//! Local Markdown resources for a coding-agent session.
//!
//! This module is intentionally independent of the rest of the application so
//! that the session and CLI layers can adopt it incrementally. It makes all
//! on-disk roots explicit through [`ResourcePaths`].
//!
//! Resource files are untrusted when they come from a workspace. Reads reject
//! symbolic-link files and paths that resolve outside their configured root;
//! writes additionally verify the complete prompt-directory path before
//! creating or replacing a file.

use std::{
    cell::Cell,
    collections::{BTreeMap, HashSet},
    env,
    error::Error,
    fmt,
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Component, Path, PathBuf},
    process,
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use serde::{Deserialize, Serialize};
use tar::{Archive, Builder, EntryType, Header};
use time::{OffsetDateTime, format_description::well_known::Rfc3339};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

/// Largest accepted resource file, including prompt templates and skills.
pub const MAX_RESOURCE_BYTES: usize = 2 * 1024 * 1024;
/// Version of the portable prompt-archive format understood by this build.
pub const ARCHIVE_VERSION: u32 = 1;
/// Largest number of tar members accepted from one untrusted archive.
pub const MAX_ARCHIVE_ENTRIES: usize = 10_000;
/// Largest total decompressed archive payload retained or read.
pub const MAX_ARCHIVE_BYTES: usize = 64 * 1024 * 1024;

const ARCHIVE_MANIFEST_NAME: &str = "manifest.json";
const ARCHIVE_USER_DIRECTORY: &str = "user";
const ARCHIVE_WORKSPACE_DIRECTORY: &str = "project";

static TEMP_FILE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Result type used by this module.
pub type Result<T> = std::result::Result<T, ResourceError>;

/// Errors that callers can present directly to a user.
#[derive(Debug)]
pub enum ResourceError {
    Io {
        action: &'static str,
        path: PathBuf,
        source: io::Error,
    },
    InvalidTemplateName {
        name: String,
        reason: String,
    },
    TemplateExists {
        path: PathBuf,
    },
    TemplateNotFound {
        name: String,
        directory: PathBuf,
    },
    EmptyTemplate,
    TemplateTooLarge {
        size: usize,
        limit: usize,
    },
    PromptDirectoryEscapes {
        directory: PathBuf,
        root: PathBuf,
    },
    UnsafePromptDestination {
        path: PathBuf,
        reason: &'static str,
    },
    UnterminatedQuote,
    ArchiveInvalid(String),
    ArchiveTooLarge {
        limit: usize,
    },
    ArchiveFutureVersion {
        version: u32,
        supported: u32,
    },
}

impl fmt::Display for ResourceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                action,
                path,
                source,
            } => write!(formatter, "{action} {}: {source}", path.display()),
            Self::InvalidTemplateName { name, reason } => {
                write!(formatter, "{name:?} cannot be a prompt name: {reason}")
            }
            Self::TemplateExists { path } => {
                write!(formatter, "a prompt already exists at {}", path.display())
            }
            Self::TemplateNotFound { name, directory } => {
                write!(
                    formatter,
                    "no prompt named {name:?} in {}",
                    directory.display()
                )
            }
            Self::EmptyTemplate => formatter.write_str("a prompt needs some text"),
            Self::TemplateTooLarge { size, limit } => {
                write!(
                    formatter,
                    "this prompt is {size} bytes, over the {limit}-byte limit"
                )
            }
            Self::PromptDirectoryEscapes { directory, root } => write!(
                formatter,
                "the prompt directory {} resolves outside its allowed root {}",
                directory.display(),
                root.display()
            ),
            Self::UnsafePromptDestination { path, reason } => {
                write!(formatter, "{} is unsafe: {reason}", path.display())
            }
            Self::UnterminatedQuote => {
                formatter.write_str("unterminated quote in prompt-template arguments")
            }
            Self::ArchiveInvalid(reason) => {
                write!(formatter, "this is not a valid prompt archive: {reason}")
            }
            Self::ArchiveTooLarge { limit } => {
                write!(
                    formatter,
                    "this archive expands beyond the {limit}-byte limit"
                )
            }
            Self::ArchiveFutureVersion { version, supported } => {
                write!(
                    formatter,
                    "this archive is version {version}; this build understands version {supported}"
                )
            }
        }
    }
}

impl Error for ResourceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

fn io_error(action: &'static str, path: impl Into<PathBuf>, source: io::Error) -> ResourceError {
    ResourceError::Io {
        action,
        path: path.into(),
        source,
    }
}

/// Explicit roots used to load and save resources.
///
/// `workspace` is normally the active working directory. Discovery walks from
/// it toward the nearest `.git` entry, inclusive. `agent_dir` is the user
/// configuration directory (for example, `~/.goshcoder/agent`). `user_home`
/// only controls the optional global `~/.agents/skills` location and can be
/// disabled in tests or sandboxed sessions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourcePaths {
    pub workspace: PathBuf,
    pub agent_dir: PathBuf,
    pub user_home: Option<PathBuf>,
}

impl ResourcePaths {
    /// Builds paths from caller-provided workspace and agent directories.
    pub fn new(workspace: impl AsRef<Path>, agent_dir: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            workspace: absolute_path(workspace.as_ref())?,
            agent_dir: absolute_path(agent_dir.as_ref())?,
            user_home: default_user_home(),
        })
    }

    /// Builds paths using the process working directory as the workspace.
    pub fn from_current_dir(agent_dir: impl AsRef<Path>) -> Result<Self> {
        let workspace = env::current_dir()
            .map_err(|error| io_error("determine the current directory", ".", error))?;
        Self::new(workspace, agent_dir)
    }

    /// Overrides the global skill home directory without changing other roots.
    pub fn with_user_home(mut self, user_home: impl AsRef<Path>) -> Result<Self> {
        self.user_home = Some(absolute_path(user_home.as_ref())?);
        Ok(self)
    }

    /// Disables discovery of global `~/.agents/skills`.
    pub fn without_user_home(mut self) -> Self {
        self.user_home = None;
        self
    }

    /// Returns the prompt directory for one save scope.
    pub fn prompt_dir(&self, scope: PromptScope) -> PathBuf {
        match scope {
            PromptScope::User => self.agent_dir.join("prompts"),
            PromptScope::Workspace => self.workspace.join(".goshcoder").join("prompts"),
        }
    }

    fn prompt_root(&self, scope: PromptScope) -> &Path {
        match scope {
            PromptScope::User => &self.agent_dir,
            PromptScope::Workspace => &self.workspace,
        }
    }

    fn template_locations(&self) -> [(PathBuf, &Path); 3] {
        [
            (self.agent_dir.join("prompts"), &self.agent_dir),
            (self.workspace.join(".pi").join("prompts"), &self.workspace),
            (
                self.workspace.join(".goshcoder").join("prompts"),
                &self.workspace,
            ),
        ]
    }
}

fn default_user_home() -> Option<PathBuf> {
    env::var_os("HOME")
        .or_else(|| env::var_os("USERPROFILE"))
        .and_then(|path| absolute_path(Path::new(&path)).ok())
}

/// Scope in which a prompt is stored.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum PromptScope {
    /// A prompt in the configured agent directory, shared between workspaces.
    #[default]
    User,
    /// A prompt committed beneath `.goshcoder/prompts` in one workspace.
    Workspace,
}

impl fmt::Display for PromptScope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::User => "user",
            Self::Workspace => "workspace",
        })
    }
}

/// A loaded `AGENTS.md`, `AGENTS.override.md`, or `CLAUDE.md`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContextFile {
    pub path: PathBuf,
    pub content: String,
}

/// A slash-expandable Markdown prompt template.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Template {
    pub name: String,
    pub description: String,
    pub argument_hint: String,
    pub path: PathBuf,
    pub body: String,
}

/// An Agent Skills-compatible Markdown skill.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub path: PathBuf,
    pub body: String,
    pub disable_model_invocation: bool,
}

/// The complete local resource snapshot for one session.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResourceSet {
    pub context_files: Vec<ContextFile>,
    pub templates: Vec<Template>,
    pub skills: Vec<Skill>,
    pub custom_system: String,
    pub append_system: String,
    pub custom_system_source: Option<PathBuf>,
    pub append_system_sources: Vec<PathBuf>,
    pub warnings: Vec<String>,
}

/// A prompt-only view for list and reload callers.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TemplateListing {
    pub templates: Vec<Template>,
    pub warnings: Vec<String>,
}

/// Discovers all local resources rooted at `paths`.
pub fn discover(paths: &ResourcePaths) -> Result<ResourceSet> {
    let ancestors = ancestor_directories_until_repository(&paths.workspace)?;
    let workspace_boundary = ancestors
        .first()
        .cloned()
        .unwrap_or_else(|| paths.workspace.clone());
    let mut warnings = Vec::new();

    let context_files =
        discover_context_files(paths, &ancestors, &workspace_boundary, &mut warnings);
    let (custom_system, custom_system_source, append_system, append_system_sources) =
        discover_system_overrides(paths, &mut warnings);
    let templates = discover_templates(paths, &mut warnings);
    let skills = discover_skills(paths, &ancestors, &workspace_boundary, &mut warnings);

    Ok(ResourceSet {
        context_files,
        templates,
        skills,
        custom_system,
        append_system,
        custom_system_source,
        append_system_sources,
        warnings,
    })
}

/// Convenience wrapper for callers that have separate path arguments.
pub fn discover_at(
    workspace: impl AsRef<Path>,
    agent_dir: impl AsRef<Path>,
) -> Result<ResourceSet> {
    let paths = ResourcePaths::new(workspace, agent_dir)?;
    discover(&paths)
}

/// Lists the ancestors used for workspace-scoped resources.
///
/// The returned order is outermost to innermost, so a repository-root context
/// file appears before one in the active child directory. The nearest `.git`
/// entry (file or directory) is included and terminates the walk. When no
/// repository marker exists, the walk ends at the filesystem root to preserve
/// normal ancestor-context behavior.
pub fn ancestor_directories_until_repository(path: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let mut current = absolute_path(path.as_ref())?;
    let mut reverse = Vec::new();

    loop {
        reverse.push(current.clone());
        if is_repository_root(&current) {
            break;
        }
        let Some(parent) = current.parent() else {
            break;
        };
        if parent == current {
            break;
        }
        current = parent.to_path_buf();
    }

    reverse.reverse();
    Ok(reverse)
}

/// Returns the nearest directory containing a `.git` entry, if there is one.
pub fn repository_root(path: impl AsRef<Path>) -> Result<Option<PathBuf>> {
    let mut current = absolute_path(path.as_ref())?;
    loop {
        if is_repository_root(&current) {
            return Ok(Some(current));
        }
        let Some(parent) = current.parent() else {
            return Ok(None);
        };
        if parent == current {
            return Ok(None);
        }
        current = parent.to_path_buf();
    }
}

fn is_repository_root(directory: &Path) -> bool {
    fs::symlink_metadata(directory.join(".git")).is_ok()
}

fn discover_context_files(
    paths: &ResourcePaths,
    ancestors: &[PathBuf],
    workspace_boundary: &Path,
    warnings: &mut Vec<String>,
) -> Vec<ContextFile> {
    let mut seen = HashSet::new();
    let mut files = Vec::new();
    let mut add = |path: PathBuf, root: &Path| {
        let absolute = normalize_absolute(&path);
        if seen.contains(&absolute) {
            return true;
        }
        if let Some(content) = read_safe_resource(&absolute, root, "context", warnings) {
            seen.insert(absolute.clone());
            files.push(ContextFile {
                path: absolute,
                content,
            });
            return true;
        }
        false
    };

    let _ = add(paths.agent_dir.join("AGENTS.md"), &paths.agent_dir);
    for directory in ancestors {
        let override_path = directory.join("AGENTS.override.md");
        if context_candidate_exists(&override_path) && add(override_path, workspace_boundary) {
            continue;
        }

        let agents_path = directory.join("AGENTS.md");
        if context_candidate_exists(&agents_path) && add(agents_path, workspace_boundary) {
            continue;
        }

        let _ = add(directory.join("CLAUDE.md"), workspace_boundary);
    }
    files
}

fn context_candidate_exists(path: &Path) -> bool {
    fs::symlink_metadata(path)
        .is_ok_and(|metadata| metadata.file_type().is_file() || metadata.file_type().is_symlink())
}

fn discover_system_overrides(
    paths: &ResourcePaths,
    warnings: &mut Vec<String>,
) -> (String, Option<PathBuf>, String, Vec<PathBuf>) {
    let system_candidates = [
        (paths.agent_dir.join("SYSTEM.md"), &paths.agent_dir, false),
        (
            paths.workspace.join(".pi").join("SYSTEM.md"),
            &paths.workspace,
            true,
        ),
        (
            paths.workspace.join(".goshcoder").join("SYSTEM.md"),
            &paths.workspace,
            true,
        ),
    ];

    let mut custom_system = String::new();
    let mut custom_system_source = None;
    for (candidate, root, is_workspace_file) in system_candidates {
        if let Some(content) = read_safe_resource(&candidate, root, "system prompt", warnings) {
            if content.is_empty() {
                continue;
            }
            if is_workspace_file {
                warnings.push(format!(
                    "{} replaces the whole system prompt; review it if this workspace is not yours",
                    display_path(&candidate)
                ));
            }
            custom_system_source = Some(candidate);
            custom_system = content;
            break;
        }
    }

    let append_candidates = [
        (paths.agent_dir.join("APPEND_SYSTEM.md"), &paths.agent_dir),
        (
            paths.workspace.join(".pi").join("APPEND_SYSTEM.md"),
            &paths.workspace,
        ),
        (
            paths.workspace.join(".goshcoder").join("APPEND_SYSTEM.md"),
            &paths.workspace,
        ),
    ];
    let mut append_parts = Vec::new();
    let mut append_system_sources = Vec::new();
    for (candidate, root) in append_candidates {
        if let Some(content) =
            read_safe_resource(&candidate, root, "append system prompt", warnings)
            && !content.is_empty()
        {
            append_system_sources.push(candidate);
            append_parts.push(content);
        }
    }

    (
        custom_system,
        custom_system_source,
        append_parts.join("\n\n"),
        append_system_sources,
    )
}

fn discover_templates(paths: &ResourcePaths, warnings: &mut Vec<String>) -> Vec<Template> {
    let mut seen_names = HashSet::new();
    let mut templates = Vec::new();

    for (directory, root) in paths.template_locations() {
        if !safe_directory_for_read(&directory, root, "prompt directory", warnings) {
            continue;
        }
        let entries = match sorted_directory_entries(&directory, "read prompt directory", warnings)
        {
            Some(entries) => entries,
            None => continue,
        };
        for entry in entries {
            let file_name = entry.file_name();
            let Some(file_name) = file_name.to_str() else {
                warnings.push(format!(
                    "prompt {} skipped: its name is not valid UTF-8",
                    display_path(&entry.path())
                ));
                continue;
            };
            let Some(name) = markdown_stem(file_name) else {
                continue;
            };
            if name.is_empty() {
                warnings.push(format!(
                    "prompt {} skipped: it has an empty command name",
                    display_path(&entry.path())
                ));
                continue;
            }
            if seen_names.contains(name) {
                continue;
            }

            let path = entry.path();
            let Some(body) = read_safe_resource(&path, root, "prompt", warnings) else {
                continue;
            };
            let (metadata, content) = parse_frontmatter(&body);
            let description = metadata
                .get("description")
                .cloned()
                .filter(|description| !description.is_empty())
                .unwrap_or_else(|| first_non_empty_line(&content));
            seen_names.insert(name.to_owned());
            templates.push(Template {
                name: name.to_owned(),
                description,
                argument_hint: metadata.get("argument-hint").cloned().unwrap_or_default(),
                path: normalize_absolute(&path),
                body: content,
            });
        }
    }

    templates.sort_by(|left, right| left.name.cmp(&right.name));
    templates
}

fn discover_skills(
    paths: &ResourcePaths,
    ancestors: &[PathBuf],
    workspace_boundary: &Path,
    warnings: &mut Vec<String>,
) -> Vec<Skill> {
    let mut locations: Vec<(PathBuf, &Path)> =
        vec![(paths.agent_dir.join("skills"), &paths.agent_dir)];
    if let Some(home) = paths.user_home.as_deref() {
        locations.push((home.join(".agents").join("skills"), home));
    }
    locations.push((paths.workspace.join(".pi").join("skills"), &paths.workspace));
    locations.push((
        paths.workspace.join(".goshcoder").join("skills"),
        &paths.workspace,
    ));
    for directory in ancestors {
        locations.push((directory.join(".agents").join("skills"), workspace_boundary));
    }

    let mut seen_paths = HashSet::new();
    let mut seen_names = HashSet::new();
    let mut skills = Vec::new();
    for (location, root) in locations {
        discover_skills_in_directory(
            &location,
            root,
            &mut seen_paths,
            &mut seen_names,
            &mut skills,
            warnings,
        );
    }
    skills.sort_by(|left, right| left.name.cmp(&right.name));
    skills
}

fn discover_skills_in_directory(
    location: &Path,
    root: &Path,
    seen_paths: &mut HashSet<PathBuf>,
    seen_names: &mut HashSet<String>,
    skills: &mut Vec<Skill>,
    warnings: &mut Vec<String>,
) {
    if !safe_directory_for_read(location, root, "skill directory", warnings) {
        return;
    }
    walk_skills(
        location, location, root, seen_paths, seen_names, skills, warnings,
    );
}

fn walk_skills(
    directory: &Path,
    location: &Path,
    root: &Path,
    seen_paths: &mut HashSet<PathBuf>,
    seen_names: &mut HashSet<String>,
    skills: &mut Vec<Skill>,
    warnings: &mut Vec<String>,
) {
    let Some(entries) = sorted_directory_entries(directory, "read skill directory", warnings)
    else {
        return;
    };

    for entry in entries {
        let path = entry.path();
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) => {
                warnings.push(format!("skill {}: {error}", display_path(&path)));
                continue;
            }
        };
        if metadata.file_type().is_symlink() {
            warnings.push(format!(
                "skill {} skipped: it is a symbolic link",
                display_path(&path)
            ));
            continue;
        }
        if metadata.file_type().is_dir() {
            if entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            walk_skills(
                &path, location, root, seen_paths, seen_names, skills, warnings,
            );
            continue;
        }
        if !metadata.file_type().is_file() {
            continue;
        }

        let file_name = entry.file_name();
        let Some(file_name) = file_name.to_str() else {
            warnings.push(format!(
                "skill {} skipped: its name is not valid UTF-8",
                display_path(&path)
            ));
            continue;
        };
        let at_location_root = directory == location;
        let is_skill_file =
            file_name == "SKILL.md" || (at_location_root && markdown_stem(file_name).is_some());
        if !is_skill_file {
            continue;
        }

        let absolute = normalize_absolute(&path);
        if !seen_paths.insert(absolute.clone()) {
            continue;
        }
        let Some(body) = read_safe_resource(&absolute, root, "skill", warnings) else {
            continue;
        };
        let (metadata, content) = parse_frontmatter(&body);
        let name = metadata
            .get("name")
            .map(|name| name.trim().to_owned())
            .filter(|name| !name.is_empty())
            .or_else(|| {
                if file_name == "SKILL.md" {
                    path.parent()
                        .and_then(Path::file_name)
                        .and_then(|name| name.to_str())
                        .map(str::to_owned)
                } else {
                    markdown_stem(file_name).map(str::to_owned)
                }
            });
        let Some(name) = name else {
            warnings.push(format!(
                "skill {} skipped: it has no usable name",
                display_path(&path)
            ));
            continue;
        };
        let description = metadata
            .get("description")
            .map(|description| description.trim().to_owned())
            .unwrap_or_default();
        if description.is_empty() {
            warnings.push(format!(
                "skill {} has no description and was skipped",
                display_path(&path)
            ));
            continue;
        }
        if !seen_names.insert(name.clone()) {
            warnings.push(format!(
                "duplicate skill {name:?} at {} was skipped",
                display_path(&path)
            ));
            continue;
        }
        skills.push(Skill {
            name,
            description,
            path: absolute,
            body: content,
            disable_model_invocation: parse_bool(
                metadata
                    .get("disable-model-invocation")
                    .map(String::as_str)
                    .unwrap_or_default(),
            ),
        });
    }
}

/// Lists prompt templates without re-reading contexts, skills, or system files.
pub fn list_templates(paths: &ResourcePaths) -> Result<TemplateListing> {
    let mut warnings = Vec::new();
    let templates = discover_templates(paths, &mut warnings);
    Ok(TemplateListing {
        templates,
        warnings,
    })
}

/// Replaces only the prompt section of an already-discovered resource set.
pub fn reload_templates(set: &ResourceSet, paths: &ResourcePaths) -> Result<ResourceSet> {
    let listing = list_templates(paths)?;
    let mut updated = set.clone();
    updated.templates = listing.templates;
    updated
        .warnings
        .retain(|warning| !warning.starts_with("prompt"));
    updated.warnings.extend(listing.warnings);
    Ok(updated)
}

/// Options controlling a prompt save.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SaveTemplateOptions {
    pub description: String,
    pub argument_hint: String,
    pub scope: PromptScope,
    /// Names that templates must not shadow, with or without a leading slash.
    pub reserved_names: Vec<String>,
    /// Escapes captured text so all dollar sequences render literally.
    pub literal: bool,
    /// Allows replacement of an existing regular prompt file.
    pub overwrite: bool,
}

/// Successful result of saving a prompt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SaveTemplateResult {
    pub path: PathBuf,
    /// Higher-precedence prompt that hides this saved prompt, if any.
    pub shadowed_by: Option<PathBuf>,
}

/// Validates a prompt name before it becomes both a filename and slash command.
pub fn validate_template_name(name: &str, reserved_names: &[String]) -> Result<()> {
    if name.is_empty() {
        return Err(ResourceError::InvalidTemplateName {
            name: name.to_owned(),
            reason: "a name is required".to_owned(),
        });
    }
    let bytes = name.as_bytes();
    if !bytes[0].is_ascii_alphanumeric()
        || !bytes
            .iter()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(ResourceError::InvalidTemplateName {
            name: name.to_owned(),
            reason: "use ASCII letters, digits, '.', '-' and '_', starting with a letter or digit"
                .to_owned(),
        });
    }
    if name.contains("..") {
        return Err(ResourceError::InvalidTemplateName {
            name: name.to_owned(),
            reason: "names may not contain '..'".to_owned(),
        });
    }
    if name.starts_with("skill:") {
        return Err(ResourceError::InvalidTemplateName {
            name: name.to_owned(),
            reason: "names beginning with \"skill:\" are reserved for Agent Skills".to_owned(),
        });
    }
    if reserved_names
        .iter()
        .any(|reserved| name.eq_ignore_ascii_case(reserved.trim().trim_start_matches('/')))
    {
        return Err(ResourceError::InvalidTemplateName {
            name: name.to_owned(),
            reason: "that name is reserved by a built-in command".to_owned(),
        });
    }
    Ok(())
}

/// Returns whether `name` passes the filesystem and slash-command checks.
pub fn is_valid_template_name(name: &str) -> bool {
    validate_template_name(name, &[]).is_ok()
}

/// Saves a Markdown template through a same-directory temporary file and rename.
pub fn save_template(
    paths: &ResourcePaths,
    name: &str,
    body: &str,
    options: SaveTemplateOptions,
) -> Result<SaveTemplateResult> {
    validate_template_name(name, &options.reserved_names)?;
    let mut body = body.trim().to_owned();
    if body.is_empty() {
        return Err(ResourceError::EmptyTemplate);
    }
    if options.literal {
        body = escape_placeholders(&body);
    }
    let content = render_template_file(&options.description, &options.argument_hint, &body);
    if content.len() > MAX_RESOURCE_BYTES {
        return Err(ResourceError::TemplateTooLarge {
            size: content.len(),
            limit: MAX_RESOURCE_BYTES,
        });
    }

    let directory = paths.prompt_dir(options.scope);
    check_template_directory(&directory, paths.prompt_root(options.scope), true)?;
    let path = directory.join(format!("{name}.md"));
    match fs::symlink_metadata(&path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            return Err(ResourceError::UnsafePromptDestination {
                path,
                reason: "it is a symbolic link",
            });
        }
        Ok(metadata) if !metadata.file_type().is_file() => {
            return Err(ResourceError::UnsafePromptDestination {
                path,
                reason: "it is not a regular file",
            });
        }
        Ok(_) if !options.overwrite => {
            return Err(ResourceError::TemplateExists { path });
        }
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(io_error("inspect prompt", path, error)),
    }

    atomic_write(&path, content.as_bytes())?;
    Ok(SaveTemplateResult {
        shadowed_by: shadowing_template(paths, name, &path),
        path,
    })
}

/// Renders the simple frontmatter shape understood by [`parse_frontmatter`].
pub fn render_template_file(description: &str, argument_hint: &str, body: &str) -> String {
    let mut content = String::new();
    if !description.is_empty() || !argument_hint.is_empty() {
        content.push_str("---\n");
        if !description.is_empty() {
            content.push_str("description: ");
            content.push_str(&single_line(description));
            content.push('\n');
        }
        if !argument_hint.is_empty() {
            content.push_str("argument-hint: ");
            content.push_str(&single_line(argument_hint));
            content.push('\n');
        }
        content.push_str("---\n");
    }
    content.push_str(body);
    content.push('\n');
    content
}

fn shadowing_template(paths: &ResourcePaths, name: &str, written: &Path) -> Option<PathBuf> {
    let written = normalize_absolute(written);
    for (directory, root) in paths.template_locations() {
        let candidate = directory.join(format!("{name}.md"));
        if normalize_absolute(&candidate) == written {
            return None;
        }
        if is_safe_regular_file(&candidate, root) {
            return Some(candidate);
        }
    }
    None
}

/// Result of removing one template.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemoveTemplateResult {
    pub path: PathBuf,
    /// A symbolic link was removed rather than its target.
    pub removed_symbolic_link: bool,
}

/// Removes only the prompt of the requested scope.
///
/// A leaf symbolic link is itself removed safely. A redirected directory is
/// rejected before anything is deleted.
pub fn remove_template(
    paths: &ResourcePaths,
    name: &str,
    scope: PromptScope,
) -> Result<RemoveTemplateResult> {
    validate_template_name(name, &[])?;
    let directory = paths.prompt_dir(scope);
    if !check_template_directory(&directory, paths.prompt_root(scope), false)? {
        return Err(ResourceError::TemplateNotFound {
            name: name.to_owned(),
            directory,
        });
    }
    let path = directory.join(format!("{name}.md"));
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            return Err(ResourceError::TemplateNotFound {
                name: name.to_owned(),
                directory,
            });
        }
        Err(error) => return Err(io_error("inspect prompt", path, error)),
    };
    if !metadata.file_type().is_file() && !metadata.file_type().is_symlink() {
        return Err(ResourceError::UnsafePromptDestination {
            path,
            reason: "it is not a regular file or symbolic link",
        });
    }
    let removed_symbolic_link = metadata.file_type().is_symlink();
    fs::remove_file(&path).map_err(|error| io_error("remove prompt", &path, error))?;
    Ok(RemoveTemplateResult {
        path,
        removed_symbolic_link,
    })
}

impl ResourceSet {
    /// Finds one discovered template by slash-command name.
    pub fn find_template(&self, name: &str) -> Option<&Template> {
        self.templates.iter().find(|template| template.name == name)
    }

    /// Expands a template or skill command, or reports that `input` is ordinary
    /// user text.
    pub fn expand_input(&self, input: &str) -> Result<Expansion> {
        let input = input.trim();
        let split_at = input
            .char_indices()
            .find_map(|(index, character)| character.is_whitespace().then_some(index));
        let (command, rest) = match split_at {
            Some(index) => (&input[..index], input[index..].trim()),
            None => (input, ""),
        };

        if let Some(template) = self
            .templates
            .iter()
            .find(|template| command == format!("/{}", template.name))
        {
            let arguments = split_arguments(rest)?;
            return Ok(Expansion::Template {
                name: template.name.clone(),
                text: expand_template(&template.body, &arguments),
            });
        }

        if let Some(name) = command.strip_prefix("/skill:")
            && let Some(skill) = self.skills.iter().find(|skill| skill.name == name)
        {
            let mut text = format!(
                "Follow the {:?} skill from {}:\n\n{}",
                skill.name,
                display_path(&skill.path),
                skill.body
            );
            if !rest.is_empty() {
                text.push_str("\n\nUser: ");
                text.push_str(rest);
            }
            return Ok(Expansion::Skill {
                name: skill.name.clone(),
                text,
            });
        }

        Ok(Expansion::NotResource)
    }

    /// Builds the system prompt used by a session from this resource snapshot.
    ///
    /// An explicit prompt wins over a discovered `SYSTEM.md`; `APPEND_SYSTEM.md`
    /// is appended in either case. The default tool framing is intentionally
    /// small and can be replaced later by a runtime-specific prompt.
    pub fn build_system_prompt<I, S>(
        &self,
        explicit: &str,
        cwd: impl AsRef<Path>,
        tool_names: I,
    ) -> String
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let selected: HashSet<String> = tool_names
            .into_iter()
            .map(|name| name.as_ref().to_owned())
            .collect();
        let mut base = explicit.trim().to_owned();
        if base.is_empty() {
            base = self.custom_system.trim().to_owned();
        }
        if base.is_empty() {
            let mut tools = Vec::new();
            for name in [
                "read",
                "write",
                "edit",
                "bash",
                "grep",
                "find",
                "ls",
                "web_search",
            ] {
                if selected.contains(name) {
                    tools.push(format!("- {name}: {}", tool_snippet(name)));
                }
            }
            if tools.is_empty() {
                tools.push("(none)".to_owned());
            }
            let mut guidelines = vec![
                "- Be concise in your responses".to_owned(),
                "- Show file paths clearly when working with files".to_owned(),
            ];
            if selected.contains("read") {
                guidelines.insert(
                    0,
                    "- Use read to examine files instead of cat or sed".to_owned(),
                );
            }
            base = format!(
                "You are an expert coding assistant operating inside GoshCoder, a coding agent harness. You help users by reading files, executing commands, editing code, and writing new files.\n\nAvailable tools:\n{}\n\nGuidelines:\n{}",
                tools.join("\n"),
                guidelines.join("\n")
            );
        }

        if !self.append_system.trim().is_empty() {
            base.push_str("\n\n");
            base.push_str(self.append_system.trim());
        }
        if !self.context_files.is_empty() {
            base.push_str(
                "\n\n<project_context>\n\nProject-specific instructions and guidelines:\n\n",
            );
            for file in &self.context_files {
                base.push_str(&format!(
                    "<project_instructions path={:?}>\n{}\n</project_instructions>\n\n",
                    display_path(&file.path),
                    seal_delimiters(&file.content)
                ));
            }
            base.push_str("</project_context>");
        }
        if selected.contains("read") {
            let visible = self
                .skills
                .iter()
                .filter(|skill| !skill.disable_model_invocation)
                .collect::<Vec<_>>();
            if !visible.is_empty() {
                base.push_str("\n\n<available_skills>\n");
                for skill in visible {
                    base.push_str(&format!(
                        "<skill><name>{}</name><description>{}</description><location>{}</location></skill>\n",
                        xml_escape(&skill.name),
                        xml_escape(&skill.description),
                        xml_escape(&display_path(&skill.path)),
                    ));
                }
                base.push_str("</available_skills>");
            }
        }
        let cwd = absolute_path(cwd.as_ref()).unwrap_or_else(|_| cwd.as_ref().to_path_buf());
        base.push_str("\nCurrent working directory: ");
        base.push_str(&display_path(&cwd));
        base
    }

    /// Creates a display-ready summary for `/resources`-style interfaces.
    pub fn report(&self, paths: &ResourcePaths) -> ResourceReport {
        ResourceReport {
            workspace: paths.workspace.clone(),
            agent_dir: paths.agent_dir.clone(),
            context_files: self
                .context_files
                .iter()
                .map(|file| file.path.clone())
                .collect(),
            templates: self
                .templates
                .iter()
                .map(|template| ResourceReportEntry {
                    command: format!("/{}", template.name),
                    description: template.description.clone(),
                    path: template.path.clone(),
                })
                .collect(),
            skills: self
                .skills
                .iter()
                .map(|skill| ResourceReportEntry {
                    command: format!("/skill:{}", skill.name),
                    description: skill.description.clone(),
                    path: skill.path.clone(),
                })
                .collect(),
            custom_system_source: self.custom_system_source.clone(),
            append_system_sources: self.append_system_sources.clone(),
            warnings: self.warnings.clone(),
        }
    }
}

/// Result of [`ResourceSet::expand_input`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Expansion {
    NotResource,
    Template { name: String, text: String },
    Skill { name: String, text: String },
}

impl Expansion {
    /// Returns the expanded text when this was a resource command.
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Template { text, .. } | Self::Skill { text, .. } => Some(text),
            Self::NotResource => None,
        }
    }

    /// Returns whether expansion consumed a resource command.
    pub fn is_resource(&self) -> bool {
        !matches!(self, Self::NotResource)
    }
}

/// Splits template arguments with shell-like quotes and backslash escapes.
pub fn split_arguments(input: &str) -> Result<Vec<String>> {
    let mut arguments = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;

    for character in input.trim().chars() {
        if escaped {
            current.push(character);
            escaped = false;
            continue;
        }
        if character == '\\' && quote != Some('\'') {
            escaped = true;
            continue;
        }
        if let Some(open_quote) = quote {
            if character == open_quote {
                quote = None;
            } else {
                current.push(character);
            }
            continue;
        }
        match character {
            '\'' | '"' => quote = Some(character),
            ' ' | '\t' | '\n' => {
                if !current.is_empty() {
                    arguments.push(std::mem::take(&mut current));
                }
            }
            _ => current.push(character),
        }
    }
    if escaped {
        current.push('\\');
    }
    if quote.is_some() {
        return Err(ResourceError::UnterminatedQuote);
    }
    if !current.is_empty() {
        arguments.push(current);
    }
    Ok(arguments)
}

/// Expands `$1`, `$@`, `$ARGUMENTS`, defaults, and argument slices in a
/// Markdown template.
///
/// Replacement is one left-to-right pass: values supplied in an argument are
/// never scanned again as placeholders.
pub fn expand_template(body: &str, arguments: &[String]) -> String {
    let all_arguments = arguments.join(" ");
    let mut result = String::with_capacity(body.len());
    let mut index = 0;

    while index < body.len() {
        let remaining = &body[index..];
        if remaining.starts_with('$')
            && let Some((consumed, replacement)) =
                expand_placeholder(remaining, arguments, &all_arguments)
        {
            result.push_str(&replacement);
            index += consumed;
            continue;
        }
        let character = remaining
            .chars()
            .next()
            .expect("index always points to a character boundary");
        result.push(character);
        index += character.len_utf8();
    }
    result.trim().to_owned()
}

/// Escapes captured text so its dollar sequences reproduce exactly after
/// [`expand_template`].
///
/// Every dollar is doubled rather than only known placeholders. That also
/// preserves an existing `$$` literal escape and makes future placeholder
/// syntax safe by construction.
pub fn escape_placeholders(body: &str) -> String {
    body.replace('$', "$$")
}

/// Returns whether invoking a template would interpret a dollar sequence.
pub fn has_placeholders(body: &str) -> bool {
    let mut index = 0;
    while index < body.len() {
        let remaining = &body[index..];
        if remaining.starts_with('$') && expand_placeholder(remaining, &[], "").is_some() {
            return true;
        }
        let character = remaining
            .chars()
            .next()
            .expect("index always points to a character boundary");
        index += character.len_utf8();
    }
    false
}

fn expand_placeholder(
    text: &str,
    arguments: &[String],
    all_arguments: &str,
) -> Option<(usize, String)> {
    if let Some(rest) = text.strip_prefix("$$") {
        let consumed = text.len() - rest.len();
        return Some((consumed, "$".to_owned()));
    }

    if let Some((consumed, key, default)) = default_placeholder(text) {
        let value = argument_value(key, arguments, all_arguments);
        return Some((
            consumed,
            if value.is_empty() {
                default.to_owned()
            } else {
                value.to_owned()
            },
        ));
    }

    if let Some((consumed, start, length)) = slice_placeholder(text) {
        let replacement = match start.parse::<usize>() {
            Ok(start) if start > 0 => {
                let start = start - 1;
                if start >= arguments.len() {
                    String::new()
                } else {
                    let end = match length {
                        None => arguments.len(),
                        Some(length) => match length.parse::<usize>() {
                            Ok(length) => start.saturating_add(length).min(arguments.len()),
                            Err(_) => start,
                        },
                    };
                    arguments[start..end].join(" ")
                }
            }
            _ => String::new(),
        };
        return Some((consumed, replacement));
    }

    if let Some(digits) = text
        .strip_prefix('$')
        .and_then(|remaining| take_ascii_digits(remaining).filter(|digits| !digits.is_empty()))
    {
        let consumed = 1 + digits.len();
        let replacement = match digits.parse::<usize>() {
            Ok(index) if index > 0 && index <= arguments.len() => arguments[index - 1].clone(),
            _ => String::new(),
        };
        return Some((consumed, replacement));
    }
    if let Some(rest) = text.strip_prefix("$ARGUMENTS") {
        return Some((text.len() - rest.len(), all_arguments.to_owned()));
    }
    if let Some(rest) = text.strip_prefix("$@") {
        return Some((text.len() - rest.len(), all_arguments.to_owned()));
    }
    None
}

fn default_placeholder(text: &str) -> Option<(usize, &str, &str)> {
    let rest = text.strip_prefix("${")?;
    let default_index = rest.find(":-")?;
    let key = &rest[..default_index];
    if !is_argument_key(key) {
        return None;
    }
    let after_default = &rest[default_index + 2..];
    let closing_index = after_default.find('}')?;
    let default = &after_default[..closing_index];
    let consumed = 2 + default_index + 2 + closing_index + 1;
    Some((consumed, key, default))
}

fn slice_placeholder(text: &str) -> Option<(usize, &str, Option<&str>)> {
    let rest = text.strip_prefix("${@:")?;
    let start = take_ascii_digits(rest)?;
    if start.is_empty() {
        return None;
    }
    let after_start = &rest[start.len()..];
    if let Some(after_close) = after_start.strip_prefix('}') {
        let consumed = text.len() - after_close.len();
        return Some((consumed, start, None));
    }
    let after_colon = after_start.strip_prefix(':')?;
    let length = take_ascii_digits(after_colon)?;
    if length.is_empty() {
        return None;
    }
    let after_length = &after_colon[length.len()..];
    let after_close = after_length.strip_prefix('}')?;
    let consumed = text.len() - after_close.len();
    Some((consumed, start, Some(length)))
}

fn take_ascii_digits(value: &str) -> Option<&str> {
    let length = value
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_digit())
        .count();
    Some(&value[..length])
}

fn is_argument_key(key: &str) -> bool {
    key == "@"
        || key == "ARGUMENTS"
        || (!key.is_empty() && key.bytes().all(|byte| byte.is_ascii_digit()))
}

fn argument_value<'a>(key: &str, arguments: &'a [String], all_arguments: &'a str) -> &'a str {
    if matches!(key, "@" | "ARGUMENTS") {
        return all_arguments;
    }
    match key.parse::<usize>() {
        Ok(index) if index > 0 && index <= arguments.len() => &arguments[index - 1],
        _ => "",
    }
}

/// Splits simple YAML frontmatter from a Markdown body.
///
/// The format intentionally remains line-oriented and dependency-free: only
/// `key: value` entries in an opening `---` block are interpreted.
pub fn parse_frontmatter(input: &str) -> (BTreeMap<String, String>, String) {
    let metadata = BTreeMap::new();
    let normalized = input.replace("\r\n", "\n");
    if !normalized.starts_with("---\n") {
        return (metadata, input.to_owned());
    }
    let rest = &normalized[4..];
    let Some(end) = rest.find("\n---\n") else {
        return (metadata, input.to_owned());
    };
    let mut metadata = BTreeMap::new();
    for line in rest[..end].split('\n') {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        metadata.insert(
            key.trim().to_owned(),
            value.trim().trim_matches(['"', '\'']).to_owned(),
        );
    }
    (metadata, rest[end + 5..].to_owned())
}

/// A display-ready resource summary for a future `/resources` command.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResourceReport {
    pub workspace: PathBuf,
    pub agent_dir: PathBuf,
    pub context_files: Vec<PathBuf>,
    pub templates: Vec<ResourceReportEntry>,
    pub skills: Vec<ResourceReportEntry>,
    pub custom_system_source: Option<PathBuf>,
    pub append_system_sources: Vec<PathBuf>,
    pub warnings: Vec<String>,
}

/// One command-like resource in a [`ResourceReport`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceReportEntry {
    pub command: String,
    pub description: String,
    pub path: PathBuf,
}

impl ResourceReport {
    /// Formats a concise, deterministic text report suitable for a terminal UI.
    pub fn render(&self) -> String {
        let mut output = format!(
            "Context files: {}\nPrompt templates: {}\nSkills: {}\n",
            self.context_files.len(),
            self.templates.len(),
            self.skills.len()
        );
        if let Some(path) = &self.custom_system_source {
            output.push_str("Custom system: ");
            output.push_str(&display_path(path));
            output.push('\n');
        }
        for path in &self.append_system_sources {
            output.push_str("Append system: ");
            output.push_str(&display_path(path));
            output.push('\n');
        }
        for path in &self.context_files {
            output.push_str("  ");
            output.push_str(&display_path(path));
            output.push('\n');
        }
        for template in &self.templates {
            output.push_str("  ");
            output.push_str(&template.command);
            if !template.description.is_empty() {
                output.push_str(" — ");
                output.push_str(&template.description);
            }
            output.push('\n');
        }
        for skill in &self.skills {
            output.push_str("  ");
            output.push_str(&skill.command);
            if !skill.description.is_empty() {
                output.push_str(" — ");
                output.push_str(&skill.description);
            }
            output.push('\n');
        }
        if !self.warnings.is_empty() {
            output.push_str("Warnings:\n");
            for warning in &self.warnings {
                output.push_str("  ");
                output.push_str(warning);
                output.push('\n');
            }
        }
        output
    }
}

/// Metadata recorded at the root of each portable prompt archive.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default)]
pub struct ArchiveManifest {
    pub version: u32,
    pub tool: String,
    pub created: String,
    pub prompts: usize,
}

/// One raw prompt suitable for backup or restore.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchivedPrompt {
    pub name: String,
    pub scope: PromptScope,
    /// Includes frontmatter exactly as it was saved.
    pub body: String,
}

/// Prompts collected from on-disk scopes before archive encoding.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PromptCollection {
    pub prompts: Vec<ArchivedPrompt>,
    pub warnings: Vec<String>,
}

/// Collects safe regular prompt files for a portable archive writer.
pub fn collect_prompts(paths: &ResourcePaths, scopes: &[PromptScope]) -> Result<PromptCollection> {
    let mut prompts = Vec::new();
    let mut warnings = Vec::new();
    let mut seen_scopes = HashSet::new();

    for scope in scopes {
        if !seen_scopes.insert(*scope) {
            continue;
        }
        let directory = paths.prompt_dir(*scope);
        match check_template_directory(&directory, paths.prompt_root(*scope), false) {
            Ok(false) => continue,
            Ok(true) => {}
            Err(error) => {
                warnings.push(error.to_string());
                continue;
            }
        }
        let entries =
            match sorted_directory_entries(&directory, "read prompt directory", &mut warnings) {
                Some(entries) => entries,
                None => continue,
            };
        for entry in entries {
            let path = entry.path();
            let file_name = entry.file_name();
            let Some(file_name) = file_name.to_str() else {
                warnings.push(format!(
                    "prompt {} skipped: its name is not valid UTF-8",
                    display_path(&path)
                ));
                continue;
            };
            let Some(name) = markdown_stem(file_name) else {
                continue;
            };
            if let Err(error) = validate_template_name(name, &[]) {
                warnings.push(format!("prompt {} skipped: {error}", display_path(&path)));
                continue;
            }
            let Some(body) =
                read_safe_resource(&path, paths.prompt_root(*scope), "prompt", &mut warnings)
            else {
                continue;
            };
            prompts.push(ArchivedPrompt {
                name: name.to_owned(),
                scope: *scope,
                body,
            });
        }
    }
    prompts.sort_by(|left, right| {
        left.scope
            .cmp(&right.scope)
            .then_with(|| left.name.cmp(&right.name))
    });
    Ok(PromptCollection { prompts, warnings })
}

/// Outcome for one prompt passed to [`restore_prompts`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RestoreOutcome {
    pub name: String,
    pub scope: PromptScope,
    pub path: Option<PathBuf>,
    pub skipped: bool,
    pub reason: Option<String>,
}

/// Controls safe direct restoration of already-decoded prompt records.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RestoreOptions {
    pub overwrite: bool,
    pub dry_run: bool,
    pub reserved_names: Vec<String>,
    /// If nonempty, restore only prompts whose names occur in this list.
    pub only: Vec<String>,
}

/// Restores already-decoded prompt records without following unsafe paths.
pub fn restore_prompts(
    paths: &ResourcePaths,
    prompts: &[ArchivedPrompt],
    options: &RestoreOptions,
) -> Result<Vec<RestoreOutcome>> {
    let wanted: HashSet<&str> = options.only.iter().map(String::as_str).collect();
    let mut outcomes = Vec::with_capacity(prompts.len());

    for prompt in prompts {
        if !wanted.is_empty() && !wanted.contains(prompt.name.as_str()) {
            continue;
        }
        let mut outcome = RestoreOutcome {
            name: prompt.name.clone(),
            scope: prompt.scope,
            path: None,
            skipped: false,
            reason: None,
        };
        if let Err(error) = validate_template_name(&prompt.name, &options.reserved_names) {
            skip_restore(&mut outcome, error.to_string());
            outcomes.push(outcome);
            continue;
        }
        if prompt.body.len() > MAX_RESOURCE_BYTES {
            skip_restore(
                &mut outcome,
                format!("prompt content exceeds the {MAX_RESOURCE_BYTES}-byte resource limit"),
            );
            outcomes.push(outcome);
            continue;
        }

        let directory = paths.prompt_dir(prompt.scope);
        match check_template_directory(
            &directory,
            paths.prompt_root(prompt.scope),
            !options.dry_run,
        ) {
            Ok(_) => {}
            Err(error) => {
                skip_restore(&mut outcome, error.to_string());
                outcomes.push(outcome);
                continue;
            }
        }
        let path = directory.join(format!("{}.md", prompt.name));
        outcome.path = Some(path.clone());
        match fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                skip_restore(&mut outcome, "the destination is a symbolic link");
                outcomes.push(outcome);
                continue;
            }
            Ok(metadata) if !metadata.file_type().is_file() => {
                skip_restore(&mut outcome, "the destination is not a regular file");
                outcomes.push(outcome);
                continue;
            }
            Ok(_) if !options.overwrite => {
                skip_restore(&mut outcome, "a prompt with that name already exists");
                outcomes.push(outcome);
                continue;
            }
            Ok(_) | Err(_) if options.dry_run => {
                outcomes.push(outcome);
                continue;
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => {
                skip_restore(&mut outcome, format!("inspect destination: {error}"));
                outcomes.push(outcome);
                continue;
            }
        }
        if let Err(error) = atomic_write(&path, prompt.body.as_bytes()) {
            skip_restore(&mut outcome, error.to_string());
        }
        outcomes.push(outcome);
    }
    Ok(outcomes)
}

fn skip_restore(outcome: &mut RestoreOutcome, reason: impl Into<String>) {
    outcome.skipped = true;
    outcome.reason = Some(reason.into());
}

/// Writes prompts as a gzipped tar archive compatible with the Go frontend.
///
/// The manifest is intentionally informational. Archive member names are
/// constructed from validated prompt names and never from arbitrary paths.
pub fn write_archive<W: Write>(output: &mut W, prompts: &[ArchivedPrompt]) -> Result<()> {
    let now = OffsetDateTime::now_utc();
    let created = now
        .format(&Rfc3339)
        .map_err(|error| archive_invalid("format archive timestamp", error))?;
    let modified_at = now.unix_timestamp().max(0) as u64;
    let manifest = ArchiveManifest {
        version: ARCHIVE_VERSION,
        tool: "goshcoder".to_owned(),
        created,
        prompts: prompts.len(),
    };
    let mut manifest_contents = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| archive_invalid("encode archive manifest", error))?;
    manifest_contents.push(b'\n');

    let encoder = GzEncoder::new(output, Compression::default());
    let mut archive = Builder::new(encoder);
    write_archive_file(
        &mut archive,
        ARCHIVE_MANIFEST_NAME,
        &manifest_contents,
        modified_at,
    )?;
    for prompt in prompts {
        validate_template_name(&prompt.name, &[])?;
        if prompt.body.len() > MAX_RESOURCE_BYTES {
            return Err(ResourceError::TemplateTooLarge {
                size: prompt.body.len(),
                limit: MAX_RESOURCE_BYTES,
            });
        }
        let path = format!(
            "{}/{}.md",
            archive_scope_directory(prompt.scope),
            prompt.name
        );
        write_archive_file(&mut archive, &path, prompt.body.as_bytes(), modified_at)?;
    }

    let encoder = archive
        .into_inner()
        .map_err(|error| archive_invalid("finish tar archive", error))?;
    encoder
        .finish()
        .map_err(|error| archive_invalid("finish gzip archive", error))?;
    Ok(())
}

/// The decoded contents of a prompt archive.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ArchiveRead {
    pub prompts: Vec<ArchivedPrompt>,
    pub manifest: ArchiveManifest,
    pub warnings: Vec<String>,
}

/// Decodes a portable prompt archive without trusting its paths or metadata.
///
/// The decompressed stream, member count, individual payloads, and retained
/// prompt data are separately bounded. Only regular Markdown files below the
/// `user/` and `project/` archive directories can become prompts.
pub fn read_archive<R: Read>(input: &mut R) -> Result<ArchiveRead> {
    let budget = ArchiveBudget::new();
    let decoder = GzDecoder::new(input);
    let reader = BoundedArchiveReader {
        inner: decoder,
        budget: budget.clone(),
    };
    let mut archive = Archive::new(reader);
    let entries = archive
        .entries()
        .map_err(|error| archive_read_error("start reading tar archive", error, &budget))?;
    let mut result = ArchiveRead::default();
    let mut entry_count = 0_usize;
    let mut retained_bytes = 0_usize;

    for next in entries {
        let mut entry =
            next.map_err(|error| archive_read_error("read archive entry", error, &budget))?;
        entry_count = entry_count.saturating_add(1);
        if entry_count > MAX_ARCHIVE_ENTRIES {
            return Err(ResourceError::ArchiveInvalid(format!(
                "this archive holds more than {MAX_ARCHIVE_ENTRIES} entries"
            )));
        }

        let entry_type = entry.header().entry_type();
        if entry_type.is_dir() {
            check_archive_budget(&budget)?;
            continue;
        }
        let raw_path = String::from_utf8_lossy(entry.path_bytes().as_ref()).into_owned();
        let path = clean_archive_path(&raw_path);
        if !entry_type.is_file() {
            result
                .warnings
                .push(format!("{path} is not a regular file and was skipped"));
            check_archive_budget(&budget)?;
            continue;
        }

        let declared_size = entry.size();
        if declared_size > MAX_RESOURCE_BYTES as u64 {
            result.warnings.push(format!(
                "{path} declares {declared_size} bytes, over the {MAX_RESOURCE_BYTES}-byte limit, and was skipped"
            ));
            check_archive_budget(&budget)?;
            continue;
        }

        let mut contents = Vec::with_capacity(declared_size as usize);
        (&mut entry)
            .take((MAX_RESOURCE_BYTES + 1) as u64)
            .read_to_end(&mut contents)
            .map_err(|error| archive_read_error("read archive entry content", error, &budget))?;
        check_archive_budget(&budget)?;
        if contents.len() > MAX_RESOURCE_BYTES {
            result.warnings.push(format!(
                "{path} expands past the {MAX_RESOURCE_BYTES}-byte limit and was skipped"
            ));
            continue;
        }

        if path == ARCHIVE_MANIFEST_NAME {
            if let Ok(manifest) = serde_json::from_slice(&contents) {
                result.manifest = manifest;
            } else {
                result
                    .warnings
                    .push("the archive manifest could not be read".to_owned());
            }
            continue;
        }

        let Some(scope) = archive_path_scope(&path) else {
            result.warnings.push(format!(
                "{path} is not in a recognized archive directory and was skipped"
            ));
            continue;
        };
        let Some(file_name) = path.rsplit('/').next() else {
            result
                .warnings
                .push(format!("{path} is not a Markdown prompt and was skipped"));
            continue;
        };
        let Some(name) = markdown_stem(file_name) else {
            result
                .warnings
                .push(format!("{path} is not a Markdown prompt and was skipped"));
            continue;
        };
        if let Err(error) = validate_template_name(name, &[]) {
            result.warnings.push(format!("{path} was skipped: {error}"));
            continue;
        }

        retained_bytes = retained_bytes.saturating_add(contents.len());
        if retained_bytes > MAX_ARCHIVE_BYTES {
            return Err(ResourceError::ArchiveTooLarge {
                limit: MAX_ARCHIVE_BYTES,
            });
        }
        result.prompts.push(ArchivedPrompt {
            name: name.to_owned(),
            scope,
            body: String::from_utf8_lossy(&contents).into_owned(),
        });
    }
    check_archive_budget(&budget)?;

    if result.manifest.version > ARCHIVE_VERSION {
        return Err(ResourceError::ArchiveFutureVersion {
            version: result.manifest.version,
            supported: ARCHIVE_VERSION,
        });
    }
    Ok(result)
}

fn archive_scope_directory(scope: PromptScope) -> &'static str {
    match scope {
        PromptScope::User => ARCHIVE_USER_DIRECTORY,
        PromptScope::Workspace => ARCHIVE_WORKSPACE_DIRECTORY,
    }
}

fn archive_path_scope(path: &str) -> Option<PromptScope> {
    if path.starts_with(&format!("{ARCHIVE_WORKSPACE_DIRECTORY}/")) {
        Some(PromptScope::Workspace)
    } else if path.starts_with(&format!("{ARCHIVE_USER_DIRECTORY}/")) {
        Some(PromptScope::User)
    } else {
        None
    }
}

fn write_archive_file<W: Write>(
    archive: &mut Builder<W>,
    path: &str,
    contents: &[u8],
    modified_at: u64,
) -> Result<()> {
    let mut header = Header::new_gnu();
    header
        .set_path(path)
        .map_err(|error| archive_invalid("set archive member path", error))?;
    header.set_mode(0o600);
    header.set_size(contents.len() as u64);
    header.set_mtime(modified_at);
    header.set_entry_type(EntryType::Regular);
    header.set_cksum();
    archive
        .append(&header, contents)
        .map_err(|error| archive_invalid("write archive member", error))
}

fn archive_invalid(operation: &str, error: impl fmt::Display) -> ResourceError {
    ResourceError::ArchiveInvalid(format!("{operation}: {error}"))
}

fn archive_read_error(operation: &str, error: io::Error, budget: &ArchiveBudget) -> ResourceError {
    if budget.exceeded.get() {
        ResourceError::ArchiveTooLarge {
            limit: MAX_ARCHIVE_BYTES,
        }
    } else {
        archive_invalid(operation, error)
    }
}

fn check_archive_budget(budget: &ArchiveBudget) -> Result<()> {
    if budget.exceeded.get() {
        Err(ResourceError::ArchiveTooLarge {
            limit: MAX_ARCHIVE_BYTES,
        })
    } else {
        Ok(())
    }
}

#[derive(Clone)]
struct ArchiveBudget {
    consumed: Rc<Cell<usize>>,
    exceeded: Rc<Cell<bool>>,
}

impl ArchiveBudget {
    fn new() -> Self {
        Self {
            consumed: Rc::new(Cell::new(0)),
            exceeded: Rc::new(Cell::new(false)),
        }
    }
}

/// Bounds bytes emitted by the decompressor, not the compressed input size.
struct BoundedArchiveReader<R> {
    inner: R,
    budget: ArchiveBudget,
}

impl<R: Read> Read for BoundedArchiveReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let consumed = self.budget.consumed.get();
        let allowance = MAX_ARCHIVE_BYTES.saturating_add(1);
        if consumed >= allowance {
            self.budget.exceeded.set(true);
            return Ok(0);
        }
        let permitted = buffer.len().min(allowance - consumed);
        let read = self.inner.read(&mut buffer[..permitted])?;
        let consumed = consumed.saturating_add(read);
        self.budget.consumed.set(consumed);
        if consumed > MAX_ARCHIVE_BYTES {
            self.budget.exceeded.set(true);
        }
        Ok(read)
    }
}

fn clean_archive_path(path: &str) -> String {
    let mut components = Vec::new();
    let normalized = path.replace('\\', "/");
    for component in normalized.split('/') {
        match component {
            "" | "." => {}
            ".." => {
                if components.last().is_some_and(|previous| *previous != "..") {
                    components.pop();
                } else {
                    components.push(component);
                }
            }
            _ => components.push(component),
        }
    }
    if components.is_empty() {
        ".".to_owned()
    } else {
        components.join("/")
    }
}

fn check_template_directory(directory: &Path, root: &Path, create: bool) -> Result<bool> {
    let directory = absolute_path(directory)?;
    let root = absolute_path(root)?;
    if !directory.starts_with(&root) {
        return Err(ResourceError::PromptDirectoryEscapes { directory, root });
    }

    if let Some(existing) = deepest_existing_at_or_below(&root, &directory)? {
        let resolved_root = fs::canonicalize(&root)
            .map_err(|error| io_error("resolve prompt root", &root, error))?;
        let resolved_existing = fs::canonicalize(&existing)
            .map_err(|error| io_error("resolve prompt directory", &existing, error))?;
        if !resolved_existing.starts_with(&resolved_root) {
            return Err(ResourceError::PromptDirectoryEscapes { directory, root });
        }
    }

    if !create
        && matches!(
            fs::symlink_metadata(&directory),
            Err(error) if error.kind() == io::ErrorKind::NotFound
        )
    {
        return Ok(false);
    }
    if create {
        fs::create_dir_all(&directory)
            .map_err(|error| io_error("create prompt directory", &directory, error))?;
    }

    let metadata = fs::symlink_metadata(&directory)
        .map_err(|error| io_error("inspect prompt directory", &directory, error))?;
    if !metadata.file_type().is_dir() && !metadata.file_type().is_symlink() {
        return Err(ResourceError::UnsafePromptDestination {
            path: directory,
            reason: "the prompt directory is not a directory",
        });
    }
    let resolved_root =
        fs::canonicalize(&root).map_err(|error| io_error("resolve prompt root", &root, error))?;
    let resolved_directory = fs::canonicalize(&directory)
        .map_err(|error| io_error("resolve prompt directory", &directory, error))?;
    if !resolved_directory.starts_with(&resolved_root) {
        return Err(ResourceError::PromptDirectoryEscapes { directory, root });
    }
    Ok(true)
}

fn deepest_existing_at_or_below(root: &Path, directory: &Path) -> Result<Option<PathBuf>> {
    let mut probe = root.to_path_buf();
    match fs::symlink_metadata(&probe) {
        Ok(_) => {}
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io_error("inspect prompt root", probe, error)),
    }
    let mut deepest = probe.clone();
    let remainder = directory
        .strip_prefix(root)
        .expect("directory was checked as rooted");
    for component in remainder.components() {
        if !matches!(component, Component::Normal(_)) {
            continue;
        }
        probe.push(component.as_os_str());
        match fs::symlink_metadata(&probe) {
            Ok(_) => deepest = probe.clone(),
            Err(error) if error.kind() == io::ErrorKind::NotFound => break,
            Err(error) => return Err(io_error("inspect prompt directory", probe, error)),
        }
    }
    Ok(Some(deepest))
}

fn atomic_write(path: &Path, contents: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| ResourceError::UnsafePromptDestination {
            path: path.to_path_buf(),
            reason: "it has no parent directory",
        })?;
    let file_name = path
        .file_name()
        .ok_or_else(|| ResourceError::UnsafePromptDestination {
            path: path.to_path_buf(),
            reason: "it has no file name",
        })?;

    let mut temporary_file = None;
    for _ in 0..32 {
        let sequence = TEMP_FILE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{}.{}.{}.tmp",
            file_name.to_string_lossy(),
            process::id(),
            sequence
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        match options.open(&candidate) {
            Ok(file) => {
                temporary_file = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io_error("create temporary prompt", candidate, error)),
        }
    }
    let (temporary, mut file) = temporary_file.ok_or_else(|| {
        io_error(
            "create temporary prompt",
            parent,
            io::Error::new(
                io::ErrorKind::AlreadyExists,
                "could not allocate a unique temporary prompt name",
            ),
        )
    })?;
    let write_result = (|| -> Result<()> {
        file.write_all(contents)
            .map_err(|error| io_error("write temporary prompt", &temporary, error))?;
        file.sync_all()
            .map_err(|error| io_error("sync temporary prompt", &temporary, error))?;
        drop(file);
        fs::rename(&temporary, path).map_err(|error| io_error("replace prompt", path, error))?;
        #[cfg(unix)]
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .map_err(|error| io_error("set prompt permissions", path, error))?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

fn safe_directory_for_read(
    directory: &Path,
    root: &Path,
    kind: &str,
    warnings: &mut Vec<String>,
) -> bool {
    let metadata = match fs::symlink_metadata(directory) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return false,
        Err(error) => {
            warnings.push(format!("{kind} {}: {error}", display_path(directory)));
            return false;
        }
    };
    if metadata.file_type().is_symlink() {
        warnings.push(format!(
            "{kind} {} skipped: it is a symbolic link",
            display_path(directory)
        ));
        return false;
    }
    if !metadata.file_type().is_dir() {
        warnings.push(format!(
            "{kind} {} skipped: it is not a directory",
            display_path(directory)
        ));
        return false;
    }
    match resolves_within(directory, root) {
        Ok(true) => true,
        Ok(false) => {
            warnings.push(format!(
                "{kind} {} skipped: it resolves outside {}",
                display_path(directory),
                display_path(root)
            ));
            false
        }
        Err(error) => {
            warnings.push(format!("{kind} {}: {error}", display_path(directory)));
            false
        }
    }
}

fn sorted_directory_entries(
    directory: &Path,
    action: &'static str,
    warnings: &mut Vec<String>,
) -> Option<Vec<fs::DirEntry>> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => {
            warnings.push(format!("{action} {}: {error}", display_path(directory)));
            return None;
        }
    };
    let mut collected = Vec::new();
    for entry in entries {
        match entry {
            Ok(entry) => collected.push(entry),
            Err(error) => warnings.push(format!("{action} {}: {error}", display_path(directory))),
        }
    }
    collected.sort_by_key(|entry| entry.file_name());
    Some(collected)
}

fn read_safe_resource(
    path: &Path,
    root: &Path,
    kind: &str,
    warnings: &mut Vec<String>,
) -> Option<String> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return None,
        Err(error) => {
            warnings.push(format!("{kind} {}: {error}", display_path(path)));
            return None;
        }
    };
    if metadata.file_type().is_symlink() {
        warnings.push(format!(
            "{kind} {} skipped: it is a symbolic link",
            display_path(path)
        ));
        return None;
    }
    if !metadata.file_type().is_file() {
        warnings.push(format!(
            "{kind} {} skipped: it is not a regular file",
            display_path(path)
        ));
        return None;
    }
    match resolves_within(path, root) {
        Ok(true) => {}
        Ok(false) => {
            warnings.push(format!(
                "{kind} {} skipped: it resolves outside {}",
                display_path(path),
                display_path(root)
            ));
            return None;
        }
        Err(error) => {
            warnings.push(format!("{kind} {}: {error}", display_path(path)));
            return None;
        }
    }
    match read_limited_text(path) {
        Ok(content) => Some(content),
        Err(error) => {
            warnings.push(format!("{kind} {}: {error}", display_path(path)));
            None
        }
    }
}

fn is_safe_regular_file(path: &Path, root: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|metadata| {
        metadata.file_type().is_file() && resolves_within(path, root).unwrap_or(false)
    })
}

fn resolves_within(path: &Path, root: &Path) -> io::Result<bool> {
    let root = fs::canonicalize(root)?;
    let path = fs::canonicalize(path)?;
    Ok(path.starts_with(root))
}

enum BoundedReadError {
    Io(io::Error),
    TooLarge,
}

impl fmt::Display for BoundedReadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::TooLarge => write!(formatter, "resource exceeds {MAX_RESOURCE_BYTES} bytes"),
        }
    }
}

fn read_limited_text(path: &Path) -> std::result::Result<String, BoundedReadError> {
    let file = File::open(path).map_err(BoundedReadError::Io)?;
    let mut bytes = Vec::with_capacity(MAX_RESOURCE_BYTES.min(8192));
    file.take((MAX_RESOURCE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(BoundedReadError::Io)?;
    if bytes.len() > MAX_RESOURCE_BYTES {
        return Err(BoundedReadError::TooLarge);
    }
    Ok(String::from_utf8_lossy(&bytes).trim().to_owned())
}

fn markdown_stem(file_name: &str) -> Option<&str> {
    let (stem, extension) = file_name.rsplit_once('.')?;
    extension.eq_ignore_ascii_case("md").then_some(stem)
}

fn first_non_empty_line(text: &str) -> String {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default()
        .to_owned()
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "t" | "true" | "yes" | "y"
    )
}

fn single_line(value: &str) -> String {
    value.replace(['\r', '\n'], " ").trim().to_owned()
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(normalize_absolute(path));
    }
    let current = env::current_dir()
        .map_err(|error| io_error("determine the current directory", ".", error))?;
    Ok(normalize_absolute(&current.join(path)))
}

fn normalize_absolute(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    normalized.push(component.as_os_str());
                }
            }
            Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

/// Prevents untrusted context from closing the XML-ish system-prompt framing.
pub fn seal_delimiters(content: &str) -> String {
    content
        .replace("</project_instructions>", "<\u{200b}/project_instructions>")
        .replace("<project_instructions", "<\u{200b}project_instructions")
        .replace("</project_context>", "<\u{200b}/project_context>")
        .replace("<project_context>", "<\u{200b}project_context>")
}

fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn tool_snippet(name: &str) -> &str {
    match name {
        "read" => "Read file contents",
        "write" => "Create or overwrite files",
        "edit" => "Apply exact text replacements",
        "bash" => "Run shell commands",
        "grep" => "Search file contents (respects .gitignore)",
        "find" => "Find files by glob pattern (respects .gitignore)",
        "ls" => "List directory contents",
        "web_search" => "Search the web and return cited sources",
        _ => name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct Scratch(PathBuf);

    impl Scratch {
        fn new(label: &str) -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock is after epoch")
                .as_nanos();
            let path = env::temp_dir().join(format!(
                "goshcoder-resources-{label}-{}-{nonce}-{}",
                process::id(),
                TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).expect("create scratch directory");
            Self(path)
        }

        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_file(path: &Path, content: &str) {
        fs::create_dir_all(path.parent().expect("fixture path has a parent"))
            .expect("create fixture parent");
        fs::write(path, content).expect("write fixture");
    }

    fn paths(workspace: &Path, agent_dir: &Path) -> ResourcePaths {
        ResourcePaths::new(workspace, agent_dir)
            .expect("resource paths")
            .without_user_home()
    }

    #[test]
    fn context_discovery_honors_override_and_stops_at_repository_boundary() {
        let outer = Scratch::new("context-boundary");
        let repository = outer.path().join("repository");
        let workspace = repository.join("nested").join("src");
        let agent_dir = outer.path().join("agent");
        write_file(&outer.path().join("AGENTS.md"), "outside repository");
        write_file(
            &repository.join(".git").join("HEAD"),
            "ref: refs/heads/main",
        );
        write_file(&repository.join("AGENTS.md"), "repository instructions");
        write_file(&workspace.join("AGENTS.md"), "ordinary child instructions");
        write_file(
            &workspace.join("AGENTS.override.md"),
            "child override instructions",
        );
        write_file(&agent_dir.join("AGENTS.md"), "user instructions");

        let set = discover(&paths(&workspace, &agent_dir)).expect("discover resources");
        let contents = set
            .context_files
            .iter()
            .map(|file| file.content.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            contents,
            [
                "user instructions",
                "repository instructions",
                "child override instructions"
            ]
        );
        assert!(!contents.iter().any(|content| content.contains("outside")));
        assert!(
            !contents
                .iter()
                .any(|content| content.contains("ordinary child"))
        );
    }

    #[test]
    fn system_files_are_prioritized_and_reported() {
        let root = Scratch::new("system");
        let workspace = root.path().join("workspace");
        let agent_dir = root.path().join("agent");
        write_file(&agent_dir.join("SYSTEM.md"), "user system");
        write_file(&workspace.join(".pi").join("SYSTEM.md"), "workspace system");
        write_file(&agent_dir.join("APPEND_SYSTEM.md"), "user append");
        write_file(
            &workspace.join(".pi").join("APPEND_SYSTEM.md"),
            "workspace append",
        );

        let resource_paths = paths(&workspace, &agent_dir);
        let set = discover(&resource_paths).expect("discover resources");
        assert_eq!(set.custom_system, "user system");
        assert_eq!(set.append_system, "user append\n\nworkspace append");
        assert_eq!(
            set.custom_system_source.as_deref(),
            Some(agent_dir.join("SYSTEM.md").as_path())
        );
        let report = set.report(&resource_paths).render();
        assert!(report.contains("Custom system:"));
        assert!(report.contains("Context files: 0"));
    }

    #[test]
    fn templates_parse_frontmatter_and_expand_arguments_without_recursion() {
        let root = Scratch::new("expansion");
        let workspace = root.path().join("workspace");
        let agent_dir = root.path().join("agent");
        write_file(
            &workspace.join(".pi").join("prompts").join("review.md"),
            "---\r\ndescription: Review code\r\nargument-hint: <focus>\r\n---\r\nReview $1 and ${2:-tests}: $@ / ${@:2:2}",
        );
        let set = discover(&paths(&workspace, &agent_dir)).expect("discover resources");
        let template = set
            .find_template("review")
            .expect("discover review template");
        assert_eq!(template.description, "Review code");
        assert_eq!(template.argument_hint, "<focus>");

        let expanded = set
            .expand_input(r#"/review auth "edge cases" more"#)
            .expect("expand template");
        assert_eq!(
            expanded.text(),
            Some("Review auth and edge cases: auth edge cases more / edge cases more")
        );
        assert_eq!(
            split_arguments(r#"one "two words" 'three words' four\ words"#).expect("split"),
            ["one", "two words", "three words", "four words"]
        );
        assert!(matches!(
            split_arguments("'missing"),
            Err(ResourceError::UnterminatedQuote)
        ));
        assert_eq!(
            expand_template("value: $1", &["$ARGUMENTS".to_owned()]),
            "value: $ARGUMENTS"
        );
        assert_eq!(
            expand_template(
                "$${@:2:2} ${@:2:2}",
                &["one".into(), "two".into(), "three".into()]
            ),
            "${@:2:2} two three"
        );
        assert_eq!(
            expand_template(
                "${@:2:92233720368547758079223372036854775807}",
                &["one".into(), "two".into()]
            ),
            ""
        );
        assert_eq!(expand_template("${1:-main} ${@:-all}", &[]), "main all");
        assert!(has_placeholders("cost $$5; $1; ${@:2}"));
        assert!(!has_placeholders("ordinary $HOME text"));
    }

    #[test]
    fn literal_saves_preserve_dollars_and_remove_only_the_requested_scope() {
        let root = Scratch::new("save");
        let workspace = root.path().join("workspace");
        let agent_dir = root.path().join("agent");
        let resource_paths = paths(&workspace, &agent_dir);
        let literal = "run $1; echo $$; use $HOME";
        save_template(
            &resource_paths,
            "captured",
            literal,
            SaveTemplateOptions {
                literal: true,
                ..SaveTemplateOptions::default()
            },
        )
        .expect("save literal prompt");
        save_template(
            &resource_paths,
            "captured",
            "workspace copy",
            SaveTemplateOptions {
                scope: PromptScope::Workspace,
                ..SaveTemplateOptions::default()
            },
        )
        .expect("save workspace prompt");

        let set = discover(&resource_paths).expect("discover saved prompts");
        assert_eq!(
            set.expand_input("/captured alpha")
                .expect("expand saved literal")
                .text(),
            Some(literal)
        );
        let removed =
            remove_template(&resource_paths, "captured", PromptScope::User).expect("remove user");
        assert!(!removed.removed_symbolic_link);
        let set = discover(&resource_paths).expect("rediscover");
        assert_eq!(
            set.find_template("captured")
                .map(|template| template.body.as_str()),
            Some("workspace copy")
        );
    }

    #[test]
    fn skills_stay_inside_the_repository_and_require_descriptions() {
        let outer = Scratch::new("skills-boundary");
        let repository = outer.path().join("repository");
        let workspace = repository.join("nested");
        let agent_dir = outer.path().join("agent");
        write_file(
            &repository.join(".git").join("HEAD"),
            "ref: refs/heads/main",
        );
        write_file(
            &outer
                .path()
                .join(".agents")
                .join("skills")
                .join("outside")
                .join("SKILL.md"),
            "---\nname: outside\ndescription: must not load\n---\nbody",
        );
        write_file(
            &repository
                .join(".agents")
                .join("skills")
                .join("inside")
                .join("SKILL.md"),
            "---\nname: inside\ndescription: useful skill\n---\nbody",
        );
        write_file(
            &workspace
                .join(".goshcoder")
                .join("skills")
                .join("missing.md"),
            "# no frontmatter",
        );

        let set = discover(&paths(&workspace, &agent_dir)).expect("discover skills");
        assert_eq!(
            set.skills
                .iter()
                .map(|skill| skill.name.as_str())
                .collect::<Vec<_>>(),
            ["inside"]
        );
        assert!(
            set.warnings
                .iter()
                .any(|warning| warning.contains("has no description"))
        );
    }

    #[test]
    fn collection_restore_and_archive_round_trip_are_safe() {
        let root = Scratch::new("backup");
        let source_workspace = root.path().join("source-workspace");
        let source_agent = root.path().join("source-agent");
        let source = paths(&source_workspace, &source_agent);
        save_template(
            &source,
            "review",
            "Review carefully.",
            SaveTemplateOptions::default(),
        )
        .expect("save source prompt");
        let collection =
            collect_prompts(&source, &[PromptScope::User]).expect("collect source prompts");
        assert_eq!(collection.prompts.len(), 1);
        assert!(collection.warnings.is_empty());

        let destination = paths(
            &root.path().join("destination-workspace"),
            &root.path().join("destination-agent"),
        );
        let outcomes = restore_prompts(
            &destination,
            &collection.prompts,
            &RestoreOptions::default(),
        )
        .expect("restore prompt records");
        assert_eq!(outcomes.len(), 1);
        assert!(!outcomes[0].skipped);
        assert!(
            list_templates(&destination)
                .expect("list restored templates")
                .templates
                .iter()
                .any(|template| template.name == "review")
        );

        let mut bytes = Vec::new();
        write_archive(&mut bytes, &collection.prompts).expect("write archive");
        assert!(!bytes.is_empty());
        let mut archive = bytes.as_slice();
        let decoded = read_archive(&mut archive).expect("read archive");
        assert_eq!(decoded.manifest.version, ARCHIVE_VERSION);
        assert_eq!(decoded.manifest.tool, "goshcoder");
        assert_eq!(decoded.prompts, collection.prompts);
        assert!(decoded.warnings.is_empty());

        let mut input = &b"not an archive"[..];
        assert!(matches!(
            read_archive(&mut input),
            Err(ResourceError::ArchiveInvalid(_))
        ));
    }

    #[test]
    fn archive_reader_only_accepts_regular_markdown_prompts_in_known_scopes() {
        let manifest = serde_json::to_vec(&ArchiveManifest {
            version: ARCHIVE_VERSION,
            tool: "another-tool".to_owned(),
            created: "2026-01-01T00:00:00Z".to_owned(),
            prompts: 1,
        })
        .expect("serialize manifest");
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        {
            let mut archive = Builder::new(&mut encoder);
            append_test_archive_entry(
                &mut archive,
                ARCHIVE_MANIFEST_NAME,
                EntryType::Regular,
                &manifest,
            );
            append_test_archive_entry(
                &mut archive,
                "project/review.md",
                EntryType::Regular,
                b"Review the change.",
            );
            append_test_archive_entry(
                &mut archive,
                "untrusted/authorized_keys.md",
                EntryType::Regular,
                b"not a prompt",
            );
            append_test_archive_entry(&mut archive, "user/link.md", EntryType::Symlink, b"");
            archive.into_inner().expect("finish tar");
        }
        let bytes = encoder.finish().expect("finish gzip");
        let mut input = bytes.as_slice();
        let decoded = read_archive(&mut input).expect("read hostile archive safely");

        assert_eq!(
            decoded.prompts,
            vec![ArchivedPrompt {
                name: "review".to_owned(),
                scope: PromptScope::Workspace,
                body: "Review the change.".to_owned(),
            }]
        );
        assert!(
            decoded
                .warnings
                .iter()
                .any(|warning| warning.contains("not in a recognized archive directory"))
        );
        assert!(
            decoded
                .warnings
                .iter()
                .any(|warning| warning.contains("not a regular file"))
        );
    }

    #[test]
    fn archive_reader_refuses_a_newer_format_version() {
        let mut bytes = Vec::new();
        {
            let encoder = GzEncoder::new(&mut bytes, Compression::default());
            let mut archive = Builder::new(encoder);
            let manifest = serde_json::to_vec(&ArchiveManifest {
                version: ARCHIVE_VERSION + 1,
                ..ArchiveManifest::default()
            })
            .expect("serialize future manifest");
            append_test_archive_entry(
                &mut archive,
                ARCHIVE_MANIFEST_NAME,
                EntryType::Regular,
                &manifest,
            );
            let encoder = archive.into_inner().expect("finish tar");
            encoder.finish().expect("finish gzip");
        }
        let mut input = bytes.as_slice();
        assert!(matches!(
            read_archive(&mut input),
            Err(ResourceError::ArchiveFutureVersion {
                version,
                supported: ARCHIVE_VERSION
            }) if version == ARCHIVE_VERSION + 1
        ));
    }

    fn append_test_archive_entry<W: Write>(
        archive: &mut Builder<W>,
        path: &str,
        entry_type: EntryType,
        contents: &[u8],
    ) {
        let mut header = Header::new_gnu();
        header.set_path(path).expect("set test archive path");
        header.set_entry_type(entry_type);
        header.set_mode(0o600);
        header.set_size(contents.len() as u64);
        header.set_cksum();
        archive
            .append(&header, contents)
            .expect("append test archive entry");
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_context_templates_and_skills_never_leak_their_target() {
        use std::os::unix::fs::symlink;

        let root = Scratch::new("symlink-discovery");
        let workspace = root.path().join("workspace");
        let agent_dir = root.path().join("agent");
        let secret = root.path().join("private-key");
        write_file(&workspace.join(".git").join("HEAD"), "ref: refs/heads/main");
        write_file(&secret, "PRIVATE KEY MATERIAL");
        fs::create_dir_all(workspace.join(".goshcoder").join("prompts"))
            .expect("create prompt directory");
        fs::create_dir_all(workspace.join(".goshcoder").join("skills").join("leak"))
            .expect("create skill directory");
        fs::create_dir_all(workspace.join(".pi")).expect("create pi directory");
        write_file(&workspace.join("CLAUDE.md"), "safe fallback context");
        symlink(&secret, workspace.join("AGENTS.md")).expect("link context");
        symlink(
            &secret,
            workspace.join(".goshcoder").join("prompts").join("leak.md"),
        )
        .expect("link template");
        symlink(
            &secret,
            workspace
                .join(".goshcoder")
                .join("skills")
                .join("leak")
                .join("SKILL.md"),
        )
        .expect("link skill");
        symlink(&secret, workspace.join(".pi").join("SYSTEM.md")).expect("link system prompt");
        symlink(&secret, workspace.join(".pi").join("APPEND_SYSTEM.md"))
            .expect("link appended system prompt");

        let set = discover(&paths(&workspace, &agent_dir)).expect("discover resources");
        assert!(
            !set.context_files
                .iter()
                .any(|file| file.content.contains("PRIVATE"))
        );
        assert!(
            set.context_files
                .iter()
                .any(|file| file.content == "safe fallback context")
        );
        assert!(
            !set.templates
                .iter()
                .any(|template| template.body.contains("PRIVATE"))
        );
        assert!(
            !set.skills
                .iter()
                .any(|skill| skill.body.contains("PRIVATE"))
        );
        assert!(set.custom_system.is_empty());
        assert!(set.append_system.is_empty());
        assert!(
            set.warnings
                .iter()
                .filter(|warning| warning.contains("symbolic link"))
                .count()
                >= 5
        );
    }

    #[cfg(unix)]
    #[test]
    fn saves_and_restores_refuse_a_prompt_directory_redirected_outside_workspace() {
        use std::os::unix::fs::symlink;

        let root = Scratch::new("symlink-save");
        let workspace = root.path().join("workspace");
        let agent_dir = root.path().join("agent");
        let outside = root.path().join("outside");
        fs::create_dir_all(workspace.join(".goshcoder")).expect("create workspace config");
        fs::create_dir_all(&outside).expect("create outside directory");
        symlink(&outside, workspace.join(".goshcoder").join("prompts"))
            .expect("redirect prompt directory");

        let resource_paths = paths(&workspace, &agent_dir);
        let error = save_template(
            &resource_paths,
            "trap",
            "do not write outside",
            SaveTemplateOptions {
                scope: PromptScope::Workspace,
                ..SaveTemplateOptions::default()
            },
        )
        .expect_err("reject redirected prompt directory");
        assert!(matches!(
            error,
            ResourceError::PromptDirectoryEscapes { .. }
        ));
        assert!(!outside.join("trap.md").exists());
        let listing = list_templates(&resource_paths).expect("list untrusted directory");
        assert!(listing.templates.is_empty());
        assert!(
            listing
                .warnings
                .iter()
                .any(|warning| warning.contains("symbolic link"))
        );

        let outcomes = restore_prompts(
            &resource_paths,
            &[ArchivedPrompt {
                name: "trap".to_owned(),
                scope: PromptScope::Workspace,
                body: "also blocked".to_owned(),
            }],
            &RestoreOptions::default(),
        )
        .expect("restore reports a skipped record");
        assert_eq!(outcomes.len(), 1);
        assert!(outcomes[0].skipped);
        assert!(!outside.join("trap.md").exists());
    }

    #[cfg(unix)]
    #[test]
    fn save_refuses_to_overwrite_a_symlinked_prompt_file() {
        use std::os::unix::fs::symlink;

        let root = Scratch::new("symlink-leaf");
        let workspace = root.path().join("workspace");
        let agent_dir = root.path().join("agent");
        let resource_paths = paths(&workspace, &agent_dir);
        let prompt_directory = resource_paths.prompt_dir(PromptScope::User);
        fs::create_dir_all(&prompt_directory).expect("create prompt directory");
        let victim = root.path().join("victim");
        write_file(&victim, "leave this unchanged");
        symlink(&victim, prompt_directory.join("trap.md")).expect("link prompt");

        let error = save_template(
            &resource_paths,
            "trap",
            "replacement",
            SaveTemplateOptions {
                overwrite: true,
                ..SaveTemplateOptions::default()
            },
        )
        .expect_err("refuse symlink leaf");
        assert!(matches!(
            error,
            ResourceError::UnsafePromptDestination { .. }
        ));
        assert_eq!(
            fs::read_to_string(&victim).expect("read victim"),
            "leave this unchanged"
        );

        let removed = remove_template(&resource_paths, "trap", PromptScope::User)
            .expect("remove link itself");
        assert!(removed.removed_symbolic_link);
        assert_eq!(
            fs::read_to_string(&victim).expect("read victim"),
            "leave this unchanged"
        );
    }
}
