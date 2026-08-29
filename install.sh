#!/bin/sh
# GoshCoder installer for Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/goshitsarch-eng/goshcoder/main/install.sh | sh
#
# By default this downloads the latest published release for your platform and
# verifies its SHA-256 against the signed checksums file before installing. If
# no release is published yet it falls back to building from source, which needs
# a Go toolchain. Re-running the script upgrades an existing installation.
#
# Options (also settable as environment variables):
#   --dir <path>       install location            (GOSHCODER_INSTALL_DIR)
#   --version <tag>    release tag to install      (GOSHCODER_VERSION)
#   --from-source      always build from source    (GOSHCODER_FROM_SOURCE=1)
#   --no-modify-path   do not offer to edit shell profiles
set -eu

REPO="goshitsarch-eng/goshcoder"
BINARY="goshcoder"
GO_MIN="1.26.6"

INSTALL_DIR="${GOSHCODER_INSTALL_DIR:-}"
VERSION="${GOSHCODER_VERSION:-latest}"
FROM_SOURCE="${GOSHCODER_FROM_SOURCE:-0}"
MODIFY_PATH=1

TMPDIR_CREATED=""
cleanup() { [ -n "$TMPDIR_CREATED" ] && rm -rf "$TMPDIR_CREATED"; }
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------------

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
	C_BOLD=$(printf '\033[1m'); C_RED=$(printf '\033[31m')
	C_GREEN=$(printf '\033[32m'); C_YELLOW=$(printf '\033[33m')
	C_DIM=$(printf '\033[2m'); C_OFF=$(printf '\033[0m')
else
	C_BOLD=''; C_RED=''; C_GREEN=''; C_YELLOW=''; C_DIM=''; C_OFF=''
fi

info() { printf '%s==>%s %s\n' "$C_BOLD" "$C_OFF" "$*"; }
warn() { printf '%swarning:%s %s\n' "$C_YELLOW" "$C_OFF" "$*" >&2; }
ok()   { printf '%s✓%s %s\n' "$C_GREEN" "$C_OFF" "$*"; }
die()  { printf '%serror:%s %s\n' "$C_RED" "$C_OFF" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------------------
# arguments
# ---------------------------------------------------------------------------

while [ $# -gt 0 ]; do
	case "$1" in
		--dir) [ $# -ge 2 ] || die "--dir needs a path"; INSTALL_DIR="$2"; shift 2 ;;
		--dir=*) INSTALL_DIR="${1#--dir=}"; shift ;;
		--version) [ $# -ge 2 ] || die "--version needs a tag"; VERSION="$2"; shift 2 ;;
		--version=*) VERSION="${1#--version=}"; shift ;;
		--from-source) FROM_SOURCE=1; shift ;;
		--no-modify-path) MODIFY_PATH=0; shift ;;
		-h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
		*) die "unknown option: $1 (try --help)" ;;
	esac
done

# ---------------------------------------------------------------------------
# platform detection
# ---------------------------------------------------------------------------

detect_platform() {
	os=$(uname -s 2>/dev/null || echo unknown)
	arch=$(uname -m 2>/dev/null || echo unknown)
	case "$os" in
		Linux) OS=linux ;;
		Darwin) OS=darwin ;;
		MINGW*|MSYS*|CYGWIN*)
			die "this script is for Linux and macOS; on Windows run install.ps1 in PowerShell" ;;
		*) die "unsupported operating system: $os" ;;
	esac
	case "$arch" in
		x86_64|amd64) ARCH=amd64 ;;
		arm64|aarch64) ARCH=arm64 ;;
		*) die "unsupported architecture: $arch (build from source with --from-source)" ;;
	esac
}

# ---------------------------------------------------------------------------
# install directory
# ---------------------------------------------------------------------------

choose_install_dir() {
	[ -n "$INSTALL_DIR" ] && return 0

	# Prefer a user-writable location already on PATH so the installer never
	# needs root and the binary is runnable without a profile edit.
	gobin=""
	gopath_bin=""
	if have go; then
		gobin=$(go env GOBIN 2>/dev/null || true)
		gopath=$(go env GOPATH 2>/dev/null || true)
		[ -n "$gopath" ] && gopath_bin="$gopath/bin"
	fi

	for candidate in "${GOBIN:-}" "$gobin" "$HOME/.local/bin" "$gopath_bin"; do
		[ -n "$candidate" ] || continue
		case ":$PATH:" in
			*":$candidate:"*) INSTALL_DIR="$candidate"; return 0 ;;
		esac
	done

	# Nothing suitable is on PATH; fall back to the XDG-style default and let
	# suggest_path() offer to add it.
	INSTALL_DIR="$HOME/.local/bin"
}

# ---------------------------------------------------------------------------
# download helpers
# ---------------------------------------------------------------------------

fetch() { # fetch <url> <dest>
	if have curl; then
		curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 -o "$2" "$1"
	elif have wget; then
		wget -q --tries=3 --timeout=20 -O "$2" "$1"
	else
		die "need curl or wget to download; install one, or use --from-source"
	fi
}

fetch_stdout() {
	if have curl; then
		curl -fsSL --retry 3 --retry-delay 2 --connect-timeout 20 "$1"
	elif have wget; then
		wget -q --tries=3 --timeout=20 -O - "$1"
	else
		die "need curl or wget to download; install one, or use --from-source"
	fi
}

resolve_version() {
	[ "$VERSION" != "latest" ] && return 0
	info "resolving the latest release"
	# Follow the releases/latest redirect rather than parsing the API, so the
	# script works without a GitHub token and without a JSON parser.
	if have curl; then
		resolved=$(curl -fsSLI -o /dev/null -w '%{url_effective}' \
			"https://github.com/$REPO/releases/latest" 2>/dev/null || true)
	else
		resolved=$(wget -q --max-redirect=10 -S -O /dev/null \
			"https://github.com/$REPO/releases/latest" 2>&1 |
			awk '/^  Location: /{print $2}' | tail -1 || true)
	fi
	case "$resolved" in
		*/releases/tag/*) VERSION="${resolved##*/tag/}" ;;
		*) return 1 ;;
	esac
	[ -n "$VERSION" ] || return 1
	return 0
}

verify_checksum() { # verify_checksum <file> <expected-sha256>
	actual=""
	if have sha256sum; then
		actual=$(sha256sum "$1" | cut -d' ' -f1)
	elif have shasum; then
		actual=$(shasum -a 256 "$1" | cut -d' ' -f1)
	else
		warn "no sha256sum or shasum available; skipping integrity verification"
		return 0
	fi
	[ "$actual" = "$2" ] || die "checksum mismatch for $(basename "$1")
  expected $2
  actual   $actual
Refusing to install a binary that does not match the published checksum."
	ok "checksum verified"
}

install_from_release() {
	resolve_version || return 1
	stripped="${VERSION#v}"
	archive="${BINARY}_${stripped}_${OS}_${ARCH}.tar.gz"
	base="https://github.com/$REPO/releases/download/$VERSION"

	info "downloading $BINARY $VERSION for $OS/$ARCH"
	fetch "$base/$archive" "$TMPDIR_CREATED/$archive" 2>/dev/null || return 1

	# The checksums file is what makes the download trustworthy over a plain
	# redirect chain; refuse to install if it is missing or does not list us.
	if fetch "$base/checksums.txt" "$TMPDIR_CREATED/checksums.txt" 2>/dev/null; then
		expected=$(awk -v f="$archive" '$2 == f || $2 == "*"f {print $1}' \
			"$TMPDIR_CREATED/checksums.txt" | head -1)
		[ -n "$expected" ] || die "checksums.txt does not list $archive"
		verify_checksum "$TMPDIR_CREATED/$archive" "$expected"
	else
		die "could not download checksums.txt; refusing to install unverified binary"
	fi

	tar -xzf "$TMPDIR_CREATED/$archive" -C "$TMPDIR_CREATED" ||
		die "could not extract $archive"
	found=$(find "$TMPDIR_CREATED" -type f -name "$BINARY" -perm -u+x 2>/dev/null | head -1)
	[ -n "$found" ] || found=$(find "$TMPDIR_CREATED" -type f -name "$BINARY" | head -1)
	[ -n "$found" ] || die "release archive did not contain a $BINARY binary"
	SOURCE_BINARY="$found"
	return 0
}

# ---------------------------------------------------------------------------
# source build
# ---------------------------------------------------------------------------

version_ge() { # version_ge <have> <want>
	[ "$1" = "$2" ] && return 0
	lower=$(printf '%s\n%s\n' "$1" "$2" | sort -t. -k1,1n -k2,2n -k3,3n | head -1)
	[ "$lower" = "$2" ]
}

install_from_source() {
	have go || die "building from source needs Go $GO_MIN or newer; install it from https://go.dev/dl/"

	srcdir=""
	if [ -f "./go.mod" ] && grep -q '^module goshcoder' ./go.mod 2>/dev/null; then
		srcdir="$(pwd)"
		info "building from the checkout in $srcdir"
	else
		have git || die "building from source needs git (or run this script from a checkout)"
		srcdir="$TMPDIR_CREATED/src"
		info "cloning $REPO"
		git clone --depth 1 "https://github.com/$REPO.git" "$srcdir" >/dev/null 2>&1 ||
			die "could not clone https://github.com/$REPO.git"
	fi

	# The requirement is whatever go.mod actually states, read from the tree
	# being built rather than from a constant here that can drift away from it.
	required=$(sed -n 's/^go \([0-9][0-9.]*\).*/\1/p' "$srcdir/go.mod" 2>/dev/null | head -1)
	[ -n "$required" ] || required="$GO_MIN"
	gover=$(go env GOVERSION 2>/dev/null | sed 's/^go//; s/[^0-9.].*$//')

	if [ -n "$gover" ] && ! version_ge "$gover" "$required"; then
		# go.mod states a hard minimum, not a preference: an older toolchain
		# refuses the build outright. Go resolves this itself when it is allowed
		# to, so ask it to -- GOTOOLCHAIN is exported for this build only, which
		# also overrides a distribution that pinned it to "local".
		info "Go $gover is older than the $required this project requires"
		info "fetching the Go $required toolchain (set GOTOOLCHAIN=local to refuse)"
		GOTOOLCHAIN="go${required}+auto"
		export GOTOOLCHAIN
	fi

	info "compiling (this takes a minute)"
	ver=$(cd "$srcdir" && git describe --tags --dirty --match 'v*' 2>/dev/null ||
		printf '0.4.0-dev+%s' "$(cd "$srcdir" && git rev-parse --short HEAD 2>/dev/null || echo unknown)")
	commit=$(cd "$srcdir" && git rev-parse HEAD 2>/dev/null || echo '')
	date=$(date -u +%Y-%m-%dT%H:%M:%SZ)
	if ! ( cd "$srcdir" && CGO_ENABLED=0 go build -trimpath \
		-ldflags "-s -w -X main.Version=$ver -X main.Commit=$commit -X main.BuildDate=$date" \
		-o "$TMPDIR_CREATED/$BINARY" ./cmd/goshcoder ); then
		if [ -n "$gover" ] && ! version_ge "$gover" "$required"; then
			die "build failed: this needs Go $required and $gover is installed.
  Either install Go $required from https://go.dev/dl/, or allow Go to fetch it
  by running this installer without GOTOOLCHAIN=local in the environment."
		fi
		die "build failed"
	fi
	SOURCE_BINARY="$TMPDIR_CREATED/$BINARY"
}

# ---------------------------------------------------------------------------
# PATH help
# ---------------------------------------------------------------------------

on_path() {
	case ":$PATH:" in *":$INSTALL_DIR:"*) return 0 ;; *) return 1 ;; esac
}

suggest_path() {
	on_path && return 0
	line="export PATH=\"$INSTALL_DIR:\$PATH\""
	warn "$INSTALL_DIR is not on your PATH"
	if [ "$MODIFY_PATH" -eq 0 ] || [ ! -t 0 ]; then
		printf '  add this to your shell profile:\n    %s\n' "$line"
		return 0
	fi
	profile=""
	case "${SHELL:-}" in
		*zsh) profile="$HOME/.zshrc" ;;
		*bash) [ -f "$HOME/.bashrc" ] && profile="$HOME/.bashrc" || profile="$HOME/.bash_profile" ;;
		*fish) profile="$HOME/.config/fish/config.fish"; line="fish_add_path $INSTALL_DIR" ;;
		*) profile="$HOME/.profile" ;;
	esac
	printf '  add %s to PATH in %s? [y/N] ' "$INSTALL_DIR" "$profile"
	read -r answer || answer=n
	case "$answer" in
		[yY]*)
			mkdir -p "$(dirname "$profile")"
			printf '\n# added by the goshcoder installer\n%s\n' "$line" >> "$profile"
			ok "updated $profile — run: source $profile" ;;
		*) printf '  add this to your shell profile:\n    %s\n' "$line" ;;
	esac
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

detect_platform
choose_install_dir
TMPDIR_CREATED=$(mktemp -d 2>/dev/null || mktemp -d -t goshcoder)

SOURCE_BINARY=""
if [ "$FROM_SOURCE" = "1" ]; then
	install_from_source
else
	if ! install_from_release; then
		warn "no published release found for $OS/$ARCH; falling back to a source build"
		install_from_source
	fi
fi

[ -n "$SOURCE_BINARY" ] || die "internal error: nothing was built or downloaded"

mkdir -p "$INSTALL_DIR" || die "could not create $INSTALL_DIR"
target="$INSTALL_DIR/$BINARY"

# Install through a temporary file in the destination directory and rename, so
# that upgrading while a session is running never leaves a half-written binary
# and never fails with ETXTBSY.
staged="$target.new.$$"
cp "$SOURCE_BINARY" "$staged" || die "could not write to $INSTALL_DIR (try --dir ~/.local/bin)"
chmod 0755 "$staged"
mv -f "$staged" "$target" || { rm -f "$staged"; die "could not install to $target"; }

ok "installed $target"
"$target" version 2>/dev/null || true

suggest_path

printf '\n%sNext steps%s\n' "$C_BOLD" "$C_OFF"
printf '  %s1.%s authenticate:  %sgoshcoder auth login anthropic%s\n' "$C_DIM" "$C_OFF" "$C_BOLD" "$C_OFF"
printf '     %s(or: goshcoder auth set anthropic, or export ANTHROPIC_API_KEY)%s\n' "$C_DIM" "$C_OFF"
printf '  %s2.%s check providers: %sgoshcoder providers%s\n' "$C_DIM" "$C_OFF" "$C_BOLD" "$C_OFF"
printf '  %s3.%s start coding:    %sgoshcoder%s\n' "$C_DIM" "$C_OFF" "$C_BOLD" "$C_OFF"
