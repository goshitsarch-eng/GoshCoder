<#
.SYNOPSIS
    Installs GoshCoder on Windows.

.DESCRIPTION
    Downloads the latest published release for this machine's architecture and
    verifies its SHA-256 against the published checksums file before installing.
    When no release is published it falls back to building from source, which
    requires a Go toolchain. Re-running upgrades an existing installation.

.PARAMETER InstallDir
    Where to place goshcoder.exe. Defaults to %LOCALAPPDATA%\Programs\goshcoder.

.PARAMETER Version
    Release tag to install. Defaults to the latest release.

.PARAMETER FromSource
    Always build from source instead of downloading a release.

.PARAMETER NoModifyPath
    Do not add the install directory to the user PATH.

.EXAMPLE
    irm https://raw.githubusercontent.com/goshitsarch-eng/goshcoder/main/install.ps1 | iex

.EXAMPLE
    .\install.ps1 -FromSource -InstallDir C:\tools\bin
#>
[CmdletBinding()]
param(
    [string] $InstallDir = $env:GOSHCODER_INSTALL_DIR,
    [string] $Version    = $(if ($env:GOSHCODER_VERSION) { $env:GOSHCODER_VERSION } else { 'latest' }),
    [switch] $FromSource,
    [switch] $NoModifyPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repo   = 'goshitsarch-eng/goshcoder'
$Binary = 'goshcoder'
$GoMin  = [version] '1.26.6'

function Write-Step { param([string] $Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Ok   { param([string] $Message) Write-Host "[ok] $Message" -ForegroundColor Green }
function Write-Warn { param([string] $Message) Write-Host "warning: $Message" -ForegroundColor Yellow }
function Stop-WithError { param([string] $Message) Write-Host "error: $Message" -ForegroundColor Red; exit 1 }

function Test-Command { param([string] $Name) [bool] (Get-Command $Name -ErrorAction SilentlyContinue) }

# ---------------------------------------------------------------------------
# platform
# ---------------------------------------------------------------------------

function Get-TargetArch {
    # PROCESSOR_ARCHITECTURE reports the *process* architecture, which is x86
    # for a 32-bit PowerShell on a 64-bit machine; the W6432 variant reports
    # the machine, so prefer it when present.
    $raw = $env:PROCESSOR_ARCHITEW6432
    if (-not $raw) { $raw = $env:PROCESSOR_ARCHITECTURE }
    switch ($raw) {
        'AMD64' { return 'amd64' }
        'ARM64' { return 'arm64' }
        'x86'   { Stop-WithError '32-bit Windows is not supported' }
        default { Stop-WithError "unsupported architecture: $raw" }
    }
}

# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

function Resolve-LatestVersion {
    Write-Step 'resolving the latest release'
    try {
        # Follow the releases/latest redirect rather than calling the API, so
        # this works unauthenticated and without rate limits.
        $response = Invoke-WebRequest -Uri "https://github.com/$Repo/releases/latest" `
            -MaximumRedirection 0 -ErrorAction SilentlyContinue -UseBasicParsing
        $location = $response.Headers['Location']
    } catch {
        $location = $_.Exception.Response.Headers['Location']
    }
    if ($location -and $location -match '/tag/(?<tag>[^/]+)$') { return $Matches['tag'] }
    return $null
}

function Get-FileHashHex { param([string] $Path) (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant() }

function Install-FromRelease {
    param([string] $Arch, [string] $Workdir)

    if ($Version -eq 'latest') {
        $resolved = Resolve-LatestVersion
        if (-not $resolved) { return $null }
        $script:Version = $resolved
    }

    $stripped = $Version -replace '^v', ''
    $archive  = "${Binary}_${stripped}_windows_${Arch}.zip"
    $base     = "https://github.com/$Repo/releases/download/$Version"
    $zipPath  = Join-Path $Workdir $archive
    $sumPath  = Join-Path $Workdir 'checksums.txt'

    Write-Step "downloading $Binary $Version for windows/$Arch"
    try {
        Invoke-WebRequest -Uri "$base/$archive" -OutFile $zipPath -UseBasicParsing
    } catch {
        return $null
    }

    # The checksums file is what makes the download trustworthy; refuse to
    # install anything it does not vouch for.
    try {
        Invoke-WebRequest -Uri "$base/checksums.txt" -OutFile $sumPath -UseBasicParsing
    } catch {
        Stop-WithError 'could not download checksums.txt; refusing to install an unverified binary'
    }

    $expected = $null
    foreach ($line in Get-Content $sumPath) {
        $parts = $line -split '\s+', 2
        if ($parts.Count -eq 2 -and ($parts[1].TrimStart('*')) -eq $archive) {
            $expected = $parts[0].ToLowerInvariant()
            break
        }
    }
    if (-not $expected) { Stop-WithError "checksums.txt does not list $archive" }

    $actual = Get-FileHashHex $zipPath
    if ($actual -ne $expected) {
        Stop-WithError "checksum mismatch for $archive`n  expected $expected`n  actual   $actual`nRefusing to install a binary that does not match the published checksum."
    }
    Write-Ok 'checksum verified'

    $extractDir = Join-Path $Workdir 'extract'
    Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
    $exe = Get-ChildItem -Path $extractDir -Recurse -Filter "$Binary.exe" | Select-Object -First 1
    if (-not $exe) { Stop-WithError "release archive did not contain $Binary.exe" }
    return $exe.FullName
}

# ---------------------------------------------------------------------------
# source build
# ---------------------------------------------------------------------------

function Install-FromSource {
    param([string] $Workdir)

    if (-not (Test-Command 'go')) {
        Stop-WithError "building from source needs Go $GoMin or newer; install it from https://go.dev/dl/"
    }

    $goVersionRaw = (& go env GOVERSION) -replace '^go', '' -replace '[^0-9.].*$', ''
    try {
        if ([version] $goVersionRaw -lt $GoMin) {
            Write-Warn "Go $goVersionRaw is older than the recommended $GoMin"
            Write-Warn 'earlier releases carry standard-library vulnerabilities reachable from this tree'
        }
    } catch { }

    $srcDir = $null
    if ((Test-Path './go.mod') -and ((Get-Content './go.mod' -First 1) -match '^module goshcoder')) {
        $srcDir = (Get-Location).Path
        Write-Step "building from the checkout in $srcDir"
    } else {
        if (-not (Test-Command 'git')) {
            Stop-WithError 'building from source needs git (or run this script from a checkout)'
        }
        $srcDir = Join-Path $Workdir 'src'
        Write-Step "cloning $Repo"
        & git clone --depth 1 "https://github.com/$Repo.git" $srcDir 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) { Stop-WithError "could not clone https://github.com/$Repo.git" }
    }

    Write-Step 'compiling (this takes a minute)'
    Push-Location $srcDir
    try {
        $ver = & git describe --tags --dirty --match 'v*' 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $ver) {
            $short = & git rev-parse --short HEAD 2>$null
            if (-not $short) { $short = 'unknown' }
            $ver = "0.2.0-dev+$short"
        }
        $commit = & git rev-parse HEAD 2>$null
        if (-not $commit) { $commit = '' }
        $date = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')

        $out = Join-Path $Workdir "$Binary.exe"
        $env:CGO_ENABLED = '0'
        & go build -trimpath `
            -ldflags "-s -w -X main.Version=$ver -X main.Commit=$commit -X main.BuildDate=$date" `
            -o $out ./cmd/goshcoder
        if ($LASTEXITCODE -ne 0) { Stop-WithError 'build failed' }
        return $out
    } finally {
        Pop-Location
    }
}

# ---------------------------------------------------------------------------
# PATH
# ---------------------------------------------------------------------------

function Add-ToUserPath {
    param([string] $Directory)

    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (-not $userPath) { $userPath = '' }
    $entries = $userPath -split ';' | Where-Object { $_ -ne '' }
    foreach ($entry in $entries) {
        if ($entry.TrimEnd('\') -ieq $Directory.TrimEnd('\')) {
            return $false  # already present
        }
    }
    $updated = if ($userPath) { "$userPath;$Directory" } else { $Directory }
    [Environment]::SetEnvironmentVariable('Path', $updated, 'User')
    # Also update this session so the caller can run goshcoder immediately.
    $env:Path = "$env:Path;$Directory"
    return $true
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if (-not $InstallDir) {
    $InstallDir = Join-Path $env:LOCALAPPDATA 'Programs\goshcoder'
}

$arch    = Get-TargetArch
$workdir = Join-Path ([System.IO.Path]::GetTempPath()) ("goshcoder-install-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $workdir -Force | Out-Null

try {
    $source = $null
    if ($FromSource) {
        $source = Install-FromSource -Workdir $workdir
    } else {
        $source = Install-FromRelease -Arch $arch -Workdir $workdir
        if (-not $source) {
            Write-Warn "no published release found for windows/$arch; falling back to a source build"
            $source = Install-FromSource -Workdir $workdir
        }
    }
    if (-not $source) { Stop-WithError 'internal error: nothing was built or downloaded' }

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    $target = Join-Path $InstallDir "$Binary.exe"

    # Windows refuses to overwrite a running executable, so move any existing
    # binary aside first; the leftover is cleaned up on the next upgrade.
    if (Test-Path $target) {
        $backup = "$target.old"
        Remove-Item $backup -Force -ErrorAction SilentlyContinue
        try {
            Move-Item -Path $target -Destination $backup -Force
        } catch {
            Stop-WithError "could not replace $target - close any running goshcoder session and retry"
        }
    }
    Copy-Item -Path $source -Destination $target -Force
    Remove-Item "$target.old" -Force -ErrorAction SilentlyContinue

    Write-Ok "installed $target"
    & $target version

    if (-not $NoModifyPath) {
        if (Add-ToUserPath -Directory $InstallDir) {
            Write-Ok "added $InstallDir to your user PATH (restart other terminals to pick it up)"
        }
    } else {
        Write-Warn "PATH not modified; add $InstallDir yourself to run goshcoder from anywhere"
    }

    Write-Host ''
    Write-Host 'Next steps' -ForegroundColor White
    Write-Host '  1. authenticate:    goshcoder auth login anthropic'
    Write-Host '     (or: goshcoder auth set anthropic, or $env:ANTHROPIC_API_KEY = "...")'
    Write-Host '  2. check providers: goshcoder providers'
    Write-Host '  3. start coding:    goshcoder'
} finally {
    Remove-Item -Recurse -Force $workdir -ErrorAction SilentlyContinue
}
