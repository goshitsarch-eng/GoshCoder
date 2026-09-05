#!/bin/sh
# Collect licence text for Rust crates included in a release archive.
#
# Cargo.lock records checksums and SPDX expressions but does not contain the
# notices that must accompany the binary. Cargo metadata identifies the exact
# resolved dependency graph; each package's registry source contains its
# declared licence file or a conventional top-level licence notice.
#
# A crate with no discoverable notice fails the build. Shipping an archive with
# unknown terms would be a release defect, not a warning.
#
# Usage: collect-licenses.sh <rust-target> <destdir>
set -eu

target="$1"
dest="$2"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$dest"

command -v python3 >/dev/null 2>&1 || {
	echo "collect-licenses: Python 3 is required to read Cargo metadata" >&2
	exit 1
}

cargo metadata --format-version 1 --locked --filter-platform "$target" >"$work/metadata.json"

python3 - "$work/metadata.json" "$dest" "$target" <<'PY'
import json
import pathlib
import re
import shutil
import sys

metadata_path = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
target = sys.argv[3]
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
packages = {package["id"]: package for package in metadata["packages"]}
resolve = metadata.get("resolve") or {}
root = resolve.get("root")
nodes = {node["id"]: node for node in resolve.get("nodes", [])}

if not root:
    raise SystemExit("collect-licenses: Cargo metadata did not identify a root package")

reachable = set()
pending = [root]
while pending:
    package_id = pending.pop()
    if package_id in reachable:
        continue
    reachable.add(package_id)
    node = nodes.get(package_id, {})
    pending.extend(node.get("dependencies", []))

notice_names = ("LICENSE*", "LICENCE*", "COPYING*", "NOTICE*")
missing = []
index = []
exceptions_root = pathlib.Path("scripts/license-exceptions")

for package_id in sorted(reachable):
    if package_id == root:
        continue
    package = packages.get(package_id)
    if package is None:
        missing.append(f"{package_id} (missing metadata)")
        continue

    package_root = pathlib.Path(package["manifest_path"]).parent
    candidates = []
    declared = package.get("license_file")
    if declared:
        declared_path = pathlib.Path(declared)
        if not declared_path.is_absolute():
            declared_path = package_root / declared_path
        if declared_path.is_file():
            candidates.append(declared_path)
    if not candidates:
        for pattern in notice_names:
            candidates.extend(path for path in package_root.glob(pattern) if path.is_file())
    if not candidates:
        exception = exceptions_root / package["name"]
        if exception.is_dir():
            candidates.extend(path for path in exception.iterdir() if path.is_file())
    candidates = sorted(set(candidates))
    if not candidates:
        missing.append(f'{package["name"]} {package["version"]} ({package_root})')
        continue

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", f'{package["name"]}-{package["version"]}')
    crate_destination = destination / safe_name
    crate_destination.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        shutil.copy2(candidate, crate_destination / candidate.name)
    source = package.get("source") or "path"
    index.append(f'{package["name"]} {package["version"]} {source}')

if missing:
    for package in missing:
        print(f"collect-licenses: no licence file for {package}", file=sys.stderr)
    raise SystemExit(
        "collect-licenses: refusing to build an archive with unlicensed dependencies"
    )

(destination / "modules.txt").write_text(
    "Third-party Rust crates resolved for this build "
    f"({target}).\nEach crate's licence text is alongside it in this directory.\n\n"
    + "\n".join(index)
    + "\n",
    encoding="utf-8",
)
PY
