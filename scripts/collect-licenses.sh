#!/bin/sh
# Collect the licence text of every Go module linked into the binary.
#
# A Go binary statically links its dependencies, and MIT and BSD alike require
# the copyright notice to accompany the software. A release archive carrying
# only GoshCoder's own LICENSE is therefore missing notices it is obliged to
# ship. go.sum cannot stand in for them: it records module versions and
# integrity hashes, not licence texts, and the builder's module cache is not
# part of the archive.
#
# Only modules that survive into the build are collected -- the list comes from
# `go list -deps` for the target platform rather than from go.mod -- so a
# platform-specific dependency is included exactly where it is linked and
# nowhere else.
#
# A dependency with no discoverable licence fails the build. Shipping one whose
# terms are unknown is the failure this script exists to prevent, and a warning
# nobody reads would not prevent it.
#
# Usage: collect-licenses.sh <goos> <goarch> <destdir>
set -eu

goos="$1"
goarch="$2"
dest="$3"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir -p "$dest"

GOOS="$goos" GOARCH="$goarch" go list -deps \
	-f '{{if .Module}}{{.Module.Path}}|{{.Module.Version}}|{{.Module.Dir}}{{end}}' \
	./cmd/goshcoder | sort -u >"$work/modules"

# Read from a file rather than a pipe: a pipeline runs the loop in a subshell,
# where a recorded failure could not escape to the check below.
while IFS='|' read -r path version dir; do
	# The main module is the thing being licensed rather than a dependency,
	# and the standard library ships with the toolchain, not in the archive.
	[ -n "$path" ] || continue
	[ -n "$dir" ] || continue
	[ "$path" != "goshcoder" ] || continue

	find "$dir" -maxdepth 1 -type f \
		\( -iname 'LICENSE*' -o -iname 'LICENCE*' -o -iname 'COPYING*' \
		-o -iname 'NOTICE*' \) >"$work/found"

	if [ ! -s "$work/found" ]; then
		# A few modules declare their terms in a README and ship no licence
		# file. Those are recorded by hand under license-exceptions, so the
		# notice in the archive is a decision somebody made and can audit
		# rather than something this script inferred from prose.
		exception="$(dirname "$0")/license-exceptions/$path"
		if [ -d "$exception" ]; then
			find "$exception" -maxdepth 1 -type f >"$work/found"
		fi
	fi

	if [ ! -s "$work/found" ]; then
		echo "collect-licenses: no licence file for $path in $dir" >&2
		printf '%s\n' "$path" >>"$work/missing"
		continue
	fi

	mkdir -p "$dest/$path"
	while IFS= read -r candidate; do
		cp "$candidate" "$dest/$path/$(basename "$candidate")"
	done <"$work/found"
	printf '%s %s\n' "$path" "$version" >>"$work/index"
done <"$work/modules"

if [ -f "$work/missing" ]; then
	echo "collect-licenses: refusing to build an archive with unlicensed dependencies" >&2
	exit 1
fi

{
	echo "Third-party Go modules linked into this build ($goos/$goarch)."
	echo "Each module's licence text is alongside it in this directory."
	echo
	sort -u "$work/index"
} >"$dest/modules.txt"
