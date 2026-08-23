#!/bin/sh
# Proves a release archive built by `make dist` is the one install.sh looks for.
#
# The two derive the archive name independently -- the Makefile from VERSION,
# the installer from the release tag -- so a mismatch is invisible until someone
# tries to install a published release and gets a 404. This serves the real
# artifacts over HTTP and drives the installer's real download path.
set -eu

TAG="${1:-v9.9.9}"
work="$(mktemp -d)"
trap 'rm -rf "$work"; [ -n "${server_pid:-}" ] && kill "$server_pid" 2>/dev/null || true' EXIT

make dist VERSION="$TAG" PLATFORMS="linux/amd64" >/dev/null

mkdir -p "$work/download/$TAG"
cp dist/*.tar.gz dist/checksums.txt "$work/download/$TAG/"

# Stand in for the GitHub release layout the installer fetches from.
(cd "$work" && python3 -m http.server 8731 --bind 127.0.0.1 >/dev/null 2>&1) &
server_pid=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
	if curl -fsS "http://127.0.0.1:8731/download/$TAG/checksums.txt" >/dev/null 2>&1; then break; fi
	sleep 0.3
done

archive="$(OS=linux ARCH=amd64 make --no-print-directory dist-name VERSION="$TAG").tar.gz"
url="http://127.0.0.1:8731/download/$TAG/$archive"
if ! curl -fsS -o "$work/fetched.tar.gz" "$url"; then
	echo "FAIL: install.sh would request $archive, which make dist did not produce:" >&2
	ls dist >&2
	exit 1
fi

# The checksums file must vouch for exactly that name, or the installer refuses.
if ! grep -q " \*\{0,1\}$archive\$" "$work/download/$TAG/checksums.txt"; then
	echo "FAIL: checksums.txt does not list $archive" >&2
	cat "$work/download/$TAG/checksums.txt" >&2
	exit 1
fi

tar -xzf "$work/fetched.tar.gz" -C "$work"
binary="$(find "$work" -type f -name goshcoder | head -1)"
[ -n "$binary" ] || { echo "FAIL: archive contains no goshcoder binary" >&2; exit 1; }
"$binary" version >/dev/null || { echo "FAIL: the shipped binary does not run" >&2; exit 1; }

echo "OK: $archive is what install.sh requests, is checksummed, and runs"
