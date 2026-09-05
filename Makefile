# GoshCoder Rust build and release automation.
#
# `make` builds a stamped Rust binary into bin/. `make check` is the same
# formatting, lint, test, and advisory gate used by CI.

SHELL := /bin/sh
CARGO ?= cargo
BIN_DIR ?= bin
BINARY := goshcoder
TARGET_DIR ?= $(CURDIR)/target

# A tagged build reports the tag; an untagged one reports the development
# revision, so a local binary never claims to be a published release.
VERSION ?= $(shell git describe --tags --dirty --match 'v*' 2>/dev/null \
	|| printf '0.5.0-dev+%s' "$$(git rev-parse --short HEAD 2>/dev/null || echo unknown)")

# Release archives are named without the tag's leading "v", which is what both
# installers derive from a release tag.
DIST_VERSION := $(VERSION:v%=%)

# Platforms produced by `make dist`. cargo-zigbuild supplies the native C
# toolchains required to link Rust targets from the release builder.
PLATFORMS := \
	linux/amd64 linux/arm64 \
	darwin/amd64 darwin/arm64 \
	windows/amd64 windows/arm64

INSTALL_DIR ?= $(if $(CARGO_INSTALL_ROOT),$(CARGO_INSTALL_ROOT)/bin,$(HOME)/.cargo/bin)

.PHONY: all build install uninstall run clean check fmt fmt-check vet test test-race \
        test-hermetic cover lint vuln tools dist dist-name checksums clean-dist help

all: build

## build: compile a release Rust binary into bin/
build:
	@mkdir -p "$(BIN_DIR)"
	CARGO_TARGET_DIR="$(TARGET_DIR)" GOSHCODER_VERSION="$(VERSION)" \
		$(CARGO) build --release --locked --bin "$(BINARY)"
	@cp "$(TARGET_DIR)/release/$(BINARY)" "$(BIN_DIR)/$(BINARY)"
	@echo "built $(BIN_DIR)/$(BINARY) ($(VERSION))"

## install: build and copy the binary onto PATH
install: build
	@mkdir -p "$(INSTALL_DIR)"
	@cp "$(BIN_DIR)/$(BINARY)" "$(INSTALL_DIR)/$(BINARY)"
	@echo "installed $(INSTALL_DIR)/$(BINARY)"
	@command -v "$(BINARY)" >/dev/null 2>&1 || \
		echo "warning: $(INSTALL_DIR) is not on PATH; add it to your shell profile"

## uninstall: remove the installed binary
uninstall:
	@rm -f "$(INSTALL_DIR)/$(BINARY)"
	@echo "removed $(INSTALL_DIR)/$(BINARY)"

## run: build and start an interactive session
run: build
	@"$(BIN_DIR)/$(BINARY)"

## check: the full Rust formatting, check, lint, test, and advisory gate
check: fmt-check vet lint test test-hermetic vuln
	@echo "all checks passed"

## fmt: rewrite Rust sources with rustfmt
fmt:
	$(CARGO) fmt --all

## fmt-check: fail if Rust sources are not rustfmt-clean
fmt-check:
	$(CARGO) fmt --all -- --check

## vet: type-check all Rust targets without producing a release binary
vet:
	CARGO_TARGET_DIR="$(TARGET_DIR)" $(CARGO) check --workspace --all-targets --locked

## test: run all Rust unit and integration tests
test:
	CARGO_TARGET_DIR="$(TARGET_DIR)" $(CARGO) test --workspace --all-targets --locked

## test-race: compatibility alias kept for older scripts; `test` is the gate
test-race: test

## test-hermetic: prove tests ignore ambient provider credentials
test-hermetic:
	AWS_ACCESS_KEY_ID=hermeticity-probe \
	AWS_SECRET_ACCESS_KEY=hermeticity-probe \
	AWS_REGION=hermeticity-probe \
	AWS_PROFILE=hermeticity-probe \
		CARGO_TARGET_DIR="$(TARGET_DIR)" $(CARGO) test --workspace --all-targets --locked

## cover: run coverage when cargo-llvm-cov is installed
cover:
	@if command -v cargo-llvm-cov >/dev/null 2>&1; then \
		CARGO_TARGET_DIR="$(TARGET_DIR)" cargo llvm-cov --workspace --all-targets; \
	else \
		echo "cargo-llvm-cov is not installed; run 'make tools'"; \
	fi

## tools: install current Rust audit, coverage, and cross-build tools
tools:
	$(CARGO) install cargo-audit --locked
	$(CARGO) install cargo-llvm-cov --locked
	$(CARGO) install cargo-zigbuild --locked

## lint: fail on every Clippy warning for every Rust target
lint:
	CARGO_TARGET_DIR="$(TARGET_DIR)" $(CARGO) clippy --workspace --all-targets --all-features --locked -- -D warnings

## vuln: check Cargo.lock against RustSec advisories when cargo-audit is installed
vuln:
	@if command -v cargo-audit >/dev/null 2>&1; then \
		cargo audit; \
	else \
		echo "cargo-audit is not installed; run 'make tools'"; \
	fi

## dist: cross-compile signed-release archive contents for every platform
dist: clean-dist
	@command -v cargo-zigbuild >/dev/null 2>&1 || { \
		echo "cargo-zigbuild is required for cross-platform releases; run 'make tools'"; exit 1; \
	}
	@mkdir -p dist
	@for platform in $(PLATFORMS); do \
		os=$${platform%/*}; arch=$${platform#*/}; \
		case "$$platform" in \
			linux/amd64) target=x86_64-unknown-linux-musl ;; \
			linux/arm64) target=aarch64-unknown-linux-musl ;; \
			darwin/amd64) target=x86_64-apple-darwin ;; \
			darwin/arm64) target=aarch64-apple-darwin ;; \
			windows/amd64) target=x86_64-pc-windows-gnu ;; \
			windows/arm64) target=aarch64-pc-windows-gnu ;; \
			*) echo "unsupported release platform $$platform" >&2; exit 1 ;; \
		esac; \
		ext=""; [ "$$os" = "windows" ] && ext=".exe"; \
		out="dist/$(BINARY)_$(DIST_VERSION)_$${os}_$${arch}"; \
		echo "building $$out$$ext"; \
		CARGO_TARGET_DIR="$(TARGET_DIR)/dist" GOSHCODER_VERSION="$(VERSION)" \
			$(CARGO) zigbuild --release --locked --target "$$target" --bin "$(BINARY)" || exit 1; \
		mkdir -p "$$out"; \
		cp "$(TARGET_DIR)/dist/$$target/release/$(BINARY)$$ext" "$$out/$(BINARY)$$ext" || exit 1; \
		cp README.md NOTICE LICENSE "$$out/"; \
		sh scripts/collect-licenses.sh "$$target" "$$out/licenses" || exit 1; \
		base=$$(basename "$$out"); \
		if [ "$$os" = "windows" ]; then \
			(cd dist && zip -qr "$$base.zip" "$$base"); \
		else \
			(cd dist && tar czf "$$base.tar.gz" "$$base"); \
		fi; \
		rm -rf "$$out"; \
	done
	@$(MAKE) --no-print-directory checksums

## dist-name: print the archive basename the installers expect (used by CI)
dist-name:
	@printf '%s_%s_%s_%s\n' '$(BINARY)' '$(DIST_VERSION)' "$${OS:-linux}" "$${ARCH:-amd64}"

## checksums: write SHA-256 sums for release archives
checksums:
	@cd dist && (sha256sum *.tar.gz *.zip 2>/dev/null || shasum -a 256 *.tar.gz *.zip) \
		| awk '$$2 != "checksums.txt" && $$2 != "*checksums.txt"' > checksums.txt
	@echo "wrote dist/checksums.txt"

clean-dist:
	@rm -rf dist "$(TARGET_DIR)/dist"

## clean: remove build output
clean: clean-dist
	@rm -rf "$(BIN_DIR)" "$(TARGET_DIR)" coverage.out
	@echo "cleaned"

## help: list available targets
help:
	@awk '/^## / { sub(/^## /, "  "); print }' $(MAKEFILE_LIST) | sort
