# GoshCoder build and release automation.
#
# `make` builds a stamped binary into bin/. `make install` puts it on PATH.
# `make check` is the gate that CI runs and that should pass before any commit.

SHELL := /bin/sh
GO ?= go
BIN_DIR ?= bin
BINARY := goshcoder
PKG := ./cmd/goshcoder

# Minimum toolchain. Earlier Go 1.26 patch releases carry standard-library
# vulnerabilities that govulncheck reports as reachable from this tree.
GO_MIN_VERSION := 1.26.6

# A tagged build reports the tag; an untagged one reports the development
# version with the short revision appended, so no build ever claims to be a
# release it is not.
VERSION ?= $(shell git describe --tags --dirty --match 'v*' 2>/dev/null \
	|| printf '0.2.0-dev+%s' "$$(git rev-parse --short HEAD 2>/dev/null || echo unknown)")
COMMIT ?= $(shell git rev-parse HEAD 2>/dev/null)
BUILD_DATE ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)

LDFLAGS := -s -w \
	-X main.Version=$(VERSION) \
	-X main.Commit=$(COMMIT) \
	-X main.BuildDate=$(BUILD_DATE)

# Platforms produced by `make dist`.
PLATFORMS := \
	linux/amd64 linux/arm64 \
	darwin/amd64 darwin/arm64 \
	windows/amd64 windows/arm64

INSTALL_DIR ?= $(if $(GOBIN),$(GOBIN),$(shell $(GO) env GOPATH)/bin)

.PHONY: all build install uninstall run clean check fmt fmt-check vet test test-race \
        cover lint vuln tools dist checksums help

all: build

## build: compile a stamped binary into bin/
build:
	@mkdir -p $(BIN_DIR)
	$(GO) build -trimpath -ldflags '$(LDFLAGS)' -o $(BIN_DIR)/$(BINARY) $(PKG)
	@echo "built $(BIN_DIR)/$(BINARY) ($(VERSION))"

## install: build and copy the binary onto PATH
install: build
	@mkdir -p '$(INSTALL_DIR)'
	@cp $(BIN_DIR)/$(BINARY) '$(INSTALL_DIR)/$(BINARY)'
	@echo "installed $(INSTALL_DIR)/$(BINARY)"
	@command -v $(BINARY) >/dev/null 2>&1 || \
		echo "warning: $(INSTALL_DIR) is not on PATH; add it to your shell profile"

## uninstall: remove the installed binary
uninstall:
	@rm -f '$(INSTALL_DIR)/$(BINARY)'
	@echo "removed $(INSTALL_DIR)/$(BINARY)"

## run: build and start an interactive session
run: build
	@$(BIN_DIR)/$(BINARY)

## check: the full gate - format, vet, lint, tests with the race detector, vulns
check: fmt-check vet lint test-race vuln
	@echo "all checks passed"

## fmt: rewrite sources with gofmt
fmt:
	$(GO) fmt ./...

## fmt-check: fail if any file is not gofmt-clean
fmt-check:
	@unformatted=$$(gofmt -l ./cmd ./internal); \
	if [ -n "$$unformatted" ]; then \
		echo "gofmt needed:"; echo "$$unformatted"; exit 1; \
	fi

## vet: run the full go vet analyser set
vet:
	$(GO) vet -all ./...

## test: run the test suite
#
# AWS_* is cleared because the Bedrock credential-resolution tests assert the
# behaviour of an unconfigured environment; ambient credentials from a CI
# runner or a developer shell would otherwise make them fail spuriously.
test:
	env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
		-u AWS_PROFILE -u AWS_REGION -u AWS_DEFAULT_REGION \
		$(GO) test -count=1 ./...

## test-race: run the test suite under the race detector
test-race:
	env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
		-u AWS_PROFILE -u AWS_REGION -u AWS_DEFAULT_REGION \
		$(GO) test -race -count=1 ./...

## cover: write a coverage profile and print the per-package summary
cover:
	env -u AWS_ACCESS_KEY_ID -u AWS_SECRET_ACCESS_KEY -u AWS_SESSION_TOKEN \
		$(GO) test -count=1 -coverprofile=coverage.out ./...
	$(GO) tool cover -func=coverage.out | tail -1

## tools: install the optional external analysers into GOPATH/bin
tools:
	$(GO) install honnef.co/go/tools/cmd/staticcheck@latest
	$(GO) install golang.org/x/vuln/cmd/govulncheck@latest

## lint: run staticcheck (skipped with a warning when not installed)
lint:
	@if command -v staticcheck >/dev/null 2>&1; then \
		staticcheck ./...; \
	elif [ -x "$$($(GO) env GOPATH)/bin/staticcheck" ]; then \
		"$$($(GO) env GOPATH)/bin/staticcheck" ./...; \
	else \
		echo "staticcheck not installed; run 'make tools'"; \
	fi

## vuln: run govulncheck (skipped with a warning when not installed)
vuln:
	@if command -v govulncheck >/dev/null 2>&1; then \
		govulncheck ./...; \
	elif [ -x "$$($(GO) env GOPATH)/bin/govulncheck" ]; then \
		"$$($(GO) env GOPATH)/bin/govulncheck" ./...; \
	else \
		echo "govulncheck not installed; run 'make tools'"; \
	fi

## dist: cross-compile release archives for every supported platform
dist: clean-dist
	@mkdir -p dist
	@for platform in $(PLATFORMS); do \
		os=$${platform%/*}; arch=$${platform#*/}; \
		ext=""; [ "$$os" = "windows" ] && ext=".exe"; \
		out="dist/$(BINARY)_$(VERSION)_$${os}_$${arch}"; \
		echo "building $$out$$ext"; \
		mkdir -p "$$out"; \
		GOOS=$$os GOARCH=$$arch CGO_ENABLED=0 \
			$(GO) build -trimpath -ldflags '$(LDFLAGS)' -o "$$out/$(BINARY)$$ext" $(PKG) || exit 1; \
		cp README.md NOTICE "$$out/"; \
		if [ "$$os" = "windows" ]; then \
			(cd dist && zip -qr "$$(basename $$out).zip" "$$(basename $$out)"); \
		else \
			(cd dist && tar czf "$$(basename $$out).tar.gz" "$$(basename $$out)"); \
		fi; \
		rm -rf "$$out"; \
	done
	@$(MAKE) --no-print-directory checksums

## checksums: write SHA-256 sums for the built archives
checksums:
	@cd dist && (sha256sum * 2>/dev/null || shasum -a 256 *) > checksums.txt.tmp \
		&& grep -v checksums.txt < checksums.txt.tmp > checksums.txt \
		&& rm -f checksums.txt.tmp
	@echo "wrote dist/checksums.txt"

.PHONY: clean-dist
clean-dist:
	@rm -rf dist

## clean: remove build output
clean: clean-dist
	@rm -rf $(BIN_DIR) coverage.out
	@echo "cleaned"

## help: list available targets
help:
	@grep -hE '^## ' $(MAKEFILE_LIST) | sed 's/^## /  /' | sort
