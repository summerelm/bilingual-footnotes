.PHONY: architecture bootstrap build check journey lock sync test test-seams

ALIGNMENT_PACKAGE := packages/bilingual-text-align
EPUB_PACKAGE := packages/epub-bilingual-footnotes
UI_PACKAGE := packages/bilingual-text-align-ui
UV := uv --cache-dir $(CURDIR)/.uv-cache
PACKAGE_PATHS := $(CURDIR)/$(ALIGNMENT_PACKAGE)/src:$(CURDIR)/$(EPUB_PACKAGE)/src:$(CURDIR)/$(UI_PACKAGE)/src

bootstrap: sync
	git config --local core.hooksPath .githooks

sync:
	$(UV) sync --locked --all-packages --all-groups

check: lock
	$(MAKE) -C $(ALIGNMENT_PACKAGE) lint type coverage
	$(MAKE) -C $(EPUB_PACKAGE) lint type coverage
	$(MAKE) -C $(UI_PACKAGE) lint type coverage
	$(MAKE) architecture
	$(MAKE) test-seams

lock:
	$(UV) lock --check

test:
	$(MAKE) -C $(ALIGNMENT_PACKAGE) test
	$(MAKE) -C $(EPUB_PACKAGE) test
	$(MAKE) -C $(UI_PACKAGE) test

journey:
	$(MAKE) -C $(EPUB_PACKAGE) journey

architecture:
	PYTHONPATH=$(PACKAGE_PATHS) $(UV) run --locked --package epub-bilingual-footnotes lint-imports --config $(CURDIR)/.importlinter --cache-dir $(CURDIR)/.import_linter_cache

test-seams:
	python3 tools/check_test_seams.py

build:
	$(MAKE) -C $(ALIGNMENT_PACKAGE) build
	$(MAKE) -C $(EPUB_PACKAGE) build-self
	$(MAKE) -C $(UI_PACKAGE) build-self
