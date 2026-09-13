PACKAGE_NAME := epub-bilingual-footnotes
TEST_ARGS := -m "not journey"
include ../../tools/package.mk

.PHONY: build build-self journey

PROJECT_VERSION := $(shell $(UV) version --package $(PACKAGE_NAME) --short)
WHEEL := dist/epub_bilingual_footnotes-$(PROJECT_VERSION)-py3-none-any.whl
ALIGNMENT_PROJECT := ../bilingual-text-align
ALIGNMENT_VERSION := $(shell $(UV) version --package bilingual-text-align --short)
ALIGNMENT_WHEEL := $(ALIGNMENT_PROJECT)/dist/bilingual_text_align-$(ALIGNMENT_VERSION)-py3-none-any.whl
journey:
	$(UV) run --locked pytest --maxfail=1 -m journey tests/test_alice_journey.py
build:
	$(MAKE) -C $(ALIGNMENT_PROJECT) build
	$(MAKE) build-self
build-self:
	$(UV) build --package $(PACKAGE_NAME) --out-dir $(CURDIR)/dist
	$(UV_BASE) run --no-cache --isolated --no-project --with $(ALIGNMENT_WHEEL) --with ./$(WHEEL) bilingual-footnotes --help
