PACKAGE_NAME := bilingual-text-align-ui
include ../../tools/package.mk

.PHONY: build build-self

PROJECT_VERSION := $(shell $(UV) version --package $(PACKAGE_NAME) --short)
WHEEL := dist/bilingual_text_align_ui-$(PROJECT_VERSION)-py3-none-any.whl
ALIGNMENT_VERSION := $(shell $(UV) version --package bilingual-text-align --short)
ALIGNMENT_WHEEL := ../bilingual-text-align/dist/bilingual_text_align-$(ALIGNMENT_VERSION)-py3-none-any.whl
EPUB_VERSION := $(shell $(UV) version --package epub-bilingual-footnotes --short)
EPUB_WHEEL := ../epub-bilingual-footnotes/dist/epub_bilingual_footnotes-$(EPUB_VERSION)-py3-none-any.whl

build:
	$(MAKE) -C ../epub-bilingual-footnotes build
	$(MAKE) build-self
build-self:
	$(UV) build --package $(PACKAGE_NAME) --out-dir $(CURDIR)/dist
	$(UV_BASE) run --no-cache --isolated --no-project --with $(ALIGNMENT_WHEEL) --with $(EPUB_WHEEL) --with ./$(WHEEL) bilingual-align-ui --help
