PACKAGE_NAME := bilingual-text-align
include ../../tools/package.mk

.PHONY: build

PROJECT_VERSION := $(shell $(UV) version --package $(PACKAGE_NAME) --short)
WHEEL := dist/bilingual_text_align-$(PROJECT_VERSION)-py3-none-any.whl

build:
	$(UV) build --package $(PACKAGE_NAME) --out-dir $(CURDIR)/dist
	$(UV_BASE) run --no-cache --isolated --no-project --with ./$(WHEEL) bilingual-align --help
	$(UV_BASE) run --no-cache --isolated --no-project --with ./$(WHEEL) bilingual-align-setup --help
	$(UV_BASE) run --no-cache --isolated --no-project --with ./$(WHEEL) bilingual-align-resources --help
