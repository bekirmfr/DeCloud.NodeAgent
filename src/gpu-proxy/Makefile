# DeCloud GPU Proxy — build daemon + shim
#
# The shim can be built in two modes:
# Prerequisites:
#   Daemon: CUDA toolkit (nvcc, libcudart, libcuda)
#   Shim:   Just a C compiler (no CUDA needed)
#
# Usage:
#   make              — build both
#   make daemon       — build daemon only (requires CUDA)
#   make shim         — build shim (dynamic, links host glibc)
#   make shim-compat  — build shim (Docker Ubuntu 20.04, universal glibc compat)
#   make clean        — remove build artifacts
#   make install      — install to /usr/local

CC       ?= gcc
NVCC     ?= nvcc
CFLAGS   := -Wall -Wextra -O2 -g
LDFLAGS  :=

# CUDA paths (override with CUDA_HOME=/path/to/cuda)
CUDA_HOME ?= /usr/local/cuda
CUDA_INC   = $(CUDA_HOME)/include
CUDA_LIB   = $(CUDA_HOME)/lib64

# Output
BUILD_DIR := build
DAEMON    := $(BUILD_DIR)/gpu-proxy-daemon
SHIM      := $(BUILD_DIR)/libdecloud_cuda_shim.so
SHIM_STATIC := $(BUILD_DIR)/libdecloud_cuda_shim-static.so

.PHONY: all daemon shim shim-compat test clean install

all: daemon shim

# ----------------------------------------------------------------
# Daemon: links against CUDA runtime
# ----------------------------------------------------------------
daemon: $(DAEMON)

$(DAEMON): daemon/gpu_proxy_daemon.c proto/gpu_proxy_proto.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) \
		-I$(CUDA_INC) \
		-o $@ $< \
		-L$(CUDA_LIB) -lcuda -lcudart -lpthread -ldl \
		-Wl,-rpath,$(CUDA_LIB)

# ----------------------------------------------------------------
# Shim: no CUDA dependency, just vsock + LD_PRELOAD
# ----------------------------------------------------------------
shim: $(SHIM)

$(SHIM): shim/cuda_shim.c proto/gpu_proxy_proto.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC \
		-o $@ $< \
		-lpthread

# ----------------------------------------------------------------
# Shim (compat): built inside Ubuntu 20.04 container for universal
# glibc compatibility. Requires: docker
# ----------------------------------------------------------------
SHIM_COMPAT := $(BUILD_DIR)/libdecloud_cuda_shim-compat.so

.PHONY: shim-compat

shim-compat: | $(BUILD_DIR)
	docker run --rm -v $(CURDIR):/src -w /src ubuntu:20.04 \
	    sh -c "apt-get update -qq && \
	           apt-get install -y -qq gcc libc6-dev linux-libc-dev >/dev/null 2>&1 && \
	           gcc -Wall -Wextra -O2 -g -shared -fPIC \
	               -o build/libdecloud_cuda_shim-compat.so \
	               shim/cuda_shim.c -lpthread"
	@echo "Compat shim built: $(SHIM_COMPAT) (glibc 2.31+, universal)"

# ----------------------------------------------------------------
# Tests: no CUDA dependency — runs on any machine
# ----------------------------------------------------------------
TEST_PROTO := $(BUILD_DIR)/test_proto

test: $(TEST_PROTO)
	@echo "Running protocol tests..."
	@./$(TEST_PROTO)

$(TEST_PROTO): tests/test_proto.c proto/gpu_proxy_proto.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<

# ----------------------------------------------------------------
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

# ----------------------------------------------------------------
# Install
# ----------------------------------------------------------------
PREFIX ?= /usr/local

install: all
	install -d $(DESTDIR)$(PREFIX)/bin
	install -d $(DESTDIR)$(PREFIX)/lib
	install -m 755 $(DAEMON) $(DESTDIR)$(PREFIX)/bin/gpu-proxy-daemon
	install -m 644 $(SHIM)   $(DESTDIR)$(PREFIX)/lib/libdecloud_cuda_shim.so

# ----------------------------------------------------------------
# Install static shim (preferred for 9p share distribution)
# ----------------------------------------------------------------
install-shim-compat: shim-compat
	install -d $(DESTDIR)$(PREFIX)/lib
	install -d $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim
	install -m 644 $(SHIM_COMPAT) $(DESTDIR)$(PREFIX)/lib/libdecloud_cuda_shim.so
	install -m 644 $(SHIM_COMPAT) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libdecloud_cuda_shim.so
	@echo "Compat shim installed to $(PREFIX)/lib and 9p share directory"