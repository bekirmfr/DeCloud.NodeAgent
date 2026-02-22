# DeCloud GPU Proxy — build daemon + shim
#
# Prerequisites:
#   Daemon: CUDA toolkit (nvcc, libcudart, libcuda)
#   Shim:   Just a C compiler (no CUDA needed)
#
# Usage:
#   make              — build both
#   make daemon       — build daemon only (requires CUDA)
#   make shim         — build shim only (no CUDA needed)
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

.PHONY: all daemon shim clean install

all: daemon shim

# ----------------------------------------------------------------
# Daemon: links against CUDA runtime
# ----------------------------------------------------------------
daemon: $(DAEMON)

$(DAEMON): daemon/gpu_proxy_daemon.c proto/gpu_proxy_proto.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) \
		-I$(CUDA_INC) \
		-o $@ $< \
		-L$(CUDA_LIB) -lcudart -lpthread \
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
