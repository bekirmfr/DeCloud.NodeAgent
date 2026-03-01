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
#   make shim-compat  — build shim in Ubuntu 20.04 container (universal)
#   make all-shims-compat — build ALL shims (runtime + driver + nvml) in container
#   make clean        — remove build artifacts
#   make install      — install to /usr/local

CC       ?= gcc
MUSL_CC  ?= musl-gcc
NVCC     ?= nvcc
# -I. ensures #include "proto/..." resolves from project root, not source
# file directory. Critical for shim/cuda_shim.c which includes
# "proto/gpu_proxy_proto.h" — without -I., gcc only searches shim/proto/
# (relative to source file), which doesn't exist.
CFLAGS   := -Wall -Wextra -O2 -g -I.
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
SHIM_COMPAT := $(BUILD_DIR)/libdecloud_cuda_shim-compat.so
DRIVER_SHIM := $(BUILD_DIR)/libcuda.so.1
NVML_SHIM   := $(BUILD_DIR)/libnvidia-ml.so.1
CUBLAS_STUB := $(BUILD_DIR)/libcublas_stub.so

# Transport header dependency (shared by all shims)
TRANSPORT_DEPS := shim/transport.h shim/transport.c proto/gpu_proxy_proto.h

.PHONY: all daemon shim shim-compat all-shims-compat cublas-stub test clean install \
        install-shim-compat install-all-shims-compat

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
# Runtime API Shim (LD_PRELOAD): no CUDA dependency
# ----------------------------------------------------------------
shim: $(SHIM)

$(SHIM): shim/cuda_shim.c shim/libcudart.version $(TRANSPORT_DEPS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared \
		-fPIC \
		-o $@ $< \
		-Wl,-soname,libcudart.so.12 \
		-Wl,--version-script=shim/libcudart.version \
		-lpthread -ldl

# ----------------------------------------------------------------
# Driver API Shim (libcuda.so.1): no CUDA dependency
# Needs -ldl for dlsym/RTLD_DEFAULT used by cuGetProcAddress
# ----------------------------------------------------------------
driver-shim: $(DRIVER_SHIM)

$(DRIVER_SHIM): shim/cuda_driver_shim.c $(TRANSPORT_DEPS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC \
		-o $@ $< \
		-lpthread -ldl
	ln -sf libcuda.so.1 $(BUILD_DIR)/libcuda.so

# ----------------------------------------------------------------
# NVML Shim (libnvidia-ml.so.1): no CUDA dependency
# ----------------------------------------------------------------
nvml-shim: $(NVML_SHIM)

$(NVML_SHIM): shim/nvml_shim.c $(TRANSPORT_DEPS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC \
		-o $@ $< \
		-lpthread

# ----------------------------------------------------------------
# cuBLAS Stub (libcublas.so.12): separate library with correct
# @@libcublas.so.12 version tags. Required because libggml-cuda.so
# has DT_NEEDED: libcublas.so.12 with versioned symbol imports.
# Copying the cuda_shim binary as libcublas.so.12 gives wrong tags
# (@@libcudart.so.12) causing silent dlopen failure.
# See: GPU_PROXY_DEBUGGING_JOURNAL.md, Problem 17 (Day 4)
# ----------------------------------------------------------------
cublas-stub: $(CUBLAS_STUB)

$(CUBLAS_STUB): stubs/cublas_stub.c stubs/libcublas.version | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC \
		-o $@ $< \
		-Wl,-soname,libcublas.so.12 \
		-Wl,--version-script=stubs/libcublas.version

# ----------------------------------------------------------------
# Compat shims: built inside Ubuntu 20.04 container for universal
# glibc compatibility (glibc 2.31+). Requires: docker
#
# -I/src is the container equivalent of -I. — ensures
# "proto/gpu_proxy_proto.h" resolves from the mounted project root.
# ----------------------------------------------------------------

shim-compat: | $(BUILD_DIR)
	docker run --rm -v $(CURDIR):/src -w /src ubuntu:20.04 \
	    sh -c "apt-get update -qq && \
	           apt-get install -y -qq gcc libc6-dev linux-libc-dev >/dev/null 2>&1 && \
	           gcc -Wall -Wextra -O2 -I/src -shared -fPIC \
	               -Wl,-soname,libcudart.so.12 \
	               -Wl,--version-script=shim/libcudart.version \
	               -o build/libdecloud_cuda_shim-compat.so \
	               shim/cuda_shim.c -lpthread -ldl"
	@echo "Compat shim built: $(SHIM_COMPAT) (glibc 2.31+, universal)"

# Build ALL three shims in container for maximum glibc compatibility.
# This is the preferred build target for deployment.
all-shims-compat: | $(BUILD_DIR)
	docker run --rm -v $(CURDIR):/src -w /src ubuntu:20.04 \
	    sh -c "apt-get update -qq && \
	           apt-get install -y -qq gcc libc6-dev linux-libc-dev >/dev/null 2>&1 && \
	           gcc -Wall -Wextra -O2 -I/src -shared -fPIC \
	               -Wl,-soname,libcudart.so.12 \
	               -Wl,--version-script=shim/libcudart.version \
	               -o build/libdecloud_cuda_shim-compat.so \
	               shim/cuda_shim.c -lpthread -ldl && \
	           gcc -Wall -Wextra -O2 -I/src -shared -fPIC \
	               -o build/libcuda.so.1 \
	               shim/cuda_driver_shim.c -lpthread -ldl && \
	           ln -sf libcuda.so.1 build/libcuda.so && \
	           gcc -Wall -Wextra -O2 -I/src -shared -fPIC \
	               -o build/libnvidia-ml.so.1 \
	               shim/nvml_shim.c -lpthread && \
	           ln -sf libnvidia-ml.so.1 build/libnvidia-ml.so && \
	           gcc -Wall -Wextra -O2 -shared -fPIC \
	               -Wl,-soname,libcublas.so.12 \
	               -Wl,--version-script=stubs/libcublas.version \
	               -o build/libcublas_stub.so \
	               stubs/cublas_stub.c"
	@echo "All compat shims built (glibc 2.31+, universal):"
	@echo "  $(BUILD_DIR)/libdecloud_cuda_shim-compat.so  (Runtime API, LD_PRELOAD)"
	@echo "  $(BUILD_DIR)/libcuda.so.1                    (Driver API, dlopen target)"
	@echo "  $(BUILD_DIR)/libnvidia-ml.so.1               (NVML, dlopen target)"
	@echo "  $(BUILD_DIR)/libcublas_stub.so               (cuBLAS stub, @@libcublas.so.12 tags)"

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
	install -d $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim
	install -m 755 $(DAEMON) $(DESTDIR)$(PREFIX)/bin/gpu-proxy-daemon
	install -m 644 $(SHIM) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libdecloud_cuda_shim.so

# ----------------------------------------------------------------
# Install compat shim (runtime API only)
# ----------------------------------------------------------------
install-shim-compat: shim-compat
	install -d $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim
	install -m 644 $(SHIM_COMPAT) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libdecloud_cuda_shim.so
	@echo "Compat shim installed to $(PREFIX)/lib/decloud-gpu-shim (9p share only)"

# ----------------------------------------------------------------
# Install ALL shims (preferred for 9p share distribution)
# ----------------------------------------------------------------
install-all-shims-compat: all-shims-compat
	install -d $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim
	# Runtime API shim (VM delivery only — NEVER install to system lib paths)
	install -m 644 $(SHIM_COMPAT) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libdecloud_cuda_shim.so
	# Driver API shim (libcuda.so.1)
	install -m 644 $(DRIVER_SHIM) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcuda.so.1
	ln -sf libcuda.so.1 $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcuda.so
	# NVML shim (libnvidia-ml.so.1)
	install -m 644 $(NVML_SHIM) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libnvidia-ml.so.1
	ln -sf libnvidia-ml.so.1 $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libnvidia-ml.so
	# cuBLAS stub (separate build with correct @@libcublas.so.12 version tags)
	install -m 644 $(CUBLAS_STUB) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcublas_stub.so
	# Convenience copies for VM deployment
	cp $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libdecloud_cuda_shim.so \
	   $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcudart.so.12
	@echo "All compat shims installed to $(PREFIX)/lib/decloud-gpu-shim (VM delivery only)"