# DeCloud GPU Proxy — build daemon + shim
#
# The shim can be built in two modes:
# Prerequisites:
#   Daemon: CUDA toolkit (nvcc, libcudart, libcuda)
#   Shim:   Just a C compiler (no CUDA needed)
#
# Usage:
#   make              — build both daemon + runtime shim (requires CUDA)
#   make daemon       — build daemon only (requires CUDA)
#   make shim         — build runtime shim (dynamic, links host glibc)
#   make driver-shim  — build driver API shim (libcuda.so.1)
#   make nvml-shim    — build NVML shim (libnvidia-ml.so.1)
#   make cublas-stub  — build cuBLAS stub with correct version tags
#   make cublaslt-stub — build cuBLAS Lt stub (DT_NEEDED placeholder)
#   make pytorch-stub — build PyTorch compat stubs (libcuda_pytorch_stubs.so)
#   make all-shims    — build ALL shims + stubs (no Docker, host glibc)
#   make shim-compat  — build runtime shim in Ubuntu 20.04 container
#   make all-shims-compat — build ALL shims + stubs in container (preferred)
#   make pytorch-stub-compat — build PyTorch stubs in container
#   make test         — run protocol tests (no CUDA needed)
#   make clean        — remove build artifacts
#   make install      — install daemon + shim to /usr/local
#   make install-all-shims-compat — install compat shims to 9p share dir

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
BUILD_DIR      := build
DAEMON         := $(BUILD_DIR)/gpu-proxy-daemon
SHIM           := $(BUILD_DIR)/libdecloud_cuda_shim.so
SHIM_STATIC    := $(BUILD_DIR)/libdecloud_cuda_shim-static.so
SHIM_COMPAT    := $(BUILD_DIR)/libdecloud_cuda_shim-compat.so
DRIVER_SHIM    := $(BUILD_DIR)/libcuda.so.1
NVML_SHIM      := $(BUILD_DIR)/libnvidia-ml.so.1
CUBLAS_STUB    := $(BUILD_DIR)/libcublas_stub.so
CUBLASLT_STUB  := $(BUILD_DIR)/libcublasLt_stub.so
PYTORCH_STUB   := $(BUILD_DIR)/libcuda_pytorch_stubs.so

# Transport header dependency (shared by all shims)
TRANSPORT_DEPS := shim/transport.h shim/transport.c proto/gpu_proxy_proto.h

.PHONY: all daemon shim driver-shim nvml-shim cublas-stub cublaslt-stub pytorch-stub \
        all-shims shim-compat all-shims-compat pytorch-stub-compat \
        test clean \
        install install-shim-compat install-all-shims install-all-shims-compat

all: daemon shim

# ----------------------------------------------------------------
# all-shims: build ALL shims + stubs on host (no Docker).
# Used by install.sh when Docker is unavailable.
# ----------------------------------------------------------------
all-shims: shim driver-shim nvml-shim cublas-stub cublaslt-stub pytorch-stub
	@echo "All shims built (host glibc — may have compat issues on older distros):"
	@echo "  $(SHIM)            (Runtime API, LD_PRELOAD)"
	@echo "  $(DRIVER_SHIM)     (Driver API, dlopen target)"
	@echo "  $(NVML_SHIM)       (NVML, dlopen target)"
	@echo "  $(CUBLAS_STUB)     (cuBLAS stub, @@libcublas.so.12 tags)"
	@echo "  $(CUBLASLT_STUB)   (cuBLAS Lt stub, DT_NEEDED placeholder)"
	@echo "  $(PYTORCH_STUB)    (PyTorch compat stubs, @@libcudart.so.12 tags)"

# ----------------------------------------------------------------
# Daemon: links against CUDA runtime
# ----------------------------------------------------------------
daemon: $(DAEMON)

$(DAEMON): daemon/gpu_proxy_daemon.c proto/gpu_proxy_proto.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) \
		-I$(CUDA_INC) \
		-o $@ $< \
		-L$(CUDA_LIB) -lcuda -lcudart -lcublas -lpthread -ldl \
		-Wl,-rpath,$(CUDA_LIB)

# ----------------------------------------------------------------
# Runtime API Shim (LD_PRELOAD): no CUDA dependency
# ----------------------------------------------------------------
shim: $(SHIM)

$(SHIM): shim/cuda_shim.c shim/libcudart.version $(TRANSPORT_DEPS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC \
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
#
# FIX: Previous Makefile listed -ldl as a prerequisite instead of
# a linker flag, causing 'No rule to make target -ldl' errors.
# ----------------------------------------------------------------
cublas-stub: $(CUBLAS_STUB)

$(CUBLAS_STUB): stubs/cublas_stub.c stubs/libcublas.version proto/gpu_proxy_proto.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC \
		-o $@ $< \
		-Wl,-soname,libcublas.so.12 \
		-Wl,--version-script=stubs/libcublas.version \
		-ldl

# ----------------------------------------------------------------
# cuBLAS Lt Stub (libcublasLt.so.12): minimal placeholder that
# satisfies libggml-cuda.so's DT_NEEDED: libcublasLt.so.12.
# No versioned symbols needed — no actual cublasLt calls are made
# in the VM; real compute is proxied via the daemon on the host.
# ----------------------------------------------------------------
cublaslt-stub: $(CUBLASLT_STUB)

$(CUBLASLT_STUB): stubs/cublasLt_stub.c stubs/libcublasLt.version | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC \
		-Wl,-soname,libcublasLt.so.12 \
		-Wl,--version-script=stubs/libcublasLt.version \
		-o $@ $

# ----------------------------------------------------------------
# PyTorch Compatibility Stubs (libcuda_pytorch_stubs.so)
#
# Supplies CUDA Runtime symbols required by PyTorch 2.3.x, Stable
# Diffusion Forge, and bitsandbytes that are absent from the main
# libdecloud_cuda_shim.so:
#
#   libtorch_cuda.so         — cudaGraphInstantiateWithFlags,
#                              cudaGraphGetNodes, cudaGraphGetEdges, ...
#   libbitsandbytes_cuda121.so — cudaMemPrefetchAsync
#   libc10_cuda.so           — cudaMallocAsync, cudaFreeAsync,
#                              cudaStreamCreateWithPriority,
#                              cudaDeviceGetStreamPriorityRange, +9 more
#
# Must appear BEFORE libdecloud_cuda_shim.so in LD_PRELOAD so these
# specific @@libcudart.so.12 versioned symbols resolve here first.
# Safe for all workloads — stubs are pure no-ops for Ollama/ggml
# (those symbols are never called by the ggml CUDA backend).
# ----------------------------------------------------------------
pytorch-stub: $(PYTORCH_STUB)

$(PYTORCH_STUB): stubs/cuda_pytorch_stubs.c shim/libcudart.version | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC \
		-Wl,-soname,libcuda_pytorch_stubs.so \
		-Wl,--version-script=shim/libcudart.version \
		-o $@ $<

pytorch-stub-compat: | $(BUILD_DIR)
	docker run --rm -v $(CURDIR):/src -w /src ubuntu:20.04 \
	    sh -c "apt-get update -qq && \
	           apt-get install -y -qq gcc libc6-dev >/dev/null 2>&1 && \
	           gcc -Wall -Wextra -O2 -shared -fPIC \
	               -Wl,-soname,libcuda_pytorch_stubs.so \
	               -Wl,--version-script=shim/libcudart.version \
	               -o build/libcuda_pytorch_stubs.so \
	               stubs/cuda_pytorch_stubs.c"
	@echo "PyTorch compat stub built (glibc 2.31+): $(PYTORCH_STUB)"

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

# ----------------------------------------------------------------
# Build ALL shims + stubs in container for maximum glibc compat.
# This is the preferred build target for deployment.
#
# FIX: Added -I/src to cublas_stub.c build (includes proto header
# via ../proto/gpu_proxy_proto.h). Added cublasLt stub build.
# ----------------------------------------------------------------
all-shims-compat: | $(BUILD_DIR)
	docker run --rm -v $(CURDIR):/src -w /src ubuntu:20.04 \
	    sh -c "apt-get update -qq && \
	           apt-get install -y -qq gcc libc6-dev linux-libc-dev >/dev/null 2>&1 && \
	           echo '==> Building runtime API shim...' && \
	           gcc -Wall -Wextra -O2 -I/src -shared -fPIC \
	               -Wl,-soname,libcudart.so.12 \
	               -Wl,--version-script=shim/libcudart.version \
	               -o build/libdecloud_cuda_shim-compat.so \
	               shim/cuda_shim.c -lpthread -ldl && \
	           echo '==> Building driver API shim...' && \
	           gcc -Wall -Wextra -O2 -I/src -shared -fPIC \
	               -o build/libcuda.so.1 \
	               shim/cuda_driver_shim.c -lpthread -ldl && \
	           ln -sf libcuda.so.1 build/libcuda.so && \
	           echo '==> Building NVML shim...' && \
	           gcc -Wall -Wextra -O2 -I/src -shared -fPIC \
	               -o build/libnvidia-ml.so.1 \
	               shim/nvml_shim.c -lpthread && \
	           ln -sf libnvidia-ml.so.1 build/libnvidia-ml.so && \
	           echo '==> Building cuBLAS stub...' && \
	           gcc -Wall -Wextra -O2 -I/src -shared -fPIC \
	               -Wl,-soname,libcublas.so.12 \
	               -Wl,--version-script=stubs/libcublas.version \
	               -o build/libcublas_stub.so \
	               stubs/cublas_stub.c -ldl && \
	           echo '==> Building cuBLAS Lt stub...' && \
	           gcc -Wall -Wextra -O2 -shared -fPIC \
					-Wl,-soname,libcublasLt.so.12 \
					-Wl,--version-script=stubs/libcublasLt.version \
					-o build/libcublasLt_stub.so \
					stubs/cublasLt_stub.c && \
	           echo '==> Building PyTorch compat stubs...' && \
	           gcc -Wall -Wextra -O2 -shared -fPIC \
	               -Wl,-soname,libcuda_pytorch_stubs.so \
	               -Wl,--version-script=shim/libcudart.version \
	               -o build/libcuda_pytorch_stubs.so \
	               stubs/cuda_pytorch_stubs.c"
	@echo ""
	@echo "All compat shims built (glibc 2.31+, universal):"
	@echo "  $(BUILD_DIR)/libdecloud_cuda_shim-compat.so  (Runtime API, LD_PRELOAD)"
	@echo "  $(BUILD_DIR)/libcuda.so.1                    (Driver API, dlopen target)"
	@echo "  $(BUILD_DIR)/libnvidia-ml.so.1               (NVML, dlopen target)"
	@echo "  $(BUILD_DIR)/libcublas_stub.so               (cuBLAS stub, @@libcublas.so.12 tags)"
	@echo "  $(BUILD_DIR)/libcublasLt_stub.so             (cuBLAS Lt stub, DT_NEEDED placeholder)"
	@echo "  $(BUILD_DIR)/libcuda_pytorch_stubs.so        (PyTorch compat stubs, @@libcudart.so.12 tags)"

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
# Install: daemon + runtime shim (requires CUDA for daemon build)
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
#
# SECURITY: Shims go ONLY to decloud-gpu-shim/ (for VM delivery
# via 9p share). NEVER install to /usr/local/lib/ directly — that
# poisons the host's ldconfig cache and causes the daemon to load
# our shim instead of the real CUDA runtime (circular dependency:
# daemon -> shim -> tries to connect to daemon -> timeout).
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
	# cuBLAS Lt stub (DT_NEEDED placeholder)
	install -m 644 $(CUBLASLT_STUB) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcublasLt_stub.so
	# PyTorch compat stubs (must precede libdecloud_cuda_shim.so in LD_PRELOAD)
	install -m 644 $(PYTORCH_STUB) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcuda_pytorch_stubs.so
	# Convenience copies for VM deployment (Ollama/llama.cpp expect these names)
	cp $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libdecloud_cuda_shim.so \
	   $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcudart.so.12
	@echo "All compat shims installed to $(PREFIX)/lib/decloud-gpu-shim (VM delivery only)"

install-all-shims: all-shims
	install -d $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim
	install -m 644 $(SHIM_COMPAT) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libdecloud_cuda_shim.so
	install -m 644 $(DRIVER_SHIM) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcuda.so.1
	ln -sf libcuda.so.1 $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcuda.so
	install -m 644 $(NVML_SHIM) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libnvidia-ml.so.1
	ln -sf libnvidia-ml.so.1 $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libnvidia-ml.so
	install -m 644 $(CUBLAS_STUB) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcublas_stub.so
	install -m 644 $(CUBLASLT_STUB) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcublasLt_stub.so
	install -m 644 $(PYTORCH_STUB) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcuda_pytorch_stubs.so
	cp $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libdecloud_cuda_shim.so \
	   $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcudart.so.12
	@echo "All shims installed to $(PREFIX)/lib/decloud-gpu-shim (host glibc)"