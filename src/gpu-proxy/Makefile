# DeCloud GPU Proxy — build daemon + shims
#
# Prerequisites:
#   Daemon: CUDA toolkit (nvcc, libcudart, libcuda)
#   Shims:  Just a C compiler (no CUDA needed)
#
# Usage:
#   make              — build daemon + runtime API shim
#   make daemon       — build daemon only (requires CUDA)
#   make shim         — build runtime API shim (LD_PRELOAD, dynamic)
#   make driver-shim  — build CUDA Driver API shim (libcuda.so.1)
#   make nvml-shim    — build NVML shim (libnvidia-ml.so.1)
#   make all-shims    — build all three shims (dynamic, host glibc)
#   make shim-compat  — build runtime API shim (Docker Ubuntu 20.04)
#   make all-shims-compat — build all shims (Docker Ubuntu 20.04, universal)
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
DRIVER_SHIM := $(BUILD_DIR)/libcuda.so.1
NVML_SHIM   := $(BUILD_DIR)/libnvidia-ml.so.1

# Shared transport files (compiled into each shim)
TRANSPORT_DEPS := shim/transport.h shim/transport.c proto/gpu_proxy_proto.h

.PHONY: all daemon shim driver-shim nvml-shim all-shims \
        shim-compat all-shims-compat test clean install install-shim-compat

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
# Runtime API Shim: LD_PRELOAD for CUDA Runtime API (cudaGetDeviceCount, etc.)
# ----------------------------------------------------------------
shim: $(SHIM)

$(SHIM): shim/cuda_shim.c proto/gpu_proxy_proto.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC \
		-o $@ $< \
		-lpthread

# ----------------------------------------------------------------
# Driver API Shim: libcuda.so.1 for dlopen-based GPU discovery (Ollama)
# ----------------------------------------------------------------
driver-shim: $(DRIVER_SHIM)

$(DRIVER_SHIM): shim/cuda_driver_shim.c $(TRANSPORT_DEPS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC \
		-o $@ $< \
		-lpthread
	ln -sf libcuda.so.1 $(BUILD_DIR)/libcuda.so

# ----------------------------------------------------------------
# NVML Shim: libnvidia-ml.so.1 for VRAM monitoring (Ollama)
# ----------------------------------------------------------------
nvml-shim: $(NVML_SHIM)

$(NVML_SHIM): shim/nvml_shim.c $(TRANSPORT_DEPS) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -shared -fPIC \
		-o $@ $< \
		-lpthread
	ln -sf libnvidia-ml.so.1 $(BUILD_DIR)/libnvidia-ml.so

# ----------------------------------------------------------------
# All shims (dynamic, host glibc)
# ----------------------------------------------------------------
all-shims: shim driver-shim nvml-shim

# ----------------------------------------------------------------
# Runtime API Shim (compat): built inside Ubuntu 20.04 container
# for universal glibc compatibility. Requires: docker
# ----------------------------------------------------------------
SHIM_COMPAT := $(BUILD_DIR)/libdecloud_cuda_shim-compat.so

shim-compat: | $(BUILD_DIR)
	docker run --rm -v $(CURDIR):/src -w /src ubuntu:20.04 \
	    sh -c "apt-get update -qq && \
	           apt-get install -y -qq gcc libc6-dev linux-libc-dev >/dev/null 2>&1 && \
	           gcc -Wall -Wextra -O2 -g -shared -fPIC \
	               -o build/libdecloud_cuda_shim-compat.so \
	               shim/cuda_shim.c -lpthread"
	@echo "Compat shim built: $(SHIM_COMPAT) (glibc 2.31+, universal)"

# ----------------------------------------------------------------
# All shims (compat): built inside Ubuntu 20.04 container for
# universal glibc compatibility across guest distros.
# ----------------------------------------------------------------
all-shims-compat: | $(BUILD_DIR)
	docker run --rm -v $(CURDIR):/src -w /src ubuntu:20.04 \
	    sh -c "apt-get update -qq && \
	           apt-get install -y -qq gcc libc6-dev linux-libc-dev >/dev/null 2>&1 && \
	           gcc -Wall -Wextra -O2 -shared -fPIC \
	               -o build/libdecloud_cuda_shim-compat.so \
	               shim/cuda_shim.c -lpthread && \
	           gcc -Wall -Wextra -O2 -shared -fPIC \
	               -o build/libcuda.so.1 \
	               shim/cuda_driver_shim.c -lpthread && \
	           ln -sf libcuda.so.1 build/libcuda.so && \
	           gcc -Wall -Wextra -O2 -shared -fPIC \
	               -o build/libnvidia-ml.so.1 \
	               shim/nvml_shim.c -lpthread && \
	           ln -sf libnvidia-ml.so.1 build/libnvidia-ml.so"
	@echo "All compat shims built (glibc 2.31+, universal):"
	@echo "  $(BUILD_DIR)/libdecloud_cuda_shim-compat.so  (Runtime API, LD_PRELOAD)"
	@echo "  $(BUILD_DIR)/libcuda.so.1                    (Driver API, dlopen target)"
	@echo "  $(BUILD_DIR)/libnvidia-ml.so.1               (NVML, dlopen target)"

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
# Install all shims (preferred for 9p share distribution)
# ----------------------------------------------------------------
install-shim-compat: shim-compat
	install -d $(DESTDIR)$(PREFIX)/lib
	install -d $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim
	install -m 644 $(SHIM_COMPAT) $(DESTDIR)$(PREFIX)/lib/libdecloud_cuda_shim.so
	install -m 644 $(SHIM_COMPAT) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libdecloud_cuda_shim.so
	@echo "Compat shim installed to $(PREFIX)/lib and 9p share directory"

install-all-shims-compat: all-shims-compat
	install -d $(DESTDIR)$(PREFIX)/lib
	install -d $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim
	# Runtime API shim
	install -m 644 $(SHIM_COMPAT) $(DESTDIR)$(PREFIX)/lib/libdecloud_cuda_shim.so
	install -m 644 $(SHIM_COMPAT) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libdecloud_cuda_shim.so
	# Driver API shim (libcuda.so.1)
	install -m 644 $(DRIVER_SHIM) $(DESTDIR)$(PREFIX)/lib/libcuda.so.1
	install -m 644 $(DRIVER_SHIM) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcuda.so.1
	ln -sf libcuda.so.1 $(DESTDIR)$(PREFIX)/lib/libcuda.so
	ln -sf libcuda.so.1 $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libcuda.so
	# NVML shim (libnvidia-ml.so.1)
	install -m 644 $(NVML_SHIM) $(DESTDIR)$(PREFIX)/lib/libnvidia-ml.so.1
	install -m 644 $(NVML_SHIM) $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libnvidia-ml.so.1
	ln -sf libnvidia-ml.so.1 $(DESTDIR)$(PREFIX)/lib/libnvidia-ml.so
	ln -sf libnvidia-ml.so.1 $(DESTDIR)$(PREFIX)/lib/decloud-gpu-shim/libnvidia-ml.so
	@echo "All compat shims installed to $(PREFIX)/lib and 9p share directory"
