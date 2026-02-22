# DeCloud GPU Proxy — build daemon + shim + guest agent
#
# Prerequisites:
#   Daemon: CUDA toolkit (nvcc, libcudart, libcuda)
#   Shim:   Just a C compiler (no CUDA needed)
#   Agent:  Just a C compiler (no CUDA needed; uses nvidia-smi at runtime)
#
# Usage:
#   make              — build daemon + shim
#   make daemon       — build daemon only (requires CUDA)
#   make shim         — build shim only (no CUDA needed)
#   make agent        — build guest agent only (no CUDA needed)
#   make test         — build and run protocol tests (no CUDA needed)
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
AGENT     := $(BUILD_DIR)/decloud-gpu-agent

.PHONY: all daemon shim agent test clean install

all: daemon shim agent

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
# Guest Agent: no CUDA dependency, polls nvidia-smi
# ----------------------------------------------------------------
agent: $(AGENT)

$(AGENT): agent/gpu_guest_agent.c proto/gpu_proxy_proto.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) \
		-o $@ $< \
		-lpthread

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
	install -m 755 $(AGENT)  $(DESTDIR)$(PREFIX)/bin/decloud-gpu-agent
	install -m 644 $(SHIM)   $(DESTDIR)$(PREFIX)/lib/libdecloud_cuda_shim.so
