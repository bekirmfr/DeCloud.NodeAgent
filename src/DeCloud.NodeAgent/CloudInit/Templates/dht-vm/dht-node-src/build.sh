#!/bin/bash
#
# Build script for DeCloud DHT Node binary
# Cross-compiles for amd64 and arm64, then base64-encodes the output.
#
# Usage:
#   ./build.sh              # Build for both architectures
#   ./build.sh amd64        # Build for amd64 only
#   ./build.sh arm64        # Build for arm64 only
#
# Output:
#   ../dht-node-amd64.b64   (placed next to templates, not in source dir)
#   ../dht-node-arm64.b64
#
# Requirements:
#   - Go 1.23+ installed
#   - Network access for initial dependency download (go mod tidy)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$(dirname "$SCRIPT_DIR")"
ARCHITECTURES="${1:-amd64 arm64}"

cd "$SCRIPT_DIR"

echo "=== DeCloud DHT Node Build ==="
echo "Source: $SCRIPT_DIR"
echo "Output: $OUTPUT_DIR"
echo ""

# Ensure dependencies are resolved
if [ ! -f go.sum ]; then
    echo "Resolving Go dependencies..."
    go mod tidy
fi

for ARCH in $ARCHITECTURES; do
    BINARY_NAME="dht-node-${ARCH}"
    B64_NAME="${BINARY_NAME}.b64"
    BINARY_PATH="${OUTPUT_DIR}/${BINARY_NAME}"
    B64_PATH="${OUTPUT_DIR}/${B64_NAME}"

    echo "Building for linux/${ARCH}..."

    CGO_ENABLED=0 GOOS=linux GOARCH="${ARCH}" \
        go build -trimpath -ldflags="-s -w" -o "${BINARY_PATH}" .

    BINARY_SIZE=$(stat -c%s "${BINARY_PATH}" 2>/dev/null || stat -f%z "${BINARY_PATH}" 2>/dev/null)
    echo "  Binary: ${BINARY_PATH} ($(( BINARY_SIZE / 1024 / 1024 ))MB)"

    # Base64 encode
    base64 -w0 "${BINARY_PATH}" > "${B64_PATH}"
    B64_SIZE=$(stat -c%s "${B64_PATH}" 2>/dev/null || stat -f%z "${B64_PATH}" 2>/dev/null)
    echo "  Base64: ${B64_PATH} ($(( B64_SIZE / 1024 / 1024 ))MB)"

    # Clean up raw binary (only the .b64 is needed at runtime)
    rm -f "${BINARY_PATH}"

    echo "  Done."
    echo ""
done

echo "=== Build complete ==="
echo "Place the .b64 files alongside the cloud-init templates:"
ls -lh "${OUTPUT_DIR}"/*.b64 2>/dev/null || echo "(no .b64 files found — build may have failed)"
