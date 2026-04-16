#!/usr/bin/env bash

PROTO_DIR="./proto"
OUT_DIR="./proto_gen"

mkdir -p "$OUT_DIR"

set -ex
protoc \
  --proto_path="$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  $(find "$PROTO_DIR" -name "*.proto")

echo "✅ 所有 .proto 文件已编译到 ${OUT_DIR}/"