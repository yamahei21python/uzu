# vLLM 0.32.0 + DFlash block-table fix
#
# vLLM 0.32.0 serves DFlash speculative decoding at ~0 acceptance because the
# draft replays its FULL CUDA graph with a stale kernel block table (#63587).
# No released version contains the fix (#64059); this image backports it.
#
# Build (from the repository root):
#   docker build -f docker/vllm-0320-dflash/Dockerfile -t vllm:0.32.0-dflash .
#
# Verify inside the image:
#   docker run --rm vllm:0.32.0-dflash python -c \
#     "from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator as S; print(hasattr(S, '_map_kernel_block_tables'))"

FROM vllm/vllm-openai:v0.32.0

COPY patches/vllm-0320-dflash-kernel-block-table.patch /tmp/dflash-fix.patch

# Locate the directory that holds the vllm package instead of hardcoding the
# python version; the base image moves it between releases.
RUN SITE="$(python -c 'import os, vllm; print(os.path.dirname(os.path.dirname(vllm.__file__)))')" \
    && cd "$SITE" \
    && patch -p1 --dry-run < /tmp/dflash-fix.patch \
    && patch -p1 < /tmp/dflash-fix.patch \
    && rm /tmp/dflash-fix.patch \
    && python -c "from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator as S; assert hasattr(S, '_map_kernel_block_tables'), 'patch missing'"
