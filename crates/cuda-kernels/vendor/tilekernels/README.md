# Tile Kernels

Optimized GPU kernels for LLM operations, built with [TileLang](https://github.com/tile-ai/tilelang). TileLang is a domain-specific language for expressing high-performance GPU kernels in Python, featuring easy migration, agile development, and automatic optimization.

Most kernels in this project approach the limit of hardware performance regarding the compute intensity and memory bandwidth. Some of them have already been used in internal training and inference scenarios. However, they do not represent best practices and we are actively working on improving the code quality and documentation.

## Features

- **Gating** — Top-k expert selection and scoring for Mixture of Experts routing
- **MoE Routing** — Token-to-expert mapping, fused expansion/reduction and weight normalization
- **Quantization** — Per-token, per-block, and per-channel FP8/FP4/E5M6 casting with fused SwiGLU+quantization ops
- **Transpose** — Batched transpose operations
- **Engram** — Engram gating kernels with fused RMSNorm, forward/backward passes and weight gradient reduction
- **Manifold HyperConnection** — Hyper-connection kernels including Sinkhorn normalization and mix splitting/application
- **Modeling** — High-level `torch.autograd.Function` wrappers composing low-level kernels into trainable layers (engram gate, mHC pipeline)

## Requirements

- Python 3.10 or higher
- PyTorch 2.10 or higher
- TileLang 0.1.9 or higher
- NVIDIA SM90 or SM100 architecture GPU
- CUDA Toolkit 13.1 or higher

## Installation

### Install a local development version

```bash
pip install -e ".[dev]"
```

### Install a release version

```bash
pip install tile-kernels
```

## Testing

Tests using pytest:

### Test single test file

```bash
pytest tests/transpose/test_transpose.py -n 4 # Correctness only with 4 workers
pytest tests/transpose/test_transpose.py --run-benchmark # Correctness + Benchmarking
```

### Pressure test

```bash
TK_FULL_TEST=1 pytest -n 4 --count 2
```

## Project Structure

```txt
tile_kernels/
├── moe/        # Mixture of Experts routing related kernels
├── quant/      # FP8/FP4/E5M6 quantization
├── transpose/  # Batched transpose
├── engram/     # Engram gating kernels
├── mhc/        # Manifold HyperConnection kernels
├── modeling/   # High-level autograd modeling layers (engram, mHC)
├── torch/      # PyTorch reference implementations
└── testing/    # Test and benchmark utilities
```

## Acknowledgement

This project is built on [TileLang](https://github.com/tile-ai/tilelang). Thanks and respect to the developers!

## License

This code repository is released under [the MIT License](LICENSE).

## Citation

```bibtex
@misc{tilekernels,
      title={TileKernels},
      author={Xiangwen Wang, Chenhao Xu, Huanqi Cao, Rui Tian, Weilin Zhao, Kuai Yu and Chenggang Zhao},
      year={2026},
      publisher = {GitHub},
      howpublished = {\url{https://github.com/deepseek-ai/TileKernels}},
}
```
