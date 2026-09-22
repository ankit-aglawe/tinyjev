# E5 — footprint gate on the E1(b) checkpoint (Qwen3-0.6B LoRA 5e-5), base M1 / 16 GB (2026-09-22)

Scored with Kev's own `kev.benchmark --remote` against `tinyjev serve` on this machine, transfer-v4
development partition (764 records, 656 clean questions). H100 fp32 reference for the same checkpoint: 0.6250.

| build | acc | ECE | cov@5% | Brier | p50 | p95 | weights |
|---|---|---|---|---|---|---|---|
| MLX fp16 | 0.6204 | 0.140 | 0.076 | 0.505 | 65 ms | 106 ms | 1.2 GB |
| MLX INT8 (Linear layers, g64) | 0.6204 | 0.136 | 0.078 | 0.506 | 58 ms | 109 ms | ~0.6 GB |
| MLX 4-bit (Linear layers, g64) | 0.5991 | 0.097 | 0.064 | 0.491 | 59 ms | 110 ms | ~0.35 GB |

INT8 is free on accuracy and calibration. 4-bit costs 2.1 pp on held-out accuracy (fails the < 1 pp
gate) while its ECE/Brier improve, which is softened over-confidence, not better decisions. Quantization
does not change latency on an M1; it changes memory. Embeddings and the fp32 numpy head are never quantized.

Files: `e1b-mlx-{fp16,int8,int4}-m1.report.json` are Kev's benchmark reports.
