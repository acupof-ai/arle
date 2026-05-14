# DSv4 Decode nsys Trace Artifacts

Date: 2026-05-14

This directory keeps the trace evidence for
[`docs/experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md`](../../experience/errors/2026-05-14-dsv4-decode-nccl-bottleneck.md).
The artifacts were pulled from the remote H20 host after the run; the record no
longer depends on `/tmp`.

## Run

Model: `/root/DeepSeek-V4-Flash`

Serving shape:

```bash
./target/release/infer \
  --model-path /root/DeepSeek-V4-Flash \
  --port 18084 \
  --max-seq-len 900000 \
  --kv-cache-dtype fp8 \
  --num-slots 1 \
  --disable-cuda-graph \
  --deepseek-distributed-layers 43 \
  --mem-fraction-static 0.1
```

Important environment:

```bash
INFER_CUDA_DEVICES=0,1,2,3,4,5,6,7
ARLE_DSV4_INCREMENTAL_KV=1
ARLE_DSV4_LOG_TOPK=0
ARLE_CUDA_DISABLE_MARLIN_W4_FP8=1
LD_LIBRARY_PATH=/tmp/arle-nccl-227-lib:/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH:-}
```

Profiler capture:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output=/tmp/arle-dsv4-decode-nsys \
  ./target/release/infer ...
```

The service used CUDA profiler API signal hooks: `SIGUSR1` started capture and
`SIGUSR2` stopped capture after the 32-token streaming decode request.

## Files

| File | Purpose |
|---|---|
| `arle-dsv4-decode-nsys.nsys-rep.gz` | Compressed raw Nsight Systems report, tracked through Git LFS. |
| `arle-dsv4-decode-nsys.sqlite.gz` | Compressed exported Nsight Systems SQLite database, tracked through Git LFS. |
| `arle-dsv4-decode-nsys-stats.txt` | Human-readable `nsys stats` output. |
| `arle-dsv4-decode-nsys-client.json` | Client-side result for the profiled 32-token streaming request. |
| `arle-dsv4-decode-nsys.log` | Service log captured during the nsys run. |
| `arle-dsv4-default-after.json` | Non-nsys 32/64-token decode benchmark after the TP/EP instrumentation changes. |
| `arle-http-dsv4-run.log` | Current default TP=8/EP=8 server log containing request traces and benchmark output. |

## Rehydrate

```bash
gzip -dk arle-dsv4-decode-nsys.nsys-rep.gz
gzip -dk arle-dsv4-decode-nsys.sqlite.gz
nsys stats arle-dsv4-decode-nsys.nsys-rep
sqlite3 arle-dsv4-decode-nsys.sqlite
```

## SHA256

```text
2febdffa636e1b73b6cfda52fe082f76253c46b6a8b87b547994858b6b84b71e  arle-dsv4-decode-nsys-client.json
6a979f093087d431d492badf8d0f2d9a26eefad1e3a8f49e24303f9a7d1f901d  arle-dsv4-decode-nsys-stats.txt
f698828ab7d25a9a4b9b6ecddda369ff9c5b3d125e65909d09732e14a951d657  arle-dsv4-decode-nsys.log
1bd7ac7247a20447cf05d29920b2b811219f9d9059d5b24d4f04f89f6cf1db31  arle-dsv4-decode-nsys.nsys-rep.gz
b627841dab50abf691db8c76cdd75c5be5b76404a5c91cc15e43cd4f52b390ee  arle-dsv4-decode-nsys.sqlite.gz
fc58110bf22874625a962af5b89cb557da57740c53742bf9179fec2bbf64b1cc  arle-dsv4-default-after.json
aa69786857eec380b5c3106801bcf9dad47d77f9f69b53c5c1fceff7b925e487  arle-http-dsv4-run.log
```
