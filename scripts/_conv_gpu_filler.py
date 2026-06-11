#!/usr/bin/env python3
# Controlled VRAM-constraint filler for the Qwen C4-reject experiment.
# Holds ~25 GiB on the visible GPU so the Qwen3.6 serve's POST-weights free VRAM
# falls below one slot's KV at max context (RoPE-capped 262144 tokens), driving
# Qwen35Model::kv_budget_num_slots to affordable=0 → the C4 fail-closed reject.
# Without this the reject branch is unreachable on a 96 GB H20 (free ≫ per_slot).
import sys, time, torch
mib = int(sys.argv[1]) if len(sys.argv) > 1 else 25000
numel = (mib * 1024 * 1024) // 2  # float16 = 2 bytes
buf = torch.zeros(numel, dtype=torch.float16, device="cuda:0")
torch.cuda.synchronize()
free, total = torch.cuda.mem_get_info()
print(f"filler holding {mib} MiB on cuda:0; free now {free>>20} MiB / total {total>>20} MiB", flush=True)
time.sleep(1800)
