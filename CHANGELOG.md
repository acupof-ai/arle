# Changelog

Progress record — one line per event (phase exit · default flip · verdict),
detail in the linked wins/errors entry. Oldest sections are condensed.

- [docs/stability-policy.md](docs/stability-policy.md)
- [docs/support-matrix.md](docs/support-matrix.md)

## [Unreleased]

- **The standalone correct-inference gate could not fail (#400).** `scripts/needle_gate.py` printed its per-length SUMMARY lines and exited 0 unless `--check` was passed, and a request that never reached the model — dead serve, refused connection, malformed transcript — was appended as a null result and counted as a model miss. The bare form `needle_gate.py <lengths> 3 0.0` is how the agent contract, CONTRIBUTING, the Justfile and roughly thirty wins entries invoke it, so the invocation that licensed correctness was a report, not a gate. The exit code now carries the verdict: 0 on pass, 1 when any length falls under the exact-hit threshold, and 2 for a request error in every mode including the new `--report` opt-out, so a run that could not fetch is never read as a miss. `--check` becomes an accepted alias for the default and `lever_gate.sh` takes `--report` on its baseline-envelope path, where `validate_summary` is the independent verdict; lever's own gating was not the hole and is unchanged, and the temp arm exits through its own path and is unaffected. What the gate checks was not tightened. A new shell test drives the real script against an in-process server and demonstrates each exit code on the input that produces it, including both `--report` arms; it is registered in the pre-push test list. Bench-entry exemption: gate harness scripts only. Live confirmation that a healthy model still passes default mode on a real serve is pending a free card.

- **The bench spec claimed a validation the runner never performed (#401).** §5 stated that ITL is valid only when each SSE event carries one completion token and that the runner rejects event/token count mismatches. `bench_throughput.py` records one timestamp per non-empty chunk and increments `output_events` at :236, and compares that count to nothing anywhere in the file; the rejection did not exist. The claim was also inverted by later work, since MTP and speculative decode legitimately emit several accepted tokens in one event — the code says so at :249 — so building the described gate would have rejected valid runs. The doc now states what is actually measured: ITL is a per-event interval, it equals a per-token interval only on a one-token-per-event configuration, and the decode `tok/s` figure derived from it is an events-per-second number otherwise. The runner captures no tokens-per-event count, only the aggregate `completion_tokens` from final usage, so per-token ITL under those configurations is not merely unvalidated but uncomputable from what is recorded; that is now named as a measurement gap rather than a gate. Two stale §3.4 defaults were corrected against the argparse help text. Bench-entry exemption: documentation only, no runner or gate code changed.

- **A pre-push selftest arm fired only 98.5% of the time (#399).** The rule-8 branch-only world built a real two-commit repository and cited the lane commit's sha truncated to nine characters. That prefix derives from the commit timestamp, and the sha matcher deliberately rejects an all-decimal string so an ordinary number is not read as a sha, so roughly one run in sixty-seven produced a prefix the matcher skipped, the arm found nothing to flag, and the selftest reported a failure indistinguishable from a real regression. It blocked a push on main once. Both selftest world builders now pin the commit dates, making the cited sha `b3a8a500b` on every run; the other thirty-five arms use only literal strings and were left alone. The matcher was not widened to accept all-decimal shas, since that would re-break the far more common case of a plain number in a document; that miss is now recorded as a known limit in the rule's docstring. Bench-entry exemption: pre-PR checker, dev-only tooling.

- **Five Prometheus counters that no backend can populate are no longer exported (#398).** The page-tier copy and wait counters — `demote_mset_copy_bytes`/`_ms`, `promote_mget_copy_bytes`/`_ms` and `fetch_wait_ms` — are written only inside a branch requiring a page tier that both carries capacity and copies. The tree holds three `KvPageTier` implementations and none satisfies both: CUDA returns a literal 0 capacity, and Metal, the only capacity-bearing tier, returns a literal true for zero-copy. Exporting them as `0` stated that the operation ran zero times, when the instrument does not exist in any supported configuration; the series are now absent from the metrics endpoint and the values stay in `/v1/stats` and JSONL, where 0 is the true reading of a counter that cannot run. The struct fields are kept and annotated at their definition with why each is zero and which backend would set it, because a future copying page tier would populate them. The gate asserts absence rather than a zero value, and checks that two live sibling counters in the same block still render, so a renderer that emitted nothing would fail it. No dashboard or document in the repository referenced the five series. Entry: [docs/experience/wins/2026-09-12-omit-impossible-page-tier-copy-metrics.md](docs/experience/wins/2026-09-12-omit-impossible-page-tier-copy-metrics.md).

- **Two KV reuse counters were summing different quantities and shipping the total (#397).** `Engine::kv_system_metrics` added the engine's per-block reuse DECISION, taken once per prefix block in the classify loop, to the backend tier store's READ count, so a block that was both decided reusable and read from the store was reported about twice. The two sets are not the same set — a block whose tier location is unknown counts as a host decision with no store read, and store reads also happen outside prefix reuse — so this conflated two metrics rather than cleanly doubling one, and dividing by two would have been the wrong fix. Both counters now come from the tier store alone, assigned rather than folded, and the engine-side increments are removed because the assignment made them write-only; the resident-page count in the same loop is unchanged. The blast radius lands exactly on the broken arm: CUDA pins page-tier capacity to zero so the decision path never ran there and its numbers do not move, while Metal `--kv-disk` steps down toward half — the old value was wrong. Two unit gates cover both shapes, and the capacity-bearing one was confirmed to fail against the restored original code at 2 against 1 and 8 against 4, so the test is known to go red on the defect it guards. Entry: [docs/experience/wins/2026-09-12-kv-tier-reuse-hit-single-source.md](docs/experience/wins/2026-09-12-kv-tier-reuse-hit-single-source.md); the live Metal confirmation is pending-remote.

- **The FlashQLA/DSpark verify batch went green without proving the treatment routed chunked (#334).** The one-command A/B across attn_tp 1-8 is the gate for the routing change and for the varlen deletion behind it, and in its DEFAULT mode — the mode that actually verifies the change — the GDR routing row was emitted as INFO while only FAIL rows reach the exit code. A treatment arm that never took the chunked path, from a wrong binary, a dropped cubin, or a geometry the router rejects, printed one line and the batch exited 0; the routing check was a hard gate only in the opposite direction, under the flag-off arm. The default run now requires the treatment arm to observe chunked routing or go red, the base arm stays informational because it may legitimately run varlen before the change, and a mock seam plus a treatment-routed-zero case prove the red path. The proof method is the point: the new test was run against the OLD script and observed to record the routing row as INFO, so the gap is demonstrated rather than asserted. No check was weakened. The script is scripts-only, has never executed on a card, and its first GPU run doubles as its live validation.

- **A doc citing a commit that will not survive its own merge now fails before the PR opens (#396).** Short shas recorded in wins and errors entries die structurally: the lane is rebased or squash-merged and the sha becomes unreachable, which is why 87 backticked hashes in the docs resolve to nothing today. Rule 8 of the pre-PR checker scans only ADDED lines in `docs/**.md` and fails a backticked 7-12 hex token that is not reachable from the origin/main merge-base. Testing reachability rather than mere resolvability catches the second and subtler shape — a sha that is a real commit right now but exists only on the feature branch, so the citation is already stale at the moment it is written. The rule is deliberately narrow: pre-existing lines are structurally unreachable by it, so the historical dead shas are left alone rather than swept; 16-character data hashes, decimals and full shas are excluded; and an upstream-repo sha passes when the project is named on the line. What it does not catch is listed in the PR body, on the principle that a check people learn to ignore is worse than no check.

- **Two support-matrix claims cited commits that do not exist (#395).** A sweep of `docs/` found 87 backticked hash-like strings that resolve to no object in this repository, and the cause is structural rather than careless: a short sha captured in an entry goes unreachable the moment its lane is rebased or squash-merged, and the same change can end up in history under several shas. Only the two citations that were actively misleading a reader are corrected — the OPD pivot's retirement commit and the Vulkan row's HF-to-GGUF fixes — and in both cases the sha is dropped rather than replaced, because the sentence carries its meaning without it. The other 85 are deliberately left alone: they were correct when written, and rewriting historical entries to chase shas that no longer resolve risks deleting real provenance for no gain. A follow-up adds a hygiene check that fires only on a newly added documentation line citing an unresolvable sha.

- **`lane.sh new` no longer strands a half-created worktree, and it is now tested (#387).** A lane add that left a directory with edited files but no `.git` link, no `worktree list` entry and no branch made every retry impossible: the `-e` precheck blocked the name while `git worktree remove` could not see the directory. Both failure shapes now run the same rollback — remove the directory the add created, prune, and delete the branch only when it did not pre-exist. The success check is the part that was subtly wrong: `rev-parse --is-inside-work-tree` resolves a stranded directory to the PARENT repository and so reports success, and the fix compares the directory's own path to its `--show-toplevel` instead. The rollback is safe because the pre-existing `[ -e "$path" ]` precheck proves the path absent immediately before the add, not because of the name validator. `scripts/tests/test_lane_new_rollback.sh` covers five cases including the one where the directory vanishes entirely and the `cd` fails, and is registered in the pre-push list. The trigger itself stays recorded as cause unknown. Entry: [docs/experience/errors/2026-09-12-lane-new-leaves-unregistered-directory.md](docs/experience/errors/2026-09-12-lane-new-leaves-unregistered-directory.md).

- **A git push that says "correct access rights" may have nothing to do with keys (#394).** Two lane pushes failed with git's cannot-read-from-remote text, which is its generic wording for any failure of the SSH command rather than a verdict on credentials. `ssh -T git@github.com` reported `command not found: _kaku_wrapped_ssh` while `/usr/bin/ssh -T` authenticated normally: the bare `ssh` resolved to a shell function captured without the implementation it calls. The rule recorded is to run `type ssh` before suspecting a key, and to confirm credentials against the binary explicitly, because a wrapper shadowing `/usr/bin/ssh` fails identically to an auth rejection at git's error-reporting layer. Entry: [docs/experience/errors/2026-09-12-ssh-shell-shim-fails-as-access-rights.md](docs/experience/errors/2026-09-12-ssh-shell-shim-fails-as-access-rights.md).

- **The architecture docs match the split tree again (#393).** `architecture.md` and `codebase-map.md` still described the pre-split shape: two crates that now exist, `infer-kvspace` and `infer-model`, were missing from the package tables entirely; `Engine` and `ServeHandle` were documented as generic over the backend and KV pool when both now hold `Box<dyn>`; `ForwardMode` was listed with six variants against the code's four; `RawLogits` was placed in `infer-api` although it lives in the train crate's extension; the CUDA `BackendExecutor` impl, the DeepEP sidecar path and `TpConfig` were all cited at locations they had moved from; and `infer-api` was credited with a metal feature and backend edges it does not have. Two commit hashes cited in codebase-map do not resolve to any object in the repository and are deleted rather than replaced, since the sentences stand without them. The migration notes marked as old homes are left alone — they are history, not drift.

- **Every pending-remote GPU run is queued with its command and decision rule (#392).** Deferred GPU work had accumulated faster than cards freed, with each item's command, device requirement and decision rule sitting in its own wins entry or agenda row; whoever got the next free card would have spent the first hour rediscovering what to run. Thirteen entries now carry the stable facts — exact command, the device and why, what the run decides, the prereg row it opens, and what it blocks or unblocks — derived from the agenda, the registry, the verify scripts and the per-item entries rather than from recollection. The ordering encodes real dependencies rather than preference: the one-card registry parity batch runs first because its per-gate metric logs are the measured clean-run margins the Phase-2 tolerance tightenings need, #334 must merge before the FlashQLA/DSpark verify grid because that grid's script is not on main, and that grid in turn gates the #300 varlen deletion. The fa2 sm70 gate runs on the separate V100 host concurrently with all H20 work. Reconstructing the list from the tree surfaced five items no one had remembered, including the second never-executed registry gate. The document deliberately records commands and rules, not which card is free or which run is in progress, since that state rots within the hour; the one dated section is labelled as such.

- **The DSv4 TP=8 parity gate ran green without comparing anything; it now runs from the batch and can fail (#391).** `dsv4_parity` was recorded as never executing because `parity_gpu_batch.sh` skips any gate tagged `:model`, and the wiring to run it had simply never been done. Reading the launcher for that wiring turned up the worse defect: `scripts/dsv4_multigpu_parity.sh` spawned its ranks, echoed rank 0's tokens next to the oracle, and exited 0 unconditionally — it never compared them, so a green run from it proved nothing at all. It now parses rank 0's first clean token, compares it to the oracle documented at `crates/infer-cuda/examples/dsv4_parity.rs:81`, and prints ALL PASS or a FAIL naming both values. Ranks also bind to a physical GPU list rather than hard-wired indices 0..7, so the batch can run on whatever set is free. `parity_gpu_batch.sh` gains a multi-rank phase: it releases its single-card claim, reserves the pinned set or the first free contiguous N, launches, frees, and re-claims one card for the remaining gates; without the checkpoint path or enough free SM90 cards it stays SKIP with the concrete reason rather than passing silently. The shell harness now drives the real launcher through both a correct oracle and a wrong first token, so the FAIL path is proven rather than assumed. The gate still checks only the first prefill token at TP=N, which the registry gate_gap states. The fa2 sm70 gate remains a documented hardware skip. The 8-GPU run is pending-remote.

- **Eight user-facing support claims corrected against the code (#390).** An audit of the four reference docs against the tree found the support matrix asserting things the code contradicts, and the expensive ones were the ones a reader would act on. GGUF was listed as production on CUDA and Metal; neither loader depends on `infer-gguf` and both require safetensors, so it ships on the experimental Vulkan and HIP backends instead, with `Q3_K` having no host launcher. `INFER_METAL_DFLASH_MAX_ROWS` was documented as defaulting to 4 and the code defaults to 16. W4A16 was credited with a Marlin prefill arm; `marlin_w4a16_gemm_cuda` is an FFI declaration with no caller and the router has no Marlin or dequant arm at any M, while W8A16's Marlin is real. W2A16 was listed as experimental and is an enum variant with no format, loader arm or kernel, so it is now a separate not-implemented row. The xgrammar build line named a `--features real` that does not exist at the root and would fail; the real chain is the `grammar` feature. DSv4 FP8/FP4 was still marked pending kernels although it is dispatched on the serving path, which also contradicted the FP8-MoE production row in the same document. Vulkan was marked as having no serving path although a serve builder is registered and selectable. Vanilla Qwen3-MoE was described as running a numerically wrong CUDA forward; it now fails closed at engine load. In environment.md, a documented `cargo test --release --test e2e` command was deleted: there is no such test target and no Rust reader of the variable it set, so the section now describes its one real consumer. Docs-only, net -6 lines.

- **Every parity-gate tolerance now says where it came from (#389).** The positive-arm audit found that most bounds were round numbers with no arithmetic beside them, so a reader could not tell a derived bound from a guessed one. Fourteen gates now carry the derivation in place where it closes — e4m3 round-to-nearest rms at 3.1% per operand and 6.25% worst-bin, int8 at 0.45%, bf16 store at 2.3e-3, the two-operand product at 4.4%, the flashmla pack repack supremum at 16/272 and its f32-scale bind at 17/273, the dsa bf16-amax half-ULP at 3.9e-3 — and an explicit `bound not derived; clean-run-set at <geometry>; needs a supremum` where no closed supremum exists, rather than a plausible-looking number. No existing bound value was changed. One structural gap is closed: marlin_w8a16 previously only compared Marlin against the dequant fallback by ratio, so both lanes being jointly wrong passed; it gains an absolute `MAX_REL_L2 = 0.20` on both lanes, derived from the INT8 group-quant floor amax/127 over GROUP=128 at roughly twice the analytic ceiling, with the 3x negative teeth reading 0.67 for a 3.35x margin so they still trip. One audit claim is retracted in the gate comment: paged_quant_attn does not share deepgemm's quantization term, because its oracle reads the same decoded pool bytes as the kernel, so the gap is f32 tile accumulation, per-split merge order, fast __expf and one bf16 store. Three tightenings — fa3 TOL_FP8_ANCHOR 0.22, dsv4_tp DG_SLOPE 0.14, dspark_sampler 0.02 — are identified and deliberately deferred until a card is free to measure the clean-run margin.

- **The GPU sampler is back, in-process and node-wide (#388).** `bbd422973` disabled the old sampler because its background thread forked `nvidia-smi` every 2 s and stalled the H20 driver 2-5 ms per step under DSpark batched decode; the promise to replace it is now closed. `crates/infer-server/src/gpu_nvml.rs` dlopens `libnvidia-ml.so.1` and reads utilisation, memory, temperature and power for up to eight devices once per observe tick, with no new link dependency, no extra thread and no infer-cuda involvement; a missing library, a missing symbol or any partial read yields no sample rather than zeroed gauges, so an absent reading is distinguishable from a real zero. `TICK_INTERVAL` is the single constant behind both the loop sleep and the documented cadence. The always-None plumbing it replaces is deleted — `BackendStats.gpu`, the engine overwrite, and the dead first-wins `WireStats.gpu` relay merge, including a fourth literal in the multiproc relay that was still populating the field. Both the single-process handle and the engine-less multiproc coordinator feed the same snapshot the JSONL, `/metrics` and `/v1/stats` consumers already read, and the dashboard charts one line per device with an across-device average, collapsing to the previous single-line form on one GPU. Default-off behind `ARLE_OBSERVE_GPU=1`: polling 5x less often in-process is a favourable prior, not a measurement, so it stays opt-in until the matched A/B clears it. Entry: [docs/experience/wins/2026-09-12-inprocess-nvml-sampler.md](docs/experience/wins/2026-09-12-inprocess-nvml-sampler.md). The A/B is pending-remote.

- **DSpark's c=8 zero-acceptance has a named scheduling candidate and a one-command decider (#386).** The pooled confidence goodput budget in `dspark_verify_lens` is the one by-design path that can make the printed acceptance rate collapse as concurrency rises: its first-draft admission bar is 0.53·r/(211+0.53r), which is 0.002506 at one slot and 0.019699 at eight, a 7.86x rise. A first-row confidence in that interval is drafted at c=1 and refused at c=8; keeps=0 truncates the chain to its anchor, `drafted` never increments, and the bench prints 0/0 as 0.0000. Every other batch-sensitive site — the spec_max_batch gate, the BF16 KV class gate, the block-size clamp, the per-row seed conditions, the two-row plan minimum — is tabulated with its c=1 and c=8 values and none of them explains the symptom. The head's real first-row confidence distribution cannot be read from code and is stated as the unknown rather than assumed. Scheduling cannot explain drafts that are proposed and then rejected, so the reading is decided by counters, not argument: `scripts/dspark_c8_accept_branch.py` drives the same load at both concurrencies, reads the spec_decode chains/drafted/accepted deltas, and prints exactly one of NOT-SEEDING, BUDGET-ZERO-KEEPS or PROPOSED-AND-REJECTED. It refuses a c=8 verdict unless the c=1 positive control drafted, and its selftest includes two counter traps that must refuse a branch rather than name one. Entry: [docs/experience/errors/2026-09-12-dspark-c8-confidence-budget-admits-zero.md](docs/experience/errors/2026-09-12-dspark-c8-confidence-budget-admits-zero.md). The run is pending-remote; all eight cards are held.

- **The pod TileLang pin is read from the tree being built, and rule 3 now sees the shell override form (#384).** `scripts/pod-build-env.sh` resolved the pin from a hardcoded build-tree path and silently fell back to a literal `0.1.13` when the grep missed — harmless only while that fallback equalled the real pin — and never printed which interpreter it chose, so a wrong-version build stayed invisible until the AOT probe aborted with a 101 that read as a defect in the diff under review. It now reads the pin from the built tree's `requirements-build.txt`, fails loudly when that git-tracked file exists without a `tilelang==` line, keeps the literal fallback only when the file is absent — a half-finished sync, where a hard failure would turn a recoverable build dead — and prints the resolved interpreter and pin on every resolution. Incident: [docs/experience/errors/2026-09-12-handrolled-pod-check-resolves-wrong-tilelang.md](docs/experience/errors/2026-09-12-handrolled-pod-check-resolves-wrong-tilelang.md). Separately, rule 3 of `scripts/lane_pr_precheck.py` excluded `-` from its lookbehind, so the shell parameter-expansion default escaped the machine-local-path check entirely; that is how a literal pod path reached the incident entry's prose. The lookbehind is now strict and the shell default is re-allowed only inside executable `.sh` code, never in a `.sh` comment and never outside shell. Measured across the tree, that idiom has 39 sites in `.sh` and none outside it, so the carve-out is scoped rather than a migration. Selftest: 30 cases.

- **Real-CUDA clippy is now gated the same way the check is (#383).** Rule 5 of `scripts/lane_pr_precheck.py` demanded `CUDA_CHECK_EXIT=0`, which is `cargo check` — and check runs no clippy lints, so deprecated items and unused-mut stayed invisible to it while the Mac lint compiled the CUDA code out entirely. Four hotfixes came through that one gap, one per target class: library (#368), root crate (#370), examples (#375), tests (#378). Rule 7 fires on the same trigger and requires a `CLIPPY_EXIT=0` line from a pod `cargo clippy --workspace --all-targets --features cuda,nccl -- -D warnings` without no-cuda; the trigger is extracted into a single predicate so the two rules cannot drift apart. Selftest: 25 cases.

- **The last dead negative control is a real tooth (#385).** `flashmla_prefill_parity`'s index family copied the kernel's own output on the host, edited one slot, and asserted it differed from the clean oracle — true by construction, no kernel rerun, unable to fail since the day it was written. It now re-runs the CSA index builder on a corrupted selection and requires the indices the kernel produces to differ from the clean oracle and to equal the oracle rebuilt from that selection, so a builder that silently ignored the bad input fails too. The replacement key is chosen to be in range and causally visible at that token, which is what stops the builder from dropping it. With this, all three defects the negative-control audit found are closed.

- **The parity gates' negative controls audited; three defects fixed (#380, #381, #382).** All 21 registry gates were read for whether their negative control can actually fail: is the perturbation on the path the positive arm measures, are the families distinct, and is the control's bound the positive arm's bound. Seventeen were sound. Three were not. (a) `flashmla_prefill_parity` computed `neg_indices = bad_indices != ref_indices` after already asserting `indices == ref_indices`, so that tooth was true by construction and never reran the kernel; its two sibling gates feed the sabotage through the kernel and were correct. (b) The fa3 hd256 shim's B=8 FP8 tooth averaged one corrupted row over 4608 head-rows, clearing its 5e-3 cap by 4 percent, so more checked rows would have silently disabled it; the attention teeth now assert on the corrupted row's own violation fraction (#380). (c) `moe_routing`'s `m_indices` family had only transitive coverage although a separate kernel produces it, and now has its own sabotage; `totals` is derived from counts by construction, which is recorded as a derivation rather than left as an open question, and every sabotage now asserts that the other families stay green (#381). Two gates never execute on the H20 batch device (fa2 needs sm_70, `dsv4_parity` needs the multi-rank model launcher), and the batch summary now names them instead of letting a green run read as full coverage. Separately, the three FlashMLA pack-gate oracles are corrected: the bit-exact CSA compare targeted raw f32 where the upload rounds to bf16, the FP8 NoPE compare used per-lane relative error on data that crosses zero, and two decode negative controls corrupted a data byte rather than the E8M0 scale byte (#382). Method note: [docs/experience/wins/2026-09-12-negcontrol-audit-method.md](docs/experience/wins/2026-09-12-negcontrol-audit-method.md). GPU runs are pending-remote.

- **A parity gate that is not registered never runs, and that is now a pre-PR failure (#374, #377).** `scripts/parity_gpu_batch.sh` derives its gate list from the `correctness_gate` values in `operators/registry.toml` and reads `ALL PASS` from the positive arm and `NEGATIVE CONTROL OK` from the negative one. A gate missing from the registry was therefore never built and never run, and a gate printing a different marker was recorded as a failure while working correctly; both were silent. Rule 6 of `scripts/lane_pr_precheck.py` now fires when a changed `crates/infer-cuda/examples/*.rs` takes `--negative-control` and requires both the registry entry and the marker. Checked against main when it was added: 26 examples, 20 triggered, 0 failures. Found by hand on the DSpark whole-drafter-step batch-invariance gate (#374), which had both defects and now covers one target sequence run alone against the same sequence at slots 0 and 3 of an 8-live-slot batch, with pos-swap and kv-base-swap negative controls; its GPU run is pending-remote.

- **The no-cuda lint blind spot is closed across the whole target matrix (#368, #370, #375, #378).** Every CI lint runs `--features cuda,no-cuda,...`, which compiles out `cfg(not(feature = "no-cuda"))` code, so four separate deletions and visibility changes passed CI and broke the real-CUDA build: a deleted `TapeDtype::nvrtc_prelude` whose only caller sat behind that gate (#368), a Vulkan `set_submit_cap` whose caller was in the root crate (#370), nine pre-existing lints in the train CUDA examples that a clean library finally exposed (#375), and three more in autograd's CUDA-only tests (#378). The instances span library, root crate, example and test targets, which is the full matrix. `cargo check` does not run clippy lints, so the `CUDA_CHECK_EXIT` requirement did not catch them; a `CLIPPY_EXIT` requirement under the same trigger is the follow-up.

- **CUDA OPD control surface moved into train; the MarlinW4A8 weight format deleted (#376, #379).** The train-only CUDA methods and LoRA types left `infer-api` for a `crates/train/src/cuda_opd_ext.rs` extension trait, so the serving crate keeps only the served `/v1/raw_logits` path (16 files, +212/-254). Separately the `WeightFormat::MarlinW4A8` variant, its validation and Display arms, one dead legacy kernel name in the cuda-kernels build, and four producer scripts are deleted (13 files, +11/-1003); the surviving MoE `w4a8_grouped_gemm` is a different code path and is untouched.

- **DSv4 multi-GPU attention output projection gated (#335).** The per-rank head slice, the grouped FP8 low-rank projection and the row-sharded output projection have a parity example at DSv4-Flash geometry for TP 1, 2, 4 and 8, decode at 1 and 8 tokens and a 32-token prefill chunk, compared with an f64 reference of the unsharded path. It covers the DeepGEMM lane H20 takes by default and the scalar fallback lane separately, with negative controls for a shifted head slice, a wrong group mapping, a dropped rank and a corrupted scale. With this, every kernel family the audit flagged has a code-side gate; all CUDA gates await one `scripts/parity_gpu_batch.sh` run.

- **Vulkan device-to-host reads work on non-UMA devices (#338).** `DeviceBuffer::copy_to_host` mapped device memory directly, which fails on MoltenVK and discrete GPUs when the buffer is device-local; it now stages through a host-visible buffer in that case and keeps the direct map for host-visible memory, so the serving forward is unchanged. The loader upload test that failed on MoltenVK passes without modification.

- **V100 attention kernel gated; GPU batch reports skipped gates (#336, #337).** (a) The FA2 sm70 attention kernel, whose only serving caller is the MTP speculative head, has a parity example at Qwen3.5 geometry (GQA 8/2 and 24/4, head_dim 256) with B=1 and B=8, prefill lengths across tile boundaries and a 4096-key case; `scripts/parity_gpu_batch.sh` records a gate that prints `SKIP:` as skipped rather than passed or failed, so this gate reads SKIP on H20 and runs on V100 (#336). (b) The kernel coverage audit now lists each gated family with its gate file and PR; every CUDA gate still awaits its GPU run (#337).

- **Vulkan shaders audited and gated at served geometry (#333).** All 38 Vulkan kernels are listed with their caller and test; 22 run on the Qwen3.6-27B-Q8_0 forward. New tests decode the quantized weights from the GGUF spec and check Q4_K/Q5_K/Q6_K/Q8_0 matrix-vector products at K 5120 and 17408, the MoE expert-indexed product at 256 experts top-8, flash attention at head_dim 256 with 24 query / 4 KV heads up to 4096 keys, and partial rotary 64 of 256. Setting `ARLE_REQUIRE_VULKAN_DEVICE=1` makes a missing Vulkan device fail the tests instead of skipping them.

- **Metal custom kernels gated (#332).** The six hand-written Metal kernel families in mlx-sys (GDR step and its tape variant, tape replay, the two-pass verify attention, and the M=16 4-bit quantized matmul at group sizes 32/64/128) have parity tests against f64 references at Qwen3.6-35B-A3B-4bit geometry, with BF16 quantization scales as shipped, attention lengths from 16 to 4113 keys including an 8-bit KV case, and one negative control per family; the tests run on the Metal CI runner. Exposing the GDR kernel to the tests first rebuilt a scalar input in every layer and slowed c=1 decode 3.1%; after the fix the 4B A/B is -0.17% TPOT and the needle gate matches ([entry](docs/experience/wins/2026-09-11-metal-kernel-parity-op-boundary.md)). The Qwen3.6-35B run is deferred because the machine lacks free memory.

- **Qwen DSpark drafter attention gated (#331).** The drafter's ring attention (single-slot and batched forms) has a parity example at Qwen3.8-27B-DSpark geometry from its config.json: 40 q / 8 KV heads, head_dim 128, block 7, no sliding window, so the context ring holds the full request (32775 rows at the serving default of 32768 tokens). It runs B=1 and B=8 with unequal contexts from 256 to 4096 keys, requires the single-slot and batched forms to agree exactly, and adds a small wrapping ring as a kernel-contract check. This is the second candidate for the DSpark acceptance gap after the varlen GDR kernel (#322); the GPU run is pending-remote.

- **A100-tier INT8/FP8 KV decode kernel gated (#330).** `paged_attention_quantized_fa3_cuda`, the decode and short-verify kernel for `--kv-cache-dtype int8|fp8` on sm_80..sm_89 hosts, has a parity example at Qwen3.5-4B geometry (16 q heads, 4 KV heads, head_dim 256) over INT8 and FP8 pools, with scattered and rotated page tables, query lengths 1 to 8, and single-split and 8/16-split decode; each pool and split mode is corrupted separately under `--negative-control`. The audit's hd128 gap in the TileLang paged-attention table is closed as unreachable: every CUDA-served target model uses head_dim 256, and the head_dim 128 drafters run a different attention kernel. The GPU run is pending-remote.

- **Chunked FlashQLA at attn_tp=8; Marlin gates gain negative controls (#328, #329).** (a) The (h_k, h_v) = (2, 6) FlashQLA kernels are added and gated, so Qwen3.5/3.6 multi-row GDN advances take the chunked path at every TP size and the varlen recurrent kernel has no production caller; the TP8 needle/lever gate and the new kernel bundle are pending-remote ([entry](docs/experience/wins/2026-09-11-flashqla-tp8-chunked.md), #329). (b) The three Marlin quantized GEMM gates (W8A16, FP8, FP4) corrupt each compared family under `--negative-control`, so `scripts/parity_gpu_batch.sh` now requires a negative mode from every gate except the model-level `dsv4_parity` (#328).

- **Qwen3.5/3.6 at attn_tp=2 and 4 run multi-row GDN advances through chunked FlashQLA (#327).** The (8,24) and (4,12) head-shard kernels were already compiled but excluded by two geometry checks, so prompt prefill, batched spec verify and DSpark rollback replay fell back to the varlen recurrent kernel on those TP sizes. They now take the same chunked path as attn_tp=1; attn_tp=8 still uses varlen until its (2,6) kernels exist. This changes the default prefill path at TP2/TP4, so TTFT, the needle/lever correctness gate and the DSpark acceptance A/B are pending-remote ([entry](docs/experience/wins/2026-09-11-flashqla-tp2-chunked.md)).

- **Parity gates, fifth batch (#322, #323, #326).** (a) The varlen conv1d + GDR prefill path that batched spec verify and DSpark rollback replay take at attn_tp≥2 has a parity example against an f64 recurrence at the (16,48) and (8,24) head shards, and prints its difference from the chunked FlashQLA path at row lengths 5, 17 and 64 from the same initial state; the GPU run decides whether the varlen kernel explains the DSpark acceptance gap that #292 left open (#322). (b) The FA3 hd256 paged-attention shims have one at Qwen3.6-27B geometry over BF16, INT8 and FP8 pools, with non-identity page tables and per-token scales (#323). (c) The FlashMLA sparse prefill has one for CSA and HCA layers at chunk sizes 128, 2048 and 4096 (#326). GPU runs are pending-remote through `scripts/parity_gpu_batch.sh`.

- **Parity gates: HCA decode, per-family negative controls, one-command GPU batch (#321, #324, #325).** (a) The FlashMLA decode gate covers the HCA (compression ratio 128) layers, about half of DSv4-Flash's decode layers, and both FlashMLA gates run B=1 as well as B=8 (#324). (b) Every parity gate now corrupts each compared family separately under `--negative-control` and fails if any family's corruption goes undetected; before, one detected corruption was enough (#321). (c) `scripts/parity_gpu_batch.sh <dir>` builds every gate listed in `operators/registry.toml`, runs each positive and negative on a claimed GPU, and writes a results table; a gate without a negative mode fails unless allowlisted, and only the model-level `dsv4_parity` is (#325). GPU runs are pending-remote.

- **Kernel parity gates, fourth batch, and the pre-push rebuild guard removed (#317, #319, #320).** (a) DSv4 decode-side kernels have parity examples at production shape: the FP8 grouped SwiGLU / down decode GEMMs at hidden 4096 / intermediate 2048 with top-6 routing over expert ids across 0..255, every output row checked, and the TP Q repack bit-exact at TP 2/4/8 (#319). The FlashMLA sparse decode on the CSA layers has one at B=1 and B=8 with per-row positions, rotated page tables, a nonzero attention sink and a separate check of the FP8 KV pack (#320). GPU runs are pending-remote. (b) The pre-push hook's rebuild guards are deleted. Once the snapshot rsync stamps updated files with the current time and the lock covers rsync through cargo, a changed input always rebuilds, so the guards could only fail legitimate pushes; a test now reads the hook's rsync flags and fails if mtime preservation returns ([entry](docs/experience/errors/2026-09-11-prepush-fresh-guard-was-redundant.md), #317).

- **Kernel parity gates, third batch, and a pre-push snapshot race (#314, #316, #318).** (a) The DSv4 MoE routing family (route, count, scan, pack, scatter, combine) has a parity example covering learned-bias and hash routing at 256 experts / top-6, including a regime where scores rather than bias decide the selection (#314). The DeepGEMM grouped FP8 prefill GEMM has one at the DSv4-Flash down and gate/up shapes with realistic block scales, all 256 expert groups and every output column checked (#316). GPU runs are pending-remote. (b) A hook shell test ran the real pre-push hook with the lock disabled and the default snapshot path, so any push touching `scripts/` rsynced a fixture tree over the shared snapshot with `--delete` while a peer's hook was compiling from it; nested runs now use a private snapshot unless a test names its own ([entry](docs/experience/errors/2026-09-11-prepush-nested-fixture-wiped-shared-snapshot.md), #318).

- **Kernel parity gates, second batch (#311–#315).** (a) DSpark filter / draft-sample / chain-accept and rms_norm / silu_mul / split / embedding have host-written parity examples at production vocab (Qwen 151936, DSv4-Flash 129280), the production 151936×5120 embedding table, and DSv4-Flash hidden 4096 / MoE intermediate 2048 (#312). DSpark draft attention and the DSA indexer have one at DSv4-Flash geometry: 64 heads, head_dim 512 split 448 nope + 64 rope, index 64×128 top-512 (#313). GPU runs are pending-remote. (b) Every unsafe block in `vulkan-sys` carries a SAFETY comment; name reads use ash's bounded `*_as_c_str()` accessors, and CI runs clippy `--all-targets -D warnings` on the three Vulkan crates (#311). (c) The pre-push rebuild guard keyed on the pushed file range, so a push that changed only `examples/` failed with a false cross-contamination report; it now keys on the files rsync updated in lib-affecting paths ([entry](docs/experience/errors/2026-09-11-prepush-fresh-guard-examples-only-false-positive.md), #315).

- **Kernel parity gates and cleanup (#307–#310).** (a) GDR decode, conv1d decode and argmax have host-written parity examples at production geometry, each with a negative control that must fail; GPU runs are pending-remote (#308). Writing the argmax oracle found that the host sampler picked a NaN logit that the CUDA strict-`>` scan never selects; the host path now matches the CUDA contract (ties go to the lowest index, NaN is skipped, an all-NaN row returns 0) ([entry](docs/experience/errors/2026-09-11-host-argmax-nan-divergence.md)). (b) FP8 E4M3 paged-KV quantize has a round-trip test over discontinuous pages, bounded by the reference value's half-ULP times the scale (#310). (c) The remaining 17 plain `dead_code` allows outside infer-cuda were adjudicated by per-crate clippy: 3 items deleted, 6 cfg-gated, 5 kept with a reason, and one allow removed from a field that is read, net −28 lines (#307). (d) CI lints the transport-parity examples with `nccl` enabled, and nine inline `ncclGetUniqueId` unsafe blocks collapse into one safe `nccl::unique_id()` (#309).

- **Follow-up batch — dead code, kernel audit, gate references, CI coverage (#303–#306).** (a) 65 plain `dead_code` allows removed from infer-cuda and cuda-kernels and adjudicated by CUDA clippy: 3 unused functions deleted, 3 allows kept with a stated reason, net −125 lines (#303). (b) A kernel parity-coverage audit lists every production CUDA kernel, the gate that covers it, and whether the gate runs at production geometry; FlashMLA sparse decode, the FA3 paged shims, GDR/conv1d decode and argmax have no numeric gate, and attn_tp≥2 GDR runs at a geometry no gate covers. Kernels with zero callers across the whole tree were deleted: the FP8 KV pack wrapper and five unused TileLang chunk AOT rows, which changes the prebuilt kernel bundle hash (#306, [audit](docs/plans/2026-09-11-kernel-parity-coverage.md)). (c) `operators/registry.toml` gate references are executable: each entry names a gate file, `vendor-trusted: <e2e path>`, or `none`, and every concrete gate declares `gate_scope = "production"` or a `gate_gap`; hygiene fails on a missing path or an undeclared scope, with a selftest world for each (#305). (d) CI now compiles CUDA lib tests, cuda-kernels tests with the `cuda` feature, the train/autograd CUDA examples, `infer-hip` with its feature, a `cargo check` of `infer-vulkan` with its feature, and the infer-model / infer-gguf tests; the Lint job grew by 27–52 s (#304).

- **Pre-push false-Fresh root cause fixed (#302).** The hook built from `git archive | tar -x` and `rsync -a`, so a changed source file kept its commit timestamp; when the commit was older than the last build, cargo saw source mtime <= output mtime and reported a content-changed crate Fresh, in one lane or across lanes sharing the target. The snapshot now syncs with `rsync -rlpD --checksum` (changed files get mtime=now, identical files keep theirs), every cargo-bearing hook builds from one shared snapshot under the machine lock, and cargo-free pushes use a private snapshot with no lock. See [the entry](docs/experience/errors/2026-09-10-prepush-snapshot-mtime-from-commit-time.md).

- **Simplification batch — seam, executor, CI coverage, pre-push (#292–#301).** (a) Batched spec verify and DSpark batched rollback replay advance GDR per slot through the FlashQLA chunked recurrence that c=1 uses whenever it is available, instead of the varlen recurrent kernel that no parity gate covers; the kernel is the suspected cause of the 13% c=1 / 0% c=8 DSpark acceptance gap, numerics pending-remote (#292, [entry](docs/experience/wins/2026-09-10-batched-spec-verify-gdr-uses-chunked-fq.md)). With attn_tp≥2 the local head geometry has no FlashQLA instantiation, so multi-GPU deployments still take varlen; its deletion is prepared as draft #300, blocked on the H20 A/B. (b) `KvSlotAccounting` folded back into `KvAllocator` — it existed only to narrow `submit`'s write surface, which #289 removed — and `infer-api` drops its unused `infer-util` dependency (#293). (c) A duplicate `accept_commit` in `infer-model` (the production path is `infer_plan::spec_accept_greedy`), three stale `dead_code` allows on live qwen35 forward functions, and two lint warnings removed (#295). (d) CI now compiles and runs `infer-metal`'s own lib tests (#296): the `kv_ssd` test module had not compiled (#294), and on its first day the gate caught a #293×#296 merge-order break in four test imports (#299). Three offline pytest modules are wired into CI (#294). (e) The pre-push hook skips the shell-test batch when a push touches none of their inputs (`scripts/`, `.githooks/`, `.github/`, `.gitignore`, `crates/cuda-kernels/`), so a docs-only push no longer holds the SSH connection long enough for the remote to close it (#298). (f) Comments referencing the deleted `infer/` crate point at current paths (#297), and task-history labels are stripped from code comments (#301).

- **Gate and infrastructure batch — six host-side correctness gates merged (#274–#291).** (a) A reference that scores zero exact hits across every needle length now fails the lever gate instead of passing any treatment; a partial-only baseline (zero exact, nonzero partial) is the pinned red shape (#287). (b) `bench_ab.sh` defines `die()` before its first use — it runs under `set -uo pipefail` without `-e`, so the late-defined guard had printed `command not found` and run both A/B arms identically; the cross-arm diff now refuses a single-arm comparison in either direction (#288). (c) Per-backend feature gates scattered through the four leaf binary builders are replaced by one `BackendCapabilities { mtp_spec_decode, kv_ssd_tier }` declared at `register_backend`, checked once before executor construction; adding a load-time feature is one struct field + one check + one per-backend declaration, and refusal names the selected backend (#289). (d) The FA3 per-slot quantized-attention workspace is itemized in the capacity solve and the over-budget config is rejected before allocation, pinned by a host test that makes the workspace the marginal bytes crossing the 1-slot line (#290; mechanism in #281). (e) Concurrent pod lanes no longer cross-execute: build/run state is per-lane (`/root/arle-ops-<lane>`) and a run rejects a receipt whose `tree` is not its own before the sha/GPU checks, so an identical-sha binary from another tree is refused (#291). (f) The qwen35 executor dispersion continued through d8–d10: spec accept scan and page math (#274), the spec draft plan and accepted-token constructor (#277), and prefill snapshot cuts / row slicing (#280) now compute as pure values in `infer-plan`.


- **Batched MTP on Qwen3.8-27B-NVFP4 — verdict: accepted with bounds, profitable at c≤4 and rejected at c≥8.** The batched draft is correct and ships: chains engage at every concurrency (~4,200, against 0 at c≥4 before the fix), needle 12/12 exact DET, lever gate PASS. Decode ms/tok against the no-spec control on the same binary: c=1 −17.3%, c=4 −1.3%, c=8 +27.7%, c=16 +29.3%, c=32 +39.7%. The mechanism is break-even, not a kernel defect — verify sends 3N rows through the trunk and returns 1+d·a tokens per step, so it pays only while `1 + d·a > cost_ratio`; the measured cost ratio is 1.6 at c=1 and 2.51 at c=32 against 1+2·0.467 = 1.93, putting the c=32 break-even accept rate at 0.755 versus the measured 0.467. The lever is drafter quality. See [the entry](docs/experience/errors/2026-09-10-batched-mtp-acceptance-break-even.md).

- **Architecture refactor — phase exit through Step 2: the host/device line is drawn by dependency graph, and four crates sit above it.** The criterion replaces "no device calls": a crate is above the line iff `cuda-kernels` is absent from `cargo tree`, not gated out of it. Verified on `main`: `infer-plan` (18 deps), `infer-kvspace` (29), `infer-quant` (33), `infer-seam` (20) all return zero matches, with the same grep hitting `infer-cuda` once and `infer-api` three times as its positive control. Seven changes landed: weight-layout decisions moved to `infer-quant` with a CPU oracle (#255), the speculative-decode host core moved to `infer-plan` (#256), the pod tree gained an identity check and sync bootstrap (#257), `DeviceBatch` and runtime backend dispatch replaced the seam's `Box<dyn Any>` return (#258), `infer-kvspace` was created and took the DSv4 prefix codec (#259), four inline TP two-phase-commit copies collapsed into two named seam operations (#260), and prefix reuse-license policy moved behind `PrefixPoolIndex` (#262). Step 3a (dispersing the executor's six blocks) is next; `infer-protocol` and `infer-model` in the target structure do not exist yet. See [the refactor plan](docs/plans/2026-09-09-architecture-refactor.md), [kv batch descriptor](docs/experience/wins/2026-09-09-kv-batch-descriptor-above-seam.md), [spec host core](docs/experience/wins/2026-09-09-spec-decode-host-core-infer-plan.md), [prefix codec](docs/experience/wins/2026-09-10-infer-kvspace-dsv4-prefix-codec.md), [reuse license](docs/experience/wins/2026-09-10-prefix-reuse-license-infer-kvspace.md), [TP two-phase commit](docs/experience/wins/2026-09-10-tp-two-phase-commit-seam.md).

- **Hygiene checks now prove they can fail, and bench runs are pre-registered** — adopted from the aupai harness. `scripts/check_repo_hygiene.py --selftest` builds a world per git-backed check by copying the real artifact and breaking it, asserts FAIL there and PASS on the same world unbroken; six checks covered, 0.71 s, wired into CI / `make hygiene` / pre-push. `scripts/prereg.py` writes the hypothesis before a run and closes the row with result / finding / decision separated; a row left running past 24h fails hygiene.

- docs(design): design note 1, a prefix cache for hybrid models — snapshots at page boundaries bound to the attention pages, `prompt_len - 1` cap, content keys on disk / page identity in memory, and the 2026-09-02 identity failure ([note](docs/design/hybrid-prefix-cache.md), [plan](docs/plans/2026-09-02-design-theses.md)).
- docs(design): design note 2, speculative decoding as a correctness-preserving transform — verify owns correctness, the block drafter rides the prefix cache; the gain region ends at c≥4 on H20 and where drafter and target mismatch (DSpark acceptance 13% c=1 / 0% c=8 on Qwen3.8-27B), with tileRL's spec net-negative at every measurement point as the comparison case. New `scripts/spec_parity.py` diffs greedy vs draft token ids over N prompts with a negative control that proves the gate can go red ([note](docs/design/speculative-decoding.md), [plan](docs/plans/2026-09-02-design-theses.md)).
- docs(design): design note 3, the backend seam as a cost contract — two host-only traits (`BackendExecutor` = submit/poll + `StepLimits` + capability accessors defaulting to `None`; `KvPool`), capability traits with zero default bodies, loop-fork precedent for families whose decode is not submit/poll shaped; 49 → 15 methods; the CPU smoke path is the placeholder `MetalExecutor` over the shared `HostPagedKvPool`, 16/16 engine tests green with no accelerator ([note](docs/design/seam-cost-contract.md), [plan](docs/plans/2026-09-02-design-theses.md)).
- docs(design): design note 5, matched measurement and the gate's positive control — TTFT/ITL as separate SLOs, same-shell three-trial A/B inside the drift band, needle-ladder envelope over byte identity, loud gate skips, three flag-deletion waves (66 → 49 serve flags) ([note](docs/design/matched-measurement.md), [plan](docs/plans/2026-09-02-design-theses.md)).
- docs(design): design note 4, memory is the product — resident bytes and bytes-read-per-token as the two numbers every feature is judged by; one resident layout per weight (Marlin source freed inline, derived operands rebuilt per call in scratch), itemized prefill working set; the 2026-08-20 stored-twice failure and its same-day recurrence ([note](docs/design/memory-is-the-product.md), [plan](docs/plans/2026-09-02-design-theses.md)).
- **`arle --doctor` prints the Metal resource-guard solve — weights, runtime headroom, static state, anti-swap reserve, KV budget, planned slots — and the rejection path itemizes the fixed requirement; JSON gains a `resource` object, inspection schema v3 → 4.** See [wins 2026-09-09](docs/experience/wins/2026-09-09-doctor-prints-resource-solve.md).
- docs(design): `what-breaks.md` — ten errors-corpus entries selected for root causes that generalize beyond this codebase, one paragraph each (Symptom / Root cause / Rule); the failure page paired with design note 5 ([page](docs/design/what-breaks.md), [plan](docs/plans/2026-09-02-design-theses.md)).
- **Metal prefix restore survives past the first turn — verdict: accepted.** Turn 3+ of a multi-turn conversation licensed 0 prefix blocks and re-prefilled the whole prompt: a restored slot's republish minted new logical ids for the shared pages and pruned every boundary snapshot, and the snapshots it did leave were keyed to a page chain the radix never hands out. Qwen3.5-0.8B, 12 agent-shaped turns: turns 2–12 median TTFT 2.01 s → 180 ms (mlx-lm 0.31.2 on the same weights: 249 ms); restored output byte-identical to cold, needle 18/18 DET. See `docs/experience/wins/2026-09-02-metal-prefix-restore-survives-turns.md`.

- **DSv4 slot budget — verdict: accepted. Prefill-transient reserve: the solve subtracts one prefill chunk's itemized working set (1352MB at TP=4) before planning slots.** Exact previously-OOMing config (FP8 TP=4, 128K total tokens, no request cap): pre-fix 18 slots + all-rank OOM at tick 4432; post-fix 17 slots, c=8 16/16 complete, 0 OOM, 0 preempts; a 3.3GB-free shared-GPU rank now rejects at boot instead of OOMing mid-serve. Same day: the FP8 checkpoint's 86 capture alloc nodes rooted to the dense-BF16 `wo_a` cuBLAS lane and fixed (0 alloc nodes both checkpoints), and the precision-matrix entry corrected — FP8 experts win c=1 (59.5 vs 44.4 tok/s; the 26.5 was census contamination). See `docs/experience/wins/2026-08-24-dsv4-budget-prefill-reserve.md`, `2026-08-24-dsv4-precision-matrix.md`.

- **GSPO `math-opd` lane — end-to-end on `Qwen3.8-27B-NVFP4` (single H20).** Boxed-answer grader + length-shaped reward (`1 - α·(len-len_min)/(len_max-len_min+ε)`, α=0.3) on the agent-opd GSPO loop. Three blockers fixed at root: RoPE clamp for the CC KV pool, an FP8-Marlin LoRA promote arm composing `marlin_fp8_to_e4m3` + `dequantize_fp8_block_scaled_to_bf16` (no new CUDA), and the `--lora-target-set` default flipped `all-linear` → `attention-qv` (all-linear's permanent BF16 promotion does not fit a 27B quantized student on one GPU). Smoke: 2 rounds, 8/8 groups `capped==0`, `is_ratio` ≈1.001, `clip_frac` <1, eval 0.875 → 0.875, both syncs clean. See `docs/experience/wins/2026-08-24-gspo-math-opd-smoke.md`.

- **GSPO `math-opd` run 1 — verdict: rejected (zero length compression).** 12 rounds, α=0.3 relative-within-group penalty. Eval accuracy 0.48→0.48 (noise), eval median pinned at 8192=max_tokens. Root cause: wrong samples got reward=0 regardless of length → no gradient on unsolved tasks (where the model rambles). Fix: absolute penalty on every sample — correct → `max(0, 1-α·len/L0)`, wrong → `-β·len/L0` (defaults α=0.5, L0=4096, β=0.05). See `docs/experience/errors/2026-08-24-gspo-math-opd-zero-length-gradient.md`.

- **Repo cleaning machinery adopted from deepseek-harness** — frozen wins/errors archive with sha256 manifest gate (`scripts/archive_experience.py`), residue sweeper (`scripts/clean_repo.py`), archive skill, all 22 CI actions SHA-pinned, cargo-machete nightly (first sweep removed 4 dead deps, incl. `infer-core` from both backends), loud lever-gate skips. See `docs/experience/wins/2026-08-24-repo-cleaning-machinery.md`.

- `Qwen3.8-27B-NVFP4` passes every probe that `ThinkingCap-Qwen3.6-27B-NVFP4` fails (tool calls included), at 52.3 tok/s with a 33%-acceptance MTP head — so NVFP4 training is viable and the corruption is a property of that one checkpoint, not the FP4 path.
- Note that the DSpark drafter RoPE fix corrected a real dropped-config defect but did not move draft acceptance (13% before and after); the Qwen3.8 acceptance collapse is unexplained, see `docs/experience/errors/2026-08-23-dspark-rope-fix-was-not-the-cause.md`.
- Record the agent-OPD baseline for `Qwen3.8-27B-FP8` on the localized swe-smith corpus — 4/10 solved, 0.617 mean partial credit, every task control-validated; see `docs/experience/wins/2026-08-23-agent-opd-qwen38-baseline.md`.
- **DSv4 c=1 decode graph — default flip: armed by default; `ARLE_DSV4_DECODE_GRAPH=0` selects the eager arm.** Matched A/B on `c1-graph-v26`, TP=4 on H20, 32k agent prompts: c=1 decode 40.8 → 44.2 tok/s (+8.3%), ITL p50 24.1 → 22.2 ms, p99 47.7 → 44.1 ms; eager arm proven at 0 captures. c=8/16 unchanged (gate is c=1-only), DSpark unchanged, MMLU 171/200 in both arms with 0 per-item diffs, needle envelope identical through 32768. Capture audit 0 alloc / 0 free / 0 host nodes, positive-control verified. An earlier +21.3% reading is superseded: the eager baseline itself improved 14.6% via concurrent Marlin and Markov-head work. Twelve replay defects fixed on the way. See `docs/experience/wins/2026-08-23-dsv4-c1-decode-graph.md`.
- Record that NVFP4 serving corrupts tool-call generation while FP8 on the same request is clean, which is what took every agent-OPD rollout to `edited=false`; cause unknown, see `docs/experience/errors/2026-08-23-nvfp4-tool-calls-corrupt.md`.
- **Verification harness — Metal gate, CI needle gate, concurrent arm, bench_compare fix.** `needle_gate.py --check` standalone exit-0/1 gate; `lever_gate.sh GATE_PROFILE=metal`; temp + concurrent arms integrated; `bench_compare.py` rewritten for v1 snapshots with COLLAPSE detection; Metal CI runs the needle ladder on the 0.8B model with `ARLE_METAL_AVAILABLE_RESERVE_MB`/`ARLE_METAL_RUNTIME_HEADROOM_MB` + `--system-reserve-bytes`/`--memory-budget-bytes` overrides (7 GiB GitHub runners cannot fit the 6 GiB reserve + 4 GiB headroom defaults). See `docs/experience/wins/2026-08-23-verification-harness-metal-gate.md`.
- **Rust toolchain bumped 1.95.0 → 1.98.0** (next trait solver on nightly, blog 2026-08-21; stable gets the `chunks_exact_to_as_chunks` clippy lint). 20+ byte-conversion sites converted to `as_chunks`; pod/Dockerfile pins bumped in lockstep. Three lint lanes exit 0.
- First agent-OPD rollouts on an NVFP4 27B reach the model: 16/16 `edited=false` with the tool path proven working, so the gap is sustaining tool use under a long agent context rather than emitting it; see `docs/experience/wins/2026-08-23-agent-opd-nvfp4-baseline.md`.
- **TP batched decode — default flip: the rows>1 `is_single` gate is gone; TP decode batches go through the batched paged forward instead of B per-row forwards.** TP2 NVFP4-27B on H20: c≥2 ITL 1.55×–17.89×, aggregate 87 → 1,300 tok/s at c=32; c=1 wash; concurrent needle 54/54 exact. See `docs/experience/wins/2026-08-23-tp-batched-decode.md`.
- Emit Anthropic `thinking` content blocks only when the client enabled extended thinking. A chat template that reasons by default was sending them to every client, which aborts Claude Code's stream and took cc-harness agent rollouts to zero reward; see `docs/experience/errors/2026-08-23-anthropic-thinking-blocks-abort-claude-code.md`.
- **`--numa-pin` — characterized as single-rank-inert: the pin call is gated on `!cfg.is_single()` (`loader.rs:60`), so single-rank serves never pin; a same-binary A/B on Qwen3.6-35B-A3B-FP8 washed at c=1/16/32 (both arms unpinned).** Multi-rank evidence (2026-06-12, TP=8) already supports default ON; flag stays as the multi-rank opt-out. See `docs/experience/wins/2026-08-23-numa-pin-single-rank-inert.md`.
- **DeepGEMM min-routes mid-band — characterized, no flip: `--qwen35-deepgemm-min-routes 129` vs 1024 on Qwen3.6-35B-A3B-FP8, 1×H20. c=16 control identical (gate works); c=32 (R=256) wash (itl_p50 64.8 vs 66.6 ms); c=64 over KV capacity. The JIT/TMA small-band overhead persists into the FP8 contiguous mid-band; 1024 stays.** See `docs/experience/wins/2026-08-23-deepgemm-min-routes-mid-band.md`.
- **TP decode graph — default flip: the `model.tp.is_single()` gate is gone; whole-step decode graph now arms and replays under tensor parallelism.** NCCL/one-shot IPC all-reduces are stream-ordered and graph-capturable; eager fallback disarms on capture failure. TP2 NVFP4-27B on H20: c=1 ITL +9.9 %, c≥2 wash (GPU-bound ceiling), needle envelope identical to eager. See `docs/experience/wins/2026-08-22-tp-decode-graph-industry-path.md`.
- Close the LoRA merge loop for an NVFP4 base: `dequantize_fp4_marlin_to_bf16` plus FP8 slot setup and a `retired_marlin` keepalive, so `rubric-opd`/`agent-opd` can sync a trained LoRA back into an NVFP4 rollout engine. Verified on Qwen3.6-27B-NVFP4; see `docs/experience/wins/2026-08-22-nvfp4-lora-merge-loop.md`.
- **Flag deletion wave 3 follow-up — accepted: `examples/dsv4_resident_ab.rs` (601 lines) and the fused-WQKV override chain deleted (the axis was a proven +18.4 % winner with a preflight fallback); a dangling `set_dsv4_moe_contig_decode` doc comment removed.** See `docs/experience/wins/2026-08-22-flag-deletion-wave3.md`.
- **Flag deletion wave 3 — accepted: `--dspark-confidence-threshold`/`--mtp-adaptive`/`--mtp-min-accept`/`--dsv4-flashmla-decode` deleted with full chains (incl. the FlashMLA override API and the resident A/B example's scalar arm), `dsv4_decode_reuse_enabled()` shim removed; serve flags 53 → 49.** See `docs/experience/wins/2026-08-22-flag-deletion-wave3.md`.
- **Flag deletion wave 2 — accepted: `--qwen35-fa3`/`--qwen35-deepgemm`/`--qwen35-moe-decode-kernel` deleted (off-arms 2.76×/15.4×/10.8× worse), Metal warmup seam default fixed to match shipped false, 12 stale doc refs cleaned**. See `docs/experience/wins/2026-08-22-flag-deletion-wave2.md`.
- **FIX: decode graph was hardcoded OFF by the flag-deletion wave — serve mapping now hardcodes on (the off-arm costs −58.7 % TPOT); caught by the SOTA-defaults audit.** See `docs/experience/errors/2026-08-22-decode-graph-hardcoded-off-in-flag-deletion.md`.
- Gate the shared NVFP4 base against `marlin_fp4_gemm` rather than the group layout: the repack flushes lifted values under 2.0 to zero, so the two layouts only agree on inputs that avoid that step. Kernel verified correct under the new oracle; see `docs/experience/errors/2026-08-22-marlin-fp4-parity-wrong-oracle.md`.
- **`--dsv4-moe-transport` promoted from `ARLE_DSV4_MOE_TRANSPORT` env** — the `--deepep-*` flags were inert without it; flag wins, env stays as fallback. Closes the codex P2.
- **Flag deletion wave — accepted: 10 proven A/B flags deleted (qwen35-decode-graph/-batched-decode/-gpu-router/-fa3-decode-splits, max-num-batched-tokens, dsv4-decode-reuse, metal-pipeline/-paged-kv-read, pool-model, extra_args), `--comm-backend` default reverted Auto→Nccl (unlicensed flip in one-shot 51–53 vs NCCL 70–80), 5 env aliases merged, runtime-flags statics fixed**. See `docs/experience/wins/2026-08-22-flag-deletion-wave.md`.
- **Batched DSpark on quantized KV — closed as rejected: op_timing pins the c≥8 loss on mixed-step scheduler stalls (57× 1.7 s vs 19× 1.97 s), and chunked-prefill 512/1024 does not fix it (c=4 −29 %, c=8 +5 % noise, c=16 wash).** Gate stays off; DSpark ships per-row at c=1. See `docs/experience/errors/2026-08-22-batched-dspark-quant-kv-verify-loses.md`.
- **Batched DSpark on quantized KV — rejected again with the verify-shape MMA kernel: c=8 19.2 vs 21.8, c=16 11.2 vs 13.9 tok/s per-row; verify attention ruled out as the cause, gate restored.** See `docs/experience/errors/2026-08-22-batched-dspark-quant-kv-verify-loses.md`.
- Read a shared NVFP4 base in its Marlin layout (`marlin_fp4_to_bf16`), so an OPD student can borrow the serving engine's frozen NVFP4 weights instead of holding a second copy (9 GB/rank vs FP8). Bit-exact against the group layout on sm_90; see `docs/experience/wins/2026-08-22-marlin-nvfp4-base-share-parity.md`.
- **OPD recompute chunk derives from the rank sequence — accepted.** Was a 4096
  constant. `chunk * rank_seq <= 2^30`, capped 16,384; `--opd-seq-chunk`
  overrides. Global 262,144 step: cp=4 692.9 → 618.4 s, cp=2 1,362.8 → 1,163.4 s,
  peak flat. `--fp8-native-gemm` stays opt-in (loss moves 0.23 %, six to ten
  times the envelope); the frozen-weight dequant cache measured as no effect.
  ([entry](docs/experience/wins/2026-08-22-opd-recompute-chunk-from-rank-sequence.md))

- **FA3 fp8 prefill attention from 64K tokens — accepted.** The quantized-pool
  prefill shim runs FA3's e4m3 kernels for prefill chunks over ≥64K KV (one
  descale per row/kv-head); 220K TTFT 129.7 → 108.2 s (−17 %), 32K and decode
  unchanged, needle 13/13 DET.
  ([entry](docs/experience/wins/2026-08-22-fa3-fp8-prefill-attention-long-context.md))

- **Global sequence 262,144 trains on 2 GPUs — accepted.** Sequence-parallel
  linear-attention core with cross-rank state carry (the all-to-all form's
  transient was O(global seq)), layer param grads parked on host, and the CP
  core collapsed into one tape entry. cp=2: 131,072 loss 3.034898 peak 64.6 GB,
  262,144 loss 1.561557 peak 85.7 GB; cp=4 262,144 loss 1.560897 peak 65.1 GB.
  Parity rungs bit-identical (4,096 9.857565 / 16,384 11.229959).
  ([entry](docs/experience/wins/2026-08-22-sp-linear-attention-core-262144-cp4.md))

- **MTP on the 32 K chain — measured, no flip.** Same binary, c=1/4/8/16 decode
  tok/s: no-spec 73.0/42.1/25.9/14.8, MTP d=2 84.3/42.0/24.8/14.0, d=4
  83.0/42.1/26.3/15.1; TTFT identical. +15 % at c=1 only (already the `auto`
  default); inert from c=4 until the verify step is batched. Closes the
  quantized-KV unification plan.
  ([plan](docs/plans/2026-08-22-quantized-kv-attention-unification.md))

- **CUDA drops Qwen3 dense and KIVI per-channel K.** `model_type=qwen3` fails at
  load; every quantized KV pool is per-(token, head) K+V on the tensor-core
  decode kernel. −7,340 lines incl. `decode_attention_quantized.cu`, the dense
  executor, the HD128 TileLang rows and the `--no-cuda-graph` flag.
  ([entry](docs/experience/wins/2026-08-22-delete-qwen3-dense-cuda-and-kivi.md))

- **Quantized paged attention on tensor cores — accepted; the only quantized
  decode path.** Phase 1 of the quantized-KV unification plan. Qwen3.8-27B-
  NVFP4, fp8 KV, 32 K prompts: per-request decode tok/s c=16 9.9–10.2 → 14.3
  (+40 %), c=32 5.9–6.0 → 8.9–9.0 (+50 %), c=1 wash; kernel 2.7–3.9×; needle 12/12 DET, eval
  177/200. Deletes the scalar kernel and the varlen fallback (−510 lines).
  ([entry](docs/experience/wins/2026-08-22-paged-attention-quantized-tensor-core.md))

- **Quantized paged attention dequantizes K/V once per GQA group — accepted.**
  One CTA serves the q-heads sharing a kv-head (group size scales with batch).
  Qwen3.8-27B-NVFP4, fp8 KV, 32 K prompts: per-request decode tok/s c=16 7.9 →
  10.0–10.2 (+27–29 %), c=32 4.5 → 6.0–6.1 (+33–36 %), c=1 wash; needle 12/12 DET, eval
  179/200 (unchanged).
  ([entry](docs/experience/wins/2026-08-21-paged-attention-quantized-gqa-shared-dequant.md))

- **`--spec-type auto` now resolves on the multiproc path.** The 0.5.8 default
  was lowered only in `serve_http`, which the multiproc coordinator never runs,
  so multi-GPU DSv4 workers loaded without the MTP head. Same-day cleanup also
  made `--mtp-draft-tokens` a loud error when `auto` finds no head, and deleted
  the closed-axis decode GEMM probe.
  ([entry](docs/experience/errors/2026-08-21-spec-auto-missing-on-multiproc.md))

- **W4AFP8 GEMV extended to M>1 for DSpark verify — rejected: 8.8% slower than
  CUTLASS grouped GEMM (42.4 vs 46.5 tok/s on the same long prompt), reverted in
 the same pass pins DSpark to c=1 (−32% at c=8, −47.7% at c=16 —
  the sequential draft tax scales with batch while verify savings do not).**
  ([entry](docs/experience/wins/2026-08-21-w4afp8-gemv-decode-lane.md))

## [0.5.8] - 2026-08-21

Two 4-bit checkpoint families now serve on CUDA, and the decode profile that
ranked their optimisation work turned out to have been taken at the wrong batch.

### NVFP4 serving — both families

- **DeepSeek-V4-Flash NVFP4 → W4AFP8, converted at load.** Routed MoE experts
  ship as E2M1 packed in I8 with F8_E8M0 per-1x32-block scales; the SGLang W4A8
  CUTLASS kernel wants signed INT4 with BF16 per-1x128-block scales. The
  conversion runs on GPU per expert, byte-neutral, and never retains the source,
  which is what keeps TP=2 inside 96 GB/GPU.
  ([TP=2](docs/experience/wins/2026-08-18-nvfp4-w4afp8-tp2-serve.md) ·
  [TP=4 decode](docs/experience/wins/2026-08-19-w4afp8-tp4-decode.md) ·
  [GEMV decode lane](docs/experience/wins/2026-08-21-w4afp8-gemv-decode-lane.md))
- **Qwen3.8-27B NVFP4, one resident layout.** Marlin `kFE2M1f` at decode; above
  the DeepGEMM prefill floor the same Marlin layout is widened to E4M3 in
  scratch for DeepGEMM's native FP8 MMA, 84 → 274 TFLOPS. Against
  Qwen3.6-27B-FP8 on one H20: 32K long-agent ITL +21.3% / +20.7% / +13.2% /
  +5.5% at c=1/4/8/16, resident 22.36 GB against 29.36, KV pool 1,779,114
  against 1,582,506.
  ([entry](docs/experience/wins/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md))

### Default flip

- **`--spec-type auto` is the default, and it is implemented.** It speculates
  whenever the checkpoint declares an MTP head (`mtp_num_hidden_layers`, which
  Qwen3.5 nests under `text_config`, or `num_nextn_predict_layers`; GLM ships 0).
  c=1 goes 20.50 → 11.94 ms per committed token at d=2; d=4 is not better. Above c=1 it is inert. Needle ladder exact x3 DET at
  four lengths, with engagement proven by a counter rather than by identical
  text — speculation is output-preserving under greedy, so the ladder alone
  cannot tell "ran and was correct" from "never ran".
  ([entry](docs/experience/wins/2026-08-21-spec-type-auto-default.md))

- **`--metal-warmup` defaults off: Metal cold start 0.62 → 0.35 s.**
  ([entry](docs/experience/wins/2026-08-20-cold-start-warmup-off-config-dedup.md))

### Decode

- **The decode profile was taken at the wrong batch, and it inverts.** Every
  lever had been ranked off Marlin 68.3% / attention 1.6%. At serving
  concurrency it is **attention 80.6% at c=16 and 82.7% at c=32**, Marlin 13.9%
  / 12.7% — attention scales with batch x context while the weight read does
  not, so their shares cross.
  ([entry](docs/experience/errors/2026-08-21-decode-profile-taken-at-the-wrong-batch.md))
- **Paged-attention KV row in one vector load.** A lane's 8 KV bytes are
  contiguous, so 48 `LDG.E.U8` become 4 `LDG.E.64`, and the KV scale leaves the
  per-element loop (`FMUL` 300 → 260). `cuobjdump` was the gate: it proved the compiler was not already
  vectorising, then caught two regressions in the fix before either reached a
  bench.
  ([entry](docs/experience/wins/2026-08-21-paged-attention-vector-load.md))

### Eval

- **NVFP4 matches same-base FP8, and the kernel work costs nothing.** 200 items,
  greedy, FP8 KV, one GPU back to back: NVFP4 with this release **179/200
  (89.5%)**, NVFP4 control 180/200 (90.0%), `Qwen/Qwen3.8-27B-FP8` **177/200
  (88.5%)**. At n=200 the binomial spread is ~4.2 items, so all three are
  indistinguishable — the paged-attention change is quality-neutral, and the
  4-bit checkpoint matches the 8-bit one on the same base. This closes the
  measurement debt of comparing Qwen3.8-NVFP4 against Qwen3.6-FP8, two different
  models. **Not a GSM8K result**: the items come from the repo's own
  `examples/opd/gsm8k-train.jsonl`, the TRAIN split; only the arm-to-arm
  difference on identical inputs carries.

### Verdicts


- **Metal decode — three directions ruled out on Qwen3.8-27B-MLX-2bit (M4 Pro,
  27.1 tok/s ceiling): fused GDR postprocessing within noise (21.5 → 21.6 tok/s,
  MLX async dispatch already hides the launch), the strong norm+gate-in-scan
  variant saves 0.009% of a step by traffic math, and the 2-bit matmul is at 83%
  of bandwidth.**
  ([entry](docs/experience/errors/2026-08-21-metal-decode-three-directions-ruled-out.md))
- **Conv-pair backward recompute — rejected: 6,720 MiB of tape tensors bought
  0 MiB residency (69,665 MiB before and after at local 65,536, bit-identical).
  Tensor bytes are not residency.**
  ([entry](docs/experience/errors/2026-08-20-tensor-bytes-are-not-residency.md))
- **`--kv-recall` — rejected and deleted: shrinking the attended
  set buys nothing while HBM is not the binding constraint, and the cost is paid
  regardless. L2/L3 keep only the lossless capacity path; the five correctness
  fixes stand as tiering plumbing.**
  ([entry](docs/experience/wins/2026-08-18-kv-recall-repaired-cp-and-selector.md))
- **DSv4 32K context cap — accepted (deleted): obsolete four days after the
  demand-paged joint (num_slots, pool_tokens) budget solve landed; `serve` passes
  `max_position_embeddings` through unmodified and starts at
  `max_prompt_tokens=1048576` (budget solver plans 4 slots on 4×H20, no crash).**
  ([entry](docs/experience/wins/2026-08-19-dsv4-context-cap-deleted.md))
- **Spec decode on NVFP4 — rejected for both paths: no spec 60.2 tok/s, DSpark
  16.8 (−72%), MTP 13.9 (−77%); the 6.5× single-token decode speedup makes the
  verify forward dominate, so speculation's arithmetic is inverted.**
  ([entry](docs/experience/wins/2026-08-19-nvfp4-marlin-tensorcore.md))
- **Per-channel FP8 on Marlin — accepted: the 145 GEMMs still on a scalar
  batched GEMV route to Marlin `kFE4M3fn` (already instantiated, no nvcc cost).
  +100.2% at c=16 (235.9 → 472.4 tok/s), NVFP4 now passes same-base FP8 through
  c=8 (+33.4% c=1 … +2.5% c=8); numerics 31/31, engagement proven by counter.**
  ([entry](docs/experience/wins/2026-08-19-marlin-fp8-per-channel.md))
- **Marlin blocks-per-SM search — accepted (`MARLIN_MAX_BLOCKS_PER_SM=5`, was
  pinned to 1): +4.5% decode, needle 6/6. It exposed two latent bugs
  (shared-mem budget lowered in place across chunks; lock buffer sized for one
  block per SM — an out-of-bounds write at bps=5), both fixed. The 32K-crash
  revert experiment was retracted as an unengaged arm — the revert arm never hit
  a partial prefix restore in 207 requests; cause not established, search stays.**
  ([entry](docs/experience/errors/2026-08-19-blocks-per-sm-search-two-latent-bugs.md))
- **Wave-2 dead-code deletion W8A16 27B champion-row A/B — accepted: ITL p50
  16.72 vs 16.70 ms, p99 +0.9%, TTFT −0.3%, all inside the noise floor; perf
  license granted for the one live-path touch (the unused `max_shared_mem`
  Marlin kernel arg).**
  ([entry](docs/experience/wins/2026-08-18-dead-code-deletion-wave2.md))
- **All-GPTQ W4A16 on V100 — rejected for non-expert weights: garbage output
  ("1+1=" → "iginigin_2222222"); the GPTQ kernel is correct for experts only.
  Workaround: dequantize non-expert weights to BF16 in checkpoint prep (+2 GB
  VRAM, 30 GB loaded, fits 32 GB).**
  ([entry](docs/experience/wins/2026-08-17-autoround-w4a16-v100.md))
  above M=32, not a replacement for Marlin.** Driven with one group as a dense
  GEMM it is 0.65x Marlin at M=1 and 0.90x at M=16, then 1.08x at M=32, 1.46x at
  M=48 and 1.90x at M=64. Not tile waste: at M=1 Marlin achieves 1,594 GB/s
  against the collective's 952 GB/s while reading more bytes, so wgmma and TMA do
  not help at one row. The same sweep relocated the lever — **Marlin costs the
  same at M=8 as at M=1** (0.0629 ms for 1, 4 and 8 rows), so the attack on its
  68.3% of decode is row count, which concurrency and speculative decode
  (`M = b*(d+1)`) both supply. Two latent crashes fixed on the way: the grouped
  GEMM aborted for any odd expert count on TMA descriptor alignment, then on the
  CUTLASS workspace start; even counts are unchanged.
  [entry](docs/experience/errors/2026-08-21-sm90-collective-loses-below-m32.md)
- **`gdr_decode_batch_kernel` — measured: latency-bound, not at a ceiling.** 13.0%
  of decode GPU time at 17.5-19.4% of compute peak and 20.2-22.5% of memory, IPC
  0.66, achieved occupancy 30.9% against an uncapped theoretical 100%. 61.6% of
  the stall is Long Scoreboard.
  [entry](docs/experience/wins/2026-08-21-gdr-decode-batch-is-latency-bound.md)

- **cp=2 131,072 — verdict: below the card by ~2–5 GB; the a2a linear-attention
  core is the ceiling.** Layer replays chunked end-to-end (linear projections,
  CP full attention over q tiles with ring FA3 tiled-q support, core over head
  groups): linear-layer replay 20.3 → 6.9 GB, full-attn 25.9 → ~4 GB, live peak
  75 GB of 97.5. Remaining deficit is allocator hoard around the core's
  O(global-seq) transient. Five stacked faults fixed on the way (NCCL teardown
  hang, pre-backward hoard, per-chunk trim, ring backward k-extents, flashqla
  geometry). Next step is a sequence-parallel core with cross-rank state carry.
  NVFP4 frozen base numerically validated (Δloss 0.012%), sharing unwired.
  [entry](docs/experience/errors/2026-08-21-cp2-131072-stacked-faults-and-the-a2a-core-ceiling.md)

## [0.5.7] - 2026-08-21

Seven days, 547 commits. Three threads: NVFP4 becomes a serving format worth
using, context-parallel training gets its ceiling back, and the CUDA operator
layer gets an organization.

**NVFP4 is now the smaller model, not just the smaller file.** A 23.4 GB 4-bit
checkpoint went from 39.3 GB resident — 10 GB *more* than the FP8 model it
competes with — to 22.4 GB, 7 GB *less*, by deriving DeepGEMM's prefill operand
from Marlin's resident layout instead of keeping both. Prefill moved off Marlin's
BF16 widening onto the FP8 tensor cores (84 → 274 TFLOPS), and two kernels on
that path were 4.0x and 3.7x off. Against Qwen3.6-27B-FP8 on the 32K agent
workload it now leads on ITL at every concurrency.

**CP training** recovered its 2-GPU sequence ceiling (114,688 → 131,072) and had
a 6.4x gradient error root-caused to one dropped `* 2` in a byte offset.

**~90 typed CUDA launchers** replaced hand-rolled FFI across seven phase exits.

- **REFACTOR (phase exit T5 + T8 closure) — CUDA operator organization complete: infer-cuda moe.rs split into qwen/dsv4/dsv4_deepep policy files; registry bound for qwen35/dsv4 MoE experts and dsv4 transport (15 semantic, 45 implementations); multi-model receipt (NVFP4 + two FP8 27B) identical counters/text, needle 12/12** (2026-08-21, [wins](docs/experience/wins/2026-08-20-cuda-operator-organization-t-series.md))
- **REFACTOR (phase exits T0–T4, T6, T7, T8-prep) — CUDA operator organization: ~90 typed launchers, infer-cuda raw FFI 217 sites → 0 (GDR fn-pointer table and peer-held moe.rs excepted), loader split 6,722 → 2,859+3,273+2,302, load-time quant storage validation, 12-operator registry, autograd NVRTC catalog+identity; remote receipt on H20: route counters and greedy text identical, needle 12/12 both binaries. T5 deferred (peer holds moe.rs)** (2026-08-20..[wins](docs/experience/wins/2026-08-20-cuda-operator-organization-t-series.md))
- **VERDICT (accept) — Qwen quant-linear dispatch consolidation (T2 tranche 1): one dispatcher, one route owner per weight family; all five remote gate classes pass on H20 — numerical parity, 5/5 route counters + identical completion, decode-graph capture, needle 12/12 ×2 families + lever PASS, 32K A/B within noise at c=1/4/8/16 (first c=1 baseline OOM was a foreign 22 GB resident; clean-GPU re-run 3/3, zero OOM), eval_harness 3/3** (2026-08-20, [errors](docs/experience/errors/2026-08-20-quant-linear-dispatch-consolidation-pending-remote.md))
- **PERF — two prefill kernels on the NVFP4 path, 4.0x and 3.7x: non-GEMM overhead 838 → 303 ms (24.5% → 10.5% of the quantised path), and the 32K chain leads on ITL at every concurrency (+21.3% / +20.7% / +13.2% / +5.5% at c=1/4/8/16)** (2026-08-20, [wins](docs/experience/wins/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md), [rows](docs/baselines.md))
- **PERF — NVFP4 prefill on FP8 tensor cores: resident 39.35 → 22.36 GB (FP8 29.36), KV pool 1,302,407 → 1,779,114, 32K chain ITL +21.3/+18.2/+11.7/+3.9% at c=1/4/8/16** (2026-08-20, [wins](docs/experience/wins/2026-08-20-nvfp4-widen-to-e4m3-deepgemm-prefill.md), [rows](docs/baselines.md))
- **REFACTOR — NVFP4 single serving path: `fp4_route` dequant/GEMV variants, five dead W4A8 sidecar fields, two unreachable guards, one duplicated DeepGEMM launch removed; −458 lines; unsupported shape/SM tier now fails at load with the reason** (2026-08-20)
- **VERDICT (accept) — Qwen speculative decoding honours the thinking budget: `--max-thinking-tokens 8` holds at 8 reasoning tokens on three prompts; the unlimited control runs 439/600/600. W8A16 lm_head routing in the same commit is unvalidated — no checkpoint on the box can reach it (both tie word embeddings to a BF16 tensor)** (2026-08-20, [errors](docs/experience/errors/2026-08-20-qwen-spec-budget-and-w8-lm-head.md))
- **PERF — 2-GPU CP training ceiling 114,688 → 131,072: linear-attention core gets its own checkpoint sub-group; CP transport frees as it consumes** (2026-08-20, [wins](docs/experience/wins/2026-08-20-cp2-ceiling-114688-to-131072.md), [rows](docs/baselines.md))
- **PERF — Marlin no longer stores the model twice: freeing pre-repack bytes returns 18.7 GB, KV pool 281,577 → 790,603, 32K long-agent chain full recomputes 24 → 2, ITL ahead of FP8 at every point (+21.3% / +18.4% / +11.5% at c=1/4/8)** (2026-08-20, [wins](docs/experience/wins/2026-08-20-marlin-source-freed-18gb.md), [rows](docs/baselines.md))
- **FIX — Marlin fp32-reduce buffer sized for one block per SM while the grid is `sms × blocks_per_sm`: CUDA_ERROR_ILLEGAL_ADDRESS past ~512 tokens; root cause of the 33K prefill crash and of the MARLIN_MAX_BLOCKS_PER_SM=1 pin working** (2026-08-20, [errors](docs/experience/errors/2026-08-20-marlin-reduce-buffer-sized-for-one-block-per-sm.md))
- **VERDICT (accept) — #228 batched FlashMLA decode corruption fixed by (2026-06-15, indices reader pitch = writer pitch), verified on pod 2026-08-19: batch=4 byte_parity=true; batch=8 5/6 pass (1 failure = #229, separate non-FlashMLA bug). Issue closed.** (2026-08-19; [wins](docs/experience/wins/2026-08-19-batched-flashmla-decode-verified.md), [#228](https://github.com/cklxx/arle/issues/228), [#229](https://github.com/cklxx/arle/issues/229))
- **VERDICT (close — not reproducible) — #229 DSv4 concurrent-decode digit corruption: 40 `dsv4_parity` batch-decode trials (20 needle + 20 repeated-pattern, batch=8, TP=4) produced 0 failures; model experts are NVFP4, and the FP8 MoE kernel suspected in the #229 doc was replaced by the W4AFP8/NVFP4 path. Issue closed.** (2026-08-20; [errors](docs/experience/errors/2026-07-06-dsv4-concurrent-decode-digit-corruption-unresolved.md), [#229](https://github.com/cklxx/arle/issues/229))
- **PERF — NVFP4 c=16 decode +51.7% (472.4 → 716.6 tok/s): the last four load sites had no Marlin repack, so 34% of FP8 GEMM calls stayed on the scalar GEMV** (2026-08-19, [wins](docs/experience/wins/2026-08-19-nvfp4-marlin-remaining-load-sites.md), [rows](docs/baselines.md))
- **VERDICT (reject) — Marlin decode occupancy: three tunings, all reverted; warp occupancy 20.7% → 30.7% while throughput fell, the shared-memory over-request removal is a wash, and buying occupancy with registers spills** (2026-08-19, [errors](docs/experience/errors/2026-08-19-marlin-decode-is-not-occupancy-limited.md))
- **FIX — CP ring FA3 pair offsets are bytes: one dropped `* 2` gave 6.4× wrong training gradients for two days; cp=2 grad_norm 14.01 → 2.15 vs cp=1's 2.20** (2026-08-19, [wins](docs/experience/wins/2026-08-19-cp-ring-fa3-byte-offset-fix.md), [errors](docs/experience/errors/2026-08-19-cp-training-gradients-regressed-and-the-gate-is-dead.md))
- **ERROR — CP training gradients 6.4× off single-card (grad_norm 14.01 vs 2.20, 27B cp=2 seq=32768); the 0.8B CP correctness arm has been unrunnable since FlashQLA went default-on 2026-08-05** (2026-08-19; [errors](docs/experience/errors/2026-08-19-cp-training-gradients-regressed-and-the-gate-is-dead.md), [rows](docs/baselines.md)) — bisect pending
- **BASELINE (re-anchor) — cp=4 training seq ceiling 229376; seq=131072 step is 17.5× the 2026-08-03 row (3100 s → 177.5 s)** (2026-08-19, [wins](docs/experience/wins/2026-08-19-cp4-seq-ceiling-229376-and-17x-step.md), [rows](docs/baselines.md))
- **FIX — one prefill dequant arm firing at `M >= 2` cost NVFP4 5× aggregate throughput at c≥2, inverted both spec-decode paths, and crashed the server at 34K: 11.56 G FP8 params re-materialised to BF16 every step** (2026-08-19; [errors](docs/experience/errors/2026-08-19-fp8-dequant-arm-shadows-decode.md), [rows](docs/baselines.md)) — after-arm re-measure pending on the fixed binary
- **FIX — whole-slot park works under CP: 9,970/9,970 refusals → 390/390 round-trips, promote 130 ms @ 10K tokens, needle 48/48** (2026-08-19..[wins](docs/experience/wins/2026-08-19-cp-slot-park-works-l2-l3-nonzero.md), [errors](docs/experience/errors/2026-08-19-cp-park-refused-so-l2-l3-never-written.md))
- **REFACTOR — dead-code deletion wave 4: −426 lines, non-CUDA crates, zero live paths touched** (2026-08-19, [wins](docs/experience/wins/2026-08-19-dead-code-deletion-wave4.md))
- **FEAT — Qwen3.8-27B-NVFP4 mixed-precision inference: NVFP4 MLP + FP8 per-channel attention on H20** (2026-08-18, [wins](docs/experience/wins/2026-08-18-qwen38-27b-nvfp4-inference.md))
- **FEAT — KV-recall × CP: shard-filtered recall under 2D parallelism; needle 21/21 TP=2 CP=2** (2026-08-18..[wins](docs/experience/wins/2026-08-18-kv-recall-cp-shard-filtered.md))
- **FEAT — MXFP4 W4A16 weights on Metal (opt-in); 9B pilot: affine ladder row retained (8K recall regression reproduced in stock mlx_lm)** (2026-08-18, [wins](docs/experience/wins/2026-08-18-mxfp4-metal-qwen35-9b.md))
- **FEAT — NVFP4 checkpoint loads + FP8 inference: DSv4-Flash-0731 TP=4 on H20** (2026-08-18; [wins](docs/experience/wins/2026-08-18-nvfp4-load-fp8-infer.md))
- **REFACTOR — dead-code deletion: −7,155 lines across 65 files, zero live paths touched** (2026-08-18..[wins](docs/experience/wins/2026-08-18-dead-code-deletion.md))
- **REFACTOR — dead-code deletion wave 2: −5,819 lines, CUDA C++/FFI + cross-crate, zero live paths touched** (2026-08-18..[wins](docs/experience/wins/2026-08-18-dead-code-deletion-wave2.md))
- **FIX — CP prefill snapshotted blind-tail: skip `prefill_row_snapshotted` under 2D; needle ladder 21/21** (2026-08-18, [wins](docs/experience/wins/2026-08-18-cp-prefill-snapshotted-blind-tail-fix.md), [errors](docs/experience/errors/2026-08-18-cp-state-chain-pre-advance-recv.md))
- **FEAT — DP coordinator: least-in-flight multi-group routing** (2026-08-17, [wins](docs/experience/wins/2026-08-17-dp-coordinator.md))
- **PERF — TP/CP NCCL collectives onto comm_stream** (2026-08-17, [wins](docs/experience/wins/2026-08-17-collectives-to-comm-stream.md))
- **PERF — DeepEP host stalls → on-device event ordering** (2026-08-17, [wins](docs/experience/wins/2026-08-17-deepep-host-stalls-event-ordering.md))
- **FEAT (accept) — 35B A3B AutoRound W4A16 runs on V100 (sm_70) via ARLE** (2026-08-17; [wins](docs/experience/wins/2026-08-17-autoround-w4a16-v100.md))
- **FEAT (accept) — CP T3.1: B2 CP decode head-sharding across the cp group** (2026-08-17, [wins](docs/experience/wins/2026-08-17-b2-cp-decode-head-sharding.md), [plan](docs/plans/2026-08-16-cp-ideal-state.md))
- **FEAT (accept) — CP T2: engine prefill context parallelism, replicated KV** (2026-08-16; [wins](docs/experience/wins/2026-08-16-cp-t2b-replicated-kv-prefill.md), [plan](docs/plans/2026-08-16-cp-ideal-state.md))
- **FIX — GDR prefill recurrent kernel: missing `__syncthreads()` smem race** (2026-08-16, [errors](docs/experience/errors/2026-08-16-gdr-prefill-smem-race.md))
- **FIX — windowed-GKD backward: residency bounded to one window; free-after-backward UAF** (2026-08-16)
- **REFACTOR (accept) — CP T1: tape-free ring-attention core shared via cuda-kernels** (2026-08-16, [wins](docs/experience/wins/2026-08-16-cp-t1-ring-core-extraction.md))
- **FIX (accept) — serve lifecycle: explicit memory budget wins; an engine cannot outlive its supervisor** (2026-08-16; [wins](docs/experience/wins/2026-08-16-serve-explicit-budget-and-parent-watchdog.md))
- **FIX (accept) — share-frozen-base: alias fused QKV/gate-up slices, no duplicate FP8 base** (2026-08-16; [wins](docs/experience/wins/2026-08-16-share-frozen-base-fused-slices.md))
- **FEAT (accept) — `--lora-merge-fp8`: 27B all-linear LoRA merge fits one GPU** (2026-08-16, [wins](docs/experience/wins/2026-08-16-lora-merge-requant-fp8.md))
- **FIX (accept) — OPD long-seq OOM: cached teacher hidden + O(n) student forward** (2026-08-16, [wins](docs/experience/wins/2026-08-16-opd-65536-longseq-oom-fix.md))
- **FIX (accept) — reasoning the model produced always reaches the client** (2026-08-15; [wins](docs/experience/wins/2026-08-15-openai-reasoning-content-lane.md))
- **VERDICT — agent-OPD parameter-update path executed on real claude rollouts** (2026-08-15; [wins](docs/experience/wins/2026-08-15-agent-opd-update-path-first-execution.md))
- **VERDICT (accept) — FP8 non-zero-delta merge verified over two rounds on one GPU** (2026-08-15, [wins](docs/experience/wins/2026-08-15-rubric-single-gpu-judge-residency.md))
- **VERDICT (resolve) — DSv4 first-token flip under concurrency = near-tied logit pair; no runtime defect** (2026-08-15; [wins](docs/experience/wins/2026-08-15-dsv4-first-token-flip-near-tied-pair.md), closes #202)
- **FIX (accept) — frozen-base ownership: one invariant, no per-site frees** (2026-08-15; `8c0ac637c`, `24202f656`, [wins](docs/experience/wins/2026-08-15-frozen-base-ownership-single-invariant.md), [review](docs/plans/2026-08-14-frozen-base-sharing-correctness.md))
- **PERF (accept) — w2s gates computed on device: s/step 3.614 → 2.742 (−24.1%)** (2026-08-14; `7b9b13393`, [bench](docs/experience/wins/2026-08-14-w2s-device-gates-and-chunked-regularizers.md))
- **VERDICT — agent-OPD runs end to end on one GPU through the real claude harness** (2026-08-14; smoke at `7b9b13393`, log `/host/aopd-smoke-0814.log`)
- **VERDICT — w2s 60-step e2e on 27B-FP8: confidence threshold is a near-switch (0.99 skips nothing, 0.9 skips 80% on GSM8K)** (2026-08-13; [bench](docs/experience/wins/2026-08-13-w2s-e2e-confidence-near-switch.md))
- **FIX (accept) — LoRA-targeted projections keep the trainer-owned base under frozen-base sharing** (2026-08-14; `7c4c9082f`, [design](docs/plans/2026-08-14-frozen-base-sharing-correctness.md))
- **FIX (accept) — OPD `--engine-offload student` step-1 NaN root-caused: frozen-base alias use-after-free** (2026-08-14; `a1a3fda92`, `ef486bd86`, `4b8b02f9f`, [bench](docs/experience/wins/2026-08-14-opd-offload-student-alias-uaf.md))
- **BASELINE (re-anchor) — Qwen3.6-27B-FP8 DSpark and DSv4-Flash-FP8 8xH20 DSpark re-measured at ** (2026-08-14; [bench](docs/experience/wins/2026-08-14-sampling-penalties-verified-on-both-runtimes.md), [errors](docs/experience/errors/2026-08-14-raw-completion-continuation-flips-with-concurrency.md))

## [0.5.6] - 2026-08-14
- **FIX — single-GPU OPD: teacher pool sizing + engine-offload starvation** (2026-08-14; `f1f568d1a`, `c7f9c68ad`, [errors](docs/experience/errors/2026-08-14-opd-engine-offload-starves-autograd-forward.md))
- **PERF (accept) — OPD bf16 bridge event-ordered: 2.24× over the legacy sync; KV pool trim moved off the host** (2026-08-14, `49b469456`, `7fa81cf6d`, `35a773d52`, [bench](docs/experience/wins/2026-08-14-bf16-bridge-event-ordered.md))
- **FIX — `train w2s --save-every N`; VRAM on every step line** (2026-08-13)
- **REFACTOR (accept) — five monolithic impl blocks holding 30-84% of their file** (2026-08-13, [method](docs/experience/wins/2026-08-13-orthogonal-axes-expanded-into-method-names.md))
- **REFACTOR (accept) — backend_cuda.rs 12874 → 2680 + 21 concept-named modules** (2026-08-13)
- **MEASURE — w2s step budget: the four KL terms are 46.6%, the student forward 12.8%** (2026-08-13, [bench](docs/experience/wins/2026-08-13-w2s-step-budget-kl-terms-dominate.md))
- **REFACTOR (accept) — crates/train deletion, config layering, opd.rs split** (2026-08-13)
- **FIX (accept) — w2s no longer round-trips the FP8 base through host** (2026-08-13, [errors](docs/experience/errors/2026-08-13-w2s-fp8-base-offload-roundtrip-was-lossy.md))
- **FIX (accept) — prefix-cache metrics report actual restored work** (2026-08-13, [bench](docs/experience/wins/2026-08-13-kv-prefix-metrics-and-oversubscription-slice.md))
- **FEATURE (accept) — FA3 quantized KV paths A+B for qwen35** (2026-08-13, [bench](docs/experience/wins/2026-08-13-fa3-quant-paths.md))
- **FEATURE (accept) — DSpark spec decode with quantized KV; L2 tier demote/promote verified** (2026-08-13, [bench](docs/experience/wins/2026-08-13-dspark-quant-kv.md))
- **FIX — HTTP sampling penalties validated at ingress; logit_bias survives the multiproc relay and the greedy fast path** (2026-08-13)
- **INFRA — watchdog startup grace 120s→300s; one-off scripts pruned; conversion and quantization unified** (2026-08-13)

## [0.5.5] - 2026-08-13
- **FEATURE (accept) — batched paged decode for FP8/INT8 KV pools** (2026-08-13)
- **REFACTOR (accept) — unify qwen35 FP8/INT8 KV on the NHD split-KV kernel; delete the dead TileLang FP8 path** (2026-08-13)
- **BENCH — KV dtype comparison on H20, ThinkingCap-Qwen3.6-27B-FP8** (2026-08-13)

## [0.5.4] - 2026-08-12
- **FEATURE (accept) — INT8 KV cache support for Qwen3.5 paged attention** (2026-08-12; `b20859520`)
- **PERF (accept) — FP8 dequant GEMV floor lowered to M>=2: WMMA GEMM replaces cuBLAS for small batches** (2026-08-12; `b20859520`)

## [0.5.3] - 2026-08-11
- **PERF (accept) — DSv4 whole-slot KV tier serialization simplified: swap_out/swap_in persist only mutable fields; FP32 carry skipped via `fp32_carry_stale`** (2026-08-11)
- **FIX (accept) — DeepGEMM native build fixes for CUDA 12.9** (2026-08-11)
- **VERDICT (reject) — FlashQLA `block_DV=32` improves wave count but fails numerical parity** (2026-08-10, [error](docs/experience/errors/2026-08-10-flashqla-block-dv32-numerical-kill.md))
- **VERDICT (reject and revert) — unmeasured CUDA split, fast-math, and GEMV changes regressed correctness** (2026-08-10, [error](docs/experience/errors/2026-08-10-unmeasured-cuda-micro-optimizations-regressed-correctness.md))
- **CALIBRATION — the anchor's `nsys` window over-states prefill kernel shares by 2.02×** (2026-08-09; `nsys` at [bench](docs/experience/wins/2026-08-09-pack-quantize-warp-per-block.md))
- **PERF (accept) — `pack_quantize` at 16 B loads: 5.13×, still bit-identical** (2026-08-09, [bench](docs/experience/wins/2026-08-09-pack-quantize-warp-per-block.md))
- **PERF (accept) — `pack_quantize` was instruction-bound; one warp per block gives 3.67× and −2.98% anchor wall** (2026-08-09, [bench](docs/experience/wins/2026-08-09-pack-quantize-warp-per-block.md))
- **MODEL (supersede) — the anchor window is now an exact partition; prefill arithmetic is at the hardware floor** (2026-08-09, [bench](docs/experience/wins/2026-08-09-anchor-window-partitioned-exactly-prefill-arithmetic-is-finished.md))
- **FIX (root cause confirmed) — the Qwen3.6 trunk's final RMSNorm applied `w` instead of `(1+w)`; every eval on this model before today is a floor** (2026-08-08, [entry](docs/experience/errors/2026-08-08-qwen36-final-norm-missing-offset.md))
- **VERDICT (close the lever) — the anchor's FP8 GEMM is 57.7% of all kernel time and runs at ~90% of FP8 peak** (2026-08-08, [bench](docs/experience/wins/2026-08-08-anchor-fp8-gemm-is-at-90-percent-of-peak.md))
- **PERF (accept) — DSpark draft attention was launched once per slot at 192 blocks; batching the slot axis gives ITL mean −10.4%** (2026-08-08, [bench](docs/experience/wins/2026-08-08-dspark-draft-attention-slot-batched.md))
- **VERDICT (accept) — agent-opd rollout concurrency; production config is cp=4 × G=2** (2026-08-08, [bench](docs/experience/wins/2026-08-07-agent-opd-rollout-fleet.md))
- **BASELINE — decode re-anchored on a decode-shaped workload: draft attention is 30.5% of a tick; the prior anchor priced it at 4.3%** (2026-08-08; `nsys` at [bench](docs/experience/wins/2026-08-08-decode-shaped-reanchor-draft-attention-is-30pct.md))
- **VERDICT (reject the ranking, mechanism confirmed) — FA3 decode-verify is 29.2% of roofline and 0.39% of GPU time; the anchor is a prefill benchmark** (2026-08-08; `nsys` at [entry](docs/experience/errors/2026-08-08-anchor-is-a-prefill-benchmark-decode-levers-ranked-off-it.md))

## [0.5.2] - 2026-08-21
- **BASELINE — corrected Qwen3.6-27B DSpark anchor complete** (2026-08-10; runtime runner [bench](docs/experience/wins/2026-08-10-qwen36-27b-corrected-baseline.md))
- **FIX — benchmark warmup no longer primes a measured prefix** (2026-08-10; [error](docs/experience/errors/2026-08-10-benchmark-warmup-contaminated-cold-session.md))
- **FIX — DFlash draft norms restore Qwen3 plain-weight semantics** (2026-08-10, [error](docs/experience/errors/2026-08-10-dflash-draft-norm-offset.md))
- **FIX — the canonical fixed-output benchmark now forces `ignore_eos=true`** (2026-08-10; [error](docs/experience/errors/2026-08-10-fixed-output-benchmark-allowed-early-eos.md))
- **PERF — V100 (sm_70) prefill: W4A16 dequant→FP16 GEMM + GDR/FA2 tuning**

## [0.5.1] - 2026-08-07
- **VERDICT (accept, end-to-end null) — the prefix sidecar serialized 146.8 MiB per element; bulk copy is −9.5% on the operation and 0.9% of wall** (2026-08-07, [bench](docs/experience/wins/2026-08-07-prefix-sidecar-serialize-bulk-copy.md))
- **FIX — agent-opd cp>1: rank 0 owns rollout, followers mirror the update stream**
- **VERDICT (confirmed) — agent-opd cp=2 fix validated; the cc-rollout training loop closes end-to-end under the new defaults** (2026-08-07/08, pod GPUs 4+5, [error entry](docs/experience/errors/2026-08-07-agent-opd-cp2-rollout-divergence-deadlock.md))

## [0.5.0] - 2026-08-07
- **VERDICT (accept) — the c≥4 DSpark decode regression is CLOSED; anchor re-anchored on ** (2026-08-07; [bench](docs/experience/wins/2026-08-07-dspark-rollback-replay-batched.md))
- **VERDICT (accept) — the DSpark verify linear core is batched; long-agent anchor re-anchored on ** (2026-08-07, [bench](docs/experience/wins/2026-08-07-dspark-verify-linear-core-batched.md))
- **DEFAULT FLIP + VERDICT (accept B / reject C) — `--checkpoint-reload-device` on by default; pinned checkpoint pool stays off** (2026-08-06.. + this flip, [bench](docs/experience/wins/2026-08-06-checkpoint-reload-and-pinned-offload.md))
- **WASH — reshape/rmsnorm backward heal is correct but a no-op for the profiled cost; `7da312d0d` kept** (2026-08-06; [error](docs/experience/errors/2026-08-06-healed-the-wrong-reshape-backward-not-recompute-forward.md))
- **VERDICT (reject) — OPD_SEQ_CHUNK 4096→8192 is a null; backward wall scales with total work on CPU** (2026-08-06; pod-only, [error](docs/experience/errors/2026-08-06-opd-chunk-knob-null-backward-is-total-work-cpu.md))
- **VERDICT (reject) — native FP8 training forward halves the GEMM cluster but moves the step wall 1%** (2026-08-06+[error](docs/experience/errors/2026-08-06-native-fp8-forward-optimized-the-wrong-17-percent.md))
- **FIX — REPL/OCR load caps slots at 1: Qwen3.5-9B now fits in 48 GB** (2026-08-06; [win](docs/experience/wins/2026-08-06-repl-single-slot-load.md))
- **FIX — CUDA serve auto-downloads HF model ids, matching Metal** (2026-08-06; [win](docs/experience/wins/2026-08-06-cuda-serve-auto-download.md))
- **DEFAULT FLIP — FlashQLA GDN chunkwise backward default-on: 80K training step 1.99×, backward 2.14×** (2026-08-05, [win](docs/experience/wins/2026-08-05-flashqla-gdn-backward-default-on-2x.md))
- **CHARACTERIZATION — an 80K training step is one kernel: GDN chunked-scan backward is 71%; FA3 is worth 3.54× at 80K, vs 2.17×** (2026-08-05; [win](docs/experience/wins/2026-08-05-80k-training-step-is-one-kernel.md))
- **DEFAULT FLIP — FA3 is the unconditional CP ring path; `ARLE_CP_RING_FA3` deleted** (2026-08-05, [win](docs/experience/wins/2026-08-05-80k-training-step-is-one-kernel.md))
- **VERDICT — the prefill gap was a stub build: FlashQLA was never compiled into the pod binary; TTFT 31.08 → 25.01 s** (2026-08-05, [win](docs/experience/wins/2026-08-05-flashqla-was-never-compiled-into-the-pod-binary.md))
- **VERDICT — the decode step reaches parity with SGLang; the gap is now entirely prefill** (2026-08-04+[budget](docs/experience/wins/2026-08-04-w8a16-decode-step-kernel-budget.md))
- **DEFAULT FLIP — FA3 decode split ceiling derived from the SM count: −11.2% decode step at batch 1** (2026-08-04++[win](docs/experience/wins/2026-08-04-fa3-decode-splits-fill-the-sms.md))
- **ACCEPT (perf) — FA3 replaces the scalar CP ring-attention kernels: 2.17× per training step; default OFF pending grad parity** (2026-08-04+++[win](docs/experience/wins/2026-08-04-fa3-ring-attention-2x.md))
- **ACCEPT — GDR chunk-prepare native CUDA: 289× per launch, −10% training fwd wall, losses bit-identical** (2026-08-03, [win](docs/experience/wins/2026-08-03-gdr-prepare-native-289x.md))
- **PHASE EXIT — CP×DP mesh verified end-to-end; 131072 cp=4 runs clean; the training step is attributed** (2026-08-03++++[win](docs/experience/wins/2026-08-03-cpxdp-verified-and-training-step-attributed.md))
- **ACCEPT — per-token O(cached-pages) scan → O(1) counter: −6.0% decode ITL (cumulative −29.4%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-resident-page-scan-per-token.md))
- **ACCEPT — T6 GDN decode kernel: −2.8% decode ITL (cumulative −24.9%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t6-gdn-decode-kernel.md))
- **ACCEPT — T5b lm_head GEMV → cuBLASLt: −2.8% decode ITL (cumulative −22.7%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t5b-lmhead-cublas.md))
- **DEFAULT FLIP — `--qwen35-decode-graph` ON (serve + seam default)** (2026-08-03)
- **ACCEPT — T4 whole-step decode graph under paged KV: −7.9% decode ITL (cumulative −20.5%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t4-paged-decode-graph.md))
- **ACCEPT — T2 qkv + qkvz row-fusion: −2.5% decode ITL (cumulative −13.7%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t2-qkv-row-fusion.md))
- **ACCEPT — T5 small-M bf16 GEMV → cuBLAS: −5.1% decode ITL (cumulative −11.5%)** (2026-08-03; [win](docs/experience/wins/2026-08-03-t5-small-m-gemv-to-cublas.md))
- **ACCEPT — T3 in_proj_b+a row-fusion: −4.7% decode ITL (cumulative −6.7% with T1)** (2026-08-03, [win](docs/experience/wins/2026-08-03-t3-in-proj-ba-fusion.md))
- **ACCEPT — T1 gate+up row-fusion: −2.1% decode ITL; Marlin fixed-grid correction re-ranks #196** (2026-08-03, [win](docs/experience/wins/2026-08-03-t1-gate-up-fusion.md))
- **VERDICT — W8A16 matched A/B vs SGLang: same kernel, same weights; the GEMM matches and SGLang decodes 1.57× faster — the gap sits in our runtime** (2026-08-02, [entry](docs/experience/wins/2026-08-02-w8a16-sglang-matched-ab.md))
- **ACCEPT — device-native cat: matched A/B verdict, strict win (−10.6 GB host RSS, ~5.6× faster)** (2026-08-02, [win](docs/experience/wins/2026-08-02-device-cat-ab-strict-win.md))
- **PHASE EXIT — W8A16 Marlin tensor-core GEMM: bf16-class decode at half the weight VRAM** (2026-08-02, [win](docs/experience/wins/2026-08-02-w8a16-marlin-tensorcore.md))
- **PHASE EXIT — real 27B 256K CP training runs end-to-end; the VRAM wall is measured at 94.2 GB/GPU (cp=2 fits)** (2026-08-02, [win](docs/experience/wins/2026-08-02-linear-attn-cp-a2a-reorder-256k-runs.md))
- **ACCEPT — FA3 for batch==1 prefill (−4%) and the driver-context thread-lottery fix** (2026-08-02, [win](docs/experience/wins/2026-08-02-fa3-batch1-prefill-and-ctx-bind.md))
- **DEFAULT FLIP — `--qwen35-gdr-chunked` ON, licensed by the chat-format battery** (2026-08-02)
- **VERDICT — the chunked-GDR GSM collapse adjudicated: bf16 drift in a margin-sensitive harness; kernels correct; chat-format quality identical** (2026-08-02; probe [error](docs/experience/errors/2026-08-02-gdr-chunked-gsm-collapse-was-a-knife-edge-harness.md))
- **REVERT — `--qwen35-gdr-chunked` default back to OFF: GSM8K 11/100 vs 46/100** (2026-08-02; flip revert)
- **ACCEPT — FlashQLA chunked GDR generalized to head geometry and made real: 33K prefill −27%** (2026-08-02 + [win](docs/experience/wins/2026-08-02-flashqla-chunked-gdr-h48.md), [error](docs/experience/errors/2026-08-02-pod-b64-arg-truncation-stale-binary.md))
- **PHASE EXIT — the 27B step is profiled end to end; the backlog is re-ranked off measured share** (2026-08-01; [win](docs/experience/wins/2026-08-01-prefill-and-decode-step-budget.md), row in [docs/baselines.md](docs/baselines.md))
- **REJECT — `--qwen35-decode-graph` is a no-op under paged KV** (2026-08-01; not landed, [error](docs/experience/errors/2026-08-01-decode-graph-flag-is-a-noop-under-paged-kv.md))
- **ACCEPT — CP training now actually rings: fixed a `self.cp` split-brain, pod-verified FAIL→PASS** (2026-08-01, [error](docs/experience/errors/2026-08-01-cp-split-brain-forward-read-self-cp-not-arg.md))
- **REJECT — the draft attention is ALU-bound, but removing the IDIV only pays in a microbench** (2026-08-01; not landed, [error](docs/experience/errors/2026-08-01-draft-attention-idiv-win-is-microbench-only.md))
- **REJECT — the DSpark draft attention's per-key reduction axis** (2026-08-01; reverted in [error](docs/experience/errors/2026-08-01-draft-attention-reduction-axis-was-not-the-cost.md))
- **ACCEPT — ISO-Merger grafts one RL skill onto another, same-lineage, data-free** (2026-08-01, [win](docs/experience/wins/2026-08-01-iso-merger-same-lineage-27b-graft.md))
- **PHASE EXIT — CP correctness core complete: ring full-attn + zigzag load-balance + linear-attn all-to-all-to-head, all CPU-gated** (2026-07-31, [win](docs/experience/wins/2026-07-31-cp-zigzag-seqshard-per-row-positions.md), [win](docs/experience/wins/2026-07-31-linear-attn-cp-all-to-all-to-head.md), [win](docs/experience/wins/2026-07-30-cp-ring-attention-and-all-to-all.md))
- **DEFAULT FLIP — DSpark static confidence truncation deleted; the head now drives the paper's goodput budget** (2026-07-30; [win](docs/experience/wins/2026-07-30-dspark-markov-confidence-batched.md))
- **DEFAULT FLIP — OPD seq-chunked recompute is unconditional; verdict still PENDING** (2026-07-30, [win](docs/experience/wins/2026-07-30-seq-chunk-bake-in-and-dparam-offload.md))
- **DEFAULT FLIP — `--dspark-conf-threshold` 0.5 → 0: the shipped default made spec decode slower than no spec decode** (2026-07-30; [win](docs/experience/wins/2026-07-30-dspark-markov-confidence-batched.md))
- **ACCEPT the batching, REJECT the confidence truncation — DSpark markov+confidence checkpoints now speculate at concurrency** (2026-07-30, [win](docs/experience/wins/2026-07-30-dspark-markov-confidence-batched.md))
- **REJECT — data-free MoE expert merge (Qwen3.6-35B-A3B, 256→N)** (2026-07-30; [error](docs/experience/errors/2026-07-30-moe-expert-merge-collapse.md))
- **DEFAULT FLIP — `--spec-max-batch` 1 → 16: Qwen3.5/3.6 DSpark now speculates at concurrency** (2026-07-29, [win](docs/experience/wins/2026-07-29-dspark-varlen-replay-c16-win.md), [win](docs/experience/wins/2026-07-29-dspark-batched-draft-across-slots.md))
- **ACCEPT — context-parallel N=2 OPD writeback runs end-to-end** (2026-07-29, [win](docs/experience/wins/2026-07-29-context-parallel-n2-writeback-works.md), [error](docs/experience/errors/2026-07-29-cp-nccl-wedge-is-hashmap-param-order.md))
- **ACCEPT — Qwen3.5/3.6 paged full attention: one launch per layer** (2026-07-28, [win](docs/experience/wins/2026-07-28-fa3-one-launch-per-layer.md))
- **ACCEPT — Qwen3.5/3.6 converges onto the host-authoritative KV page mirror** (2026-07-28, [win](docs/experience/wins/2026-07-28-qwen35-host-authoritative-kv-mirror.md))
- **ACCEPT — training-system correctness program, Phases 1–5** (2026-07-27; commits)
- **CLOSE (Phase 7a — long agent writeback)** (2026-07-28; forward-peak win [win](docs/experience/wins/2026-07-28-opd-writeback-forward-peak-freed.md), [decomposition](docs/research/2026-07-27-opd-writeback-wall-decomposition.md))
- **REJECT (premise) — ISO near-isospectral premise fails on the DSpark head** (2026-07-28; [errors](docs/experience/errors/2026-07-28-iso-premise-fails-on-dspark-head.md))
- **WITHDRAWN — "DSpark is net-negative once decode is fast"** (filed 2026-07-27 as a REJECT; withdrawn 2026-07-28)
- **ACCEPT — Qwen3.6-35B-A3B MoE is the faster serving target on 1×H20** (2026-07-28; [champion row](docs/baselines.md))
- **ACCEPT — sm_90 paged decode attention routes to vendored FA3** (2026-07-27 + win: [2026-07-27-fa3-paged-decode-32k-2.76x](docs/experience/wins/2026-07-27-fa3-paged-decode-32k-2.76x.md))
- **ACCEPT — DSpark markov path batches by speculating its own chain** (2026-07-26 + win: [2026-07-26-dspark-markov-chain-self-speculation](docs/experience/wins/2026-07-26-dspark-markov-chain-self-speculation.md))
- **ACCEPT — Agent RFT uses generation-time behavior probabilities** (2026-07-26; win: [2026-07-26-agent-rft-sidecar-denominator](docs/experience/wins/2026-07-26-agent-rft-sidecar-denominator.md))
- **DEFAULT FLIP — OPD carry GDN backward routes through the device chunked path** (2026-07-26 + bench: [2026-07-26-carry-gdn-device-reroute-tranche2](docs/experience/wins/2026-07-26-carry-gdn-device-reroute-tranche2.md))
- **REJECT (current form) — online markov-head self-RL cannot reach training scale; the markov path taxes 22.5%** (2026-07-26; bench: [2026-07-26-markov-head-online-selfrl-cannot-reach-scale](docs/experience/errors/2026-07-26-markov-head-online-selfrl-cannot-reach-scale.md))
- **FINDING — the DSpark draft is a good ranker and a bad argmax** (2026-07-26; bench: [2026-07-26-dspark-draft-is-a-good-ranker-bad-argmax](docs/experience/wins/2026-07-26-dspark-draft-is-a-good-ranker-bad-argmax.md))
- **AMEND — DSpark block size is a lever at concurrency** (2026-07-26; bench: [2026-07-26-dspark-block-size-is-a-lever-at-concurrency](docs/experience/wins/2026-07-26-dspark-block-size-is-a-lever-at-concurrency.md))
- **ACCEPT — one ragged-window launch per DSpark draft layer** (2026-07-26; bench: [2026-07-26-dspark-ragged-window-draft-attention](docs/experience/wins/2026-07-26-dspark-ragged-window-draft-attention.md))
- **ACCEPT (mechanism only, no serving delta) — one batched argmax per DSpark tick** (2026-07-26, `308c8b247`; bench: [2026-07-26-dspark-batched-argmax-tick](docs/experience/wins/2026-07-26-dspark-batched-argmax-tick.md))
- **WORKLOAD — bench workload is multi-turn agent sessions at the TraceLab medians** (2026-07-26; bench: [2026-07-26-long-agent-32k-is-the-workload](docs/experience/wins/2026-07-26-long-agent-32k-is-the-workload.md))
- **PHASE EXIT — spec-decode concurrency gate; three dispatch ladders → one `route_decode`** (2026-07-26; win: [2026-07-26-spec-decode-concurrency-gate](docs/experience/wins/2026-07-26-spec-decode-concurrency-gate.md))
- **DEFAULT — `spec_max_batch = 1`** (2026-07-26)
- **VERDICT — #128 DSpark accept-or-kill: KEEP as a c=1 feature; the 07-20 +63.8% vs 07-25 +5% gap was the dataset** (2026-07-26)
- **VERDICT — backward re-offload lifts the OPD-writeback device wall 24576→32768; 256K needs LA-chunk, beyond more offload** (2026-07-25; win: [2026-07-25-backward-reoffload-device-wall-24576-to-32768](docs/experience/wins/2026-07-25-backward-reoffload-device-wall-24576-to-32768.md))
- **REJECT — #127 "train a DSv4 draft head"; the trained head is public** (2026-07-25; docs/architecture-dsv4.md §7 corrected)
- **VERDICT — #160 device-fit park gate closed: backstop only, unreachable in practice** (2026-07-25, #160 closed; wins: [2026-07-24-dsv4-band-exhaustion-park-gate](docs/experience/wins/2026-07-24-dsv4-band-exhaustion-park-gate.md))
- **REJECT — "DSv4 cold boot is serialized on rank 0"** (2026-07-25, #181 closed not-planned)
- **DEFAULT FLIP — Qwen KV pool sizing: measured VRAM outranks the page floor** (2026-07-25, #178; wins: [2026-07-25-kv-pool-floor-yields-to-measured-vram](docs/experience/wins/2026-07-25-kv-pool-floor-yields-to-measured-vram.md))
- **DEFAULT FLIP — `--kv-disk` with a zero derived budget degrades to no-tier** (2026-07-25, #158)
- **VERDICT — DSpark V100 TP-lockstep stall: FIXED, measured** (2026-07-25, #168; errors: [2026-07-21-dspark-v100-tp-lockstep-stall-kill](docs/experience/errors/2026-07-21-dspark-v100-tp-lockstep-stall-kill.md))
- **DEFAULT FLIP — writeback-offload threshold 4096 → 16384** (2026-07-24, #172; wins: [2026-07-24-writeback-offload-dial-back](docs/experience/wins/2026-07-24-writeback-offload-dial-back.md))
- **ACCEPT — FP8 quant loss on 27B: −0.25% PPL vs bf16** (2026-07-24, #174; wins: [2026-07-24-ppl-harness-fp8-matrix](docs/experience/wins/2026-07-24-ppl-harness-fp8-matrix.md))
- **REJECT — group-stagger admission for CC preamble prefix reuse** (2026-07-24, reverted in errors: [2026-07-24-group-stagger-premise-false](docs/experience/errors/2026-07-24-group-stagger-premise-false.md))
- **ACCEPT — agent-OPD sandbox staging outside the repo** (2026-07-24+++)
- **ACCEPT — batched linear-attention CUDA device path** (2026-07-24 +)
- **ACCEPT — systematic review-fix sweep (26 findings); one relay regression fixed** (2026-07-24 +)
- **DEFAULT FLIP — self-opd distill path fused → dense** (2026-07-24)
- **REJECT — checkpoint-gate ×4 tightening reverted** (2026-07-24)
- **ACCEPT — agent-OPD rollout is idle-bound; concurrent mega-rollout GO** (2026-07-24)
- **ACCEPT — sm_120 FP8 MoE prefill: CUTLASS grouped GEMM (G2)** (2026-07-22)
### Removed (dead surface — `crates/train`, 2026-07-23)
- **train-crate systematic simplification, −4,134 LOC.**

## [0.4.0] - 2026-07-22
### Added
- **DSpark train sidecar** (`--dspark-train` / serve background trainer); batched verify (B>1); Qwen3.5/3.6 MTP speculative decode; agent-OPD cc-harness path; V100 (sm_70) serving substrate; ThinkingCap-27B-FP8; unified direct L3 storage (`kv-tier`); DSv4 local-NVMe cold load; qualified kernel artifact flow.
### Fixed
- **#167 Qwen3.6 temp>0 sampled-tail garbage; DSv4 extension-prompt prefix reuse; DSv4 plan-repair / HostPagedKvPool fatal at c32; SM-gate Qwen FP8 dense DeepGEMM to Hopper-only (`major == 9`); DSpark draft latent sliding-window.**
### Verdicts (selected)
- **2026-07-25 — bf16 tape Stage 1a (frozen prefix K/V) rejected: no VRAM win on Qwen3.6-27B.**
- **2026-07-21 — #167 closed: Qwen3.6 temp>0 sampled-tail garbage fixed (accept).**
- **2026-07-20 — DSpark train sidecar Phase 1 shipped (accept, end-to-end verified).**
- **2026-07-17 — DSv4 cold-boot #69 closed: fixed in code, disk-bound residual.**
- **2026-07-17 — DSv4 extension-prompt prefix reuse fixed (accept, wash).**
- **2026-07-17 — DSv4 prefill chunk default 128→2048 (default flip, accept).**
- **2026-07-17 — #164/#162 CLOSED (accept): c32 × 300 s oversubscription survival with real preemption (192 events, zero teardowns).**
- **2026-07-16 — adversarial review of the day's commits fixed 8 confirmed defects pre-deployment.**
- **2026-07-16 — DSv4 FP32 probe scratch hoisted off per-slot state (accept): per_slot 9618→338 MB, slot clamp 2→59.**
- **2026-07-16 — DSv4 FP32 prefill compressor grid-parallelized; serial probe kernel deleted (accept).**
- **2026-07-16 — DSv4 FP32 probe limited to prefill; unblocks DSpark (MTP) decode.**
- **2026-07-16 — DSv4 FP32 compressor extended to all compression boundaries.**
- **2026-07-16 — DSv4 FP32 compressor promoted to default.**
- **2026-07-15 — DSv4 long-context correctness blocked.**
- **2026-07-15 — DSv4 MegaMoE retained but correctness-blocked.**
- **2026-07-14 — DSv4 DSpark TP=4 concurrency licensed.**
- **2026-07-14 — DSv4 DSpark prompt router licensed for H20 TP=4.**
- **2026-07-14 — V100 (sm_70) prefill `cudaErrorNotSupported` fixed.**
- **2026-07-14 — DSv4 DSpark correctness PASS, opt-in unchanged.**

## [0.3.0] - 2026-07-12
### Added
- **DSpark block-draft speculative decoding** (`--spec-type dspark`); unified kernel set; content-addressed prebuilt kernel bundle; strategy-driven agent-OPD harness.
### Changed
- **2026-07-11 — DSv4 decode-region KV reuse default ON**
### Eval

- **NVFP4 matches same-base FP8, and the kernel work costs nothing.** 200 items,
  greedy, FP8 KV, one GPU back to back: NVFP4 with this release **179/200
  (89.5%)**, NVFP4 control 180/200 (90.0%), `Qwen/Qwen3.8-27B-FP8` **177/200
  (88.5%)**. At n=200 the binomial spread is ~4.2 items, so all three are
  indistinguishable — the paged-attention change is quality-neutral, and the
  4-bit checkpoint matches the 8-bit one on the same base. This closes the
  measurement debt of comparing Qwen3.8-NVFP4 against Qwen3.6-FP8, two different
  models. **Not a GSM8K result**: the items come from the repo's own
  `examples/opd/gsm8k-train.jsonl`, the TRAIN split; only the arm-to-arm
  difference on identical inputs carries.

### Verdicts
- **2026-07-11 — DSpark draft-KV: cap full-layer at per-request ceiling** (Qwen3.6-27B, CUDA)
- **2026-07-11 — DSpark/DFlash block-draft spec-decode: P1 LICENSED** (Qwen3.6-27B, CUDA)
- **2026-07-11 — DSv4 decode-region reuse: DEFAULT FLIPPED ON** (`--dsv4-decode-reuse`, was opt-in)
- **2026-07-11 — Agent-OPD round −30.1% wall (H20 GPU1 3-arm A/B), quality-neutral**
- **2026-07-10 — DSv4 finish-write-through decode-region reuse: crash-fix gate PASS (opt-in `--dsv4-decode-reuse`), default flip pending perf**
- **2026-07-10 — DSpark-on-OPD default flip: quality-neutral LICENSED (opt-in), concurrency ≥4 DEFERRED**
- **2026-07-10 — DSv4 Route A prefix reuse "identity formula fix": REVERTED**
- **2026-07-10 — Qwen FP8 small-M dense GEMM: DeepGEMM from M=2 LICENSED; M=1 GEMV variants KILLED**
- **2026-07-10 — DSv4 KV-reuse Phases 2b+3b SHIPPED** (#154)
- **2026-07-10 — DSpark on the OPD rollout serve: wall-clock POSITIVE** (first e2e A/B, CC-as-harness, 16 real swe_smith tasks)
- **2026-07-10 — DSv4 prefix reuse RELICENSED (Phase 2a, content-keyed host-resident state pool)**
- **2026-07-10 — DSpark sampled (temp>0) spec decode LICENSED**
- **2026-07-10 — DSpark partial-ctx drafting (P2.5) LICENSED; sampling RNG cleared**
- **2026-07-10 — DSpark trained heads NO-LICENSE (z-lab backbone stays); P2 sampling verify KILLED as-is**
- **2026-07-10 — DSv4 Route A prefix reuse KILLED pending content-keyed redesign; warm-cache needle regression FIXED**
- **2026-07-10 — Qwen3.6 DSpark block draft LICENSED (short-ctx greedy)**
- **DSv4 decode-kernel levers #141/#142/#143 LICENSED (2026-07-04).**
- **Agent-OPD toy-corpus capability lane KILLED; harness + 12-round loop SHIPPED (2026-07-03).**
- **Phase 2 re-scoped; whole-step decode CUDA graph RE-KILLED (2026-06-21).**
### CUDA
- **Qwen3.6 serves on CUDA (2026-06-29); Qwen3.5-122B-A10B at TP4; GLM-5.2 (`glm_moe_dsa`, DSv4-DSA family) wired on the DSv4 path.**
### Metal
- **Qwen3.6 NextN/MTP spec decode shipped (2026-06-21)**
### Server
- **`/v1/chat/completions` now supports `stream=true`** (SSE `chat.completion.chunk` frames with `reasoning_content`/`content` deltas; closes the R5 tranche-2 deferral, #79)
### Repo
- **Renamed `agent-infer` → `arle`**

## [0.2.1] — 2026-06-15
> Consolidated section: tags `v0.1.5` (2026-05-02), `v0.2.0` and `v0.2.1`
> (both 2026-06-15) were cut without changelog sections. Everything below
> spans v0.1.4 → v0.2.1; per-tag artifacts live on GitHub Releases.
### Runtime rewrite — `infer-*` stack becomes the serving truth (2026-06-04)
- Breaking.
### Training surface — OPD-only (2026-05-18)
- Breaking.
### DSv4 perf campaign — adopt official kernels (2026-06-06 → 06-15)
- Official DSA indexer default-on: decode 124 ms → 26 ms flat @4096.
- FlashMLA `sparse_fwd` + FP8 DeepGEMM prefill default-on: 7.2 s → 3.48 s.
- Phase 0 debt closed 2026-06-10 (#56–#59).
- Seam-level KV-dtype dispatch `--kv-cache-dtype` (default bf16 unchanged); INT8/FP8 correctness LICENSED, opt-in pending a perf license (2026-06-12).
- Phase 1 batched-lane keystone closed (#61 2026-06-11, #60 2026-06-15): DSv4 B>1 decode takes the batched serving lane by default; residual c>1 throughput lever is DP-attn (#89).
### OPD train (CUDA) — new beta surface
- OPD mainline queue moved from experiment-only to operator-facing workflow; end-to-end OPD CUDA training stack landed on Qwen3-0.6B.
### Observability
- Low-overhead HTTP `request_trace` JSON summaries for streaming and buffered requests (TTFT, latency, throughput, KV/prefix-cache state, scheduler phase EMA).
- DSv4 Nsight trace artifacts committed under `docs/trace-artifacts/` (2026-05-14/15).
### CUDA
- DSv4 decode scratch reuse and uninitialized-allocation pass across attention, MoE, and compressor buffers (2026-05-15).
- DSv4 B=1 padded BF16 combine reduce-scatter default-on (`ARLE_DSV4_COMBINE_REDUCE_SCATTER`).
- **W4-hybrid prefill graph capture closes the 4k/c=4 gap — Tier 1 STRONG PROCEED** (`a56b7a9`/`c44788f` 2026-05-10; opt-in via `INFER_PREFILL_GRAPH=1` + `INFER_HYBRID_W4A8_PREFILL=1`, `35fc3cf`).
### Long-context (cross-backend)
- **RoPE scaling support** (YARN / Linear / NtkAware)
### Structured-output (xgrammar)
- `crates/xgrammar-sys` Rust safe wrapper over upstream `mlc-ai/xgrammar` v0.1.34, Phase 1 FFI scaffold (codex's #26).
### Metal
- Qwen3.5-0.8B MLX 4bit single-request step-driver: 305.5 tok/s mean / 304.7 p50 on M4 Pro 20c for `1024/256`.

> Older releases (0.1.x — pre-rewrite): see [CHANGELOG-history.md](CHANGELOG-history.md)
### 2026-08-04 — default flip: DSpark train sidecar `learning_rate` 1e-4 → 1e-3
  - See [errors/2026-08-03-dspark-online-sidecar-degrades-regardless-of-loss.md](docs/experience/errors/2026-08-03-dspark-online-sidecar-degrades-regardless-of-loss.md)
### 2026-08-04 — removed: the DSpark online train sidecar
  - See [errors/2026-08-04-dspark-bias-floor-model-was-wrong-twice.md](docs/experience/errors/2026-08-04-dspark-bias-floor-model-was-wrong-twice.md)
