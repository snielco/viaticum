# KV slot move test (2026-11-01)

**Question:** can a llama.cpp session's KV cache be saved on one GPU host, copied to a different GPU host
(different vendor generation), restored there, and continue correctly without re-processing the prompt?

**Answer: yes.** In 16 of 16 test queries the restored session produced exactly the same tokens as the source and as
a cold re-prefill. A 100k-token session moved in ~42 s on 1 GbE, against a 530 s re-prefill on the destination.

## Setup

| | host `strix` | host `observer` |
|---|---|---|
| GPU | AMD Radeon 8060S iGPU (Ryzen AI MAX+ 395, gfx1151), 128 GB unified memory | AMD Radeon AI PRO R9700, 32 GB (gfx1201) |
| Backend | llama.cpp b11960, Vulkan | llama.cpp b11960, Vulkan (same binary sha256) |
| Serving | llama-swap v248 | llama-swap v248 |
| Network | 10 GbE | 1 GbE |

A third machine (the router host, 1 GbE) relayed copies between the two, so copies were capped at ~110 MB/s.

- Model: Qwen3.8-27B UD-Q4_K_XL (identical GGUF sha256 on both hosts). Hybrid architecture: 16 full-attention layers,
  48 Gated-DeltaNet (recurrent) layers, plus an MTP layer.
- Server flags: q8_0 KV, `--ctx-checkpoints 8`, MTP speculative decoding (draft 3, `-ctkd/-ctvd q8_0`),
  `--slot-save-path`, and `--cache-ram 0` on the 1-slot test route so nothing could be served from the RAM prompt cache.
- Document: Python standard library sources with 3 "needle" codes at 10/50/90 %; the question asks for all three.
  Greedy decoding (temperature 0, top_k 1), 48 tokens.
- Scripts in this folder: `add_kv_routes.py` (adds/removes test routes in a llama-swap `config.yaml`),
  `kvmove_test.py` (client, one step per subcommand), `copy_slot.sh` (relay copy + sha256 on both ends).
  Raw results: `runs/*/results.jsonl`. The tokenized documents are not included; `kvmove_test.py build` regenerates them.

## Phase A: same host (strix)

| Step | Route / slot | Result |
|---|---|---|
| Prefill 29,905 tokens | `test-kv-q38` / 0 | 108.5 s (276 tok/s) |
| Save | same | 0.35 s, **1,578 MB** (KV + recurrent ≈1,216 MB, 2 context checkpoints 299 MiB, MTP draft state 63 MiB) |
| Reference query | same | 3/3 needles, prompt_n 31 (cache_n 29,905), MTP 24/30 accepted |
| Unload → reload → restore | same | restore **0.22 s** (4.5 s with model load), n_restored 29,905; log: "restored 2 context checkpoint(s)", "restored the draft state" |
| Query after restore | same | 3/3, prompt_n 31, **generated tokens identical** to the reference, MTP 24/30 |
| Erase → cold query (control) | same | prompt_n 29,936, **108.3 s**, same tokens |
| Restore into a 2-slot server | `-p2` / 1 | **FAILED: `n_stream mismatch`** |
| Restore into 2-slot **unified KV** | `-p2u` / 1 (`--kv-unified --kv-unified-per-slot 262144`) | OK, 0.22 s, 3/3, prompt_n 31, MTP 24/30 |
| Restore into f16 KV (negative control) | `-f16` / 0 | rejected: `incompatible key rotation (256 instead of 0)` → `failed to restore kv cache` |

Findings:
1. **Save/restore across a process restart works** with a hybrid model, context checkpoints and MTP draft state:
   b11960 saves and restores all three.
2. **The KV stream count has to match, unless KV is unified.** Without `--kv-unified`, `--parallel N` gives N streams,
   so a file from a 1-slot server will not load into a 2-slot server. With `--kv-unified` there is 1 stream whatever
   the slot count, and the file loads into any slot.
3. **KV type must match.** Quantized K is stored Hadamard-rotated and the file records this, so f16 ↔ q8_0 fails
   cleanly instead of loading garbage.
4. File size ≈ 52.8 KB/token at 30k tokens including checkpoints and draft state; the KV + recurrent part alone is
   ≈40 KB/token.

### Converting a slot file between KV types (desk research, not built)
The rotation is a Walsh-Hadamard transform applied to Q/K (and V when quantized) before the cache write, only for
quantized types. An offline converter looks feasible (dequantize, undo/apply the rotation per head, requantize, fix
the header fields; only the attention layers need it, the recurrent checkpoints are type-independent), but the result
would be approximate, the on-disk format is not versioned for these fields, and no upstream tool exists.
**Decision: don't build it.** Pin one KV type per route and refuse moves between hosts that differ.

## Phase B: cross-host, observer ↔ strix

All copies sha256-matched on both ends (read back with `O_DIRECT`).

| Move | Tokens | File | Prefill on source | Save | Copy | Restore | Re-prefill on destination | Answer |
|---|---|---|---|---|---|---|---|---|
| observer → strix | 29,905 | 1.58 GB | 34.9 s | 1.4 s | 14.2 s | 0.24 s | 108 s | identical, prompt_n 31 |
| observer → strix | 100,912 | 4.21 GB | 163 s | 3.7 s | 37.0 s | 0.62 s | **530 s** (measured) | identical, prompt_n 31 |
| strix → observer | 29,905 | 1.58 GB | 108 s | 0.35 s | 14.5 s | 1.5 s | (35 s) | identical, prompt_n 31 |
| strix → observer, follow-up turn | 100,990 (doc + Q + answer) | 4.21 GB | (530 s) | 0.94 s | 37.9 s | 4.4 s | (163 s) | identical, prompt_n 18 |
| observer → strix, 2-slot unified KV server, slot 1 | 100,912 | 4.21 GB | 163 s | 3.7 s | (37 s) | 0.60 s | 530 s | identical, prompt_n 31 |

- **Every query in both phases produced the same generated tokens**, at 30k and 100k, same-host, cross-host both ways,
  the multi-slot shape and the cold re-prefill controls. MTP acceptance was identical too (24/30, 32/40, 33/39).
  The two hosts' save files differ in 68 % of their bytes (GPU float noise), yet continue identically.
- Every restore logged "restored 2 context checkpoint(s)" and "restored the draft state" (63 MiB at 30k, 212 MiB at 100k).
- Restores on observer are slower (1.5 s / 4.4 s): an older NVMe drive (Samsung 960 EVO) and no Resizable BAR. Still small.
- File size: 52.8 KB/token at 30k and 41.7 KB/token at 100k (checkpoints and draft are a smaller share as a session grows).
- Copies ran at ~110 MB/s, the 1 GbE limit of the relay host (`runs/20261101-B/copies.txt`).
- A save leaves the file in page cache, and under systemd that is charged to the llama-swap service's cgroup
  (21.5 GB at one point, over a `MemoryHigh=20G` limit). `sync` + `posix_fadvise(DONTNEED)` on the file brings it back
  down. A parking feature must do the same, or write with `O_DIRECT`.

## Verdict
Cross-host KV moves are real with this stack (llama.cpp b11960, Qwen3.8-27B hybrid + MTP, Vulkan on gfx1201 ↔ gfx1151).
A 100k-token session moves in ~42 s on 1 GbE (save + copy + restore) instead of a ~9 min re-prefill: about 13x faster,
with bit-identical continuation in these tests. On 10 GbE the copy would be ~4 s.

Rules a router must enforce before a move:
1. Same GGUF (check sha256: **restore does not check the model identity**; a same-shaped model loads silently).
2. Same llama.cpp state format: same build on both hosts (cross-build restore not tested).
3. Same KV types (`-ctk/-ctv` and `-ctkd/-ctvd`), otherwise restore is rejected.
4. Same KV stream count: the same `--parallel` without unified KV, or `--kv-unified` everywhere.
5. Enough context in the destination slot for the restored tokens.
6. Restore only into an idle slot; continue a restored session by extending it (a follow-up turn), not by branching
   inside it.

Not tested: restore across llama.cpp builds; other models (e.g. MoE, pure-transformer); full 256k sessions; CUDA ↔ Vulkan.
