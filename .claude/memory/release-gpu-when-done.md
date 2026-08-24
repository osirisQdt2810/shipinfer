# Release the GPU the moment the task no longer needs it

Operator, 23 Aug 2026: `khi bạn đã xong task và không cần đến gpu nữa thì phải tắt đi nhé,
tránh trường vram bị leak, tiến trình còn đó khiến cho người khác không dùng được gpu`.

**Why:** this box is shared. A finished-but-alive process keeps its CUDA context — roughly
220–480 MiB per GPU even when idle — and on a 24 GiB card a handful of those starve the next
person. The operator watches VRAM continuously via `~/workspaces/tools/vram_log.sh`, so a
leak is visible to them whether or not it is mentioned.

**How to apply:**
- Bound every GPU run up front: `timeout <n>` on the command, and a `finally` that drops
  tensors and calls `torch.cuda.empty_cache()`. A crash must not be the thing that frees it.
- Prefer `docker run --rm` so the container cannot outlive the command.
- After any GPU task, verify rather than assume:
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
  Empty output is the goal. `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader`
  should be back near its idle floor (~15–20 MiB here).
- Distinguish a leak from live work before killing anything: a background agent's probe
  inside its own `timeout` is working, not leaking. Check elapsed time and parent first.
- Relates to [[run-in-container]]: a container that exits takes its contexts with it, which
  is the cheapest version of this discipline.
