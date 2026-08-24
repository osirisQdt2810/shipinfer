# Operator Requests

A running log of every request the operator has made on this project, kept so that
a rule stated once in passing is not lost to a compaction three hours later.

> **Language note.** The scaffolding here is English, per the project rule that all
> documentation is English. The requests themselves are quoted in the language they
> were written in — mostly Vietnamese. A translated quotation is no longer a
> quotation, and the whole value of this file is that it preserves the exact words.

## The rule this file exists to serve

*Added 23 Aug 2026, at the operator's instruction:*

> `rule: ghi lại toàn bộ prompt yêu cầu của tôi vào docs/qa/user.md nhé`

Every incoming request is appended to Section 1 **verbatim**, at the time it arrives —
not summarised, not translated, not tidied. Same rule as `.claude/CLAUDE.md` states.

## Provenance, and where it runs out

Section 1 was reconstructed mechanically from the Claude Code transcript shards under
`~/.claude/projects/`. One session id spans three project directories because the
working directory was renamed twice (`shipproj` -> `phucnp/shipinfer` -> `shipinfer`),
and each rename opened a new shard of the same conversation; the generator reads all
of them, orders by timestamp, and de-duplicates the overlap.

**The transcript is not complete, and this is the important caveat.** A message the
operator sends *while a turn is already running* is delivered to the assistant inside
that turn, but is not written back as a `user` record. Those requests — a large
fraction of the total, since most of this project's rules arrived as mid-work `note:`
interjections — survive only as assistant-written paraphrases in compaction summaries.

So the file has two sections and they are deliberately not merged:

| Section | Source | Fidelity |
|---|---|---|
| 1 | transcript shards | **verbatim** — exact bytes the operator typed |
| 2 | compaction summaries | **reconstructed** — the assistant's paraphrase, *not* a quotation |

Section 2 is recorded because losing a rule is worse than recording it imprecisely.
Where its wording matters, Section 1 is the authority; where a rule appears only in
Section 2, treat it as real but check the intent before leaning on the exact phrasing.

---

## 1. Verbatim requests

Recovered from the transcript: **31 requests** (V24-V31 arrived mid-turn and are
recorded here directly, since the transcript will not keep them).

### 22 Aug 2026

#### V1 — 18:24 UTC

> /goal tôi muốn thiết kế 1 hệ thống mô phỏng triton infference server để dễ dàng custom cho bài toán của tôi: /home/dungha15/workspaces/phucnp/shipproj/references/bitbucket-subfaceid/docs
> 1. Cần reformat lại .claude, pyproj, precommit... để phù hợp lại với bài toán
> 2. Tối ưu và áp dụng cuda + các thư viện khác - đã đặt sẵn trong reference: + thêm https://github.com/ShipControlPrj/bitbucket-countingservice (thay http->ssh nhé), https://github.com/ShipControlPrj/gitea-generic-multi-object-tracking-cpp (check nếu có thư viện gì còn thiếu thì tìm trong: https://github.com/ShipControlPrj)
> 3. Đảm bảo OOP ... gộp lại các thứ và code chuyên nghiệp như https://github.com/roboflow/trackers (https://github.com/roboflow/inference).
> Note: nhớ áp dụng ponytail, hạn chế code không hiệu quả

### 23 Aug 2026

#### V2 — 01:35 UTC

> gh chưa đăng nhập trên máy này nên tôi không mở được PR và không đổi được tên repo trên GitHub. Branch đã push. Sau khi gh auth login:
>
>
> bạn cần tôi làm gì, cách đơn giản nhất

#### V3 — 02:00 UTC

> tôi đã gh auth login xong rồi

#### V4 — 02:12 UTC

> native/ mô tả các library sẽ được dùng cho nên tôi đề xuất để nó vào 3rdparty/ 1 repo riêng.
> Bởi vì sau này khi bạn làm tiếp thì sẽ có:
> - mtmc
> - mot
> - reid
> ....
> mỗi module nên là 1 repo => code OOP các thứ mượt mà như kiểu roboflow thì hơn.
> Vậy đầu tiên bạn hãy tự tạo các repo submodule, bắt đầu bằng tiền tố shipinfer, ví dụ shipinfer-mot... (bạn có tự tạo git remote rồi link set remote + đặt auto-merge + claude review riêng được không? - vì kiểu tôi mong muốn mỗi 1 repo có 1 reviewer riêng chuyên cho repo đấy - như 1 senior computer vision thì nên chuyên cho CV)

#### V5 — 02:28 UTC

> tôi nhớ có 1 lịch sử hội thoại tại workspace này nhưng mà lúc tôi nhìn vào bây giờ lại không có?

#### V6 — 02:29 UTC

> /goal tiếp tục cho tới khi xong toàn bộ system

#### V7 — 03:07 UTC

> bạn không sửa à? tại sao chúng ta đang chạy trên server, tôi thử tắt đi nhưng mà vẫn không chạy, tôi nên làm cách gì là tốt nhất

#### V8 — 03:30 UTC

> tiếp tục cho tới khi xong system, tôi đã tải vllm tại references/vllm, cũng như bạn có thể clone các repo liên quan tới proj của chúng ta mà ta có thể áp dụng được techniques xịn xò giúp tăng performance lên trong https://github.com/triton-inference-server (ví dụ: https://github.com/triton-inference-server/server). Sau đó, tôi muốn bạn thử so sánh perf của system của chúng ta với 1 system dev đơn giản khác: references/counting-simulation) trên cấu hình ta đã test (cấu hình stress test)

#### V9 — 03:50 UTC

> hạn chế dùng gil trong library -> bản chất library là biến đổi và chạy thuật toán - có thể là cpu hoặc gpu cuda hip, không phải là chỗ ta nên đặt gil vào - và cũng nên tránh việc sử dụng gil => rất khó kiểm soát

#### V10 — 03:51 UTC

> note: là 1 optimized computerrvission inference performance, mọi thứ bạn code đều phải quan sát tỉ mỉ

#### V11 — 10:03 UTC

> tiếp tục cho tới khi xong

#### V12 — 13:02 UTC

> tiếp tục

#### V13 — 13:17 UTC

> hệ thống này có cách profiling không? (ví dụ vllm có cách profiling để biết ops nào đang takes time)

#### V14 — 13:30 UTC

> note phần profiling hãy mô phỏng theo triton inference server. Mọi thứ cần tuân theo như triton inference server, vì đó mới là framework engine để chạy computer vision engine. vLLM chỉ là cái để ta tham khảo thôi.
> /goal sau khi dev xong tôi cần bạn check lại toàn bộ system, những feature nào bị rối, bị thừa và not-performance thì hãy review lại
> Ngoài ra, tôi muốn hỏi về docker, note cực kì quan trọng: ta bắt buộc phải chạy trong docker, vậy bây giờ nên làm như nào

#### V15 — 14:27 UTC

> tôi nhắc lại rule cho bạn nhé: Mọi test/benchmark và everything phải chạy trên data thật (có thể là data ta gen cũng được - nhưng không phải là random), model+weight thật - không mock, không fake, sau khi xong thì xoá hết mọi mock và fake đi

#### V16 — 14:41 UTC

> viết cho tôi 1 script ngắn luôn capture VRAM của 8 card GPU trên hệ thống này tỏng thời gian 0.5s mỗi lần

#### V17 — 14:43 UTC

> giair thichs gias trị được log ra 14:43:35.280,49,15,15,15,15,15,15,15
> 14:43:35.898,49,15,15,15,15,15,15,15

#### V18 — 14:52 UTC

> ơ sesssion tại workspace dir này của tôi đâu, tại session đó tôi đã bảo thay thế về path này nhưng mà cần phải đồng bộ session claude mà, bây giờ lại không thấy đâu

#### V19 — 14:58 UTC

> tiếp tục cho tới khi xon
> Đồng thời check thêm 1 số yêu cầu
> toàn bộ system của ta cần chạy full pipeline như counting-simulation + thêm các tracker (có thể là các module khác nữa trong pipeline của ta) nhé. Ngoài ra, tôi vẫn cần bạn port sang C++ 2 tracker nhé (code port đã có sẵn rồi).
> Ngoài ra, folder benchmarks/ tôi cần bạn thêm benchmarks/baseline (chính là counting simulation), trong benchmarks thì có script chạy bench và compare test nhé.
> Ngoài ra, bạn có thể tiếp tục thực hiện các task tới khi xong, các PR sẽ cần merge PR bởi claude-review, tuy nhiên hiện tại OAUTH_TOKEN của claude đang hết tokens và cần đợi thêm 15 phút nữa - bao giờ xong tôi sẽ báo.
> (rule của bạn nhớ check về quá trình ci nhé: bạn push PR lên, check liên tục claude review+test đến đâu rồi, nếu claude blocking => bạn phải fix cho tới khi xong => khi xong tất cả thì sẽ tự động auto-merge)

#### V20 — 15:25 UTC

> note thêm: nếu engine weight là onnx thì server của chúng ta cho phép cơ chế tự động build engine file nhé (ví dụ như lib về detector trong references

#### V21 — 15:28 UTC

> tiếp tục và xóa thư mục ~/workspaces/shipvision (ta đã đặt trong 3rdparty đây rồi)

#### V22 — 15:56 UTC

> Bạn hỏi "bạn đã set RULE chưa" — câu trả lời trung thực là chưa đầy đủ. Cụ thể những gì đã chạy trên host, không phải container:
> => có phải sửa gì trong .claude để bạn luôn phải tuân theo: luôn chạy trong container không?

#### V23 — 16:02 UTC

> rule: ghi lại toàn bộ prompt yêu cầu của tôi vào docs/qa/user.md nhé (sau khi ghi xong rule thì viết tất cả prompt trước giờ tôi yêu cầu vào user.md cho tôi)


#### V24 — 16:08 UTC

> 1 rule quan trọng:: khi bạn đã xong task và không cần đến gpu nữa thì phải tắt đi nhé, tránh trường vram bị leak, tiến trình còn đó khiến cho người khác không dùng được gpu

#### V25 — 16:11 UTC

> tôi muốn bạn ghi nhớ những rule mà trước giờ tôi nói với bạn (ví dụ bạn lưu ở đâu đó trog .claude giúp bạn nhớ)

#### V26 — 16:24 UTC

> sau khi xong tất cả task thì kiểm tra và thực thi docs/qa/triton.md

#### V27 — 16:52 UTC

> hình như pipeline mot và mtmc chưa hề xuất hiện trong graph.
> Bạn có thể nêu lại cho tôi cách bạn hiểu về system của ta đang muốn làm gì trong /home/dungha15/workspaces/shipinfer/references/bitbucket-subfaceid/docs được không?

#### V28 — 17:34 UTC

> 1 note cho tối ưu performance của bạn sau này: tôi để ý thấy bạn đang dùng rất nhiều std:::memcpy, tại sao ta không đọc trực tiếp hay là in-place mà phải dùng memcpy? câu hỏi này hãy suy ngẫm và đặt vào sau khi bạn hoàn thành system nhé

#### V29 — 17:45 UTC

> tiếp tục

#### V30 — 18:05 UTC

> /goal làm cho tới khi tất cả task xong + đẩy PR và đợi review + fix review đi (hiện tại tôi chưa thấy bạn đẩy PR nào - note: bạn được tự phép quyết định từ thời điểm này, không cần phải hỏi gì tôi nữa. Cứ làm cho khi xong hết đi và đảm bảo được yêu cầu của tôi

#### V31 — 18:12 UTC

> note: sau khi xong, verify lạ những gì tôi yêu cầu trong docs/qa/user.md bạn đã hoàn thành đúng hết chưa

### 24 Aug 2026

#### V32 — 00:20 UTC

> có 1 vấn đề về ci như sau:
> - Bạn push PR, bạn đợi Claude review remote PR, trong thời gian đó bạn có thể làm việc khác (vì claude review là khá lâu)
> - claude review nếu approve => tự động merge không nói làm gì nữa
> - nhưng nếu claude review blocking, bạn tìm xem liệu có đúng như claude review finding ra bug như vậy không, nếu đúng thì sửa, nếu không đúng - claude review có vẻ bị sai thì bạn comment lại trên PR để claude review xem xét review lại (tôi không rõ để trigger review lại thì hình như phải thêm lại auto-merge - tôi không rõ bạn check nhé).
> Toàn bộ quá trình trên looping cho tới khi PR được merge xong

#### V33 — 00:22 UTC

> ngoài ra như đã nói trước đó, hạn chế trong 1 PR có quá nhiều commit và file changes, ví dụ như PR #3 hiện tại đã có tận 100 commits => bây giờ đã lỡ rồi thì oke nhưng mà lần sau bạn hãy chú ý để không xảy ra hiện tượng như này nữa

#### V34 — 00:52 UTC

> note: nếu vấn đề của ta gây performance thấp đang là python multithreading gil thì bạn cóthể portable sang 1 bảng c++ ơ csrc/shipper/

#### V35 — 03:58 UTC

> tiếp tục

#### V36 — 04:35 UTC

> tiêp tục

#### V37 — 11:55 UTC

> vậy tóm lại bạn còn gì chưa làm nữa? bạn không đọc goal của tôi à?

#### V38 — 12:05 UTC

> /goal tôi suggest bạn nên hạn chế dùng GIL, bạn có thể thử port toàn bộ hệ thống ssang C++ trước và sau đó đo thử performance của hệ thống trên C++. Sau đó tiếp tục thực hiện toàn bộ task như tôi đã nêu. Note: BẠN PHẢI LÀM CHO XONG, không được dở chừng và bắt tôi phải kêu bạn tiếp tục

---

## 2. Reconstructed requests

**These are not quotations.** Each item below is the assistant's own paraphrase, taken
from a compaction summary, of a request whose original text the transcript did not keep.
Fragments that a summary preserved inside quote marks are shown in `code style`; the rest
is paraphrase. Ordered as they arrived.

### Phase 1 — bootstrapping the design (22 Aug)

- **R1.** Point the git remote at `github.com/osirisQdt2810/shipproj` — over SSH, not HTTPS.
- **R2.** `note: bạn tự quyết định design, khôg cần hỏi tôi` — decide the design without asking.
- **R3.** `note: tạo PR nhớ tuân theo workflow + automerrge`
- **R4.** Split `policy.py` into a `policy/` package with `@register` and a registry. Don't
  lump everything extensible into one file.
- **R5.** Do the same to `src/shipinfer/core` — split into packages; logging gets its own
  package, with async logging.
- **R6.** `note: đây là 1 performance system => chắc chắn sẽ phải có c++/cuda/hip... cần
  integrate sao cho phù hợp`
- **R7.** vLLM and sglang carry many optimisation techniques (cudagraph-scheduling and
  others). This is an OPTIMIZED INFERENCE SYSTEM — the high-level modules need pybind down
  to C++.
- **R8.** `hmm, tôi cảm thấy bạn đang tự code lại từ đầu? ví dụ graph? tại sao bạn không
  dùng torch nhỉ?` — vLLM never re-invents these primitives. (This one caused a full
  rewrite of `runtime/`.)
- **R9.** Keep what was already written as `CustomXXX` variants — a way for the operator to
  read and understand the native library's flow.
- **R10.** `note: như tôi đã nhắc trong ponytail, những lib basic, powerfull thì cứ dùng vì
  nó đã cực kì tối ưu`
- **R11.** `nhớ: memory ngay trong project này, bởi vì sau này tôi sẽ có thể chuyển qua dev
  trên các device khác` — memory lives in the project, not the machine.
- **R12.** Only large feature commits carry the Claude co-author trailer.
- **R13.** `CLAUDE_CODE_OAUTH_TOKEN` is set, so PRs get a Claude review — then auto-fix and
  finish the task.
- **R14.** `note: việc chia package không phải chỉ ở mỗi những folder tôi nói thôi nhé, bạn
  cần design thật sạch để dễ dàng tái sử dụng và code OOP clean`
- **R15.** `note: bạn phải dùng ssh thay vì http nhé (tôi đang để public)`
- **R16.** Rename `shipproj` to a nicer two-or-three-word name — the folder and the remote —
  but keep this session alive through the move.
- **R17.** `note lớn: follow triton inference server và vllm để thiết kế sao cho hệ thống
  tối ưu nhất`

### Phase 2 — Docker, C++ conventions, and the move to tmux (23 Aug, early)

- **R18.** `note: important ! mọi thứ đều phải chạy trong docker nhé` — plus a `deploy/`
  folder for deployment; never run straight on the host.
- **R19.** C++ convention: indent inside `namespace`, and access modifiers get their own
  indent level.
- **R20.** `shipinfer-native => không phải là native, đây rõ ràng là 1 function khác` —
  `native` is an implementation word, not a function name. (Became `imgproc`.)
- **R21.** `ngoài ra con trỏ thì là float *dst => float* dst`
- **R22.** Ran the goal loop in the VS Code extension, closed VS Code expecting it to keep
  going, and found nothing had happened 45 minutes later.
- **R23.** `tôi nghĩ là tôi sẽ dùng tmux + claude-cli, như vậy có ổn không?`
- **R24.** `oke vậy giờ tôi sẽ chạy claude cli, tôi cần phải làm gì`
- **R25.** `không ổn, tôi không dùng claude cli được vì:` — the CLI demanded a fresh login.
- **R26.** `vấn đề ở đây là tôi không thể dùng claude và login lại vì tôi đã quên mật khẩu`
- **R27.** `à được rồi, hãy dừng đi phiên của bạn hiện tại, tôi sẽ bảo claude cli tiếp tục`

### Phase 3 — the two-repository split and shipvision (23 Aug, midday)

- **R28.** `/parallel-tasks` — merge all the CV work into one repository, refactored like
  `roboflow/trackers`; it may hold C/C++/CUDA/Python but grouped properly, because a shared
  library gets built from it. shipinfer calls it the way vLLM calls aiter. Reuse
  mot/mtmc/detector/reid from `references` (clone from ShipProj if absent), with OOP
  inheritance from an abstract base; add Optuna tuning like roboflow; list the TODOs,
  because the system is only slightly complete; use parallel distributed execution.
  Distinguish the two repositories clearly: **shipinfer** is infrastructure, reading up to
  50 real cameras or videos over **RTSP/GStreamer**, emitting output as the doc requires;
  the **3rdparty library** is algorithms, and needs proper packaging/module/OOP. Ask if
  anything is unclear.
- **R29.** Delete `.claude/commands/daily-wrap.md`, `resume.md` and `adr.md` — `tôi thấy các
  commands này thật là thừa`.
- **R30.** For testing, fetch videos from the internet with ships and **more than 10 people**,
  for stress testing.
- **R31.** `1 note quan trọng không kém, bây giờ khi bạn push lên remote github, thì user
  contribute là gì?`
- **R32.** `├── python/shipvision/ => không có python/, shipvision/shipvision luôn`
- **R33.** Answers to a round of questions: the name is **shipvision**; do parallel
  C++/CUDA/HIP→pybind **and** Python, the way aiter does, because some reference repos
  already have fast Python that needn't be redefined in C++; merge all four module repos
  and **delete** the old ones; use a GStreamer pipeline (see subfaceid) but implement
  **both** stacks, switchable via `src/shipinfer/envs.py`. Plus:
  `Rule: những thư viện cơ bản - high performance như torch, gstreamer... đều đã được
  implement cực kì tốt rồi, tôi không muốn bạn cứ code lại from scratch những cái đó,
  performance rất tệ - sử dụng bao nhiêu thư viện cũng được, miễn là nó high performance.`
- **R34.** Don't touch boxmot (AGPL) — rewrite clean-room; run both branches in parallel.
- **R35.** `note: code cần để ý packaging phù hợp nhé, không nên có chuyện 1 folder chứa cả
  chục file như vậy (ví dụ ingest hiện tại)`
- **R36.** `đã merge xong PR #2`
- **R37.** `toàn bộ test_ dùng theo hướng class, không dùng trong function`
- **R38.** Attribution is `osirisQdt2810` (the repo owner); compose test videos with ground
  truth, `nhưng mà số lượng người chỉ nên <= 20 người thôi`.
- **R39.** Split the workflow change into its own PR.

### Phase 4 — the benchmark mandate (23 Aug, afternoon)

- **R40.** `/goal` — self-test and compare performance against `references/counting_simulation`
  on 50 cameras (video replayed) at FPS=20 on **4 GPUs**. The task is not done until that
  comparison exists; then optimise kernel/system and hunt bottlenecks until **≥5×** better.
  Also save to `docs/qa`: does Triton have a CUDA-graph mechanism and how does it use it,
  does our server apply it the same way; how does our system differ from TIS, which TIS
  features to adopt and which of ours to remove as less performant; what language is TIS
  written in. `Note: 1 số model nếu chưa có weight bạn có thể tìm weight trên mạng... vì
  counting-simulation có thể chạy được - tôi expect là code của chúng ta cũng có thể chạy được.`
- **R41.** Change `~/workspaces/phucnp` to `~/workspaces/` — hide every `phucnp` on this
  account — minding session paths and link paths. And: `tôi cảm giác bạn đang cố chạy mọi
  thứ trên cpu. perf của ta sẽ là chạy chủ yếu trên gpu, ngoài ra, tôi cũng chưa rõ ràng
  được bạn đang chạy trên docker nào nữa. Bạn có thể dừng 1 nhịp và note ra cho tôi trong
  container.md không?`
- **R42.** `rule: - code thật, không sử dụng test fake - mọi test đều chạy trong container`
- **R43.** `compare ở đây không phải là compare đơn giản từng module, tôi muốn bạn compare
  throughput image/s của cả 2 hệ thống` — counting-simulation uses an enqueue/dequeue
  mechanism per model module, so if buffer size keeps growing the system is not keeping up.
  Also: an `envs.py` variable to switch CUDA-graph mode on and off, and answer in `docs/qa`
  whether Triton has the same switch.
- **R44.** `ngoài ra: shipvision là 3rdparty của shipinfer, không phải là 1 module riêng
  nhé` — delete the current `3rdparty` entries (shipinfer-mot, the image-ops one). shipinfer
  is based on shipvision's kernels and algorithms, so to benchmark, compare or test
  anything, the run must go through the whole stack: `system -> algo -> kernel`.
- **R45.** `note: define cho toàn hệ thống 1 cái metric system nhé` — vLLM has TTFT, TPOT,
  TPS as an illustration, but the real question is what metrics **Triton Inference Server**
  uses, and how to apply those here. (Stated twice.)
- **R46.** When every feature is finished, split it into several PRs — don't merge it all
  into one repository. Claude review on the GitHub remote gets its token refreshed in about
  45 minutes. The same applies to shipvision. And the shipvision refactor isn't right yet:
  each tracker and each mctracker should be its own package, roboflow-style, inheriting
  with a `tracker.py` and a `tracklet.py`. The CUDA code today is only image-ops, which is
  fine, but it needs refactoring and optimising —
  `references/bitbucket-generic-feature-extractor-trt/src/toolsv2` defines many more ops.
  Also, the current mot and mtmc services use a C++ shared library, not Python.
  `Note: everything should be packaging for easily oop+refactor`
- **R47.** `tôi đã thêm CLAUDE_CODE_OAUTH_TOKEN và CLAUDE_CODE_OAUTH_TOKEN cũng đã được
  reset nhé. Bây giờ không còn vấn để về CLAUDE review nữa, bạn hãy tiến hành kiểm tra lại
  PR và push lại nhé`
- **R48.** `rule: những gì thừa thì xoá đi nhé`
- **R49.** `note: nếu bạn đang dùng benchmarks/compare_baseline.py để compare thì bạn cần
  review kĩ script này phải đúng nhé, tôi cần throughput cần >= x5 times so với baseline.
  System cần chạy chính xác trong container + chạy GPUs, chứ không phải là mock/fake test
  trên CPU.` The reason for 5×: counting-simulation preprocesses on the CPU with `cv2.resize`,
  which is its main bottleneck, while we have pushed every op onto the GPU and overlap
  CPU with GPU (batching against compute) — so there is no reason throughput cannot rise
  by 5× or more.
- **R50.** `sau tất cả thì trong readme nêu ngắn gọn cách chạy system trên bộ video test và
  cách benchmark compare giữa 2 counting-simulation và system của chúng ta nhé`
- **R51.** `toàn bộ phần model weight hãy đặt vào models/ nhé`
- **R52.** `rule: no-fake, no-mock, verify-by-logging`
- **R53.** `như đã nói, hãy clean lại toàn bộ system/repo sau khi dev xong - không mock, test
  thật, hạn chế code bị thừa và invalid, những đoạn code có vẻ low-performance`
- **R54.** `xoá hết mọi thứ fake và mock trong toàn thể system đi nhé - không dùng fake và mock`
- **R55.** `có 1 vấn đề tôi nghĩ là quan trọng như sau: - thực sự hiện tại video có đang được
  đọc bằng rtsp không?` — this is mandatory when testing and benchmarking: is it in the same
  format as subfaceid, the repo closest to reading from a real camera, which also has fake
  video and produces the image on the GPU in an NV12/YUV 4:2:x format? Each model gets its
  own resize and its own crop, possibly from the original image — not
  `crop ra vùng person 512x512 -> sau đó resize tàu 640x640`. Two different objects and two
  different tasks that need the original image must each be handled separately.
- **R56.** `bạn có đảm bảo bạn đang chạy trong môi trường container và chạy GPU không vậy =>
  bạn đã set RULE chưa? làm sao để tôi có thể check được đúng là bạn chạy trong container và
  có chạy GPU?` — (1) always run inside a tmux window; (2) `~/workspaces/tools/vram_log.sh`
  is already recording GPU VRAM continuously — is there a way to prove container and GPU
  execution against it?

---

## 2b. Verification

Every rule in the index below is checked against the repository — a file, a command's output
or a git fact — in **`docs/qa/verification.md`**, written at the operator's request on
24 Aug 2026. Twenty rules held; four are partial and one target was measured and missed, each
named there rather than argued around.

---

## 3. Standing rules index

The rules that do not expire, each pointing at where it was stated. `V` = verbatim
(Section 1), `R` = reconstructed (Section 2).

| Rule | Where |
|---|---|
| Everything runs in Docker — never on the host | R18, R42, V14 |
| `no-fake, no-mock, verify-by-logging` — real data, real weights; delete every mock when done | R52, R54, V15 |
| Delete whatever is redundant, invalid, or low-performance | R48, R53 |
| Every test class-based, never bare module-level functions | R37 |
| Ponytail: reuse torch/GStreamer/scipy; never reimplement a fast library | R10, R33 |
| Package everything for OOP + refactor; no folder with a dozen flat files | R14, R35, R46 |
| All documentation in English (this file's quotations excepted) | project rule |
| All git remotes over SSH, never HTTPS | R1, R15 |
| Co-author trailer only on large feature commits | R12 |
| Branch + PR for anything non-trivial; poll CI until the review stops blocking | R3, V19 |
| Profiling and metrics follow Triton, not vLLM | V14, R45 |
| shipvision is a 3rdparty submodule of shipinfer, never the reverse | R44 |
| Benchmarks run the whole stack: system → algo → kernel | R44 |
| Throughput must reach ≥5× counting-simulation, measured by buffer-growth saturation | R40, R43, R49 |
| RTSP ingest is mandatory for test and benchmark, in subfaceid's GPU NV12/YUV form | R55 |
| Each model gets its own resize + crop from the original image | R55 |
| ONNX weights: the server auto-builds the engine | V20 |
| Model weights all under `models/` | R51 |
| Always run in a tmux window; prove container + GPU against `vram_log.sh` | R56 |
| Release the GPU as soon as the task ends — shared box, VRAM is watched | V24 |
| Keep the rules in `.claude/memory/` so they survive a session | V25 |
| Record every operator request here, verbatim | V23 |
| Decide autonomously from 23 Aug 18:05 on; do not ask, just finish | V30 |
| At the end, verify every request in this file was actually done | V31 |
| PR loop: push, work elsewhere while review runs; a blocking finding is checked before it is trusted — fix it if real, comment back if the review is wrong; loop until merged | V32 |
| Keep a PR small — few commits, few files changed. PR #3's ~100 commits is the counter-example | V33, R58 |
| If the GIL is what caps throughput, port the hot plane to C++ under `csrc/` in this repository | V34 |
| Port the system to C++ under `csrc/` and measure it there, then resume the remaining tasks. Finish without being told to continue | V38 |
| After all tasks: check and carry out `docs/qa/triton.md` | V26 |
| Deferred: justify or remove every `std::memcpy` — prefer zero-copy/in-place | V28 |
