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

#### V39 — 13:40 UTC

> làm đến đâu rồi, tại ssao tôi không thấy bạn update gì thêm nữa vậy? lý do là gì và tại ssao cứ bắt tôi phải nhắn trong khi bạn chưa hề làm xong?

#### V40 — 13:52 UTC

> ngoài ra, tôi không muốn code csrc như này, tôi muốn cấu trúc csrc giống với src luôn. ngoài ra bạn không cần phải thiết kế C++ theo hướng include 1 bbên rồi src 1 bên như vậy, bạn cứ thoải mái 2 file .h và .cpp trong cùng 1 file thôi, có vấn đề gì đâu?

#### V52 — 14:25 UTC

> tiếp tục

#### V91 — 14:40 UTC

> note: C++ chỉ cần có cách thuật toans track đã sử dụng trong motservice và mtmcservice thôi

#### V53 — 15:30 UTC

> tiếpt ục

#### V54 — 16:10 UTC

> tiếp tục

#### V55 — 16:40 UTC

> 1. đổi 3rdparty/shipvision/shipvision/tracking -> 3rdparty/shipvision/shipvision/mot
> 2. đổi 3rdparty/shipvision/shipvision/tracking/core -> trackers
> 3. có cách nào để đưa các tracker native vào luôn trong trackers không? (recommend nên như vậy - nếu không được thì thôi) - vì bản chất tôi thấy nó là algo tracker mà (làm tương tự với mtmc)
> 4. làm tương tự với mtmc (mot và mtmc gần như nhau)

#### V56 — 16:45 UTC

> ví dụ deepsorrtv2 c++ có thể đặt trong 3rdparty/shipvision/shipvision/tracking/core/deepsortv2/tracker.py?

#### V57 — 16:50 UTC

> tôi nghĩ vẫn nên để trong mot/trackers/deepsortv2/tracker.py - vì bản chất nó vẫn là 1 thuật toán track của deepsort, chỉ là implementation nó khác đi thôi

#### V58 — 17:05 UTC

> tôi nhớ trong mot-service hay là mtmc-service làm gì có implement full c++ các thuật toán tracker khác ngoài deepsortv2 đâu nhỉ?

#### V59 — 17:10 UTC

> không, đã code rồi thì giữ nguyên đi

#### V60 — 17:20 UTC

> không cần xoá nhé, ngoài ra fill thêm code các trackers khác trên roboflow

#### V61 — 17:35 UTC

> đặt 3rdparty/shipvision/shipvision/mtmc/tracker.py -> 3rdparty/shipvision/shipvision/mtmc/trackers/ để ta có base và theo cơ chế registry.

#### V62 — 17:50 UTC

> thực tế cái shipvission (ví dụ mtmc) expose là tracker, mà tracker thì ta có thể dev theo kiểu nào cũng được (ví dụ native hoặc đơn giản là backend python).
> Tương tự với các matchers, cluster...
> Do đó, kiểu code refactor cần đáp ứng được tính flexible - ta có thể native 1 phần hoặc native toàn bộ code tracker
> Ngoài ra, 3rdparty/shipvision/shipvision/mtmc/identity.py theo tôi hiểu là 1 component của 1 loại tracker hoặc 1 loại tracker, nên đặt ở ngay thư mục mtmc không chuẩn lắm

#### V63 — 18:00 UTC

> ví dụ như hiện tại 3rdparty/shipvision/shipvision/mtmc/trackers/cluster/tracker.py có dùng kiểu self.lock trên python - tôi dự đoán là sẽ không hiệu quả và ảnh hưởng performance, ví dụ như trong mtmc service đã có 1 thư viện tracker native c++ riêng rồi?

#### V64 — 18:10 UTC

> không, ý của tôi là đúng là nó phải có lock, nhưng mà nếu ta xác định dùng lock thì có thể nên dùng c++ native perf sẽ tốt hơn nhiều => nên có 1 bản native tracker c++ tương ứng (hãy nhìn vào c++ trackers/ trong mtmcservice => đã implement sẵn 1 số loại tracker rồi)

#### V65 — 18:15 UTC

> 3rdparty/shipvision/shipvision/mot/backends/base.py => đặt là 3rdparty/shipvision/shipvision/mot/backends/native.py (tương tự với mtmc)

#### V66 — 18:20 UTC

> 3rdparty/shipvision/csrc/shipvision/mtmc/core đây là 3rdparty/shipvision/csrc/shipvision/mtmc/matchers chứ không phải core

#### V67 — 18:30 UTC

> cách design chưa chuẩn: 3rdparty/shipvision/csrc/bindings/module.cpp
> Phần bindings chỉ nên là define binding thôi, không được có các function như là run_nms, run_crop...
> ví dụ ta chỉ binding ops nms, nv12_letterbox... thôi hoặc là binding tracker... chứ không được viết nguyên code trong các file bindings như này

#### V68 — 18:40 UTC

> tại sao lại không có mạng, tôi thử: git clone --recursive https://github.com/roboflow/tracker vẫn bình thường (trong references)

#### V69 — 18:45 UTC

> không còn tracking mà phân biệt rõ mot và mtmc

#### V70 — 18:55 UTC

> rule: khi bạn thực hiện quá nhiều feature mà chưa commit, bạn phải tách nhỏ feature ra thành các PR riêng biệt để đảm bảo reviewer không bị quá tải

#### V71 — 19:05 UTC

> note: tôi chưa hiểu tại sao cứ phải đặt py::gil_scoped_release release; rất gây down performance. nên đặt chỗ nào hợp lí hơn. Ví dụ nếu ta đã đặt ngoài tracker rồi thì thôi, (vì nếu tracker thì cứ đặt lock sau mỗi step track), còn ví dụ các function mtmc_threshold_similarity là các function con trong mỗi track, nó không thể bị lỗi race condition được. => Bạn review xem có đúng không?
> Tức là: đã đặt lock tránh condition thì nên đặt tại mỗi bước trong track()

#### V72 — 19:20 UTC

> bạn hãy nghĩ đơn giản như này: shipvission chỉ là thư viện về thuật toán, bao gồm c++ và binding của nó; hoặc là native python sẵn.
> Vậy thì:
> - ta cần gil để tránh race condition khi nào? => khi mà nhiều thread cùng động vào buffer chung nào đó đúng không?
> - nếu là detection thì điều này dường như không xảy ra, vì abstract nhất của nó là .detect() thì ứ thế mà detect thôi
> - còn nếu là tracking (ví dụ: mot/mtmc) thì sau mỗi step tracker() có thể gây ra racecondition => ta đặt ngay threading.lock ngay tại đầu hàm .track() của nó là được mà?
> Nên nhớ: SHIPVISSION chỉ là thư viện ta gọi thuật toán

#### V73 — 19:30 UTC

> ủa tại sao lại cần như này: py::gil_scoped_release không phải lock — nó là điều ngược lại. Nó nhả GIL để các thread Python khác chạy được trong lúc C++ làm việc
> python trên tầng system của ta đang dùng có threading à?

#### V74 — 19:40 UTC

> thực tế, python dùng threading sẽ cực kì chậm

#### V75 — 19:45 UTC

> à nhưng mà không sao, bản chất ta có thể switch sang system C++ mà

#### V76 — 19:55 UTC

> bạn để ý là ví dụ trong vllm họ không hề dùng threading (hoặc dùng rất ít), những phần core wrap thuật toán để launching họ hầu hết dùng multi-process - dùng threading trong python cực kì down performance

#### V77 — 20:05 UTC

> oke tiếp tục, làm tiếp khi xong

#### V78 — 20:40 UTC

> note: bạn đẩy PR cũng phải theo thứ tự nhé (vì claude review có thể blocking và bạn cần sửa lại) - do đó không được puhs 1 lần nhiều PR

#### V79 — 20:55 UTC

> tôi chưa thích cách bạn binding, theo tôi hiểu nó gồm các bước:
> 1. pybind
> 2. transform type của pybind -> c++
> 3. thực hiện thuật toán
> Không nên có gil gì ở đây cả.
> Vậy thì tôi muốn 1.pybind chỉ nằm trong bindings/
> còn từ 2 và 3 thì define trong 3rdparty/shipvision/csrc/shipvision (nếu nó chằng chéo và nhiều thuật toán img_ops, tracker,...) dùng chung 1 loại dtype thì bạn có thể define chung ngoài shipvision/, còn nếu không thì trong mỗi thuật toán thì define riêng rồi transform riêng - tóm lại là tách bạch phần bindings/ và từ binding->code c++ thuật toán trogn shipvission.
> Ngoài ra, hạn chế và tôi vẫn không muốn dùng gil_release - lí do: phần shipvision chỉ define thuật toán,  tối đa là có thêm lock mutex đặt ở BaseTracker, chứ không được đặt gil gì ở đây cả - vì tôi chưa hiểu gil là tránh multithread truy cập, như vậy nếu đặt gil ta phải chặn từ tầng python chứ không đặt ở tầng algorithm


### 25 Aug 2026

#### V80 — 11:30 UTC

> tôi đã bảo rồi nhưng hình như bạn đã quên: https://github.com/osirisQdt2810/shipvision/pull/2/changes hãy tách nhỏ pr ra, pr quá to rồi (ghì mà tận 45 commit, 290 file changes vậy) => nếu bạn chưa đặt rule cho yêu cầucủa tôi thì hãy đặt ngay đi nhé, lại quên context rồi


#### V81 — 13:05 UTC

> tương tự như vậy, hãy nhìn xem https://github.com/osirisQdt2810/shipinfer/pull/8/changes cũng quá to này

#### V82 — 13:40 UTC

> như nào rồi

#### V83 — 16:45 UTC

> tạm thời hãy dừng lại 1 chút và thảo luận với tôi về architecture design của hệ thống này.
> Ban đầu tôi nghĩ rằng hệ thống nên thiết kế theo hướng:
> - Giả sử có N camera, G gpus, K=N/G
> - pipeline:: mỗi gpu đọc/decode K camera/video => đưa ảnh raw nv412 qua 1 cụm scheduler/balancer (mà ta hay nghĩ nó là triton inference server) để dispatch/balânce được GPU nào đang rảnh/bận - nêus rảnh thì disspatch tới GPU đó. Vậy câu hỏi ở đây: ví dụ với reid ta cũng có nên loadbalancing không?
> - Nó cũng gần tương tự như pipeline mtmc_deepstream.py (tôi chưa confirm được pipeline đúng hay không). Khác 1 điều là pseudo pipeline trên chỉ đơn giản là detect đã được gán vào nvinferserver, đi qua preprocess rồi sau đó được dynamic dispatch là các cropped object. Còn ở đây, tôi đang phân vân ta dispatch nên là full ảnh hay là ảnh cropped thôi. tôi nghĩ nên là ảnh cropped thôi .
> - Tôi nghĩ rằng pipeline trên đang chính là pipeline mà ssystem của ta đang xây dựng.
> Tôi cần thảo luận với bạn 2 điều:
> +) Nếu đúng là vậy, tôi nghĩ rằng mình nên abstract hóa pipeline lại, mỗi thành phần gstelement plugin có thể là 1 phần trong đồ thị của chúng ta pipeline của chúng ta. Và như vậy tức là, ta có thể song song hóa code 1 phiên bản pipeline bằng gstreamer để so sánh xem performance khi sử dụng gsstreamer như nào. Việc switch cách thực thi pipeline có thể được bật bằng 1 var trong envs.py
> +) Theo bạn, cách thiết kế design như này đã đủ tốt chưa. Và cách thiết kế system hiện tại có tuân theo design trên không? 
> Note: Ưu tiên thảo luận với tôi về vấn đề này đã nhé.

#### V84 — 17:20 UTC

> So với mtmc_deepstream.py
> Sketch đó là topology B, không phải C: mỗi nhánh GPU có nvstreammux(gpu-id) → nvinferserver (config riêng per GPU) → nvtracker → sgie → probe;
> tôi nghĩ file này là topology C chứ, vì C tôi nghĩ là cách giải quyết tốt nhất cho bài toán trên single-node này rồi (hạn chế rất nhiều được hiện tượng imbalance). Nếu ko phải, các pipeline lớn trên thế giới họ xử lý dạng bài toán CV này như thế nào?

#### V85 — 17:50 UTC

> Mục tiêu cuối cùng của tôi là C, và như vậy tôi nghĩ rằng pipeline có thể abstract hóa được theo C và hiện tại ta có 3 instance có thể sử dụng abstract pipeline này:
> +) B (chỉ cần chỉnh 1 chút là được đúng không?)
> +) C hẳn
> +) deepstream => phần sẽ cần code thêm
> Nếu bạn không còn hỏi gì thì tôi muốn bạn thêm task ở trên, và hoàn thiện tất cả các task khác

#### V86 — 18:20 UTC

> note; khi bạn không biết code như nào, hãy nhìn vào các bản code profession như triton inference sserver hoặc là vllm

#### V87 — 18:55 UTC

> tiếp tục

#### V88 — 19:40 UTC

> C++ dataplane tôi thấy đang chưa giống với python (tôi thấy khác hẳn luôn ấy - có thực sự là đã port sang chưa)

#### V89 — 19:55 UTC

> B. Port thật: C++ plane phải mirror kiến trúc Python plane: note khi ta update python thì c++ cũng phải sync theo nhé

### 26 Aug 2026

#### V90 — 01:20 UTC

> tiếptuccj

*(A typo of "tiếp tục" — continue. Logged as typed, per the rule.)*

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
| When uncommitted work has grown to several features, **split it into separate PRs** before pushing — a reviewer who is overloaded reviews nothing | V70 |
| **Push those PRs one at a time, in order.** Never open several at once: review can block, and a blocked PR has to be fixed before the next one is opened on top of it | V78 |
| **A PR is at most ~15 commits and ~25 files. Check the numbers *before* opening it, not in the body afterwards.** Past that, split by package or seam and push the pieces one at a time. Acknowledging that a PR is oversized is not the same as splitting it | V80, V70, V33 |
| **`bindings/` holds pybind declarations and nothing else.** The pybind-type → C++-type conversion and the algorithm both live under `csrc/shipvision/`: a shared dtype at the top level if several algorithm families use it, otherwise per-algorithm | V79, V69 |
| **No `gil_scoped_release` anywhere in shipvision.** It is an algorithm library; thread discipline belongs to the caller. A `std::mutex` on a stateful tracker base is the most it may hold | V79 |
| If the GIL is what caps throughput, port the hot plane to C++ under `csrc/` in this repository | V34 |
| Port the system to C++ under `csrc/` and measure it there, then resume the remaining tasks. Finish without being told to continue | V38 |
| After all tasks: check and carry out `docs/qa/triton.md` | V26 |
| Deferred: justify or remove every `std::memcpy` — prefer zero-copy/in-place | V28 |
| `csrc/` mirrors `src/`'s package layout; a thing's `.h` and `.cpp` live next to each other, no split `include/` — in this repo *and* in shipvision | V40, V50 |
| Never end a turn while work remains — enforced by a mechanism, not an intention | V41, V51 |
| `X.cpp` pairs with `X.h`, never `X.hpp` | V44 |
| C++ style: indent the body of a `namespace X {` | V45 |
| Professional CUDA/HIP convention — `GPU_CHECK(...)` and `gpu*` aliases, not a bespoke macro over an inline helper | V46 |
| Show the todo list and get it confirmed before executing a large batch | V47 |
| Reuse the Python names (script, class, function) in C++; do not invent a second vocabulary | V48 |
| Order: Plane 3 and Triton first; the ≥5× whole-system optimisation is the **final** goal | V49 |
| `shipvision`'s `tracking/` and `mtmc/` need real module packaging — one package per algorithm, in the shape of roboflow/trackers `src/trackers/core` — so adding and optimising an algorithm is clean | V50 |
| Every MOT/MTMC algorithm from the previous C++ services must exist in shipvision too, callable through the C++ ops binding rather than only in Python | V50 |
| **Reference implementations first** — when the way to build something is not clear, read how Triton Inference Server or vLLM does it before inventing — their shape is the default, and a departure from it is stated with its reason. | V86 |
| **Two planes, one architecture** — `csrc/` mirrors the Python data plane seam for seam — instance thread + queue, dispatcher + policy, batch window, fair-queue eviction order, graph, reassembly, ingest — and a change to a Python data-plane seam is not done until the C++ seam is synced and the cross-plane parity test agrees. | V88–V89 |

### Verbatim, V41–V51

- **V41** — *tiếp theo, mục quan trọng nhất: tôi muốn biết tại sao lí do gì khiến bạn không làm
  trong khi vẫn còn task chưa xong? tôi cần bạn khắc phục được điêfu đó trước*
- **V42** — *thế nào rồi*
- **V43** — *todo cho tôi những gì bạn làm, bạn đã nắm được tôi yêu cầu những gì chưa*
- **V44** — *note: 1 file X/cpp thì tươngứng của nó nên là X.h chứ không phải là X.hpp*
- **V45** — *ngoài ra như tôi đã yêu cầu trước đó, style c++ của tôi là cần indent ví dụ từ
  namespace X{ => indent mới tới code*
- **V46** — *hãy code theo như convention cuda hay code đi, ví dụ: ta dùng GPU_CHECK(...) chứ ai
  lại dùng SHIPINFER_CUDA rồi gọi cuda_check inline như vậy bao giờ - bạn hãy quan sát thêm các
  thư viện code cuda/hip professtional nữa*
- **V47** — *trước hết, bạn từ từ lại, gửi tôi list todo đã - chưa cần phải cố gắng giải quyết
  hết mọi thứ vội, tôi muốn xem bạn đang định làm gì và tất cả task đã đúng ý tôi chưa*
- **V48** — *note: bạn có thể dùng tên cũ (script, class, function...) như ở trong python cũng
  được mà, không nhất thiết cứ phải nghĩ ra tên mới*
- **V49** (24 Aug 2026, 14:20 UTC) — *bạn có thể thực hiện phase 3 + triton first, hãy đặt optimize perf của toàn system
  x5 là goal cuối cùng ta cần tối ưu*
- **V50** — *ngoài ra, như đã nói trước đó 3rdparty/shipvision/csrc => bạn không cần phải câu nệ
  đặt include và src tách ra 1 bên mà có thể gộp lại 1 chỗ là được nhé, miễn là packaging hợp lí
  để sau này refactor oop cho dễ. Ngoài ra, tôi thực sự chưa ưng lắm với cách design
  3rdparty/shipvision/shipvision/tracking và 3rdparty/shipvision/shipvision/mtmc, code chưa được
  packaging module hóa chuẩn để sau này dễ dàng thêm thuật toán và tối ưu - ví dụ phần tracking
  mot 3rdparty/shipvision/shipvision/tracking/trackers bạn có thể code như roboflow phần core
  https://github.com/roboflow/trackers/tree/develop/src/trackers/core này không => cảm giác
  refactor rất chuẩn và rất đẹp mắt, dễ phát triển sau này. Ngoài ra, tôi cũng muốn bạn triển
  khai được các thuật toán service mà tôi đã triển khai trước đó trong mot, mtmc viết bằng c++
  thì code bên này shipvision cũng phải có - dù chúng đã có 1 version bên python cũng không sao
  => vẫn có thể gọi được ops binding c++ thay vì python. Phần: C10 (tmux) — có cần retrofit
  không, hay timeout + --rm là đủ khi các run giờ chỉ tính bằng phút? thì bạn tự quyết theo hướng
  bạn nghĩ là tối ưu nhất*
- **V51** — *ngoài ra bạn vẫn chưa trả lời tôi, liệu bạn có còn xảy ra hiện tượng "quên" - tức là
  task chưa xong bạn vẫn dừng lại và bắt tôi phải tự gõ tiếp tục hay không*
