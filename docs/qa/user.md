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

#### V92 — 06:56 UTC

> tiếp tục

#### V93 — 11:10 UTC

> tiep tuc

#### V94 — 11:58 UTC

> tiếp tục\\

#### V95 — 12:37 UTC

> tiếp tục, check liên tục 3 phút 1 lần

#### V96 — 12:51 UTC

> tiếp tục đi

*(delivered by the operator's stall-watchdog session, not typed by hand)*

#### V97 — 13:15 UTC

> tiếp tục đi

*(the stall-watchdog session again)*

#### V98 — 13:32 UTC

> tiếp tục đi

*(the stall-watchdog session)*

#### V99 — 13:47 UTC

> tiếp tục đi

*(the stall-watchdog session)*

#### V100 — 14:02 UTC

> tiếp tục đi

*(the stall-watchdog session)*

#### V101 — 14:18 UTC

> tiếp tục đi

*(the stall-watchdog session)*

#### V102 — 14:50 UTC

> tiếp tục đi

*(the stall-watchdog session)*

#### V103 — 15:19 UTC

> tiếp tục đi

*(the stall-watchdog session)*

#### V104 — 15:30 UTC

> tiếp tục đi

*(the stall-watchdog session)*

#### V105 — 15:54 UTC

> tiếp tục đi

*(the stall-watchdog session)*

#### V106 — 16:1x UTC

> list cho tôi hiện tại trong toàn bộ feature mà chúng ta đã bàn, bạn đã làm xong cái gì và cái gì còn todo

*(A status request: list everything discussed — done vs still todo. Answered from
`.claude/TASKS.md` in the session; the ledger itself is the canonical answer.)*

#### V107 — 16:2x UTC

> list cho tôi hiện tại trong toàn bộ feature mà chúng ta đã bàn, bạn đã làm xong cái gì và cái gì còn todo??? tại sao bạn không trả lời???

*(Repeat of V106 with a complaint about the missing answer — legitimate: the report was
being prepared while round-2 test work continued instead of being answered first. Full
done/todo report delivered in-session immediately.)*

#### V108 — 16:3x UTC

> note: "T4 · DeepStream tier — hoàn thiện mtmc_deepstream.py của bạn thành competitor benchmark."
> Đây không phải là benchmark, mà thực sự tôi muốn đưa deepstream vào làm 1 pipeline kế thừa từ
> abstract pipeline như ta đã từng nói trước đó (chính là topology absstraction). cái file
> deepstream_mtmc.py đó chẳng là tôi gửi bạn tham khảo và xem xét để nhìn ra topology mà hệ thống
> của ta đang muốn hướng tới thôi.
> Tóm gọn lại đang có 1 absstract topology: xương sống của cả system: input api server/offline
> inference engine nhận đầu vào là camera url/video => đi vào pipeline: ingest (có thể gstreamer,
> cv...) -> output.
> Mà hiện có 3 topology như ta đã đề cập trước đó cần triển khai:
> - threading (là hiện tại)
> - multi-process theo hướng shard (fleet launcher - nhưng mà mỗi launcher tách riêng biệt không
>   shared pool gì)
> - multi-process sharding nhưng làm theo hướng như inference-engine - mỗi launcher tách biệt xử
>   lý nhưng mà chúng share nhau pool để giảm tải workload imbalânce
> - deepstream topology

*(T4 re-scoped: DeepStream is **not** a competitor benchmark. It is a fourth topology —
a pipeline implementation inheriting the abstract topology/pipeline backbone, registered
like the others. `mtmc_deepstream.py` was reference material to show the target shape, not
a deliverable to finish. The taxonomy the system implements: **threading** (current
single-process default), **fleet** (multi-process shards, nothing shared), **service**
(multi-process shards sharing an instance pool to absorb imbalance — PR #26), and
**deepstream**. The backbone: API server / offline engine takes camera URL/video → ingest
(gstreamer, cv, …) → pipeline → output.)*

#### V109 — 16:3x UTC

> ở bước "2. CI PR (compile csrc trong CI + sửa prompt review) — sửa .github/workflows/ nên cần
> bạn merge tay.", tôi cho phép bạn tự merge luôn, đến khi nào tôi ra lệnh không được tự merge,
> phải là tôi merge thì lúc đó rule mới hiệu nghiệm

*(A standing grant: I may merge the workflows-editing CI PR myself — and self-merge stays
permitted in general — until the operator explicitly revokes it. The revocation, when it
comes, is what re-establishes "operator merges". Recorded in the standing-rules index.)*

#### V110 — 16:4x UTC

> ngoài ra bạn còn cần notice thêm 1 điều cực kì quan trọng: việc shared workload đó đúng là rất
> tốt, nhưng dễ triển khai ở pipeline đơn giản: detection -> reid -> track thôi. Ở đây của chúng
> ta ngoài detect,track còn có segment, reid, thậm chí như trong docs còn có thêm cả ocr . mtmc
> nữa đó nhé

*(Design warning for the service topology: workload sharing is easy on a simple
detect → reid → track chain, but this pipeline also carries segment, reid, OCR and MTMC.
The tier must hold for the full DAG — many stateless crop-stage models with very different
payload sizes, and stateful stages (track, MTMC) that must never cross processes. Recorded
against T3/T4 in the ledger; the per-model dispatcher-level sharing seam is the answer and
the ring budget is the cost that scales with it.)*


#### V111 — 16:5x UTC

> sao tôi thấy 1 số điều tôi nói không ở trong user.md nhỉ

*(The operator opened the repository's `docs/qa/user.md` in the IDE — the copy last merged
by docs snapshot #21, through ~V89. Every request since lives in the working copy
(`/tmp/mps/docs/qa/user.md`, V90–V111) that lands on main in the next docs snapshot PR.
Fixed the visibility now by refreshing the repository working-tree file with the current
copy, uncommitted, so the IDE shows everything; the snapshot PR carries it to main.)*

#### V112 — 17:0x UTC

> tiếp tục đi

*(the stall-watchdog session)*

#### V113 — 18:2x UTC

> tiếp tục đi

*(the stall-watchdog session)*

#### V114 — 20:2x UTC

> tiếp tuuc

*(continue — typo'd; the flow continues: #30 merged, lever 2's plan in flight)*


#### V115 — 21:0x UTC

> tiếp tục đi

*(the stall-watchdog session)*


#### V116 — 21:1x UTC

> tiếp tục đi

*(the stall-watchdog session)*


#### V117 — 22:0x UTC

> tiếp tục đi

*(the stall-watchdog session)*


#### V118 — 23:3x UTC

> tiếp tục đi

*(the stall-watchdog session)*


#### V119 — 23:2x UTC (26 Aug; first logged with a drifted clock estimate as 27 Aug 00:1x)

> tiếp tục đi

*(the stall-watchdog session)*


#### V120 — 00:0x UTC, 27 Aug

> tiếp tục đi

*(the stall-watchdog session)*


#### V121 — 01:4x UTC, 27 Aug

> tiếp tục đi

*(the stall-watchdog session)*


#### V122 — 02:2x UTC, 27 Aug

> tiếp tục đi

*(the stall-watchdog session)*


#### V123 — 02:4x UTC, 27 Aug

> tiếp tục đi

*(the stall-watchdog session)*


#### V124 — 02:5x UTC, 27 Aug

> tôi vẫn chưa hiểu tại sao lại có src/shipinfer/runtime/ops và xử lý image_ops ở đây. Thực tế
> như tôi đã nói từ trước,, các thuật toán chuyên thuộc về xử lý ảnh cần được đặt trong
> shipvision, tại shipinfer chủ yếu lo phần system và nếu muốn gọi thuật toán thì gọi từ
> shipvision mà. Ngoài ra, 3rdparty/shipvision có vẻ cũng không phải main latest mới nhất của
> nó đúng không, nó cực kì khác với những gì tôi tưởng tượng và yêu cầu trước đó

*(Two architecture concerns: (1) image-processing algorithms belong in shipvision —
shipinfer is the system layer that CALLS them — so why does `runtime/ops` hold image_ops
implementations; (2) the submodule pointer/checkout does not look like shipvision's latest
main. Answered in-session with the recorded rationale for the split, an acknowledgment of
the recent drift (#30/#31 grew real algorithms inside torch_ops), a fact-check of the
gitlink vs shipvision main vs the on-disk checkout, and a migration lane opened per the
standing principle from V50.)*


#### V125 — 03:0x UTC, 27 Aug

> shipvision cần luôn được checkout về main để đảm bảo thuật toán là mới nhâts

*(Standing rule: the shipvision submodule checkout must always be at its latest main so the
algorithms are current — done immediately for the primary working tree (was parked on the
stale `feat/detection` branch; now at `8e62786` == shipvision origin/main, which the
project's gitlink already pinned), and recorded in the standing-rules index: keep working
checkouts on shipvision main, and bump the parent's pinned gitlink promptly whenever
shipvision main moves.)*


#### V126 — 03:2x UTC, 27 Aug

> tiếp tục đi

*(the stall-watchdog session)*


### V127 · 27 Aug 2026, ~09:21 UTC — continue (after a process restart)

> tiếp tục

(Arrived immediately after the harness notification that the P4-PR2c coder agent had no
completion record — the previous Claude Code process exited while it ran. Treated as the
standing continue instruction; the agent's branch existed with no commits, so it was resumed
from its saved transcript.)

### V128 · 27 Aug 2026, ~10:4x UTC — hỏi tổng kết: nhiều PR nhưng chưa rõ đã làm gì

> tôi thấy bạn mở rất nhiều PR, nhưng mà vẫn chưa rõ hiện tại bạn đã làm những cái gì rồi ấy

(Kèm ngữ cảnh IDE: operator đang mở src/shipinfer/pipeline/graph/detect.py.)

### V129 · 27 Aug 2026, ~11:0x UTC — TẠM DỪNG: chất vấn kiến trúc tổng thể

> tôi cảm thấy bạn đang code không chuẩn lắm:
> - tại sao lại từ deepstream topology -> gọi xuống deepstream command???
> - hình dung ban đầu về hệ thống của tôi như sau (bạn check nhé):
> 1. 2 hướng đầu vào là InferenceEngine offline và api-server, nhận đầu vào là camera link stream hoặc video
> 2. input đi qua topology để xử lý (topology như ta đã bảo có thể là deepstream hoặc fleet)
> 3. topology sẽ là chuỗi pipeline, ví dụ như decode -> detect/segment/ocr -> reid -> tracker -> mtmc tracker -> output. Ở đây bạn đang code module server/ thực sự tôi đang thấy cực kì rối rắm và ko có 1 architecture nào hoàn chỉnh cả
> 4. Ví dụ như tại bước decode, output có thể là ảnh nv12 ngay trên gpu hoặc trên cpu raw như bgr,... ->forward qua các element tiếp theo.
> Bạn có thể tạm dừng, nêu lại kiến trúc tổng thể hệ thống mà bạn đang muốn dessign -> những module nào đang phụ trách phần gì -> sau đó nếu cảm thấy ko đúng với ý của tôi thì bạn hỏi lại tôi có được không

HÀNH ĐỘNG: dừng mọi lane mới; trình bày kiến trúc as-built; đối chiếu 4 điểm; hỏi lại các
điểm lệch. Đã relay trạng thái tạm dừng cho shipinfer-f6 (twin đang mổ crop-stage).

### V130 · 27 Aug 2026, ~11:2x UTC — hỏi khái niệm giữa thảo luận kiến trúc

> TENSOR request là gì

### V131 · 27 Aug 2026, ~11:4x UTC — làm rõ vision kiến trúc, yêu cầu mô tả chi tiết + cách implement

> không giống tôi hình dung lắm về architecture này:
> - đầu vào phải là camera url/video. Lí do vì sao? vì tôi muốn nó là 1 pipeline có thể decode->computer-vision->output (chuỗi như tôi bảo trên). Mà bên cạnh decode đơn giản (ví dụ dùng cv.read_video -> dùng gstreamer pipeline để đọc và xử lý ngay trên GPU). (Chuỗi pipeline computer vision tôi mong muốn, bạn hãy đọc trong references/bitbucket-subfaceid/docs <- đây chính là mục tiêu target của system này)
> - note: với chế độ inference offline/api server, ta ban đầu có thể khởi tạo thẳng M camera/video và shard đều ra. Khi thêm 1 luồng camera nữa thì cứ theo cơ chế round-robin đơn giản.
> - topology ở đây tôi hiểu là gì? chính là pipeline ở trên - nhưng mà ta abstract được để mỗi thành phần có thể code khác nhau - oop khác nhau (ví dụ: element decode -> có thể dùng gstreamer hay là cv. element detect có thể implement lại nvinfer của deepstream với customize model của ta, hoặc đơn giản là đi qua model của ta luôn....)
> - về mặt thiết kế, tôi hình dung đơn giản giống như là mtmc_deepstream.py vậy, mỗi cụm camera có thể được decode tại 1 streammux => sharding tại nvinferserver (ở đây tôi hiểu là chỉ server để load-balancing thôi - distribute image raw, hoặc có thể là xử lý infer distribute cropped ảnh luôn...)
> Bạn có thể hình dung ra hệ thống của tôi không? Bạn có thể mô tả lại 1 cách chi tiết hơn và trình bày cách implement architecture của system này như nào được không?

HÀNH ĐỘNG: đọc references/bitbucket-subfaceid/docs (target chain) + mtmc_deepstream.py,
rồi trình bày kiến trúc chi tiết + kế hoạch implement. TOPOLOGY THEO ĐỊNH NGHĨA CỦA
OPERATOR = chuỗi element trừu tượng hóa (mỗi element nhiều implementation), KHÔNG phải
bố trí process. Đây là câu trả lời cho Q1/Q2 của V129: camera-vào-cả-hai-mode (offline
engine + api server, round-robin khi thêm stream); "nvinferserver" hiểu là server
load-balancing phân phối ảnh raw/crop = vai trò của model pool/scheduler hiện có.

### V132 · 27 Aug 2026, ~12:0x UTC — DUYỆT cả 3 câu kiến trúc; yêu cầu giải thích rõ hơn sơ đồ

> cả 3 câu tôi đều oke.
> Nhưng mà bạn vẽ "Hệ thống của tôi" tôi chưa hiểu lắm, bạn nói rõ hơn được không?

QUYẾT ĐỊNH CHỐT (V132, ràng buộc từ giờ):
1. Track/MTMC = element TRONG chuỗi, nhưng shard-được để tách process khi cần (giữ cả hai cửa).
2. GIỮ endpoint tensor KServe làm mặt phụ của engine.
3. TÊN CHỐT: `topology` = chuỗi element khai báo; `runner` = cách thực thi (inprocess/fleet/deepstream).

### V133 · 27 Aug 2026, ~12:2x UTC — yêu cầu mô tả vật lý/runtime chi tiết

> bạn mô tả kĩ hơn, shard ra là shard ra gì, mỗi tầng có bao nhiêu tiến trình, bao nhiêu GPU đang chạy, nó kĩ hơn về hệ thống 1 chút

### V134 · 27 Aug 2026, ~12:4x UTC — zoom vào bên trong một shard

> bạn nói kĩ hơn trong mỗi shard được không

### V135 · 27 Aug 2026, ~12:5x UTC — chê khó hiểu, yêu cầu giải thích chậm + chi tiết pool

> giải thích kĩ hơn về POOL CỤC BỘ, ngoài ra, bạn viết thực sự khó hiểu quá, tôi viaãn chưa hiểu từ sau đoạn: [trích đoạn HÀNG ĐỢI CÔNG BẰNG] phiền bạn nói rõ ra (kiểu detail hơn như là: scheduler kéo round-robin => 1 scheduler duy nhất thôi phải không?)

GHI CHÚ CÁCH TRÌNH BÀY: operator muốn kiểu hỏi-đáp cụ thể (ai làm? bao nhiêu cái?),
ít sơ đồ nén, nhiều lời kể theo chân một frame/worker cụ thể. Áp dụng cho các giải
thích kiến trúc về sau.

### V136 · 27 Aug 2026, ~13:0x UTC — yêu cầu flow chart 2 GPU/2 shard/2 cam + các usecase

> nói lại từ đầu luồng hoạt động trong mỗi shard, pool cục bộ này có share nhau giữa các tiến trình (các shard), hay chỉ nội bộ trong 1 tiến trình thôi. Nói kĩ ra được không, nói bằng flow chart, không phải bằng lời, có thể lấy ví dụ đơn giản có 2 GPU/2 shard/xử lý 2 camera -> đưa ra cả những usecase như gpu này đang quá tải mà gpu ko quá tải thì làm sao? khi raw image decode đang ở trên GPU (nv12 GPU) rồi thì làm sao, có shared nhau image không hay là chỉ shared vùng cropped image?...

### V137 · 27 Aug 2026, ~13:2x UTC — CHỈNH THIẾT KẾ: share VRAM-first (CUDA IPC), không đi qua RAM; tổng quát hóa pool; decode mặc định ra VRAM

> 1. note: tôi không cấm: process này với tay vào VRAM của GPU khác, chỉ cần đảm bảo perf tốt và accuracy chuẩn là được.
> 2. Không ổn rồi, hiện tại shared memory/pool bạn đang dùng cả với RAM=> không ổn, ban đầu tôi expect là: nó giống như là handleCudaIPC vậy, chỉ cần cầm handle thôi, hoặc đơn giản là cudaMempcy tới các pool lẫn nhau. Việc sử dụng RAM thay vì VRAM gây down performance cực kì mạnh. => Bạn hãy thiết kế lại nhé.
> 3. Từ đó, liệu ta có thể tổng quát hóa được "pool" này không? bản chất là ta muốn shared data lẫn nhau:
> - data ở đây có 2 loại: RAM hoặc VRAM => mode default là VRAM (và tôi muốn pipeline default sẽ là gstreamer decode ra VRAM image -> bạn tham khảo bên subfaceid service cách decode)
> Bạn có hiểu ý tôi nói không?

QUYẾT ĐỊNH RÀNG BUỘC MỚI (V137):
1. KHÔNG cấm cross-process/cross-GPU VRAM access — tiêu chí duy nhất: perf + accuracy.
   (Ghi đè cách đọc doc §1 cũ; "tránh P2P" trong new-system-architecture.md hết hiệu lực.)
2. Mesh dữ liệu giữa shard: VRAM-first — CUDA IPC handle / cudaMemcpyPeer; RAM chỉ là
   fallback mode. Payload KHÔNG đi qua RAM khi hai đầu đều GPU.
3. "Pool" tổng quát hóa = typed buffer pool hai location (VRAM default | RAM), decode
   mặc định = gstreamer ra NV12 TRÊN VRAM (tham khảo subfaceid decode).

### V138 · 27 Aug 2026, ~13:5x UTC — duyệt thiết kế DataPool; hỏi tiền lệ thế giới; đề xuất pool nhiều tầng

> thiết kế như vậy tôi nghĩ oke (bạn check xem các hệ thống lớn trên thế giới có design kiểu như vậy không?) ngoài ra, theo bạn thì như vậy ta nên trao đổi ảnh raw, hay là cropped ảnh raw? Liệu khi detect với cùng 1 số lượng ảnh thì có bị quá tải không? => hình như là có: 1 ảnh với nhiều người lúc detect sẽ lâu hơn nhiều 1 ảnh với ít người => như vậy tôi nghĩ ta có thể build theo nhiều tầng pool: pool shared ở image raw và pool shared ở các tầng dưới. Bạn nên biết trong các tác vụ ở trên thì:
> - embedding
> - detect
> - segment
> là 3 task có thể gây load imbalance nhất

QUYẾT ĐỊNH V138: thiết kế DataPool/vé/probe-theo-cặp ĐƯỢC DUYỆT. Chỉ đạo mới: spill
NHIỀU TẦNG — tầng frame-raw và tầng crop; 3 stage dễ lệch tải nhất: embedding, detect,
segment.

### V139 · 27 Aug 2026, ~14:0x UTC — giả định triển khai: NVLink full giữa mọi GPU trong node

> trên thực tế lúc triển khai chắc chắc sẽ có nvlink link giữa tất cả các gpu trong cùng node nên không sao

GHI NHẬN: production node = all-to-all NVLink (NVSwitch-class) → bảng định tuyến sẽ là
"direct mọi cặp". Probe theo-cặp VẪN GIỮ trong thiết kế (chi phí ~ms một lần lúc bắt tay;
bảo vệ dev-box hiện tại vốn CÓ cặp PXB độc, và mọi deployment không-NVSwitch).

### V140 · 27 Aug 2026, ~14:1x UTC — CHỐT GIL (i); viết docs/arch.md; triển khai lại top-down; gRPC thay command

> sửa theo (i).
> Vậy ta đã hình dung sơ bộ tổng thể hệ thống, tôi cần bạn ghi rõ ràng, giải thích kĩ càng hệ thống của chúng ta trong docs/arch.md
> Sau đó bắt đầu triển khai từ trên xuống dưới lại theo đúng hệ thống này. Cần phải OOP, refactor chuẩn để không gặp vấn đề khi ta mở rộng, và code packaging phải dễ nhìn, nhìn vào là ta biết ngay từ tầng api server -> sharding các thứ như thế nào... ngoài ra, bơiỉ vì ở tầng python, bạn có thể sử dụng grpc như ở bên vllm để connect giữa 2 tiến trình được không? sẽ hay hơn rất nhiều khi bạn sử dụng trò "command" như vậy (thực tế bạn hãy xóa luôn cách dùng gọi command giữa 2 tiến trình như này đi nhé)

QUYẾT ĐỊNH V140 (ràng buộc):
1. GIL fix = (i): shipvision nhả GIL quanh đoạn thuần-native + per-thread CUDA stream đi
   cùng nhau. LUẬT V70 ĐƯỢC SỬA: từ "không đụng GIL" → "release quanh native, cấm acquire".
2. Viết docs/arch.md giải thích kĩ toàn hệ thống (English theo luật docs; bản dịch tiếng
   Việt nếu operator yêu cầu riêng).
3. Triển khai lại TOP-DOWN theo kiến trúc mới: OOP chuẩn, packaging tự-giải-thích
   (api → launch/sharding → topology → engine → datapool nhìn phát biết ngay).
4. Control-plane giữa tiến trình = gRPC (như vLLM), XÓA HẲN cơ chế command()/argv giữa
   parent-child. (Spawn process vẫn phải có, nhưng mọi điều khiển/health/add-camera đi
   qua gRPC service, không nhét vào argv+env.)
5. V129 pause được thay bằng: bắt đầu triển khai theo kiến trúc mới.

### V141 · 27 Aug 2026, ~15:0x UTC — tiếp tục

> tiếp tục

(Ngữ cảnh: #52 round 1 BLOCKING với 3 finding thật — ADR-002/ADR-015 bị đảo mà không nêu tên
+ chưa trả lời chi phí context per-peer; V127–V140 chưa snapshot vào repo; bảng P2P chưa có
artifact truy vết. Đang sửa: ADR-016, snapshot user.md+TASKS.md, K-neighborhood + ngân sách
context, chờ artifacts probe từ shipinfer-f6.)

### V142 · 27 Aug 2026, ~13:3x UTC — BỎ HẾT GIL trong shipvision (đảo V140 (i); V70 khôi phục)

> note: bỏ hết mọi GIL, có chậm cũng kệ, tôi không muốn nhìn thây GIL ở trong repo của tôi, shipvission chỉ có duy nhất nhiệm vụ là deliver thuật toán - nhiều nhất chỉ có đặt muxer quanh các bước tracker.track()

(Quyết định: V140 (i) — "release GIL trong shipvision + per-thread streams" — bị HỦY. V70 đứng
nguyên: shipvision không đụng GIL, chỉ deliver thuật toán; tối đa một mutex quanh
tracker.track(). Hậu quả chấp nhận: convoy C1b còn đó; đường xử lý được phép là phía server
(V34: port hot plane sang C++ trong csrc/ của shipinfer) chứ không phải trong shipvision.
Hành động: dừng coder Phase-0 ở /tmp/sv0, xóa worktree/branch, nhả queue shipvision, sửa
docs/arch.md §7/§10 ở PR tiếp theo.)

### V143 · 28 Aug 2026, ~06:5x UTC — tiếp tục (sau khi tiến trình Claude Code khởi động lại)

> tiếp tục

Ba coder đang chạy dở (C3 r2, C4, C6 build) bị giết cùng tiến trình cũ; worktree còn diff chưa commit.
Nối lại cả ba từ transcript, rồi tiếp tục hàng đợi C3 → C4 → C6.

### V144 · 28 Aug 2026, ~07:0x UTC — arch.md đã ghi chưa, và có code theo arch đó không?

> bạn đã ghi về arch.md mới của chúng ta chưa vậy? và thực sự bạn có nhớ là phải code tuân theo arch đó chưa vậy (tôi nhớ đã request rồi)

Có: `docs/arch.md` trên main từ #52, Section 3 giữ dòng V140 ("design of record; top-down"), §7 đã theo V142;
kế hoạch Phase C được đối chiếu từng mục với arch.md (lượt plan "mù" đầu tiên bị bỏ vì đọc nhầm checkout cũ),
mỗi brief và mỗi PR body dẫn section nó thực hiện.

### V145 · 28 Aug 2026, ~13:0x UTC — VIẾT NGẮN LẠI; một logger; envs.py kiểu omnia; checkout main + rebase

> rule: hiện tại tôi thấy thực sự bạn đang viết documentation cực kì nhiều, code thì ít. Bạn có thể đặt 1 rule giới hạn lại lượng chữ documentation để bớt lại được không? Ngoài ra, review lại toàn bộ system để xem có filter lọc bớt được số lượng chữ quá dài trong documentation lại không? Hãy viết ngắn gọn và súc tích để người đọc xem là hiểu.
> 1. tôi muốn tất cả đều chung 1 log, tại sao lại có nhiều logger _LOG khác nhau thế kia
> 2. trong envs.py tôi chỉ đơn giản cần https://github.com/osirisQdt2810/omnia/blob/main/src/omnia/envs.py định nghĩa như này thôi, sau đó import dạng shipinfer_envs.ENVIRONMENT_VAR, không cần phải viết cao siêu như này, hoặc nếu được thì thêm doc vào cũng được, nma tôi không muốn dùng biến global kiểu như INGEST_BACKEND, INGEST_HWACCEL...
> có thể làm kiểu như: "OMNIA_SMART_NOTES_IMPROVE_PROMPT_TEMPERATURE": lambda: EnvVar(...)?
> => khi lấy envs. OMNIA_SMART_NOTES_IMPROVE_PROMPT_TEMPERATURE là có ngay được giá trị là bool, int, str, choices hay gì rồi mà không cần phải cast nữa?
> 3. ngoài ra, tôi muốn bạn list lại cho tôi bạn đang làm cái gì, bạn có thể làm theo hướng: checkout về main mới nhất, sau đó là làm các feature dựa trên rebase từ main được không? hiện tại code vẫn đang khá cũ
> 4. rule: Khi bạn dev 1 feature mới, 1 branch mới, nếu trước đó ta đã có đẩy PR và được merge lên main rồi thì hãy rebase các feature hiện tại đang làm trên main

### V146 · 28 Aug 2026, ~13:1x UTC — shipvision/mtmc: `core` → `matchers`, và phải expose tracker

> trong shipvision, 3rdparty/shipvision/csrc/shipvision/mtmc/core phải gọi là matchers mới đúng. Ngoài ra tại sao trong mtmc này lại không có tracker? hãy nhìn vào mtmc-service trong references/ đó, expose interface là tracker - implement các loại tracker chứ không phải là implemnet các loại matcher

### V147 · 28 Aug 2026, ~13:2x UTC — vLLM dùng RPC gì? abstract transport của control plane

> ngoài ra, về phần grpc giao tiếp giữa các processes, bạn hãy check lại trong vllm, hình như là họ hay dùng rpc gì đó để giao tiếp thì phải, nếu giao tiếp khác với src/shipinfer/launch/proto, tìm cách abstract oop nó xem có được không

### V148 · 28 Aug 2026, ~17:1x UTC — test luồng full pipeline thật, xoá hết mock; input là video

> vậy thì tôi yêu cầu bạn, từ giờ khi test hệ thống, hãy test trên luồng full pipeline thật từ decode->output, tuỳ vào bạn đang implement feature  dùng topology gì (ví dụ fleet, deepstream, threading...) tôi không muốn bạn code ra 1 mớ bòng bòng chỉ chạy mock. Ngoài ra, như tôi đã nói rồi, hãy xoá mọi mock.py được sử dụng

> input đơn giản dùng video, thay vì dùng camera url vì ta không hề có camera url hiện tại

### V149 · 31 Aug 2026, ~15:0x UTC — main không đọc được: không map được vào architecture.md, quá nhiều docs, quá nhiều hàm thừa

> thực sự nnhìn vào remote main hiện tại, tôi không biết đọc code thế nào để nó match được docs/qa/architecture.md mà chúng ta đã đặt ra như ban đầu. Bạn code quá khó hiểu, quá nhiều docs, quá nhiều hàm thừa mà tôi cảm giác nó không cần thiết. Nhìn code, tôi chẳng thể biết nên đọc từ tầng trên -> tầng dưới tở đâu trong code đó

**Resolution (asked and answered, 31 Aug):** all three, **sequentially, one package per PR** —
split the oversized file, cut the prose inside the files touched, delete the superfluous
helpers there. Start with `runners/` (worst on both metrics). Slower, but every PR is
reviewable and main stays green.

Measured baseline at the time of the complaint (commit b450acc):
`src/shipinfer` = 275 files, 52 202 lines, of which **20 572 are prose** (39%; 0.86 prose lines
per code line). `runners/` 1.77, `topology/` 1.40, `api/` 1.38. `docs/` is only 2 768 lines, so
the bloat is INSIDE the source. Nine files over 850 lines, worst `runners/inprocess.py` at
2 121 with eight responsibilities in one class and the per-frame loop (`_walk`) at line 1487.
94 single-use private helpers under 12 lines. NOTE: the doc the operator named as
`docs/qa/architecture.md` is actually `docs/arch.md`.

### V150 · 31 Aug 2026, ~15:2x UTC — viết docs/system-design.md: đọc code top-down, đủ mọi component, và timeline phần chưa xong

> tôi nên đọc code như nào để thấy được system từ trên xuống dưới, hãy ghi vào trong docs/system-design.md nhé. ghi đầy đủ toàn bộ components mà chúng ta đang build, mục tiêu của từng pcomponents và ý nghĩa của chúng. những timeline chưa xong và dự định nó là các component gì

**Follows V149.** The answer to V149 is therefore: ONE doc that is a reading order (not more
prose in the source), plus the code changes V149 already agreed. This doc is the map; V149's
splits are what make the territory match it.

### V151 · 2 Sep 2026, ~01:5x UTC — chuyển sang session mới; để lại trạng thái đủ để chỉ gõ "tiếp tục"

> tôi muốn đổi qua 1 session mới, bạn hãy làm gì đó để ở session mới tôi chỉ việc gõ tiếp tục

**A handoff request, not a work request.** What it asks for is that no state lives only in a
session's head: every in-flight branch pushed, every decision written down, and the next action
named in the two files a session reads at start-up (`.claude/JOURNAL.md`, `.claude/TASKS.md`).
Answered by RESUME-HERE at the top of the journal plus a `[~]` ledger line per open branch.

### V152 · 2 Sep 2026, ~03:2x UTC — tiếp tục (the first message of the handed-over session)

> tiếp tục

**The V151 handoff being exercised.** No new scope: the journal's RESUME HERE entry names the
order — open `chore/docs-caps-ratchet`, then `fix/source-unavailable-redaction`, then P6-D1/D2/D3.

### V153 · 4 Sep 2026, ~08:0x UTC — PR quá ngắn là tốn kém; một PR phải phục vụ một feature

> hạn chế đẩy các PR quá ngắn, quá ít content và nội dung. Ví dụ hiện tại bạn đẩy các PR đều là tests(), tại sao không gộp lại làm 1. 1 PR được tạo ra có nghĩa là nó phục vụ cho 1 feature nào đó, có thể là implement, fix bug, increase perf... 1 PR quá ngắn thực sự gây tốt

**Aimed at three PRs from this session and it lands.** #125 was one test helper, #126 was two
lines of production code, #127 was a single test file — three separate review cycles, three
bodies, three merges, for what one feature-shaped PR would have carried. The unit is a
**feature**: an implementation, a bug fix, a measured speed-up. "It is green and it is small"
is not a reason to open one; a review round costs the reviewer and the operator more than the
change is worth at that size.

Not a licence to go back to 100-commit PRs either — the ~15 commit / ~25 file cap from V80
still stands. The rule is that a PR should be *whole*, not that it should be big: build the
feature and its tests together and open that, instead of slicing a feature into a `feat()`
and a `test()` and a `docs()`.

### V154 · 4 Sep 2026, ~08:2x UTC — đừng chờ ý kiến; tự quyết định

> không cần chơ ý kiến của tôi, bạn hãy làm theo hướng mà bạn nghĩ là tốt nhất

**A standing grant, and it retires a habit rather than a single question.** It arrived while
V146b sat marked `[!]` on a question I had just written for the operator ("should shipvision's
CUDA-free subtree get an offline g++ test target?"). That is now mine to answer, and so is the
next one of that shape: an `[!]` is for something that is genuinely unsafe or impossible
without them -- a GPU that needs resetting, a container image that cannot be built here, a
credential -- not for a design call I am able to make and defend.

It does not retire the confirmations that exist for safety: merging someone else's PR,
deleting a pushed branch, anything outward-facing or hard to reverse.

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
| **Decide it yourself** — do not park a design call as an operator question. `[!]` is for genuinely blocked (a dead GPU, an unbuildable image, a credential), not for a judgement you can make and defend; safety confirmations still stand | **V154** |
| **A PR serves one feature** — implement / fix / speed-up, with its tests inside it. Do not slice one feature into `feat()` + `test()` + `docs()`, and do not open a two-line PR at all; still inside V80's ~15 commit / ~25 file cap | **V153** |
| **System tests run the real chain, decode → output**, on the topology the feature uses (fleet / deepstream / threading); mock-only verification is not verification | **V148** |
| **Delete every `mock.py` in use** — `backends/mock.py`, `topology/elements/mock.py` | **V148**, R52, R54, V15 |
| Test input is a **video file**, not a camera URL — there is no camera to reach | **V148** |
| **Documentation is capped**: module docstring ≤ 15 lines, class/function ≤ 10, comment block ≤ 4; write short and dense | **V145** |
| One logger for the whole process — not one `_LOG` per module | **V145** |
| `envs.py` is a dict of `NAME: lambda` + `__getattr__`; `envs.NAME` is already typed; no module-global env objects | **V145** |
| Rebase every in-flight branch on `main` after each merge; the working checkout stays on latest `main` | **V145** |
| shipvision `mtmc` exposes **trackers** (like `references/mtmcservice`); the matcher directory is `matchers`, in C++ too | **V146** |
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
| **Cross-GPU / cross-process VRAM access is ALLOWED**; the only criteria are perf and accuracy. Shared data is VRAM-first (CUDA-IPC slab handles once at mesh join, per-buffer tickets); RAM is the fallback mode, never the default payload path | V137, V138, V139 |
| **Default decode is GStreamer → NV12 straight into VRAM** (subfaceid-style); BGR-on-CPU is the fallback | V137 |
| **No GIL code in shipvision, ever** — it only delivers algorithms; at most a mutex around `tracker.track()`. V140 (i) was revoked the same day; V70 stands. If the GIL caps throughput, the fix lives on the server side (V34: `csrc/` in shipinfer), and slowness is accepted over GIL code in shipvision | **V142**, V70 |
| **Names are fixed:** `topology` = the declarative element chain; `runner` = how it executes (inprocess · fleet · deepstream-compiler). track/mtmc are in-chain elements; the KServe tensor endpoint stays as the engine's side door | V132 |
| **`docs/arch.md` is the design of record**; implementation proceeds top-down from it, OOP, packaging readable from api → sharding | V140 |
| **Processes talk gRPC (vLLM style); the argv "command" mechanism between parent and child is deleted**, not wrapped | V140 |
| Explanations of the system: say WHO does it and HOW MANY, use flowcharts, walk one frame through — no prose-only descriptions | V135, V136 |
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
| **Self-merge is granted, standing until the operator explicitly revokes it** — the revocation, when it comes, is what re-establishes "operator merges". Exercised once: #28. | V109 |
| Poll the open PR's checks and review every ~3 minutes while it is in flight | V95 |
| **The shipvision checkout is always its latest main** — working trees keep the submodule on shipvision main, and the parent's pinned gitlink is bumped promptly when shipvision main moves, so the algorithms in play are always current | V125 |
| **Workload sharing must hold for the full DAG** — segment, reid (person & ship), OCR and MTMC beside detect/track, not the simple detect→reid→track chain; stateful stages stay pinned per shard and the ring budget scales with the shared set | V110 |

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
