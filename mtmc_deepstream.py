import sys
import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

# -----------------------------------------------------------------------------
# 1. PAD PROBE: CỬA NGÕ GOM METADATA VỀ CENTRAL MATCHING (KHÔNG CHUYỂN RAW FRAME)
# -----------------------------------------------------------------------------
def mtmc_collector_probe_cb(pad, info, gpu_id):
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    # Lấy Metadata đã đóng gói từ VRAM
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        cam_id = frame_meta.pad_index
        frame_num = frame_meta.frame_num
        
        # Duyệt qua từng Bounding Box để trích xuất Feature Vector
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            local_track_id = obj_meta.object_id
            
            # (Đẩy struct [gpu_id, cam_id, frame_num, local_track_id, reid_vector] 
            #  vào Central Queue để xử lý Cross-Camera Matching ở Central CPU/Process)

            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------
# 2. XÂY DỰNG VÀ LINK PIPELINE CHI TIẾT
# -----------------------------------------------------------------------------
def build_and_link_pipeline():
    Gst.init(None)
    pipeline = Gst.Pipeline.new("ds-2gpu-4cam-decoupled-pipeline")

    # Cấu hình phân bổ 4 Camera trên 2 GPU
    gpu_branches = [
        {"gpu_id": 0, "cams": [0, 1]},  # GPU 0 gánh Cam 0, Cam 1
        {"gpu_id": 1, "cams": [2, 3]}   # GPU 1 gánh Cam 2, Cam 3
    ]

    for branch in gpu_branches:
        gpu_id = branch["gpu_id"]
        cam_ids = branch["cams"]

        # A. KHỞI TẠO CÁC ELEMENT CHO NHÁNH GPU NÀY
        sub_mux = Gst.ElementFactory.make("nvstreammux", f"sub_mux_gpu_{gpu_id}")
        sub_mux.set_property("gpu-id", gpu_id)
        sub_mux.set_property("batch-size", len(cam_ids))
        sub_mux.set_property("width", 1920)
        sub_mux.set_property("height", 1080)
        sub_mux.set_property("live-source", 1)

        pgie = Gst.ElementFactory.make("nvinferserver", f"pgie_gpu_{gpu_id}")
        pgie.set_property("config-file-path", f"configs/pgie_triton_config_gpu{gpu_id}.txt")

        tracker = Gst.ElementFactory.make("nvtracker", f"tracker_gpu_{gpu_id}")
        tracker.set_property("gpu-id", gpu_id)
        tracker.set_property("ll-config-file", "configs/tracker_config.yml")
        tracker.set_property("ll-lib-file", "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so")

        sgie_reid = Gst.ElementFactory.make("nvinferserver", f"sgie_reid_gpu_{gpu_id}")
        sgie_reid.set_property("config-file-path", f"configs/sgie_reid_triton_config_gpu{gpu_id}.txt")

        sink = Gst.ElementFactory.make("fakesink", f"sink_gpu_{gpu_id}")

        # B. THÊM TẤT CẢ ELEMENT CỦA NHÁNH VÀO PIPELINE CHÍNH
        elements_to_add = [sub_mux, pgie, tracker, sgie_reid, sink]
        for elem in elements_to_add:
            if not elem:
                sys.exit(f"[ERROR] Không thể tạo Element ở GPU {gpu_id}")
            pipeline.add(elem)

        # C. KẾT NỐI ĐỘNG (DYNAMIC LINKING) CÁC CAMERA RTSP VÀO SUB_MUX
        for idx, cam_id in enumerate(cam_ids):
            src_bin = Gst.ElementFactory.make("nvurisrcbin", f"src_cam_{cam_id}")
            src_bin.set_property("uri", f"rtsp://127.0.0.1:8554/live/cam_{cam_id}")
            src_bin.set_property("gpu-id", gpu_id)
            pipeline.add(src_bin)

            # Lấy Request Pad dạng sink_N từ sub_mux
            sub_mux_sink_pad = sub_mux.get_request_pad(f"sink_{idx}")

            # nvurisrcbin tạo Pad động (Dynamic Pad) nên phải gán qua Callback Function
            def cb_pad_added(element, pad, target_pad):
                if pad.get_name().startswith("src"):
                    pad.link(target_pad)

            src_bin.connect("pad-added", cb_pad_added, sub_mux_sink_pad)

        # D. LIÊN KẾT CHUỖI TĨNH (STATIC LINKING) NỘI BỘ NHÁNH GPU
        # [sub_mux] ➔ [pgie] ➔ [tracker] ➔ [sgie_reid] ➔ [sink]
        if not sub_mux.link(pgie):
            sys.exit(f"[ERROR] Fail to link sub_mux -> pgie on GPU {gpu_id}")
        if not pgie.link(tracker):
            sys.exit(f"[ERROR] Fail to link pgie -> tracker on GPU {gpu_id}")
        if not tracker.link(sgie_reid):
            sys.exit(f"[ERROR] Fail to link tracker -> sgie_reid on GPU {gpu_id}")
        if not sgie_reid.link(sink):
            sys.exit(f"[ERROR] Fail to link sgie_reid -> sink on GPU {gpu_id}")

        # E. GẮN PAD PROBE TẠI ĐẦU RA CỦA SGIE_REID ĐỂ RÚT METADATA
        sgie_src_pad = sgie_reid.get_static_pad("src")
        if sgie_src_pad:
            sgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, mtmc_collector_probe_cb, gpu_id)

    return pipeline

# -----------------------------------------------------------------------------
# 3. CHẠY PIPELINE
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    pipeline = build_and_link_pipeline()
    loop = GLib.MainLoop()
    pipeline.set_state(Gst.State.PLAYING)
    print("[INFO] Pipeline 4-Cam / 2-GPU đã chạy thành công...")

    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)