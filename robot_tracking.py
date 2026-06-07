"""
=============================================================
  Bullseye Tracking  –  YOLOv8 (best.pt)  +  ESP32 PWM
  v2: Fix lag DroidCam WiFi + chuyển inference sang CUDA
=============================================================
  Fix lag:
    1. Thread riêng đọc camera liên tục → luôn lấy frame MỚI NHẤT,
       bỏ hết frame cũ tích tụ trong buffer WiFi.
    2. DEVICE = "cuda" → YOLO chạy trên GPU NVIDIA, nhanh ~10x.
    3. Resize frame xuống 480p trước khi đưa vào YOLO → giảm
       thêm thời gian inference mà không ảnh hưởng độ chính xác.
=============================================================
  Cấu hình nhanh:
    MODEL_PATH   – đường dẫn tới best.pt
    ESP32_IP     – IP của ESP32
    CAMERA_URL   – URL DroidCam / IP-Cam
    CONF_THRESH  – ngưỡng confidence (0.0–1.0)
    IOU_THRESH   – ngưỡng IoU NMS
    INFER_WIDTH  – chiều rộng resize trước inference (px)
=============================================================
"""

import cv2
import requests
import time
import threading
import numpy as np
from collections import deque
from ultralytics import YOLO

# ===================== CẤU HÌNH =====================
MODEL_PATH  = "best.pt"           # <-- chỉ định đường dẫn model
ESP32_IP    = "172.20.10.4"
CAMERA_URL  = "http://172.20.10.2:4747/video"

CONF_THRESH  = 0.40   # ngưỡng confidence box (giảm nếu bỏ sót, tăng nếu nhận nhầm)
IOU_THRESH   = 0.45   # ngưỡng IoU cho NMS
DEVICE       = "cuda" # ← GPU NVIDIA; đổi "cpu" nếu muốn test không dùng GPU
INFER_WIDTH  = 640    # resize frame về chiều rộng này trước inference
                      # 480 → nhanh hơn / 640 → chính xác hơn (khuyến nghị)

# ── 3 vùng khoảng cách (dựa trên diện tích bbox / frame) ─────
STOP_AREA_THRESHOLD = 0.30   # dừng hẳn khi bbox > 30% frame
SLOW_AREA_THRESHOLD = 0.12   # giảm tốc khi bbox > 12%
SAFE_AREA_THRESHOLD = 0.05   # tiến chậm khi bbox < 5% (quá xa)

# ── Ngưỡng lệch (px) ──────────────────────────────────────────
ERR_SMALL = 75
ERR_LARGE = 170

# ── Tốc độ PWM (0–255) ────────────────────────────────────────
SPEED_MIN   = 220
SPEED_MAX   = 255
SPEED_FWD   = 210
SPEED_ALIGN = 230

# ── Dead zone & thời gian ổn định ─────────────────────────────
ALIGN_DEAD_ZONE      = 93
ALIGN_STABLE_TIME    = 0.2
FOLLOW_DEAD_ZONE     = 120
FINETUNE_DEAD_ZONE   = 130
FINETUNE_STABLE_TIME = 0.3

# ── Timing gửi lệnh ───────────────────────────────────────────
SEND_INTERVAL = 0.20
CMD_HOLD_TIME = 0.12
SMOOTH_WINDOW = 5
FRAME_SKIP    = 1


# ===================== KHỞI TẠO MODEL =====================
print(f"[MODEL] Đang load {MODEL_PATH} ...")
model = YOLO(MODEL_PATH)
model.conf = CONF_THRESH
model.iou  = IOU_THRESH
# Warm-up (giảm độ trễ frame đầu)
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(dummy, verbose=False, device=DEVICE)
print("[MODEL] Load xong —", model.names)

# ===================== CAMERA THREAD (chống lag WiFi) =====================
# Vấn đề: OpenCV buffer tích tụ frame cũ khi WiFi chậm.
# Giải pháp: thread liên tục đọc và GHI ĐÈ, main loop chỉ lấy frame MỚI NHẤT.

class CameraStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            print("[LỖI] Không mở được camera.")
            exit()
        self.frame  = None
        self.lock   = threading.Lock()
        self.active = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        """Đọc liên tục, chỉ giữ frame mới nhất."""
        while self.active:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        """Trả về frame mới nhất (hoặc None nếu chưa có)."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.active = False
        self.cap.release()

print("[CAM] Đang kết nối camera...")
cam = CameraStream(CAMERA_URL)
# Chờ frame đầu tiên
for _ in range(50):
    if cam.read() is not None:
        break
    time.sleep(0.1)
print("[CAM] OK — Nhấn 'q' để thoát.\n")

# ── State machine ──────────────────────────────────────────────
STATE_ALIGN    = "ALIGN"
STATE_FOLLOW   = "FOLLOW"
STATE_FINETUNE = "FINETUNE"
state = STATE_ALIGN

align_stable_since    = None
finetune_stable_since = None

cx_history = deque(maxlen=SMOOTH_WINDOW)

last_cmd     = None
last_send    = 0.0
cmd_set_time = 0.0
pending_cmd  = None

frame_count = 0


# ===================== PHÁT HIỆN HỒNG TÂM (YOLOv8) =====================

def detect_bullseye(frame):
    """
    Phát hiện hồng tâm bằng YOLOv8.

    Trả về:
        (cx, cy, x1, y1, x2, y2, area_ratio, conf)  nếu tìm thấy
        None                                          nếu không thấy

    Chiến lược: lấy box có confidence CAO NHẤT trong số tất cả
    detection class 'Target'.
    """
    h, w = frame.shape[:2]

    # ── Resize để tăng tốc inference (scale bbox về toạ độ gốc sau) ──
    scale = INFER_WIDTH / w
    infer_h = int(h * scale)
    infer_frame = cv2.resize(frame, (INFER_WIDTH, infer_h))

    results = model.predict(
        infer_frame,
        verbose=False,
        device=DEVICE,
        conf=CONF_THRESH,
        iou=IOU_THRESH
    )

    best_box  = None
    best_conf = 0.0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            # Chỉ xét class 0 = 'Target'
            if cls_id != 0:
                continue
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box  = box

    if best_box is None:
        return None

    # Toạ độ bounding box → scale ngược về kích thước frame gốc
    x1i, y1i, x2i, y2i = map(int, best_box.xyxy[0])
    x1 = int(x1i / scale);  y1 = int(y1i / scale)
    x2 = int(x2i / scale);  y2 = int(y2i / scale)
    bw = x2 - x1
    bh = y2 - y1

    # Centroid của box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    area_ratio = (bw * bh) / (w * h)

    return cx, cy, x1, y1, x2, y2, area_ratio, best_conf


# ===================== HÀM TIỆN ÍCH =====================

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def spin_speed(abs_err: int) -> int:
    if abs_err < ERR_SMALL:
        return SPEED_MIN
    elif abs_err < ERR_LARGE:
        return (SPEED_MIN + SPEED_ALIGN) // 2
    else:
        return SPEED_ALIGN


def spin_command(err: int):
    spd = spin_speed(abs(err))
    if err > 0:
        return "right", -spd, spd
    else:
        return "left", spd, -spd


def follow_command(cx_smooth: int, frame_w: int, area_ratio: float):
    err = cx_smooth - frame_w // 2

    if area_ratio > STOP_AREA_THRESHOLD:
        if abs(err) > ALIGN_DEAD_ZONE:
            spd = spin_speed(abs(err))
            if err > 0:
                return "right", -spd, spd
            else:
                return "left", spd, -spd
        return "stop", 0, 0

    if area_ratio > SLOW_AREA_THRESHOLD:
        t    = (area_ratio - SLOW_AREA_THRESHOLD) / (STOP_AREA_THRESHOLD - SLOW_AREA_THRESHOLD)
        base = int(SPEED_FWD + t * (SPEED_MIN - SPEED_FWD))
        base = clamp(base, SPEED_MIN, SPEED_FWD)
    elif area_ratio < SAFE_AREA_THRESHOLD:
        base = SPEED_MIN
    else:
        base = SPEED_FWD

    if abs(err) <= FOLLOW_DEAD_ZONE:
        return "forward", base, base

    spd = spin_speed(abs(err))
    if err > 0:
        return "right", -spd, spd
    else:
        return "left", spd, -spd


def fine_tune_command(cx_smooth: int, frame_w: int, now: float) -> tuple:
    global finetune_stable_since

    err = cx_smooth - frame_w // 2

    if abs(err) <= FINETUNE_DEAD_ZONE:
        if finetune_stable_since is None:
            finetune_stable_since = now
        elif now - finetune_stable_since >= FINETUNE_STABLE_TIME:
            finetune_stable_since = None
            return "stop", 0, 0, True
        return "stop", 0, 0, False
    else:
        finetune_stable_since = None
        if err > 0:
            return "right", -SPEED_MIN, SPEED_MIN, False
        else:
            return "left", SPEED_MIN, -SPEED_MIN, False


def send_track(cmd: str, spd_l: int, spd_r: int):
    url = f"http://{ESP32_IP}/track?cmd={cmd}&l={spd_l}&r={spd_r}"
    def _go():
        try:
            requests.get(url, timeout=0.3)
        except Exception as e:
            print(f"[SEND ERR] {e}")
    threading.Thread(target=_go, daemon=True).start()


# ===================== VÒNG LẶP CHÍNH =====================
while True:
    frame = cam.read()
    if frame is None:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    annotated    = frame.copy()
    result       = detect_bullseye(frame)

    cmd, spd_l, spd_r = "stop", 0, 0
    target_found = False
    err          = 0
    conf_disp    = 0.0

    if result is not None:
        cx, cy, x1, y1, x2, y2, area_ratio, conf_disp = result

        cx_history.append(cx)
        cx_smooth = int(np.mean(cx_history))
        err       = cx_smooth - frame.shape[1] // 2

        target_found = True
        now = time.time()

        # ── Vẽ kết quả nhận diện ──────────────────────────
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(annotated, (cx_smooth, cy), 8, (0, 255, 255), -1)   # centroid mượt
        cv2.circle(annotated, (cx, cy),        5, (255, 0, 255), -1)   # centroid thô

        label = f"Target {conf_disp:.2f}  err={err:+d}  area={area_ratio:.2f}  [{state}]"
        cv2.putText(annotated, label,
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ════════════════════════════════════════════════
        #  STATE MACHINE
        # ════════════════════════════════════════════════
        if state == STATE_ALIGN:
            if abs(err) <= ALIGN_DEAD_ZONE:
                if align_stable_since is None:
                    align_stable_since = now
                elif now - align_stable_since >= ALIGN_STABLE_TIME:
                    state = STATE_FOLLOW
                    align_stable_since = None
                    print("[STATE] ALIGN → FOLLOW")
                cmd, spd_l, spd_r = "stop", 0, 0
            else:
                align_stable_since = None
                cmd, spd_l, spd_r  = spin_command(err)

        elif state == STATE_FOLLOW:
            if abs(err) > FOLLOW_DEAD_ZONE * 3:
                state = STATE_ALIGN
                align_stable_since    = None
                finetune_stable_since = None
                cmd, spd_l, spd_r = spin_command(err)
                print("[STATE] FOLLOW → ALIGN (lệch quá)")
            else:
                cmd, spd_l, spd_r = follow_command(cx_smooth, frame.shape[1], area_ratio)
                if cmd == "stop" and area_ratio > STOP_AREA_THRESHOLD:
                    state = STATE_FINETUNE
                    finetune_stable_since = None
                    print("[STATE] FOLLOW → FINETUNE")

        elif state == STATE_FINETUNE:
            cmd, spd_l, spd_r, done = fine_tune_command(cx_smooth, frame.shape[1], now)
            if done:
                print("[STATE] FINETUNE → DONE (đã căn tâm chính xác)")
            if area_ratio < SLOW_AREA_THRESHOLD:
                state = STATE_FOLLOW
                finetune_stable_since = None
                print("[STATE] FINETUNE → FOLLOW (mục tiêu lùi xa)")

        # ── Dead zone lines ────────────────────────────────
        fc = frame.shape[1] // 2
        dz_color = (0, 80, 255) if state == STATE_FINETUNE else (0, 200, 255)
        dz = FINETUNE_DEAD_ZONE if state == STATE_FINETUNE else ALIGN_DEAD_ZONE
        cv2.line(annotated, (fc - dz, 0), (fc - dz, frame.shape[0]), dz_color, 1)
        cv2.line(annotated, (fc + dz, 0), (fc + dz, frame.shape[0]), dz_color, 1)

    # ── Mất mục tiêu ──────────────────────────────────────
    if not target_found:
        cmd, spd_l, spd_r = "stop", 0, 0
        if state in (STATE_FOLLOW, STATE_FINETUNE):
            state = STATE_ALIGN
            align_stable_since    = None
            finetune_stable_since = None
            print("[STATE] → ALIGN (mất mục tiêu)")

    # ── Giữ lệnh CMD_HOLD_TIME trước khi đổi ──────────────
    now = time.time()
    if cmd != pending_cmd:
        pending_cmd  = cmd
        cmd_set_time = now

    confirmed_cmd = cmd if (now - cmd_set_time) >= CMD_HOLD_TIME else last_cmd

    if confirmed_cmd is not None and (
            confirmed_cmd != last_cmd or (now - last_send) >= SEND_INTERVAL):
        send_track(confirmed_cmd, spd_l, spd_r)
        last_cmd  = confirmed_cmd
        last_send = now
        print(f"[{state:8}] {confirmed_cmd:8s}  L={spd_l:+4d}  R={spd_r:+4d}  conf={conf_disp:.2f}")

    # ── HUD ───────────────────────────────────────────────
    hud_colors = {
        STATE_ALIGN:    (0, 165, 255),
        STATE_FOLLOW:   (0, 255, 0),
        STATE_FINETUNE: (0, 80, 255),
    }
    color_hud = hud_colors.get(state, (255, 255, 255))
    cv2.putText(annotated,
                f"[{state}] {confirmed_cmd}  L={spd_l} R={spd_r}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_hud, 2)

    # ── Thanh khoảng cách ─────────────────────────────────
    if target_found:
        bar_x, bar_y, bar_h, bar_total = 10, 45, 14, 220
        cv2.rectangle(annotated,
                      (bar_x, bar_y), (bar_x + bar_total, bar_y + bar_h),
                      (40, 40, 40), -1)
        filled = int(clamp(area_ratio / STOP_AREA_THRESHOLD, 0, 1) * bar_total)
        if area_ratio > STOP_AREA_THRESHOLD:
            bar_color = (0, 0, 220)
        elif area_ratio > SLOW_AREA_THRESHOLD:
            bar_color = (0, 200, 255)
        else:
            bar_color = (0, 210, 60)
        cv2.rectangle(annotated,
                      (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                      bar_color, -1)
        slow_x = int(SLOW_AREA_THRESHOLD / STOP_AREA_THRESHOLD * bar_total) + bar_x
        cv2.line(annotated, (slow_x, bar_y), (slow_x, bar_y + bar_h), (0, 200, 255), 2)
        cv2.putText(annotated,
                    f"area={area_ratio:.2f}  conf={conf_disp:.2f}",
                    (bar_x + bar_total + 5, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

    # ── Thanh progress căn giữa ───────────────────────────
    show_align_bar = (state == STATE_ALIGN    and target_found)
    show_ft_bar    = (state == STATE_FINETUNE and target_found)
    if show_align_bar or show_ft_bar:
        dz    = FINETUNE_DEAD_ZONE if show_ft_bar else ALIGN_DEAD_ZONE
        label = "FineTune" if show_ft_bar else "Align"
        color = (0, 80, 255) if show_ft_bar else (0, 165, 255)
        pct   = clamp(1.0 - abs(err) / (frame.shape[1] / 2), 0, 1)
        bar_w = int(200 * pct)
        cv2.rectangle(annotated, (10, 65), (210, 80), (50, 50, 50), -1)
        cv2.rectangle(annotated, (10, 65), (10 + bar_w, 80), color, -1)
        cv2.putText(annotated, label, (215, 78),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.imshow("Bullseye Tracking (YOLOv8)", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Dừng xe khi thoát ─────────────────────────────────────────
send_track("stop", 0, 0)
time.sleep(0.3)
cam.release()
cv2.destroyAllWindows()
print("Done.")
