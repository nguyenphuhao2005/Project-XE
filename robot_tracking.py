import cv2
import requests
import time
import threading
import numpy as np
from collections import deque
from ultralytics import YOLO

# ===================== CẤU HÌNH =====================
ESP32_IP             = "172.20.10.4"
CAMERA_URL           = 'http://172.20.10.2:4747/video'
CONFIDENCE_THRESHOLD = 0.4
TARGET_CLASS_ID      = 0                 # 0 = person

# ── 3 vùng khoảng cách (dựa trên diện tích bbox / frame) ────
STOP_AREA_THRESHOLD  = 0.30# Dừng hẳn khi bbox > 22% frame
SLOW_AREA_THRESHOLD  = 0.12  # Bắt đầu giảm tốc khi bbox > 12%
SAFE_AREA_THRESHOLD  = 0.05  # Bbox < 5% → tiến chậm (quá xa)

# ── Tốc độ xoay theo 3 mức độ lệch ─────────────────────────
ERR_SMALL = 75  # px: ngưỡng lệch nhỏ
ERR_LARGE = 170 # px: ngưỡng lệch lớn

# ── Tốc độ PWM (0–255) ───────────────────────────────────────
SPEED_MIN      = 220
SPEED_MAX      = 255
SPEED_FWD      = 210
SPEED_ALIGN    = 230

# ── Ngưỡng căn giữa ──────────────────────────────────────────
ALIGN_DEAD_ZONE    = 93 # sai lệch cho phép khi xoay căn giữa
ALIGN_STABLE_TIME  = 0.2 # giây: phải giữ giữa liên tục bao lâu mới chuyển sang FOLLOW
FOLLOW_DEAD_ZONE   = 120 #qd zone khi đang FOLLOW (rộng hơn để không lắc)

# ── Tinh chỉnh khung hình sau khi FOLLOW xong ────────────────
# Chỉnh 2 giá trị này nếu muốn căn chặt/lỏng hơn lúc đỗ
FINETUNE_DEAD_ZONE  = 130#one rất hẹp để căn tâm chính xác nhất
FINETUNE_STABLE_TIME = 0.3#iây: giữ tâm đủ lâu thì mới coi là xong

# ── Timing ───────────────────────────────────────────────────
SEND_INTERVAL  = 0.20
CMD_HOLD_TIME  = 0.12
SMOOTH_WINDOW  = 5
FRAME_SKIP     = 1
YOLO_IMGSZ     = 320

# ===================== KHỞI TẠO =====================
import torch
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')

if torch.cuda.is_available():
    model.to('cuda')
    print(f"OK — GPU: {torch.cuda.get_device_name(0)}")
else:
    print("OK — Không có GPU, chạy CPU (cài torch+cuda để nhanh hơn)")

print("Connecting to camera...")
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    exit()
print("OK — Press 'q' to quit.")

# ── State machine ─────────────────────────────────────────────
# ALIGN   : xe xoay tại chỗ cho đến khi mục tiêu vào giữa
# FOLLOW  : tiến về phía mục tiêu + giữ tâm
# FINETUNE: đã đến đích, tinh chỉnh khung hình về tâm chính xác nhất
STATE_ALIGN    = "ALIGN"
STATE_FOLLOW   = "FOLLOW"
STATE_FINETUNE = "FINETUNE"
state = STATE_ALIGN

align_stable_since    = None  # thời điểm cx bắt đầu nằm trong ALIGN_DEAD_ZONE
finetune_stable_since = None  # thời điểm cx bắt đầu nằm trong FINETUNE_DEAD_ZONE

# ── Moving average cx ────────────────────────────────────────
cx_history = deque(maxlen=SMOOTH_WINDOW)

# ── Gửi lệnh ─────────────────────────────────────────────────
last_cmd     = None
last_send    = 0.0
cmd_set_time = 0.0
pending_cmd  = None

frame_count = 0


# ===================== HÀM TIỆN ÍCH =====================

def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def spin_speed(abs_err: int) -> int:
    """3 mức tốc độ xoay theo độ lệch."""
    if abs_err < ERR_SMALL:
        return SPEED_MIN
    elif abs_err < ERR_LARGE:
        return (SPEED_MIN + SPEED_ALIGN) // 2
    else:
        return SPEED_ALIGN


def spin_command(err: int):
    """Xoay tại chỗ (4 bánh) để căn giữa."""
    spd = spin_speed(abs(err))
    if err > 0:
        return "right", -spd, spd
    else:
        return "left", spd, -spd


def follow_command(cx_smooth: int, frame_w: int, area_ratio: float):
    """
    Điều khiển khi FOLLOW:
      STOP zone + chưa căn  → xoay tại chỗ (ALIGN_DEAD_ZONE)
      STOP zone + đã căn    → "stop" → báo hiệu chuyển sang FINETUNE
      SLOW zone             → tiến chậm dần
      SAFE zone             → tiến tốc độ bình thường
    """
    err = cx_smooth - frame_w // 2

    # ── STOP: quá gần ────────────────────────────────────────
    if area_ratio > STOP_AREA_THRESHOLD:
        if abs(err) > ALIGN_DEAD_ZONE:
            # Chưa vào tâm → xoay tại chỗ trước khi dừng
            spd = spin_speed(abs(err))
            if err > 0:
                return "right", -spd, spd
            else:
                return "left", spd, -spd
        # Đã vào tâm → dừng, báo hiệu để chuyển FINETUNE
        return "stop", 0, 0

    # ── Tính base speed theo vùng khoảng cách ────────────────
    if area_ratio > SLOW_AREA_THRESHOLD:
        t    = (area_ratio - SLOW_AREA_THRESHOLD) / (STOP_AREA_THRESHOLD - SLOW_AREA_THRESHOLD)
        base = int(SPEED_FWD + t * (SPEED_MIN - SPEED_FWD))
        base = clamp(base, SPEED_MIN, SPEED_FWD)
    elif area_ratio < SAFE_AREA_THRESHOLD:
        base = SPEED_MIN
    else:
        base = SPEED_FWD

    # ── Trong dead zone: tiến thẳng ──────────────────────────
    if abs(err) <= FOLLOW_DEAD_ZONE:
        return "forward", base, base

    # ── Ngoài dead zone: xoay tại chỗ ───────────────────────
    spd = spin_speed(abs(err))
    if err > 0:
        return "right", -spd, spd
    else:
        return "left", spd, -spd


def fine_tune_command(cx_smooth: int, frame_w: int, now: float) -> tuple:
    """
    Tinh chỉnh khung hình SAU KHI đã FOLLOW xong.
    Chỉ kích hoạt ở STATE_FINETUNE — không can thiệp vào ALIGN/FOLLOW.

    Logic:
      - Dùng FINETUNE_DEAD_ZONE (rất hẹp, mặc định 20px) để căn chính xác.
      - Luôn dùng SPEED_MIN (tốc độ thấp nhất) để xoay tinh tế.
      - Phải giữ tâm liên tục FINETUNE_STABLE_TIME giây mới coi là xong.
      - Trả về (cmd, spd_l, spd_r, done):
            done=True  → đã căn xong, có thể dừng hẳn
            done=False → vẫn đang tinh chỉnh
    """
    global finetune_stable_since

    err = cx_smooth - frame_w // 2

    if abs(err) <= FINETUNE_DEAD_ZONE:
        # Trong tâm → đếm thời gian giữ ổn định
        if finetune_stable_since is None:
            finetune_stable_since = now
        elif now - finetune_stable_since >= FINETUNE_STABLE_TIME:
            # Giữ đủ lâu → xong hoàn toàn
            finetune_stable_since = None
            return "stop", 0, 0, True   # done=True
        # Chưa đủ thời gian giữ → dừng chờ
        return "stop", 0, 0, False
    else:
        # Ngoài tâm → reset bộ đếm và xoay tinh chỉnh
        finetune_stable_since = None
        if err > 0:
            return "right", -SPEED_MIN, SPEED_MIN, False
        else:
            return "left", SPEED_MIN, -SPEED_MIN, False


def send_track(cmd: str, spd_l: int, spd_r: int):
    """Gửi HTTP không đồng bộ, không block main loop."""
    url = f"http://{ESP32_IP}/track?cmd={cmd}&l={spd_l}&r={spd_r}"
    def _go():
        try:
            requests.get(url, timeout=0.3)
        except Exception as e:
            print(f"[SEND ERR] {e}")
    threading.Thread(target=_go, daemon=True).start()


# ===================== VÒNG LẶP CHÍNH =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    results   = model(frame, conf=CONFIDENCE_THRESHOLD, imgsz=YOLO_IMGSZ, verbose=False)
    annotated = results[0].plot()
    boxes     = results[0].boxes

    cmd, spd_l, spd_r = "stop", 0, 0
    target_found = False

    if boxes is not None:
        for box in boxes:
            if int(box.cls[0]) != TARGET_CLASS_ID:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx = (x1 + x2) // 2

            cx_history.append(cx)
            cx_smooth = int(np.mean(cx_history))

            box_area   = (x2 - x1) * (y2 - y1)
            frame_area = frame.shape[0] * frame.shape[1]
            area_ratio = box_area / frame_area
            err        = cx_smooth - frame.shape[1] // 2

            target_found = True
            now = time.time()

            # ════════════════════════════════════════════
            #  STATE MACHINE
            # ════════════════════════════════════════════
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
                    # Lệch quá nhiều → căn lại từ đầu
                    state = STATE_ALIGN
                    align_stable_since    = None
                    finetune_stable_since = None
                    cmd, spd_l, spd_r = spin_command(err)
                    print("[STATE] FOLLOW → ALIGN (lệch quá)")
                else:
                    cmd, spd_l, spd_r = follow_command(cx_smooth, frame.shape[1], area_ratio)
                    # ★ follow_command trả "stop" khi đã vào STOP zone + căn đủ
                    #   → chuyển sang FINETUNE để tinh chỉnh lần cuối
                    if cmd == "stop" and area_ratio > STOP_AREA_THRESHOLD:
                        state = STATE_FINETUNE
                        finetune_stable_since = None
                        print("[STATE] FOLLOW → FINETUNE")

            elif state == STATE_FINETUNE:
                # ★ Gọi hàm tinh chỉnh riêng — chỉ kích hoạt ở đây
                cmd, spd_l, spd_r, done = fine_tune_command(cx_smooth, frame.shape[1], now)
                if done:
                    print("[STATE] FINETUNE → DONE (đã căn tâm chính xác)")
                # Nếu mục tiêu bỗng đi ra xa (người lùi lại) → quay lại FOLLOW
                if area_ratio < SLOW_AREA_THRESHOLD:
                    state = STATE_FOLLOW
                    finetune_stable_since = None
                    print("[STATE] FINETUNE → FOLLOW (mục tiêu lùi xa)")

            # ── Debug overlay ────────────────────────────
            cv2.circle(annotated, (cx_smooth, (y1 + y2) // 2), 6, (0, 255, 255), -1)
            cv2.putText(annotated,
                        f"err={err:+d}  area={area_ratio:.2f}  [{state}]",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Vẽ vùng dead zone (đổi màu theo state)
            fc = frame.shape[1] // 2
            dz_color = (0, 80, 255) if state == STATE_FINETUNE else (0, 200, 255)
            dz = FINETUNE_DEAD_ZONE if state == STATE_FINETUNE else ALIGN_DEAD_ZONE
            cv2.line(annotated, (fc - dz, 0), (fc - dz, frame.shape[0]), dz_color, 1)
            cv2.line(annotated, (fc + dz, 0), (fc + dz, frame.shape[0]), dz_color, 1)
            break

    if not target_found:
        cmd, spd_l, spd_r = "stop", 0, 0
        if state in (STATE_FOLLOW, STATE_FINETUNE):
            state = STATE_ALIGN
            align_stable_since    = None
            finetune_stable_since = None
            print(f"[STATE] {state} → ALIGN (mất mục tiêu)")

    # ── Giữ lệnh CMD_HOLD_TIME trước khi đổi ────────────────
    now = time.time()
    if cmd != pending_cmd:
        pending_cmd  = cmd
        cmd_set_time = now

    confirmed_cmd = cmd if (now - cmd_set_time) >= CMD_HOLD_TIME else last_cmd

    # ── Gửi theo SEND_INTERVAL ───────────────────────────────
    if confirmed_cmd is not None and (
            confirmed_cmd != last_cmd or (now - last_send) >= SEND_INTERVAL):
        send_track(confirmed_cmd, spd_l, spd_r)
        last_cmd  = confirmed_cmd
        last_send = now
        print(f"[{state:8}] {confirmed_cmd:8s}  L={spd_l:+4d}  R={spd_r:+4d}")

    # ── HUD ─────────────────────────────────────────────────
    hud_colors = {
        STATE_ALIGN:    (0, 165, 255),   # cam
        STATE_FOLLOW:   (0, 255, 0),     # xanh lá
        STATE_FINETUNE: (0, 80, 255),    # xanh dương
    }
    color_hud = hud_colors.get(state, (255, 255, 255))
    cv2.putText(annotated,
                f"[{state}] {confirmed_cmd}  L={spd_l} R={spd_r}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_hud, 2)

    # ── Thanh khoảng cách ────────────────────────────────────
    if target_found:
        bar_x, bar_y, bar_h, bar_total = 10, 45, 14, 220
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_total, bar_y + bar_h), (40, 40, 40), -1)
        filled = int(clamp(area_ratio / STOP_AREA_THRESHOLD, 0, 1) * bar_total)
        if area_ratio > STOP_AREA_THRESHOLD:
            bar_color = (0, 0, 220)
        elif area_ratio > SLOW_AREA_THRESHOLD:
            bar_color = (0, 200, 255)
        else:
            bar_color = (0, 210, 60)
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), bar_color, -1)
        slow_x = int(SLOW_AREA_THRESHOLD / STOP_AREA_THRESHOLD * bar_total) + bar_x
        cv2.line(annotated, (slow_x, bar_y), (slow_x, bar_y + bar_h), (0, 200, 255), 2)
        cv2.putText(annotated, f"area={area_ratio:.2f}", (bar_x + bar_total + 5, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, bar_color, 1)

    # ── Thanh progress căn giữa (ALIGN / FINETUNE) ───────────
    show_align_bar = (state == STATE_ALIGN and target_found)
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

    cv2.imshow("Person Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Dừng xe khi thoát ───────────────────────────────────────
send_track("stop", 0, 0)
time.sleep(0.3)
cap.release()
cv2.destroyAllWindows()
print("Done.")
