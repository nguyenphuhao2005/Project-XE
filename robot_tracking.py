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

STOP_AREA_THRESHOLD  = 0.3              # Dừng nếu bbox > 50% frame (quá gần)
SAFE_AREA_THRESHOLD  = 0.05             # Bbox < 5% → tiến chậm (quá xa)

# ── PID (dùng chung cho cả ALIGN và FOLLOW) ──────────────────
KP = 0.08
KI = 0.001
KD = 0.05

# ── Tốc độ PWM (0–255) ───────────────────────────────────────
SPEED_MIN      = 170    # Tốc độ tối thiểu để bánh còn quay
SPEED_MAX      = 200   # Tốc độ tối đa tiến/lùi
SPEED_FWD      = 180  # Tốc độ tiến mặc định khi đã căn giữa
SPEED_ALIGN    = 180   # Tốc độ xoay tại chỗ khi đang căn giữa (nhỏ hơn → mượt hơn)

# ── Ngưỡng căn giữa ──────────────────────────────────────────
ALIGN_DEAD_ZONE   = 90   # px: sai lệch cho phép khi xoay căn giữa
ALIGN_STABLE_TIME = 0.4  # giây: phải giữ giữa liên tục bao lâu mới chuyển sang FOLLOW
FOLLOW_DEAD_ZONE  = 50   # px: dead zone khi đang FOLLOW (rộng hơn để không lắc)

# ── Timing ───────────────────────────────────────────────────
SEND_INTERVAL  = 0.20   # giây (PHẢI < CMD_TIMEOUT_MS của ESP32 = 0.4s)
CMD_HOLD_TIME  = 0.12   # giây: giữ lệnh tối thiểu trước khi đổi
SMOOTH_WINDOW  = 5      # số frame lấy trung bình cx
FRAME_SKIP     = 2      # xử lý YOLO mỗi N frame

# ===================== KHỞI TẠO =====================
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')
print("OK")

print("Connecting to camera...")
cap = cv2.VideoCapture(CAMERA_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print("ERROR: Cannot open camera.")
    exit()
print("OK — Press 'q' to quit.")

# ── State machine ─────────────────────────────────────────────
# ALIGN : xe xoay tại chỗ cho đến khi mục tiêu vào giữa
# FOLLOW: mục tiêu đã căn giữa → tiến về phía mục tiêu + tiếp tục PID
STATE_ALIGN  = "ALIGN"
STATE_FOLLOW = "FOLLOW"
state = STATE_ALIGN

align_stable_since = None   # thời điểm cx bắt đầu nằm trong ALIGN_DEAD_ZONE

# ── PID ──────────────────────────────────────────────────────
pid_integral  = 0.0
pid_prev_err  = 0.0
pid_last_time = time.time()

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


def pid_update(error: float) -> float:
    """Trả về u ∈ [-1, 1]. Dương = lệch phải."""
    global pid_integral, pid_prev_err, pid_last_time
    now = time.time()
    dt  = max(now - pid_last_time, 1e-3)
    pid_last_time = now

    pid_integral += error * dt
    pid_integral  = clamp(pid_integral, -300, 300)
    derivative    = (error - pid_prev_err) / dt
    pid_prev_err  = error

    u = KP * error + KI * pid_integral + KD * derivative
    return clamp(u / 100.0, -1.0, 1.0)


def pid_reset():
    global pid_integral, pid_prev_err, pid_last_time
    pid_integral  = 0.0
    pid_prev_err  = 0.0
    pid_last_time = time.time()


def spin_command(err: int):
    """
    Xoay TẠI CHỖ (1 bánh tiến, 1 bánh lùi) để căn giữa.
    Tốc độ tỉ lệ với |err| nhưng giới hạn ở SPEED_ALIGN.
    Trả về (cmd, spd_l, spd_r).
    """
    ratio = clamp(abs(err) / 200.0, 0.3, 1.0)   # 0.3–1.0
    spd   = int(SPEED_ALIGN * ratio)
    spd   = clamp(spd, SPEED_MIN, SPEED_ALIGN)

    if err > 0:   # mục tiêu lệch phải → xoay phải: trái tiến, phải lùi
        return "right", spd, -spd
    else:         # mục tiêu lệch trái → xoay trái: trái lùi, phải tiến
        return "left", -spd, spd


def follow_command(cx_smooth: int, frame_w: int, area_ratio: float):
    """
    Differential drive khi đang FOLLOW.
    Trả về (cmd, spd_l, spd_r).
    """
    global pid_integral

    if area_ratio > STOP_AREA_THRESHOLD:
        return "stop", 0, 0

    if area_ratio < SAFE_AREA_THRESHOLD:
        return "forward", SPEED_MIN, SPEED_MIN

    err = cx_smooth - frame_w // 2

    if abs(err) <= FOLLOW_DEAD_ZONE:
        pid_integral = 0
        return "forward", SPEED_FWD, SPEED_FWD

    u     = pid_update(err)
    delta = int(abs(u) * (SPEED_MAX - SPEED_MIN))
    base  = SPEED_FWD

    if u > 0:   # lệch phải
        return "right", clamp(base + delta, SPEED_MIN, SPEED_MAX), clamp(base - delta, SPEED_MIN, SPEED_MAX)
    else:       # lệch trái
        return "left",  clamp(base - delta, SPEED_MIN, SPEED_MAX), clamp(base + delta, SPEED_MIN, SPEED_MAX)


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

    results   = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
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
                    # Bắt đầu đếm thời gian giữ giữa
                    if align_stable_since is None:
                        align_stable_since = now
                    elif now - align_stable_since >= ALIGN_STABLE_TIME:
                        # Đã giữ giữa đủ lâu → chuyển sang FOLLOW
                        state = STATE_FOLLOW
                        align_stable_since = None
                        pid_reset()
                        print("[STATE] ALIGN → FOLLOW")
                    # Vẫn đang trong dead zone, chờ thêm → dừng tại chỗ
                    cmd, spd_l, spd_r = "stop", 0, 0
                else:
                    # Chưa vào giữa → xoay tại chỗ
                    align_stable_since = None
                    cmd, spd_l, spd_r  = spin_command(err)

            else:  # STATE_FOLLOW
                if abs(err) > FOLLOW_DEAD_ZONE * 3:
                    # Lệch quá nhiều (mục tiêu di chuyển ngang mạnh)
                    # Quay về ALIGN để căn lại trước
                    state = STATE_ALIGN
                    align_stable_since = None
                    pid_reset()
                    cmd, spd_l, spd_r = spin_command(err)
                    print("[STATE] FOLLOW → ALIGN (lệch quá)")
                else:
                    cmd, spd_l, spd_r = follow_command(cx_smooth, frame.shape[1], area_ratio)

            # ── Debug overlay ────────────────────────────
            cv2.circle(annotated, (cx_smooth, (y1 + y2) // 2), 6, (0, 255, 255), -1)
            cv2.putText(annotated,
                        f"err={err:+d}  area={area_ratio:.2f}  [{state}]",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Vẽ vùng dead zone căn giữa
            fc = frame.shape[1] // 2
            cv2.line(annotated, (fc - ALIGN_DEAD_ZONE, 0),
                     (fc - ALIGN_DEAD_ZONE, frame.shape[0]), (0, 200, 255), 1)
            cv2.line(annotated, (fc + ALIGN_DEAD_ZONE, 0),
                     (fc + ALIGN_DEAD_ZONE, frame.shape[0]), (0, 200, 255), 1)
            break

    if not target_found:
        # Mất mục tiêu → dừng và quay về ALIGN để chờ
        cmd, spd_l, spd_r = "stop", 0, 0
        if state == STATE_FOLLOW:
            state = STATE_ALIGN
            align_stable_since = None
            pid_reset()
            print("[STATE] FOLLOW → ALIGN (mất mục tiêu)")

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
        print(f"[{state:6}] {confirmed_cmd:8s}  L={spd_l:+4d}  R={spd_r:+4d}")

    # ── HUD ─────────────────────────────────────────────────
    color_hud = (0, 255, 0) if state == STATE_FOLLOW else (0, 165, 255)
    cv2.putText(annotated,
                f"[{state}] {confirmed_cmd}  L={spd_l} R={spd_r}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_hud, 2)

    # Thanh progress căn giữa (chỉ hiện khi ALIGN)
    if state == STATE_ALIGN and target_found:
        fc   = frame.shape[1] // 2
        pct  = clamp(1.0 - abs(err) / (frame.shape[1] / 2), 0, 1)
        bar_w = int(200 * pct)
        cv2.rectangle(annotated, (10, 45), (210, 62), (50, 50, 50), -1)
        cv2.rectangle(annotated, (10, 45), (10 + bar_w, 62), (0, 165, 255), -1)
        cv2.putText(annotated, "Align", (215, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    cv2.imshow("Person Tracking", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Dừng xe khi thoát ───────────────────────────────────────
send_track("stop", 0, 0)
time.sleep(0.3)
cap.release()
cv2.destroyAllWindows()
print("Done.")
