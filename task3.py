import cv2
import numpy as np

template = cv2.imread('variant-9.png', cv2.IMREAD_GRAYSCALE)
if template is None:
    print("无法读取标记图像")
    exit()

orb = cv2.ORB_create()
kp_template, des_template = orb.detectAndCompute(template, None)
if des_template is None:
    print("模板特征点太少")
    exit()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

cap = cv2.VideoCapture(0)

# 用于存储所有帧中心坐标的列表
centers = []

print("跟踪程序已启动（含平均坐标记录），按 'q' 退出...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    if des_frame is not None and len(des_frame) > 0:
        matches = bf.match(des_template, des_frame)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 10:
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                h, w = template.shape
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)
                center = np.mean(dst, axis=0).flatten().astype(int)
                cx, cy = center[0], center[1]

                # 存储当前帧的中心坐标（选项9的修改）
                centers.append((cx, cy))

                # 绘制边框和中心点
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({cx}, {cy})", (cx+20, cy-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('Marker Tracking (Step 3 with average)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 计算并输出平均坐标（选项9的修改）
if centers:
    avg_x = sum(p[0] for p in centers) / len(centers)
    avg_y = sum(p[1] for p in centers) / len(centers)
    print(f"会话期间标记的平均坐标: ({avg_x:.2f}, {avg_y:.2f})")
    print(f"共记录 {len(centers)} 帧")
else:
    print("未检测到任何标记")