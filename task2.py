import cv2
import numpy as np

# 读取标记图像（模板）
template = cv2.imread('variant-9.png', cv2.IMREAD_GRAYSCALE)
if template is None:
    print("无法读取标记图像，请检查路径")
    exit()

# 初始化ORB
orb = cv2.ORB_create()
kp_template, des_template = orb.detectAndCompute(template, None)
if des_template is None:
    print("模板图像特征点太少")
    exit()

# 匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 打开摄像头
cap = cv2.VideoCapture(0)

print("跟踪程序已启动，按 'q' 退出...")

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
                # 计算中心点
                center = np.mean(dst, axis=0).flatten().astype(int)
                cx, cy = center[0], center[1]

                # 绘制边框和中心点
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"({cx}, {cy})", (cx+20, cy-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow('Marker Tracking (Step 2)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()