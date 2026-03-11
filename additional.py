import cv2
import numpy as np

# 读取标记图像（模板）
template = cv2.imread('variant-9.png', cv2.IMREAD_GRAYSCALE)
if template is None:
    print("无法读取标记图像，请检查路径")
    exit()

# 读取苍蝇图像（带透明通道）
fly = cv2.imread('fly64.png', cv2.IMREAD_UNCHANGED)
if fly is None:
    print("无法读取苍蝇图像 fly64.png")
    exit()

# 获取苍蝇图像的尺寸和通道数
fly_h, fly_w = fly.shape[:2]
if fly.shape[2] == 4:
    # 分离BGR和Alpha通道
    fly_bgr = fly[:, :, :3]
    fly_alpha = fly[:, :, 3] / 255.0  # 归一化到[0,1]
else:
    # 如果没有Alpha通道，直接使用图像，并创建全不透明掩码
    fly_bgr = fly
    fly_alpha = np.ones((fly_h, fly_w), dtype=np.float32)

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

print("跟踪程序 + 苍蝇叠加已启动，按 'q' 退出...")

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
                # 计算标记中心
                center = np.mean(dst, axis=0).flatten().astype(int)
                cx, cy = center[0], center[1]

                # 绘制标记边框（绿色）
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

                # 计算苍蝇图像的放置位置（使苍蝇中心与标记中心重合）
                x = cx - fly_w // 2
                y = cy - fly_h // 2

                # 确保区域在画面内
                if x >= 0 and y >= 0 and x + fly_w <= frame.shape[1] and y + fly_h <= frame.shape[0]:
                    # 提取画面上要放置苍蝇的ROI
                    roi = frame[y:y+fly_h, x:x+fly_w]
                    # 如果苍蝇有透明通道，进行alpha混合
                    if fly_alpha is not None:
                        # 将fly_bgr和roi按alpha混合
                        blended = (fly_bgr * fly_alpha[:, :, np.newaxis] + 
                                   roi * (1 - fly_alpha[:, :, np.newaxis])).astype(np.uint8)
                        frame[y:y+fly_h, x:x+fly_w] = blended
                    else:
                        # 无透明通道，直接覆盖
                        frame[y:y+fly_h, x:x+fly_w] = fly_bgr
                else:
                    # 如果苍蝇超出边界，可以选择不画或缩小（这里简单忽略）
                    pass

    cv2.imshow('Marker Tracking with Fly', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()