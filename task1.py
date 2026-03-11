import cv2
import numpy as np

# 读取图像
img = cv2.imread('variant-9.png')
if img is None:
    print("无法读取图像，请检查路径")
    exit()

# 构建高斯金字塔，将每一层保存到列表中
pyramid = [img]
while True:
    # 下采样：尺寸减半
    down = cv2.pyrDown(pyramid[-1])
    if down.shape[0] < 30 or down.shape[1] < 30:  # 设置一个最小尺寸阈值
        break
    pyramid.append(down)

# 显示金字塔：可以将所有层水平拼接（需要调整高度一致）
# 先将所有层缩放到相同高度（例如第一层的高度）
base_h = pyramid[0].shape[0]
resized_layers = []
for layer in pyramid:
    # 按比例缩放高度到 base_h
    h, w = layer.shape[:2]
    scale = base_h / h
    new_w = int(w * scale)
    resized = cv2.resize(layer, (new_w, base_h))
    resized_layers.append(resized)

# 水平拼接
combined = np.hstack(resized_layers)

# 显示
cv2.imshow('Image Pyramid', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()