import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread(r'D:\zhangergao\pythonProject\mylearningtest\image\test1.png', cv2.IMREAD_GRAYSCALE)

# 计算梯度
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # 水平梯度
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # 垂直梯度

# 计算梯度幅度
gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
gradient_magnitude = np.uint8(gradient_magnitude)  # 转换为 uint8 类型以便显示

# 使用 matplotlib 显示原始图像和梯度图
plt.figure(figsize=(10, 5))

# 原始图像
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 梯度图
plt.subplot(1, 2, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off')

plt.show()
