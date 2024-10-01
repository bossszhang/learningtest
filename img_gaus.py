import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0, std=25):
    """给图像添加高斯噪声"""
    # 将图像转换为 NumPy 数组
    image_array = np.array(image)

    # 生成与图像同样大小的高斯噪声
    noise = np.random.normal(mean, std, image_array.shape).astype(np.uint8)

    # 将噪声添加到原图像
    noisy_image_array = image_array + noise

    # 确保像素值在合法范围内 [0, 255]
    noisy_image_array = np.clip(noisy_image_array, 0, 255)

    # 返回带噪声的图像
    return Image.fromarray(noisy_image_array.astype(np.uint8))


# 读取图像并转换为 RGB
image_path = r'D:\zhangergao\pythonProject\mylearningtest\image\test1.png'  # 替换为你的图像路径
original_image = Image.open(image_path).convert('RGB')

# 添加高斯噪声
noisy_image = add_gaussian_noise(original_image)

# 显示原图和结果图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Image with Gaussian Noise")
plt.imshow(noisy_image)
plt.axis('off')

plt.show()
