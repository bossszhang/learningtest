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
image_path = r'D:\zhangergao\pythonProject\mylearningtest\image\image0.jpg'
reflect0_path = r'D:\zhangergao\pythonProject\mylearningtest\image\reflect0_0.jpg'
reflect1_path = r'D:\zhangergao\pythonProject\mylearningtest\image\reflect0_1.jpg'
reflect2_path = r'D:\zhangergao\pythonProject\mylearningtest\image\reflect0_2.jpg'

image = Image.open(image_path).convert('RGB')
reflect0 = Image.open(reflect0_path).convert('RGB')
reflect1 = Image.open(reflect1_path).convert('RGB')
reflect2 = Image.open(reflect2_path).convert('RGB')


# 将图片转换为NumPy数组
image_np = np.array(image, dtype=np.float32)
reflect0_np = np.array(reflect0, dtype=np.float32)
reflect1_np = np.array(reflect1, dtype=np.float32)
reflect2_np = np.array(reflect2, dtype=np.float32)

# 两张图片相减，取绝对值（防止负值）
diff_np_0 = np.abs(reflect0_np - image_np)
diff_np_1 = np.abs(reflect1_np - image_np)
diff_np_2 = np.abs(reflect2_np - image_np)

# 将差异结果归一化到0-255，并转换为uint8类型
diff_np_0 = np.clip(diff_np_0, 0, 255).astype(np.uint8)
diff_np_1 = np.clip(diff_np_1, 0, 255).astype(np.uint8)
diff_np_2 = np.clip(diff_np_2, 0, 255).astype(np.uint8)

# # 将结果转换回PIL Image并显示
diff_img_0 = Image.fromarray(diff_np_0)
diff_img_1 = Image.fromarray(diff_np_1)
diff_img_2 = Image.fromarray(diff_np_2)
# diff_img.show()

# 使用matplotlib显示原图、去除阴影的图像和差异图
plt.figure(figsize=(10, 5))

# plt.subplot(3, 3, 1)
# plt.imshow(image)
# plt.title("Image")
# plt.subplot(3, 3, 2)
# plt.imshow(reflect0)
# plt.title("Reflect0")
# plt.subplot(3, 3, 3)
# plt.imshow(diff_img_0)
# plt.title("Difference_0 (Shadow)")
# plt.subplot(3, 3, 4)
# plt.imshow(image)
# plt.title("Image")
# plt.subplot(3, 3, 5)
# plt.imshow(reflect1)
# plt.title("Reflect1")
# plt.subplot(3, 3, 6)
# plt.imshow(diff_img_1)
# plt.title("Difference_1 (Shadow)")
# plt.subplot(3, 3, 7)
# plt.imshow(image)
# plt.title("Image")
# plt.subplot(3, 3, 8)
# plt.imshow(reflect2)
# plt.title("Reflect2")
# plt.subplot(3, 3, 9)
# plt.imshow(diff_img_2)
# plt.title("Difference_2 (Shadow)")

plt.subplot(1, 3, 1)
plt.imshow(diff_img_0)
plt.title("Difference_0 (Shadow)")
plt.subplot(1, 3, 2)
plt.imshow(diff_img_1)
plt.title("Difference_1 (Shadow)")
plt.subplot(1, 3, 3)
plt.imshow(diff_img_2)
plt.title("Difference_2 (Shadow)")

plt.tight_layout()  # 自动调整子图间距
plt.show()







# # 添加高斯噪声
# noisy_image = add_gaussian_noise(original_image)
#
# # 显示原图和结果图
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(original_image)
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.title("Image with Gaussian Noise")
# plt.imshow(noisy_image)
# plt.axis('off')
#
# plt.show()
