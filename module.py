import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

# transform the type of the image to float32
def image_int2float(image):
    if image is not None:
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
    return image

def image_float2int(image):
    if image is not None:
        if image.dtype == np.float32:
            image = (image * 255).astype(np.uint8)
    return image


def gaussian_kernel(size):
    # create the kernel
    kernel = np.zeros((size, size))
    
    # calculate the center of the kernel
    center = math.floor(size / 2) - 1
    
    sigma = 0.3 * ((size - 1) / 2 - 1) + 0.8
    
    # fill in the kernel
    for x in range(size):
        for y in range(size):
            kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    
    # normalize the kernel
    kernel /= np.sum(kernel)
    
    return kernel

def cross_correlation_2d(image, kernel):
    # transform the type of the image to float32
    image_float = image_int2float(image)

    # get the size of the image
    height, width = image.shape[0], image.shape[1]

    # get the size of the kernel
    kernel_size = kernel.shape[0]

    # calculate the padding size
    padding = math.ceil((kernel_size - 1) / 2)  

    # padding the input image
    image_padded = np.pad(image_float, ((padding, padding), (padding, padding), (0, 0)), mode='edge')

    # calculate the cross_correlation
    new_image = np.zeros((height, width, 3))

    for k in range(3):
        for i in range(0, height):
            for j in range(0, width):
                new_image[i, j, k] = np.sum(image_padded[i:i+kernel_size, j:j+kernel_size, k] * kernel)

    return new_image

def get_laplacian(image, kernel):
    # transform the type of the image to float32
    image_float = image_int2float(image)
    
    # get the low pass image
    low_pass_image = cross_correlation_2d(image, kernel)

    # get the high pass image
    high_pass_image = np.where(image_float >= low_pass_image, image_float - low_pass_image, 0)

    high_pass_image = image_add(image_float, high_pass_image, 1, 0.5)

    return high_pass_image

def image_add(image1, image2, a1, a2):
    
    fusion = np.where((1 - a1 * image1) < a2 * image2, 1, a1 * image1 + a2 * image2)
    
    return fusion

def image_resize(image1,image2):
    height = min(image1.shape[0], image2.shape[0])
    width = min(image1.shape[1], image2.shape[1])

    image1 = cv2.resize(image1, (height, width))
    image2 = cv2.resize(image2, (height, width))

    return image1, image2

def hybrid_image(image1, image2, kernel, a1, a2):
    image1, image2 = image_resize(image1, image2)

    low_pass = cross_correlation_2d(image1, kernel)

    laplacian = get_laplacian(image2, kernel)
    
    hybrid = image_add(low_pass, laplacian, a1, a2)

    low_pass_int = image_float2int(low_pass)
    laplacian_int = image_float2int(laplacian)
    hybrid_int = image_float2int(hybrid)
    
    return low_pass_int, laplacian_int, hybrid_int