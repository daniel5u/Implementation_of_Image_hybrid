from module import hybrid_image, gaussian_kernel, image_float2int
import numpy as np
import matplotlib.pyplot as plt
import cv2

def save_image(path, img):
    """
    Save the image to the specified path.
    """
    try:
        cv2.imwrite(path, img * 255)
        print(f"Picture saved to: {path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def load_image(path):
    """
    Load an image from the specified path and convert it to a numpy array.
    """
    try:
        img = cv2.imread(path)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
def main():
    # Read Picture
    img1_path = 'images/einstein.jpg'
    img2_path = 'images/marilynn.jpg'

    # Load the images
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)

    # Check whether the picture is loaded successfully
    if img1 is None or img2 is None:
        print("Image reading failed, please check the path and file name!")
        return

    #define the kernel
    kernel = gaussian_kernel(15)

    #generate the requied images
    low_freq, laplacian, hybrid = hybrid_image(img1, img2, kernel, 0.8, 0.3)


    # Save the requied images
    save_image('images/low_freq.jpg', low_freq)
    save_image('images/laplacian.jpg', laplacian)
    save_image('images/hybrid.jpg', hybrid)

if __name__ == "__main__":
    main()