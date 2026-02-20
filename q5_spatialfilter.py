import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. KERNEL DEFINITIONS
# ============================================================

def get_kernels():
    # Average kernel
    avg = np.ones((3, 3)) / 9

    # Sobel
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    # Prewitt
    prewitt_x = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])

    prewitt_y = np.array([[-1, -1, -1],
                          [0, 0, 0],
                          [1, 1, 1]])

    # Laplace
    laplace = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]])

    return {
        "Average": avg,
        "Sobel X": sobel_x,
        "Sobel Y": sobel_y,
        "Prewitt X": prewitt_x,
        "Prewitt Y": prewitt_y,
        "Laplace": laplace,
    }

# ============================================================
# 2. APPLY OPENCV FILTERS
# ============================================================

def apply_opencv_filters(image, kernels):
    results = {}
    for name, kernel in kernels.items():
        filtered = cv2.filter2D(image, -1, kernel)
        results[name] = filtered
    return results

# ============================================================
# 3. DISPLAY & SAVE RESULTS
# ============================================================

def show_results(title, images_dict, original):
    os.makedirs("output_images", exist_ok=True)
    total = len(images_dict) + 1
    cols = 4
    rows = int(np.ceil(total / cols))

    plt.figure(figsize=(12, 3 * rows))
    plt.subplot(rows, cols, 1)
    plt.imshow(original, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    for i, (name, img) in enumerate(images_dict.items(), start=2):
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap="gray")
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"output_images/{title}.png")
    plt.close()
    print(f"{title} saved.")

# ============================================================
# 4. MAIN FUNCTION
# ============================================================

def main():
    image = cv2.imread("/home/wasima/Documents/lenna.png", cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found.")
        return

    kernels = get_kernels()
    results = apply_opencv_filters(image, kernels)
    show_results("spatial_filtering", results, image)
    print("\nSpatial filtering experiment completed!")

# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
