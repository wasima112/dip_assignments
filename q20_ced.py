import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import deque

def main():
    img = cv2.imread("/home/wasima/Documents/lenna.png", cv2.IMREAD_GRAYSCALE)
    canny_edge_detection(img, filtered=True)
    canny_edge_detection(img, filtered=False)


def canny_edge_detection(img, filtered):
    if filtered:
        img_filtered = noise_reduction(img, kernel_size = 7)
    else:
        img_filtered = img

    # Step 2: Gradient calculation
    Gx, Gy, G, theta = gradient_estimation(img_filtered)

    # Step 3: Non-maximum suppression
    nms = non_maximum_suppression(G, theta)

    # Step 4: Double thresholding (adaptive ratios)
    thresh, weak, strong = threshold(nms, low_ratio=0.1, high_ratio=0.3)

    # Step 5: Hysteresis (iterative)
    edges = hysteresis(thresh, weak, strong)

    if filtered:
        img_set = [
            img,
            img_filtered,
            cv2.convertScaleAbs(G),
            nms,
            cv2.convertScaleAbs(Gx),
            cv2.convertScaleAbs(Gy),
            thresh,
            edges
        ]
        titles = ['Original', 'Img_filtered', 'Gradient', 'NMS', 'Gx', 'Gy', 'double-thresholding', 'edges-with-filter']
    else:
        img_set = [
            img,
            cv2.convertScaleAbs(Gx),
            nms,
            thresh,
            cv2.convertScaleAbs(Gy),
            cv2.convertScaleAbs(G),
            edges
        ]
        titles = ['Original', 'Gx', 'NMS', 'Double-thresholding', 'Gy', 'Gradient', 'edges-without-filter']
    display(img_set, titles)


# Step 1: Noise reduction
def noise_reduction(img, kernel_size=5, sigma=1.4):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


# Step 2: Gradient estimation
def gradient_estimation(img):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]], dtype=np.float32)

    Gx = cv2.filter2D(img.astype(np.float32), -1, sobel_x)
    Gy = cv2.filter2D(img.astype(np.float32), -1, sobel_y)

    G = np.hypot(Gx, Gy)  # float magnitude
    theta = np.arctan2(Gy, Gx)

    return Gx, Gy, G, theta


# Step 3: Non-maximum suppression
def non_maximum_suppression(G, theta):
    h, w = G.shape
    Z = np.zeros((h, w), dtype=np.float32)
    angle = theta * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, h-1):
        for j in range(1, w-1):
            q = r = 0
            a = angle[i, j]

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = G[i, j+1], G[i, j-1]
            elif (22.5 <= a < 67.5):
                q, r = G[i-1, j+1], G[i+1, j-1]
            elif (67.5 <= a < 112.5):
                q, r = G[i-1, j], G[i+1, j]
            elif (112.5 <= a < 157.5):
                q, r = G[i-1, j-1], G[i+1, j+1]

            if (G[i, j] >= q) and (G[i, j] >= r):
                Z[i, j] = G[i, j]
            else:
                Z[i, j] = 0

    return cv2.convertScaleAbs(Z)


# Step 4: Double thresholding
def threshold(img, low_ratio=0.1, high_ratio=0.3):
    high = img.max() * high_ratio
    low = high * low_ratio
    strong, weak = 255, 75

    res = np.zeros_like(img, dtype=np.uint8)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img >= low) & (img < high))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong


# Step 5: Iterative hysteresis
def hysteresis(img, weak, strong=255):
    M, N = img.shape
    edges = img.copy()
    q = deque()

    # Initialize queue with strong edges
    strong_i, strong_j = np.where(edges == strong)
    for i, j in zip(strong_i, strong_j):
        q.append((i, j))

    # BFS propagation
    while q:
        i, j = q.popleft()
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < M and 0 <= nj < N:
                    if edges[ni, nj] == weak:
                        edges[ni, nj] = strong
                        q.append((ni, nj))

    # Remove remaining weak edges
    edges[edges != strong] = 0
    return edges


# Display function
def display(img_set, titles):
    plt.figure(figsize=(16, 12))
    for i in range(len(img_set)):
        plt.subplot(2, 4, i+1)
        plt.imshow(img_set[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
