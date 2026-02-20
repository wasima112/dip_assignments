import cv2
import numpy as np

# Function to create histogram image
def create_hist_image(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)

    hist_img = np.zeros((300, 256), dtype=np.uint8)
    for x in range(1, 256):
        cv2.line(
            hist_img,
            (x - 1, 300 - int(hist[x - 1][0])),
            (x, 300 - int(hist[x][0])),
            255
        )
    return cv2.cvtColor(hist_img, cv2.COLOR_GRAY2BGR)


# Function for gamma correction (Nonlinear)
def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")
    return cv2.LUT(image, table)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Original Color ---
    color_frame = frame.copy()
    color_gray_for_hist = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
    hist_color = create_hist_image(color_gray_for_hist)
    color_panel = np.vstack((
        color_frame,
        cv2.resize(hist_color, (color_frame.shape[1], 300))
    ))

    # --- Linear : Grayscale ---
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    hist_gray = create_hist_image(gray_frame)
    gray_panel = np.vstack((
        gray_bgr,
        cv2.resize(hist_gray, (gray_bgr.shape[1], 300))
    ))

    # --- Nonlinear : Gamma Correction ---
    gamma_frame = gamma_correction(gray_frame, gamma=2.0)  # nonlinear mapping
    gamma_bgr = cv2.cvtColor(gamma_frame, cv2.COLOR_GRAY2BGR)
    hist_gamma = create_hist_image(gamma_frame)
    gamma_panel = np.vstack((
        gamma_bgr,
        cv2.resize(hist_gamma, (gamma_bgr.shape[1], 300))
    ))

    # Match all heights
    height = color_panel.shape[0]
    gray_panel = cv2.resize(gray_panel, (color_panel.shape[1], height))
    gamma_panel = cv2.resize(gamma_panel, (color_panel.shape[1], height))

    # Combine horizontally
    combined = np.hstack((color_panel, gray_panel, gamma_panel))

    cv2.imshow(
        "Original | Linear (Gray) | Nonlinear (Gamma) with Histograms",
        combined
    )

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
