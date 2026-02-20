import cv2
import numpy as np
import matplotlib.pyplot as plt

#================= Image I/O ======================
def load_grayscale_image(path, target_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, target_size)

#================= Patching =======================
def split_into_tiles(img, s):
    h, w = img.shape
    tile_h, tile_w = h // s, w // s
    tiles = []
    for i in range(s):
        for j in range(s):
            y0, y1 = i * tile_h, (i + 1) * tile_h
            x0, x1 = j * tile_w, (j + 1) * tile_w
            tiles.append(img[y0:y1, x0:x1])
    return tiles, tile_h, tile_w

def reconstruct_from_tiles(tiles, s, tile_h, tile_w):
    h, w = s * tile_h, s * tile_w
    reconstructed = np.zeros((h, w), dtype=np.uint8)
    idx = 0
    for i in range(s):
        for j in range(s):
            y0, y1 = i * tile_h, (i + 1) * tile_h
            x0, x1 = j * tile_w, (j + 1) * tile_w
            reconstructed[y0:y1, x0:x1] = tiles[idx]
            idx += 1
    return reconstructed

#================= Equalization ===================
def histogram(img):
    hist = np.zeros(256, dtype=int)
    for value in img.ravel():
        hist[value] += 1
    return hist

def pdf(hist):
    return hist / hist.sum()

def cdf(pdf):
    return np.cumsum(pdf)

def apply_equalization(img):
    hist_vals = histogram(img)
    pdf_vals = pdf(hist_vals)
    cdf_vals = cdf(pdf_vals)
    mapping = np.round(cdf_vals * 255).astype(np.uint8)
    return mapping[img]

def equalize_tile(tile):
    return apply_equalization(tile)

#================= Built-in AHE and CLAHE ===================
def apply_builtin_ahe(img_gray, clip_limit=40.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)

def apply_builtin_clahe(img_gray, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_gray)

def apply_clahe_variants(img_gray, clip_limits, tile_grid_size=(8, 8)):
    variants = []
    for clip in clip_limits:
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(img_gray)
        variants.append(enhanced)
    return variants

#================= Visualization ===================
def display_tiles(tiles, titles):
    n = len(tiles)
    cols = 2
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4 * cols, 3 * rows))
    for i, tile in enumerate(tiles):
        plt.subplot(rows, cols, i + 1)
        if(tile.ndim == 2):   # image
            
            plt.imshow(tile, cmap='gray')
            plt.title(titles[i])
            plt.axis('off')
        else: # histogram
            plt.bar(range(256), tile)
            plt.title(titles[i])
            plt.xlim([0, 255])
            plt.xlabel('Intensity Value')
            plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

#================= Main Workflow ===================
def main():
    img_path = "/home/wasima/Documents/lenna.png"
    target_size = (2000, 2000)
    s = 2

    img_gray = load_grayscale_image(img_path, target_size)

    # Manual AHE via tile-wise HE
    tiles, tile_h, tile_w = split_into_tiles(img_gray, s)
    equalized_tiles = [equalize_tile(tile) for tile in tiles]
    manual_ahe = reconstruct_from_tiles(equalized_tiles, s, tile_h, tile_w)

    display_tiles(tiles, [f'Original Tile {i+1}' for i in range(len(tiles))])

    operations = ['Linear', 'Gamma-correction', 'equalized_tiles']

    # Apply linear and non-linear operations on tiles
    for op in operations:
        processed_tiles = []
        
        for tile in tiles:
            if op == 'Linear':
                processed_tiles.append( cv2.add(tile, 20))
            elif op == 'Gamma-correction':
                gamma_corrected = np.clip(255 * (tile / 255) ** 0.5, 0, 255).astype(np.uint8)
                processed_tiles.append(gamma_corrected)
            elif op == 'equalized_tiles':
                equalized_tile = equalize_tile(tile)
                processed_tiles.append(equalized_tile)

            


        display_tiles(processed_tiles, [f'{op} Tile {i+1}' for i in range(len(processed_tiles))])

      

    # CLAHE with different clip limits
    clip_values = [40.0, 1.0, 2.0, 5.0, 10.0]
    clahe_variants = apply_clahe_variants(img_gray, clip_values)

    # histogram for all images (original + AHE + CLAHE variants)
    histograms = []

    for img in [img_gray, manual_ahe, *clahe_variants]:
        hist = histogram(img)
        histograms.append(hist)
        


    # Rename last variant to "AHE (with bilinear)"
    titles = ['Original', 'AHE'] + ['AHE (with bilinear)'] + [f'CLAHE clip={c}' for c in clip_values[1:]] 

    display_tiles([img_gray, manual_ahe] + clahe_variants, titles)
    display_tiles(histograms, [t + ' Hist' for t in titles])


if __name__ == '__main__':
    main()
