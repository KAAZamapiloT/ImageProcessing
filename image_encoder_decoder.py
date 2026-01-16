import sys

import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------
# CONFIG TABLES
# -------------------------------------------------
SPATIAL_SIZES = [100, 200, 400, 800]  # indices 0–3
INTENSITY_BITS = [1, 2, 4, 8]  # indices 0–3


# -------------------------------------------------
# IMAGE INPUT + PREPROCESSING
# -------------------------------------------------
def read_image_gray(path):
    """
    Read image and convert to grayscale manually.
    Uses matplotlib only for raw image loading.
    """
    img = plt.imread(path)

    # If image is RGB → convert to grayscale manually
    if img.ndim == 3:
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]
        img = 0.299 * r + 0.587 * g + 0.114 * b

    # Normalize to 0–255 uint8
    if img.max() <= 1.0:
        img = img * 255.0

    return img.astype(np.uint8)


def center_crop(img):
    """Crop central square region manually"""
    h, w = img.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return img[y0 : y0 + side, x0 : x0 + side]


# -------------------------------------------------
# SPATIAL SAMPLING (MANUAL)
# -------------------------------------------------
def spatial_sample(img, out_size):
    """
    Nearest-neighbor sampling implemented manually
    """
    in_h, in_w = img.shape
    out = np.zeros((out_size, out_size), dtype=np.uint8)

    for i in range(out_size):
        for j in range(out_size):
            src_x = int(i * in_h / out_size)
            src_y = int(j * in_w / out_size)
            out[i, j] = img[src_x, src_y]

    return out


# -------------------------------------------------
# INTENSITY QUANTIZATION (MANUAL)
# -------------------------------------------------
def quantize(img, bits):
    """
    Uniform intensity quantization
    """
    levels = 2**bits
    out = np.zeros_like(img, dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            norm = img[i, j] / 255.0
            q = round(norm * (levels - 1))
            out[i, j] = int(q * (255 / (levels - 1)))

    return out


# -------------------------------------------------
# ENCODER
# -------------------------------------------------
def make_header(spatial_idx, intensity_idx):
    """
    4-bit header:
    [ spatial(2 bits) | intensity(2 bits) ]
    """
    return (spatial_idx << 2) | intensity_idx


def encode_image(img, spatial_idx, intensity_idx, filename):
    size = SPATIAL_SIZES[spatial_idx]
    bits = INTENSITY_BITS[intensity_idx]

    img = center_crop(img)
    img = spatial_sample(img, size)
    img = quantize(img, bits)

    header = make_header(spatial_idx, intensity_idx)

    with open(filename, "wb") as f:
        f.write(bytes([header]))
        f.write(img.tobytes())

    print("[ENCODER]")
    print(" Resolution:", size, "x", size)
    print(" Intensity:", bits, "bits")


# -------------------------------------------------
# DECODER
# -------------------------------------------------
def decode_image(filename):
    with open(filename, "rb") as f:
        header = f.read(1)[0]

        spatial_idx = (header >> 2) & 0b11
        intensity_idx = header & 0b11

        size = SPATIAL_SIZES[spatial_idx]
        img_data = f.read()

    img = np.frombuffer(img_data, dtype=np.uint8)
    img = img.reshape((size, size))

    return img, spatial_idx, intensity_idx


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python image_encoder_decoder.py input_image")
        return

    img = read_image_gray(sys.argv[1])

    print("\nSelect Spatial Resolution:")
    print(" 0 → 100×100")
    print(" 1 → 200×200")
    print(" 2 → 400×400")
    print(" 3 → 800×800")
    spatial_idx = int(input("Enter choice (0–3): "))

    print("\nSelect Intensity Resolution:")
    print(" 0 → 1 bit (2 levels)")
    print(" 1 → 2 bits (4 levels)")
    print(" 2 → 4 bits (16 levels)")
    print(" 3 → 8 bits (256 levels)")
    intensity_idx = int(input("Enter choice (0–3): "))

    filename = "encoded_image.bin"

    encode_image(img, spatial_idx, intensity_idx, filename)

    decoded, s_idx, i_idx = decode_image(filename)

    plt.imshow(decoded, cmap="gray")
    plt.title(
        f"Decoded Image\n"
        f"{SPATIAL_SIZES[s_idx]}×{SPATIAL_SIZES[s_idx]}, "
        f"{INTENSITY_BITS[i_idx]} bits"
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
