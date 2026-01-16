# TEAM 52

"""
Uday Singh  ( 202351150 )
Sakasham Singh  ( 202351124 )
Hudad Harsh Ajaybhai (  202351048 )
"""

"""
Image Encoder–Decoder with Spatial Sampling and Intensity Quantization

This program demonstrates a simple image encoding and decoding system
based on:
1. Spatial resolution reduction (sampling)
2. Intensity resolution reduction (quantization)

The encoded image is stored in a custom binary format consisting of:
- A 4-bit header (2 bits for spatial resolution, 2 bits for intensity resolution)
- Raw quantized pixel data
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------
# SUPPORTED RESOLUTIONS
# --------------------------------------------------------------------
_SPATIAL_RESOLUTION_DIMENSIONS_ = [100, 200, 400, 800]
"""
List mapping spatial resolution index to output image size.
Index:
0 → 100×100
1 → 200×200
2 → 400×400
3 → 800×800
"""

RESOLUTION_INTENSITY_SELECTION = [1, 2, 4, 8]
"""
List mapping intensity resolution index to number of bits per pixel.
Index:
0 → 1 bit
1 → 2 bits
2 → 4 bits
3 → 8 bits
"""


# --------------------------------------------------------------------
# IMAGE INPUT AND PREPROCESSING
# --------------------------------------------------------------------
def read_grayscale_image(path):
    """
    Reads an image from disk and converts it to grayscale.
    """
    image = plt.imread(path)

    if image.ndim == 3:
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = image

    if gray.max() <= 1.0:
        gray = gray * 255.0

    return gray.astype(np.uint8)


def center_crop(image):
    """
    Crops the central square region of an image.
    """
    h, w = image.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return image[y0:y0 + side, x0:x0 + side]


# --------------------------------------------------------------------
# SPATIAL SAMPLING
# --------------------------------------------------------------------
def spatial_sample(image, out_size):
    """
    Performs spatial sampling using nearest-neighbor interpolation.

    """
    in_h, in_w = image.shape
    sampled = np.zeros((out_size, out_size), dtype=np.uint8)

    for i in range(out_size):
        for j in range(out_size):
            src_i = int(i * in_h / out_size)
            src_j = int(j * in_w / out_size)
            sampled[i, j] = image[src_i, src_j]

    return sampled


# --------------------------------------------------------------------
# INTENSITY QUANTIZATION
# --------------------------------------------------------------------
def quantize(image, bits):
    """
    Applies uniform intensity quantization.

    """
    levels = 2 ** bits
    quantized = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            normalized = image[i, j] / 255.0
            q = round(normalized * (levels - 1))
            quantized[i, j] = int(q * (255 / (levels - 1)))

    return quantized


# --------------------------------------------------------------------
# ENCODING UTILITIES
# --------------------------------------------------------------------
def make_header(spatial_idx, intensity_idx):
    """
    Creates a 4-bit header encoding spatial and intensity resolution.

    Header Format
    -------------
    [ spatial index (2 bits) | intensity index (2 bits) ]

    """
    return (spatial_idx << 2) | intensity_idx


def encode_image(img, spatial_idx, intensity_idx, filename):

    size = _SPATIAL_RESOLUTION_DIMENSIONS_[spatial_idx]
    bits = RESOLUTION_INTENSITY_SELECTION[intensity_idx]

    img = center_crop(img)
    img = spatial_sample(img, size)
    img = quantize(img, bits)

    header = make_header(spatial_idx, intensity_idx)

    with open(filename, "wb") as f:
        f.write(bytes([header]))
        f.write(img.tobytes())

    print("[ENCODER]")
    print(" Spatial Resolution:", size, "x", size)
    print(" Intensity Resolution:", bits, "bits")


# --------------------------------------------------------------------
# DECODER
# --------------------------------------------------------------------
def decode_image(filename):
    with open(filename, "rb") as f:

        header = f.read(1)[0]

        spatial_idx = (header >> 2) & 0b11
        intensity_idx = header & 0b11

        size = _SPATIAL_RESOLUTION_DIMENSIONS_[spatial_idx]
        img_data = f.read()

    image = np.frombuffer(img_data, dtype=np.uint8)
    image = image.reshape((size, size))

    return image, spatial_idx, intensity_idx


# --------------------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python image_encoder_decoder.py <input_image>")
        return

    img = read_grayscale_image(sys.argv[1])

    print("\nSelect Spatial Resolution:")
    print(" 0 → 100×100")
    print(" 1 → 200×200")
    print(" 2 → 400×400")
    print(" 3 → 800×800")
    spatial_idx = int(input("Enter choice (0–3): "))

    print("\nSelect Intensity Resolution:")
    print(" 0 → 1 bit")
    print(" 1 → 2 bits")
    print(" 2 → 4 bits")
    print(" 3 → 8 bits")
    intensity_idx = int(input("Enter choice (0–3): "))

    filename = "encoded_image.bin"

    encode_image(img, spatial_idx, intensity_idx, filename)

    decoded, s_idx, i_idx = decode_image(filename)

    plt.imshow(decoded, cmap="gray")
    plt.title(
        f"Decoded Image\n"
        f"{_SPATIAL_RESOLUTION_DIMENSIONS_[s_idx]}×{_SPATIAL_RESOLUTION_DIMENSIONS_[s_idx]}, "
        f"{RESOLUTION_INTENSITY_SELECTION[i_idx]} bits"
    )
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()


