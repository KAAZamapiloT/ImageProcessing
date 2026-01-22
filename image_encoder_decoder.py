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
#READS IMAGE AND CONVERTS IT TO GRAYSCALE IMAGE
def read_grayscale_image(filepath):

    image = plt.imread(filepath)
    if image.ndim==3:
        r=image[:,:,0]
        g=image[:,:,1]
        b=image[:,:,2]

        gray = (0.299 * r+ 0.587 * g + 0.114 * b)
    else:
        gray = image
    if gray.max() <= 1.0:
        gray = gray * 255.0

    return gray.astype(np.uint8)

#CROPS CENTRAL SQUARE REGION OF IMAGE
def center_crop(image):
    h,w=image.shape

    side =min(h,w)
    y=(h-side)//2
    x=(w-side)//2

    return image[y:y+side,x:x+side]

# --------------------------------------------------------------------
# SPATIAL SAMPLING
# --------------------------------------------------------------------

# performs spatial sampling using nearest neighbour interpolation
def spatial_sample(image,output_size):

    input_height, input_width = image.shape

    sampled=np.zeros((output_size, output_size), dtype=np.uint8)

    for i in range(output_size):
        for j in range(output_size):
            source_i = int(i * input_width / output_size)
            source_j = int(j * input_height / output_size)
            sampled[i, j] = image[source_i, source_j]

    return sampled
# --------------------------------------------------------------------
# INTENSITY QUANTIZATION
# --------------------------------------------------------------------
# Applies uniform intensity quantization
def quantize(image, bits):
    levels = 2 ** bits
    quantized = np.zeros_like(image, dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            normalized=image[i, j]/255.0

            q=round(normalized*(levels-1))

            quantized[i, j]=int(q * (255/(levels-1)))

    return quantized


# --------------------------------------------------------------------
# ENCODING UTILITIES
# --------------------------------------------------------------------

#creates a 4-bit custom header with format [spatial index | intesnsity index]
def make_header(spatial_index,intesity_index):
    return(spatial_index<<2)|intesity_index

def encode_image(img, spatial_idx, intensity_idx, filename):
    img = center_crop(img)
    img = spatial_sample(img, _SPATIAL_RESOLUTION_DIMENSIONS_[spatial_idx])
    img = quantize(img, RESOLUTION_INTENSITY_SELECTION[intensity_idx])

    header = (spatial_idx << 2) | intensity_idx

    with open(filename, "wb") as f:
        f.write(bytes([header]))
        f.write(img.tobytes())


def decode_image(filename):
    with open(filename, "rb") as f:
        header = f.read(1)[0]
        size = _SPATIAL_RESOLUTION_DIMENSIONS_[(header >> 2) & 3]
        data = f.read()

    image = np.frombuffer(data, dtype=np.uint8).reshape((size, size))
    return image



# --------------------------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python image_encoder_decoder.py <input_image>")
        return

    # Read input image
    img = read_grayscale_image(sys.argv[1])

    spatial_idx = int(input(
        "Select Spatial Resolution (0:100, 1:200, 2:400, 3:800): "
    ))

    intensity_idx = int(input(
        "Select Intensity Resolution (0:1bit, 1:2bit, 2:4bit, 3:8bit): "
    ))

    filename = "encoded_image.bin"

    # Encode and decode
    encode_image(img, spatial_idx, intensity_idx, filename)
    decoded = decode_image(filename)

    # -------------------------------------------------
    # VISUALIZATION (Matplotlib ONLY)
    # -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(center_crop(img), cmap="gray")
    axes[0].set_title("Original (Cropped)")
    axes[0].axis("off")

    axes[1].imshow(decoded, cmap="gray")
    axes[1].set_title(
        f"Decoded Image\n"
        f"{_SPATIAL_RESOLUTION_DIMENSIONS_[spatial_idx]}×"
        f"{_SPATIAL_RESOLUTION_DIMENSIONS_[spatial_idx]}, "
        f"{RESOLUTION_INTENSITY_SELECTION[intensity_idx]} bits"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

