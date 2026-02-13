# 🧪 Lab 1 — Image Encoding & Decoding

## 🎯 Objective
Understand **spatial resolution (sampling)** and **intensity resolution (quantization)** by building a simple **image encoder–decoder system**.

---

## 🖼️ Input & Preprocessing
- Accept user-uploaded image
- Crop the **central square region** ✂️
- Use cropped image for all processing

---

## 📐 Spatial Resolution (Sampling)
User selects one resolution:

| Index | Resolution |
|------:|------------|
| 00 | 100 × 100 |
| 01 | 200 × 200 |
| 10 | 400 × 400 |
| 11 | 800 × 800 |

➡️ Resize square image to selected resolution

---

## 🎚️ Intensity Resolution (Quantization)
User selects bit depth:

| Index | Bits | Levels |
|------:|-----:|-------:|
| 00 | 1 | 2 |
| 01 | 2 | 4 |
| 10 | 4 | 16 |
| 11 | 8 | 256 |

➡️ Quantize pixel intensities accordingly

---

## ⚙️ Encoding Algorithm
1. Apply **sampling** 📉  
2. Apply **quantization** 🎛️  
3. Generate header  
4. Store header + pixel data in file 💾

---

## 📦 Custom File Format
### Header (4 bits)
[ S1 S0 | Q1 Q0 ]

- `S1 S0` → Spatial resolution index  
- `Q1 Q0` → Intensity resolution index  

### Data
- Sequential quantized pixel values

---

## 🔓 Decoding Algorithm
1. Read encoded file 📂  
2. Extract header  
3. Decode resolution & bit depth  
4. Reconstruct image 🖼️  

---

## ✅ Outcome
- Observe effects of **downsampling** & **quantization**
- Learn compact image representation
- Build a full **encode → decode** pipeline 🔁

---

## ⚠️ Notes
- Lower resolution → loss of detail 🔍  
- Lower bit depth → visible artifacts 🧱  
- Foundation for **image compression** 📉



# 🧪 Lab 2 — Affine Transformations on Digital Images

## 🎯 Objective
Understand and implement **2D affine transformations** on a digital image by applying  
**scaling, rotation, translation, and shearing** — **without using any built-in image processing libraries** ⚠️.

---

## 🧠 Problem Statement
Design and implement a program that:
- Loads a digital image 📂
- Allows **interactive affine transformations** 🕹️
- Generates a correctly sized **transformed output image** 🖼️

👉 All transformations are implemented **manually** using matrix mathematics.

---

## 🖼️ Image Format
- **Input:** 24-bit uncompressed BMP  
- **Output:** 24-bit uncompressed BMP  
- **Color Model:** RGB (3 channels × 8-bit) 🎨

---

## ⚙️ Features Implemented

### 🔁 Affine Transformations
- Horizontal scaling ↔️  
- Vertical scaling ↕️  
- Rotation about origin 🔄  
- Rotation about image center 🎯  
- Translation (x, y) 📦  
- Shearing (horizontal & vertical) 🪜  

---

### 🧮 Transformation System
- Homogeneous coordinates (3×3 matrices)  
- Transformation composition via matrix multiplication ✖️  
- **Inverse affine mapping** for accurate resampling 🔁  

---

### 🔍 Resampling
- **Bilinear interpolation** 📐  
- Proper boundary clamping 🚧  
- Automatic output image size computation 📏  

---

### 🧩 Image Handling
- Manual BMP loader (header parsing, padding handling) 🧾  
- Manual BMP writer 💾  
- Bottom-up BMP pixel handling ⬆️  

---

### 🕹️ User Interaction
- Interactive command-line interface 💻  
- Apply transformations incrementally ➕  
- Save output image 💾  
- Reset transformation matrix ♻️  
- Revert to original image on failure 🔙  

---

## ⌨️ Commands Supported

| Command | Description |
|------|------------|
| `scale` | Apply horizontal & vertical scaling 📐 |
| `rotate` | Rotate about origin 🔄 |
| `rotate_center` | Rotate about image center 🎯 |
| `translate` | Translate image 📦 |
| `shear` | Apply shearing 🪜 |
| `apply` | Apply accumulated transformation ⚙️ |
| `save` | Save transformed image 💾 |
| `revert` | Restore original image 🔙 |
| `reset` | Reset transformation matrix ♻️ |
| `help` | Show command list ❓ |
| `exit` | Exit program 🚪 |

---

## 🧠 Core Concepts Used
- Affine transformations  
- Matrix multiplication ✖️  
- Inverse mapping 🔁  
- Bilinear interpolation 📐  
- Coordinate clamping 🚧  
- Bounding box computation 📦  

---

## 🛠️ Compilation

### Using g++
```bash
g++ -std=c++17 affine.cpp -o affine



# 🧪 Lab 5 — Image Restoration (Spatial & Frequency Domain)

## 🎯 Objective

To design and implement a complete **Image Restoration System in C++** that:

* Estimates the **type of noise** present in an image
* Applies appropriate **spatial-domain filters**
* Applies appropriate **frequency-domain filters**
* Evaluates restoration quality using **PSNR (Peak Signal-to-Noise Ratio)**

The system is implemented from scratch, including FFT and filtering logic.

---

## 🧠 Problem Statement

For the given grayscale images (1–7):

1. Estimate the noise type statistically
2. Apply suitable spatial filters
3. Apply frequency-domain restoration techniques
4. Evaluate restoration quality using PSNR

No high-level image-processing libraries (like OpenCV) were used.

---

## 🖼️ Image Format

* **Input:** 8-bit grayscale TIFF
* **Output:** 8-bit grayscale TIFF
* TIFF handling implemented using `libtiff`

---

## 🔎 Noise Estimation

Noise type is estimated using statistical analysis:

* Mean
* Variance
* Skewness
* Kurtosis
* Zero ratio
* Max ratio
* Histogram uniformity
* FFT magnitude analysis

### Supported Noise Models

* Gaussian Noise
* Salt & Pepper Noise
* Uniform Noise
* Rayleigh Noise
* Exponential Noise
* Erlang (Gamma) Noise
* Periodic Noise
* Spatially varying noise

Automatic classification is rule-based using statistical thresholds.

---

## 🧩 Spatial Domain Filters

| Filter               | Purpose                  |
| -------------------- | ------------------------ |
| Mean Filter          | Uniform noise reduction  |
| Median Filter        | Salt & pepper removal    |
| Adaptive Median      | Heavy impulse noise      |
| Gaussian Filter      | Gaussian noise reduction |
| Geometric Mean       | Multiplicative noise     |
| Harmonic Mean        | Salt noise               |
| Contra-Harmonic      | Salt or pepper selective |
| Log-domain filtering | Rayleigh/Erlang noise    |

---

## 🔄 Frequency Domain Filters

Implemented using a custom 2D FFT (Cooley–Tukey algorithm).

| Filter           | Description            |
| ---------------- | ---------------------- |
| Inverse Filter   | Deconvolution          |
| Wiener Filter    | Noise-aware deblurring |
| FFT Notch Filter | Periodic noise removal |

### Features

* Custom 1D and 2D FFT
* Zero-padding to power-of-two
* Inverse FFT reconstruction
* Notch rejection filtering
* Frequency-domain deblurring

---

## 📈 Quality Metric

### PSNR (Peak Signal-to-Noise Ratio)

```
PSNR = 10 log10((MAX^2) / MSE)
```

Where:

* `MSE` = Mean Squared Error
* `MAX` = Maximum intensity value (255 for 8-bit images)

Higher PSNR indicates better restoration quality.

---

## ⚙️ Program Modes

| Mode              | Description              |
| ----------------- | ------------------------ |
| `estimate`        | Detect noise type only   |
| `auto`            | Auto-detect and restore  |
| Manual strategy   | Apply selected filter    |
| Spatial filters   | Kernel-based restoration |
| Frequency filters | FFT-based restoration    |

---

## 🛠️ Compilation (Windows – MSVC)

Open **x64 Native Tools Command Prompt** and run:

```bash
cl ImageRestoration.cpp /std:c++20 /EHsc ^
   /I C:\vcpkg\installed\x64-windows\include ^
   /link /LIBPATH:C:\vcpkg\installed\x64-windows\lib tiff.lib
```

---

## ▶️ Running the Program

```bash
ImageRestoration.exe
```

To run using input file:

```bash
ImageRestoration.exe < input.txt
```

To capture console output:

```bash
ImageRestoration.exe < input.txt > output.txt
```

---

## 🔬 Observations

* All provided test images exhibited strong periodic components.
* FFT-based notch filtering significantly improved PSNR.
* Spatial filters were less effective for structured periodic noise.
* PSNR ranged approximately between **35 dB and 54 dB** depending on image complexity.

---

## ✅ Outcome

This lab demonstrates:

* Practical noise modeling and classification
* Multi-domain restoration techniques
* Manual implementation of FFT
* End-to-end analyze → restore → evaluate pipeline
* Performance evaluation using PSNR

This forms a foundation for advanced image processing and frequency-domain restoration techniques.

---




## 🖥️ Running in x64 Native Tools Command Prompt (Windows – MSVC)

### 1️⃣ Open the Correct Terminal

* Open **Start Menu**
* Search for: **x64 Native Tools Command Prompt for VS 2022**
* Open it

> This ensures the MSVC compiler (`cl`) and linker are properly configured.

---

### 2️⃣ Navigate to Project Directory

```bash
cd H:\image_processing\Lab
```

(Replace with your actual project path.)

---

### 3️⃣ Compile the Program

```bash
cl ImageRestoration.cpp /std:c++20 /EHsc ^
   /I C:\vcpkg\installed\x64-windows\include ^
   /link /LIBPATH:C:\vcpkg\installed\x64-windows\lib tiff.lib
```

If compilation succeeds, this generates:

```
ImageRestoration.exe
```

---

### 4️⃣ Run the Program

```bash
ImageRestoration.exe
```

or

```bash
.\ImageRestoration.exe
```

---

### 5️⃣ Run Using Input File (Optional)

```bash
ImageRestoration.exe < input.txt
```

To capture console output:

```bash
ImageRestoration.exe < input.txt > output.txt
```

---

### ⚠️ Common Issues

| Issue                    | Solution                             |
| ------------------------ | ------------------------------------ |
| `'cl' is not recognized` | Open x64 Native Tools Command Prompt |
| TIFF linking error       | Verify vcpkg include and lib paths   |
| Architecture mismatch    | Use x64 prompt instead of x86        |

---

✅ After successful execution, restored TIFF images will be generated in the project directory and PSNR values will be displayed in the console.
