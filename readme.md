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
