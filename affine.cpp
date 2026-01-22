/*****************************************************************************************
 * File: affine.cpp
 *
 * Purpose:
 * --------
 * Implements 2D affine transformations on a 24-bit uncompressed BMP image
 * without using any external image processing libraries.
 *
 * Supported transformations:
 *  - Scaling (horizontal, vertical)
 *  - Rotation (about origin, about image center)
 *  - Translation
 *  - Shearing
 *
 * Implementation notes:
 * ---------------------
 * - Uses homogeneous coordinates (3×3 matrices)
 * - Transformations are accumulated, then applied in one pass
 * - Inverse mapping is used to avoid holes
 * - Bilinear interpolation is used for resampling
 * - Output image size is computed automatically
 *
 * Constraints:
 * ------------
 * - No image or math libraries
 * - Manual BMP parsing and writing
 *
 * Author:
 * -------
 * Uday Singh
 *****************************************************************************************/


/* =========================
   Vector3<T>
   =========================
   Generic 3-component vector.

   Usage:
   - Stores RGB pixels when T = uint8_t
   - Can represent (x, y, 1) homogeneous coordinates

   Members:
   - x, y, z : vector components

   Operators:
   - +  : component-wise addition
   - -  : component-wise subtraction
   - *  : scalar multiplication
*/


/* =========================
   Matrix3<T>
   =========================
   3×3 matrix for affine transformations in homogeneous coordinates.

   Layout:
   [ a  b  tx ]
   [ c  d  ty ]
   [ 0  0  1  ]

   Static Functions:
   - Identity()  : returns identity matrix
   - Multiply()  : matrix multiplication (A * B)
*/


/* =========================
   ImageObject
   =========================
   Represents an image and all affine transformation logic.

   Responsibilities:
   - Load / save BMP images
   - Store RGB pixel data
   - Accumulate affine transformations
   - Apply transformations via inverse mapping
   - Perform bilinear interpolation
*/


/* Constructor
   -----------
   ImageObject(const std::string& file)

   Loads a 24-bit uncompressed BMP image.
   Initializes:
   - Width, Height
   - Pixel buffer
   - Transformation matrix (identity)
*/


/* Transformation Builders
   -----------------------
   These functions MODIFY the transformation matrix only.
   No pixels are changed until ApplyAffine() is called.

   - H_Scale(float sx)        : horizontal scaling
   - V_Scale(float sy)        : vertical scaling
   - Rotate(float angle)      : rotation about origin (degrees)
   - RotateAboutCenter(angle) : rotation about image center
   - H_Translate(float tx)    : horizontal translation
   - V_Translate(float ty)    : vertical translation
   - H_Shear(float shx)       : horizontal shear
   - V_Shear(float shy)       : vertical shear
*/


/* ResetTo(const ImageObject& src)
   -------------------------------
   Restores pixels and dimensions from another image.
   Resets transformation matrix to identity.

   Used for:
   - revert command
   - failure recovery
*/


/* Clear()
   -------
   Clears pixel buffer and resets image state.
   Used when output becomes invalid.
*/


/* InvertAffine(Matrix3)
   --------------------
   Computes inverse of a 2D affine matrix.

   Required for:
   - inverse mapping during resampling

   Returns identity if matrix is non-invertible.
*/


/* ComputeTransformedBounds(...)
   ----------------------------
   Transforms the four image corners and computes
   an axis-aligned bounding box.

   Used to determine output image size.
*/


/* ApplyAffine()
   -------------
   Applies the accumulated affine transformation.

   Pipeline:
   1. Compute transformed bounds
   2. Allocate output image
   3. Build inverse affine matrix
   4. For each output pixel:
      - Map to source using inverse matrix
      - Sample using bilinear interpolation
      - Clamp coordinates to valid range

   Returns:
   - New ImageObject with transformed image
*/


/* loadBMP(...)
   -----------
   Reads a 24-bit uncompressed BMP file.

   Handles:
   - BMP headers
   - Row padding
   - Bottom-up storage
*/


/* SaveBMP(filename)
   -----------------
   Writes image as a 24-bit uncompressed BMP file.
*/


/* =========================
   HandleUser
   =========================
   Command-line controller for ImageObject.

   Responsibilities:
   - Parse user commands
   - Apply transformations incrementally
   - Apply / save / revert images safely
   - Prevent invalid states
*/


/* Run()
   -----
   Starts interactive command loop.

   Commands:
   - scale
   - rotate
   - rotate_center
   - translate
   - shear
   - apply
   - save
   - revert
   - reset
   - help
   - exit
*/


/* main()
   ------
   Program entry point.

   - Prompts for input BMP file
   - Creates HandleUser instance
   - Starts interactive session
*/


#include<iostream>
#include<vector>
#include<string>
#include <fstream>
#include <cstdint>
#include <algorithm>

#include <cmath>



#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct BMPInfoHeader {
    uint32_t biSize;
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};
#pragma pack(pop)



template<typename T>
class Vector3{
  public:
  Vector3(){

  }
  Vector3(T x,T y,T z){
      this->x=x;
      this->y=y;
      this->z=z;
  }
  Vector3<T> operator+(const Vector3<T>& other) const {
    return Vector3<T>(x + other.x, y + other.y, z + other.z);
  }
  Vector3<T> operator-(const Vector3<T>& other) const {
    return Vector3<T>(x - other.x, y - other.y, z - other.z);
  }
  Vector3<T> operator*(const T& scalar) const {
    return Vector3<T>(x * scalar, y * scalar, z * scalar);
  }

T x,y,z;
};


template<typename T>
class Matrix3{
    public:
    T m[3][3];

        static Matrix3 Identity() {
            Matrix3 I{};
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    I.m[i][j] = (i == j) ? 1 : 0;
            return I;
        }

        static Matrix3 Multiply(const Matrix3& A, const Matrix3& B) {
            Matrix3 R{};
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    R.m[i][j] = 0;
                    for (int k = 0; k < 3; k++) {
                        R.m[i][j] += A.m[i][k] * B.m[k][j];
                    }
                }
            }
            return R;
        }

};

class ImageObject{
    public:
    ImageObject(const std::string& FilePath){
        TransformationMatrix = Matrix3<float>::Identity();
        loadBMP(FilePath, Width, Height, pixels);
    }


    void H_Scale(float Scale){
        Matrix3<float> S = Matrix3<float>::Identity();
               S.m[0][0] = Scale;
               TransformationMatrix = Matrix3<float>::Multiply(S, TransformationMatrix);
    };
    void V_Scale(float Scale){
        Matrix3<float> S = Matrix3<float>::Identity();
               S.m[1][1] = Scale;
               TransformationMatrix = Matrix3<float>::Multiply(S, TransformationMatrix);
    };

    // angle in degress
    void Rotate(float angle){
        float rad = angle * 3.14159265f / 180.0f;
               // Roatation Matrix and multiplying transformation matrix with roation one
                Matrix3<float> R = Matrix3<float>::Identity();
                R.m[0][0] =  cos(rad);
                R.m[0][1] = -sin(rad);
                R.m[1][0] =  sin(rad);
                R.m[1][1] =  cos(rad);
                TransformationMatrix = Matrix3<float>::Multiply(R, TransformationMatrix);
    }

    void RotateAboutCenter(float angle) {
        float rad = angle * 3.14159265f / 180.0f;

        float cx = Width  / 2.0f;
        float cy = Height / 2.0f;

        // T1: translate center to origin
        Matrix3<float> T1 = Matrix3<float>::Identity();
        T1.m[0][2] = -cx;
        T1.m[1][2] = -cy;

        // R: rotation
        Matrix3<float> R = Matrix3<float>::Identity();
        R.m[0][0] =  cos(rad);
        R.m[0][1] = -sin(rad);
        R.m[1][0] =  sin(rad);
        R.m[1][1] =  cos(rad);

        // T2: translate back
        Matrix3<float> T2 = Matrix3<float>::Identity();
        T2.m[0][2] = cx;
        T2.m[1][2] = cy;

        // Compose: T2 * R * T1 * current
        TransformationMatrix =
            Matrix3<float>::Multiply(
                T2,
                Matrix3<float>::Multiply(
                    R,
                    Matrix3<float>::Multiply(T1, TransformationMatrix)
                )
            );
    }


    void H_Translate(float x){
      Matrix3<float> T = Matrix3<float>::Identity();
      T.m[0][2] = x;
      TransformationMatrix = Matrix3<float>::Multiply(T, TransformationMatrix);
    }

    void V_Translate(float x){
      Matrix3<float> T = Matrix3<float>::Identity();
      T.m[1][2] = x;
      TransformationMatrix = Matrix3<float>::Multiply(T, TransformationMatrix);
    }

    void H_Shear(float x){
          Matrix3<float> S = Matrix3<float>::Identity();
          S.m[0][1] = x;
          TransformationMatrix = Matrix3<float>::Multiply(S, TransformationMatrix);
    }

    void V_Shear(float x){
          Matrix3<float> S = Matrix3<float>::Identity();
          S.m[1][0] = x;
          TransformationMatrix = Matrix3<float>::Multiply(S, TransformationMatrix);
    }

    void ResetTo(const ImageObject& src) {
        Width = src.Width;
        Height = src.Height;
        pixels = src.pixels; // deep copy
        TransformationMatrix = Matrix3<float>::Identity();
    }


    Matrix3<float> InvertAffine(const Matrix3<float>& M) {
        Matrix3<float> inv = Matrix3<float>::Identity();

        float a  = M.m[0][0];
        float b  = M.m[0][1];
        float tx = M.m[0][2];

        float c  = M.m[1][0];
        float d  = M.m[1][1];
        float ty = M.m[1][2];

        float det = a * d - b * c;

        if (det == 0) {
            std::cerr << "Error: Affine matrix is not invertible\n";
            return inv; // identity fallback (safe)
        }

        float invDet = 1.0f / det;

        inv.m[0][0] =  d * invDet;
        inv.m[0][1] = -b * invDet;
        inv.m[0][2] = (b * ty - d * tx) * invDet;

        inv.m[1][0] = -c * invDet;
        inv.m[1][1] =  a * invDet;
        inv.m[1][2] = (c * tx - a * ty) * invDet;

        inv.m[2][0] = 0;
        inv.m[2][1] = 0;
        inv.m[2][2] = 1;

        return inv;
    }


    ImageObject ApplyAffine() {
        float minX, minY, maxX, maxY;

        ComputeTransformedBounds(
            TransformationMatrix,
            Width, Height,
            minX, minY, maxX, maxY
        );

        // include +1 to account for inclusive pixel coordinates
        int outWidth  = (int)std::ceil(maxX - minX + 1.0f);
        int outHeight = (int)std::ceil(maxY - minY + 1.0f);

        if (outWidth <= 0) outWidth = 1;
        if (outHeight <= 0) outHeight = 1;

        ImageObject output(outWidth, outHeight);

        // translation to move minX,minY to origin
        Matrix3<float> offset = Matrix3<float>::Identity();
        offset.m[0][2] = -minX;
        offset.m[1][2] = -minY;

        Matrix3<float> finalM = Matrix3<float>::Multiply(offset, TransformationMatrix);
        Matrix3<float> inv = InvertAffine(finalM);

        // safety: if source image is empty return a blank image
        if (Width <= 0 || Height <= 0) return output;

        for (int y = 0; y < outHeight; y++) {
            for (int x = 0; x < outWidth; x++) {

                float xin = inv.m[0][0] * x + inv.m[0][1] * y + inv.m[0][2];
                float yin = inv.m[1][0] * x + inv.m[1][1] * y + inv.m[1][2];

                // base integer pixel (unclamped) for interpolation
                int x1f = (int)std::floor(xin);
                int y1f = (int)std::floor(yin);
                int x2f = x1f + 1;
                int y2f = y1f + 1;

                // fractions
                float dx = xin - (float)x1f;
                float dy = yin - (float)y1f;

                // clamp sampling indices to valid range so we always sample valid pixels
                auto clampi = [](int v, int lo, int hi) {
                    return std::max(lo, std::min(v, hi));
                };

                int sx11 = clampi(x1f, 0, Width  - 1);
                int sx21 = clampi(x2f, 0, Width  - 1);
                int sy11 = clampi(y1f, 0, Height - 1);
                int sy12 = clampi(y2f, 0, Height - 1);

                const Vector3<uint8_t>& p11 = pixels[sy11 * Width + sx11];
                const Vector3<uint8_t>& p21 = pixels[sy11 * Width + sx21];
                const Vector3<uint8_t>& p12 = pixels[sy12 * Width + sx11];
                const Vector3<uint8_t>& p22 = pixels[sy12 * Width + sx21];

                auto blend_channel = [&](uint8_t c11, uint8_t c21, uint8_t c12, uint8_t c22) -> uint8_t {
                    float top    = (1.0f - dx) * (float)c11 + dx * (float)c21;
                    float bottom = (1.0f - dx) * (float)c12 + dx * (float)c22;
                    float v = (1.0f - dy) * top + dy * bottom;
                    int vi = (int)std::lround(v);
                    vi = std::max(0, std::min(255, vi));
                    return (uint8_t)vi;
                };

                int outIdx = y * outWidth + x;
                output.pixels[outIdx].x = blend_channel(p11.x, p21.x, p12.x, p22.x); // R
                output.pixels[outIdx].y = blend_channel(p11.y, p21.y, p12.y, p22.y); // G
                output.pixels[outIdx].z = blend_channel(p11.z, p21.z, p12.z, p22.z); // B
            }
        }

        return output;
    }


    std::vector<Vector3<uint8_t>> pixels; // RGBRGBRGB.....
    int Width;
    int Height;
private:
    Matrix3<float> TransformationMatrix;

    void ComputeTransformedBounds(
        const Matrix3<float>& M,
        int width, int height,
        float& minX, float& minY,
        float& maxX, float& maxY
    ) {
        float corners[4][3] = {
            {0.0f,                 0.0f,                  1.0f},
            {static_cast<float>(width - 1), 0.0f,          1.0f},
            {0.0f,                 static_cast<float>(height - 1), 1.0f},
            {static_cast<float>(width - 1), static_cast<float>(height - 1), 1.0f}
        };

        minX = minY =  1e9f;
        maxX = maxY = -1e9f;

        for (int i = 0; i < 4; i++) {
            float x = M.m[0][0]*corners[i][0] +
                      M.m[0][1]*corners[i][1] +
                      M.m[0][2];

            float y = M.m[1][0]*corners[i][0] +
                      M.m[1][1]*corners[i][1] +
                      M.m[1][2];

            minX = std::min(minX, x);
            minY = std::min(minY, y);
            maxX = std::max(maxX, x);
            maxY = std::max(maxY, y);
        }
    }

    ImageObject(int w, int h) {
        Width = w;
        Height = h;
        pixels.resize(w * h);
        TransformationMatrix = Matrix3<float>::Identity();
    }

    bool loadBMP(
        const std::string& filename,
            int& width,
            int& height,
            std::vector<Vector3<uint8_t>>& pixels
    ){
        std::ifstream file(filename, std::ios::binary);
            if (!file) {
                std::cerr << "Cannot open BMP file\n";
                return false;
            }

            BMPFileHeader fileHeader;
            BMPInfoHeader infoHeader;

            file.read(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
            file.read(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

            // Validate BMP
            if (fileHeader.bfType != 0x4D42) { // 'BM'
                std::cerr << "Not a BMP file\n";
                return false;
            }

            if (infoHeader.biBitCount != 24 || infoHeader.biCompression != 0) {
                std::cerr << "Only uncompressed 24-bit BMP supported\n";
                return false;
            }

            width  = infoHeader.biWidth;
            height = infoHeader.biHeight;

            pixels.resize(width * height);

            int rowPadding = (4 - (width * 3) % 4) % 4;

            file.seekg(fileHeader.bfOffBits, std::ios::beg);

            // BMP is stored bottom-up
            for (int y = height - 1; y >= 0; y--) {
                for (int x = 0; x < width; x++) {
                    uint8_t b, g, r;
                    file.read((char*)&b, 1);
                    file.read((char*)&g, 1);
                    file.read((char*)&r, 1);

                    pixels[y * width + x] = Vector3<uint8_t>(r, g, b);
                }
                file.ignore(rowPadding);
            }

            return true;
    }
    public:
    void Clear() {
        pixels.clear();
        Width = Height = 0;
        TransformationMatrix = Matrix3<float>::Identity();
    }

    bool SaveBMP(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Cannot write BMP file\n";
            return false;
        }

        int rowPadding = (4 - (Width * 3) % 4) % 4;
        int imageSize = (Width * 3 + rowPadding) * Height;

        BMPFileHeader fileHeader{};
        BMPInfoHeader infoHeader{};

        fileHeader.bfType = 0x4D42; // 'BM'
        fileHeader.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
        fileHeader.bfSize = fileHeader.bfOffBits + imageSize;

        infoHeader.biSize = sizeof(BMPInfoHeader);
        infoHeader.biWidth = Width;
        infoHeader.biHeight = Height;
        infoHeader.biPlanes = 1;
        infoHeader.biBitCount = 24;
        infoHeader.biCompression = 0;
        infoHeader.biSizeImage = imageSize;

        file.write(reinterpret_cast<char*>(&fileHeader), sizeof(fileHeader));
        file.write(reinterpret_cast<char*>(&infoHeader), sizeof(infoHeader));

        uint8_t padding[3] = {0, 0, 0};

        // BMP is bottom-up
        for (int y = Height - 1; y >= 0; y--) {
            for (int x = 0; x < Width; x++) {
                const Vector3<uint8_t>& p = pixels[y * Width + x];
                file.put(p.z); // B
                file.put(p.y); // G
                file.put(p.x); // R
            }
            file.write(reinterpret_cast<char*>(padding), rowPadding);
        }

        return true;
    }

};



class HandleUser {
public:
HandleUser(const std::string& inputFile)
    : original(inputFile), img(inputFile), output(inputFile), applied(false) {}


    void Run() {
        std::string cmd;

        PrintHelp();

        while (true) {
            std::cout << "\n> ";
            std::cin >> cmd;

            if (cmd == "scale") {
                float sx, sy;
                std::cout << "Enter sx sy: ";
                std::cin >> sx >> sy;
                img.H_Scale(sx);
                img.V_Scale(sy);
                std::cout << "Scaling applied.\n";
            }

            else if (cmd == "rotate") {
                float angle;
                std::cout << "Enter angle (degrees): ";
                std::cin >> angle;
                img.Rotate(angle);
                std::cout << "Rotation about origin applied.\n";
            }

            else if (cmd == "rotate_center") {
                float angle;
                std::cout << "Enter angle (degrees): ";
                std::cin >> angle;
                img.RotateAboutCenter(angle);
                std::cout << "Rotation about center applied.\n";
            }

            else if (cmd == "translate") {
                float tx, ty;
                std::cout << "Enter tx ty: ";
                std::cin >> tx >> ty;
                img.H_Translate(tx);
                img.V_Translate(ty);
                std::cout << "Translation applied.\n";
            }

            else if (cmd == "shear") {
                float shx, shy;
                std::cout << "Enter shx shy: ";
                std::cin >> shx >> shy;
                img.H_Shear(shx);
                img.V_Shear(shy);
                std::cout << "Shearing applied.\n";
            }

            else if (cmd == "apply") {
                try {
                    ImageObject tmp = img.ApplyAffine();
                    output = std::move(tmp);
                    applied = true;
                    std::cout << "Affine transformation applied.\n";
                }
                catch (const std::exception& e) {
                    std::cerr << "Apply failed: " << e.what() << "\n";
                    std::cerr << "Image reverted to original state.\n";
                    img.ResetTo(original);
                    applied = false;
                    output.Clear();
                }
            }
            else if (cmd == "revert") {
                img.ResetTo(original);
                applied = false;
                output.Clear();
                std::cout << "Image reverted to original state.\n";
            }


            else if (cmd == "save") {
                if (!applied) {
                    std::cout << "Apply transformation first.\n";
                    continue;
                }
                std::string outFile;
                std::cout << "Enter output BMP file name: ";
                std::cin >> outFile;
                output.SaveBMP(outFile);
                std::cout << "Output saved.\n";
            }

            else if (cmd == "reset") {
                img = original;
                applied = false;
                std::cout << "Transformations reset.\n";
            }

            else if (cmd == "help") {
                PrintHelp();
            }

            else if (cmd == "exit") {
                std::cout << "Exiting...\n";
                break;
            }

            else {
                std::cout << "Unknown command. Type 'help'.\n";
            }
        }
    }

private:
    ImageObject original;
    ImageObject img;
    ImageObject output;
    bool applied = false;

    void PrintHelp() {
        std::cout << "\nCommands:\n";
        std::cout << " scale           -> scale image (sx sy)\n";
        std::cout << " rotate          -> rotate about origin\n";
        std::cout << " rotate_center   -> rotate about image center\n";
        std::cout << " translate       -> translate image\n";
        std::cout << " shear           -> shear image\n";
        std::cout << " apply           -> apply affine transformation\n";
        std::cout << " save            -> save output image\n";
        std::cout << " revert          -> revert image to original state\n";
        std::cout << " reset           -> reset transformation matrix only\n";
        std::cout << " help            -> show commands\n";
        std::cout << " exit            -> quit program\n";
    }

};


int main() {
    std::string inputFile;
    std::cout << "Enter input BMP file name: ";
    std::cin >> inputFile;

    HandleUser app(inputFile);
    app.Run();

    return 0;
}
