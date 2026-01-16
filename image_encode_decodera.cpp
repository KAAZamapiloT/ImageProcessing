#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>

using namespace std;

// --------------------------------------------------
// CONFIG TABLES
// --------------------------------------------------
const int SPATIAL_SIZES[4]   = {100, 200, 400, 800};
const int INTENSITY_BITS[4]  = {1, 2, 4, 8};

// --------------------------------------------------
// SIMPLE IMAGE STRUCT
// --------------------------------------------------
struct Image {
    int width;
    int height;
    vector<uint8_t> data;
};

// --------------------------------------------------
// PGM IMAGE LOADER (MANUAL)
// --------------------------------------------------
Image readPGM(const string& filename)
{
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Cannot open image file\n";
        exit(1);
    }

    string magic;
    file >> magic;
    if (magic != "P5") {
        cerr << "Only binary PGM (P5) supported\n";
        exit(1);
    }

    int width, height, maxVal;
    file >> width >> height >> maxVal;
    file.ignore(1); // skip newline

    Image img;
    img.width = width;
    img.height = height;
    img.data.resize(width * height);

    file.read(reinterpret_cast<char*>(img.data.data()), width * height);
    file.close();

    return img;
}

// --------------------------------------------------
// CENTER CROP (MANUAL)
// --------------------------------------------------
Image centerCrop(const Image& img)
{
    int side = min(img.width, img.height);
    int x0 = (img.width  - side) / 2;
    int y0 = (img.height - side) / 2;

    Image out;
    out.width = side;
    out.height = side;
    out.data.resize(side * side);

    for (int y = 0; y < side; y++) {
        for (int x = 0; x < side; x++) {
            out.data[y * side + x] =
                img.data[(y + y0) * img.width + (x + x0)];
        }
    }
    return out;
}

// --------------------------------------------------
// SPATIAL SAMPLING (NEAREST NEIGHBOR)
// --------------------------------------------------
Image spatialSample(const Image& img, int outSize)
{
    Image out;
    out.width = out.height = outSize;
    out.data.resize(outSize * outSize);

    for (int i = 0; i < outSize; i++) {
        for (int j = 0; j < outSize; j++) {
            int srcX = i * img.height / outSize;
            int srcY = j * img.width  / outSize;
            out.data[i * outSize + j] =
                img.data[srcX * img.width + srcY];
        }
    }
    return out;
}

// --------------------------------------------------
// INTENSITY QUANTIZATION (MANUAL)
// --------------------------------------------------
Image quantize(const Image& img, int bits)
{
    int levels = 1 << bits;
    Image out = img;

    for (size_t i = 0; i < img.data.size(); i++) {
        float norm = img.data[i] / 255.0f;
        int q = round(norm * (levels - 1));
        out.data[i] = static_cast<uint8_t>(
            q * (255.0f / (levels - 1))
        );
    }
    return out;
}

// --------------------------------------------------
// HEADER (4 BITS LOGICAL)
// --------------------------------------------------
uint8_t makeHeader(int spatialIdx, int intensityIdx)
{
    return (spatialIdx << 2) | intensityIdx;
}

// --------------------------------------------------
// ENCODER
// --------------------------------------------------
void encode(const Image& img,
            int spatialIdx,
            int intensityIdx,
            const string& filename)
{
    Image proc = centerCrop(img);
    proc = spatialSample(proc, SPATIAL_SIZES[spatialIdx]);
    proc = quantize(proc, INTENSITY_BITS[intensityIdx]);

    uint8_t header = makeHeader(spatialIdx, intensityIdx);

    ofstream out(filename, ios::binary);
    out.write(reinterpret_cast<char*>(&header), 1);
    out.write(reinterpret_cast<char*>(proc.data.data()),
              proc.data.size());
    out.close();

    cout << "[ENCODED]\n";
    cout << "Resolution: " << SPATIAL_SIZES[spatialIdx] << " x "
         << SPATIAL_SIZES[spatialIdx] << "\n";
    cout << "Intensity: " << INTENSITY_BITS[intensityIdx] << " bits\n";
}

// --------------------------------------------------
// DECODER
// --------------------------------------------------
Image decode(const string& filename,
             int& spatialIdx,
             int& intensityIdx)
{
    ifstream in(filename, ios::binary);

    uint8_t header;
    in.read(reinterpret_cast<char*>(&header), 1);

    spatialIdx   = (header >> 2) & 0b11;
    intensityIdx = header & 0b11;

    int size = SPATIAL_SIZES[spatialIdx];

    Image img;
    img.width = img.height = size;
    img.data.resize(size * size);

    in.read(reinterpret_cast<char*>(img.data.data()),
            img.data.size());
    in.close();

    return img;
}

// --------------------------------------------------
// SAVE DECODED IMAGE AS PGM
// --------------------------------------------------
void writePGM(const Image& img, const string& filename)
{
    ofstream out(filename, ios::binary);
    out << "P5\n" << img.width << " " << img.height << "\n255\n";
    out.write(reinterpret_cast<const char*>(img.data.data()),
              img.data.size());
    out.close();
}

// --------------------------------------------------
// MAIN
// --------------------------------------------------
int main(int argc, char* argv[])
{
    if (argc < 2) {
        cout << "Usage: ./img_codec input.pgm\n";
        return 0;
    }

    Image img = readPGM(argv[1]);

    int spatialIdx, intensityIdx;

    cout << "\nSpatial Resolution:\n";
    cout << "0 → 100x100\n1 → 200x200\n2 → 400x400\n3 → 800x800\n";
    cin >> spatialIdx;

    cout << "\nIntensity Resolution:\n";
    cout << "0 → 1 bit\n1 → 2 bits\n2 → 4 bits\n3 → 8 bits\n";
    cin >> intensityIdx;

    encode(img, spatialIdx, intensityIdx, "encoded_image.bin");

    int sIdx, iIdx;
    Image decoded = decode("encoded_image.bin", sIdx, iIdx);

    writePGM(decoded, "decoded_image.pgm");

    cout << "Decoded image saved as decoded_image.pgm\n";
    return 0;
}
