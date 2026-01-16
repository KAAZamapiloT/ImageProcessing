#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>

using namespace cv;
using namespace std;

// -----------------------------
// CONFIG TABLES
// -----------------------------
const vector<int> SPATIAL_SIZES = {100, 200, 400, 800};
const vector<int> INTENSITY_BITS = {1, 2, 4, 8};

// -----------------------------
// IMAGE PREPROCESSING
// -----------------------------
Mat centerCropGray(const Mat& img)
{
    Mat gray;
    if (img.channels() == 3)
        cvtColor(img, gray, COLOR_BGR2GRAY);
    else
        gray = img.clone();

    int h = gray.rows;
    int w = gray.cols;
    int side = min(h, w);

    int y = (h - side) / 2;
    int x = (w - side) / 2;

    return gray(Rect(x, y, side, side));
}

Mat spatialSample(const Mat& img, int size)
{
    Mat resized;
    resize(img, resized, Size(size, size), 0, 0, INTER_AREA);
    return resized;
}

Mat quantize(const Mat& img, int bits)
{
    int levels = 1 << bits;
    Mat quantized(img.size(), CV_8U);

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            float norm = img.at<uchar>(i, j) / 255.0f;
            int q = round(norm * (levels - 1));
            quantized.at<uchar>(i, j) =
                static_cast<uchar>(q * (255.0f / (levels - 1)));
        }
    }
    return quantized;
}

// -----------------------------
// ENCODER
// -----------------------------
uint8_t makeHeader(int spatialIdx, int intensityIdx)
{
    // 4-bit header: [ spatial(2) | intensity(2) ]
    return static_cast<uint8_t>((spatialIdx << 2) | intensityIdx);
}

void encodeImage(const string& inputPath,
                 int spatialIdx,
                 int intensityIdx,
                 const string& outputFile)
{
    Mat img = imread(inputPath);
    if (img.empty())
    {
        cerr << "Error: Cannot read input image\n";
        exit(1);
    }

    Mat proc = centerCropGray(img);
    proc = spatialSample(proc, SPATIAL_SIZES[spatialIdx]);
    proc = quantize(proc, INTENSITY_BITS[intensityIdx]);

    uint8_t header = makeHeader(spatialIdx, intensityIdx);

    ofstream out(outputFile, ios::binary);
    out.write(reinterpret_cast<char*>(&header), 1);
    out.write(reinterpret_cast<char*>(proc.data),
              proc.total() * proc.elemSize());
    out.close();

    cout << "[ENCODED]\n";
    cout << " Resolution: " << SPATIAL_SIZES[spatialIdx] << "x"
         << SPATIAL_SIZES[spatialIdx] << "\n";
    cout << " Intensity: " << INTENSITY_BITS[intensityIdx] << " bits\n";
}

// -----------------------------
// DECODER
// -----------------------------
Mat decodeImage(const string& inputFile,
                int& spatialIdx,
                int& intensityIdx)
{
    ifstream in(inputFile, ios::binary);
    if (!in)
    {
        cerr << "Error: Cannot open encoded file\n";
        exit(1);
    }

    uint8_t header;
    in.read(reinterpret_cast<char*>(&header), 1);

    spatialIdx = (header >> 2) & 0b11;
    intensityIdx = header & 0b11;

    int size = SPATIAL_SIZES[spatialIdx];
    Mat img(size, size, CV_8U);

    in.read(reinterpret_cast<char*>(img.data),
            img.total() * img.elemSize());
    in.close();

    return img;
}

// -----------------------------
// MAIN
// -----------------------------
int main(int argc, char** argv)
{
    if (argc < 2)
    {
        cout << "Usage: ./img_codec input_image\n";
        return 0;
    }

    int spatialIdx, intensityIdx;

    cout << "\nSelect Spatial Resolution:\n";
    cout << " 0 → 100x100\n 1 → 200x200\n 2 → 400x400\n 3 → 800x800\n";
    cout << "Enter choice (0–3): ";
    cin >> spatialIdx;

    cout << "\nSelect Intensity Resolution:\n";
    cout << " 0 → 1 bit\n 1 → 2 bits\n 2 → 4 bits\n 3 → 8 bits\n";
    cout << "Enter choice (0–3): ";
    cin >> intensityIdx;

    string encodedFile = "encoded_image.bin";

    encodeImage(argv[1], spatialIdx, intensityIdx, encodedFile);

    int decSpatial, decIntensity;
    Mat decoded = decodeImage(encodedFile, decSpatial, decIntensity);

    string title = "Decoded Image: " +
        to_string(SPATIAL_SIZES[decSpatial]) + "x" +
        to_string(SPATIAL_SIZES[decSpatial]) + ", " +
        to_string(INTENSITY_BITS[decIntensity]) + " bits";

    imshow(title, decoded);
    waitKey(0);

    return 0;
}
