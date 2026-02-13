#include <filesystem>
#include <utility>
#include <cmath>

#include<iostream>
#include<string>
#include "fstream"
#include<vector>
#include<math.h>
#include <tiffio.h>
#include<algorithm>

#include <complex>
const double PI = 3.14159265358979323846;

using Complex = std::complex<double>;

struct ComplexImage {
    int width, height;
    std::vector<Complex> data;

    ComplexImage(int w, int h)
        : width(w), height(h), data(w*h) {}

    Complex& operator()(int y, int x) {
        return data[y*width + x];
    }
};

class ImageRow {
public:
    ImageRow(uint16_t* rowData) : m_Row(rowData) {}

    uint16_t& operator[](uint32_t x) {
        return m_Row[x];
    }

    const uint16_t& operator[](uint32_t x) const {
        return m_Row[x];
    }

private:
    uint16_t* m_Row;
};


class ImageObject{
  public:
  ImageObject(const std::string& filepath)
      : m_Path(filepath)
  {
      loadTIFF();
  }
    inline  uint32_t width()  const { return m_Width; }
    inline  uint32_t height() const { return m_Height; }
    inline  uint16_t bits()   const { return m_BitsPerSample; }
    inline  uint32_t levels() const { return m_Levels; }

    ImageRow operator[](uint32_t y) {
        return ImageRow(&m_Data[y * m_Width]);
    }

    const ImageRow operator[](uint32_t y) const {
        return ImageRow(const_cast<uint16_t*>(&m_Data[y * m_Width]));
    }

    void saveTIFF(std::string OutputPath){
        TIFF* out = TIFFOpen(OutputPath.c_str(), "w");
        if (!out)
            throw std::runtime_error("Failed to open output TIFF file");

        // Check actual data range before saving
        uint16_t saveMin = 65535, saveMax = 0;
        for (const auto& val : m_Data) {
            saveMin = std::min(saveMin, val);
            saveMax = std::max(saveMax, val);
        }

        std::cout << "Saving range: [" << saveMin << ", " << saveMax << "]\n";
        std::cout << "Saving as " << m_BitsPerSample << "-bit TIFF\n";

        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, m_Width);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, m_Height);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, m_BitsPerSample);
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);

        // Add min/max sample values for proper display
        TIFFSetField(out, TIFFTAG_SMINSAMPLEVALUE, saveMin);
        TIFFSetField(out, TIFFTAG_SMAXSAMPLEVALUE, saveMax);

        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP,
                     TIFFDefaultStripSize(out, m_Width));

        for (uint32_t row = 0; row < m_Height; row++) {
            TIFFWriteScanline(
                out,
                (void*)&m_Data[row * m_Width],
                row
            );
        }

        TIFFClose(out);
    }
    void saveTIFF8bit(std::string OutputPath){
        TIFF* out = TIFFOpen(OutputPath.c_str(), "w");
        if (!out)
            throw std::runtime_error("Failed to open output TIFF file");

        // Find data range
        uint16_t dataMin = 65535, dataMax = 0;
        for (const auto& val : m_Data) {
            dataMin = std::min(dataMin, val);
            dataMax = std::max(dataMax, val);
        }

        std::cout << "Converting from [" << dataMin << ", " << dataMax << "] to 8-bit\n";

        // Convert to 8-bit
        std::vector<uint8_t> data8bit(m_Width * m_Height);

        if (dataMax == dataMin) {
            // All pixels same value - map to middle gray
            std::fill(data8bit.begin(), data8bit.end(), 128);
        } else {
            double scale = 255.0 / (dataMax - dataMin);
            for (size_t i = 0; i < m_Data.size(); i++) {
                double normalized = (m_Data[i] - dataMin) * scale;
                data8bit[i] = static_cast<uint8_t>(std::clamp(normalized, 0.0, 255.0));
            }
        }

        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, m_Width);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, m_Height);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, m_Width));

        for (uint32_t row = 0; row < m_Height; row++) {
            TIFFWriteScanline(out, &data8bit[row * m_Width], row);
        }

        TIFFClose(out);
    }



        std::vector<uint64_t> computeHistogram() const {
            std::vector<uint64_t> hist(m_Levels, 0);

            for (uint16_t value : m_Data) {
                if (value < m_Levels) {
                    hist[value]++;
                }
            }

            return hist;
        }

  private:
  uint32_t m_Width,m_Height,m_Levels;
  uint16_t m_BitsPerSample = 0;
   std::string m_Path;
   std::vector<uint16_t> m_Data;


   // finction to load TIF file format images
   void loadTIFF(){
       TIFF* tif = TIFFOpen(m_Path.c_str(), "r");
       if (!tif)
           throw std::runtime_error("Failed to open TIFF file");

       uint16 samplesPerPixel, sampleFormat = SAMPLEFORMAT_UINT;

       TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &m_Width);
       TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &m_Height);
       TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &m_BitsPerSample);
       TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);
       TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT, &sampleFormat);

       if (samplesPerPixel != 1)
           throw std::runtime_error("Only grayscale TIFF supported");

       m_Levels = 1u << m_BitsPerSample;

       std::cout << "TIFF metadata: " << m_Width << "x" << m_Height
                 << ", " << m_BitsPerSample << " bits/sample\n";

       // Allocate buffer for one scanline in NATIVE format
       tmsize_t scanlineSize = TIFFScanlineSize(tif);
       std::cout << "Scanline size: " << scanlineSize << " bytes\n";
       std::cout << "Expected: " << m_Width * (m_BitsPerSample / 8) << " bytes\n";

       m_Data.resize(m_Width * m_Height);

       if (m_BitsPerSample == 8) {
           // 8-bit: read as uint8, then convert to uint16
           std::vector<uint8_t> scanline(m_Width);
           for (uint32_t row = 0; row < m_Height; row++) {
               TIFFReadScanline(tif, scanline.data(), row);
               for (uint32_t col = 0; col < m_Width; col++) {
                   m_Data[row * m_Width + col] = scanline[col];
               }
           }
       } else {
           // 16-bit: read directly
           for (uint32_t row = 0; row < m_Height; row++) {
               TIFFReadScanline(tif, &m_Data[row * m_Width], row);
           }
       }

       TIFFClose(tif);

       // Detect actual bit depth from data
       uint16_t actualMax = 0;
       for (const auto& val : m_Data) {
           actualMax = std::max(actualMax, val);
       }

       std::cout << "Actual data range: [0, " << actualMax << "]\n";

       // Correct bit depth if metadata is wrong
       if (actualMax > 255 && m_BitsPerSample == 8) {
           std::cout << "WARNING: Data exceeds 8-bit range! Correcting to 16-bit.\n";
           m_BitsPerSample = 16;
       }

       m_Levels = 1u << m_BitsPerSample;
   }
};





class IMAGE_RESTORATION {
public:

struct Params {
    int kernelSize = 3;
    int maxKernel = 7;

    double noiseVariance = 50.0;
    double gaussianSigma = 1.0;

    double Q = 1.5;

    double blurLength = 0.1;
    double blurAngle = 0.0;
    double wienerK = 0.01;
    double gamma = 0.001;
};
struct NoiseStats {
    double mean = 0;
    double variance = 0;
    double skewness = 0;
    double zeroRatio = 0;
    double maxRatio = 0;
};


    void ExecuteStrategy(const std::string& strategy,
                         const std::string& inputFile,
                         const std::string& outputFile,
                         const Params& params = Params())
    {
        ImageObject img(inputFile);
        ImageObject original = img;

        if (strategy == "gaussian")
            RemoveGaussianNoise(img, params);

        else if (strategy == "saltpepper")
            RemoveSaltPepperNoise(img, params);

        else if (strategy == "uniform")
            RemoveUniformNoise(img, params);

        else if (strategy == "rayleigh")
            RemoveRayleighNoise(img, params);

        else if (strategy == "erlang")
            RemoveErlangNoise(img, params);

        else if (strategy == "exponential")
            RemoveExponentialNoise(img, params);

        else if (strategy == "periodic")
            RemovePeriodicNoise(img, params);
        else if(strategy == "geometric")
            GeometricMeanFilter(img, params.kernelSize);

        else if(strategy == "harmonic")
            HarmonicMeanFilter(img, params.kernelSize);

        else if(strategy == "contraharmonic")
            ContraHarmonicFilter(img,
                                 params.kernelSize,
                                 params.Q);

        else if(strategy == "adaptive_median")
            AdaptiveMedianFilter(img,
                                 params.maxKernel);
        else if (strategy == "inverse")
           {
               std::cout << "Applying Frequency-Domain Inverse Filter...\n";
               InverseFilter(img, params);
           }

           else if (strategy == "wiener_freq")
           {
               std::cout << "Applying Frequency-Domain Wiener Filter...\n";
               WienerFilter(img, params);
           }

           else if (strategy == "periodic_fft")
           {
               std::cout << "Applying FFT-based Notch Filter...\n";
               PeriodicFFT(img, params);
           }

        else {
            std::cout << "Unknown strategy\n";
            return;
        }

        std::cout << "PSNR: "
                  << ComputePSNR(original, img)
                  << " dB\n";

        img.saveTIFF8bit(outputFile);
    }


    void AutoRestore(const std::string& inputFile,
                     const std::string& outputFile)
    {
        ImageObject img(inputFile);

        std::string detected =
            EstimateNoiseType(img);

        Params params;

        // Basic default parameters
        params.kernelSize = 3;
        params.maxKernel = 7;

        ExecuteStrategy(detected,
                        inputFile,
                        outputFile,
                        params);
    }

    std::string EstimateNoiseTypePublic(const ImageObject& img)
    {
        return EstimateNoiseType(img);
    }
private:

ComplexImage FFT2D(const ImageObject& img)
{
    int W = img.width();
    int H = img.height();

    ComplexImage F(W, H);

    // Copy image to complex
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            F(y,x) = Complex((double)img[y][x], 0.0);

    // FFT on rows
    for (int y = 0; y < H; y++) {
        std::vector<Complex> row(W);
        for (int x = 0; x < W; x++)
            row[x] = F(y,x);

        FFT1D(row, false);

        for (int x = 0; x < W; x++)
            F(y,x) = row[x];
    }

    // FFT on columns
    for (int x = 0; x < W; x++) {
        std::vector<Complex> col(H);
        for (int y = 0; y < H; y++)
            col[y] = F(y,x);

        FFT1D(col, false);

        for (int y = 0; y < H; y++)
            F(y,x) = col[y];
    }

    return F;
}




void FFT1D(std::vector<Complex>& a, bool invert)
{
    int n = a.size();

    // Bit reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j |= bit;

        if (i < j)
            std::swap(a[i], a[j]);
    }

    // Cooley–Tukey
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        Complex wlen = std::exp(Complex(0, ang));

        for (int i = 0; i < n; i += len) {
            Complex w = 1;
            for (int j = 0; j < len / 2; j++) {
                Complex u = a[i + j];
                Complex v = a[i + j + len/2] * w;

                a[i + j] = u + v;
                a[i + j + len/2] = u - v;

                w *= wlen;
            }
        }
    }

    if (invert) {
        for (int i = 0; i < n; i++)
            a[i] /= n;
    }
}

void IFFT2D(const ComplexImage& F,
            ImageObject& img)
{
    int W = img.width();
    int H = img.height();

    ComplexImage temp = F;

    // Inverse FFT on rows
    for (int y = 0; y < H; y++) {
        std::vector<Complex> row(W);
        for (int x = 0; x < W; x++)
            row[x] = temp(y,x);

        FFT1D(row, true);

        for (int x = 0; x < W; x++)
            temp(y,x) = row[x];
    }

    // Inverse FFT on columns
    for (int x = 0; x < W; x++) {
        std::vector<Complex> col(H);
        for (int y = 0; y < H; y++)
            col[y] = temp(y,x);

        FFT1D(col, true);

        for (int y = 0; y < H; y++)
            temp(y,x) = col[y];
    }

    // Copy back to image
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            img[y][x] =
                std::clamp(
                    (int)std::abs(temp(y,x)),
                    0,
                    (int)(img.levels()-1)
                );
}


void InverseFilter(ImageObject& img,
                   const Params& p)
{
    ComplexImage G = FFT2D(img);

    int W = img.width();
    int H = img.height();

    for(int u=0;u<H;u++){
        for(int v=0;v<W;v++){

            double a = p.blurLength *
                       cos(p.blurAngle*PI/180.0);
            double b = p.blurLength *
                       sin(p.blurAngle*PI/180.0);

            double val = PI*(u*a + v*b);

            Complex Huv;

            if(std::abs(val) > 1e-8)
                Huv =
                    (sin(val)/val) *
                    std::exp(Complex(0,-val));
            else
                Huv = 1.0;

            if(std::abs(Huv) > 1e-6)
                G(u,v) /= Huv;
        }
    }

    IFFT2D(G, img);
}

void WienerFilter(ImageObject& img,
                  const Params& p)
{
    ComplexImage G = FFT2D(img);

    int W = img.width();
    int H = img.height();

    for(int u=0;u<H;u++){
        for(int v=0;v<W;v++){

            double a = p.blurLength *
                       cos(p.blurAngle*PI/180.0);
            double b = p.blurLength *
                       sin(p.blurAngle*PI/180.0);

            double val = PI*(u*a + v*b);

            Complex Huv;

            if(std::abs(val) > 1e-8)
                Huv =
                    (sin(val)/val) *
                    std::exp(Complex(0,-val));
            else
                Huv = 1.0;

            Complex Hconj = std::conj(Huv);

            double denom =
                std::norm(Huv) + p.wienerK;

            G(u,v) =
                (Hconj / denom) * G(u,v);
        }
    }

    IFFT2D(G, img);
}
void PeriodicFFT(ImageObject& img,
                 const Params& p)
{
    ComplexImage F = FFT2D(img);

    int cx = img.width()/2;
    int cy = img.height()/2;

    // Example fixed notch positions
    NotchReject(F, cx+30, cy, 10);
    NotchReject(F, cx-30, cy, 10);

    IFFT2D(F, img);
}

void NotchReject(ComplexImage& F,
                 int cx,
                 int cy,
                 int radius)
{
    int W = F.width;
    int H = F.height;

    for(int u=0;u<H;u++){
        for(int v=0;v<W;v++){

            int du = u-cy;
            int dv = v-cx;

            if(du*du + dv*dv < radius*radius)
                F(u,v) = 0;
        }
    }
}



NoiseStats ComputeStatistics(const ImageObject& img)
{
    NoiseStats stats;

    uint32_t w = img.width();
    uint32_t h = img.height();
    uint64_t N = (uint64_t)w * h;

    double sum = 0;
    double sumSq = 0;

    uint64_t zeroCount = 0;
    uint64_t maxCount = 0;

    uint16_t maxLevel = img.levels() - 1;

    for(uint32_t y=0;y<h;y++){
        for(uint32_t x=0;x<w;x++){

            double val = img[y][x];

            sum += val;
            sumSq += val*val;

            if(val == 0) zeroCount++;
            if(val == maxLevel) maxCount++;
        }
    }

    stats.mean = sum / N;
    stats.variance = (sumSq / N) - (stats.mean * stats.mean);

    stats.zeroRatio = (double)zeroCount / N;
    stats.maxRatio  = (double)maxCount / N;

    // Compute skewness
    double skewSum = 0;
    for(uint32_t y=0;y<h;y++)
        for(uint32_t x=0;x<w;x++){
            double val = img[y][x];
            skewSum += pow(val - stats.mean, 3);
        }

    skewSum /= N;
    stats.skewness =
        skewSum / pow(stats.variance, 1.5);

    return stats;
}

std::string EstimateNoiseType(const ImageObject& img)
{
    std::cout << "\n=========== ADVANCED NOISE ANALYSIS ===========\n";

    NoiseStats stats = ComputeStatistics(img);
    double kurt = ComputeKurtosis(img);
    bool periodic = DetectPeriodicNoise(img);
    bool uniformHist = DetectUniformNoise(img);
    bool spatialVar = DetectSpatiallyVaryingNoise(img);

    std::cout << "Mean: " << stats.mean << "\n";
    std::cout << "Variance: " << stats.variance << "\n";
    std::cout << "Skewness: " << stats.skewness << "\n";
    std::cout << "Kurtosis: " << kurt << "\n";
    std::cout << "Zero ratio: " << stats.zeroRatio << "\n";
    std::cout << "Max ratio: " << stats.maxRatio << "\n";
    std::cout << "Periodic detected: " << periodic << "\n";
    std::cout << "Uniform histogram: " << uniformHist << "\n";
    std::cout << "Spatial variance instability: " << spatialVar << "\n\n";

    // ============================
    // 1️⃣ Periodic Noise
    // ============================
    if(periodic)
    {
        std::cout << "Detected: Periodic Noise (FFT spikes)\n";
        return "periodic_fft";
    }

    // ============================
    // 2️⃣ Salt & Pepper
    // ============================
    if(stats.zeroRatio > 0.02 || stats.maxRatio > 0.02)
    {
        std::cout << "Detected: Salt & Pepper Noise (impulses)\n";
        return "saltpepper";
    }

    // ============================
    // 3️⃣ Uniform Noise
    // ============================
    if(uniformHist && std::abs(stats.skewness) < 0.3)
    {
        std::cout << "Detected: Uniform Noise (flat histogram)\n";
        return "uniform";
    }

    // ============================
    // 4️⃣ Gaussian Noise
    // Gaussian: skew ≈ 0, kurtosis ≈ 3
    // ============================
    if(std::abs(stats.skewness) < 0.3 &&
       std::abs(kurt - 3.0) < 0.5)
    {
        std::cout << "Detected: Gaussian Noise\n";
        return "gaussian";
    }

    // ============================
    // 5️⃣ Rayleigh Noise
    // Rayleigh: positive skew (~0.6–1), kurtosis ~3.2–4
    // ============================
    if(stats.skewness > 0.5 &&
       stats.skewness < 1.5 &&
       kurt > 3.0 && kurt < 4.5)
    {
        std::cout << "Detected: Rayleigh Noise\n";
        return "rayleigh";
    }

    // ============================
    // 6️⃣ Exponential Noise
    // High skew (>1), high kurtosis (>4)
    // ============================
    if(stats.skewness > 1.0 &&
       kurt > 4.0)
    {
        std::cout << "Detected: Exponential Noise\n";
        return "exponential";
    }

    // ============================
    // 7️⃣ Erlang (Gamma) Noise
    // Moderate skew + moderate kurtosis
    // ============================
    if(stats.skewness > 0.5 &&
       kurt >= 3.0 && kurt <= 6.0)
    {
        std::cout << "Detected: Erlang (Gamma) Noise\n";
        return "erlang";
    }

    // ============================
    // 8️⃣ Spatially Varying Noise
    // ============================
    if(spatialVar)
    {
        std::cout << "Detected: Spatially Varying Noise\n";
        return "adaptive_median";
    }

    // ============================
    // Fallback
    // ============================
    std::cout << "Fallback: Gaussian (default assumption)\n";
    return "gaussian";
}



void MeanFilter(ImageObject& img, int k)
{
    uint32_t w = img.width();
    uint32_t h = img.height();
    std::vector<uint16_t> result(w*h);

    int r = k/2;

    for (uint32_t y = r; y < h-r; y++) {
        for (uint32_t x = r; x < w-r; x++) {

            uint64_t sum = 0;

            for (int j=-r;j<=r;j++)
                for (int i=-r;i<=r;i++)
                    sum += img[y+j][x+i];

            result[y*w+x] = sum / (k*k);
        }
    }

    for (size_t i=0;i<w*h;i++)
        img[i/w][i%w] = result[i];
}

void MedianFilter(ImageObject& img, int k)
{
    uint32_t w = img.width();
    uint32_t h = img.height();
    std::vector<uint16_t> result(w*h);

    int r = k/2;

    for (uint32_t y=r;y<h-r;y++){
        for (uint32_t x=r;x<w-r;x++){

            std::vector<uint16_t> window;

            for(int j=-r;j<=r;j++)
                for(int i=-r;i<=r;i++)
                    window.push_back(img[y+j][x+i]);

            std::sort(window.begin(),window.end());
            result[y*w+x] = window[window.size()/2];
        }
    }

    for (size_t i=0;i<w*h;i++)
        img[i/w][i%w] = result[i];
}

void RemoveGaussianNoise(ImageObject& img,
                         const Params& p)
{
    uint32_t w = img.width();
    uint32_t h = img.height();

    std::vector<uint16_t> result(w*h);
    int r = p.kernelSize/2;

    for(uint32_t y=r;y<h-r;y++){
        for(uint32_t x=r;x<w-r;x++){

            double mean=0, var=0;

            for(int j=-r;j<=r;j++)
                for(int i=-r;i<=r;i++)
                    mean += img[y+j][x+i];

            mean /= (p.kernelSize*p.kernelSize);

            for(int j=-r;j<=r;j++)
                for(int i=-r;i<=r;i++){
                    double d = img[y+j][x+i] - mean;
                    var += d*d;
                }

            var /= (p.kernelSize*p.kernelSize);

            double pixel = img[y][x];

            double restored =
                mean +
                std::max(var - p.noiseVariance,0.0)/std::max(var,1.0)
                * (pixel - mean);

            result[y*w+x] =
                std::clamp((int)restored,0,(int)(img.levels()-1));
        }
    }

    for(size_t i=0;i<w*h;i++)
        img[i/w][i%w] = result[i];
}
void RemoveSaltPepperNoise(ImageObject& img,
                           const Params& p)
{
    MedianFilter(img,p.kernelSize);
}

void RemoveUniformNoise(ImageObject& img,
                        const Params& p)
{
    MeanFilter(img,p.kernelSize);
}

void RemoveRayleighNoise(ImageObject& img,
                         const Params& p)
{
    uint32_t w = img.width();
    uint32_t h = img.height();
    uint16_t maxLevel = img.levels() - 1;

    // --- Step 1: Convert to double buffer ---
    std::vector<double> logBuffer(w*h);

    for(uint32_t y = 0; y < h; y++)
    {
        for(uint32_t x = 0; x < w; x++)
        {
            double val = static_cast<double>(img[y][x]);
            logBuffer[y*w + x] = std::log(1.0 + val);
        }
    }

    // --- Step 2: Mean filter in log domain ---
    int k = p.kernelSize;
    int r = k / 2;

    std::vector<double> filtered(w*h);

    for(uint32_t y = r; y < h-r; y++)
    {
        for(uint32_t x = r; x < w-r; x++)
        {
            double sum = 0.0;

            for(int j = -r; j <= r; j++)
                for(int i = -r; i <= r; i++)
                    sum += logBuffer[(y+j)*w + (x+i)];

            filtered[y*w + x] = sum / (k*k);
        }
    }

    // Preserve borders
    for(uint32_t y = 0; y < h; y++)
    {
        for(uint32_t x = 0; x < w; x++)
        {
            if(y < r || y >= h-r || x < r || x >= w-r)
                filtered[y*w + x] = logBuffer[y*w + x];
        }
    }

    // --- Step 3: Exponentiate back ---
    double minVal = 1e9;
    double maxVal = -1e9;

    for(size_t i = 0; i < filtered.size(); i++)
    {
        filtered[i] = std::exp(filtered[i]);
        minVal = std::min(minVal, filtered[i]);
        maxVal = std::max(maxVal, filtered[i]);
    }

    // --- Step 4: Normalize back to image range ---
    double scale = maxLevel / (maxVal - minVal + 1e-8);

    for(uint32_t y = 0; y < h; y++)
    {
        for(uint32_t x = 0; x < w; x++)
        {
            double norm =
                (filtered[y*w + x] - minVal) * scale;

            img[y][x] =
                std::clamp((int)norm,
                           0,
                           (int)maxLevel);
        }
    }
}


void RemoveErlangNoise(ImageObject& img,
                       const Params& p)
{
    RemoveRayleighNoise(img,p);
}

void RemoveExponentialNoise(ImageObject& img,
                            const Params& p)
{
    MeanFilter(img,p.kernelSize);
}

void RemovePeriodicNoise(ImageObject& img,
                         const Params& p)
{
    for(int i=0;i<3;i++)
        MeanFilter(img,p.kernelSize);
}

double ComputePSNR(const ImageObject& orig,
                   const ImageObject& restored)
{
    double mse = 0;
    uint32_t w = orig.width();
    uint32_t h = orig.height();

    for(uint32_t y=0;y<h;y++)
        for(uint32_t x=0;x<w;x++){
            double d = orig[y][x] - restored[y][x];
            mse += d*d;
        }

    mse /= (w*h);

    if(mse==0) return 100;

    double maxI = orig.levels()-1;

    return 10*log10((maxI*maxI)/mse);
}

void GeometricMeanFilter(ImageObject& img, int k)
{
    uint32_t w = img.width();
    uint32_t h = img.height();
    std::vector<uint16_t> result(w*h);

    int r = k/2;

    for(uint32_t y=r;y<h-r;y++){
        for(uint32_t x=r;x<w-r;x++){

            double product = 1.0;

            for(int j=-r;j<=r;j++)
                for(int i=-r;i<=r;i++)
                    product *= std::max((double)img[y+j][x+i],1.0);

            result[y*w+x] =
                pow(product, 1.0/(k*k));
        }
    }

    for(size_t i=0;i<w*h;i++)
        img[i/w][i%w] = result[i];
}

void HarmonicMeanFilter(ImageObject& img, int k)
{
    uint32_t w = img.width();
    uint32_t h = img.height();
    std::vector<uint16_t> result(w*h);

    int r = k/2;

    for(uint32_t y=r;y<h-r;y++){
        for(uint32_t x=r;x<w-r;x++){

            double sum = 0;

            for(int j=-r;j<=r;j++)
                for(int i=-r;i<=r;i++)
                    sum += 1.0 /
                           std::max((double)img[y+j][x+i],1.0);

            result[y*w+x] =
                (k*k)/sum;
        }
    }

    for(size_t i=0;i<w*h;i++)
        img[i/w][i%w] = result[i];
}
void ContraHarmonicFilter(ImageObject& img,
                          int k,
                          double Q)
{
    uint32_t w = img.width();
    uint32_t h = img.height();
    std::vector<uint16_t> result(w*h);

    int r = k/2;

    for(uint32_t y=r;y<h-r;y++){
        for(uint32_t x=r;x<w-r;x++){

            double num = 0;
            double den = 0;

            for(int j=-r;j<=r;j++)
                for(int i=-r;i<=r;i++){
                    double val = img[y+j][x+i];
                    num += pow(val, Q+1);
                    den += pow(val, Q);
                }

            if(den != 0)
                result[y*w+x] =
                    std::clamp((int)(num/den),
                               0,(int)(img.levels()-1));
        }
    }

    for(size_t i=0;i<w*h;i++)
        img[i/w][i%w] = result[i];
}
void AdaptiveMedianFilter(ImageObject& img,
                          int maxKernel)
{
    uint32_t w = img.width();
    uint32_t h = img.height();

    std::vector<uint16_t> result(w*h);

    for(uint32_t y=0;y<h;y++){
        for(uint32_t x=0;x<w;x++){

            int k = 3;

            while(true){

                int r = k/2;
                std::vector<uint16_t> window;

                for(int j=-r;j<=r;j++)
                    for(int i=-r;i<=r;i++){
                        int yy = std::clamp((int)y+j,0,(int)h-1);
                        int xx = std::clamp((int)x+i,0,(int)w-1);
                        window.push_back(img[yy][xx]);
                    }

                std::sort(window.begin(),window.end());

                uint16_t zmin = window.front();
                uint16_t zmax = window.back();
                uint16_t zmed = window[window.size()/2];
                uint16_t zxy = img[y][x];

                if(zmed > zmin && zmed < zmax){
                    if(zxy > zmin && zxy < zmax)
                        result[y*w+x] = zxy;
                    else
                        result[y*w+x] = zmed;
                    break;
                }
                else{
                    k += 2;
                    if(k > maxKernel){
                        result[y*w+x] = zmed;
                        break;
                    }
                }
            }
        }
    }

    for(size_t i=0;i<w*h;i++)
        img[i/w][i%w] = result[i];
}


bool DetectPeriodicNoise(const ImageObject& img)
{
    ComplexImage F = FFT2D(img);

    int W = img.width();
    int H = img.height();

    double meanMag = 0.0;
    double maxMag = 0.0;

    // Compute magnitude statistics
    for(int u = 0; u < H; u++)
    {
        for(int v = 0; v < W; v++)
        {
            double mag = std::abs(F(u,v));
            meanMag += mag;
            maxMag = std::max(maxMag, mag);
        }
    }

    meanMag /= (W * H);

    // If peak is significantly larger than average → periodic
    if(maxMag > 10.0 * meanMag)
        return true;

    return false;
}

bool DetectUniformNoise(const ImageObject& img)
{
    auto hist = img.computeHistogram();
    double mean = 0.0;
    double var = 0.0;

    int L = hist.size();

    double total = 0;
    for(auto h : hist) total += h;

    for(int i = 0; i < L; i++)
        mean += hist[i];

    mean /= L;

    for(int i = 0; i < L; i++)
    {
        double d = hist[i] - mean;
        var += d*d;
    }

    var /= L;

    // Small variance → flat histogram
    if(var < 0.01 * mean * mean)
        return true;

    return false;
}

double ComputeKurtosis(const ImageObject& img)
{
    uint32_t w = img.width();
    uint32_t h = img.height();
    uint64_t N = (uint64_t)w * h;

    double mean = 0;
    for(uint32_t y=0;y<h;y++)
        for(uint32_t x=0;x<w;x++)
            mean += img[y][x];

    mean /= N;

    double var = 0;
    for(uint32_t y=0;y<h;y++)
        for(uint32_t x=0;x<w;x++)
        {
            double d = img[y][x] - mean;
            var += d*d;
        }

    var /= N;

    double kurt = 0;
    for(uint32_t y=0;y<h;y++)
        for(uint32_t x=0;x<w;x++)
        {
            double d = img[y][x] - mean;
            kurt += pow(d,4);
        }

    kurt /= N;
    kurt /= (var * var + 1e-8);

    return kurt;
}

bool DetectGaussianNoise(const ImageObject& img)
{
    double kurt = ComputeKurtosis(img);

    if(std::abs(kurt - 3.0) < 0.5)
        return true;

    return false;
}

bool DetectSpatiallyVaryingNoise(const ImageObject& img)
{
    int block = 32;

    int W = img.width();
    int H = img.height();

    std::vector<double> blockVars;

    for(int by = 0; by < H; by += block)
    {
        for(int bx = 0; bx < W; bx += block)
        {
            double mean = 0;
            double var = 0;
            int count = 0;

            for(int y = by; y < std::min(by+block,H); y++)
                for(int x = bx; x < std::min(bx+block,W); x++)
                {
                    mean += img[y][x];
                    count++;
                }

            mean /= count;

            for(int y = by; y < std::min(by+block,H); y++)
                for(int x = bx; x < std::min(bx+block,W); x++)
                {
                    double d = img[y][x] - mean;
                    var += d*d;
                }

            var /= count;

            blockVars.push_back(var);
        }
    }

    // variance of block variances
    double meanVar = 0;
    for(auto v : blockVars) meanVar += v;
    meanVar /= blockVars.size();

    double varVar = 0;
    for(auto v : blockVars)
    {
        double d = v - meanVar;
        varVar += d*d;
    }

    varVar /= blockVars.size();

    if(varVar > 0.5 * meanVar)
        return true;

    return false;
}


};


class INPUT_HANDLER {
public:

    static void RUN()
    {
        IMAGE_RESTORATION engine;

        while (true)
        {
            std::string strategy;
            std::string inputFile;
            std::string outputFile;

            std::cout << "\n===== IMAGE RESTORATION ENGINE =====\n";
            std::cout << "Available strategies:\n";
            std::cout << " auto | estimate noise\n";

            std::cout << " gaussian | saltpepper | uniform\n";
            std::cout << " rayleigh | erlang | exponential\n";
            std::cout << " geometric | harmonic | contraharmonic\n";
            std::cout << " adaptive_median\n";
            std::cout << "  inverse | wiener_freq | periodic_fft\n";
            std::cout << "Type 'exit' to quit\n> ";

            std::cin >> strategy;

            if (strategy == "exit" || strategy == "quit")
            {
                std::cout << "Exiting...\n";
                return;
            }

            std::cout << "Input file: ";
            std::cin >> inputFile;

            std::cout << "Output file: ";
            std::cin >> outputFile;

            IMAGE_RESTORATION::Params params;

            // ---- Spatial Filters ----



            if (strategy == "gaussian")
            {
                std::cout << "Kernel size (odd): ";
                std::cin >> params.kernelSize;

                std::cout << "Noise variance: ";
                std::cin >> params.noiseVariance;
            }
            else if(strategy == "auto")
            {
                engine.AutoRestore(inputFile, outputFile);
                continue;
            }
            else if (strategy == "saltpepper" ||
                     strategy == "uniform" ||
                     strategy == "rayleigh" ||
                     strategy == "erlang" ||
                     strategy == "exponential" ||
                     strategy == "geometric" ||
                     strategy == "harmonic")
            {
                std::cout << "Kernel size (odd): ";
                std::cin >> params.kernelSize;
            }

            else if (strategy == "contraharmonic")
            {
                std::cout << "Kernel size (odd): ";
                std::cin >> params.kernelSize;

                std::cout << "Enter Q value:\n";
                std::cout << "Q > 0 removes pepper\n";
                std::cout << "Q < 0 removes salt\n> ";
                std::cin >> params.Q;
            }

            else if (strategy == "adaptive_median")
            {
                std::cout << "Maximum kernel size (odd, e.g. 7 or 9): ";
                std::cin >> params.maxKernel;
            }
            else if (strategy == "inverse" ||
                     strategy == "wiener_freq" ||
                     strategy == "periodic_fft")
            {
                std::cout << "Blur length: ";
                std::cin >> params.blurLength;

                std::cout << "Blur angle (degrees): ";
                std::cin >> params.blurAngle;

                if(strategy == "wiener_freq")
                {
                    std::cout << "Wiener K: ";
                    std::cin >> params.wienerK;
                }
            }
            else if(strategy == "estimate")
            {
                ImageObject img(inputFile);
                std::string detected = engine.EstimateNoiseTypePublic(img);
                std::cout<<"\n Estimation Complete \n";
                continue;
            }


            else
            {
                std::cout << "Unknown strategy.\n";
                continue;
            }

            // --- Validation ---

            if (params.kernelSize % 2 == 0 && params.kernelSize != 0)
            {
                std::cout << "Kernel must be odd. Auto-fixing.\n";
                params.kernelSize += 1;
            }

            if (params.maxKernel % 2 == 0 && params.maxKernel != 0)
            {
                std::cout << "Max kernel must be odd. Auto-fixing.\n";
                params.maxKernel += 1;
            }

            std::cout << "\nProcessing...\n";

            engine.ExecuteStrategy(strategy,
                                   inputFile,
                                   outputFile,
                                   params);

            std::cout << "Done.\n";
        }
    }
};



int main(){
    INPUT_HANDLER::RUN();
    return 0;
}
