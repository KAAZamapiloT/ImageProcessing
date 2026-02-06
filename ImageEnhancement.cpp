#include <tiffio.h>
#include <iostream>
#include<vector>
#include<string>
#include<cstdint>
#include <stdexcept>
#include <cmath>
#include <algorithm>
// compilation helpers
/*
 *


 cl ImageEnhancement.cpp ^
 /std:c++20 /EHsc ^
 /I C:\vcpkg\installed\x64-windows\include ^
 /link /LIBPATH:C:\vcpkg\installed\x64-windows\lib tiff.lib



*/
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


// this class wil handle all inversion realted tasks of image
class InvertImageEvent{
  public:
  InvertImageEvent(const std::string& in,
                       const std::string& out)
          : m_InputPath(in), m_OutputPath(out) {}
  std::string Name="Invert";
  std::string m_InputPath;
  std::string m_OutputPath;

  void execute() {
      ImageObject img(m_InputPath);
      invert(img);
      img.saveTIFF8bit(m_OutputPath);
  }
private:
void invert(ImageObject& img) {
        const uint32_t L = img.levels();

        for (uint32_t y= 0;y<img.height();y++) {
            for (uint32_t x=0;x<img.width();x++) {
                img[y][x]=L-1-img[y][x];
            }
        }
    }
};


class LogTransformEvent{
    public:
    LogTransformEvent(const std::string& in,
                         const std::string& out)
            : m_InputPath(in), m_OutputPath(out) {}
    std::string Name="log";
    std::string m_InputPath;
    std::string m_OutputPath;

    void execute() {
        ImageObject img(m_InputPath);
        LOGGG(img);
        img.saveTIFF8bit(m_OutputPath);
    }
  private:
  void LOGGG(ImageObject& img) {
          const uint32_t L = img.levels();
          float c=(static_cast<double>(L-1)/std::log(static_cast<double>(L)));
          for (uint32_t y= 0;y<img.height();y++) {
              for (uint32_t x=0;x<img.width();x++) {
                  img[y][x]=c*std::log(img[y][x]+1);
              }
          }
      }
};



class GammaTransformEvent{

    public:
    GammaTransformEvent(const std::string& in,
                         const std::string& out,double gamma)
            : m_InputPath(in), m_OutputPath(out),m_Gamma(gamma) {}
    std::string Name="gamma";
    std::string m_InputPath;
    std::string m_OutputPath;

    void execute() {
        ImageObject img(m_InputPath);
        gama(img);
        img.saveTIFF8bit(m_OutputPath);
    }
  private:
  double m_Gamma;
  void gama(ImageObject& img) {
      uint64_t diffCount = 0;

      // Find actual min/max values in the image
      uint16_t minVal = 65535, maxVal = 0;
      for (uint32_t y = 0; y < img.height(); y++) {
          for (uint32_t x = 0; x < img.width(); x++) {
              uint16_t val = img[y][x];
              minVal = std::min(minVal, val);
              maxVal = std::max(maxVal, val);
          }
      }

      std::cout << "Input range: [" << minVal << ", " << maxVal << "]\n";

      // Work in normalized [0,1] space, then scale back
      const double inputMax = static_cast<double>(maxVal);

      // Create LUT for input range
      std::vector<uint16_t> lut(static_cast<size_t>(maxVal) + 1);
      for (uint32_t i = 0; i <= maxVal; i++) {
          // Normalize to [0, 1]
          double normalized = static_cast<double>(i) / inputMax;

          // Apply gamma
          double transformed = std::pow(normalized, m_Gamma);

          // Scale back to original range
          double scaled = transformed * inputMax;

          lut[i] = static_cast<uint16_t>(std::round(scaled));
      }

      // Apply transformation
      for (uint32_t y = 0; y < img.height(); y++) {
          for (uint32_t x = 0; x < img.width(); x++) {
              uint16_t original = img[y][x];
              uint16_t transformed = lut[original];

              if (original != transformed)
                  diffCount++;

              img[y][x] = transformed;
          }
      }

      std::cout << "Different pixels: " << diffCount << "\n";
  }
};


class PieceWiseContrastEvent{
    public:
    std::string name="PieceWise";
    std::string m_InputPath;
    std::string m_OutputPath;
    PieceWiseContrastEvent(const std::string& in,
                              const std::string& out,
                              uint16_t r1, uint16_t s1,
                              uint16_t r2, uint16_t s2)
            : m_InputPath(in), m_OutputPath(out),
              r1(r1), s1(s1), r2(r2), s2(s2) {}

    void execute() {
        ImageObject img(m_InputPath);
        contrast(img);
        img.saveTIFF8bit(m_OutputPath);
    }

    private:

    uint16_t r1, s1, r2, s2;

    std::vector<uint16_t> buildLUT(uint32_t L) {
        const double maxVal = static_cast<double>(L - 1);

        if (r1 == 0 || r2 <= r1 || r2 >= maxVal)
            throw std::runtime_error("Invalid contrast parameters");

        if (s1 > maxVal || s2 > maxVal)
            throw std::runtime_error("Output levels exceed bit depth");

        std::vector<uint16_t> lut(L);

        for (uint32_t r = 0; r < L; r++) {
            double s;

            if (r <= r1) {
                s = (s1 / static_cast<double>(r1)) * r;
            }
            else if (r <= r2) {
                s = ((s2 - s1) / static_cast<double>(r2 - r1)) * (r - r1) + s1;
            }
            else {
                s = ((maxVal - s2) / (maxVal - r2)) * (r - r2) + s2;
            }

            lut[r] = static_cast<uint16_t>(
                std::lround(std::clamp(s, 0.0, maxVal))
            );
        }

        return lut;
    }

        void contrast(ImageObject& img) {
            const uint32_t L = img.levels();
            auto lut = buildLUT(L);

            uint32_t h = img.height();
            uint32_t w = img.width();

            for (uint32_t y = 0; y < h; y++) {
                for (uint32_t x = 0; x < w; x++) {
                    img[y][x] = lut[img[y][x]];
                }
            }
        }

};

// Applying intensity ramp
class IntensityRampEvent {
public:
    IntensityRampEvent(const std::string& in,
                       const std::string& out,
                       uint16_t start,
                       uint16_t end)
        : m_InputPath(in),
          m_OutputPath(out),
          m_Start(start),
          m_End(end) {}

    std::string name = "ramp";

    void execute() {
        ImageObject img(m_InputPath);
        applyRamp(img);
        img.saveTIFF(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    uint16_t m_Start, m_End;

    std::vector<uint16_t> buildRampLUT(uint32_t L,
                                       uint16_t r1,
                                       uint16_t r2)
    {
        if (r2 <= r1)
            throw std::runtime_error("Invalid ramp range");

        const double maxVal = static_cast<double>(L - 1);
        const double slope  = maxVal / (r2 - r1);

        std::vector<uint16_t> lut(L);

        for (uint32_t r = 0; r < L; r++) {
            double s;
            if (r < r1)
                s = 0.0;
            else if (r > r2)
                s = maxVal;
            else
                s = slope * (r - r1);
            lut[r] = static_cast<uint16_t>(
                std::lround(std::clamp(s, 0.0, maxVal))
            );
        }
        return lut;
    }

    void applyRamp(ImageObject& img) {
        auto lut = buildRampLUT(img.levels(), m_Start, m_End);
        uint32_t h = img.height();
        uint32_t w = img.width();

        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                img[y][x] = lut[img[y][x]];
            }
        }
    }
};

class IntensityLevelSlicingEvent {
public:

enum class Mode {
    WITHOUT_BACKGROUND,
    WITH_BACKGROUND
};
    IntensityLevelSlicingEvent(const std::string& in,
                               const std::string& out,
                               uint16_t r1,
                               uint16_t r2,
                               uint16_t k,
                               std::string mode)
        : m_InputPath(in),
          m_OutputPath(out),
          m_R1(r1),
          m_R2(r2),
          m_K(k){

              if(mode=="bg"){
                  m_Mode = Mode::WITH_BACKGROUND;
              }else{
                  m_Mode = Mode::WITHOUT_BACKGROUND;
              }
          }

    std::string name = "slice";

    void execute() {
        ImageObject img(m_InputPath);
        applySlicing(img);
        img.saveTIFF(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    uint16_t m_R1, m_R2;
    uint16_t m_K;
    Mode m_Mode;

    std::vector<uint16_t> buildLUT(uint32_t L) {
        const double maxVal = static_cast<double>(L-1);
        if (m_R2 < m_R1){
            throw std::runtime_error("Invalid slicing range");
        }
        if (m_K > maxVal){
            throw std::runtime_error("Slice value exceeds bit depth");
        }
        std::vector<uint16_t> lut(L);

        for (uint32_t r = 0; r < L; r++) {
            if (r >= m_R1 && r <= m_R2) {
                lut[r] = m_K;
            } else {
                if (m_Mode == Mode::WITH_BACKGROUND)
                    lut[r] = static_cast<uint16_t>(r);
                else
                    lut[r] = 0;
            }
        }

        return lut;
    }

    void applySlicing(ImageObject& img) {
        auto lut = buildLUT(img.levels());
        uint32_t h = img.height();
        uint32_t w = img.width();
        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                img[y][x] = lut[img[y][x]];
            }
        }
    }
};


class BitPlaneSliceEvent {
public:
    enum class Mode {
        WITHOUT_BACKGROUND,
        WITH_BACKGROUND
    };

    BitPlaneSliceEvent(const std::string& in,
                       const std::string& out,
                       uint16_t bitIndex,
                       std::string mode)
        : m_InputPath(in),
          m_OutputPath(out),
          m_Bit(bitIndex)
    {
        m_Mode = (mode == "bg") ?
                 Mode::WITH_BACKGROUND :
                 Mode::WITHOUT_BACKGROUND;
    }

    void execute() {
        ImageObject img(m_InputPath);
        apply(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    uint16_t m_Bit;
    Mode m_Mode;

    void apply(ImageObject& img) {
        const uint32_t h = img.height();
        const uint32_t w = img.width();
        const uint16_t maxVal = (1u << img.bits()) - 1;

        if (m_Bit >= img.bits())
            throw std::runtime_error("Invalid bit index");

        for (uint32_t y = 0; y < h; y++) {
            for (uint32_t x = 0; x < w; x++) {
                uint16_t pixel = img[y][x];
                uint16_t bit = (pixel >> m_Bit) & 1;

                if (m_Mode == Mode::WITH_BACKGROUND) {
                    img[y][x] = bit ? maxVal : pixel;
                } else {
                    img[y][x] = bit ? maxVal : 0;
                }
            }
        }
    }
};



class HistogramEqualizationEvent {
public:
    HistogramEqualizationEvent(const std::string& in,
                               const std::string& out)
        : m_InputPath(in), m_OutputPath(out) {}

    void execute() {
        ImageObject img(m_InputPath);
        equalize(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;

    void equalize(ImageObject& img) {
        const uint32_t L = img.levels();
        const uint64_t N = img.width() * img.height();

        auto hist = img.computeHistogram();

        std::vector<double> cdf(L, 0.0);
        cdf[0] = static_cast<double>(hist[0]) / N;

        for (uint32_t i = 1; i < L; i++) {
            cdf[i] = cdf[i - 1] + static_cast<double>(hist[i]) / N;
        }
        std::vector<uint16_t> lut(L);
        for (uint32_t i = 0; i < L; i++) {
            lut[i] = static_cast<uint16_t>(
                std::lround(cdf[i] * (L - 1))
            );
        }
        for (uint32_t y = 0; y < img.height(); y++) {
            for (uint32_t x = 0; x < img.width(); x++) {
                img[y][x] = lut[img[y][x]];
            }
        }
    }
};

class HistogramStatsEvent {
public:
    HistogramStatsEvent(const std::string& in)
        : m_InputPath(in) {}

    void execute() {
        ImageObject img(m_InputPath);
        analyze(img);
    }

private:
    std::string m_InputPath;

    void analyze(const ImageObject& img) {
        const uint32_t L = img.levels();
        const uint64_t N = static_cast<uint64_t>(img.width()) * img.height();

        auto hist = img.computeHistogram();

        // ---- basic stats ----
        uint32_t minLevel = L, maxLevel = 0;
        double mean = 0.0;
        double variance = 0.0;
        double entropy = 0.0;

        for (uint32_t i = 0; i < L; i++) {
            if (hist[i] > 0) {
                minLevel = std::min(minLevel, i);
                maxLevel = std::max(maxLevel, i);
            }
            mean += i * static_cast<double>(hist[i]);
        }
        mean /= N;

        for (uint32_t i = 0; i < L; i++) {
            double diff = i - mean;
            variance += diff * diff * hist[i];

            if (hist[i] > 0) {
                double p = static_cast<double>(hist[i]) / N;
                entropy -= p * std::log2(p);
            }
        }
        variance /= N;

        // ---- print results ----
        std::cout << "\n--- Histogram Statistics ---\n";
        std::cout << "Image size      : " << img.width() << " x " << img.height() << "\n";
        std::cout << "Bit depth       : " << img.bits() << "\n";
        std::cout << "Levels          : " << L << "\n";
        std::cout << "Used range      : [" << minLevel << ", " << maxLevel << "]\n";
        std::cout << "Mean intensity  : " << mean << "\n";
        std::cout << "Variance        : " << variance << "\n";
        std::cout << "Entropy (bits)  : " << entropy << "\n";

        printCompactHistogram(hist, N);
    }

    void printCompactHistogram(const std::vector<uint64_t>& hist, uint64_t N) {
        const uint32_t bins = 16;   // compact view
        const uint32_t L = hist.size();
        const uint32_t step = L / bins;

        std::cout << "\nHistogram (compressed):\n";

        for (uint32_t b = 0; b < bins; b++) {
            uint64_t count = 0;
            uint32_t start = b * step;
            uint32_t end = (b + 1) * step;

            for (uint32_t i = start; i < end; i++)
                count += hist[i];

            double percent = (100.0 * count) / N;
            std::cout << "[" << start << "-" << end - 1 << "] : ";

            int bars = static_cast<int>(percent / 2);
            for (int i = 0; i < bars; i++)
                std::cout << "#";

            std::cout << " (" << percent << "%)\n";
        }
    }
};

class HistogramMatchingEvent {
public:
    HistogramMatchingEvent(const std::string& src,
                           const std::string& ref,
                           const std::string& out)
        : m_SrcPath(src),
          m_RefPath(ref),
          m_OutPath(out) {}

    void execute() {
        ImageObject src(m_SrcPath);
        ImageObject ref(m_RefPath);

        if (src.levels() != ref.levels())
            throw std::runtime_error("Source and reference must have same bit depth");

        match(src, ref);
        src.saveTIFF8bit(m_OutPath);
    }

private:
    std::string m_SrcPath, m_RefPath, m_OutPath;

    void match(ImageObject& src, const ImageObject& ref) {
        const uint32_t L = src.levels();
        const uint64_t Ns = src.width() * src.height();
        const uint64_t Nr = ref.width() * ref.height();

        auto histS = src.computeHistogram();
        auto histR = ref.computeHistogram();

        // ---- compute CDFs ----
        std::vector<double> cdfS(L, 0.0), cdfR(L, 0.0);

        cdfS[0] = (double)histS[0] / Ns;
        cdfR[0] = (double)histR[0] / Nr;

        for (uint32_t i = 1; i < L; i++) {
            cdfS[i] = cdfS[i - 1] + (double)histS[i] / Ns;
            cdfR[i] = cdfR[i - 1] + (double)histR[i] / Nr;
        }

        // ---- build mapping LUT ----
        std::vector<uint16_t> lut(L);

        for (uint32_t s = 0; s < L; s++) {
            double val = cdfS[s];
            uint32_t r = 0;

            while (r < L - 1 && cdfR[r] < val)
                r++;

            lut[s] = static_cast<uint16_t>(r);
        }

        // ---- apply mapping ----
        for (uint32_t y = 0; y < src.height(); y++) {
            for (uint32_t x = 0; x < src.width(); x++) {
                src[y][x] = lut[src[y][x]];
            }
        }
    }
};


class LocalHistogramEnhancementEvent {
public:
    LocalHistogramEnhancementEvent(const std::string& in,
                                   const std::string& out,
                                   uint32_t windowSize)
        : m_InputPath(in),
          m_OutputPath(out),
          m_Window(windowSize)
    {
        if (windowSize < 3 || windowSize % 2 == 0)
            throw std::runtime_error("Window size must be odd and >= 3");
    }

    void execute() {
        ImageObject img(m_InputPath);
        enhance(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    uint32_t m_Window;

    void enhance(ImageObject& img) {
        const uint32_t L = img.levels();
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const int r = static_cast<int>(m_Window / 2);

        std::vector<uint16_t> out(W * H);

        for (int y = 0; y < static_cast<int>(H); y++) {
            for (int x = 0; x < static_cast<int>(W); x++) {

                // ---- build local histogram ----
                std::vector<uint64_t> hist(L, 0);
                uint64_t total = 0;

                for (int dy = -r; dy <= r; dy++) {
                    for (int dx = -r; dx <= r; dx++) {
                        int yy = std::clamp(y + dy, 0, (int)H - 1);
                        int xx = std::clamp(x + dx, 0, (int)W - 1);

                        uint16_t v = img[yy][xx];
                        hist[v]++;
                        total++;
                    }
                }

                // ---- local CDF for center pixel ----
                uint16_t center = img[y][x];
                uint64_t cdf = 0;

                for (uint32_t i = 0; i <= center; i++)
                    cdf += hist[i];

                double mapped =
                    static_cast<double>(cdf) * (L - 1) / total;

                out[y * W + x] =
                    static_cast<uint16_t>(std::round(mapped));
            }
        }

        // ---- write back ----
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++)
                img[y][x] = out[y * W + x];
    }
};

class BoxSmoothingEvent {
public:
    BoxSmoothingEvent(const std::string& in,
                      const std::string& out,
                      uint32_t kernel)
        : m_InputPath(in),
          m_OutputPath(out),
          m_Kernel(kernel)
    {
        if (kernel < 3 || kernel % 2 == 0)
            throw std::runtime_error("Kernel size must be odd and >= 3");
    }

    void execute() {
        ImageObject img(m_InputPath);
        smooth(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    uint32_t m_Kernel;

    void smooth(ImageObject& img) {
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const uint32_t L = img.levels();
        const int r = static_cast<int>(m_Kernel / 2);

        std::vector<uint16_t> out(W * H, 0);

        for (int y = 0; y < (int)H; y++) {
            for (int x = 0; x < (int)W; x++) {

                uint64_t sum = 0;
                uint32_t count = 0;

                for (int dy = -r; dy <= r; dy++) {
                    for (int dx = -r; dx <= r; dx++) {
                        int yy = std::clamp(y + dy, 0, (int)H - 1);
                        int xx = std::clamp(x + dx, 0, (int)W - 1);

                        sum += img[yy][xx];
                        count++;
                    }
                }

                out[y * W + x] =
                    static_cast<uint16_t>(sum / count);
            }
        }

        // write back
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++)
                img[y][x] = out[y * W + x];
    }
};

class GaussianLowPassEvent {
public:
    GaussianLowPassEvent(const std::string& in,
                          const std::string& out,
                          uint32_t kernelSize,
                          double sigma)
        : m_InputPath(in),
          m_OutputPath(out),
          m_KernelSize(kernelSize),
          m_Sigma(sigma)
    {
        if (kernelSize < 3 || kernelSize % 2 == 0)
            throw std::runtime_error("Gaussian kernel size must be odd and >= 3");
        if (sigma <= 0.0)
            throw std::runtime_error("Sigma must be > 0");
    }

    void execute() {
        ImageObject img(m_InputPath);
        applyGaussian(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    uint32_t m_KernelSize;
    double m_Sigma;

    std::vector<double> buildGaussianKernel1D() {
        int r = static_cast<int>(m_KernelSize / 2);
        std::vector<double> kernel(m_KernelSize);
        double sum = 0.0;

        for (int i = -r; i <= r; i++) {
            double v = std::exp(-(i * i) / (2.0 * m_Sigma * m_Sigma));
            kernel[i + r] = v;
            sum += v;
        }

        // normalize
        for (double& v : kernel)
            v /= sum;

        return kernel;
    }

    void applyGaussian(ImageObject& img) {
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const uint32_t L = img.levels();
        const int r = static_cast<int>(m_KernelSize / 2);

        auto kernel = buildGaussianKernel1D();

        // ---- horizontal pass ----
        std::vector<uint16_t> temp(W * H);

        for (int y = 0; y < (int)H; y++) {
            for (int x = 0; x < (int)W; x++) {
                double acc = 0.0;

                for (int k = -r; k <= r; k++) {
                    int xx = std::clamp(x + k, 0, (int)W - 1);
                    acc += kernel[k + r] * img[y][xx];
                }

                temp[y * W + x] =
                    static_cast<uint16_t>(std::clamp(acc, 0.0, (double)(L - 1)));
            }
        }

        // ---- vertical pass ----
        for (int y = 0; y < (int)H; y++) {
            for (int x = 0; x < (int)W; x++) {
                double acc = 0.0;

                for (int k = -r; k <= r; k++) {
                    int yy = std::clamp(y + k, 0, (int)H - 1);
                    acc += kernel[k + r] * temp[yy * W + x];
                }

                img[y][x] =
                    static_cast<uint16_t>(std::clamp(acc, 0.0, (double)(L - 1)));
            }
        }
    }
};

class HighPassSharpenEvent {
public:
    HighPassSharpenEvent(const std::string& in,
                         const std::string& out,
                         double strength = 1.0)
        : m_InputPath(in),
          m_OutputPath(out),
          m_Strength(strength)
    {
        if (strength <= 0.0)
            throw std::runtime_error("Sharpen strength must be > 0");
    }

    void execute() {
        ImageObject img(m_InputPath);
        sharpen(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    double m_Strength;

    void sharpen(ImageObject& img) {
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const uint32_t L = img.levels();

        // 4-connected Laplacian kernel
        const int K[3][3] = {
            {  0, -1,  0 },
            { -1,  4, -1 },
            {  0, -1,  0 }
        };

        std::vector<int> lap(W * H, 0);

        // ---- compute Laplacian (high-pass) ----
        for (int y = 1; y < (int)H - 1; y++) {
            for (int x = 1; x < (int)W - 1; x++) {

                int sum = 0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        sum += K[ky + 1][kx + 1] * img[y + ky][x + kx];
                    }
                }

                lap[y * W + x] = sum;
            }
        }

        // ---- add scaled high-pass back ----
        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {

                double sharpened =
                    img[y][x] + m_Strength * lap[y * W + x];

                img[y][x] = static_cast<uint16_t>(
                    std::clamp(sharpened, 0.0, (double)(L - 1))
                );
            }
        }
    }
};

class UnsharpHighboostEvent {
public:
    UnsharpHighboostEvent(const std::string& in,
                          const std::string& out,
                          double A)
        : m_InputPath(in),
          m_OutputPath(out),
          m_A(A)
    {
        if (A < 1.0)
            throw std::runtime_error("A must be >= 1.0");
    }

    void execute() {
        ImageObject img(m_InputPath);
        apply(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    double m_A;

    void apply(ImageObject& img) {
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const uint32_t L = img.levels();

        // Gaussian kernel
        const int G[3][3] = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };
        const int Gsum = 16;

        std::vector<double> blurred(W * H, 0.0);

        // ---- Gaussian blur ----
        for (int y = 1; y < (int)H - 1; y++) {
            for (int x = 1; x < (int)W - 1; x++) {

                int acc = 0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        acc += G[ky + 1][kx + 1] * img[y + ky][x + kx];
                    }
                }

                blurred[y * W + x] = acc / (double)Gsum;
            }
        }

        // ---- Unsharp / Highboost ----
        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {

                double original = img[y][x];
                double mask = original - blurred[y * W + x];

                double result =
                    m_A * original - blurred[y * W + x];


                img[y][x] = static_cast<uint16_t>(
                    std::clamp(result, 0.0, (double)(L - 1))
                );
            }
        }
    }
};

class GradientEdgeEnhancementEvent {
public:
    GradientEdgeEnhancementEvent(const std::string& in,
                                 const std::string& out,
                                 double k)
        : m_InputPath(in),
          m_OutputPath(out),
          m_K(k)
    {
        if (k <= 0.0)
            throw std::runtime_error("k must be > 0");
    }

    void execute() {
        ImageObject img(m_InputPath);
        enhance(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    double m_K;

    void enhance(ImageObject& img) {
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const uint32_t L = img.levels();

        // Sobel kernels
        const int Sx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };

        const int Sy[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        std::vector<uint16_t> out(W * H);

        for (int y = 1; y < (int)H - 1; y++) {
            for (int x = 1; x < (int)W - 1; x++) {

                int gx = 0, gy = 0;

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        uint16_t p = img[y + ky][x + kx];
                        gx += Sx[ky + 1][kx + 1] * p;
                        gy += Sy[ky + 1][kx + 1] * p;
                    }
                }

                double grad = std::abs(gx) + std::abs(gy);

                double enhanced =
                    img[y][x] + m_K * grad;

                out[y * W + x] = static_cast<uint16_t>(
                    std::clamp(enhanced, 0.0, (double)(L - 1))
                );
            }
        }

        // write back
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++)
                img[y][x] = out[y * W + x];
    }
};

class LaplacianSobelSharpenEvent {
public:
    LaplacianSobelSharpenEvent(const std::string& in,
                               const std::string& lapOut,
                               const std::string& sharpOut,
                               const std::string& sobelOut)
        : m_Input(in),
          m_LapOut(lapOut),
          m_SharpOut(sharpOut),
          m_SobelOut(sobelOut) {}

    void execute() {
        ImageObject img(m_Input);

        ImageObject lap = img;
        ImageObject sharp = img;
        ImageObject sobel = img;

        applyLaplacian(img, lap);
        applySharpen(img, lap, sharp);
        applySobel(img, sobel);

        lap.saveTIFF8bit(m_LapOut);
        sharp.saveTIFF8bit(m_SharpOut);
        sobel.saveTIFF8bit(m_SobelOut);
    }

private:
    std::string m_Input, m_LapOut, m_SharpOut, m_SobelOut;

    void applyLaplacian(const ImageObject& in, ImageObject& out) {
        const int k[3][3] = {
            { 0, -1,  0},
            {-1,  4, -1},
            { 0, -1,  0}
        };

        int H = in.height();
        int W = in.width();

        for (int y = 1; y < H - 1; y++) {
            for (int x = 1; x < W - 1; x++) {
                int sum = 0;
                for (int j = -1; j <= 1; j++)
                    for (int i = -1; i <= 1; i++)
                        sum += k[j+1][i+1] * in[y+j][x+i];

                out[y][x] = std::clamp(sum, 0, (int)in.levels() - 1);
            }
        }
    }

    void applySharpen(const ImageObject& orig,
                      const ImageObject& lap,
                      ImageObject& out) {

        for (uint32_t y = 0; y < orig.height(); y++)
            for (uint32_t x = 0; x < orig.width(); x++) {
                int v = orig[y][x] + lap[y][x];
                out[y][x] = std::clamp(v, 0, (int)orig.levels() - 1);
            }
    }

    void applySobel(const ImageObject& in, ImageObject& out) {
        const int gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };
        const int gy[3][3] = {
            {-1,-2,-1},
            { 0, 0, 0},
            { 1, 2, 1}
        };

        int H = in.height();
        int W = in.width();

        for (int y = 1; y < H - 1; y++) {
            for (int x = 1; x < W - 1; x++) {
                int sx = 0, sy = 0;

                for (int j = -1; j <= 1; j++)
                    for (int i = -1; i <= 1; i++) {
                        sx += gx[j+1][i+1] * in[y+j][x+i];
                        sy += gy[j+1][i+1] * in[y+j][x+i];
                    }

                int mag = std::abs(sx) + std::abs(sy);
                out[y][x] = std::clamp(mag, 0, (int)in.levels() - 1);
            }
        }
    }
};

class MedianFilterEvent {
public:
    MedianFilterEvent(const std::string& in,
                      const std::string& out,
                      uint32_t windowSize)
        : m_InputPath(in),
          m_OutputPath(out),
          m_Window(windowSize)
    {
        if (windowSize < 3 || windowSize % 2 == 0)
            throw std::runtime_error("Median window must be odd and >= 3");
    }

    void execute() {
        ImageObject img(m_InputPath);
        apply(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    uint32_t m_Window;

    void apply(ImageObject& img) {
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const uint32_t L = img.levels();
        const int r = static_cast<int>(m_Window / 2);

        std::vector<uint16_t> out(W * H);
        std::vector<uint16_t> neighborhood;
        neighborhood.reserve(m_Window * m_Window);

        for (int y = 0; y < (int)H; y++) {
            for (int x = 0; x < (int)W; x++) {

                neighborhood.clear();

                for (int dy = -r; dy <= r; dy++) {
                    for (int dx = -r; dx <= r; dx++) {
                        int yy = std::clamp(y + dy, 0, (int)H - 1);
                        int xx = std::clamp(x + dx, 0, (int)W - 1);
                        neighborhood.push_back(img[yy][xx]);
                    }
                }

                std::nth_element(
                    neighborhood.begin(),
                    neighborhood.begin() + neighborhood.size() / 2,
                    neighborhood.end()
                );

                out[y * W + x] =
                    neighborhood[neighborhood.size() / 2];
            }
        }

        // write back
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++)
                img[y][x] = out[y * W + x];
    }
};


class RobertsEdgeEvent {
public:
    RobertsEdgeEvent(const std::string& in,
                     const std::string& out)
        : m_InputPath(in),
          m_OutputPath(out) {}

    void execute() {
        ImageObject img(m_InputPath);
        apply(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;

    void apply(ImageObject& img) {
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const uint32_t L = img.levels();

        std::vector<uint16_t> out(W * H, 0);

        for (uint32_t y = 0; y < H - 1; y++) {
            for (uint32_t x = 0; x < W - 1; x++) {

                int gx =
                    img[y][x] - img[y + 1][x + 1];

                int gy =
                    img[y][x + 1] - img[y + 1][x];

                int mag = std::abs(gx) + std::abs(gy);

                out[y * W + x] =
                    static_cast<uint16_t>(
                        std::clamp(mag, 0, (int)L - 1)
                    );
            }
        }

        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++)
                img[y][x] = out[y * W + x];
    }
};

class PrewittEdgeEvent {
public:
    PrewittEdgeEvent(const std::string& in,
                     const std::string& out)
        : m_InputPath(in),
          m_OutputPath(out) {}

    void execute() {
        ImageObject img(m_InputPath);
        apply(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;

    void apply(ImageObject& img) {
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const uint32_t L = img.levels();

        const int Gx[3][3] = {
            {-1, 0, 1},
            {-1, 0, 1},
            {-1, 0, 1}
        };

        const int Gy[3][3] = {
            {-1, -1, -1},
            { 0,  0,  0},
            { 1,  1,  1}
        };

        std::vector<uint16_t> out(W * H, 0);

        for (int y = 1; y < (int)H - 1; y++) {
            for (int x = 1; x < (int)W - 1; x++) {

                int gx = 0, gy = 0;

                for (int ky = -1; ky <= 1; ky++)
                    for (int kx = -1; kx <= 1; kx++) {
                        uint16_t p = img[y + ky][x + kx];
                        gx += Gx[ky + 1][kx + 1] * p;
                        gy += Gy[ky + 1][kx + 1] * p;
                    }

                int mag = std::abs(gx) + std::abs(gy);

                out[y * W + x] =
                    static_cast<uint16_t>(
                        std::clamp(mag, 0, (int)L - 1)
                    );
            }
        }

        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++)
                img[y][x] = out[y * W + x];
    }
};

class SobelEdgeEvent {
public:
    SobelEdgeEvent(const std::string& in,
                   const std::string& out,
                   uint16_t threshold = 0)
        : m_InputPath(in),
          m_OutputPath(out),
          m_Threshold(threshold) {}

    void execute() {
        ImageObject img(m_InputPath);
        apply(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    uint16_t m_Threshold; // 0 = no threshold

    void apply(ImageObject& img) {
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const uint32_t L = img.levels();

        const int Gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };

        const int Gy[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        std::vector<uint16_t> mag(W * H, 0);
        uint16_t maxMag = 0;

        // ---- compute gradient magnitude ----
        for (int y = 1; y < (int)H - 1; y++) {
            for (int x = 1; x < (int)W - 1; x++) {

                int gx = 0, gy = 0;

                for (int ky = -1; ky <= 1; ky++)
                    for (int kx = -1; kx <= 1; kx++) {
                        uint16_t p = img[y + ky][x + kx];
                        gx += Gx[ky + 1][kx + 1] * p;
                        gy += Gy[ky + 1][kx + 1] * p;
                    }

                int g = std::abs(gx) + std::abs(gy);
                mag[y * W + x] = g;
                maxMag = std::max(maxMag, (uint16_t)g);
            }
        }

        // ---- normalize + threshold ----
        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {

                uint16_t v = mag[y * W + x];

                uint16_t norm =
                    (maxMag > 0)
                        ? static_cast<uint16_t>((v * (L - 1)) / maxMag)
                        : 0;

                if (m_Threshold > 0)
                    img[y][x] = (norm >= m_Threshold) ? (L - 1) : 0;
                else
                    img[y][x] = norm;
            }
        }
    }
};


class LaplacianSharpenEvent {
public:
    enum class Mode { FOUR, EIGHT };

    LaplacianSharpenEvent(const std::string& in,
                          const std::string& lapOut,
                          const std::string& sharpOut,
                          Mode mode)
        : m_Input(in),
          m_LapOut(lapOut),
          m_SharpOut(sharpOut),
          m_Mode(mode) {}

    void execute() {
        ImageObject img(m_Input);
        ImageObject lap = img;
        ImageObject sharp = img;

        applyLaplacian(img, lap);
        applySharpen(img, lap, sharp);

        lap.saveTIFF8bit(m_LapOut);
        sharp.saveTIFF8bit(m_SharpOut);
    }

private:
    std::string m_Input, m_LapOut, m_SharpOut;
    Mode m_Mode;

    void applyLaplacian(const ImageObject& in, ImageObject& out) {
        const int K4[3][3] = {
            { 0, -1,  0},
            {-1,  4, -1},
            { 0, -1,  0}
        };

        const int K8[3][3] = {
            {-1,-1,-1},
            {-1, 8,-1},
            {-1,-1,-1}
        };

        const int (*K)[3] = (m_Mode == Mode::FOUR) ? K4 : K8;

        const int H = in.height();
        const int W = in.width();
        const int L = in.levels();

        for (int y = 1; y < H - 1; y++) {
            for (int x = 1; x < W - 1; x++) {

                int sum = 0;
                for (int j = -1; j <= 1; j++)
                    for (int i = -1; i <= 1; i++)
                        sum += K[j+1][i+1] * in[y+j][x+i];

                out[y][x] = std::clamp(sum, 0, L - 1);
            }
        }
    }

    void applySharpen(const ImageObject& orig,
                      const ImageObject& lap,
                      ImageObject& out) {

        const int L = orig.levels();

        for (uint32_t y = 0; y < orig.height(); y++) {
            for (uint32_t x = 0; x < orig.width(); x++) {

                int v = orig[y][x] + lap[y][x];
                out[y][x] = std::clamp(v, 0, L - 1);
            }
        }
    }
};


class BandFilterEvent {
public:
    enum class Mode {
        BANDPASS,
        BANDREJECT
    };

    BandFilterEvent(const std::string& in,
                    const std::string& out,
                    uint32_t k1, double s1,
                    uint32_t k2, double s2,
                    Mode mode)
        : m_Input(in), m_Output(out),
          m_K1(k1), m_S1(s1),
          m_K2(k2), m_S2(s2),
          m_Mode(mode) {}

    void execute() {
        ImageObject img(m_Input);
        ImageObject lp1 = img;
        ImageObject lp2 = img;

        gaussian(lp1, m_K1, m_S1);
        gaussian(lp2, m_K2, m_S2);

        apply(img, lp1, lp2);
        img.saveTIFF8bit(m_Output);
    }

private:
    std::string m_Input, m_Output;
    uint32_t m_K1, m_K2;
    double m_S1, m_S2;
    Mode m_Mode;

    void gaussian(ImageObject& img, uint32_t k, double sigma) {
        GaussianLowPassEvent("", "", k, sigma).execute(); // reuse logic
    }

    void apply(ImageObject& img,
               const ImageObject& lp1,
               const ImageObject& lp2) {

        const uint32_t H = img.height();
        const uint32_t W = img.width();
        const uint32_t L = img.levels();

        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {

                double val;

                if (m_Mode == Mode::BANDPASS) {
                    val = lp2[y][x] - lp1[y][x];
                } else {
                    val = lp1[y][x] + (img[y][x] - lp2[y][x]);
                }

                img[y][x] = static_cast<uint16_t>(
                    std::clamp(val, 0.0, (double)(L - 1))
                );
            }
        }
    }
};

class WeightedAveragingEvent {
public:
    WeightedAveragingEvent(const std::string& in,
                           const std::string& out)
        : m_InputPath(in),
          m_OutputPath(out) {}

    void execute() {
        ImageObject img(m_InputPath);
        apply(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;

    void apply(ImageObject& img) {
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const uint32_t L = img.levels();

        // Weighted averaging kernel (DIP standard)
        const int K[3][3] = {
            {1, 2, 1},
            {2, 4, 2},
            {1, 2, 1}
        };
        const int Ksum = 16;

        std::vector<uint16_t> out(W * H, 0);

        for (int y = 1; y < (int)H - 1; y++) {
            for (int x = 1; x < (int)W - 1; x++) {

                int acc = 0;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        acc += K[ky + 1][kx + 1] *
                               img[y + ky][x + kx];
                    }
                }

                out[y * W + x] =
                    static_cast<uint16_t>(
                        std::clamp(acc / Ksum, 0, (int)L - 1)
                    );
            }
        }

        // write back
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++)
                img[y][x] = out[y * W + x];
    }
};

class GradientSharpenEvent {
public:
    GradientSharpenEvent(const std::string& in,
                         const std::string& out,
                         double k)
        : m_InputPath(in),
          m_OutputPath(out),
          m_K(k)
    {
        if (k <= 0.0)
            throw std::runtime_error("Sharpen factor k must be > 0");
    }

    void execute() {
        ImageObject img(m_InputPath);
        sharpen(img);
        img.saveTIFF8bit(m_OutputPath);
    }

private:
    std::string m_InputPath, m_OutputPath;
    double m_K;

    void sharpen(ImageObject& img) {
        const uint32_t W = img.width();
        const uint32_t H = img.height();
        const uint32_t L = img.levels();

        // Sobel operators
        const int Gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };

        const int Gy[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        std::vector<uint16_t> out(W * H, 0);

        for (int y = 1; y < (int)H - 1; y++) {
            for (int x = 1; x < (int)W - 1; x++) {

                int sx = 0, sy = 0;

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        uint16_t p = img[y + ky][x + kx];
                        sx += Gx[ky + 1][kx + 1] * p;
                        sy += Gy[ky + 1][kx + 1] * p;
                    }
                }

                double gradMag = std::abs(sx) + std::abs(sy);

                double sharpened =
                    img[y][x] + m_K * gradMag;

                out[y * W + x] =
                    static_cast<uint16_t>(
                        std::clamp(sharpened, 0.0, (double)(L - 1))
                    );
            }
        }

        // write back
        for (uint32_t y = 0; y < H; y++)
            for (uint32_t x = 0; x < W; x++)
                img[y][x] = out[y * W + x];
    }
};

class InputHandler {
public:
    static void run() {
        std::string cmd;
        while (true) {

            std::cout << "\nCommands:\n";
            std::cout << "invert <input> <output>\n";
            std::cout<<"log <input> <output>\n";
            std::cout<<"gamma <input> <output> <gamma_value>\n";
            std::cout<<"contrast <input> <output> <r1> <s1> <r2> <s2>\n";
            std::cout<<"ramp <input> <output> <start> <end> \n";
            std::cout<<"slice <input> <output> <start> <end> <mode>(bg,nobg)\n";
            std::cout<<"bit_slice <input> <output> <bitindex> <mode>(bg,nobg)\n";
            std::cout << "hist_eq <input> <output>\n";
            std::cout << "hist_stats <input>\n";
            std::cout << "hist_match <src> <ref> <output>\n";
            std::cout<<"local_hist <input> <output> <windowsize>\n";
            std::cout << "smooth_box <input> <output> <kernel_size>\n";
            std::cout << "gaussian <input> <output> <kernelSize> <sigma>\n";
            std::cout << "sharpen <input> <output> <strength>\n";
            std::cout << "unsharp <input> <output> <A>\n";
            std::cout << "grad_edge <input> <output> <k>\n";
             std::cout<<"lap_sobel <input> <lap_out> <sharp_out> <sobel_out>\n";
             std::cout << "median <input> <output> <windowSize>\n";
             std::cout << "roberts <input> <output>\n";
             std::cout << "prewitt <input> <output>\n";
             std::cout << "sobel <input> <output> [threshold]\n";
             std::cout << "laplacian <input> <lap_out> <sharp_out> <4|8>\n";
             std::cout << "bandpass <in> <out> <k1> <s1> <k2> <s2>\n";
             std::cout << "bandreject <in> <out> <k1> <s1> <k2> <s2>\n";
             std::cout << "weighted_avg <input> <output>\n";
             std::cout << "grad_sharpen <input> <output> <k>\n";

            std::cout << "quit\n";

            std::cin >> cmd;

            if (cmd == "invert") {
                std::string in, out;
                std::cin >> in >> out;
                InvertImageEvent(in, out).execute();
                std::cout << "Inversion done.\n";
            }
            else if (cmd == "log") {
                std::string in, out;
                std::cin >> in >> out;
                LogTransformEvent(in, out).execute();
                std::cout << "Log transformation done.\n";
            }else if (cmd == "grad_edge") {
                std::string in, out;
                double k;
                std::cin >> in >> out >> k;
                GradientEdgeEnhancementEvent(in, out, k).execute();
                std::cout << "Gradient edge enhancement done.\n";
            }
else if(cmd=="gamma"){
                std::string in,out;
                double gamma;
                std::cin>>in>>out>>gamma;
                GammaTransformEvent(in,out,gamma).execute();
                std::cout<<"Gamma transformation done.\n";
            }else if(cmd=="Contrast"){
                std::string in,out;
                uint16_t r1,s1,r2,s2;
                std::cin>>in>>out>>r1>>s1>>r2>>s2;
                PieceWiseContrastEvent(in,out,r1,s1,r2,s2).execute();
                std::cout<<"Contrast transformation done.\n";
            }
            else if (cmd == "smooth_box") {
                std::string in, out;
                uint32_t k;
                std::cin >> in >> out >> k;
                BoxSmoothingEvent(in, out, k).execute();
                std::cout << "Box smoothing done.\n";
            }
               else if(cmd=="ramp"){
                std::string in,out;
                uint16_t start,end;
                std::cin>>in>>out>>start>>end;
                IntensityRampEvent(in,out,start,end).execute();
                std::cout<<"Intensity_Ramp transformation done.\n";
            }else if(cmd=="slice"){
                std::string in,out;
                std::string mode;
                uint16_t start,end,slice;
                std::cin>>in>>out>>start>>end>>slice>>mode;
                IntensityLevelSlicingEvent(in,out,start,end,slice,mode).execute();
            } else if (cmd == "bit_slice") {
                std::string in, out;
                uint16_t index;
                std::string mode;
                std::cin >> in >> out >> index >> mode;
                BitPlaneSliceEvent(in, out, index, mode).execute();
            }else if (cmd == "hist_stats") {
                std::string in;
                std::cin >> in;
                HistogramStatsEvent(in).execute();
            }
             else if (cmd == "hist_eq") {
                std::string in, out;
                std::cin >> in >> out;
                HistogramEqualizationEvent(in, out).execute();
                std::cout << "Histogram equalization done.\n";
            }else if (cmd == "hist_match") {
                std::string src, ref, out;
                std::cin >> src >> ref >> out;
                HistogramMatchingEvent(src, ref, out).execute();
                std::cout << "Histogram matching done.\n";
            }else if(cmd=="local_hist"){
                std::string in, out;
                    uint32_t win;
                    std::cin >> in >> out >> win;
                    LocalHistogramEnhancementEvent(in, out, win).execute();

            }else if (cmd == "gaussian") {
                std::string in, out;
                uint32_t k;
                double sigma;
                std::cin >> in >> out >> k >> sigma;
                GaussianLowPassEvent(in, out, k, sigma).execute();
                std::cout << "Gaussian smoothing done.\n";
            }
            else if (cmd == "sharpen") {
                std::string in, out;
                double k;
                std::cin >> in >> out >> k;
                HighPassSharpenEvent(in, out, k).execute();
                std::cout << "Sharpening done.\n";
            }
            else if (cmd == "unsharp") {
                std::string in, out;
                double A;
                std::cin >> in >> out >> A;
                UnsharpHighboostEvent(in, out, A).execute();
                std::cout << "Unsharp / Highboost done.\n";
            }
            else if (cmd == "lap_sobel") {
                std::string in, lap, sharp, sobel;
                std::cin >> in >> lap >> sharp >> sobel;
                LaplacianSobelSharpenEvent(in, lap, sharp, sobel).execute();
                std::cout << "Laplacian + Sobel sharpening done.\n";
            }
            else if (cmd == "median") {
                std::string in, out;
                uint32_t w;
                std::cin >> in >> out >> w;
                MedianFilterEvent(in, out, w).execute();
                std::cout << "Median filtering done.\n";
            }else if (cmd == "roberts") {
                std::string in, out;
                std::cin >> in >> out;
                RobertsEdgeEvent(in, out).execute();
                std::cout << "Roberts edge detection done.\n";
            }
            else if (cmd == "prewitt") {
                std::string in, out;
                std::cin >> in >> out;
                PrewittEdgeEvent(in, out).execute();
                std::cout << "Prewitt edge detection done.\n";
            }

            else if (cmd == "sobel") {
                std::string in, out;
                uint16_t t = 0;
                std::cin >> in >> out;
                if (std::cin.peek() != '\n')
                    std::cin >> t;

                SobelEdgeEvent(in, out, t).execute();
                std::cout << "Sobel edge detection done.\n";
            }
            else if (cmd == "laplacian") {
                std::string in, lap, sharp;
                int mode;
                std::cin >> in >> lap >> sharp >> mode;

                LaplacianSharpenEvent::Mode m =
                    (mode == 8)
                        ? LaplacianSharpenEvent::Mode::EIGHT
                        : LaplacianSharpenEvent::Mode::FOUR;

                LaplacianSharpenEvent(in, lap, sharp, m).execute();
                std::cout << "Laplacian sharpening done.\n";
            }
            else if (cmd == "bandpass") {
                std::string in, out;
                uint32_t k1, k2;
                double s1, s2;
                std::cin >> in >> out >> k1 >> s1 >> k2 >> s2;
                BandFilterEvent(in, out, k1, s1, k2, s2,
                    BandFilterEvent::Mode::BANDPASS).execute();
            }

            else if (cmd == "bandreject") {
                std::string in, out;
                uint32_t k1, k2;
                double s1, s2;
                std::cin >> in >> out >> k1 >> s1 >> k2 >> s2;
                BandFilterEvent(in, out, k1, s1, k2, s2,
                    BandFilterEvent::Mode::BANDREJECT).execute();
            }
else if (cmd == "weighted_avg") {
    std::string in, out;
    std::cin >> in >> out;
    WeightedAveragingEvent(in, out).execute();
    std::cout << "Weighted averaging done.\n";
}
else if (cmd == "grad_sharpen") {
    std::string in, out;
    double k;
    std::cin >> in >> out >> k;
    GradientSharpenEvent(in, out, k).execute();
    std::cout << "Gradient sharpening done.\n";
}


            else if (cmd == "quit") {
                break;
            }
        }
    }
};



int main() {
    InputHandler handler;
    handler.run();
return 0;
}
