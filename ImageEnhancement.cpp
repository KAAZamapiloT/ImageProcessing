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
            std::cout << "quit\n> ";

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
            }else if(cmd=="gamma"){
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
            }else if(cmd=="ramp"){
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
