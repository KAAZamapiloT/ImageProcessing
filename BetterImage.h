#include<iostream>
#include<fstream>
#include<string>
#include<vector>


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

void loadJPG(){

}
void loadPNG(){}

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

