#include <tiffio.h>
#include <iostream>
#include<vector>
#include<string>
#include<cstdint>
#include <stdexcept>


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

           uint16 samplesPerPixel;

           TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &m_Width);
           TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &m_Height);
           TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE, &m_BitsPerSample);
           TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel);

           if (samplesPerPixel != 1)
               throw std::runtime_error("Only grayscale TIFF supported");

           m_Levels = 1u << m_BitsPerSample;
           m_Data.resize(m_Width * m_Height);

           for (uint32_t row = 0; row < m_Height; row++) {
               TIFFReadScanline(tif, &m_Data[row * m_Width], row);
           }

           TIFFClose(tif);
   }


};

class InvertImage{
public:
static void InvertImagefunction(std::string inputfile , std::string Outputfile){


}
};



int main() {
    std::cout << TIFFGetVersion() << std::endl;
}
