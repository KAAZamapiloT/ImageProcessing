#include <algorithm>
#include <cctype>
#include <cstdint>
#include <complex>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <tiffio.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")
#endif


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

class ConstImageRow {
public:
    ConstImageRow(const uint16_t* rowData) : m_Row(rowData) {}

    const uint16_t& operator[](uint32_t x) const {
        return m_Row[x];
    }

private:
    const uint16_t* m_Row;
};


class ImageObject{
  public:
  ImageObject() = default;

  ImageObject(uint32_t width, uint32_t height, uint16_t bits = 8)
      : m_Width(width),
        m_Height(height),
        m_Levels(0),
        m_BitsPerSample(bits),
        m_Data(static_cast<size_t>(width) * static_cast<size_t>(height), 0)
  {
      if (bits == 0 || bits > 16) {
          throw std::runtime_error("bits must be between 1 and 16");
      }
      m_Levels = 1u << bits;
  }

  ImageObject(const std::string& filepath)
      : m_Path(filepath)
  {
      loadByExtension();
  }
    inline  uint32_t width()  const { return m_Width; }
    inline  uint32_t height() const { return m_Height; }
    inline  uint16_t bits()   const { return m_BitsPerSample; }
    inline  uint32_t levels() const { return m_Levels; }

    void setBitDepth(uint16_t bits) {
        if (bits == 0 || bits > 16) {
            throw std::runtime_error("bits must be between 1 and 16");
        }
        m_BitsPerSample = bits;
        m_Levels = 1u << bits;
        clampPixels(0, maxPixelValue());
    }

    void load(const std::string& filepath) {
        m_Path = filepath;
        loadByExtension();
    }

    void save(const std::string& outputPath) {
        switch (detectFileType(outputPath)) {
            case FileType::TIFF:
                saveTIFF(outputPath);
                break;
            case FileType::JPG:
                saveJPG(outputPath);
                break;
            case FileType::PNG:
                savePNG(outputPath);
                break;
            case FileType::BMP:
                saveBMP(outputPath);
                break;
            default:
                throw std::runtime_error(
                    "Unsupported output extension. Use .tif/.tiff/.jpg/.jpeg/.png/.bmp");
        }
    }

    ImageRow operator[](uint32_t y) {
        return ImageRow(&m_Data[y * m_Width]);
    }

    ConstImageRow operator[](uint32_t y) const {
        return ConstImageRow(&m_Data[y * m_Width]);
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

    void saveJPG(const std::string& OutputPath){
        saveWithWIC(OutputPath, FileType::JPG);
    }

    void savePNG(const std::string& OutputPath){
        saveWithWIC(OutputPath, FileType::PNG);
    }

    void saveBMP(const std::string& OutputPath){
        saveWithWIC(OutputPath, FileType::BMP);
    }

    std::vector<uint16_t>& data() { return m_Data; }
    const std::vector<uint16_t>& data() const { return m_Data; }

    void fill(uint16_t value) {
        std::fill(m_Data.begin(), m_Data.end(), value);
    }

    void clampPixels(uint16_t low, uint16_t high) {
        if (low > high) std::swap(low, high);
        for (auto& px : m_Data) {
            px = std::clamp(px, low, high);
        }
    }

    void addOffset(int delta) {
        const int maxValue = static_cast<int>(maxPixelValue());
        for (auto& px : m_Data) {
            const int v = std::clamp(static_cast<int>(px) + delta, 0, maxValue);
            px = static_cast<uint16_t>(v);
        }
    }

    void multiplyScale(double factor) {
        const double maxValue = static_cast<double>(maxPixelValue());
        for (auto& px : m_Data) {
            const double v = std::clamp(static_cast<double>(px) * factor, 0.0, maxValue);
            px = static_cast<uint16_t>(v);
        }
    }

    template <typename PixelFn>
    void transformInPlace(PixelFn fn) {
        const int maxValue = static_cast<int>(maxPixelValue());
        for (uint32_t y = 0; y < m_Height; ++y) {
            for (uint32_t x = 0; x < m_Width; ++x) {
                const size_t idx = static_cast<size_t>(y) * m_Width + x;
                const int value = static_cast<int>(fn(m_Data[idx], x, y));
                m_Data[idx] = static_cast<uint16_t>(std::clamp(value, 0, maxValue));
            }
        }
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
  enum class FileType { TIFF, JPG, PNG, BMP, UNKNOWN };

  uint32_t m_Width = 0, m_Height = 0, m_Levels = 0;
  uint16_t m_BitsPerSample = 0;
  std::string m_Path;
  std::vector<uint16_t> m_Data;

  uint16_t maxPixelValue() const {
      if (m_BitsPerSample == 0) return 0;
      if (m_BitsPerSample >= 16) return 65535;
      return static_cast<uint16_t>((1u << m_BitsPerSample) - 1u);
  }

  static FileType detectFileType(const std::string& path) {
      std::string ext = std::filesystem::path(path).extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
          return static_cast<char>(std::tolower(c));
      });

      if (ext == ".tif" || ext == ".tiff") return FileType::TIFF;
      if (ext == ".jpg" || ext == ".jpeg") return FileType::JPG;
      if (ext == ".png") return FileType::PNG;
      if (ext == ".bmp") return FileType::BMP;
      return FileType::UNKNOWN;
  }

  void loadByExtension() {
      switch (detectFileType(m_Path)) {
          case FileType::TIFF: loadTIFF(); break;
          case FileType::JPG:  loadJPG(); break;
          case FileType::PNG:  loadPNG(); break;
          case FileType::BMP:  loadBMP(); break;
          default:
              throw std::runtime_error(
                  "Unsupported input extension. Use .tif/.tiff/.jpg/.jpeg/.png/.bmp");
      }
  }

  std::vector<uint8_t> to8BitNormalized() const {
      std::vector<uint8_t> data8bit(m_Data.size());
      if (m_Data.empty()) return data8bit;

      uint16_t dataMin = 65535, dataMax = 0;
      for (const auto& val : m_Data) {
          dataMin = std::min(dataMin, val);
          dataMax = std::max(dataMax, val);
      }

      if (dataMax == dataMin) {
          std::fill(data8bit.begin(), data8bit.end(), 128);
      } else {
          const double scale = 255.0 / (dataMax - dataMin);
          for (size_t i = 0; i < m_Data.size(); i++) {
              const double normalized = (m_Data[i] - dataMin) * scale;
              data8bit[i] = static_cast<uint8_t>(std::clamp(normalized, 0.0, 255.0));
          }
      }
      return data8bit;
  }

#ifdef _WIN32
  template <typename T>
  static void releaseCOM(T*& p) {
      if (p) {
          p->Release();
          p = nullptr;
      }
  }
#endif

  void loadJPG() { loadWithWIC(); }
  void loadPNG() { loadWithWIC(); }
  void loadBMP() { loadWithWIC(); }

  void loadWithWIC() {
#ifndef _WIN32
      throw std::runtime_error("JPG/PNG/BMP loading requires Windows WIC support");
#else
      HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
      const bool shouldUninit = SUCCEEDED(hr);
      if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
          throw std::runtime_error("Failed to initialize COM for image loading");
      }

      IWICImagingFactory* factory = nullptr;
      IWICBitmapDecoder* decoder = nullptr;
      IWICBitmapFrameDecode* frame = nullptr;
      IWICFormatConverter* converter = nullptr;

      auto cleanup = [&]() {
          releaseCOM(converter);
          releaseCOM(frame);
          releaseCOM(decoder);
          releaseCOM(factory);
          if (shouldUninit) CoUninitialize();
      };

      try {
          hr = CoCreateInstance(
              CLSID_WICImagingFactory,
              nullptr,
              CLSCTX_INPROC_SERVER,
              IID_PPV_ARGS(&factory)
          );
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to create WIC imaging factory");
          }

          const std::wstring widePath = std::filesystem::path(m_Path).wstring();
          hr = factory->CreateDecoderFromFilename(
              widePath.c_str(),
              nullptr,
              GENERIC_READ,
              WICDecodeMetadataCacheOnLoad,
              &decoder
          );
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to open image file with WIC");
          }

          hr = decoder->GetFrame(0, &frame);
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to read first image frame");
          }

          hr = factory->CreateFormatConverter(&converter);
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to create WIC format converter");
          }

          hr = converter->Initialize(
              frame,
              GUID_WICPixelFormat32bppBGRA,
              WICBitmapDitherTypeNone,
              nullptr,
              0.0,
              WICBitmapPaletteTypeCustom
          );
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to convert image format to BGRA");
          }

          UINT w = 0;
          UINT h = 0;
          hr = converter->GetSize(&w, &h);
          if (FAILED(hr) || w == 0 || h == 0) {
              throw std::runtime_error("Invalid image size from decoder");
          }

          const size_t pixelCount = static_cast<size_t>(w) * static_cast<size_t>(h);
          std::vector<uint8_t> bgra(pixelCount * 4u, 0);
          hr = converter->CopyPixels(
              nullptr,
              w * 4u,
              static_cast<UINT>(bgra.size()),
              bgra.data()
          );
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to copy decoded pixels");
          }

          m_Width = w;
          m_Height = h;
          m_BitsPerSample = 8;
          m_Levels = 1u << m_BitsPerSample;
          m_Data.assign(pixelCount, 0);

          for (size_t i = 0; i < pixelCount; ++i) {
              const uint8_t b = bgra[i * 4u + 0u];
              const uint8_t g = bgra[i * 4u + 1u];
              const uint8_t r = bgra[i * 4u + 2u];
              m_Data[i] = static_cast<uint16_t>((299u * r + 587u * g + 114u * b + 500u) / 1000u);
          }
      } catch (...) {
          cleanup();
          throw;
      }

      cleanup();
#endif
  }

  void saveWithWIC(const std::string& outputPath, FileType format) {
#ifndef _WIN32
      (void)outputPath;
      (void)format;
      throw std::runtime_error("JPG/PNG/BMP saving requires Windows WIC support");
#else
      if (m_Data.empty() || m_Width == 0 || m_Height == 0) {
          throw std::runtime_error("No image data to save");
      }

      GUID container = GUID_NULL;
      switch (format) {
          case FileType::JPG: container = GUID_ContainerFormatJpeg; break;
          case FileType::PNG: container = GUID_ContainerFormatPng; break;
          case FileType::BMP: container = GUID_ContainerFormatBmp; break;
          default:
              throw std::runtime_error("Invalid format for WIC encoder");
      }

      HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
      const bool shouldUninit = SUCCEEDED(hr);
      if (FAILED(hr) && hr != RPC_E_CHANGED_MODE) {
          throw std::runtime_error("Failed to initialize COM for image saving");
      }

      IWICImagingFactory* factory = nullptr;
      IWICStream* stream = nullptr;
      IWICBitmapEncoder* encoder = nullptr;
      IWICBitmapFrameEncode* frame = nullptr;
      IPropertyBag2* options = nullptr;

      auto cleanup = [&]() {
          releaseCOM(options);
          releaseCOM(frame);
          releaseCOM(encoder);
          releaseCOM(stream);
          releaseCOM(factory);
          if (shouldUninit) CoUninitialize();
      };

      try {
          hr = CoCreateInstance(
              CLSID_WICImagingFactory,
              nullptr,
              CLSCTX_INPROC_SERVER,
              IID_PPV_ARGS(&factory)
          );
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to create WIC imaging factory");
          }

          hr = factory->CreateStream(&stream);
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to create WIC stream");
          }

          const std::wstring widePath = std::filesystem::path(outputPath).wstring();
          hr = stream->InitializeFromFilename(widePath.c_str(), GENERIC_WRITE);
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to open output image file");
          }

          hr = factory->CreateEncoder(container, nullptr, &encoder);
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to create WIC encoder");
          }

          hr = encoder->Initialize(stream, WICBitmapEncoderNoCache);
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to initialize WIC encoder");
          }

          hr = encoder->CreateNewFrame(&frame, &options);
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to create WIC frame");
          }

          hr = frame->Initialize(options);
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to initialize WIC frame");
          }

          hr = frame->SetSize(m_Width, m_Height);
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to set output image size");
          }

          WICPixelFormatGUID pixelFormat = GUID_WICPixelFormat24bppBGR;
          hr = frame->SetPixelFormat(&pixelFormat);
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to set output pixel format");
          }
          if (!IsEqualGUID(pixelFormat, GUID_WICPixelFormat24bppBGR)) {
              throw std::runtime_error("Encoder rejected 24bpp BGR format");
          }

          std::vector<uint8_t> gray = to8BitNormalized();
          std::vector<uint8_t> bgr(gray.size() * 3u, 0);
          for (size_t i = 0; i < gray.size(); ++i) {
              const uint8_t v = gray[i];
              bgr[i * 3u + 0u] = v;
              bgr[i * 3u + 1u] = v;
              bgr[i * 3u + 2u] = v;
          }

          if (m_Width > (std::numeric_limits<UINT>::max() / 3u)) {
              throw std::runtime_error("Image width too large for WIC writer");
          }
          if (bgr.size() > std::numeric_limits<UINT>::max()) {
              throw std::runtime_error("Image buffer too large for WIC writer");
          }

          const UINT stride = m_Width * 3u;
          hr = frame->WritePixels(
              m_Height,
              stride,
              static_cast<UINT>(bgr.size()),
              bgr.data()
          );
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to write encoded image pixels");
          }

          hr = frame->Commit();
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to finalize output image frame");
          }

          hr = encoder->Commit();
          if (FAILED(hr)) {
              throw std::runtime_error("Failed to finalize output image");
          }
      } catch (...) {
          cleanup();
          throw;
      }

      cleanup();
#endif
  }

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

       if (samplesPerPixel != 1) {
           TIFFClose(tif);
           throw std::runtime_error("Only grayscale TIFF supported");
       }

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

class ComplexImageRow {
public:
    ComplexImageRow(std::complex<double>* rowData) : m_Row(rowData) {}

    std::complex<double>& operator[](uint32_t x) {
        return m_Row[x];
    }

    const std::complex<double>& operator[](uint32_t x) const {
        return m_Row[x];
    }

private:
    std::complex<double>* m_Row;
};

class ConstComplexImageRow {
public:
    ConstComplexImageRow(const std::complex<double>* rowData) : m_Row(rowData) {}

    const std::complex<double>& operator[](uint32_t x) const {
        return m_Row[x];
    }

private:
    const std::complex<double>* m_Row;
};

class ComplexImage {
public:
    ComplexImage() = default;

    ComplexImage(uint32_t width, uint32_t height)
        : m_Width(width),
          m_Height(height),
          m_Data(static_cast<size_t>(width) * static_cast<size_t>(height), std::complex<double>(0.0, 0.0)) {}

    explicit ComplexImage(const ImageObject& src)
        : m_Width(src.width()),
          m_Height(src.height()),
          m_Data(static_cast<size_t>(m_Width) * static_cast<size_t>(m_Height), std::complex<double>(0.0, 0.0)) {
        for (uint32_t y = 0; y < m_Height; ++y) {
            for (uint32_t x = 0; x < m_Width; ++x) {
                (*this)[y][x] = std::complex<double>(src[y][x], 0.0);
            }
        }
    }

    uint32_t width() const { return m_Width; }
    uint32_t height() const { return m_Height; }

    ComplexImageRow operator[](uint32_t y) {
        return ComplexImageRow(&m_Data[static_cast<size_t>(y) * m_Width]);
    }

    ConstComplexImageRow operator[](uint32_t y) const {
        return ConstComplexImageRow(&m_Data[static_cast<size_t>(y) * m_Width]);
    }

    std::vector<std::complex<double>>& data() { return m_Data; }
    const std::vector<std::complex<double>>& data() const { return m_Data; }

    void fill(const std::complex<double>& value) {
        std::fill(m_Data.begin(), m_Data.end(), value);
    }

    template <typename ComplexFn>
    void transformInPlace(ComplexFn fn) {
        for (uint32_t y = 0; y < m_Height; ++y) {
            for (uint32_t x = 0; x < m_Width; ++x) {
                const size_t idx = static_cast<size_t>(y) * m_Width + x;
                m_Data[idx] = fn(m_Data[idx], x, y);
            }
        }
    }

    ImageObject magnitudeImage(uint16_t bits = 8) const {
        ImageObject out(m_Width, m_Height, bits);
        if (m_Data.empty()) return out;

        double maxMagnitude = 0.0;
        for (const auto& v : m_Data) {
            maxMagnitude = std::max(maxMagnitude, std::abs(v));
        }

        if (maxMagnitude <= 0.0) return out;

        const double maxValue = (bits >= 16) ? 65535.0 : static_cast<double>((1u << bits) - 1u);
        const double scale = maxValue / maxMagnitude;

        for (uint32_t y = 0; y < m_Height; ++y) {
            for (uint32_t x = 0; x < m_Width; ++x) {
                const double mag = std::abs((*this)[y][x]);
                const double value = std::clamp(mag * scale, 0.0, maxValue);
                out[y][x] = static_cast<uint16_t>(value);
            }
        }

        return out;
    }
private:
    uint32_t m_Width = 0;
    uint32_t m_Height = 0;
    std::vector<std::complex<double>> m_Data;
};
