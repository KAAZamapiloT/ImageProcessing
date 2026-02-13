#include<iostream>
#include<string>
#include "fstream"
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
        double noiseVariance = 50.0;
        double gaussianSigma = 1.0;

        // periodic (real FFT version would use these)
        std::vector<std::pair<int,int>> notchCenters;
        int notchRadius = 5;
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

        else {
            std::cout << "Unknown strategy\n";
            return;
        }

        std::cout << "PSNR: "
                  << ComputePSNR(original, img)
                  << " dB\n";

        img.saveTIFF(outputFile);
    }

private:
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

    // log transform
    for(uint32_t y=0;y<h;y++)
        for(uint32_t x=0;x<w;x++)
            img[y][x] = std::log(1 + img[y][x]);

    MeanFilter(img,p.kernelSize);

    // exp back
    for(uint32_t y=0;y<h;y++)
        for(uint32_t x=0;x<w;x++)
            img[y][x] =
                std::clamp((int)std::exp(img[y][x]),
                           0,(int)(img.levels()-1));
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

            std::cout << "\n==== IMAGE RESTORATION ENGINE ====\n";
            std::cout << "Enter strategy "
                      << "(gaussian | saltpepper | uniform | "
                      << "rayleigh | erlang | exponential | periodic)\n";
            std::cout << "Type 'exit' to quit\n> ";

            std::cin >> strategy;

            if (strategy == "exit" || strategy == "quit")
                return;

            std::cout << "Input file: ";
            std::cin >> inputFile;

            std::cout << "Output file: ";
            std::cin >> outputFile;

            IMAGE_RESTORATION::Params params;

            // ---- Strategy-specific parameters ----

            if (strategy == "gaussian")
            {
                std::cout << "Kernel size (odd number, e.g. 3 or 5): ";
                std::cin >> params.kernelSize;

                std::cout << "Noise variance (e.g. 50): ";
                std::cin >> params.noiseVariance;
            }

            else if (strategy == "saltpepper")
            {
                std::cout << "Kernel size (odd number, e.g. 3 or 5): ";
                std::cin >> params.kernelSize;
            }

            else if (strategy == "uniform")
            {
                std::cout << "Kernel size (odd number, e.g. 3 or 5): ";
                std::cin >> params.kernelSize;
            }

            else if (strategy == "rayleigh" ||
                     strategy == "erlang")
            {
                std::cout << "Kernel size (odd number, e.g. 3 or 5): ";
                std::cin >> params.kernelSize;
            }

            else if (strategy == "exponential")
            {
                std::cout << "Kernel size (odd number, e.g. 3 or 5): ";
                std::cin >> params.kernelSize;
            }

            else if (strategy == "periodic")
            {
                std::cout << "WARNING: Real periodic removal requires FFT.\n";
                std::cout << "Kernel size for smoothing approximation: ";
                std::cin >> params.kernelSize;
            }

            else
            {
                std::cout << "Unknown strategy.\n";
                continue;
            }

            // Basic validation
            if (params.kernelSize % 2 == 0)
            {
                std::cout << "Kernel size must be odd. Auto-fixing.\n";
                params.kernelSize += 1;
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
