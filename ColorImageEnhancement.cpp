#include <tiffio.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <array>

/*
 * Compilation (Windows / MSVC + vcpkg):
 *
 *   cl ImageEnhancement_Color.cpp ^
 *   /std:c++20 /EHsc ^
 *   /I C:\vcpkg\installed\x64-windows\include ^
 *   /link /LIBPATH:C:\vcpkg\installed\x64-windows\lib tiff.lib
 *
 * Compilation (Linux / GCC):
 *
 *   g++ -std=c++20 -O2 ImageEnhancement_Color.cpp -ltiff -o imgproc
 */

// ============================================================
//  HSV <-> RGB helpers
// ============================================================

struct HSV { double h, s, v; };   // h in [0,360), s/v in [0,1]

inline HSV rgbToHSV(double r, double g, double b) {   // r,g,b in [0,1]
    double mx = std::max({r, g, b});
    double mn = std::min({r, g, b});
    double d  = mx - mn;
    HSV out;
    out.v = mx;
    out.s = (mx > 1e-10) ? d / mx : 0.0;
    if      (d < 1e-10) { out.h = 0.0; }
    else if (mx == r)   { out.h = 60.0 * std::fmod((g - b) / d, 6.0); }
    else if (mx == g)   { out.h = 60.0 * ((b - r) / d + 2.0); }
    else                { out.h = 60.0 * ((r - g) / d + 4.0); }
    if (out.h < 0.0) out.h += 360.0;
    return out;
}

inline std::array<double,3> hsvToRGB(double h, double s, double v) {
    if (s < 1e-10) return {v, v, v};
    double hh = h / 60.0;
    int    i  = static_cast<int>(hh) % 6;
    double f  = hh - static_cast<int>(hh);
    double p  = v * (1.0 - s);
    double q  = v * (1.0 - s * f);
    double t  = v * (1.0 - s * (1.0 - f));
    switch (i) {
        case 0: return {v, t, p};
        case 1: return {q, v, p};
        case 2: return {p, v, t};
        case 3: return {p, q, v};
        case 4: return {t, p, v};
        default: return {v, p, q};
    }
}

// ============================================================
//  Colormaps  (for pseudocolor)
//  t in [0,1] -> {r, g, b} each in [0,255]
// ============================================================

enum class Colormap { JET, HOT, COOL, BONE, RAINBOW, SPRING, SUMMER };

inline std::array<uint8_t,3> applyColormap(double t, Colormap cm) {
    t = std::clamp(t, 0.0, 1.0);
    double r = 0, g = 0, b = 0;

    switch (cm) {
    case Colormap::JET:
        if      (t < 0.125) { r = 0;             g = 0;               b = 0.5 + t * 4.0; }
        else if (t < 0.375) { r = 0;             g = (t-0.125)*4.0;   b = 1.0; }
        else if (t < 0.625) { r = (t-0.375)*4.0; g = 1.0;            b = 1.0-(t-0.375)*4.0; }
        else if (t < 0.875) { r = 1.0;           g = 1.0-(t-0.625)*4.0; b = 0; }
        else                { r = 1.0-(t-0.875)*4.0; g = 0;           b = 0; }
        break;

    case Colormap::HOT:
        r = std::clamp(t * 3.0,       0.0, 1.0);
        g = std::clamp(t * 3.0 - 1.0, 0.0, 1.0);
        b = std::clamp(t * 3.0 - 2.0, 0.0, 1.0);
        break;

    case Colormap::COOL:
        r = t;  g = 1.0 - t;  b = 1.0;
        break;

    case Colormap::BONE: {
        // Resembles grayscale but with a blue tint
        double x = t;
        r = (x < 3.0/4.0) ? (7.0*x/8.0) : (7.0*x/8.0 + (x-3.0/4.0)*8.0/8.0);
        g = (x < 3.0/8.0) ? (7.0*x/8.0) :
            (x < 6.0/8.0) ? (7.0*x/8.0 + (x-3.0/8.0)*4.0/3.0/8.0) :
                             (7.0*x/8.0 + 1.0/8.0);
        b = 7.0*x/8.0 + 1.0/8.0;
        r = std::clamp(r,0.0,1.0);
        g = std::clamp(g,0.0,1.0);
        b = std::clamp(b,0.0,1.0);
        break;
    }

    case Colormap::RAINBOW: {
        auto c = hsvToRGB(t * 300.0, 1.0, 1.0);
        r = c[0];  g = c[1];  b = c[2];
        break;
    }

    case Colormap::SPRING:
        r = 1.0;  g = t;  b = 1.0 - t;
        break;

    case Colormap::SUMMER:
        r = t;  g = 0.5 + t * 0.5;  b = 0.4;
        break;
    }

    return {
        static_cast<uint8_t>(std::round(std::clamp(r,0.0,1.0) * 255.0)),
        static_cast<uint8_t>(std::round(std::clamp(g,0.0,1.0) * 255.0)),
        static_cast<uint8_t>(std::round(std::clamp(b,0.0,1.0) * 255.0))
    };
}

inline Colormap parseColormap(const std::string& name) {
    if (name == "hot")    return Colormap::HOT;
    if (name == "cool")   return Colormap::COOL;
    if (name == "bone")   return Colormap::BONE;
    if (name == "rainbow")return Colormap::RAINBOW;
    if (name == "spring") return Colormap::SPRING;
    if (name == "summer") return Colormap::SUMMER;
    return Colormap::JET; // default
}

// ============================================================
//  ImageRow proxy (channel 0, backward-compat)
// ============================================================

class ImageRow {
public:
    ImageRow(uint16_t* rowData) : m_Row(rowData) {}
    uint16_t& operator[](uint32_t x)       { return m_Row[x]; }
    const uint16_t& operator[](uint32_t x) const { return m_Row[x]; }
private:
    uint16_t* m_Row;
};

// ============================================================
//  ImageObject  — multi-channel (1 = grayscale, 3 = RGB)
// ============================================================

class ImageObject {
public:
    ImageObject() = default;

    explicit ImageObject(const std::string& filepath) : m_Path(filepath) { loadTIFF(); }

    // Copy semantics required by event classes that clone an image
    ImageObject(const ImageObject&) = default;
    ImageObject& operator=(const ImageObject&) = default;

    // Factory: create blank image
    static ImageObject create(uint32_t w, uint32_t h, uint16_t bits, uint16_t channels = 1) {
        ImageObject img;
        img.m_Width = w;  img.m_Height = h;
        img.m_BitsPerSample = bits;
        img.m_Channels = channels;
        img.m_Levels   = 1u << bits;
        img.m_Data.assign(channels, std::vector<uint16_t>(w * h, 0));
        return img;
    }

    // ---- metadata ----
    uint32_t width()    const { return m_Width; }
    uint32_t height()   const { return m_Height; }
    uint16_t bits()     const { return m_BitsPerSample; }
    uint32_t levels()   const { return m_Levels; }
    uint16_t channels() const { return m_Channels; }
    bool     isColor()  const { return m_Channels >= 3; }

    // ---- pixel access ----
    uint16_t& pixel(uint32_t y, uint32_t x, uint32_t ch = 0) {
        return m_Data[ch][y * m_Width + x];
    }
    const uint16_t& pixel(uint32_t y, uint32_t x, uint32_t ch = 0) const {
        return m_Data[ch][y * m_Width + x];
    }

    // Legacy channel-0 row accessor (keeps old event code buildable)
    ImageRow operator[](uint32_t y) {
        return ImageRow(&m_Data[0][y * m_Width]);
    }
    const ImageRow operator[](uint32_t y) const {
        return ImageRow(const_cast<uint16_t*>(&m_Data[0][y * m_Width]));
    }

    // ---- histogram ----
    std::vector<uint64_t> computeHistogram(uint32_t ch = 0) const {
        std::vector<uint64_t> hist(m_Levels, 0);
        for (uint16_t v : m_Data[ch])
            if (v < m_Levels) hist[v]++;
        return hist;
    }

    // ---- save ----
    void saveTIFF(const std::string& path) {
        if (m_Channels == 1) saveTIFF16Gray(path);
        else                  saveTIFF8Color(path, false);
    }

    void saveTIFF8bit(const std::string& path) {
        if (m_Channels == 1) saveTIFF8Gray(path);
        else                  saveTIFF8Color(path, true);
    }

private:
    uint32_t m_Width = 0, m_Height = 0, m_Levels = 0;
    uint16_t m_BitsPerSample = 0;
    uint16_t m_Channels = 1;
    std::string m_Path;
    std::vector<std::vector<uint16_t>> m_Data;   // [channel][y*w+x]

    // ---- loaders ----
    void loadTIFF() {
        TIFF* tif = TIFFOpen(m_Path.c_str(), "r");
        if (!tif) throw std::runtime_error("Cannot open: " + m_Path);

        uint16_t spp = 1, sf = SAMPLEFORMAT_UINT, pc = PLANARCONFIG_CONTIG;
        TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,      &m_Width);
        TIFFGetField(tif, TIFFTAG_IMAGELENGTH,     &m_Height);
        TIFFGetField(tif, TIFFTAG_BITSPERSAMPLE,   &m_BitsPerSample);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &spp);
        TIFFGetField(tif, TIFFTAG_SAMPLEFORMAT,    &sf);
        TIFFGetField(tif, TIFFTAG_PLANARCONFIG,    &pc);

        m_Channels = (spp >= 3) ? 3 : 1;
        m_Levels   = 1u << m_BitsPerSample;
        m_Data.assign(m_Channels, std::vector<uint16_t>(m_Width * m_Height, 0));

        std::cout << "  Loaded: " << m_Width << "x" << m_Height
                  << "  channels=" << m_Channels
                  << "  bits=" << m_BitsPerSample << "\n";

        if (m_Channels == 1) {
            loadGray(tif);
        } else {
            (pc == PLANARCONFIG_CONTIG) ? loadRGBContig(tif)
                                        : loadRGBSeparate(tif);
        }
        TIFFClose(tif);
        fixBitDepth();
    }

    void loadGray(TIFF* tif) {
        if (m_BitsPerSample == 8) {
            std::vector<uint8_t> row(m_Width);
            for (uint32_t y = 0; y < m_Height; y++) {
                TIFFReadScanline(tif, row.data(), y);
                for (uint32_t x = 0; x < m_Width; x++)
                    m_Data[0][y * m_Width + x] = row[x];
            }
        } else {
            for (uint32_t y = 0; y < m_Height; y++)
                TIFFReadScanline(tif, &m_Data[0][y * m_Width], y);
        }
    }

    void loadRGBContig(TIFF* tif) {
        if (m_BitsPerSample == 8) {
            std::vector<uint8_t> row(m_Width * 3);
            for (uint32_t y = 0; y < m_Height; y++) {
                TIFFReadScanline(tif, row.data(), y);
                for (uint32_t x = 0; x < m_Width; x++) {
                    m_Data[0][y*m_Width+x] = row[x*3+0];
                    m_Data[1][y*m_Width+x] = row[x*3+1];
                    m_Data[2][y*m_Width+x] = row[x*3+2];
                }
            }
        } else {
            std::vector<uint16_t> row(m_Width * 3);
            for (uint32_t y = 0; y < m_Height; y++) {
                TIFFReadScanline(tif, row.data(), y);
                for (uint32_t x = 0; x < m_Width; x++) {
                    m_Data[0][y*m_Width+x] = row[x*3+0];
                    m_Data[1][y*m_Width+x] = row[x*3+1];
                    m_Data[2][y*m_Width+x] = row[x*3+2];
                }
            }
        }
    }

    void loadRGBSeparate(TIFF* tif) {
        for (uint16_t ch = 0; ch < 3; ch++) {
            if (m_BitsPerSample == 8) {
                std::vector<uint8_t> row(m_Width);
                for (uint32_t y = 0; y < m_Height; y++) {
                    TIFFReadScanline(tif, row.data(), y, ch);
                    for (uint32_t x = 0; x < m_Width; x++)
                        m_Data[ch][y*m_Width+x] = row[x];
                }
            } else {
                for (uint32_t y = 0; y < m_Height; y++)
                    TIFFReadScanline(tif, &m_Data[ch][y*m_Width], y, ch);
            }
        }
    }

    void fixBitDepth() {
        uint16_t mx = 0;
        for (uint16_t ch = 0; ch < m_Channels; ch++)
            for (auto v : m_Data[ch]) mx = std::max(mx, v);
        if (mx > 255 && m_BitsPerSample == 8) {
            std::cout << "  WARNING: correcting declared bit-depth to 16\n";
            m_BitsPerSample = 16;
            m_Levels = 65536;
        }
    }

    // ---- savers ----
    void setCommonTags(TIFF* out, uint16_t spp, uint16_t bps) {
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH,      m_Width);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH,     m_Height);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, spp);
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE,   bps);
        TIFFSetField(out, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);
    }

    void saveTIFF16Gray(const std::string& path) {
        TIFF* out = TIFFOpen(path.c_str(), "w");
        if (!out) throw std::runtime_error("Cannot open for writing: " + path);
        setCommonTags(out, 1, m_BitsPerSample);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, m_Width));
        for (uint32_t y = 0; y < m_Height; y++)
            TIFFWriteScanline(out, (void*)&m_Data[0][y*m_Width], y);
        TIFFClose(out);
    }

    void saveTIFF8Gray(const std::string& path) {
        uint16_t mn = 65535, mx = 0;
        for (auto v : m_Data[0]) { mn = std::min(mn,v); mx = std::max(mx,v); }
        double scale = (mx > mn) ? 255.0 / (mx - mn) : 0.0;

        TIFF* out = TIFFOpen(path.c_str(), "w");
        if (!out) throw std::runtime_error("Cannot open for writing: " + path);
        setCommonTags(out, 1, 8);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, m_Width));

        std::vector<uint8_t> row(m_Width);
        for (uint32_t y = 0; y < m_Height; y++) {
            for (uint32_t x = 0; x < m_Width; x++)
                row[x] = static_cast<uint8_t>(
                    std::clamp((m_Data[0][y*m_Width+x] - mn) * scale, 0.0, 255.0));
            TIFFWriteScanline(out, row.data(), y);
        }
        TIFFClose(out);
    }

    void saveTIFF8Color(const std::string& path, bool normalize) {
        std::array<uint16_t,3> mn = {65535,65535,65535};
        std::array<uint16_t,3> mx = {0,0,0};
        if (normalize) {
            for (uint16_t ch = 0; ch < 3; ch++)
                for (auto v : m_Data[ch]) {
                    mn[ch] = std::min(mn[ch], v);
                    mx[ch] = std::max(mx[ch], v);
                }
        }

        TIFF* out = TIFFOpen(path.c_str(), "w");
        if (!out) throw std::runtime_error("Cannot open for writing: " + path);
        setCommonTags(out, 3, 8);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, m_Width * 3));

        std::vector<uint8_t> row(m_Width * 3);
        for (uint32_t y = 0; y < m_Height; y++) {
            for (uint32_t x = 0; x < m_Width; x++) {
                for (uint16_t ch = 0; ch < 3; ch++) {
                    uint16_t v = m_Data[ch][y*m_Width+x];
                    row[x*3+ch] = normalize && mx[ch] > mn[ch]
                        ? static_cast<uint8_t>(std::clamp((v-mn[ch])*255.0/(mx[ch]-mn[ch]),0.0,255.0))
                        : static_cast<uint8_t>(std::min<uint16_t>(v, 255));
                }
            }
            TIFFWriteScanline(out, row.data(), y);
        }
        TIFFClose(out);
    }
};

// ============================================================
//  Gaussian blur helper (used by multiple events)
// ============================================================

static std::vector<double> buildGaussianKernel1D(uint32_t kSize, double sigma) {
    int r = static_cast<int>(kSize / 2);
    std::vector<double> k(kSize);
    double sum = 0.0;
    for (int i = -r; i <= r; i++) {
        k[i+r] = std::exp(-(double)(i*i) / (2.0*sigma*sigma));
        sum += k[i+r];
    }
    for (auto& v : k) v /= sum;
    return k;
}

static void gaussianBlurInPlace(ImageObject& img, uint32_t kSize, double sigma) {
    if (kSize < 3 || kSize % 2 == 0)
        throw std::runtime_error("Gaussian kernel must be odd and >= 3");
    if (sigma <= 0.0)
        throw std::runtime_error("Sigma must be > 0");

    const uint32_t W = img.width(), H = img.height(), L = img.levels();
    const int r = static_cast<int>(kSize / 2);
    auto kern = buildGaussianKernel1D(kSize, sigma);

    for (uint16_t ch = 0; ch < img.channels(); ch++) {
        std::vector<uint16_t> tmp(W * H);

        // horizontal pass
        for (int y = 0; y < (int)H; y++)
            for (int x = 0; x < (int)W; x++) {
                double acc = 0.0;
                for (int k = -r; k <= r; k++) {
                    int xx = std::clamp(x+k, 0, (int)W-1);
                    acc += kern[k+r] * img.pixel(y, xx, ch);
                }
                tmp[y*W+x] = static_cast<uint16_t>(std::clamp(acc, 0.0, (double)(L-1)));
            }

        // vertical pass
        for (int y = 0; y < (int)H; y++)
            for (int x = 0; x < (int)W; x++) {
                double acc = 0.0;
                for (int k = -r; k <= r; k++) {
                    int yy = std::clamp(y+k, 0, (int)H-1);
                    acc += kern[k+r] * tmp[yy*W+x];
                }
                img.pixel(y, x, ch) = static_cast<uint16_t>(std::clamp(acc, 0.0, (double)(L-1)));
            }
    }
}

// ============================================================
//  EXISTING EVENTS  (all updated for multi-channel)
// ============================================================

class InvertImageEvent {
public:
    InvertImageEvent(const std::string& in, const std::string& out)
        : m_In(in), m_Out(out) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t L = img.levels();
        for (uint16_t ch = 0; ch < img.channels(); ch++)
            for (uint32_t y = 0; y < img.height(); y++)
                for (uint32_t x = 0; x < img.width(); x++)
                    img.pixel(y,x,ch) = L - 1 - img.pixel(y,x,ch);
        img.saveTIFF8bit(m_Out);
        std::cout << "Inversion done.\n";
    }
private:
    std::string m_In, m_Out;
};

// ─────────────────────────────────────────────────────────────

class LogTransformEvent {
public:
    LogTransformEvent(const std::string& in, const std::string& out)
        : m_In(in), m_Out(out) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t L = img.levels();
        const double c = (L-1) / std::log((double)L);
        for (uint16_t ch = 0; ch < img.channels(); ch++)
            for (uint32_t y = 0; y < img.height(); y++)
                for (uint32_t x = 0; x < img.width(); x++)
                    img.pixel(y,x,ch) = static_cast<uint16_t>(
                        c * std::log(img.pixel(y,x,ch) + 1.0));
        img.saveTIFF8bit(m_Out);
        std::cout << "Log transform done.\n";
    }
private:
    std::string m_In, m_Out;
};

// ─────────────────────────────────────────────────────────────

class GammaTransformEvent {
public:
    GammaTransformEvent(const std::string& in, const std::string& out, double gamma)
        : m_In(in), m_Out(out), m_Gamma(gamma) {}

    void execute() {
        ImageObject img(m_In);

        // Find global max across all channels
        uint16_t globalMax = 0;
        for (uint16_t ch = 0; ch < img.channels(); ch++)
            for (uint32_t y = 0; y < img.height(); y++)
                for (uint32_t x = 0; x < img.width(); x++)
                    globalMax = std::max(globalMax, img.pixel(y,x,ch));

        if (globalMax == 0) { img.saveTIFF8bit(m_Out); return; }
        const double inputMax = static_cast<double>(globalMax);

        std::vector<uint16_t> lut(globalMax + 1);
        for (uint32_t i = 0; i <= globalMax; i++) {
            double n = i / inputMax;
            lut[i] = static_cast<uint16_t>(std::round(std::pow(n, m_Gamma) * inputMax));
        }

        for (uint16_t ch = 0; ch < img.channels(); ch++)
            for (uint32_t y = 0; y < img.height(); y++)
                for (uint32_t x = 0; x < img.width(); x++) {
                    uint16_t v = img.pixel(y,x,ch);
                    img.pixel(y,x,ch) = (v <= globalMax) ? lut[v] : v;
                }

        img.saveTIFF8bit(m_Out);
        std::cout << "Gamma transform done.\n";
    }
private:
    std::string m_In, m_Out;
    double m_Gamma;
};

// ─────────────────────────────────────────────────────────────

class PieceWiseContrastEvent {
public:
    PieceWiseContrastEvent(const std::string& in, const std::string& out,
                           uint16_t r1, uint16_t s1, uint16_t r2, uint16_t s2)
        : m_In(in), m_Out(out), r1(r1), s1(s1), r2(r2), s2(s2) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t L = img.levels();
        const double mx = L - 1;

        if (r1 == 0 || r2 <= r1 || r2 >= mx)
            throw std::runtime_error("Invalid contrast parameters");

        std::vector<uint16_t> lut(L);
        for (uint32_t r = 0; r < L; r++) {
            double s;
            if      (r <= r1) s = (s1 / (double)r1) * r;
            else if (r <= r2) s = ((s2-s1) / (double)(r2-r1)) * (r-r1) + s1;
            else              s = ((mx-s2) / (mx-r2)) * (r-r2) + s2;
            lut[r] = static_cast<uint16_t>(std::lround(std::clamp(s,0.0,mx)));
        }

        for (uint16_t ch = 0; ch < img.channels(); ch++)
            for (uint32_t y = 0; y < img.height(); y++)
                for (uint32_t x = 0; x < img.width(); x++)
                    img.pixel(y,x,ch) = lut[img.pixel(y,x,ch)];

        img.saveTIFF8bit(m_Out);
        std::cout << "Piecewise contrast done.\n";
    }
private:
    std::string m_In, m_Out;
    uint16_t r1, s1, r2, s2;
};

// ─────────────────────────────────────────────────────────────

class IntensityRampEvent {
public:
    IntensityRampEvent(const std::string& in, const std::string& out,
                       uint16_t start, uint16_t end)
        : m_In(in), m_Out(out), m_Start(start), m_End(end) {}

    void execute() {
        ImageObject img(m_In);
        if (m_End <= m_Start) throw std::runtime_error("Invalid ramp range");

        const uint32_t L = img.levels();
        const double mx = L - 1;
        const double slope = mx / (m_End - m_Start);

        std::vector<uint16_t> lut(L);
        for (uint32_t r = 0; r < L; r++) {
            double s = (r < m_Start) ? 0.0
                     : (r > m_End)   ? mx
                     : slope * (r - m_Start);
            lut[r] = static_cast<uint16_t>(std::lround(std::clamp(s,0.0,mx)));
        }

        for (uint16_t ch = 0; ch < img.channels(); ch++)
            for (uint32_t y = 0; y < img.height(); y++)
                for (uint32_t x = 0; x < img.width(); x++)
                    img.pixel(y,x,ch) = lut[img.pixel(y,x,ch)];

        img.saveTIFF(m_Out);
        std::cout << "Intensity ramp done.\n";
    }
private:
    std::string m_In, m_Out;
    uint16_t m_Start, m_End;
};

// ─────────────────────────────────────────────────────────────

class IntensityLevelSlicingEvent {
public:
    enum class Mode { WITHOUT_BG, WITH_BG };

    IntensityLevelSlicingEvent(const std::string& in, const std::string& out,
                               uint16_t r1, uint16_t r2, uint16_t k, const std::string& mode)
        : m_In(in), m_Out(out), m_R1(r1), m_R2(r2), m_K(k),
          m_Mode(mode == "bg" ? Mode::WITH_BG : Mode::WITHOUT_BG) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t L = img.levels();
        if (m_R2 < m_R1) throw std::runtime_error("Invalid slice range");

        std::vector<uint16_t> lut(L);
        for (uint32_t r = 0; r < L; r++) {
            if (r >= m_R1 && r <= m_R2)
                lut[r] = m_K;
            else
                lut[r] = (m_Mode == Mode::WITH_BG) ? r : 0;
        }

        for (uint16_t ch = 0; ch < img.channels(); ch++)
            for (uint32_t y = 0; y < img.height(); y++)
                for (uint32_t x = 0; x < img.width(); x++)
                    img.pixel(y,x,ch) = lut[img.pixel(y,x,ch)];

        img.saveTIFF(m_Out);
        std::cout << "Level slicing done.\n";
    }
private:
    std::string m_In, m_Out;
    uint16_t m_R1, m_R2, m_K;
    Mode m_Mode;
};

// ─────────────────────────────────────────────────────────────

class BitPlaneSliceEvent {
public:
    enum class Mode { WITHOUT_BG, WITH_BG };

    BitPlaneSliceEvent(const std::string& in, const std::string& out,
                       uint16_t bit, const std::string& mode)
        : m_In(in), m_Out(out), m_Bit(bit),
          m_Mode(mode == "bg" ? Mode::WITH_BG : Mode::WITHOUT_BG) {}

    void execute() {
        ImageObject img(m_In);
        if (m_Bit >= img.bits()) throw std::runtime_error("Invalid bit index");
        const uint16_t maxVal = (1u << img.bits()) - 1;

        for (uint16_t ch = 0; ch < img.channels(); ch++)
            for (uint32_t y = 0; y < img.height(); y++)
                for (uint32_t x = 0; x < img.width(); x++) {
                    uint16_t p = img.pixel(y,x,ch);
                    uint16_t b = (p >> m_Bit) & 1;
                    img.pixel(y,x,ch) = (m_Mode == Mode::WITH_BG)
                        ? (b ? maxVal : p) : (b ? maxVal : 0);
                }

        img.saveTIFF8bit(m_Out);
        std::cout << "Bit-plane slice done.\n";
    }
private:
    std::string m_In, m_Out;
    uint16_t m_Bit;
    Mode m_Mode;
};

// ─────────────────────────────────────────────────────────────

class HistogramEqualizationEvent {
public:
    HistogramEqualizationEvent(const std::string& in, const std::string& out)
        : m_In(in), m_Out(out) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t L = img.levels();
        const uint64_t N = (uint64_t)img.width() * img.height();

        // Per-channel equalization (independent channels)
        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            auto hist = img.computeHistogram(ch);
            std::vector<double> cdf(L, 0.0);
            cdf[0] = (double)hist[0] / N;
            for (uint32_t i = 1; i < L; i++)
                cdf[i] = cdf[i-1] + (double)hist[i] / N;

            std::vector<uint16_t> lut(L);
            for (uint32_t i = 0; i < L; i++)
                lut[i] = static_cast<uint16_t>(std::lround(cdf[i] * (L-1)));

            for (uint32_t y = 0; y < img.height(); y++)
                for (uint32_t x = 0; x < img.width(); x++)
                    img.pixel(y,x,ch) = lut[img.pixel(y,x,ch)];
        }

        img.saveTIFF8bit(m_Out);
        std::cout << "Histogram equalization done.\n";
    }
private:
    std::string m_In, m_Out;
};

// ─────────────────────────────────────────────────────────────

class HistogramStatsEvent {
public:
    HistogramStatsEvent(const std::string& in) : m_In(in) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t L = img.levels();
        const uint64_t N = (uint64_t)img.width() * img.height();

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::string chName = img.isColor()
                ? std::string(ch==0?"R":ch==1?"G":"B") : "Gray";
            auto hist = img.computeHistogram(ch);

            uint32_t minL = L, maxL = 0;
            double mean = 0.0, var = 0.0, entropy = 0.0;
            for (uint32_t i = 0; i < L; i++) {
                if (hist[i] > 0) { minL = std::min(minL,i); maxL = std::max(maxL,i); }
                mean += i * (double)hist[i];
            }
            mean /= N;
            for (uint32_t i = 0; i < L; i++) {
                double d = i - mean;
                var += d * d * hist[i];
                if (hist[i] > 0) { double p = (double)hist[i]/N; entropy -= p*std::log2(p); }
            }
            var /= N;

            std::cout << "\n--- Channel: " << chName << " ---\n"
                      << "  Range    : [" << minL << ", " << maxL << "]\n"
                      << "  Mean     : " << mean   << "\n"
                      << "  Variance : " << var    << "\n"
                      << "  Entropy  : " << entropy << " bits\n";

            // Compact histogram (16 bins)
            const uint32_t bins = 16, step = L / bins;
            std::cout << "  Histogram:\n";
            for (uint32_t b = 0; b < bins; b++) {
                uint64_t cnt = 0;
                for (uint32_t i = b*step; i < (b+1)*step; i++) cnt += hist[i];
                double pct = 100.0 * cnt / N;
                std::cout << "  [" << b*step << "-" << (b+1)*step-1 << "] ";
                for (int i = 0; i < (int)(pct/2); i++) std::cout << '#';
                std::cout << " (" << pct << "%)\n";
            }
        }
    }
private:
    std::string m_In;
};

// ─────────────────────────────────────────────────────────────

class HistogramMatchingEvent {
public:
    HistogramMatchingEvent(const std::string& src, const std::string& ref,
                           const std::string& out)
        : m_Src(src), m_Ref(ref), m_Out(out) {}

    void execute() {
        ImageObject src(m_Src), ref(m_Ref);
        if (src.levels() != ref.levels())
            throw std::runtime_error("Src and ref must have same bit depth");
        if (src.channels() != ref.channels())
            throw std::runtime_error("Src and ref must have same number of channels");

        const uint32_t L = src.levels();
        const uint64_t Ns = (uint64_t)src.width() * src.height();
        const uint64_t Nr = (uint64_t)ref.width() * ref.height();

        for (uint16_t ch = 0; ch < src.channels(); ch++) {
            auto hS = src.computeHistogram(ch);
            auto hR = ref.computeHistogram(ch);

            std::vector<double> cS(L,0.0), cR(L,0.0);
            cS[0] = (double)hS[0]/Ns;  cR[0] = (double)hR[0]/Nr;
            for (uint32_t i = 1; i < L; i++) {
                cS[i] = cS[i-1] + (double)hS[i]/Ns;
                cR[i] = cR[i-1] + (double)hR[i]/Nr;
            }

            std::vector<uint16_t> lut(L);
            for (uint32_t s = 0; s < L; s++) {
                uint32_t r = 0;
                while (r < L-1 && cR[r] < cS[s]) r++;
                lut[s] = static_cast<uint16_t>(r);
            }

            for (uint32_t y = 0; y < src.height(); y++)
                for (uint32_t x = 0; x < src.width(); x++)
                    src.pixel(y,x,ch) = lut[src.pixel(y,x,ch)];
        }

        src.saveTIFF8bit(m_Out);
        std::cout << "Histogram matching done.\n";
    }
private:
    std::string m_Src, m_Ref, m_Out;
};

// ─────────────────────────────────────────────────────────────

class LocalHistogramEnhancementEvent {
public:
    LocalHistogramEnhancementEvent(const std::string& in, const std::string& out, uint32_t win)
        : m_In(in), m_Out(out), m_Win(win) {
        if (win < 3 || win % 2 == 0)
            throw std::runtime_error("Window must be odd and >= 3");
    }

    void execute() {
        ImageObject img(m_In);
        const uint32_t L = img.levels(), W = img.width(), H = img.height();
        const int r = static_cast<int>(m_Win / 2);

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::vector<uint16_t> out(W * H);
            for (int y = 0; y < (int)H; y++) {
                for (int x = 0; x < (int)W; x++) {
                    std::vector<uint64_t> hist(L, 0);
                    uint64_t total = 0;
                    for (int dy = -r; dy <= r; dy++) {
                        for (int dx = -r; dx <= r; dx++) {
                            int yy = std::clamp(y+dy,0,(int)H-1);
                            int xx = std::clamp(x+dx,0,(int)W-1);
                            hist[img.pixel(yy,xx,ch)]++;
                            total++;
                        }
                    }
                    uint16_t c = img.pixel(y,x,ch);
                    uint64_t cdf = 0;
                    for (uint32_t i = 0; i <= c; i++) cdf += hist[i];
                    out[y*W+x] = static_cast<uint16_t>(
                        std::round((double)cdf * (L-1) / total));
                }
            }
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    img.pixel(y,x,ch) = out[y*W+x];
        }

        img.saveTIFF8bit(m_Out);
        std::cout << "Local histogram enhancement done.\n";
    }
private:
    std::string m_In, m_Out;
    uint32_t m_Win;
};

// ─────────────────────────────────────────────────────────────

class BoxSmoothingEvent {
public:
    BoxSmoothingEvent(const std::string& in, const std::string& out, uint32_t k)
        : m_In(in), m_Out(out), m_K(k) {
        if (k < 3 || k % 2 == 0) throw std::runtime_error("Kernel must be odd and >= 3");
    }

    void execute() {
        ImageObject img(m_In);
        const uint32_t W = img.width(), H = img.height(), L = img.levels();
        const int r = static_cast<int>(m_K / 2);

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::vector<uint16_t> out(W * H, 0);
            for (int y = 0; y < (int)H; y++) {
                for (int x = 0; x < (int)W; x++) {
                    uint64_t sum = 0; uint32_t cnt = 0;
                    for (int dy = -r; dy <= r; dy++) {
                        for (int dx = -r; dx <= r; dx++) {
                            int yy = std::clamp(y+dy,0,(int)H-1);
                            int xx = std::clamp(x+dx,0,(int)W-1);
                            sum += img.pixel(yy,xx,ch); cnt++;
                        }
                    }
                    out[y*W+x] = static_cast<uint16_t>(sum / cnt);
                }
            }
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    img.pixel(y,x,ch) = out[y*W+x];
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Box smoothing done.\n";
    }
private:
    std::string m_In, m_Out;
    uint32_t m_K;
};

// ─────────────────────────────────────────────────────────────

class GaussianLowPassEvent {
public:
    GaussianLowPassEvent(const std::string& in, const std::string& out,
                         uint32_t k, double sigma)
        : m_In(in), m_Out(out), m_K(k), m_Sigma(sigma) {}

    void execute() {
        ImageObject img(m_In);
        gaussianBlurInPlace(img, m_K, m_Sigma);
        img.saveTIFF8bit(m_Out);
        std::cout << "Gaussian blur done.\n";
    }
private:
    std::string m_In, m_Out;
    uint32_t m_K;
    double m_Sigma;
};

// ─────────────────────────────────────────────────────────────

class HighPassSharpenEvent {
public:
    HighPassSharpenEvent(const std::string& in, const std::string& out, double strength = 1.0)
        : m_In(in), m_Out(out), m_Str(strength) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t W = img.width(), H = img.height(), L = img.levels();
        const int K[3][3] = {{0,-1,0},{-1,4,-1},{0,-1,0}};

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::vector<int> lap(W * H, 0);
            for (int y = 1; y < (int)H-1; y++)
                for (int x = 1; x < (int)W-1; x++) {
                    int s = 0;
                    for (int j = -1; j <= 1; j++)
                        for (int i = -1; i <= 1; i++)
                            s += K[j+1][i+1] * img.pixel(y+j,x+i,ch);
                    lap[y*W+x] = s;
                }
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    img.pixel(y,x,ch) = static_cast<uint16_t>(std::clamp(
                        img.pixel(y,x,ch) + m_Str * lap[y*W+x], 0.0, (double)(L-1)));
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "High-pass sharpening done.\n";
    }
private:
    std::string m_In, m_Out;
    double m_Str;
};

// ─────────────────────────────────────────────────────────────

class UnsharpHighboostEvent {
public:
    UnsharpHighboostEvent(const std::string& in, const std::string& out, double A)
        : m_In(in), m_Out(out), m_A(A) {
        if (A < 1.0) throw std::runtime_error("A must be >= 1.0");
    }

    void execute() {
        ImageObject img(m_In);
        const uint32_t W = img.width(), H = img.height(), L = img.levels();
        const int G[3][3] = {{1,2,1},{2,4,2},{1,2,1}};

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::vector<double> blur(W * H, 0.0);
            for (int y = 1; y < (int)H-1; y++)
                for (int x = 1; x < (int)W-1; x++) {
                    int acc = 0;
                    for (int j = -1; j <= 1; j++)
                        for (int i = -1; i <= 1; i++)
                            acc += G[j+1][i+1] * img.pixel(y+j,x+i,ch);
                    blur[y*W+x] = acc / 16.0;
                }
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    img.pixel(y,x,ch) = static_cast<uint16_t>(std::clamp(
                        m_A * img.pixel(y,x,ch) - blur[y*W+x], 0.0, (double)(L-1)));
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Unsharp/Highboost done.\n";
    }
private:
    std::string m_In, m_Out;
    double m_A;
};

// ─────────────────────────────────────────────────────────────

class GradientEdgeEnhancementEvent {
public:
    GradientEdgeEnhancementEvent(const std::string& in, const std::string& out, double k)
        : m_In(in), m_Out(out), m_K(k) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t W = img.width(), H = img.height(), L = img.levels();
        const int Sx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
        const int Sy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::vector<uint16_t> out(W * H);
            for (int y = 1; y < (int)H-1; y++) {
                for (int x = 1; x < (int)W-1; x++) {
                    int gx=0, gy=0;
                    for (int j=-1;j<=1;j++) for (int i=-1;i<=1;i++) {
                        uint16_t p = img.pixel(y+j,x+i,ch);
                        gx += Sx[j+1][i+1]*p;  gy += Sy[j+1][i+1]*p;
                    }
                    out[y*W+x] = static_cast<uint16_t>(std::clamp(
                        img.pixel(y,x,ch) + m_K*(std::abs(gx)+std::abs(gy)),
                        0.0, (double)(L-1)));
                }
            }
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    img.pixel(y,x,ch) = out[y*W+x];
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Gradient edge enhancement done.\n";
    }
private:
    std::string m_In, m_Out;
    double m_K;
};

// ─────────────────────────────────────────────────────────────

class MedianFilterEvent {
public:
    MedianFilterEvent(const std::string& in, const std::string& out, uint32_t win)
        : m_In(in), m_Out(out), m_Win(win) {
        if (win < 3 || win % 2 == 0) throw std::runtime_error("Window must be odd and >= 3");
    }

    void execute() {
        ImageObject img(m_In);
        const uint32_t W = img.width(), H = img.height();
        const int r = static_cast<int>(m_Win / 2);

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::vector<uint16_t> out(W * H);
            std::vector<uint16_t> nb; nb.reserve(m_Win * m_Win);
            for (int y = 0; y < (int)H; y++) {
                for (int x = 0; x < (int)W; x++) {
                    nb.clear();
                    for (int dy=-r;dy<=r;dy++) for (int dx=-r;dx<=r;dx++) {
                        int yy = std::clamp(y+dy,0,(int)H-1);
                        int xx = std::clamp(x+dx,0,(int)W-1);
                        nb.push_back(img.pixel(yy,xx,ch));
                    }
                    std::nth_element(nb.begin(), nb.begin()+nb.size()/2, nb.end());
                    out[y*W+x] = nb[nb.size()/2];
                }
            }
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    img.pixel(y,x,ch) = out[y*W+x];
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Median filter done.\n";
    }
private:
    std::string m_In, m_Out;
    uint32_t m_Win;
};

// ─────────────────────────────────────────────────────────────

class RobertsEdgeEvent {
public:
    RobertsEdgeEvent(const std::string& in, const std::string& out) : m_In(in), m_Out(out) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t W = img.width(), H = img.height(), L = img.levels();

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::vector<uint16_t> out(W * H, 0);
            for (uint32_t y = 0; y < H-1; y++) {
                for (uint32_t x = 0; x < W-1; x++) {
                    int gx = img.pixel(y,x,ch) - img.pixel(y+1,x+1,ch);
                    int gy = img.pixel(y,x+1,ch) - img.pixel(y+1,x,ch);
                    out[y*W+x] = static_cast<uint16_t>(
                        std::clamp(std::abs(gx)+std::abs(gy), 0, (int)L-1));
                }
            }
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    img.pixel(y,x,ch) = out[y*W+x];
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Roberts edge detection done.\n";
    }
private:
    std::string m_In, m_Out;
};

// ─────────────────────────────────────────────────────────────

class PrewittEdgeEvent {
public:
    PrewittEdgeEvent(const std::string& in, const std::string& out) : m_In(in), m_Out(out) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t W = img.width(), H = img.height(), L = img.levels();
        const int Gx[3][3] = {{-1,0,1},{-1,0,1},{-1,0,1}};
        const int Gy[3][3] = {{-1,-1,-1},{0,0,0},{1,1,1}};

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::vector<uint16_t> out(W * H, 0);
            for (int y = 1; y < (int)H-1; y++) {
                for (int x = 1; x < (int)W-1; x++) {
                    int gx=0, gy=0;
                    for (int j=-1;j<=1;j++) for (int i=-1;i<=1;i++) {
                        uint16_t p = img.pixel(y+j,x+i,ch);
                        gx += Gx[j+1][i+1]*p;  gy += Gy[j+1][i+1]*p;
                    }
                    out[y*W+x] = static_cast<uint16_t>(
                        std::clamp(std::abs(gx)+std::abs(gy), 0, (int)L-1));
                }
            }
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    img.pixel(y,x,ch) = out[y*W+x];
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Prewitt edge detection done.\n";
    }
private:
    std::string m_In, m_Out;
};

// ─────────────────────────────────────────────────────────────

class SobelEdgeEvent {
public:
    SobelEdgeEvent(const std::string& in, const std::string& out, uint16_t threshold = 0)
        : m_In(in), m_Out(out), m_Thr(threshold) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t W = img.width(), H = img.height(), L = img.levels();
        const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
        const int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::vector<uint16_t> mag(W * H, 0);
            uint16_t maxMag = 0;
            for (int y = 1; y < (int)H-1; y++) {
                for (int x = 1; x < (int)W-1; x++) {
                    int gx=0, gy=0;
                    for (int j=-1;j<=1;j++) for (int i=-1;i<=1;i++) {
                        uint16_t p = img.pixel(y+j,x+i,ch);
                        gx += Gx[j+1][i+1]*p;  gy += Gy[j+1][i+1]*p;
                    }
                    int g = std::abs(gx)+std::abs(gy);
                    mag[y*W+x] = g;
                    maxMag = std::max(maxMag, (uint16_t)g);
                }
            }
            for (uint32_t y = 0; y < H; y++) {
                for (uint32_t x = 0; x < W; x++) {
                    uint16_t norm = maxMag > 0
                        ? static_cast<uint16_t>((mag[y*W+x]*(L-1)) / maxMag) : 0;
                    img.pixel(y,x,ch) = (m_Thr > 0)
                        ? (norm >= m_Thr ? (L-1) : 0) : norm;
                }
            }
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Sobel edge detection done.\n";
    }
private:
    std::string m_In, m_Out;
    uint16_t m_Thr;
};

// ─────────────────────────────────────────────────────────────

class LaplacianSharpenEvent {
public:
    enum class Mode { FOUR, EIGHT };

    LaplacianSharpenEvent(const std::string& in,
                          const std::string& lapOut, const std::string& sharpOut, Mode mode)
        : m_In(in), m_LapOut(lapOut), m_SharpOut(sharpOut), m_Mode(mode) {}

    void execute() {
        ImageObject img(m_In);
        ImageObject lap  = img;
        ImageObject sharp = img;

        const int K4[3][3] = {{0,-1,0},{-1,4,-1},{0,-1,0}};
        const int K8[3][3] = {{-1,-1,-1},{-1,8,-1},{-1,-1,-1}};
        const int (*K)[3]  = (m_Mode == Mode::FOUR) ? K4 : K8;
        const int H = img.height(), W = img.width(), L = img.levels();

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            for (int y = 1; y < H-1; y++)
                for (int x = 1; x < W-1; x++) {
                    int s = 0;
                    for (int j=-1;j<=1;j++) for (int i=-1;i<=1;i++)
                        s += K[j+1][i+1] * img.pixel(y+j,x+i,ch);
                    lap.pixel(y,x,ch) = static_cast<uint16_t>(std::clamp(s,0,L-1));
                }
            for (uint32_t y = 0; y < (uint32_t)H; y++)
                for (uint32_t x = 0; x < (uint32_t)W; x++) {
                    int v = img.pixel(y,x,ch) + lap.pixel(y,x,ch);
                    sharp.pixel(y,x,ch) = static_cast<uint16_t>(std::clamp(v,0,L-1));
                }
        }

        lap.saveTIFF8bit(m_LapOut);
        sharp.saveTIFF8bit(m_SharpOut);
        std::cout << "Laplacian sharpening done.\n";
    }
private:
    std::string m_In, m_LapOut, m_SharpOut;
    Mode m_Mode;
};

// ─────────────────────────────────────────────────────────────

class WeightedAveragingEvent {
public:
    WeightedAveragingEvent(const std::string& in, const std::string& out)
        : m_In(in), m_Out(out) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t W = img.width(), H = img.height(), L = img.levels();
        const int K[3][3] = {{1,2,1},{2,4,2},{1,2,1}};

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::vector<uint16_t> out(W * H, 0);
            for (int y = 1; y < (int)H-1; y++) {
                for (int x = 1; x < (int)W-1; x++) {
                    int acc = 0;
                    for (int j=-1;j<=1;j++) for (int i=-1;i<=1;i++)
                        acc += K[j+1][i+1] * img.pixel(y+j,x+i,ch);
                    out[y*W+x] = static_cast<uint16_t>(std::clamp(acc/16,0,(int)L-1));
                }
            }
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    img.pixel(y,x,ch) = out[y*W+x];
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Weighted averaging done.\n";
    }
private:
    std::string m_In, m_Out;
};

// ─────────────────────────────────────────────────────────────

class GradientSharpenEvent {
public:
    GradientSharpenEvent(const std::string& in, const std::string& out, double k)
        : m_In(in), m_Out(out), m_K(k) {}

    void execute() {
        ImageObject img(m_In);
        const uint32_t W = img.width(), H = img.height(), L = img.levels();
        const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
        const int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            std::vector<uint16_t> out(W * H, 0);
            for (int y = 1; y < (int)H-1; y++) {
                for (int x = 1; x < (int)W-1; x++) {
                    int sx=0, sy=0;
                    for (int j=-1;j<=1;j++) for (int i=-1;i<=1;i++) {
                        uint16_t p = img.pixel(y+j,x+i,ch);
                        sx += Gx[j+1][i+1]*p;  sy += Gy[j+1][i+1]*p;
                    }
                    out[y*W+x] = static_cast<uint16_t>(std::clamp(
                        img.pixel(y,x,ch) + m_K*(std::abs(sx)+std::abs(sy)),
                        0.0, (double)(L-1)));
                }
            }
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    img.pixel(y,x,ch) = out[y*W+x];
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Gradient sharpening done.\n";
    }
private:
    std::string m_In, m_Out;
    double m_K;
};

// ─────────────────────────────────────────────────────────────

class BandFilterEvent {
public:
    enum class Mode { BANDPASS, BANDREJECT };

    BandFilterEvent(const std::string& in, const std::string& out,
                    uint32_t k1, double s1, uint32_t k2, double s2, Mode mode)
        : m_In(in), m_Out(out), m_K1(k1), m_S1(s1), m_K2(k2), m_S2(s2), m_Mode(mode) {}

    void execute() {
        ImageObject img(m_In);
        ImageObject lp1 = img;
        ImageObject lp2 = img;

        gaussianBlurInPlace(lp1, m_K1, m_S1);
        gaussianBlurInPlace(lp2, m_K2, m_S2);

        const uint32_t H = img.height(), W = img.width(), L = img.levels();
        for (uint16_t ch = 0; ch < img.channels(); ch++) {
            for (uint32_t y = 0; y < H; y++) {
                for (uint32_t x = 0; x < W; x++) {
                    double val = (m_Mode == Mode::BANDPASS)
                        ? lp2.pixel(y,x,ch) - lp1.pixel(y,x,ch)
                        : lp1.pixel(y,x,ch) + (img.pixel(y,x,ch) - lp2.pixel(y,x,ch));
                    img.pixel(y,x,ch) = static_cast<uint16_t>(
                        std::clamp(val, 0.0, (double)(L-1)));
                }
            }
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Band filter done.\n";
    }
private:
    std::string m_In, m_Out;
    uint32_t m_K1, m_K2;
    double m_S1, m_S2;
    Mode m_Mode;
};

// ============================================================
//  NEW: COLOR & PSEUDOCOLOR EVENTS
// ============================================================

// ─── Pseudocolor: map grayscale intensity to a false-color RGB ────────────────
class PseudoColorEvent {
public:
    PseudoColorEvent(const std::string& in, const std::string& out, Colormap cm)
        : m_In(in), m_Out(out), m_CM(cm) {}

    void execute() {
        ImageObject src(m_In);
        if (src.isColor())
            throw std::runtime_error("pseudocolor requires a grayscale input image");

        const uint32_t W = src.width(), H = src.height(), L = src.levels();
        ImageObject dst = ImageObject::create(W, H, src.bits(), 3);

        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                double t = (double)src.pixel(y,x,0) / (L - 1);
                auto rgb = applyColormap(t, m_CM);
                dst.pixel(y,x,0) = rgb[0];  // R
                dst.pixel(y,x,1) = rgb[1];  // G
                dst.pixel(y,x,2) = rgb[2];  // B
            }
        }

        dst.saveTIFF8bit(m_Out);
        std::cout << "Pseudocolor done.\n";
    }
private:
    std::string m_In, m_Out;
    Colormap m_CM;
};

// ─── Color Balance: independently scale R, G, B channels ──────────────────────
class ColorBalanceEvent {
public:
    ColorBalanceEvent(const std::string& in, const std::string& out,
                      double rScale, double gScale, double bScale)
        : m_In(in), m_Out(out), m_Scale{rScale, gScale, bScale} {}

    void execute() {
        ImageObject img(m_In);
        if (!img.isColor()) throw std::runtime_error("color_balance requires an RGB image");

        const uint32_t L = img.levels();
        for (uint16_t ch = 0; ch < 3; ch++) {
            for (uint32_t y = 0; y < img.height(); y++)
                for (uint32_t x = 0; x < img.width(); x++) {
                    double v = img.pixel(y,x,ch) * m_Scale[ch];
                    img.pixel(y,x,ch) = static_cast<uint16_t>(
                        std::clamp(v, 0.0, (double)(L-1)));
                }
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Color balance done.\n";
    }
private:
    std::string m_In, m_Out;
    double m_Scale[3];
};

// ─── HSV Adjust: shift hue, scale saturation and value ────────────────────────
class HSVAdjustEvent {
public:
    HSVAdjustEvent(const std::string& in, const std::string& out,
                   double hShift, double sMul, double vMul)
        : m_In(in), m_Out(out), m_HShift(hShift), m_SMul(sMul), m_VMul(vMul) {}

    void execute() {
        ImageObject img(m_In);
        if (!img.isColor()) throw std::runtime_error("hsv_adjust requires an RGB image");

        const uint32_t W = img.width(), H = img.height(), L = img.levels();
        const double maxVal = L - 1;

        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                double r = img.pixel(y,x,0) / maxVal;
                double g = img.pixel(y,x,1) / maxVal;
                double b = img.pixel(y,x,2) / maxVal;

                HSV hsv = rgbToHSV(r, g, b);
                hsv.h = std::fmod(hsv.h + m_HShift + 360.0, 360.0);
                hsv.s = std::clamp(hsv.s * m_SMul, 0.0, 1.0);
                hsv.v = std::clamp(hsv.v * m_VMul, 0.0, 1.0);

                auto rgb = hsvToRGB(hsv.h, hsv.s, hsv.v);
                img.pixel(y,x,0) = static_cast<uint16_t>(std::round(rgb[0] * maxVal));
                img.pixel(y,x,1) = static_cast<uint16_t>(std::round(rgb[1] * maxVal));
                img.pixel(y,x,2) = static_cast<uint16_t>(std::round(rgb[2] * maxVal));
            }
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "HSV adjustment done.\n";
    }
private:
    std::string m_In, m_Out;
    double m_HShift, m_SMul, m_VMul;
};

// ─── Grayscale Conversion: RGB → luminance (ITU-R BT.601) ─────────────────────
class GrayscaleConvertEvent {
public:
    GrayscaleConvertEvent(const std::string& in, const std::string& out)
        : m_In(in), m_Out(out) {}

    void execute() {
        ImageObject img(m_In);
        if (!img.isColor()) {
            std::cout << "Image is already grayscale, copying.\n";
            img.saveTIFF8bit(m_Out);
            return;
        }

        const uint32_t W = img.width(), H = img.height();
        ImageObject gray = ImageObject::create(W, H, img.bits(), 1);

        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                double lum = 0.299 * img.pixel(y,x,0)
                           + 0.587 * img.pixel(y,x,1)
                           + 0.114 * img.pixel(y,x,2);
                gray.pixel(y,x,0) = static_cast<uint16_t>(std::round(lum));
            }
        }
        gray.saveTIFF8bit(m_Out);
        std::cout << "Grayscale conversion done.\n";
    }
private:
    std::string m_In, m_Out;
};

// ─── Split Channels: saves R, G, B as individual grayscale TIFFs ──────────────
class ChannelSplitEvent {
public:
    ChannelSplitEvent(const std::string& in, const std::string& prefix)
        : m_In(in), m_Prefix(prefix) {}

    void execute() {
        ImageObject img(m_In);
        if (!img.isColor()) throw std::runtime_error("split_channels requires an RGB image");

        const char* names[] = {"R", "G", "B"};
        const uint32_t W = img.width(), H = img.height();

        for (uint16_t ch = 0; ch < 3; ch++) {
            ImageObject out = ImageObject::create(W, H, img.bits(), 1);
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    out.pixel(y,x,0) = img.pixel(y,x,ch);

            std::string path = m_Prefix + "_" + names[ch] + ".tif";
            out.saveTIFF8bit(path);
            std::cout << "  Channel " << names[ch] << " -> " << path << "\n";
        }
        std::cout << "Channel split done.\n";
    }
private:
    std::string m_In, m_Prefix;
};

// ─── Merge Channels: combine 3 grayscale TIFFs into RGB ───────────────────────
class ChannelMergeEvent {
public:
    ChannelMergeEvent(const std::string& r, const std::string& g,
                      const std::string& b, const std::string& out)
        : m_R(r), m_G(g), m_B(b), m_Out(out) {}

    void execute() {
        ImageObject rImg(m_R), gImg(m_G), bImg(m_B);

        if (rImg.width() != gImg.width() || rImg.width() != bImg.width() ||
            rImg.height() != gImg.height() || rImg.height() != bImg.height())
            throw std::runtime_error("All channel images must be the same size");

        const uint32_t W = rImg.width(), H = rImg.height();
        ImageObject merged = ImageObject::create(W, H, rImg.bits(), 3);

        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                merged.pixel(y,x,0) = rImg.pixel(y,x,0);
                merged.pixel(y,x,1) = gImg.pixel(y,x,0);
                merged.pixel(y,x,2) = bImg.pixel(y,x,0);
            }
        }
        merged.saveTIFF8bit(m_Out);
        std::cout << "Channel merge done.\n";
    }
private:
    std::string m_R, m_G, m_B, m_Out;
};

// ─── Color Histogram Equalization (HSV value channel) ─────────────────────────
//     Equalizes only the V channel in HSV space,
//     preserving hue and saturation (no color shift).
class ColorHistEqEvent {
public:
    ColorHistEqEvent(const std::string& in, const std::string& out)
        : m_In(in), m_Out(out) {}

    void execute() {
        ImageObject img(m_In);
        if (!img.isColor()) {
            // fallback to grayscale equalization
            HistogramEqualizationEvent(m_In, m_Out).execute();
            return;
        }

        const uint32_t W = img.width(), H = img.height();
        const double maxVal = img.levels() - 1;
        const uint64_t N = (uint64_t)W * H;
        const uint32_t BINS = 256;

        // ---- extract V channel, build histogram ----
        std::vector<double> vBuf(W * H);
        std::vector<uint64_t> hist(BINS, 0);

        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                double r = img.pixel(y,x,0) / maxVal;
                double g = img.pixel(y,x,1) / maxVal;
                double b = img.pixel(y,x,2) / maxVal;
                HSV hsv = rgbToHSV(r, g, b);
                vBuf[y*W+x] = hsv.v;
                uint32_t bin = static_cast<uint32_t>(std::clamp(hsv.v, 0.0, 1.0) * (BINS-1));
                hist[bin]++;
            }
        }

        // ---- CDF -> LUT ----
        std::vector<double> cdf(BINS, 0.0);
        cdf[0] = (double)hist[0] / N;
        for (uint32_t i = 1; i < BINS; i++)
            cdf[i] = cdf[i-1] + (double)hist[i] / N;

        // ---- apply equalized V back to RGB ----
        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                double r = img.pixel(y,x,0) / maxVal;
                double g = img.pixel(y,x,1) / maxVal;
                double b = img.pixel(y,x,2) / maxVal;
                HSV hsv = rgbToHSV(r, g, b);

                uint32_t bin = static_cast<uint32_t>(std::clamp(hsv.v,0.0,1.0)*(BINS-1));
                hsv.v = cdf[bin];

                auto rgb = hsvToRGB(hsv.h, hsv.s, hsv.v);
                img.pixel(y,x,0) = static_cast<uint16_t>(std::round(rgb[0] * maxVal));
                img.pixel(y,x,1) = static_cast<uint16_t>(std::round(rgb[1] * maxVal));
                img.pixel(y,x,2) = static_cast<uint16_t>(std::round(rgb[2] * maxVal));
            }
        }

        img.saveTIFF8bit(m_Out);
        std::cout << "Color histogram equalization (HSV) done.\n";
    }
private:
    std::string m_In, m_Out;
};

// ─── Sepia Tone: stylistic color transform ────────────────────────────────────
class SepiaToneEvent {
public:
    SepiaToneEvent(const std::string& in, const std::string& out)
        : m_In(in), m_Out(out) {}

    void execute() {
        ImageObject img(m_In);

        // If grayscale, promote to RGB first
        if (!img.isColor()) {
            const uint32_t W = img.width(), H = img.height();
            ImageObject rgb = ImageObject::create(W, H, img.bits(), 3);
            for (uint32_t y = 0; y < H; y++)
                for (uint32_t x = 0; x < W; x++)
                    rgb.pixel(y,x,0) = rgb.pixel(y,x,1) = rgb.pixel(y,x,2) =
                        img.pixel(y,x,0);
            img = rgb;
        }

        const uint32_t W = img.width(), H = img.height(), L = img.levels();
        const double mx = L - 1;

        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                double r = img.pixel(y,x,0), g = img.pixel(y,x,1), b = img.pixel(y,x,2);
                double nr = std::clamp(r*0.393 + g*0.769 + b*0.189, 0.0, mx);
                double ng = std::clamp(r*0.349 + g*0.686 + b*0.168, 0.0, mx);
                double nb = std::clamp(r*0.272 + g*0.534 + b*0.131, 0.0, mx);
                img.pixel(y,x,0) = static_cast<uint16_t>(std::round(nr));
                img.pixel(y,x,1) = static_cast<uint16_t>(std::round(ng));
                img.pixel(y,x,2) = static_cast<uint16_t>(std::round(nb));
            }
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Sepia tone done.\n";
    }
private:
    std::string m_In, m_Out;
};

// ─── Color Negative (per-channel invert) ─────────────────────────────────────
//     Alias of InvertImageEvent — shown separately for clarity
//     (use the invert command; it already handles color)

// ─── Color Threshold: pixels outside [rLo..rHi] x [gLo..gHi] x [bLo..bHi] ──
class ColorThresholdEvent {
public:
    ColorThresholdEvent(const std::string& in, const std::string& out,
                        uint16_t rLo, uint16_t rHi,
                        uint16_t gLo, uint16_t gHi,
                        uint16_t bLo, uint16_t bHi)
        : m_In(in), m_Out(out),
          m_Lo{rLo,gLo,bLo}, m_Hi{rHi,gHi,bHi} {}

    void execute() {
        ImageObject img(m_In);
        if (!img.isColor()) throw std::runtime_error("color_thresh requires an RGB image");

        const uint32_t W = img.width(), H = img.height(), L = img.levels();

        for (uint32_t y = 0; y < H; y++) {
            for (uint32_t x = 0; x < W; x++) {
                bool inside = true;
                for (uint16_t ch = 0; ch < 3; ch++) {
                    uint16_t v = img.pixel(y,x,ch);
                    if (v < m_Lo[ch] || v > m_Hi[ch]) { inside = false; break; }
                }
                if (!inside) {
                    img.pixel(y,x,0) = img.pixel(y,x,1) = img.pixel(y,x,2) = 0;
                }
            }
        }
        img.saveTIFF8bit(m_Out);
        std::cout << "Color threshold done.\n";
    }
private:
    std::string m_In, m_Out;
    uint16_t m_Lo[3], m_Hi[3];
};

// ============================================================
//  InputHandler
// ============================================================

class InputHandler {
public:
    static void run() {
        std::string cmd;
        while (true) {
            printMenu();
            std::cin >> cmd;
            try { dispatch(cmd); }
            catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
    }

private:
    static void printMenu() {
        std::cout << R"(
======= Image Enhancement Tool =======
--- Point Operations ---
  invert      <in> <out>
  log         <in> <out>
  gamma       <in> <out> <gamma>
  contrast    <in> <out> <r1> <s1> <r2> <s2>
  ramp        <in> <out> <start> <end>
  slice       <in> <out> <r1> <r2> <k> <bg|nobg>
  bit_slice   <in> <out> <bitindex> <bg|nobg>

--- Histogram ---
  hist_eq     <in> <out>          (per-channel)
  hist_stats  <in>                (per-channel for RGB)
  hist_match  <src> <ref> <out>
  local_hist  <in> <out> <winSize>
  color_hist_eq <in> <out>        (HSV-based, preserves color)

--- Spatial Filters ---
  smooth_box  <in> <out> <kernelSize>
  gaussian    <in> <out> <kernelSize> <sigma>
  weighted_avg <in> <out>
  median      <in> <out> <winSize>
  sharpen     <in> <out> <strength>
  unsharp     <in> <out> <A>

--- Edge Detection ---
  roberts     <in> <out>
  prewitt     <in> <out>
  sobel       <in> <out> [threshold]
  grad_edge   <in> <out> <k>
  grad_sharpen <in> <out> <k>
  laplacian   <in> <lapOut> <sharpOut> <4|8>

--- Color Image Operations ---
  grayscale       <in> <out>               RGB -> gray (BT.601)
  split_channels  <in> <prefix>            saves prefix_R/G/B.tif
  merge_channels  <r> <g> <b> <out>        3 gray -> RGB
  color_balance   <in> <out> <r> <g> <b>   scale factors per channel
  hsv_adjust      <in> <out> <hShift> <sMul> <vMul>
  sepia           <in> <out>
  color_thresh    <in> <out> <rL> <rH> <gL> <gH> <bL> <bH>

--- Pseudocolor (grayscale -> false color) ---
  pseudocolor <in> <out> <colormap>
  colormaps: jet | hot | cool | bone | rainbow | spring | summer

--- Bandpass/Bandreject ---
  bandpass    <in> <out> <k1> <s1> <k2> <s2>
  bandreject  <in> <out> <k1> <s1> <k2> <s2>

  quit
)";
    }

    static void dispatch(const std::string& cmd) {
        if (cmd == "invert") {
            std::string in, out; std::cin >> in >> out;
            InvertImageEvent(in, out).execute();
        }
        else if (cmd == "log") {
            std::string in, out; std::cin >> in >> out;
            LogTransformEvent(in, out).execute();
        }
        else if (cmd == "gamma") {
            std::string in, out; double g; std::cin >> in >> out >> g;
            GammaTransformEvent(in, out, g).execute();
        }
        else if (cmd == "contrast") {
            std::string in, out;
            uint16_t r1,s1,r2,s2;
            std::cin >> in >> out >> r1 >> s1 >> r2 >> s2;
            PieceWiseContrastEvent(in, out, r1, s1, r2, s2).execute();
        }
        else if (cmd == "ramp") {
            std::string in, out; uint16_t s, e; std::cin >> in >> out >> s >> e;
            IntensityRampEvent(in, out, s, e).execute();
        }
        else if (cmd == "slice") {
            std::string in, out, mode;
            uint16_t r1,r2,k;
            std::cin >> in >> out >> r1 >> r2 >> k >> mode;
            IntensityLevelSlicingEvent(in, out, r1, r2, k, mode).execute();
        }
        else if (cmd == "bit_slice") {
            std::string in, out, mode; uint16_t idx;
            std::cin >> in >> out >> idx >> mode;
            BitPlaneSliceEvent(in, out, idx, mode).execute();
        }
        else if (cmd == "hist_eq") {
            std::string in, out; std::cin >> in >> out;
            HistogramEqualizationEvent(in, out).execute();
        }
        else if (cmd == "hist_stats") {
            std::string in; std::cin >> in;
            HistogramStatsEvent(in).execute();
        }
        else if (cmd == "hist_match") {
            std::string src, ref, out; std::cin >> src >> ref >> out;
            HistogramMatchingEvent(src, ref, out).execute();
        }
        else if (cmd == "local_hist") {
            std::string in, out; uint32_t w; std::cin >> in >> out >> w;
            LocalHistogramEnhancementEvent(in, out, w).execute();
        }
        else if (cmd == "color_hist_eq") {
            std::string in, out; std::cin >> in >> out;
            ColorHistEqEvent(in, out).execute();
        }
        else if (cmd == "smooth_box") {
            std::string in, out; uint32_t k; std::cin >> in >> out >> k;
            BoxSmoothingEvent(in, out, k).execute();
        }
        else if (cmd == "gaussian") {
            std::string in, out; uint32_t k; double s;
            std::cin >> in >> out >> k >> s;
            GaussianLowPassEvent(in, out, k, s).execute();
        }
        else if (cmd == "weighted_avg") {
            std::string in, out; std::cin >> in >> out;
            WeightedAveragingEvent(in, out).execute();
        }
        else if (cmd == "median") {
            std::string in, out; uint32_t w; std::cin >> in >> out >> w;
            MedianFilterEvent(in, out, w).execute();
        }
        else if (cmd == "sharpen") {
            std::string in, out; double k; std::cin >> in >> out >> k;
            HighPassSharpenEvent(in, out, k).execute();
        }
        else if (cmd == "unsharp") {
            std::string in, out; double A; std::cin >> in >> out >> A;
            UnsharpHighboostEvent(in, out, A).execute();
        }
        else if (cmd == "roberts") {
            std::string in, out; std::cin >> in >> out;
            RobertsEdgeEvent(in, out).execute();
        }
        else if (cmd == "prewitt") {
            std::string in, out; std::cin >> in >> out;
            PrewittEdgeEvent(in, out).execute();
        }
        else if (cmd == "sobel") {
            std::string in, out; uint16_t t = 0;
            std::cin >> in >> out;
            if (std::cin.peek() != '\n') std::cin >> t;
            SobelEdgeEvent(in, out, t).execute();
        }
        else if (cmd == "grad_edge") {
            std::string in, out; double k; std::cin >> in >> out >> k;
            GradientEdgeEnhancementEvent(in, out, k).execute();
        }
        else if (cmd == "grad_sharpen") {
            std::string in, out; double k; std::cin >> in >> out >> k;
            GradientSharpenEvent(in, out, k).execute();
        }
        else if (cmd == "laplacian") {
            std::string in, lap, sharp; int mode;
            std::cin >> in >> lap >> sharp >> mode;
            auto m = (mode == 8) ? LaplacianSharpenEvent::Mode::EIGHT
                                 : LaplacianSharpenEvent::Mode::FOUR;
            LaplacianSharpenEvent(in, lap, sharp, m).execute();
        }
        else if (cmd == "bandpass") {
            std::string in, out; uint32_t k1,k2; double s1,s2;
            std::cin >> in >> out >> k1 >> s1 >> k2 >> s2;
            BandFilterEvent(in, out, k1, s1, k2, s2,
                BandFilterEvent::Mode::BANDPASS).execute();
        }
        else if (cmd == "bandreject") {
            std::string in, out; uint32_t k1,k2; double s1,s2;
            std::cin >> in >> out >> k1 >> s1 >> k2 >> s2;
            BandFilterEvent(in, out, k1, s1, k2, s2,
                BandFilterEvent::Mode::BANDREJECT).execute();
        }
        // ---- new color commands ----
        else if (cmd == "grayscale") {
            std::string in, out; std::cin >> in >> out;
            GrayscaleConvertEvent(in, out).execute();
        }
        else if (cmd == "split_channels") {
            std::string in, prefix; std::cin >> in >> prefix;
            ChannelSplitEvent(in, prefix).execute();
        }
        else if (cmd == "merge_channels") {
            std::string r, g, b, out; std::cin >> r >> g >> b >> out;
            ChannelMergeEvent(r, g, b, out).execute();
        }
        else if (cmd == "color_balance") {
            std::string in, out; double r, g, b;
            std::cin >> in >> out >> r >> g >> b;
            ColorBalanceEvent(in, out, r, g, b).execute();
        }
        else if (cmd == "hsv_adjust") {
            std::string in, out; double h, s, v;
            std::cin >> in >> out >> h >> s >> v;
            HSVAdjustEvent(in, out, h, s, v).execute();
        }
        else if (cmd == "sepia") {
            std::string in, out; std::cin >> in >> out;
            SepiaToneEvent(in, out).execute();
        }
        else if (cmd == "color_thresh") {
            std::string in, out;
            uint16_t rl,rh,gl,gh,bl,bh;
            std::cin >> in >> out >> rl >> rh >> gl >> gh >> bl >> bh;
            ColorThresholdEvent(in, out, rl, rh, gl, gh, bl, bh).execute();
        }
        else if (cmd == "pseudocolor") {
            std::string in, out, cm; std::cin >> in >> out >> cm;
            PseudoColorEvent(in, out, parseColormap(cm)).execute();
        }
        else if (cmd == "quit") {
            std::cout << "Goodbye.\n";
            std::exit(0);
        }
        else {
            std::cout << "Unknown command: " << cmd << "\n";
        }
    }
};

// ============================================================
int main() {
    InputHandler::run();
    return 0;
}
