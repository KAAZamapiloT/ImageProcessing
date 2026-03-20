"""
run_all_filters.py
==================
Batch script – applies every available color filter from ui_imge.py to the
images in the colorimages/ folder.

Folder layout expected (place this script in your Lab folder):
    Lab/
        ui_imge.py
        run_all_filters.py          ← this file
        colorimages/
            standard_test_images/   lena_color_512.tif, peppers_color.tif, etc.
            misc/                   4.2.01.tiff … house.tiff
            aerials/                2.1.01.tiff … 2.2.24.tiff

All outputs go to:
    colorimages/output/<filter_name>/

Each output filename encodes which input was used, e.g.
    peppers__color_invert.tif

Usage:
    python run_all_filters.py
"""

import os
import sys
import traceback

# ── Make sure ui_imge.py is importable ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui_imge import (
    # ── Enum needed for band filter ────────────────────────────────────────
    BandMode,
    ColorBalanceEvent,
    ColorBandFilterEvent,
    ColorBitPlaneSliceEvent,
    ColorContrastEvent,
    ColorEdgePerChannelEvent,
    # ── Color edge detection ───────────────────────────────────────────────
    ColorEdgeSobelEvent,
    ColorGammaEvent,
    ColorGaussianEvent,
    ColorGradientEdgeEvent,
    ColorGradientSharpenEvent,
    ColorHistEqChannelEvent,
    ColorHistEqHSIEvent,
    ColorHistMatchEvent,
    # ── Color enhancement ──────────────────────────────────────────────────
    ColorInvertEvent,
    ColorLaplacianSharpenEvent,
    ColorLaplacianSobelEvent,
    ColorLevelSlicingEvent,
    ColorLocalHistEqEvent,
    ColorLogEvent,
    ColorMedianEvent,
    ColorPrewittEdgeEvent,
    # ── Assignment-6 additions ─────────────────────────────────────────────
    ColorRampEvent,
    ColorRobertsEdgeEvent,
    ColorSharpenEvent,
    # ── Color spatial filters ──────────────────────────────────────────────
    ColorSmoothBoxEvent,
    ColorUnsharpEvent,
    ColorWeightedAveragingEvent,
    HSIHueRotateEvent,
    HSISaturationEvent,
    MergeChannelsEvent,
    # ── Channel split / merge ──────────────────────────────────────────────
    SplitChannelsEvent,
)

# ═══════════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════════

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colorimages")
STD = os.path.join(BASE, "standard_test_images")
MISC = os.path.join(BASE, "misc")
AERIAL = os.path.join(BASE, "aerials")
OUT_DIR = os.path.join(BASE, "output")


# ── Shorthand helpers ──────────────────────────────────────────────────────
def std(name):
    return os.path.join(STD, name)


def misc(name):
    return os.path.join(MISC, name)


def aerial(name):
    return os.path.join(AERIAL, name)


# Commonly used images
PEPPERS = std("peppers_color.tif")
LENA_512 = std("lena_color_512.tif")
LENA_256 = std("lena_color_256.tif")
MANDRILL = std("mandril_color.tif")
SPLASH = misc("4.2.01.tiff")
BABOON = misc("4.2.03.tiff")  # same subject as mandril, different file
AIRPLANE = misc("4.2.05.tiff")
SAILBOAT = misc("4.2.06.tiff")
PEPPERS_M = misc("4.2.07.tiff")  # misc version of peppers
HOUSE = misc("house.tiff")
JELLY256 = misc("4.1.07.tiff")  # 256×256 – used where speed matters
COUPLE = misc("4.1.02.tiff")  # 256×256 portrait
AERIAL_SF = aerial("2.1.03.tiff")  # San Francisco Golden Gate – 512 color
AERIAL_SD = aerial("2.1.01.tiff")  # San Diego – 512 color
AERIAL_1K = aerial("2.2.06.tiff")  # San Francisco Bay Bridge – 1024 color


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════


def out(folder: str, filename: str) -> str:
    """Build an output path, creating the folder if needed."""
    d = os.path.join(OUT_DIR, folder)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, filename)


def run(label: str, event):
    """Execute an event, print status, catch and log any error."""
    try:
        event.execute()
        print(f"  ✓  {label}")
    except Exception as e:
        print(f"  ✗  {label}  →  {e}")
        if "--debug" in sys.argv:
            traceback.print_exc()


def section(title: str):
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Output root : {OUT_DIR}")

    # ──────────────────────────────────────────────────────────────────────
    section("1 · COLOR POINT OPERATIONS")
    # ──────────────────────────────────────────────────────────────────────
    # Invert – peppers: vivid red/green/yellow make the inversion obvious
    run(
        "Invert – peppers",
        ColorInvertEvent(PEPPERS, out("01_invert", "peppers__invert.tif")),
    )

    # Invert – splash: bright saturated colors
    run(
        "Invert – splash",
        ColorInvertEvent(SPLASH, out("01_invert", "splash__invert.tif")),
    )

    # Log transform – airplane: dark background, log lifts shadow detail
    run("Log – airplane", ColorLogEvent(AIRPLANE, out("02_log", "airplane__log.tif")))

    # Log transform – aerial San Diego: brings out shadow detail in cities
    run(
        "Log – aerial SF", ColorLogEvent(AERIAL_SF, out("02_log", "aerial_sf__log.tif"))
    )

    # Gamma < 1 (brighten) – lena
    run(
        "Gamma 0.5 – lena",
        ColorGammaEvent(LENA_512, out("03_gamma", "lena__gamma_0.5.tif"), gamma=0.5),
    )

    # Gamma > 1 (darken) – lena
    run(
        "Gamma 2.0 – lena",
        ColorGammaEvent(LENA_512, out("03_gamma", "lena__gamma_2.0.tif"), gamma=2.0),
    )

    # Gamma 0.4 – airplane: reveal shadow detail in dark fuselage
    run(
        "Gamma 0.4 – airplane",
        ColorGammaEvent(
            AIRPLANE, out("03_gamma", "airplane__gamma_0.4.tif"), gamma=0.4
        ),
    )

    # Color balance – boost red channel on sailboat (warm sunset look)
    run(
        "Color balance R boost – sailboat",
        ColorBalanceEvent(
            SAILBOAT,
            out("04_balance", "sailboat__balance_r1.5.tif"),
            r_scale=1.5,
            g_scale=1.0,
            b_scale=0.7,
        ),
    )

    # Color balance – boost blue on peppers (cool look)
    run(
        "Color balance B boost – peppers",
        ColorBalanceEvent(
            PEPPERS,
            out("04_balance", "peppers__balance_b1.5.tif"),
            r_scale=0.8,
            g_scale=1.0,
            b_scale=1.5,
        ),
    )

    # Piecewise contrast – peppers: stretch mid-tones
    run(
        "Contrast stretch – peppers",
        ColorContrastEvent(
            PEPPERS,
            out("05_contrast", "peppers__contrast.tif"),
            r1=64,
            s1=0,
            r2=192,
            s2=255,
        ),
    )

    # Piecewise contrast – aerial SF: increase contrast of ground detail
    run(
        "Contrast stretch – aerial SF",
        ColorContrastEvent(
            AERIAL_SF,
            out("05_contrast", "aerial_sf__contrast.tif"),
            r1=50,
            s1=0,
            r2=200,
            s2=255,
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    section("2 · COLOR HISTOGRAM OPERATIONS")
    # ──────────────────────────────────────────────────────────────────────
    # Hist EQ per-channel – mandrill: high saturation exposes hue drift
    run(
        "Hist EQ per-channel – mandrill",
        ColorHistEqChannelEvent(
            MANDRILL, out("06_hist_eq_channel", "mandrill__histeq_ch.tif")
        ),
    )

    # Hist EQ per-channel – aerial SD: low-contrast aerial image
    run(
        "Hist EQ per-channel – aerial SD",
        ColorHistEqChannelEvent(
            AERIAL_SD, out("06_hist_eq_channel", "aerial_sd__histeq_ch.tif")
        ),
    )

    # Hist EQ HSI – mandrill: compare with per-channel (hue preserved)
    run(
        "Hist EQ HSI – mandrill",
        ColorHistEqHSIEvent(
            MANDRILL, out("07_hist_eq_hsi", "mandrill__histeq_hsi.tif")
        ),
    )

    # Hist EQ HSI – house: good for showing preserved colors in scenes
    run(
        "Hist EQ HSI – house",
        ColorHistEqHSIEvent(HOUSE, out("07_hist_eq_hsi", "house__histeq_hsi.tif")),
    )

    # ──────────────────────────────────────────────────────────────────────
    section("3 · HSI OPERATIONS")
    # ──────────────────────────────────────────────────────────────────────
    # Saturation increase – splash: already vivid, good contrast before/after
    run(
        "HSI saturation 2.0 – splash",
        HSISaturationEvent(
            SPLASH, out("08_hsi_saturation", "splash__sat2.0.tif"), scale=2.0
        ),
    )

    # Saturation decrease – splash: desaturate toward gray
    run(
        "HSI saturation 0.3 – splash",
        HSISaturationEvent(
            SPLASH, out("08_hsi_saturation", "splash__sat0.3.tif"), scale=0.3
        ),
    )

    # Saturation on lena – subtle enhancement
    run(
        "HSI saturation 1.5 – lena",
        HSISaturationEvent(
            LENA_512, out("08_hsi_saturation", "lena__sat1.5.tif"), scale=1.5
        ),
    )

    # Hue rotate 90° – peppers: red→yellow, green→cyan etc.
    run(
        "Hue rotate 90° – peppers",
        HSIHueRotateEvent(
            PEPPERS, out("09_hue_rotate", "peppers__hue90.tif"), degrees=90
        ),
    )

    # Hue rotate 180° – splash: full color inversion in hue space
    run(
        "Hue rotate 180° – splash",
        HSIHueRotateEvent(
            SPLASH, out("09_hue_rotate", "splash__hue180.tif"), degrees=180
        ),
    )

    # Hue rotate 120° – mandrill
    run(
        "Hue rotate 120° – mandrill",
        HSIHueRotateEvent(
            MANDRILL, out("09_hue_rotate", "mandrill__hue120.tif"), degrees=120
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    section("4 · COLOR SPATIAL SMOOTHING")
    # ──────────────────────────────────────────────────────────────────────
    # Box smooth k=5 – lena: classic blur benchmark
    run(
        "Box smooth k=5 – lena",
        ColorSmoothBoxEvent(LENA_512, out("10_box_smooth", "lena__box5.tif"), kernel=5),
    )

    # Box smooth k=11 – aerial SF: reduce sensor noise in aerial image
    run(
        "Box smooth k=11 – aerial SF",
        ColorSmoothBoxEvent(
            AERIAL_SF, out("10_box_smooth", "aerial_sf__box11.tif"), kernel=11
        ),
    )

    # Box smooth k=3 – peppers: light smoothing
    run(
        "Box smooth k=3 – peppers",
        ColorSmoothBoxEvent(
            PEPPERS, out("10_box_smooth", "peppers__box3.tif"), kernel=3
        ),
    )

    # Gaussian k=5 σ=1 – lena
    run(
        "Gaussian k=5 σ=1.0 – lena",
        ColorGaussianEvent(
            LENA_512,
            out("11_gaussian", "lena__gauss5_s1.tif"),
            kernel_size=5,
            sigma=1.0,
        ),
    )

    # Gaussian k=9 σ=2 – peppers: heavier blur
    run(
        "Gaussian k=9 σ=2.0 – peppers",
        ColorGaussianEvent(
            PEPPERS,
            out("11_gaussian", "peppers__gauss9_s2.tif"),
            kernel_size=9,
            sigma=2.0,
        ),
    )

    # Gaussian k=5 σ=1.5 – aerial SD
    run(
        "Gaussian k=5 σ=1.5 – aerial SD",
        ColorGaussianEvent(
            AERIAL_SD,
            out("11_gaussian", "aerial_sd__gauss5_s1.5.tif"),
            kernel_size=5,
            sigma=1.5,
        ),
    )

    # Weighted average – sailboat: gentle smoothing of water
    run(
        "Weighted average – sailboat",
        ColorWeightedAveragingEvent(
            SAILBOAT, out("12_weighted_avg", "sailboat__wavg.tif")
        ),
    )

    # Weighted average – aerial SD
    run(
        "Weighted average – aerial SD",
        ColorWeightedAveragingEvent(
            AERIAL_SD, out("12_weighted_avg", "aerial_sd__wavg.tif")
        ),
    )

    # Median k=3 – peppers: salt-and-pepper noise removal without blur
    run(
        "Median k=3 – peppers",
        ColorMedianEvent(PEPPERS, out("13_median", "peppers__median3.tif"), window=3),
    )

    # Median k=5 – lena
    run(
        "Median k=5 – lena",
        ColorMedianEvent(LENA_512, out("13_median", "lena__median5.tif"), window=5),
    )

    # Median k=3 – house: preserve sharp edges of building
    run(
        "Median k=3 – house",
        ColorMedianEvent(HOUSE, out("13_median", "house__median3.tif"), window=3),
    )

    # ──────────────────────────────────────────────────────────────────────
    section("5 · COLOR SHARPENING")
    # ──────────────────────────────────────────────────────────────────────
    # High-pass sharpen strength=1.0 – lena
    run(
        "Sharpen str=1.0 – lena",
        ColorSharpenEvent(
            LENA_512, out("14_sharpen", "lena__sharp1.0.tif"), strength=1.0
        ),
    )

    # High-pass sharpen strength=2.0 – airplane: crisp fuselage edges
    run(
        "Sharpen str=2.0 – airplane",
        ColorSharpenEvent(
            AIRPLANE, out("14_sharpen", "airplane__sharp2.0.tif"), strength=2.0
        ),
    )

    # Unsharp/Highboost A=1.5 – lena: classic unsharp mask
    run(
        "Unsharp A=1.5 – lena",
        ColorUnsharpEvent(LENA_512, out("15_unsharp", "lena__unsharp1.5.tif"), A=1.5),
    )

    # Unsharp A=2.0 – peppers: strong highboost
    run(
        "Unsharp A=2.0 – peppers",
        ColorUnsharpEvent(PEPPERS, out("15_unsharp", "peppers__unsharp2.0.tif"), A=2.0),
    )

    # Unsharp A=1.2 – sailboat: subtle detail recovery
    run(
        "Unsharp A=1.2 – sailboat",
        ColorUnsharpEvent(
            SAILBOAT, out("15_unsharp", "sailboat__unsharp1.2.tif"), A=1.2
        ),
    )

    # Laplacian sharpen 4-connected – lena
    run(
        "Laplacian sharpen 4 – lena",
        ColorLaplacianSharpenEvent(
            LENA_512, out("16_laplacian_sharpen", "lena__lap4.tif"), mode="4"
        ),
    )

    # Laplacian sharpen 8-connected – airplane: stronger sharpening
    run(
        "Laplacian sharpen 8 – airplane",
        ColorLaplacianSharpenEvent(
            AIRPLANE, out("16_laplacian_sharpen", "airplane__lap8.tif"), mode="8"
        ),
    )

    # Laplacian sharpen 4-connected – house
    run(
        "Laplacian sharpen 4 – house",
        ColorLaplacianSharpenEvent(
            HOUSE, out("16_laplacian_sharpen", "house__lap4.tif"), mode="4"
        ),
    )

    # Laplacian-Sobel combined – airplane: combines sharpening with edge mask
    run(
        "Laplacian-Sobel – airplane",
        ColorLaplacianSobelEvent(
            AIRPLANE, out("17_lap_sobel", "airplane__lapsobel.tif")
        ),
    )

    # Laplacian-Sobel – mandrill: shows combined effect on highly textured image
    run(
        "Laplacian-Sobel – mandrill",
        ColorLaplacianSobelEvent(
            MANDRILL, out("17_lap_sobel", "mandrill__lapsobel.tif")
        ),
    )

    # Gradient sharpen k=0.3 – airplane: add gradient back into original
    run(
        "Gradient sharpen k=0.3 – airplane",
        ColorGradientSharpenEvent(
            AIRPLANE, out("18_grad_sharpen", "airplane__gradsharp0.3.tif"), k=0.3
        ),
    )

    # Gradient sharpen k=0.5 – lena
    run(
        "Gradient sharpen k=0.5 – lena",
        ColorGradientSharpenEvent(
            LENA_512, out("18_grad_sharpen", "lena__gradsharp0.5.tif"), k=0.5
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    section("6 · COLOR EDGE DETECTION")
    # ──────────────────────────────────────────────────────────────────────
    # Sobel on luminance – mandrill: lots of fine edges
    run(
        "Sobel luminance – mandrill",
        ColorEdgeSobelEvent(
            MANDRILL, out("19_sobel_edge", "mandrill__sobel_lum.tif"), threshold=0
        ),
    )

    # Sobel on luminance with threshold – airplane: clean binary edge map
    run(
        "Sobel luminance thr=30 – airplane",
        ColorEdgeSobelEvent(
            AIRPLANE, out("19_sobel_edge", "airplane__sobel_thr30.tif"), threshold=30
        ),
    )

    # Sobel on luminance – aerial SF: city grid edges
    run(
        "Sobel luminance – aerial SF",
        ColorEdgeSobelEvent(
            AERIAL_SF, out("19_sobel_edge", "aerial_sf__sobel_lum.tif"), threshold=0
        ),
    )

    # Sobel per-channel – splash: color-coded edges per channel
    run(
        "Sobel per-channel – splash",
        ColorEdgePerChannelEvent(
            SPLASH, out("20_sobel_perchannel", "splash__sobel_rgb.tif")
        ),
    )

    # Sobel per-channel – mandrill: compare with luminance version
    run(
        "Sobel per-channel – mandrill",
        ColorEdgePerChannelEvent(
            MANDRILL, out("20_sobel_perchannel", "mandrill__sobel_rgb.tif")
        ),
    )

    # Roberts cross – mandrill
    run(
        "Roberts edge – mandrill",
        ColorRobertsEdgeEvent(MANDRILL, out("21_roberts", "mandrill__roberts.tif")),
    )

    # Roberts cross – airplane: clean edges on simple shape
    run(
        "Roberts edge – airplane",
        ColorRobertsEdgeEvent(AIRPLANE, out("21_roberts", "airplane__roberts.tif")),
    )

    # Roberts cross – house: architectural edges
    run(
        "Roberts edge – house",
        ColorRobertsEdgeEvent(HOUSE, out("21_roberts", "house__roberts.tif")),
    )

    # Prewitt – mandrill: compare with Roberts
    run(
        "Prewitt edge – mandrill",
        ColorPrewittEdgeEvent(MANDRILL, out("22_prewitt", "mandrill__prewitt.tif")),
    )

    # Prewitt – aerial SF: road grid
    run(
        "Prewitt edge – aerial SF",
        ColorPrewittEdgeEvent(AERIAL_SF, out("22_prewitt", "aerial_sf__prewitt.tif")),
    )

    # Prewitt – sailboat: horizon + mast edges
    run(
        "Prewitt edge – sailboat",
        ColorPrewittEdgeEvent(SAILBOAT, out("22_prewitt", "sailboat__prewitt.tif")),
    )

    # Gradient edge enhance k=0.5 – airplane
    run(
        "Gradient edge enhance k=0.5 – airplane",
        ColorGradientEdgeEvent(
            AIRPLANE, out("23_grad_edge", "airplane__gradedge0.5.tif"), k=0.5
        ),
    )

    # Gradient edge enhance k=0.3 – lena: subtle enhancement
    run(
        "Gradient edge enhance k=0.3 – lena",
        ColorGradientEdgeEvent(
            LENA_512, out("23_grad_edge", "lena__gradedge0.3.tif"), k=0.3
        ),
    )

    # Gradient edge enhance k=0.8 – aerial SD: road network
    run(
        "Gradient edge enhance k=0.8 – aerial SD",
        ColorGradientEdgeEvent(
            AERIAL_SD, out("23_grad_edge", "aerial_sd__gradedge0.8.tif"), k=0.8
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    section("7 · COLOR BAND FILTERS")
    # ──────────────────────────────────────────────────────────────────────
    # Band pass – sailboat: isolate mid-frequency water ripple texture
    run(
        "Band pass – sailboat",
        ColorBandFilterEvent(
            SAILBOAT,
            out("24_band_pass", "sailboat__bandpass.tif"),
            k1=3,
            s1=0.5,
            k2=9,
            s2=2.0,
            mode=BandMode.BANDPASS,
        ),
    )

    # Band pass – aerial SF: isolate building-block frequencies
    run(
        "Band pass – aerial SF",
        ColorBandFilterEvent(
            AERIAL_SF,
            out("24_band_pass", "aerial_sf__bandpass.tif"),
            k1=3,
            s1=0.5,
            k2=9,
            s2=2.0,
            mode=BandMode.BANDPASS,
        ),
    )

    # Band reject – peppers: suppress mid-frequency noise while keeping
    #               fine detail and coarse structure
    run(
        "Band reject – peppers",
        ColorBandFilterEvent(
            PEPPERS,
            out("25_band_reject", "peppers__bandreject.tif"),
            k1=3,
            s1=0.5,
            k2=9,
            s2=2.0,
            mode=BandMode.BANDREJECT,
        ),
    )

    # Band reject – lena: smooth skin tones, keep fine hair detail
    run(
        "Band reject – lena",
        ColorBandFilterEvent(
            LENA_512,
            out("25_band_reject", "lena__bandreject.tif"),
            k1=5,
            s1=1.0,
            k2=11,
            s2=2.5,
            mode=BandMode.BANDREJECT,
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    section("8 · COLOR POINT – INTENSITY RAMP  &  LEVEL / BIT SLICING")
    # ──────────────────────────────────────────────────────────────────────
    # Intensity ramp [64,192] – sailboat: stretch mid-range, clip extremes
    run(
        "Intensity ramp [64,192] – sailboat",
        ColorRampEvent(
            SAILBOAT, out("26_ramp", "sailboat__ramp64_192.tif"), start=64, end=192
        ),
    )

    # Intensity ramp [32,220] – aerial SD: wide ramp for aerial imagery
    run(
        "Intensity ramp [32,220] – aerial SD",
        ColorRampEvent(
            AERIAL_SD, out("26_ramp", "aerial_sd__ramp32_220.tif"), start=32, end=220
        ),
    )

    # Intensity ramp [100,200] – peppers
    run(
        "Intensity ramp [100,200] – peppers",
        ColorRampEvent(
            PEPPERS, out("26_ramp", "peppers__ramp100_200.tif"), start=100, end=200
        ),
    )

    # Level slice [200,255] val=255 nobg – peppers: isolate brightest reds
    run(
        "Level slice [200,255] nobg – peppers",
        ColorLevelSlicingEvent(
            PEPPERS,
            out("27_level_slice", "peppers__slice200_255_nobg.tif"),
            lo=200,
            hi=255,
            val=255,
            mode="nobg",
        ),
    )

    # Level slice [100,180] val=255 bg – lena: highlight mid-tones, keep rest
    run(
        "Level slice [100,180] bg – lena",
        ColorLevelSlicingEvent(
            LENA_512,
            out("27_level_slice", "lena__slice100_180_bg.tif"),
            lo=100,
            hi=180,
            val=255,
            mode="bg",
        ),
    )

    # Level slice [150,230] val=255 nobg – splash: isolate bright colors
    run(
        "Level slice [150,230] nobg – splash",
        ColorLevelSlicingEvent(
            SPLASH,
            out("27_level_slice", "splash__slice150_230_nobg.tif"),
            lo=150,
            hi=230,
            val=255,
            mode="nobg",
        ),
    )

    # Bit plane 7 (MSB) – lena: dominant structure
    run(
        "Bit plane 7 MSB nobg – lena",
        ColorBitPlaneSliceEvent(
            LENA_512, out("28_bit_slice", "lena__bit7_nobg.tif"), bit=7, mode="nobg"
        ),
    )

    # Bit plane 7 (MSB) bg – airplane: MSB with background
    run(
        "Bit plane 7 MSB bg – airplane",
        ColorBitPlaneSliceEvent(
            AIRPLANE, out("28_bit_slice", "airplane__bit7_bg.tif"), bit=7, mode="bg"
        ),
    )

    # Bit plane 6 – peppers: second most significant bit
    run(
        "Bit plane 6 nobg – peppers",
        ColorBitPlaneSliceEvent(
            PEPPERS, out("28_bit_slice", "peppers__bit6_nobg.tif"), bit=6, mode="nobg"
        ),
    )

    # Bit plane 4 nobg – lena: transitions into noise territory
    run(
        "Bit plane 4 nobg – lena",
        ColorBitPlaneSliceEvent(
            LENA_512, out("28_bit_slice", "lena__bit4_nobg.tif"), bit=4, mode="nobg"
        ),
    )

    # Bit plane 0 (LSB) nobg – lena: pure noise plane
    run(
        "Bit plane 0 LSB nobg – lena",
        ColorBitPlaneSliceEvent(
            LENA_512, out("28_bit_slice", "lena__bit0_nobg.tif"), bit=0, mode="nobg"
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    section("9 · COLOR HISTOGRAM MATCHING")
    # ──────────────────────────────────────────────────────────────────────
    # Match peppers → splash: map peppers' histogram to splash's vivid palette
    run(
        "Hist match peppers → splash",
        ColorHistMatchEvent(
            PEPPERS, SPLASH, out("29_hist_match", "peppers_matched_to_splash.tif")
        ),
    )

    # Match aerial SD → aerial SF: normalize two aerial images to same tone
    run(
        "Hist match aerial SD → aerial SF",
        ColorHistMatchEvent(
            AERIAL_SD, AERIAL_SF, out("29_hist_match", "aerial_sd_matched_to_sf.tif")
        ),
    )

    # Match lena_256 → peppers: match portrait to colorful reference
    run(
        "Hist match lena256 → peppers",
        ColorHistMatchEvent(
            LENA_256, PEPPERS, out("29_hist_match", "lena256_matched_to_peppers.tif")
        ),
    )

    # Match couple → lena: two portraits, shows subtle tone correction
    run(
        "Hist match couple → lena",
        ColorHistMatchEvent(
            COUPLE, LENA_256, out("29_hist_match", "couple_matched_to_lena256.tif")
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    section("10 · COLOR LOCAL HISTOGRAM EQ  (uses 256×256 images – slow on 512)")
    # ──────────────────────────────────────────────────────────────────────
    # Local hist EQ window=7 – jelly beans 256×256: varied local contrast
    run(
        "Local hist EQ w=7 – jelly beans (256)",
        ColorLocalHistEqEvent(
            JELLY256, out("30_local_hist_eq", "jellybeans__localheq_w7.tif"), window=7
        ),
    )

    # Local hist EQ window=9 – lena 256
    run(
        "Local hist EQ w=9 – lena 256",
        ColorLocalHistEqEvent(
            LENA_256, out("30_local_hist_eq", "lena256__localheq_w9.tif"), window=9
        ),
    )

    # Local hist EQ window=11 – couple 256
    run(
        "Local hist EQ w=11 – couple 256",
        ColorLocalHistEqEvent(
            COUPLE, out("30_local_hist_eq", "couple256__localheq_w11.tif"), window=11
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    section("11 · CHANNEL SPLIT & MERGE")
    # ──────────────────────────────────────────────────────────────────────
    split_r = out("31_split_merge", "splash__R.tif")
    split_g = out("31_split_merge", "splash__G.tif")
    split_b = out("31_split_merge", "splash__B.tif")

    run(
        "Split channels – splash", SplitChannelsEvent(SPLASH, split_r, split_g, split_b)
    )

    # Merge back in original order → should reproduce original
    run(
        "Merge R+G+B – splash (original)",
        MergeChannelsEvent(
            split_r, split_g, split_b, out("31_split_merge", "splash__merged_rgb.tif")
        ),
    )

    # Merge in swapped order (B, G, R) → color shift
    run(
        "Merge B+G+R – splash (swapped)",
        MergeChannelsEvent(
            split_b, split_g, split_r, out("31_split_merge", "splash__merged_bgr.tif")
        ),
    )

    # Split peppers and merge with swapped channels
    pep_r = out("31_split_merge", "peppers__R.tif")
    pep_g = out("31_split_merge", "peppers__G.tif")
    pep_b = out("31_split_merge", "peppers__B.tif")

    run("Split channels – peppers", SplitChannelsEvent(PEPPERS, pep_r, pep_g, pep_b))

    run(
        "Merge G+R+B – peppers (green↔red swap)",
        MergeChannelsEvent(
            pep_g, pep_r, pep_b, out("31_split_merge", "peppers__merged_grb.tif")
        ),
    )

    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  ALL FILTERS COMPLETE")
    print(f"  Results saved to: {OUT_DIR}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
