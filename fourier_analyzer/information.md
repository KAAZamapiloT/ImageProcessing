# Fourier Analyzer Guide

## Goal
This project is a PySide6 Fourier-domain image analyzer focused on learning and experimentation:
- view Fourier spectrum and reconstruction side by side
- apply preset filters and restoration operations
- edit frequency components interactively
- test scripted combinations of existing filters

## Quick Start
1. Install dependencies:
   - `pip install pyside6 numpy opencv-python`
2. Run:
   - `python main.py`
3. In the app:
   - `File > Open Image`
   - choose a preset or tool
   - inspect result in the Reconstructed panel

## UI Layout
- `Menu`: `File | View | Tools | Presets | Help`
- `Central splitter`: `Original | Fourier Spectrum | Reconstructed`
- `Dock panels`:
  - `Preset Filters Panel`
  - `Fourier Tools Panel`
  - `Inspector Panel`
  - `CLI Terminal` (hidden by default, enable in `View`)

## Recommended Usage Flow
1. Open an image.
2. Start with `Preset Filters Panel`:
   - pick filter family and response
   - tune controls (`Cutoff`, `Order`, etc.)
   - optional `Live Preview`
   - click `Apply Preset`
3. Use Fourier tools from toolbar for local edits.
4. Hover the spectrum to inspect `(u, v)`, magnitude, phase, and radial distance.
5. Use `Undo/Redo` while experimenting.
6. Save output from `File > Save Output`.

## Preset Filters Panel
- Supports `Gaussian`, `Butterworth`, `Ideal`, `Wiener`.
- Controls are dynamic by family:
  - Gaussian/Ideal: `Cutoff`
  - Butterworth: `Cutoff`, `Order`
  - Wiener: `K`, `Blur Sigma`, `Kernel Size`
- `Live Preview` shows non-destructive preview before final apply.

## Fourier Tools (Toolbar)
Available tools:
- `Circle Zero Tool`
- `Circle Boost Tool`
- `Rectangular Mask Tool`
- `Line Suppression Tool`
- `DC Removal Tool`
- `High-Frequency Suppression Tool`
- `Low-Frequency Suppression Tool`
- `Phase Randomizer Tool`

Behavior:
- tools act at clicked spectrum location (or selected cursor position)
- conjugate symmetry is enforced automatically
- operations are vectorized using numpy masks for performance

## Presets Menu (Educational Actions)
- Remove DC Component
- Show Phase Only
- Show Magnitude Only
- Randomize Phase
- Swap Magnitude (requires secondary image)
- Apply Motion Blur
- Restore with Wiener

## CLI Terminal
- Optional panel (hidden by default)
- toggle from `View > CLI Terminal`
- useful commands:
  - `undo`, `redo`, `reset`, `reconstruct`
  - filter commands supported by `cli_parser.py`

## Script/Filter Composition
Script-style combinations are supported via helper functions that produce masks.

Common helpers:
- `ideal_lp`, `ideal_hp`
- `butter_lp`, `butter_hp`
- `gaussian_lp`, `gaussian_hp`
- `ideal_br`, `ideal_bp`
- `butter_br`, `butter_bp`
- `gaussian_br`, `gaussian_bp`
- `notch_reject`, `notch_pass`
- `notch_reject_butter`, `notch_pass_butter`
- `laplacian`, `hfe`, `homomorphic`
- `clip`, `ones`, `zeros`

Common parameters:
- `d0`, `w`, `n`, `u0`, `v0`, `alpha`, `beta`, `gamma_h`, `gamma_l`, `c`

Example:
- `clip(0.6 * gaussian_lp(d0) + 0.4 * notch_reject(u0, v0, 10), 0, 1)`

## Code Structure
- `main.py`: application entrypoint
- `gui.py`: UI architecture and interaction wiring
- `fft_engine.py`: FFT state, tools, history, reconstruction
- `filters.py`: filter generation and degradation/restoration kernels
- `cli_parser.py`: text command to filter mapping
- `utils.py`: image normalization / Qt conversion helpers

## Development Guidelines
- Keep FFT mutations in `FFTEngine` (not directly in GUI widgets).
- Push history before every FFT modification (`undo/redo` integrity).
- Prefer vectorized masks over pixel loops.
- Preserve conjugate symmetry for real-image reconstruction consistency.
- Reuse filter builders from `filters.py` instead of duplicating formulas.

## Troubleshooting
- `No image loaded`: open an image before tools/filters.
- Shape mismatch with secondary image: load same-sized image or let app resize.
- Very noisy Wiener output: increase `K` or blur kernel size.
- No visible effect from edits: increase radius/cutoff or verify active tool.
