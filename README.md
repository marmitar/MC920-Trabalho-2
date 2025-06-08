# Image Processing (MC920) - Dithering

- [Requirements](papers/enunciado.pdf)
- [Report](papers/entrega.pdf)

This project implements error-diffusion dithering for halftoning grayscale images using a straightforward Python CLI. Each pixel is thresholded to black or white, and the resulting quantization error is distributed to its neighbors.

Supported kernels:

- Floyd-Steinberg
- Stevenson-Arce
- Burkes
- Sierra
- Stucki
- Jarvis-Judice-Ninke

Scan patterns:

- left-to-right
- serpentine
- spiral
- Hilbert

The implementation relies on NumPy, with optional Numba acceleration for faster loops.

![Stevenson and Arci dithering algorithm applied to Mona Lisa](resultados/dists/monalisa_stevenson.png "Stevenson & Arci dithering")
