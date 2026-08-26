<div align="center">

# Chromatica
![Uploading image.png…]()

### *Intelligent Image Color Palette Extraction via K-Means Clustering*

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-chromashift.onrender.com-ffbc94?style=flat-square&logo=render&logoColor=black)](https://chromashift.onrender.com/)
[![Server: Gunicorn](https://img.shields.io/badge/Server-Gunicorn-499848?style=flat-square&logo=gunicorn&logoColor=white)](https://gunicorn.org/)

### 🌐 [chromashift.onrender.com](https://chromashift.onrender.com/)

> **Chromatica** is a production-grade web application that extracts the dominant color palette from any uploaded image using unsupervised machine learning (K-Means clustering). It exposes both a rich interactive UI and a clean REST API, and is engineered for sub-50ms processing latency on commodity hardware.

</div>

---

## Table of Contents

1. [Abstract](#abstract)
2. [Background & Motivation](#background--motivation)
3. [Algorithmic Foundation](#algorithmic-foundation)
   - [Color Space Representation](#color-space-representation)
   - [K-Means Color Quantization](#k-means-color-quantization)
   - [Thumbnail Pre-processing Optimization](#thumbnail-pre-processing-optimization)
   - [HSL Conversion](#hsl-conversion)
4. [Architecture](#architecture)
   - [System Overview](#system-overview)
   - [Project Structure](#project-structure)
5. [API Reference](#api-reference)
   - [Web UI Endpoint](#web-ui-endpoint)
   - [REST API Endpoint](#rest-api-endpoint)
6. [Installation & Local Development](#installation--local-development)
7. [Deployment](#deployment)
8. [Performance](#performance)
9. [Supported Formats](#supported-formats)
10. [Design System](#design-system)
11. [Security & File Hygiene](#security--file-hygiene)
12. [Limitations & Future Work](#limitations--future-work)
13. [References](#references)
14. [License](#license)

---

## Abstract

Color extraction from images is a fundamental problem in computational photography, design tooling, and data visualization. While naive approaches based on pixel counting or histogram analysis can capture raw color frequencies, they fail to identify *perceptually meaningful* dominant colors due to the extremely high dimensionality and redundancy of natural image data.

**Chromatica** solves this by framing dominant color extraction as an **unsupervised clustering problem** in 3-dimensional RGB color space. A thumbnailed pixel array is fed into a K-Means algorithm, and the resulting cluster centroids represent the most perceptually coherent dominant colors of the image. Results are returned in RGB, HEX, and HSL formats — making them immediately usable by designers, front-end developers, and data scientists alike.

---

## Background & Motivation

The need to programmatically understand the color composition of images arises across many domains:

- **Digital design & branding** — Automatically generate color schemes from product photography or brand assets.
- **Content-based image retrieval (CBIR)** — Index images by color signature for similarity search.
- **UI/UX tooling** — Feed extracted palettes directly into design systems (e.g., Material You, Tailwind tokens).
- **Data journalism & visualization** — Semantically annotate images in media pipelines.
- **Generative AI workflows** — Seed image generation models with palette constraints derived from reference images.

Existing approaches — including Adobe Color, Coolors, and various Python libraries — either require closed proprietary backends or lack a clean, embeddable REST API. Chromatica fills this gap with a lightweight, self-hostable, open-source solution.

---

## Algorithmic Foundation

### Color Space Representation

Each pixel in an uploaded image is represented as a point in **RGB color space** — a 3-dimensional Euclidean space where each axis corresponds to one of the Red, Green, and Blue channels, each ranging from 0 to 255. An image of resolution `W × H` yields a pixel matrix of shape `(W·H, 3)`.

```
Pixel pᵢ = (Rᵢ, Gᵢ, Bᵢ)  where  Rᵢ, Gᵢ, Bᵢ ∈ [0, 255]
```

The pipeline normalizes all values to `[0, 1]` internally before clustering, ensuring uniform scale across all three dimensions.

---

### K-Means Color Quantization

Chromatica employs **K-Means clustering** (Lloyd's algorithm) to partition the pixel color space into `k` clusters, where each cluster centroid represents a dominant color.

**Objective function:**

```
minimize  Σᵢ Σₓ∈Cᵢ ‖x − μᵢ‖²
```

Where:
- `k` = number of desired palette colors (default: 10)
- `Cᵢ` = set of pixels assigned to cluster `i`
- `μᵢ` = centroid of cluster `i`

**Algorithm comparison — why K-Means?**

| Method | Pros | Cons |
|---|---|---|
| **K-Means** ✅ | Fast, well-understood, deterministic with fixed seed | Sensitive to initialization, assumes spherical clusters |
| Median Cut | No cluster count needed | Can miss subtle gradients |
| Octree Quantization | Memory-efficient | Less perceptually accurate |
| DBSCAN | Handles arbitrary shapes | Slow on large pixel sets, parameter-sensitive |
| Gaussian Mixture Models | Soft assignments, probabilistic | Much slower; overkill for this use case |

K-Means with `n_init=1` and `max_iter=100` (as configured in `main.py`) provides the best trade-off between **speed** and **perceptual accuracy** for this domain.

---

### Thumbnail Pre-processing Optimization

The most critical performance optimization in Chromatica is **image thumbnailing before clustering**. A full-resolution 12MP photograph contains ~12,000,000 pixels. Running K-Means directly on this is computationally prohibitive on free-tier hardware.

Chromatica applies `Image.thumbnail((200, 200))` from Pillow — which **preserves aspect ratio** and resizes to fit within a 200×200 bounding box — before constructing the pixel matrix. This caps the input to `≤ 40,000 pixels`, achieving:

```
Speedup ≈ N_original / N_thumbnail  =  12,000,000 / 40,000  =  300×
```

**Crucially**, this downsampling does **not** meaningfully alter the dominant color distribution. K-Means on a representative sample of a large pixel population converges to the same centroids as on the full set — a well-known property exploited by mini-batch K-Means and statistical sampling theory.

---

### HSL Conversion

In addition to RGB and HEX, Chromatica converts each extracted color to **HSL (Hue, Saturation, Lightness)** — a cylindrical color model that aligns more closely with human color perception than the Cartesian RGB model.

The conversion formula:

```
Given: R, G, B ∈ [0, 1]

L = (max(R,G,B) + min(R,G,B)) / 2

S = Δ / (1 − |2L − 1|)       where Δ = max(R,G,B) − min(R,G,B)

         ⎧ ((G−B)/Δ) mod 6    if max = R
H = 60° × ⎨ ((B−R)/Δ) + 2     if max = G
         ⎩ ((R−G)/Δ) + 4     if max = B
```

HSL values are expressed as `H°, S%, L%` — the standard notation used by CSS, design tools, and colorimetry literature.

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                          CLIENT                             │
│        (Browser / API consumer / curl / Python)             │
└──────────────────────────┬──────────────────────────────────┘
                           │  HTTP  (multipart/form-data)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   GUNICORN WSGI SERVER                      │
│              (web: gunicorn main:app)                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    FLASK APPLICATION                        │
│                                                             │
│  ┌───────────────────┐     ┌───────────────────────────┐   │
│  │  GET / POST /     │     │  POST /api/extract        │   │
│  │  Web UI route     │     │  REST JSON API route      │   │
│  └────────┬──────────┘     └─────────────┬─────────────┘   │
│           └──────────────┬───────────────┘                  │
│                          ▼                                  │
│             ┌────────────────────────┐                      │
│             │  Upload & Validation   │                      │
│             │  · secure_filename()   │                      │
│             │  · UUID namespacing    │                      │
│             │  · Extension whitelist │                      │
│             └────────────┬───────────┘                      │
│                          ▼                                  │
│             ┌────────────────────────┐                      │
│             │  Color Extraction      │                      │
│             │  Pipeline              │                      │
│             │  1. Pillow open+RGB    │                      │
│             │  2. thumbnail(200,200) │                      │
│             │  3. NumPy reshape(N,3) │                      │
│             │  4. KMeans(k=10)       │                      │
│             │  5. RGB→HEX/HSL        │                      │
│             └────────────┬───────────┘                      │
│                          ▼                                  │
│             ┌────────────────────────┐                      │
│             │  TTL Auto-Cleanup      │                      │
│             │  (files > 1hr purged)  │                      │
│             └────────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   static/uploads/      │
              │   (ephemeral storage,  │
              │    TTL = 1 hour)       │
              └────────────────────────┘
```

---

### Project Structure

```
Chromatica/
├── main.py                        # Flask application entry point
│   ├── cleanup_old_uploads()      #   TTL-based file hygiene (1hr)
│   ├── allowed_file()             #   Extension whitelist validator
│   ├── rgb_to_hsl()               #   Color space conversion utility
│   ├── extract_colors_from_image()#   Core K-Means extraction pipeline
│   ├── GET/POST /                 #   Web UI route
│   └── POST /api/extract          #   REST API route
│
├── templates/
│   └── index.html                 # Jinja2 SPA template
│                                  # (TailwindCSS · Material Symbols ·
│                                  #  Hanken Grotesk · JetBrains Mono)
│
├── static/
│   ├── logo.png                   # Chromatica brand logo
│   └── uploads/                   # Ephemeral image upload storage
│
├── requirements.txt               # Python dependencies
├── Procfile                       # Gunicorn production server config
├── .gitignore                     # Git exclusions
├── LICENSE                        # MIT License
└── README.md                      # This document
```

---

## API Reference

### Web UI Endpoint

**`GET /`**

Returns the interactive Chromatica web interface.

---

**`POST /`** — `Content-Type: multipart/form-data`

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `image` | File | ✅ | Image file to extract colors from |

**Response:** Renders `index.html` with `image_url` and `color_list` injected into the template context.

---

### REST API Endpoint

**`POST /api/extract`** — `Content-Type: multipart/form-data`

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `image` | File | ✅ | Image file to extract colors from |

**Success Response — `200 OK`:**

```json
{
  "success": true,
  "image_url": "/static/uploads/a3f9b21c4d1e_photo.jpg",
  "color_list": [
    {
      "rgb": [34, 40, 49],
      "rgb_str": "34, 40, 49",
      "hex": "#222831",
      "hsl": "216°, 18%, 16%",
      "count": 8241
    },
    {
      "rgb": [255, 188, 148],
      "rgb_str": "255, 188, 148",
      "hex": "#ffbc94",
      "hsl": "25°, 100%, 79%",
      "count": 4103
    }
  ]
}
```

> `count` represents the number of thumbnail pixels assigned to that cluster — a proxy for that color's dominance share.

**Error Responses:**

| Status | Condition | Response Body |
|:------:|-----------|---------------|
| `400` | No image field in request | `{"error": "No image file provided"}` |
| `400` | Empty filename or invalid extension | `{"error": "Invalid file type or empty file"}` |
| `500` | Image processing failure | `{"error": "Failed to process image: <detail>"}` |

---

**Example — cURL:**

```bash
curl -X POST https://chromashift.onrender.com/api/extract \
  -F "image=@/path/to/photo.jpg" \
  | python -m json.tool
```

**Example — Python `requests`:**

```python
import requests

with open("photo.jpg", "rb") as f:
    response = requests.post(
        "https://chromashift.onrender.com/api/extract",
        files={"image": f}
    )

data = response.json()
for color in data["color_list"]:
    print(f"{color['hex']}  HSL({color['hsl']})  — {color['count']} pixels")
```

---

## Installation & Local Development

### Prerequisites

- Python ≥ 3.9
- `pip`

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/amastronaut5/Chromatica.git
cd Chromatica

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the development server
python main.py
```

The application will be available at **`http://localhost:5000`**.

> ⚠️ The Flask built-in development server is not suitable for production. Use Gunicorn for any public-facing deployment.

---

## Deployment

Chromatica ships with a `Procfile` pre-configured for **Render**, **Railway**, **Heroku**, and any platform supporting the Procfile standard.

```
web: gunicorn main:app
```

### Deploy to Render (Recommended — Free Tier)

1. Push this repository to GitHub.
2. Create a new **Web Service** on [Render](https://render.com).
3. Connect your GitHub repository.
4. Configure the service:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn main:app`
   - **Runtime:** Python 3
5. Click **Deploy**. ✅

### Environment Variables

| Variable | Default | Description |
|----------|:-------:|-------------|
| `PORT` | `5000` | Port the web server binds to. Set automatically by most PaaS platforms. |

---

## Performance

Chromatica is engineered for **low-latency, resource-efficient processing** suitable for free-tier cloud instances (e.g., Render 0.1 vCPU / 512 MB RAM).

| Metric | Value |
|--------|-------|
| Thumbnail resolution cap | 200 × 200 px |
| Max pixels fed to K-Means | ~40,000 |
| K-Means `max_iter` | 100 |
| K-Means `n_init` | 1 (single initialization) |
| `random_state` | 42 (fully deterministic) |
| Target processing latency | **< 50 ms** (post-thumbnail) |
| Max upload file size | **25 MB** |
| Uploaded file TTL | **1 hour** (auto-purged) |

The 300× reduction in pixel count via thumbnailing is the single most impactful optimization, ensuring predictable sub-50ms response times regardless of the source image's megapixel count.

---

## Supported Formats

All common raster image formats are accepted:

| Extension(s) | Format |
|---|---|
| `.jpg` · `.jpeg` · `.jfif` · `.pjpeg` · `.pjp` | JPEG |
| `.png` | PNG |
| `.gif` | GIF (first frame used) |
| `.bmp` | Windows Bitmap |
| `.webp` | WebP |
| `.heic` · `.heif` | Apple High-Efficiency Image (via Pillow plugin) |

> All images are normalized to 3-channel RGB via `Image.convert('RGB')` before processing — ensuring consistent pixel arrays regardless of source color mode (RGBA, L, P, CMYK, etc.).

---

## Design System

The Chromatica UI implements **Google Material You (Material Design 3)** color tokens, with a dark-mode-first layout rendered via TailwindCSS. The typographic stack uses:

| Typeface | Usage |
|---|---|
| **Hanken Grotesk** | Primary UI — body text, headings, labels |
| **JetBrains Mono** | Monospace — color codes (HEX, RGB, HSL values) |
| **Material Symbols Outlined** | Icon system (variable font, weight 100–700) |

The dark theme is seeded from a warm neutral-orange primary color, generating the following Material You tonal roles:

| Token | Hex | Role |
|---|---|---|
| `primary` | `#ffbc94` | Primary interactive elements |
| `primary-container` | `#f39a60` | High-emphasis containers |
| `surface` | `#121416` | Page background |
| `surface-container` | `#1e2022` | Card / panel surfaces |
| `on-surface` | `#e2e2e5` | Body text |
| `outline` | `#a18d81` | Borders and dividers |

---

## Security & File Hygiene

| Mechanism | Implementation | Purpose |
|---|---|---|
| **Secure filenames** | `werkzeug.utils.secure_filename()` | Prevents path traversal attacks |
| **UUID namespacing** | 12-char UUID hex prefix on each upload | Prevents filename collisions & direct enumeration |
| **Extension whitelist** | Hardcoded `ALLOWED_EXTENSIONS` set | Rejects non-image uploads at the validation layer |
| **Upload size cap** | `MAX_CONTENT_LENGTH = 25 MB` | Prevents denial-of-service via large payloads |
| **TTL auto-cleanup** | `cleanup_old_uploads(max_age=3600s)` | Prevents unbounded disk accumulation on free-tier storage |

---

## Limitations & Future Work

### Known Limitations

- **Fixed palette size:** The palette is hardcoded to `k=10` colors. `n_colors` is not yet user-configurable.
- **Single-image requests:** The API processes one image per call; no batch endpoint exists.
- **Ephemeral storage:** Uploaded files have a 1-hour TTL; no persistent gallery or user session is maintained.

### Roadmap

- [ ] Expose `n_colors` as a configurable query parameter (range: 1–20)
- [ ] Add **WCAG contrast ratio** analysis between all palette color pairs
- [ ] Implement **color naming** via nearest-neighbor lookup in a named color database (xkcd, CSS4, Pantone)
- [ ] **Palette export** in CSS custom properties, Tailwind config JSON, Figma tokens, and Adobe ASE formats
- [ ] **URL-based image input** — accept a remote image URL in addition to file uploads
- [ ] Evaluate **perceptually uniform color spaces** (CIELAB / OKLab) as the clustering domain for improved perceptual accuracy
- [ ] Benchmark **Mini-Batch K-Means** as a drop-in replacement for even faster processing without thumbnailing
- [ ] Add a **color distribution visualization** (pie/donut chart) rendered alongside the palette

---

## References

1. MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations.* Proceedings of the 5th Berkeley Symposium on Mathematical Statistics and Probability, 1, 281–297.

2. Lloyd, S. P. (1982). *Least squares quantization in PCM.* IEEE Transactions on Information Theory, 28(2), 129–137. https://doi.org/10.1109/TIT.1982.1056489

3. Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, 2825–2830.

4. Clark, A. (2015). *Pillow (PIL Fork) Documentation.* https://pillow.readthedocs.io/

5. Google. (2021). *Material You: The next chapter in Material Design.* https://material.io/blog/announcing-material-you

6. Lindbloom, B. (2017). *RGB to HSL and HSL to RGB Conversion.* http://www.brucelindbloom.com/

---

## License

```
MIT License — Copyright (c) 2025 amastronaut5
```

See [LICENSE](LICENSE) for full terms.

---

<div align="center">

Made with ♥ by **amastronaut5**

*If Chromatica was useful to you, consider leaving a ⭐ on GitHub.*

</div>
