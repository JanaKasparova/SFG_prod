# Scientific Data Processing Pipeline

This repository contains a **compact, research-friendly scientific data processing pipeline** designed for working with large datasets, including **FITS**, **HDF5**, and **image-based flat-field calibration data**. The project emphasizes clarity, minimal boilerplate, and ease of experimentation.

The architecture intentionally avoids over-engineering while maintaining clean separation of concerns.

---

## High-level workflow

The pipeline follows a linear, readable execution flow:

```
Load data  →  Calibrate / preprocess  →  Analyze  →  Visualize
```

Each stage is implemented in a dedicated module and coordinated by a small master class.

---

## Directory structure

```
project/
│
├── pipeline.py        # Master orchestration class
├── io.py              # FITS / HDF5 / flat-field loading
├── processing.py      # Calibration and preprocessing
├── analysis.py        # Scientific analysis
├── plotting.py        # Visualization and figures
├── main.py            # Entry point
└── README.md          # This file
```

This layout is intentionally shallow and easy to navigate.

---

## File-by-file responsibilities

### `pipeline.py`

**Purpose:** Orchestrates the entire workflow.

Responsibilities:
- Stores paths, configuration, and shared state
- Controls execution order
- Coordinates data flow between modules

Non-responsibilities:
- No numerical analysis
- No calibration math
- No plotting logic

The `Pipeline` class exposes a single public method:

```python
Pipeline.run()
```

which executes the full workflow.

---

### `io.py`

**Purpose:** All file input/output operations.

Responsibilities:
- Discover FITS files in a directory
- Load FITS data
- Load HDF5 data
- Load flat-field image data

Design rules:
- No data modification
- No plotting
- No scientific interpretation

Functions typically return NumPy arrays, dictionaries, or lightweight containers.

---

### `processing.py`

**Purpose:** Data calibration and preprocessing.

Responsibilities:
- Flat-field correction
- Instrumental corrections
- Normalization
- Masking invalid or saturated data

Input:
- Raw data from `io.py`

Output:
- Calibrated data suitable for analysis

This module contains all transformations that *change* the data values.

---

### `analysis.py`

**Purpose:** Extract scientific information from calibrated data.

Responsibilities:
- Statistical analysis
- Temporal analysis
- Spatial or image-based analysis
- Spectral or frequency-domain analysis

Input:
- Calibrated data

Output:
- Results (arrays, scalars, tables, metadata)

This module should not perform any plotting or file I/O.

---

### `plotting.py`

**Purpose:** Visualization and diagnostics.

Responsibilities:
- Time series plots
- Image plots
- Spectral plots
- Diagnostic figures
- Animations (if needed)

Rules:
- No numerical analysis
- No data loading
- Only consume prepared data and results

All Matplotlib-related code lives here.

---

### `main.py`

**Purpose:** Entry point for running the pipeline.

Typical responsibilities:
- Parse command-line arguments (if needed)
- Instantiate the `Pipeline`
- Call `run()`

Example:

```python
pipeline = Pipeline(fits_dir, hdf_dir, flat_dir, output_dir, logger)
pipeline.run()
```

---

## Design philosophy

- **Simple beats clever**
- **Readable beats extensible** (until proven otherwise)
- **Functions before classes** unless state is required
- **One responsibility per file**

This structure is intentionally optimized for:
- Single-researcher projects
- Iterative experimentation
- Long-term maintainability without framework overhead

---

## When to refactor

Consider further modularization only when:
- A file exceeds ~800 lines
- You need parallel or distributed processing
- Components must be reused across projects

Until then, this structure is sufficient and robust.

---

## Summary

This project layout provides:
- A clear mental model
- Minimal boilerplate
- A stable foundation for scientific analysis

The `Pipeline` class reads like a methods section in a paper, and each module has a single, well-defined role.

