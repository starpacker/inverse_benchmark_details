# Task 66: SPICY

Pressure field reconstruction from PIV using SPICY

## 📄 Paper Information

**Title**: SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications

**Link**: [https://doi.org/10.1016/j.cageo.2015.09.015](https://doi.org/10.1016/j.cageo.2015.09.015)

**GitHub Repository**: https://github.com/simpeg/simpeg

## 📊 Performance Metrics

- **PSNR**: 35.54 dB
- **SSIM**: 0.9457

## 📁 Directory Structure

```
.
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── src/                      # Source code
│   ├── main.py              # Main reconstruction code
│   └── ...                  # Additional utilities
├── notebook/                 # Jupyter notebooks
│   └── visualization.ipynb  # Tutorial and visualization
├── data/                     # Data files (see Hugging Face)
│   ├── input.*              # Input data
│   ├── gt_output.*          # Ground truth output
│   └── recon_output.*       # Reconstruction output
├── test/                     # Test files
│   ├── test_*.py            # Unit tests
│   └── test_data/           # Test data
├── docs/                     # Documentation
│   └── qa.json              # Q&A documentation
└── assets/                   # Visualization results
    └── vis_result.png       # Result visualization

```
