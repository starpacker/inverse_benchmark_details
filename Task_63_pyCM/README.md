# Task 63: pyCM

Confocal microscopy deconvolution (pyCM)

## 📄 Paper Information

**Title**: None

**Link**: [https://doi.org/10.1088/1742-6596/430/1/012007](https://doi.org/10.1088/1742-6596/430/1/012007)

**GitHub Repository**: https://github.com/xraypy/xraylarch

## 📊 Performance Metrics

- **PSNR**: 36.15 dB ← 🔧 修复前: 15.72 dB
- **SSIM**: 0.9318

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
