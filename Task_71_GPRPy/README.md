# Task 71: GPRPy

Ground-penetrating radar migration using GPRPy

## 📄 Paper Information

**Title**: None

**Link**: [https://doi.org/10.1190/tle39050332.1](https://doi.org/10.1190/tle39050332.1)

**GitHub Repository**: https://github.com/NSGeophysics/GPRPy

## 📊 Performance Metrics

- **PSNR**: 21.27 dB ← 🔧 修复前: 19.83 dB
- **SSIM**: 0.643

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
