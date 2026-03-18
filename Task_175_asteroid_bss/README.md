# Task 175: asteroid_bss

Blind source separation using FastICA

## 📄 Paper Information

**Title**: None

**Link**: [None](None)

**GitHub Repository**: https://github.com/asteroid-team/asteroid

## 📊 Performance Metrics

- **PSNR**: 31.03 dB ← 🔧 修复前: 16.90 dB
- **SSIM**: N/A (1D 音频信号)

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
