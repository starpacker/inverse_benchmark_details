# Task 16: lensless_dl

Lensless imaging reconstruction using deep learning (U-Net architecture)

## 📄 Paper Information

**Title**: Towards Robust and Generalizable Lensless Imaging with Modular Learned Reconstruction

**Link**: [doi:10.1109/TCI.2025.3539448](doi:10.1109/TCI.2025.3539448)

**GitHub Repository**: https://github.com/LCAV/LenslessPiCam

## 📊 Performance Metrics

- **PSNR**: 38.51 dB
- **SSIM**: 0.9905

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
