# Task 125: caiman_calcium

Calcium imaging spike deconvolution using OASIS algorithm

## 📄 Paper Information

**Title**: CaImAn an open source tool for scalable calcium imaging data analysis

**Link**: [doi:10.7554/eLife.38173 (Giovannucci et al., eLife, 2019)](doi:10.7554/eLife.38173 (Giovannucci et al., eLife, 2019))

**GitHub Repository**: https://github.com/flatironinstitute/CaImAn

## 📊 Performance Metrics

- **PSNR**: 24.93 dB
- **SSIM**: N/A (1D temporal signals)

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
