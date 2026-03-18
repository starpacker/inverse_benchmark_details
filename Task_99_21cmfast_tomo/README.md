# Task 99: 21cmfast_tomo

21cm tomography foreground removal using polynomial/PCA subtraction

## 📄 Paper Information

**Title**: 21cmFAST v3: A Python-integrated C code for generating 3D realizations of the cosmic 21cm signal

**Link**: [doi:10.1093/mnras/staa3408 (Murray et al., MNRAS, 2020)](doi:10.1093/mnras/staa3408 (Murray et al., MNRAS, 2020))

**GitHub Repository**: https://github.com/21cmfast/21cmFAST

## 📊 Performance Metrics

- **PSNR**: 24.12 dB (polynomial foreground removal)

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
