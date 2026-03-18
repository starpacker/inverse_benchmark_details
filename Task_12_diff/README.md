# Task 12: diff

Differentiable ray tracing for optical system optimization

## 📄 Paper Information

**Title**: Towards self-calibrated lens metrology by differentiable refractive deflectometry

**Link**: [https://doi.org/10.1364/oe.433237](https://doi.org/10.1364/oe.433237)

**GitHub Repository**: https://github.com/vccimaging/DiffDeflectometry

## 📊 Performance Metrics

- **PSNR**: N/A (differentiable optics — optimization convergence)

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
