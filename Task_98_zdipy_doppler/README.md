# Task 98: zdipy_doppler

Zodiacal light Doppler imaging using least-squares inversion

## 📄 Paper Information

**Title**: The evolution of surface magnetic fields in young solar-type stars II: the early main sequence (250-650 Myr)

**Link**: [https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.4956F/abstract](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.4956F/abstract)

**GitHub Repository**: https://github.com/folsomcp/ZDIpy

## 📊 Performance Metrics

- **PSNR**: 20.66 dB ← 🔧 修复前: 8.89 dB
- **SSIM**: 0.8690

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

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

Data files are hosted on Hugging Face:

```bash
# TODO: Add Hugging Face dataset link
```

### 3. Run Reconstruction

```bash
cd src
python main.py
```

### 4. Explore Tutorial

Open the Jupyter notebook for an interactive tutorial:

```bash
jupyter notebook notebook/visualization.ipynb
```

## 🧪 Testing

Run unit tests to validate the implementation:

```bash
# Run all tests
python -m pytest test/

# Run specific test
python test/test_function_name.py
```

## 📖 Algorithm Overview

This task implements an inverse problem solver for {task_name}.

**Key Steps**:
1. Load and preprocess input data
2. Apply the inverse problem algorithm
3. Post-process and evaluate results

For detailed implementation, see `src/main.py` and the tutorial notebook.

## 📚 Citation

If you use this code, please cite the original paper:

```bibtex
@article{{task{task_id:02d},
  title={{{paper_title}}},
  url={{{paper_link}}}
}}
```

## 📝 License

This implementation follows the license of the original repository: {github_repo}

## 🤝 Contributing

This is part of the Paper2Executable benchmark. For issues or contributions, please visit the main repository.

---

**Part of Paper2Executable Benchmark** | Task {task_id:02d} of 200
