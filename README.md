# Machine Learning Fundamentals: From Scratch Implementation

A comprehensive implementation of core machine learning algorithms and numerical methods built from first principles using Python. This project demonstrates deep understanding of the mathematical foundations behind ML algorithms, going beyond library usage to implement optimization, dimensionality reduction, and regression techniques.

## Highlights

- **SVD-based Image Compression**: Achieved 68% compression ratio with <1% reconstruction error
- **MNIST Digit Classification**: Implemented multiple approaches (SVD, PCA, Linear Regression)
- **Optimization Algorithms**: Gradient Descent, SGD, and Backtracking Line Search with convergence analysis
- **Numerical Stability Analysis**: Condition number analysis, ill-conditioned matrix handling

## Implemented Algorithms

### Optimization
| Algorithm | Description | Key Features |
|-----------|-------------|--------------|
| **Gradient Descent** | First-order optimization | Fixed/adaptive learning rates, convergence criteria |
| **Stochastic Gradient Descent** | Mini-batch optimization | Epoch-based training, data shuffling |
| **Backtracking Line Search** | Adaptive step size | Armijo condition, robust convergence |

### Dimensionality Reduction
| Algorithm | Description | Applications |
|-----------|-------------|--------------|
| **Singular Value Decomposition (SVD)** | Matrix factorization | Image compression, feature extraction |
| **Principal Component Analysis (PCA)** | Linear projection | Visualization, clustering, classification |

### Regression & Classification
| Technique | Description | Implementation |
|-----------|-------------|----------------|
| **Linear Regression** | Normal equations | Cholesky decomposition for numerical stability |
| **Polynomial Regression** | Vandermonde basis | Ridge regularization (L2) |
| **Distance-based Classification** | SVD subspace projection | Multi-class digit recognition |
| **Centroid Classification** | PCA + nearest centroid | MNIST clustering |

## Project Structure

```
├── Homework Assignments/
│   ├── HW1.ipynb          # Linear systems & SVD fundamentals
│   ├── HW2_PCA.ipynb      # PCA for digit classification
│   ├── HW3.ipynb          # Gradient Descent with backtracking
│   ├── HW3_SGD.ipynb      # Stochastic Gradient Descent
│   └── HW4.ipynb          # Regression: SGD vs GD vs Normal Equations
│
├── Algorithm Notebooks/
│   ├── svd.ipynb          # SVD implementation & image compression
│   ├── pca.ipynb          # PCA from scratch
│   ├── gd.ipynb           # Gradient Descent experiments
│   └── linear_regression.ipynb  # Regression techniques
│
└── Datasets/
    ├── train.csv          # MNIST (42,000 samples, 784 features)
    └── poly_regression_*.csv  # Synthetic regression data
```

## Key Results

### Image Compression with SVD
Compressed grayscale images using truncated SVD:
- **k=80 singular values** (out of 512): 68% compression, ~1% error
- Demonstrated trade-off between compression ratio and reconstruction quality

### MNIST Classification Performance
| Method | Task | Accuracy |
|--------|------|----------|
| SVD Projection | Binary (3 vs 4) | 84% |
| SVD Projection | 3-class (6, 9, 7) | 96-97% |
| PCA + Centroid | Multi-class | 36-44% |

### Optimization Convergence
- **Backtracking Line Search**: Robust convergence across different problem types
- **Learning Rate Analysis**: Compared α = {0.005, 0.01, 0.05} convergence behavior
- **SGD vs GD**: SGD converges faster on large datasets with proper batch sizing

## Technical Skills Demonstrated

**Mathematical Foundations**
- Linear Algebra: Matrix factorizations, eigenvalue decomposition, condition numbers
- Optimization: Convex optimization, convergence analysis, regularization theory
- Statistics: MAP estimation, loss functions, bias-variance tradeoff

**Implementation**
- NumPy for vectorized numerical computing
- Matplotlib for algorithm visualization
- Pandas for data manipulation
- SciPy for advanced linear algebra (Cholesky, Hilbert matrices)

**ML Engineering**
- Train/test split methodology
- Hyperparameter tuning (learning rates, regularization λ)
- Numerical stability considerations
- Algorithm complexity analysis

## Sample Visualizations

The notebooks include visualizations for:
- SVD image reconstruction at various ranks
- PCA 2D projections of MNIST digits
- Gradient descent convergence curves
- Polynomial regression fits with different regularization

## How to Run

```bash
# Clone the repository
git clone https://github.com/[your-username]/ml-fundamentals.git

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install numpy pandas matplotlib scipy scikit-image jupyter

# Launch Jupyter
jupyter notebook
```

## What I Learned

1. **Why understanding fundamentals matters**: Implementing algorithms from scratch reveals numerical stability issues that high-level APIs hide
2. **Optimization is not one-size-fits-all**: Different problems require different approaches (SGD for large data, closed-form for small)
3. **Regularization prevents overfitting**: Demonstrated empirically with polynomial regression
4. **Dimensionality reduction enables visualization**: PCA makes high-dimensional MNIST data interpretable

## Future Improvements

- [ ] Implement momentum and Adam optimizer variants
- [ ] Add cross-validation for hyperparameter selection
- [ ] Extend to neural network backpropagation
- [ ] Add GPU acceleration with CuPy

## Technologies

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data_Analysis-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter)

---

*This project was completed as part of a Statistical Mathematical Models course, demonstrating proficiency in implementing ML algorithms from mathematical foundations.*
