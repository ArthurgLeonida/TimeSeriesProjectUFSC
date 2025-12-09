# Time Series Forecasting: Comparative Study of Deep Learning Architectures

**Author:** Arthur Gislon Leonida  
**Institution:** Universidade Federal de Santa Catarina (UFSC)

---

## 📋 Project Overview

This repository contains a comprehensive comparative analysis of **Encoder-Decoder (Seq2Seq) architectures** for time series forecasting, evaluating their performance on both **short-term** and **long-term** prediction tasks using two distinct datasets.

The project explores the strengths and limitations of:
- **LSTM Seq2Seq** (Baseline Recurrent Neural Network)
- **Transformer with Multi-Head Attention** (Global Dependency Learning)
- **Transformer with Fourier Layer** (Frequency-Domain Attention)
- **Transformer with ProbSparse Attention** (Efficient Sparse Attention)

---

## 📊 Datasets

### 1. **Short-Term Forecasting: Electricity Load Diagrams**
- **Source:** UCI Machine Learning Repository
- **Type:** Univariate Time Series
- **Target:** Electricity consumption (kW)
- **Frequency:** 15-minute intervals
- **Task:** Predict next 24 hours of electricity consumption
- **Key Challenge:** Strong daily/weekly periodicity with high variability

### 2. **Long-Term Forecasting: ETTh1 (Electricity Transformer Temperature)**
- **Source:** Standard LSTF Benchmark Dataset
- **Type:** Multivariate Time Series
- **Target:** Oil Temperature (OT)
- **Features:** 6 electrical load indicators (HUFL, HULL, MUFL, MULL, LUFL, LULL)
- **Frequency:** Hourly measurements
- **Task:** Predict transformer temperature over extended horizons
- **Key Challenge:** Causal relationship between electrical load and temperature (physics-driven)

---

## 🗂️ Repository Structure

```
TimeSeriesProjectUFSC/
│
├── short_term.ipynb          # Short-term forecasting (Electricity Load)
├── long_term.ipynb           # Long-term forecasting (ETTh1)
├── data/                     # Dataset files
└── README.md                 # This file
```

---

## 🔬 Methodology

### Preprocessing Pipeline
1. **STL Decomposition:** Separates time series into Trend, Seasonal, and Residual components
2. **Detrending/Residual Learning:** Models focus on learning the non-trivial patterns
3. **Normalization:** StandardScaler (residuals) / MinMaxScaler (features)
4. **Data Leakage Prevention:** Fit scalers/decompositions only on training data

### Model Architectures

#### **Model A: LSTM Seq2Seq (Baseline)**
- **Encoder:** Bidirectional LSTM captures temporal dependencies
- **Decoder:** Autoregressive LSTM generates future predictions
- **Strength:** Excellent for short sequences with strong local patterns

#### **Model B: Transformer Multi-Head Attention**
- **Mechanism:** Self-attention captures global dependencies
- **Strength:** Superior for long-range dependencies
- **Trade-off:** Requires more data; computationally expensive

#### **Model C: Transformer with Fourier Layer**
- **Innovation:** Replaces attention with FFT for frequency-domain learning
- **Strength:** Highly efficient for periodic/seasonal data
- **Trade-off:** Less flexible for irregular patterns

#### **Model D: Transformer ProbSparse Attention**
- **Innovation:** Sparse attention reduces complexity from O(L²) to O(L log L)
- **Strength:** Scalable to very long sequences (>720 hours)
- **Trade-off:** Slight accuracy drop on short sequences

---

## 🎯 Key Findings

### Short-Term Forecasting (Electricity Load)
- **Winner:** LSTM Seq2Seq
- **Reason:** Short horizon (24h) favors local temporal patterns; LSTM's inductive bias is perfectly aligned
- **Insight:** Transformers underperform when relevant context is recent (no long-term dependency advantage)

### Long-Term Forecasting (ETTh1)
- **Strategy:** Multivariate Residual-Only Learning
- **Innovation:** Decoder receives **future known features** (electrical load forecast)
- **Result:** ~15% improvement over naive baseline (Trend+Seasonal only)
- **Insight:** Causal features (Load → Temperature) are critical for physical systems

---

## 📈 Performance Metrics

### Short-Term (Electricity Load)
| Model | MSE | MAE | Training Time |
|-------|-----|-----|---------------|
| LSTM | **Best** | **Best** | Moderate |
| Transformer (MHA) | Good | Good | High |
| Transformer (Fourier) | Good | Good | Moderate |
| Transformer (ProbSparse) | Good | Good | Low |

### Long-Term (ETTh1)
| Model | MSE | Correlation | Baseline Beat |
|-------|-----|-------------|---------------|
| LSTM | 0.75916 | 0.9546 | +6.00% |
| Transformer (MHA) | 0.83414 | 0.9494 | -3.29% |
| Transformer (Fourier) | 0.73998 | 0.9548 | +8.37% |
| Transformer (ProbSparse) | 0.68330 | 0.9570 | +15.39% |

---

## 🚀 How to Run

### Prerequisites
```bash
pip install torch numpy pandas matplotlib scikit-learn statsmodels
```

### Execution
1. **Short-Term Analysis:**
   ```bash
   jupyter notebook short_term.ipynb
   ```
   - Run all cells sequentially
   - Models will train and save checkpoints to `best_model_*.pth`

2. **Long-Term Analysis:**
   ```bash
   jupyter notebook long_term.ipynb
   ```
   - Ensure ETTh1 dataset is in `data/` directory
   - Models will save to `best_*.pth`

---

## 📚 Academic Insights

### When to Use Each Architecture:

1. **LSTM Seq2Seq:**
   - Short horizons (<24h)
   - Limited data
   - Local temporal patterns dominate

2. **Transformer (MHA):**
   - Long horizons (>168h)
   - Large datasets
   - Complex long-range dependencies

3. **Transformer (Fourier):**
   - Strong periodicity (daily/weekly cycles)
   - Computational efficiency is critical
   - Frequency-domain insights are valuable

4. **Transformer (ProbSparse):**
   - Very long sequences (>720h)
   - Memory-constrained environments
   - Need to balance accuracy vs. scalability

---

## 🔑 Key Takeaways

1. **Architecture matters, but context matters more:** The "best" model depends on task characteristics (horizon, data volume, patterns).
2. **Preprocessing is critical:** STL decomposition and proper data splitting prevent leakage and stabilize training.
3. **Physics > Pure Statistics:** For physical systems (ETTh1), incorporating causal features (load → temperature) is essential.
4. **LSTM is underrated:** For short horizons, LSTMs remain highly competitive despite the Transformer hype.

---

## 📄 License

This project is for academic purposes. Datasets are from public repositories (UCI, ETTh1 Benchmark).
