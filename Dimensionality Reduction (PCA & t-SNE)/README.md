# Stock Market Financial Health Analysis
## Dimensionality Reduction dengan PCA dan t-SNE

**Author:** Maulana (Informatika)  
**Mata Kuliah:** Teknik Dimensionality Reduction  
**Date:** 2026  
**Topic:** Principal Component Analysis (PCA) dan t-Distributed Stochastic Neighbor Embedding (t-SNE)

---

## 📋 Daftar Isi

1. [Deskripsi Kasus](#deskripsi-kasus)
2. [Dataset](#dataset)
3. [Metodologi](#metodologi)
4. [Implementasi](#implementasi)
5. [Hasil & Analisis](#hasil--analisis)
6. [Perbandingan PCA vs t-SNE](#perbandingan-pca-vs-tsne)
7. [Conclusion](#conclusion)
8. [Setup & Penggunaan](#setup--penggunaan)
9. [References](#references)

---

## 🎯 Deskripsi Kasus

### Masalah yang Dihadapi

Dalam analisis fundamental saham, setiap perusahaan memiliki **11+ metrik keuangan** yang mencerminkan kesehatan finansial mereka:
- Profitability metrics (ROE, profit margin, operating margin)
- Liquidity metrics (current ratio, cash ratio, quick ratio)
- Cash flow metrics (operating cash flow, net cash flow)
- Efficiency metrics (earnings per share)

**Tantangan:**
- **High-dimensional data**: 11+ dimensi membuat eksplorasi pola sulit
- **Curse of dimensionality**: Jarak euclidean tidak bermakna di high-dimension
- **Visualisasi terbatas**: Tidak bisa langsung plot 11D ke 2D
- **Computational overhead**: Clustering & modeling lebih lambat
- **Interpretasi kompleks**: Hubungan antar metrik tidak jelas

### Solusi: Dimensionality Reduction

Menggunakan **PCA** dan **t-SNE** untuk:
1. **Reduce dimensionality** dari 11D ke 2D dengan minimal information loss
2. **Discover natural clusters** dalam stock market berdasarkan fundamentals
3. **Categorize stocks** by financial health (Strong/Average/Weak)
4. **Compare linear vs non-linear** approaches

### Tujuan Analysis

| Tujuan | Metode | Output |
|--------|--------|--------|
| Feature reduction & interpretation | PCA | 6 principal components (95% variance) |
| Cluster visualization & discovery | t-SNE | 2D coordinates dengan clear separation |
| Stock categorization | Health scoring | Financial health categories |
| Method comparison | Both | Insights tentang trade-offs |

---

## 📊 Dataset

### Data Source
- **Securities data**: 505 perusahaan S&P 500 dengan sector classification (GICS)
- **Fundamentals data**: 1781 financial records across multiple years
- **Price data**: 851,265 daily OHLCV records (untuk referensi)

### Data Karakteristik

```
Total Stocks Analyzed:     429 (after cleaning)
Total Features:            11 financial metrics
Missing Values:            ~15% (handled via imputation)
Outliers:                  Retained (represent real variation)
Time Period:               Latest available per ticker
```

### Feature Selection

Dipilih 11 financial metrics yang comprehensive:

| Category | Features | Rationale |
|----------|----------|-----------|
| **Profitability** | Gross Margin, Operating Margin, Profit Margin | Core efficiency metrics |
| **Returns** | After Tax ROE, Pre-Tax ROE, Earnings Per Share | Shareholder value creation |
| **Liquidity** | Current Ratio, Cash Ratio, Quick Ratio | Short-term financial health |
| **Cash Flow** | Net Cash Flow, Operating Cash Flow | Real cash generation |

### Data Preprocessing

```
Step 1: Get latest fundamentals per ticker
        ├─ 448 companies dengan sector
        
Step 2: Select 11 financial features
        ├─ Remove rows dengan > 30% missing
        ├─ 429 stocks remain
        
Step 3: Imputation
        ├─ Fill NaN dengan median per feature
        
Step 4: Standardization (CRITICAL untuk PCA)
        ├─ Mean = 0, Std = 1 per feature
        ├─ Z-score: X_scaled = (X - mean) / std
        
Step 5: Final cleaning
        ├─ Remove remaining NaN
        ├─ 267 stocks dalam final analysis
```

---

## 🔧 Metodologi

### Principal Component Analysis (PCA)

#### Konsep
PCA adalah **linear** dimensionality reduction technique yang:
1. Mencari **principal components** (kombinasi linear dari original features)
2. PC1 = arah maximum variance
3. PC2 = maximum variance orthogonal ke PC1
4. Dst... untuk PC ke-n

#### Matematis
```
Given: Data matrix X (n_samples × n_features)

1. Standardize: X_scaled = (X - mean) / std
2. Compute covariance matrix: C = X_scaled^T × X_scaled / (n-1)
3. Eigenvalue decomposition: C = U × Σ × V^T
4. Principal components: PC_k = X_scaled × U_k
5. Explained variance: variance_k = Σ_k / sum(Σ)

Output: X_reduced (n_samples × n_components) dimana n_components << n_features
```

#### Keuntungan & Keterbatasan

| Aspek | Keuntungan | Keterbatasan |
|-------|-----------|--------------|
| **Linear** | Interpretable axes | Misses non-linear patterns |
| **Fast** | O(n) complexity | Tidak scalable ke very high-dim |
| **Parametric** | Can transform new data | Kurang fleksibel |
| **Global** | Preserves global structure | Lose local details |

#### Implementasi

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

# Hasil:
# - X_pca shape: (267, 6) → 11 features ke 6 components
# - Variance explained: 95.1%
# - Component loadings: interpretable dari original features
```

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

#### Konsep
t-SNE adalah **non-linear** dimensionality reduction yang:
1. Preserve **local neighborhood** structure
2. Similar points stay close
3. Dissimilar points pushed apart
4. Better untuk cluster visualization

#### Matematis (Simplified)
```
1. Compute pairwise distances di high-dimensional space
2. Convert ke probability distributions (Gaussian kernels)
3. Initialize random low-dimensional embedding
4. Minimize KL divergence antara high-D & low-D distributions
5. Use gradient descent dengan momentum
6. Iterate hingga convergence

Key parameter: perplexity (balance local-global, typically 5-50)
```

#### Keuntungan & Keterbatasan

| Aspek | Keuntungan | Keterbatasan |
|-------|-----------|--------------|
| **Non-linear** | Captures complex patterns | Cannot interpret axes |
| **Cluster clarity** | Better separation | Slow O(n²) complexity |
| **Local focus** | Good for exploration | Distorts global distances |
| **Non-parametric** | Flexible | Cannot transform new data directly |

#### Implementasi

```python
from sklearn.manifold import TSNE

# Input: PCA output (faster than raw features)
tsne = TSNE(
    n_components=2,
    perplexity=30,        # Balance local-global structure
    max_iter=1000,        # Iteration count
    random_state=42       # Reproducibility
)
X_tsne = tsne.fit_transform(X_pca)

# Hasil:
# - X_tsne shape: (267, 2) → 6D PCA ke 2D t-SNE
# - Strong visual cluster separation
# - Non-parametric (cannot transform new data)
```

### Financial Health Categorization

Untuk coloring/annotation, kami membuat **categorical labels** based on composite score:

```python
Health Score = 
    (ROE > 0.5 quantile)           ? 1 : 0  +
    (Profit Margin > 0.5 quantile) ? 1 : 0  +
    (Current Ratio > 0.5 quantile) ? 1 : 0

Categories:
- Score = 3 → Strong (7-8)        [Green]
- Score = 2 → Average (5-6)       [Gray]
- Score ≤ 1 → Weak (3-4)          [Red]
```

---

## 💻 Implementasi

### File Structure

```
stock-market-dimensionality-reduction/
│
├── Stock_Market_Dimensionality_Reduction.ipynb
│   ├── Data loading & preparation
│   ├── PCA implementation & analysis
│   ├── t-SNE implementation & analysis
│   ├── Clustering (K-Means)
│   ├── Comparison & insights
│   └── Visualizations
│
├── figures/
│   ├── 01_pca_variance_explained.png
│   ├── 02_pca_2d_visualization.png
│   ├── 03_tsne_2d_visualization.png
│   ├── 04_clustering_results.png
│   └── 05_pca_vs_tsne_comparison.png
│
├── data/
│   ├── securities.csv          (505 stocks, metadata)
│   ├── fundamentals.csv        (1781 records, 77 features)
│   └── prices.csv              (851K rows, OHLCV)
│
├── README.md                    (this file)
└── requirements.txt             (dependencies)
```

### Notebook Sections

1. **Data Loading & Preprocessing** (Cells 1-5)
   - Load securities & fundamentals data
   - Merge & clean (429 stocks)
   - Feature selection (11 metrics)
   - Standardization

2. **PCA Analysis** (Cells 6-9)
   - Apply PCA (95% variance)
   - Variance decomposition
   - Loading analysis & interpretation
   - 2D visualization (by health & sector)

3. **t-SNE Analysis** (Cells 10-12)
   - Apply t-SNE (on PCA output)
   - 2D visualization
   - Compare dengan PCA output

4. **Clustering** (Cells 13-14)
   - K-Means (k=5)
   - Cluster characteristics
   - Visualization

5. **Comparison & Analysis** (Cells 15-18)
   - PCA vs t-SNE comparison table
   - Comprehensive 4-panel figure
   - Insights & recommendations

### Requirements

```
pandas>=1.0.0
numpy>=1.18.0
matplotlib>=3.1.0
seaborn>=0.10.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

### Running the Analysis

```bash
# 1. Clone repository
git clone https://github.com/maulana-dimensionality-reduction.git
cd stock-market-dimensionality-reduction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create data directory & add CSV files
mkdir data
# Place securities.csv, fundamentals.csv, prices.csv in data/

# 4. Create figures directory
mkdir figures

# 5. Run Jupyter notebook
jupyter notebook Stock_Market_Dimensionality_Reduction.ipynb
```

---

## 📈 Hasil & Analisis

### PCA Results

#### Variance Explained

```
Original dimensions:       11 features
Reduced dimensions:        6 components
Variance retained:         95.1%

Breakdown:
  PC1: 28.64%  (Profitability axis - ROE, margins)
  PC2: 25.07%  (Liquidity axis - ratios)
  PC3: 14.17%  (Cash flow variations)
  PC4: 11.90%  (Operational efficiency)
  PC5:  8.27%  (Financial leverage)
  PC6:  6.85%  (Market size/scale)
```

#### Key Findings

1. **PC1 Interpretation (Profitability)**
   - High positive: Strong ROE, high margins → profitable companies
   - High negative: Weak ROE, low margins → struggling companies
   - Dominated oleh: After Tax ROE, Profit Margin, Operating Margin

2. **PC2 Interpretation (Liquidity)**
   - High positive: High liquidity ratios → strong cash position
   - High negative: Low liquidity ratios → potential cash constraints
   - Dominated oleh: Current Ratio, Cash Ratio, Quick Ratio

3. **Stock Distribution**
   - Stocks spread along diagonal (unprofitable-profitable spectrum)
   - Clear separation antara financial health categories
   - Sector membership tidak predict position (fundamentals > sector)

### t-SNE Results

#### Clustering Visualization

```
t-SNE output clearly separates:
  - Strong stocks (green) → clustered in specific region
  - Weak stocks (red) → widely scattered
  - Average stocks (gray) → intermediate region

Key difference from PCA:
  - Better local separation (clusters tighter)
  - Non-linear relationships revealed
  - Axes NOT interpretable (for visualization only)
```

#### Cluster Composition

```
Cluster 0:  11 stocks
  Dominant sector: Information Technology
  Characteristic: High performers (finance/tech)

Cluster 1: 238 stocks
  Dominant sector: Consumer Discretionary
  Characteristic: Mainstream/average companies

Cluster 2:   5 stocks
  Dominant sector: Energy
  Characteristic: Specialized, volatile (oil & gas)

Cluster 3:  10 stocks
  Dominant sector: Information Technology
  Characteristic: Growth stocks, specialized tech

Cluster 4:   3 stocks
  Dominant sector: Financials
  Characteristic: Unique profile (outliers)
```

#### Financial Health Distribution

```
Strong (7-8):     4 stocks ( 1.5%)  [Excellent fundamentals]
Average (5-6):   17 stocks ( 6.4%)  [Adequate metrics]
Weak (3-4):     246 stocks (92.1%)  [Weak or mixed profile]
```

Insight: Mayoritas stocks di-characterize sebagai "Weak" karena **tidak semua bisa excellent**. Kategori "Weak" mencakup range luas dari average ke truly weak.

### Visualizations

#### Figure 1: Variance Explained
- Individual variance per PC
- Cumulative variance (threshold 95%)
- Shows dimensionality reduction efficiency

#### Figure 2: PCA 2D Visualization
- Left: Colored by Financial Health (Strong/Average/Weak)
- Right: Colored by GICS Sector
- Observation: Health categories cluster, sectors scattered

#### Figure 3: t-SNE 2D Visualization
- Left: By Financial Health
- Right: By Sector
- Observation: Better cluster separation than PCA

#### Figure 4: Clustering Results
- K-Means clusters di PCA space
- K-Means clusters di t-SNE space
- Shows both linear & non-linear clustering

#### Figure 5: PCA vs t-SNE Comparison
- 2×2 grid showing both methods
- Colored by health & clusters
- Direct visual comparison

---

## 🔍 Perbandingan PCA vs t-SNE

### Tabel Perbandingan

| Kriteria | PCA | t-SNE |
|----------|-----|-------|
| **Type** | Linear | Non-linear |
| **Relationship Focus** | Global structure | Local neighborhoods |
| **Preservation** | Maximum variance | Pairwise distances |
| **Axes Interpretable** | ✅ Yes (loadings) | ❌ No |
| **Computation** | Fast (O(n)) | Slow (O(n²)) |
| **New Data** | ✅ Transform directly | ❌ Recompute |
| **Use Case** | Preprocessing | Exploration |
| **Clusters** | Less separated | Better separated |
| **Reliability** | Stable | Parameter-sensitive |

### Analisis Hasil untuk Dataset Ini

#### PCA Keunggulan
1. **Interpretability**: PC1 jelas = profitability, PC2 = liquidity
2. **Global view**: Melihat seluruh stock market structure
3. **Production-ready**: Dapat digunakan untuk new stocks
4. **Stable**: Reproducible results independent of random seed

#### PCA Keterbatasan
1. **Linear assumption**: Tidak capture non-linear relationships
2. **Less visual**: Clusters tidak se-separated dengan t-SNE
3. **Information mixing**: Mixed signals dalam axes

#### t-SNE Keunggulan
1. **Visual clarity**: Cluster separation much better
2. **Exploration**: Great untuk discovering patterns
3. **Peer finding**: Similar stocks naturally cluster (local neighborhoods)

#### t-SNE Keterbatasan
1. **Non-interpretable**: Axes meaningless, cannot explain structure
2. **Non-parametric**: Cannot transform new data (must recompute)
3. **Computationally expensive**: O(n²) complexity
4. **Parameter-sensitive**: Perplexity choices affect results

### Mana yang Lebih Sesuai?

#### Gunakan PCA untuk:
✅ **Feature preprocessing** sebelum machine learning
✅ **Understanding what varies** in data (interpretability)
✅ **Production systems** (new data inference)
✅ **Statistical analysis** (variance decomposition)

#### Gunakan t-SNE untuk:
✅ **Exploratory Data Analysis**
✅ **Cluster discovery**
✅ **Interactive visualization** & user exploration
✅ **Finding similar items** (peers, anomalies)

#### Untuk Stock Analysis - GUNAKAN KEDUA:
1. **PCA**: Understand fundamental dimensions (profitability vs liquidity)
2. **t-SNE**: Discover true peer groups (fundamental neighbors)
3. **Pipeline**: Raw data → PCA (95% var) → t-SNE (visualization) → Interpret

---

## 🎓 Insights & Recommendations

### Key Insights

1. **Financial Fundamentals > Industry Sector**
   - Stock clustering based on fundamentals, tidak industry
   - Tech stock yang kaya cash clusters dengan finance stocks
   - Energy stocks punya unique risk profile (isolated cluster)
   - **Implication**: Use fundamental-based grouping, not sector

2. **Profitability-Liquidity Trade-off**
   - PC1 vs PC2 menunjukkan trade-off antara profitability & liquidity
   - Profitable companies tidak selalu liquid (growth stocks)
   - Liquid companies tidak selalu profitable (defensive stocks)
   - **Implication**: Consider both dimensions, not one

3. **Majority of Stocks are "Average"**
   - 92% stocks classified as "Weak" (low health score)
   - Reflects reality: most companies have mixed profiles
   - Only 4 "Strong" stocks (excellent all metrics)
   - **Implication**: Be selective, focus on truly strong stocks

4. **Cluster Stability**
   - 5 natural clusters identified by K-Means
   - Stable across PCA & t-SNE (independent validations)
   - Suggests these are real fundamental groupings
   - **Implication**: Cluster membership meaningful for analysis

### Practical Applications

#### 1. Portfolio Construction
```
Strategy: Diversify across clusters, not sectors

Step 1: Use t-SNE to find 5 representative stocks
        from each cluster
Step 2: Avoid clustering (high correlation of fundamentals)
Step 3: Balance across health levels & sizes
Step 4: Result: Portfolio with true diversification
```

#### 2. Peer Valuation
```
Traditional: Compare P/E within same sector
Better:      Find peers in same cluster (same fundamentals)

Example:
  Stock X in Cluster 2 (energy)
  Peers: Other energy stocks in Cluster 2
  NOT:   All energy stocks (sectors don't matter)
```

#### 3. Risk Management
```
Monitor: Cluster membership stability

Signals:
  ✅ Stock in Cluster 0 → consistently strong → low risk
  ⚠️  Stock moving clusters → fundamental shift
  ❌ Stock in Cluster 2 → high volatility → manage risk
```

#### 4. Anomaly Detection
```
Use t-SNE outliers:

Far from any cluster → unique fundamental profile
  → Requires specialized analysis
  → Potential opportunity or trap

Use PCA extremes:
  
Extreme on PC1 → very profitable or unprofitable
Extreme on PC2 → very liquid or illiquid
  → Potential special situations
```

---

## 📚 References

### PCA References
1. Jolliffe, I. T. (2002). Principal Component Analysis (2nd ed.). Springer.
2. Turk, M., & Pentland, A. (1991). Eigenfaces for recognition. Journal of Cognitive Neuroscience.
3. Scikit-learn PCA Documentation: https://scikit-learn.org/stable/modules/decomposition.html#pca

### t-SNE References
1. van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. JMLR.
2. Wattenberg, M., Viégas, F., & Johnson, I. (2016). How to Use t-SNE Effectively. Distill.
3. Scikit-learn t-SNE Documentation: https://scikit-learn.org/stable/modules/manifold.html#t-sne

### Finance Application References
1. Mauboussin, M. J. (2012). Expectations Investing. Columbia Business School.
2. Damodaran, A. (2012). Investment Valuation: Tools and Techniques.
3. S&P 500 Fundamentals Dataset: https://www.kaggle.com/cdc/nyse-data

---

## 📝 Conclusion

### Summary

Studi ini mendemonstrasikan penggunaan **Dimensionality Reduction** (PCA & t-SNE) untuk menganalisis pasar saham:

1. **PCA** berhasil mereduksi 11 financial metrics ke 6 components (95% variance)
   - Interpretable axes: PC1=Profitability, PC2=Liquidity
   - Suitable untuk preprocessing & feature understanding
   - Preserves global market structure

2. **t-SNE** mengungkapkan non-linear patterns & natural clusters
   - Better visual cluster separation
   - Discovers fundamental-based peer groups
   - Suitable untuk exploratory analysis & peer finding

3. **Comparison Results**:
   - PCA more interpretable & parametric
   - t-SNE better untuk visualization & exploration
   - Keduanya valuable untuk different purposes
   - **Pipeline approach** (PCA → t-SNE) recommended

### Key Takeaway

Dalam analisis pasar saham, **fundamental relationships lebih important daripada sector classification**. Dimensionality reduction techniques (PCA & t-SNE) mengungkapkan struktur hidden ini, enabling better investment decisions through:
- Fundamental-based peer discovery
- True diversification (across clusters, not sectors)
- Risk-adjusted analysis
- Anomaly detection

---

## 👤 Author Info

**Maulana** - Informatika Student  
Universitas (Yogyakarta, Indonesia)

**Contact:**
- GitHub: @maulana-username
- Email: maulana@example.com

---

## 📄 License

This project is open source and available under the MIT License.

---

**Last Updated:** July 2026  
**GitHub Repository:** [stock-market-dimensionality-reduction](https://github.com/maulana-repo)
