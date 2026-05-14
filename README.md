# 🏠 House Price Prediction — Machine Learning Project

A complete end-to-end Machine Learning project that predicts house prices using regression models. Built with Python, this project covers the full ML pipeline — from raw data exploration to model evaluation and visualization.

---

## 📌 Project Overview

This project trains and compares two regression models — **Linear Regression** and **Decision Tree Regressor** — to predict house prices based on features like area, bedrooms, bathrooms, age, garage availability, and location score.

---

## 📁 Dataset

**File:** `house_prices.csv`

| Feature | Description |
|---|---|
| `area_sqft` | Total area of the house in square feet |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `age_years` | Age of the property in years |
| `garage` | Garage availability (yes / no) |
| `location_score` | Score representing location quality |
| `price` | **Target variable** — house price |

---

## 🔧 Tech Stack

- **Python 3.x**
- **Pandas** — data loading and manipulation
- **NumPy** — numerical computations
- **Matplotlib & Seaborn** — data visualization
- **Scikit-learn** — model training and evaluation

---

## 🚀 Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Checked shape, data types, and missing values
- Generated statistical summary with `.describe()`
- Plotted histograms for all features
- Correlation heatmap to identify relationships
- Scatter plot: Area vs Price

### 2. Data Preprocessing
- Encoded `garage` column: `yes → 1`, `no → 0`
- Separated features (X) and target (y)
- Applied **80/20 Train-Test Split**
- Applied **StandardScaler** for feature normalization (fit on train, transform on test)

### 3. Model Training

| Model | Key Parameter |
|---|---|
| Linear Regression | Default |
| Decision Tree Regressor | `max_depth=5`, `random_state=42` |

### 4. Model Evaluation

Each model was evaluated using:

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average prediction error |
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **R² Score** | Proportion of variance explained by the model |

### 5. Visualization
- Side-by-side **Actual vs Predicted** scatter plots for both models
- Price axis formatted in **K (thousands)** for readability
- Perfect prediction reference line plotted in red

---

## 📊 Results

```
Model               MAE        RMSE       R²
─────────────────────────────────────────────
Linear Regression   ...        ...        ...
Decision Tree       ...        ...        ...
```
> Fill in your actual metric values after running the script.

---

## 📂 Project Structure

```
house-price-prediction/
│
├── house_prices.csv        # Dataset
├── house_price_model.py    # Main script
└── README.md               # Project documentation
```

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

3. **Run the script**
   ```bash
   python house_price_model.py
   ```

---

## 💡 Key Learnings

- Feature scaling is critical for Linear Regression but has no effect on Decision Trees
- Correlation heatmaps help identify which features are most predictive
- Visualizing Actual vs Predicted is one of the best ways to evaluate regression models intuitively
- Always fit the scaler **only on training data** to avoid data leakage


