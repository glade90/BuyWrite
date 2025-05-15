# Final Project – Buy-Write Strategy Optimization

This project implements and tests a systematic buy-write strategy using volatility breakout signals and directional momentum filters. The strategy rotates through a portfolio of S&P 500 stocks, selling covered calls each cycle while enforcing sector exposure and correlation constraints.

## 📊 Overview

- **Objective**: Maximize Sharpe ratio through systematic options overlays
- **Asset Universe**: S&P 500
- **Methodology**:
  - Rolling simulations with dynamic rebalancing
  - Volatility z-score screening and directional filters
  - Sector and correlation controls
  - Hyperparameter optimization using Optuna

## 📁 Folder Structure

```
FinalProject/
│
├── notebooks/               # Jupyter Notebooks for analysis and visualization
├── data/                    # Input data including volatility summaries and stock prices
├── utils/                   # Generic functions
├── py/                      # Core Python modules including simulator logic
├── logs/                    # Log files from optimization runs

```

## ⚙️ Optimization

An Optuna hyperparameter search was run over training data from 2015 to 2023. The model was validated using 2024 as an out-of-sample test year. Each candidate strategy was scored on training Sharpe, with validation performance logged.

### Key Result:
- Many configurations performed well in training
- Most failed to generalize in 2024, suggesting overfitting or non-stationarity

## 📈 Conclusion

This submission represents a functional prototype of a quantitative buy-write simulator and optimization engine. Though promising in design, it currently lacks robustness in out-of-sample validation.

## 🔭 Next Steps

- **Walk-forward validation** with rolling train/test splits
- **Deleveraging logic** based on VIX regime
- **Dividend treatment** in return calculations
- **Real implied vol data** (e.g., OptionMetrics, Bloomberg)
- **Alternative distributions** for better skew modeling
- **Scenario testing** for extreme risk conditions

## 📄 Submission

This project includes:
- Final simulation notebook (`final_project_buywrite_submission.ipynb`)
- Exportable PDF version (user-generated)
- Markdown summary (`FinalProject_Conclusion.md`)
- All necessary code and data files

---

Created for CSCI-E278 Final Project. Please contact Eric Kasper for inquiries.
