import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
# Set working directory to the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

feature_rows = []

for _, row in portfolio_results.iterrows():
    ticker = row["Ticker"]
    entry_date = pd.to_datetime(row["EntryDate"])
    expiry_date = pd.to_datetime(row["ExpiryDate"])

    price_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(price_path):
        continue

    try:
        df = pd.read_csv(price_path, parse_dates=["Date"], index_col="Date").sort_index()

        # Realized Vol: forward 21-day volatility
        fwd_prices = df.loc[entry_date:expiry_date]["Close"]
        fwd_log_returns = np.log(fwd_prices / fwd_prices.shift(1))
        realized_vol = fwd_log_returns.std() * np.sqrt(252)

        # Momentum_1M: trailing 21-day return
        trailing_window = df.loc[:entry_date].iloc[-option_days:]["Close"]
        if len(trailing_window) >= 2:
            momentum = (trailing_window.iloc[-1] / trailing_window.iloc[0]) - 1
        else:
            momentum = np.nan

    except Exception as e:
        print(f"⚠️ Error for {ticker}: {e}")
        realized_vol = np.nan
        momentum = np.nan

    feature_rows.append({
        "Ticker": ticker,
        "EntryDate": entry_date,
        "Vol_ZScore": row.get("Vol_ZScore"),
        "Sector": row.get("Sector"),
        "IV_Proxy": row.get("Premium") / row.get("EntryPrice") if row.get("EntryPrice") else np.nan,
        "RealizedVol": realized_vol,
        "Momentum_1M": momentum,
        "PnL_Total_position": row.get("PnL_Total_position"),
        "PnL_Label": int(row.get("PnL_Total_position", 0) > 0)
    })

features_df = pd.DataFrame(feature_rows)

# Drop missing
ml_df = features_df.dropna(subset=["Vol_ZScore", "RealizedVol", "IV_Proxy", "Momentum_1M", "PnL_Label"])

# Define features and target
X = ml_df[["Vol_ZScore", "RealizedVol", "IV_Proxy", "Momentum_1M"]]
y = ml_df["PnL_Label"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train random forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)
rf.fit(X_train, y_train)rf.fit(X_train, y_train)

# Predict
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]
