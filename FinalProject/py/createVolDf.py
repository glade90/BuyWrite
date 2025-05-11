import numpy as np
import pandas as pd
import os
# Set working directory to the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = '..\\data\stock_prices'
VOL_SUMMARY_DIR = "../data/volatility_summary.csv"

def createVolDf(vol_window=30, zscore_window=60):
    vol_list = []

    for file in os.listdir(DATA_DIR):
        if not file.endswith(".csv"):
            continue

        ticker = file.replace(".csv", "")
        path = os.path.join(DATA_DIR, file)

        try:
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")

            # Daily log returns
            df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

            # Rolling vol
            df["RollingVol"] = df["LogReturn"].rolling(window=vol_window).std() * np.sqrt(252)

            # Drop NaNs
            temp = df[["RollingVol"]].dropna().copy()
            temp["Ticker"] = ticker
            temp.reset_index(inplace=True)

            vol_list.append(temp)

        except Exception as e:
            print(f"⚠️ Error processing {ticker}: {e}")

    # Combine all tickers
    vol_df = pd.concat(vol_list, ignore_index=True)

    # Calculate Z-scores across time within each ticker
    vol_df["Vol_ZScore"] = (
        vol_df.groupby("Ticker")["RollingVol"]
        .transform(lambda x: (x - x.rolling(zscore_window).mean()) / x.rolling(zscore_window).std())
    )

    # Save for use in screening
    vol_df.to_csv(VOL_SUMMARY_DIR, index=False)
    print("✅ Saved volatility summary with breakout scores.")