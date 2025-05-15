import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.imports import *

def build_price_df(price_dir, start_date=None, end_date=None):
    print(f"📂 Loading price data from {price_dir}")
    all_data = []
    
    for fname in os.listdir(price_dir):
        if not fname.endswith(".csv"):
            continue
        ticker = fname.replace(".csv", "")
        fpath = os.path.join(price_dir, fname)
        try:
            df = pd.read_csv(fpath, parse_dates=["Date"])
            df["Ticker"] = ticker

            # Filter by date range if provided
            if start_date:
                df = df[df["Date"] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df["Date"] <= pd.to_datetime(end_date)]

            all_data.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {fname}: {e}")

    if not all_data:
        raise ValueError("No price files loaded.")

    price_df = pd.concat(all_data, ignore_index=True)
    price_df.set_index(["Ticker", "Date"], inplace=True)
    price_df.sort_index(inplace=True)
    print(f"📂 Complete")

    return price_df

