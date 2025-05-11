import os
import time
import json
import shutil
import logging
import pandas as pd
import yfinance as yf

from pathlib import Path

# === PATH SETUP ===
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

TICKER_CSV_PATH = PROJECT_ROOT / "data/sp500_tickers.csv"
DATA_DIR = PROJECT_ROOT / "data/stock_prices"
SECTOR_OUTPUT_PATH = PROJECT_ROOT / "data/sector_map.json"
MISSING_LOG_PATH = PROJECT_ROOT / "data/missing_tickers.txt"
LOG_PATH = PROJECT_ROOT / "logs/download.log"

# === FUNCTION ===
def download_sp500_with_sectors(
    ticker_csv_path=TICKER_CSV_PATH,
    data_dir=DATA_DIR,
    sector_output_path=SECTOR_OUTPUT_PATH,
    missing_log_path=MISSING_LOG_PATH,
    start_date="2015-01-01",
    end_date="2024-04-30",
    batch_size=20,
    wait_time=5,
    clear_existing=False,
    log_path=LOG_PATH
):
    """
    Downloads historical stock prices and sector info for tickers in sp500_tickers.csv.
    Saves data to disk and returns summary statistics as a dictionary.
    """

    # Set up logging
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filemode="w"
    )

    logging.info("🚀 Starting S&P 500 data download")

    if os.path.exists(data_dir) and clear_existing:
        shutil.rmtree(data_dir)
        logging.warning(f"🧹 Cleared old data from {data_dir}")
    os.makedirs(data_dir, exist_ok=True)

    # Load tickers
    try:
        sp500_tickers = pd.read_csv(ticker_csv_path)['Ticker'].tolist()
    except Exception as e:
        logging.error(f"Failed to read ticker file: {e}")
        return {"status": "error", "message": str(e)}

    sector_map = {}
    total = len(sp500_tickers)
    saved = 0
    skipped = 0
    sector_missing = 0

    with open(missing_log_path, "w") as missing_log:

        num_batches = (total + batch_size - 1) // batch_size  # ceiling division

        for i in range(0, total, batch_size):
            batch_num = i // batch_size + 1
            batch = sp500_tickers[i:i + batch_size]

            # Console + log output
            msg = f"📦 Pulling batch {batch_num} of {num_batches} ({len(batch)} tickers)"
            print(msg)
            logging.info(msg)

            try:
                data = yf.download(batch, start=start_date, end=end_date, group_by='ticker', threads=True)
            except Exception as e:
                logging.error(f"❌ Error pulling batch: {e}")
                continue

            for ticker in batch:
                try:
                    if ticker not in data.columns.levels[0]:
                        logging.warning(f"❌ No data for {ticker}, skipping.")
                        missing_log.write(ticker + "\n")
                        skipped += 1
                        continue

                    df = data[ticker].dropna()
                    if df.empty:
                        logging.warning(f"⚠️ {ticker} returned an empty DataFrame.")
                        missing_log.write(ticker + "\n")
                        skipped += 1
                        continue

                    df.to_csv(os.path.join(data_dir, f"{ticker}.csv"))
                    logging.info(f"✅ Saved {ticker}")
                    saved += 1

                    try:
                        info = yf.Ticker(ticker).info
                        sector = info.get("sector", "Unknown")
                        if sector == "Unknown":
                            sector_missing += 1
                        sector_map[ticker] = sector
                    except Exception as e:
                        logging.warning(f"⚠️ Could not fetch sector for {ticker}: {e}")
                        sector_map[ticker] = "Unknown"
                        sector_missing += 1

                    time.sleep(0.2)

                except Exception as e:
                    logging.error(f"❌ Error saving {ticker}: {e}")
                    skipped += 1

            time.sleep(wait_time)

    with open(sector_output_path, "w") as f:
        json.dump(sector_map, f, indent=2)

    logging.info("✅ Done downloading prices and sectors!")

    return {
        "tickers_total": total,
        "tickers_saved": saved,
        "tickers_skipped": skipped,
        "sector_missing": sector_missing,
        "status": "complete"
    }

# Optional: standalone run toggle
if __name__ == "__main__":
    result = download_sp500_with_sectors(clear_existing=True)
    print(result)
