import numpy as np
import pandas as pd
import os
# Set working directory to the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import gc; gc.collect()
import logging
from scipy.optimize import differential_evolution
from tqdm import tqdm
from datetime import datetime
from BuyWritePortfolioSimulator import BuyWritePortfolioSimulator  # Adjust path if needed

# === CONFIG ===
notional_per_trade = 10000
max_iters = 30
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"optimization_{timestamp}.log")

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# === GLOBAL FOR TQDM ===
progress_bar = tqdm(total=max_iters, desc="Optimizing", ncols=100)

# === OBJECTIVE FUNCTION ===
def objective(params):
    num_positions, call_otm_pct, vol_lookback_days, tau, sector_limit, alpha = params
    param_str = f"np={num_positions:.2f}, otm={call_otm_pct:.3f}, vol={vol_lookback_days:.0f}, tau={tau:.0f}, sl={sector_limit:.0f}, alpha={alpha:.2f}"

    vol_summary_df = pd.read_csv("../data/volatility_summary.csv", parse_dates=["Date"])
    price_dir="../data/stock_prices"
    try:
        sim = BuyWritePortfolioSimulator(
            num_positions=int(num_positions),
            call_otm_pct=call_otm_pct,
            vol_lookback_days=int(vol_lookback_days),
            tau=float(tau),
            sector_limit=int(sector_limit),
            alpha=float(alpha),
            vol_summary_df=vol_summary_df,
            price_dir=price_dir,
        )
        

        results_df = sim.run()
        pnl_total = results_df["PnL_Total_position"].sum()
        num_rows = len(results_df)

        if num_rows == 0:
            raise ValueError("No trades executed")

        pct_profit = pnl_total / (notional_per_trade * num_rows)

        logging.info(f"[OK] {param_str} → %Profit = {pct_profit:.4%}")
        return -pct_profit

    except Exception as e:
        logging.warning(f"[FAIL] {param_str} → Error: {e}")
        return 1e6

    finally:
        progress_bar.update(1)

# === BOUNDS ===
bounds = [
    (2, 10),          # num_positions
    (0.01, 0.2),      # call_otm_pct
    (20, 90),         # vol_lookback_days
    (-1, 1),          # tau
    (1, 5),           # sector_limit
    (0.0, 1.0)        # alpha
]

# === MAIN ===
if __name__ == "__main__":
    logging.info("🚀 Starting BuyWrite Optimization")

    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=10,
        polish=True,
        workers=2,
        disp=False
    )

    progress_bar.close()

    logging.info("✅ Optimization Complete")
    logging.info(f"Best Parameters: {result.x}")
    logging.info(f"Max % Profit: {-result.fun:.4%}")

    print("\nBest Parameters:", result.x)
    print("Max % Profit:", -result.fun)
