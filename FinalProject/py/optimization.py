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
log_file = os.path.join(log_dir, f"optimization_{timestamp}_N_only.log")

# === LOGGING SETUP ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# === GLOBAL FOR TQDM ===
progress_bar = tqdm(total=max_iters, desc="Optimizing", ncols=100)

# === OBJECTIVE FUNCTION ===
def objective(params):
    #num_positions, call_otm_pct, theta, sector_limit,correlation_threshold = params
    num_positions = params
    #param_str = f"np={int(num_positions)}, otm={call_otm_pct * 100:.2f}%, theta={theta:.1f}, sl={sector_limit}"
    param_str = f"np={int(num_positions)}"

    vol_summary_df = pd.read_csv("../data/volatility_summary.csv", parse_dates=["Date"])
    price_dir="../data/stock_prices"
    try:
        sim = BuyWritePortfolioSimulator(
            num_positions=int(num_positions),
            #call_otm_pct=call_otm_pct,
            #theta=float(theta),
            #sector_limit=int(sector_limit),
            #correlation_threshold=float(correlation_threshold),
            vol_lookback_days=30,
            alpha=1,
            vol_summary_df=vol_summary_df,
            price_dir=price_dir,
        )
        

        results_df = sim.run()
        num_rows = len(results_df)

        if num_rows == 0:
            raise ValueError("No trades executed")

        capital_base = notional_per_trade * num_positions
        returns = results_df["PnL_Total_position"] / capital_base

        mean_return = returns.mean()
        std_return = returns.std()
        pct_profit = returns.sum()

        if std_return == 0 or len(returns) < 2:
            sharpe = -1e6
        else:
            sharpe = (mean_return / std_return) * np.sqrt(12)

        logging.info(f"[OK] {param_str} → Total Profit = {results_df["PnL_Total_position"].sum()} → %Profit = {pct_profit:.1%}, Sharpe = {sharpe:.2f}, N = {len(returns)}")

        return -sharpe  # Minimize the negative Sharpe ratio

    except Exception as e:
        logging.warning(f"[FAIL] {param_str} → Error: {e}")
        return 1e6

    finally:
        progress_bar.update(1)

# === BOUNDS ===
bounds = [
    (2, 10),          # num_positions
    #(0.01, 0.1),      # call_otm_pct
    #(-0.5, 0.5),          # theta
    #(1, 5),           # sector_limit
    #(0.75, 0.95),           # sector_limit
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
