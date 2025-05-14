import os
import sys
import argparse
import pandas as pd
import cProfile
import pstats

# Allow imports from sibling utils/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from BuyWritePortfolioSimulator import BuyWritePortfolioSimulator
from build_price_df import build_price_df

# Set working directory to the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"📂 Working directory: {os.getcwd()}")

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--profile", action="store_true", help="Run simulation with profiling")
args = parser.parse_args()

# Load pre-calculated volatility summary
vol_summary_df = pd.read_csv("../data/volatility_summary.csv", parse_dates=["Date"])

# Preload all price data
price_dir = "../data/stock_prices"
price_df = build_price_df(price_dir)

# Create simulator instance
simulator = BuyWritePortfolioSimulator(
    vol_summary_df=vol_summary_df,
    price_dir=price_dir,
    notional_per_trade=10000,
    num_positions=5,
    option_days=21,
    sector_limit=3,
    correlation_threshold=0.85,
    screen_mode='breakout',
    theta=0.5,
    alpha=1.0,
    vol_zscore_threshold=0,
    price_df=price_df,
    debug=True
)

# Define wrapped runner
def run_simulation():
    portfolio_results = simulator.run()
    portfolio_results.to_csv("../output/portfolio_results.csv", index=False)
    print("✅ Simulation complete. Results saved to portfolio_results.csv.")

# Run in profile mode or normal mode
if args.profile:
    print("🧪 Running in profiling mode...")
    cProfile.run('run_simulation()', 'profile.prof')
    stats = pstats.Stats('profile.prof')
    stats.strip_dirs().sort_stats('cumulative').print_stats(30)
else:
    run_simulation()
