import os
from BuyWritePortfolioSimulator import BuyWritePortfolioSimulator
import pandas as pd

# Set working directory to the location of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(os.getcwd())

# Load your pre-calculated volatility summary DataFrame
vol_summary_df = pd.read_csv("../data/volatility_summary.csv", parse_dates=["Date"])

# Initialize the simulator
simulator = BuyWritePortfolioSimulator(
    vol_summary_df=vol_summary_df,
    price_dir="../data/stock_prices",
    notional_per_trade=10000,
    num_positions=5,
    option_days=21,
    sector_limit=3,
    correlation_threshold=0.85,
    screen_mode='breakout',
    vol_zscore_threshold=0,
    debug=True
)

# Run the simulation
portfolio_results = simulator.run()

# Save results
portfolio_results.to_csv("../output/portfolio_results.csv", index=False)
print("Simulation complete. Results saved to portfolio_results.csv.")
