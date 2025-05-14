import optuna
import torch
import pandas as pd
import logging
from BuyWritePortfolioSimulator import BuyWritePortfolioSimulator

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

"""
notional_per_trade = 10000

def objective(trial):
    num_positions = trial.suggest_int("num_positions", 2, 10)
    call_otm_pct = trial.suggest_float("call_otm_pct", 0.01, 0.1)
    theta = trial.suggest_float("theta", -0.5, 0.5)
    

    vol_summary_df = pd.read_csv("../data/volatility_summary.csv", parse_dates=["Date"])
    price_dir = "../data/stock_prices"

    try:
        sim = BuyWritePortfolioSimulator(
            num_positions=num_positions,
            vol_summary_df=vol_summary_df,
            call_otm_pct=call_otm_pct,
            theta=theta,           
            price_dir=price_dir,
        )
        results_df = sim.run()

        capital_base = notional_per_trade * num_positions
        returns = results_df["PnL_Total_position"] / capital_base

        if len(returns) < 2 or returns.std() == 0:
            return -1e6

        sharpe = (returns.mean() / returns.std()) * torch.sqrt(torch.tensor(12.0))
        return sharpe.item()

    except Exception as e:
        return -1e6

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best trial:")
print(study.best_trial)
"""