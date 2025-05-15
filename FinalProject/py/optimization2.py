import os
import optuna
import torch
import pandas as pd
import logging
from BuyWritePortfolioSimulator import BuyWritePortfolioSimulator
from build_price_df import build_price_df
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ===== Configure Logging =====
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"buywrite_optimization_{timestamp}.log"
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ===== Print GPU Info (Console Only) =====
printGPUInfo = False
if printGPUInfo:
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())

# ===== Global Config =====
notional_per_trade = 10000

start_date = "2015-01-01"
split_date = "2024-01-01"
end_date = "2025-05-14"

# ===== Test Train Split =====
os.chdir(os.path.dirname(os.path.abspath(__file__)))
vol_summary_df = pd.read_csv("../data/volatility_summary.csv", parse_dates=["Date"])
price_dir = "../data/stock_prices"

# Load split price data
price_df = build_price_df(price_dir, start_date=start_date, end_date=end_date)

# ===== Objective Function =====
def objective(trial):
    num_positions = trial.suggest_int("num_positions", 2, 10)
    call_otm_pct = trial.suggest_float("call_otm_pct", 0.01, 0.1)
    theta = trial.suggest_float("theta", 0, 0.5)
    capital_base = num_positions * notional_per_trade

    print(f"OPTIMIZING with parameters: np={num_positions}, otm={call_otm_pct:.2%}, theta={theta:.3f}, capital_base=${capital_base:,.0f}")


    try:
        # === Training Simulation ===
        sim_train = BuyWritePortfolioSimulator(
            start_date=start_date,
            end_date=split_date,
            num_positions=num_positions,
            option_days=21,
            sector_limit=3,
            correlation_threshold=0.85,
            screen_mode='breakout',
            price_df=price_df,
            vol_summary_df=vol_summary_df,
            call_otm_pct=call_otm_pct,
            theta=theta,
            alpha=1.0,
            vol_zscore_threshold=0,
            price_dir=price_dir,
            debug=False,
        )
        results_train = sim_train.run()
        returns_train = results_train["PnL_Total_position"] / capital_base
        sharpe_train = (returns_train.mean() / returns_train.std()) * torch.sqrt(torch.tensor(12.0))
        total_pnl_train = results_train["PnL_Total_position"].sum()

        # === Validation Simulation ===
        sim_val = BuyWritePortfolioSimulator(
            start_date=split_date,
            end_date=end_date,
            num_positions=num_positions,
            option_days=21,
            sector_limit=3,
            correlation_threshold=0.85,
            screen_mode='breakout',
            price_df=price_df,
            vol_summary_df=vol_summary_df,
            call_otm_pct=call_otm_pct,
            theta=theta,
            alpha=1.0,
            vol_zscore_threshold=0,
            price_dir=price_dir,
            debug=False,
        )
        results_val = sim_val.run()
        returns_val = results_val["PnL_Total_position"] / capital_base
        sharpe_val = (returns_val.mean() / returns_val.std()) * torch.sqrt(torch.tensor(12.0))
        total_pnl_val = results_val["PnL_Total_position"].sum()

        info =  f"[OK] np={num_positions}, otm={call_otm_pct:.2%}, theta={theta:.3f} | "\
            f"Train Sharpe={sharpe_train.item():.2f}, Train PnL=${total_pnl_train:,.0f} "\
            f"| Val Sharpe={sharpe_val.item():.2f}, Val PnL=${total_pnl_val:,.0f}"
        print(info)
        # === Logging ===
        logging.info(info)

        return sharpe_train.item()

    except Exception as e:
        print(e)
        logging.error(f"[ERROR] np={num_positions}, otm={call_otm_pct:.2%}, theta={theta:.3f} → {str(e)}")
        return -1e6
    
# ===== Run Optimization =====
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

# ===== Log Best Result =====
best = study.best_trial
logging.info("========== BEST TRIAL ==========")
logging.info(f"Params: {best.params}")
logging.info(f"Sharpe: {best.value:.4f}")

print("Best trial:")
print(best)
