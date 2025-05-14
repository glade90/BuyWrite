import optuna
import torch
import pandas as pd
import logging
from BuyWritePortfolioSimulator import BuyWritePortfolioSimulator

# ===== Configure Logging =====
log_file = "buywrite_optimization.log"
logging.basicConfig(
    filename=log_file,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ===== Print GPU Info (Console Only) =====
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())

# ===== Global Config =====
notional_per_trade = 10000

# ===== Objective Function =====
def objective(trial):
    num_positions = trial.suggest_int("num_positions", 2, 10)
    call_otm_pct = trial.suggest_float("call_otm_pct", 0.01, 0.1)
    theta = trial.suggest_float("theta", 0, 0.5)

    vol_summary_df = pd.read_csv("../data/volatility_summary.csv", parse_dates=["Date"])
    price_dir = "../data/stock_prices"

    try:
        sim = BuyWritePortfolioSimulator(
            num_positions=num_positions,
            option_days=21,
            sector_limit=3,
            correlation_threshold=0.85,
            screen_mode='breakout',
            vol_summary_df=vol_summary_df,
            call_otm_pct=call_otm_pct,
            theta=theta,
            alpha=1.0,
            vol_zscore_threshold=0,
            price_dir=price_dir,
            debug=False,
        )
        results_df = sim.run()

        capital_base = notional_per_trade * num_positions
        returns = results_df["PnL_Total_position"] / capital_base

        if len(returns) < 2 or returns.std() == 0:
            return -1e6

        sharpe = (returns.mean() / returns.std()) * torch.sqrt(torch.tensor(12.0))
        total_pnl = results_df["PnL_Total_position"].sum()

        logging.info(
            f"[OK] np={num_positions}, otm={call_otm_pct:.2%}, theta={theta:.3f}, "
            f"Sharpe={sharpe.item():.2f}, Total PnL=${total_pnl:,.2f}"
        )

        return sharpe.item()

    except Exception as e:
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
