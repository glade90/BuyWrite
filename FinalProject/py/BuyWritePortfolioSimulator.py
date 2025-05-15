
import os
import json
import time        
import pandas as pd
from collections import defaultdict
from datetime import datetime
from IPython.display import display
from simulate_buy_write import simulate_buy_write

# This is a simulation of a buy-write portfolio strategy.
# It screens for stocks based on volatility and momentum, enters positions, and manages expirations.
# The code includes functions for screening stocks, calculating momentum, checking correlations,
# entering positions, rolling or replacing positions, and logging trades.

class BuyWritePortfolioSimulator:
    def __init__(self, 
                 vol_summary_df,
                 price_dir,
                 start_date=None,
                 end_date=None,
                 notional_per_trade=10000,
                 num_positions=5,
                 option_days=21,
                 sector_limit=3,
                 correlation_threshold=0.85,
                 screen_mode='breakout',
                 vol_lookback_days = 30,
                 call_otm_pct= 0.05,        
                 vol_zscore_threshold=0,
                 theta=0.0,
                 alpha=1.0,
                 price_df=None,
                 debug=False):
        
        os.makedirs("../logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = f"../logs/buywrite_log.txt"

        with open("../data/sector_map.json") as f:
            self.ticker_sector_map = json.load(f)

        self._price_cache = {}
        self.debug = debug
        self.screen_mode = screen_mode
        self.start_date = start_date if start_date is not None else vol_summary_df["Date"].min()
        self.end_date = end_date if end_date is not None else vol_summary_df["Date"].max()
        self.vol_summary_df = vol_summary_df.copy()
        self.price_dir = price_dir
        self.notional = notional_per_trade
        self.num_positions = num_positions
        self.option_days = option_days
        self.sector_limit = sector_limit
        self.correlation_threshold = correlation_threshold
        self.vol_zscore_threshold = vol_zscore_threshold
        self.sector_counts = defaultdict(int)
        self.call_otm_pct = call_otm_pct
        self.vol_lookback_days = vol_lookback_days
        self.price_df = price_df
        self.theta = theta
        self.alpha = alpha
        self.summary_log = []
        self.active_positions = []  # list of dicts for each open position
        self.trade_log = []         # stores all completed trades
        self.date_index = sorted(vol_summary_df["Date"].unique())

    def _screen_and_rank_universe(self, current_date):
        """
        Apply volatility rank, sector limit, and correlation filter.
        Determine direction based on momentum and apply vol threshold.
        """
        if self.debug:
            print("SCREENING: ", current_date)

        # Only filter relevant rows for the current date
        day_df = self.vol_summary_df[self.vol_summary_df["Date"] == current_date]

        existing_tickers = {p["Ticker"] for p in self.active_positions}

        # Split into active and available universe without .copy()
        day_df_active = day_df[day_df["Ticker"].isin(existing_tickers)]
        day_df = day_df[~day_df["Ticker"].isin(existing_tickers)]

        # Rank
        if self.screen_mode == "breakout":
            ranked = day_df.sort_values("Vol_ZScore", ascending=False)
        else:
            ranked = day_df.sort_values("RollingVol", ascending=False)

        # Combine active tickers first
        ranked = pd.concat([day_df_active, ranked], ignore_index=True)

        selected = []

        for i in range(min(len(ranked), self.num_positions * 10)):
            row = ranked.iloc[i]
            ticker = row["Ticker"]
            sector = self.ticker_sector_map.get(ticker, "Unknown")
            momentum = row.get("Momentum_1M")
            zscore = row.get("Vol_ZScore")

            if pd.isna(zscore) or abs(zscore) < self.vol_zscore_threshold:
                continue

            phi = -1 * self.alpha * self.theta
            direction = "long" if momentum >= phi else "short"

            # Update direction in-place in master DataFrame
            self.vol_summary_df.loc[
                (self.vol_summary_df["Ticker"] == ticker) &
                (self.vol_summary_df["Date"] == current_date),
                "Direction"
            ] = direction

            is_corr, max_corr = self._is_correlated(ticker)
            #if self.sector_counts[sector] >= self.sector_limit:
            #    continue
            if is_corr:
                continue

            # Build new row dict manually instead of modifying original row
            new_row = row.to_dict()
            new_row["Direction"] = direction
            new_row["MaxCorrelation"] = max_corr

            selected.append(new_row)

            if direction == "short":
                self.sector_counts[sector] -= 1
            else:
                self.sector_counts[sector] += 1

            if len(selected) >= self.num_positions:
                break

        return pd.DataFrame(selected)

    def _inject_momentum(self, lookback=21):
        """
        Calculate 1M momentum and merge into self.vol_summary_df.
        """
        all_momentum = []

        for ticker in self.vol_summary_df["Ticker"].unique():
            try:
                df = self._load_price_data(ticker)
                if df is None or "Close" not in df:
                    continue

                df = df.sort_index()
                df["Momentum_1M"] = df["Close"].pct_change(periods=lookback)

                temp = df[["Momentum_1M"]].assign(Ticker=ticker).reset_index()
                all_momentum.append(temp)

            except Exception as e:
                if self.debug:
                    print(f"⚠️ Momentum calc failed for {ticker}: {e}")

        if all_momentum:
            momentum_df = pd.concat(all_momentum, ignore_index=True)
            self.vol_summary_df = self.vol_summary_df.merge(
                momentum_df,
                on=["Ticker", "Date"],
                how="left"
            )


    def _is_correlated(self, candidate_ticker):
        """
        Returns (is_too_correlated, max_correlation, avg_correlation).
        """
        if not self.active_positions or self.correlation_threshold is None:
            return False, None

        active_tickers = [p["Ticker"] for p in self.active_positions]

        candidate_df = self._load_price_data(candidate_ticker)
        if candidate_df is None or candidate_df.empty:
            return True, None

        candidate_series = pd.DataFrame() if candidate_ticker in active_tickers else candidate_df["Close"].rename(candidate_ticker)

        active_dfs = []
        for ticker in active_tickers:
            df = self._load_price_data(ticker)
            if df is not None and not df.empty:
                active_dfs.append(df["Close"].rename(ticker))

        combined_df = pd.concat([candidate_series] + active_dfs, axis=1).dropna()
        if combined_df.shape[0] < 20:
            return True, None

        corr_matrix = combined_df.corr()

        if len(active_tickers) == 1:
            corr_val = corr_matrix.loc[candidate_ticker, active_tickers[0]]
            max_corr = abs(corr_val)

        else:
            corr_vals = corr_matrix.loc[candidate_ticker, active_tickers]
            max_corr = corr_vals.abs().max()

        return max_corr > self.correlation_threshold, max_corr

    
    def _load_price_data(self, ticker):
        if ticker in self._price_cache:
            return self._price_cache[ticker]
        
        if self.price_df is not None:
            df = self.price_df.loc[ticker]
            if df.empty:
                return None
            #df = df.set_index("Date").sort_index()
            self._price_cache[ticker] = df
            return df

        try:
            df = pd.read_csv(f"{self.price_dir}/{ticker}.csv", parse_dates=["Date"])
            df = df.set_index("Date").sort_index()
            self._price_cache[ticker] = df
            return df
        except Exception as e:
            print(f"❌ Failed to load data for {ticker}: {e}")
            return None

    def _enter_position(self, ticker, entry_date):
        df = self._load_price_data(ticker)
        if df is None:
            return None

        start_idx = df.index.get_loc(entry_date) if entry_date in df.index else None
        if start_idx is None or start_idx < self.vol_lookback_days:
            start_idx = self.vol_lookback_days

        # Slice from entry_date - lookback through expiry
        slice_start = df.index[start_idx - self.vol_lookback_days]
        slice_end_idx = start_idx + self.option_days
        if slice_end_idx >= len(df):
            return None
        slice_end = df.index[slice_end_idx]

        trade_window_df = df[(df.index >= slice_start) & (df.index <= slice_end)]

        # Pull direction from vol_summary
        direction_row = self.vol_summary_df[
            (self.vol_summary_df["Ticker"] == ticker) &
            (self.vol_summary_df["Date"] == pd.to_datetime(entry_date))
        ]
        if direction_row.empty or pd.isna(direction_row.iloc[0].get("Direction")):
            return None
        direction = direction_row.iloc[0].get("Direction")

        # Run trade simulation
        trade_df = simulate_buy_write(
            entry_date=entry_date,
            df=trade_window_df,
            notional=self.notional,
            option_days=self.option_days,
            direction=direction,
            vol_lookback_days=self.vol_lookback_days,
            call_otm_pct=self.call_otm_pct,
        )

        if trade_df.empty:
            return None

        trade = trade_df.iloc[0].to_dict()
        trade["Ticker"] = ticker

        vol_row = self.vol_summary_df[
            (self.vol_summary_df["Ticker"] == ticker) &
            (self.vol_summary_df["Date"] == pd.to_datetime(entry_date))
        ]

        if not vol_row.empty:
            trade["Volatility"] = vol_row["RollingVol"].values[0]
            trade["Vol_ZScore"] = vol_row["Vol_ZScore"].values[0]
            trade["Momentum_1M"] = vol_row["Momentum_1M"].values[0]

        else:
            trade["Volatility"] = None
            trade["Vol_ZScore"] = None
            trade["Momentum_1M"] = None

        _, max_corr = self._is_correlated(ticker)
        trade["MaxCorrelation"] = max_corr

        if self.debug:
            print(f"📈 Entering {ticker} on {entry_date.date()} (Dir={direction}, Corr={max_corr})")

        self.active_positions.append(trade)

    def _roll_or_replace(self, expiry_date):
        new_positions = []
        exited = []
        rolled = []
        replaced = []
        new_buys = []
        pnl_this_round = 0.0

        candidates_df = self._screen_and_rank_universe(expiry_date)
        surviving_positions = []

        for pos in self.active_positions:
            if pd.to_datetime(pos["ExpiryDate"]) > expiry_date:
                # Still active, keep it
                surviving_positions.append(pos)
                continue

            # Expired — process it
            pos["MaxCorrelation"] = pos.get("MaxCorrelation")  # Ensure correlation is tracked
            self.trade_log.append(pos)
            pnl_this_round += pos["PnL_Total_position"]
            exited.append(pos["Ticker"])

            if pos["Outcome"] == "Option Assigned":
                replaced.append(pos["Ticker"])
            else:
                rolled.append(pos["Ticker"])
                self._enter_position(pos["Ticker"], expiry_date)

        self.active_positions = surviving_positions  # update only once

        for _, row in candidates_df.iterrows():
            ticker = row["Ticker"]
            if len(self.active_positions) >= self.num_positions:
                break
            self._enter_position(ticker, expiry_date)
            new_buys.append(ticker)

        from datetime import datetime
        summary_lines = [f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 📆 Expiry Date: {expiry_date.date()}"]

        if rolled:
            summary_lines.append("🔁 Rolled Positions:")
            for ticker in rolled:
                try:
                    pos = next(p for p in self.active_positions if p["Ticker"] == ticker)
                    shares = pos["NumberShares"]
                    direction = pos["Direction"]
                    summary_lines.append(f"   - {ticker}: {shares:.4f} shares ({direction})")
                except StopIteration:
                    summary_lines.append(f"   - {ticker}: N/A")

        if replaced:
            summary_lines.append("✅ Called Away (Exited): " + ", ".join(replaced))

        if new_buys:
            summary_lines.append("🛒 New Positions:")
            for ticker in new_buys:
                try:
                    pos = next(p for p in self.active_positions if p["Ticker"] == ticker)
                    shares = pos["NumberShares"]
                    direction = pos["Direction"]
                    summary_lines.append(f"   - {ticker}: {shares:.4f} shares ({direction})")
                except StopIteration:
                    summary_lines.append(f"   - {ticker}: N/A")

        summary_lines.append(f"💰 Period PnL: ${pnl_this_round:,.2f}")
        summary_lines.append("─" * 60)
        summary_lines.append(f"🏛 Sector Mix: {dict(self.sector_counts)}")

        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n\n")


    def run(self):
        import time
        import numpy as np

        start_time = time.time()
        print("🚦 Starting simulation...")

        self._inject_momentum()
        capital_base = self.num_positions * self.notional  # Defined once at the top

        valid_dates = self.vol_summary_df.dropna(subset=["Vol_ZScore", "Momentum_1M"]).groupby("Date").size()
        valid_dates = valid_dates[self.start_date:self.end_date]
        
        first_valid_date = valid_dates[valid_dates >= self.num_positions].index[0]

        if self.debug:
            print(f"🚀 Starting simulation on {first_valid_date.date()} ({self.screen_mode} mode)")

        candidates_df = self._screen_and_rank_universe(first_valid_date)
        for _, row in candidates_df.head(self.num_positions).iterrows():
            self._enter_position(row["Ticker"], row["Date"])

        expiry_schedule = sorted({pd.to_datetime(p["ExpiryDate"]) for p in self.active_positions})
        last_reported_year = None

        while expiry_schedule:
            next_expiry = expiry_schedule.pop(0)

            # Yearly P&L reporting
            year = next_expiry.year
            if last_reported_year is None:
                last_reported_year = year

            if year != last_reported_year:
                prior_year = last_reported_year
                prior_trades = [
                    p for p in self.trade_log 
                    if pd.to_datetime(p["ExpiryDate"]).year == prior_year
                ]
                prior_pnl = sum(p["PnL_Total_position"] for p in prior_trades)
                return_pct = (prior_pnl / capital_base) * 100 if capital_base > 0 else float('nan')
                print(f"    📅 {prior_year} Total P&L: ${prior_pnl:,.2f} ({return_pct:.2f}% return on ${capital_base:,.0f} capital)")
                last_reported_year = year

            self._roll_or_replace(next_expiry)
            new_expiries = {pd.to_datetime(p["ExpiryDate"]) for p in self.active_positions}
            expiry_schedule = sorted(set(expiry_schedule).union(new_expiries))

            if (expiry_schedule == []) or (expiry_schedule[0] > pd.to_datetime(self.end_date)):
                print(f"    📅 Time interval ended on {self.end_date}")
                break

            if self.debug:
                print(f"\n🔄 Processing expiry: {next_expiry.date()}")

        if self.debug:
            if self.active_positions:
                print("\n📌 Final active positions:")
                for p in self.active_positions:
                    print(f"   - {p['Ticker']}: Entry={p['EntryDate']}, Expiry={p['ExpiryDate']}")
                first_expiry = min(pd.to_datetime(p["ExpiryDate"]) for p in self.active_positions)
                print(f"🔔 First expiry scheduled: {first_expiry}")
            else:
                print("\n📭 No active positions at end of simulation.")

        for pos in self.trade_log:
            pos["Sector"] = self.ticker_sector_map.get(pos["Ticker"], "Unknown")

        elapsed = time.time() - start_time
        total_pnl = sum(p["PnL_Total_position"] for p in self.trade_log)

        # Estimate per-position return stream
        returns = [p["PnL_Total_position"] / capital_base for p in self.trade_log if capital_base > 0]

        if len(returns) >= 2:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns, ddof=1)
            sharpe = (mean_ret / std_ret) * np.sqrt(12) if std_ret > 0 else float('nan')
            print(f"📈 Sharpe Ratio: {sharpe:.2f}")
        else:
            print("📈 Sharpe Ratio: N/A (not enough data)")

        print(f"✅ Simulation complete. Time taken: {elapsed:.2f} seconds.")
        print(f"💰 Total P&L: ${total_pnl:,.2f}")
        print(f"💵 Total P&L: {100 * total_pnl / capital_base:.2f}%")
        if isinstance(self.trade_log, list):
            self.trade_log = pd.DataFrame(self.trade_log)
        self.trade_log["CumulativePnL"] = self.trade_log["PnL_Total_position"].cumsum()

        return pd.DataFrame(self.trade_log)


