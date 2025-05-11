
import os
import json
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
                 notional_per_trade=10000,
                 num_positions=5,
                 option_days=21,
                 sector_limit=3,
                 correlation_threshold=0.85,
                 screen_mode='breakout',
                 vol_lookback_days = 30,
                 call_otm_pct= 0.05,        
                 vol_zscore_threshold=0,
                 tau=0.0,
                 alpha=1.0,
                 debug=False):
        
        os.makedirs("../logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file_path = f"../logs/buywrite_log_{timestamp}.txt"

        with open("../data/sector_map.json") as f:
            self.ticker_sector_map = json.load(f)

        self.debug = debug
        self.screen_mode = screen_mode
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
        self.tau = tau
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
            print("SCREEENING: ", current_date)
        df = self.vol_summary_df.copy()
        day_df = df[df["Date"] == current_date].copy()

        existing_tickers = [p["Ticker"] for p in self.active_positions]
        
        day_df_active = day_df[day_df["Ticker"].isin(existing_tickers)]
        day_df = day_df[~day_df["Ticker"].isin(existing_tickers)]
        

        if self.screen_mode == "breakout":
            ranked = day_df.sort_values("Vol_ZScore", ascending=False)
        else:
            ranked = day_df.sort_values("RollingVol", ascending=False)

        ranked = pd.concat([day_df_active, ranked], ignore_index=True)

        sector_counts = defaultdict(int)
        
        selected = []

        for _, row in ranked.head(self.num_positions * 2).iterrows():
            ticker = row["Ticker"]
            sector = self.ticker_sector_map.get(ticker, "Unknown")
            momentum = row.get("Momentum_1M")
            zscore = row.get("Vol_ZScore")
            if zscore is None or abs(zscore) < self.vol_zscore_threshold:  # use threshold = 0 for now
                continue

            phi = -1 * self.alpha * self.tau

            direction = "long" if momentum >= phi else "short"
            
            row["Direction"] = direction
            self.vol_summary_df.loc[
                (self.vol_summary_df["Ticker"] == ticker) &
                (self.vol_summary_df["Date"] == current_date),
                "Direction"
            ] = direction
            is_corr, max_corr = self._is_correlated(ticker)
            
            #active_tickers = [p["Ticker"] for p in self.active_positions]
            #if ticker in active_tickers:
            #    continue

            if sector_counts[sector] >= self.sector_limit:
                continue
            if is_corr:
                continue
            row["MaxCorrelation"] = max_corr

            selected.append(row)
            if direction == "short":
                sector_counts[sector] -= 1
            else:
                sector_counts[sector] += 1
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

                temp = df[["Momentum_1M"]].copy()
                temp["Ticker"] = ticker
                temp.reset_index(inplace=True)
                all_momentum.append(temp)
            except Exception as e:
                print(f"⚠️ Momentum calc failed for {ticker}: {e}")

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
        try:
            df = pd.read_csv(f"{self.price_dir}/{ticker}.csv", parse_dates=["Date"])
            df = df.set_index("Date").sort_index()
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
            return None  # not enough history

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
        
        for pos in self.active_positions.copy():
            self.active_positions.remove(pos)
            if pd.to_datetime(pos["ExpiryDate"]) < expiry_date:
                continue

            pos["MaxCorrelation"] = pos.get("MaxCorrelation")  # Ensure correlation is tracked
            self.trade_log.append(pos)
            pnl_this_round += pos["PnL_Total_position"]
            exited.append(pos["Ticker"])

            if pos["Outcome"] == "Option Assigned":
                replaced.append(pos["Ticker"])
            else:
                rolled.append(pos["Ticker"])
                self._enter_position(pos["Ticker"], expiry_date)


        for _, row in candidates_df.iterrows():
            ticker = row["Ticker"]
            if len(self.active_positions) >= self.num_positions:
                break
            self._enter_position(ticker, expiry_date)
            new_buys.append(ticker)

        from datetime import datetime
        summary_lines = []
        summary_lines.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📆 Expiry Date: {expiry_date.date()}")

        if rolled:
            summary_lines.append("🔁 Rolled Positions:")
            for ticker in rolled:
                try:
                    shares = next(p for p in self.active_positions if p["Ticker"] == ticker)["NumberShares"]
                    summary_lines.append(f"   - {ticker}: {shares:.4f} shares")
                except StopIteration:
                    summary_lines.append(f"   - {ticker}: N/A")

        if replaced:
            summary_lines.append("✅ Called Away (Exited): " + ", ".join(replaced))

        if new_buys:
            summary_lines.append("🛒 New Positions:")
            for ticker in new_buys:
                try:
                    shares = next(p for p in self.active_positions if p["Ticker"] == ticker)["NumberShares"]
                    summary_lines.append(f"   - {ticker}: {shares:.4f} shares")
                except StopIteration:
                    summary_lines.append(f"   - {ticker}: N/A")

        summary_lines.append(f"💰 Period PnL: ${pnl_this_round:,.2f}")

        correlations = [p.get("MaxCorrelation") for p in self.active_positions if p.get("MaxCorrelation") is not None]
        #if correlations:
            #avg_corr = sum(correlations) / len(correlations)
            #summary_lines.append(f"📊 Avg Correlation of Active Portfolio: {avg_corr:.2f}")

        summary_lines.append("─" * 60)
        summary_lines.append(f"🏛 Sector Mix: {dict(self.sector_counts)}")

        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write("\n".join(summary_lines) + "\n\n")

    def run(self):
        self._inject_momentum()
        valid_dates = self.vol_summary_df.dropna(subset=["Vol_ZScore", "Momentum_1M"]).groupby("Date").size()

        first_valid_date = valid_dates[valid_dates >= self.num_positions].index[0]
        if self.debug:
            print(f"🚀 Starting simulation on {first_valid_date.date()} ({self.screen_mode} mode)")

        candidates_df = self._screen_and_rank_universe(first_valid_date)
        for _, row in candidates_df.head(self.num_positions).iterrows():
            self._enter_position(row["Ticker"], row["Date"])

        expiry_schedule = sorted({pd.to_datetime(p["ExpiryDate"]) for p in self.active_positions})
        
        #print(f"expiry_schedule{expiry_schedule}")
        #print(f"Active positions: {self.active_positions}")

        while expiry_schedule:
            next_expiry = expiry_schedule.pop(0)
            self._roll_or_replace(next_expiry)
            new_expiries = {pd.to_datetime(p["ExpiryDate"]) for p in self.active_positions}
            expiry_schedule = sorted(set(expiry_schedule).union(new_expiries))
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

        return pd.DataFrame(self.trade_log)