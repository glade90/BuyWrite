import numpy as np
import pandas as pd
from scipy.stats import norm
from math import exp, log, sqrt
from datetime import datetime, timedelta
from typing import Union
from black_scholes import black_scholes_call_price, black_scholes_put_price
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def simulate_buy_write(entry_date,df, notional, call_otm_pct=0.02, option_days=21, vol_lookback_days=30, direction="long"):
    """
    Simulates a buy-write (or short-write) strategy on a single stock DataFrame.

    Parameters:
    - df: DataFrame with Date index and a 'Close' column
    - notional: dollar amount per trade
    - call_otm_pct: OTM % for call or put strike
    - option_days: holding period
    - vol_lookback_days: days for historical vol estimate
    - direction: 'long' (buy+call) or 'short' (short+put)

    Returns:
    - DataFrame of trade-level results
    """
    trades = []
    df = df.sort_index()
    returns = df["Close"].pct_change().rolling(vol_lookback_days).std() * np.sqrt(252)
    returns = returns.dropna()
    returns = returns[returns.index >= entry_date]
    df = df[df.index >= entry_date]

    i = 0
    while i < len(df) - option_days:
        T = option_days / 252
        r = 0.02
        sigma = returns.iloc[i]

        entry_date = df.index[i]
        expiry_date = df.index[i + option_days]

        entry_price = df["Close"].iloc[i]
        spot_price_close = df["Close"].iloc[i + option_days]
        strike = entry_price * (1 + call_otm_pct if direction == "long" else 1 - call_otm_pct)

        if np.isnan(sigma) or sigma <= 0:
            premium = 0.0
        else:
            if direction == "long":
                premium = black_scholes_call_price(entry_price, strike, T, r, sigma)
            else:
                premium = black_scholes_put_price(entry_price, strike, T, r, sigma)

        quantity = notional / entry_price

        if direction == "long":
            if spot_price_close > strike:
                exit_price = strike
                outcome = "Option Assigned"
                pnl_stock = strike - entry_price
                pnl_unrealized = 0
                pnl_realized = pnl_stock + premium
            else:
                exit_price = spot_price_close
                outcome = "Call Expired"
                pnl_stock = spot_price_close - entry_price
                pnl_unrealized = pnl_stock
                pnl_realized = premium
        else:
            if spot_price_close < strike:
                exit_price = strike
                outcome = "Option Assigned"
                pnl_stock = entry_price - strike
                pnl_unrealized = 0
                pnl_realized = pnl_stock + premium
            else:
                exit_price = spot_price_close
                outcome = "Put Expired"
                pnl_stock = entry_price - spot_price_close
                pnl_unrealized = pnl_stock
                pnl_realized = premium

        pnl_option = premium
        option_type = "Call" if direction == "long" else "Put"
        pnl_total = pnl_option + pnl_stock
        pnl_percent = pnl_total / notional

        trades.append({
            "NumberShares": quantity,
            "Direction": direction,
            "EntryDate": entry_date,
            "EntryPrice": entry_price,
            "ExpiryDate": expiry_date,
            "ExitPrice": exit_price,
            "SpotPrice_Close": spot_price_close,
            "Strike": strike,
            "Premium": premium,
            "Outcome": outcome,
            "OptionType": option_type,
            "PnL_Option": pnl_option,
            "PnL_Stock": pnl_stock,
            "PnL_Realized": pnl_realized,
            "PnL_Unrealized": pnl_unrealized,
            "PnL_Total": pnl_total,
            "PnL_Percent": pnl_percent,
            "PnL_Option_Position": pnl_option * quantity,
            "PnL_Stock_Position": pnl_stock * quantity,
            "PnL_Realized_Position": pnl_realized * quantity,
            "PnL_Unrealized_Position": pnl_unrealized * quantity,
            "PnL_Total_position": pnl_total * quantity,
        })

        i += option_days

    return pd.DataFrame(trades)
