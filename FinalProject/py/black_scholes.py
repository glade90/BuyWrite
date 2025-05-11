from scipy.stats import norm
import numpy as np

def black_scholes_call_price(S, K, T, r, sigma):
    """
    Black-Scholes formula for a European call option.

    Parameters:
    - S: Spot price (entry price of the stock)
    - K: Strike price (OTM call)
    - T: Time to maturity (in years)
    - r: Risk-free rate (e.g., 0.02)
    - sigma: Volatility (annualized)

    Returns:
    - Call option price (fair value)
    """
    if T <= 0 or sigma <= 0:
        return 0.0  # Handle edge cases
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put_price(S, K, T, r, sigma):
    """Calculates European put price using call price and put-call parity."""
    call_price = black_scholes_call_price(S, K, T, r, sigma)
    return call_price + K * np.exp(-r * T) - S
