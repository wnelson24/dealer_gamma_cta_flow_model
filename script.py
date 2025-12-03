"""
USDJPY Dealer-Gamma + CTA Positioning Reflexivity Model
Author: William Nelson
Purpose: Provide a flow-aware prediction of vol-surface moves around macro events.

Requirements:
- Refinitiv Eikon (Eikon Data API)
- numpy, pandas, scipy, scikit-learn
"""

###############################################################
# 0. IMPORTS
###############################################################

import eikon as ek
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


###############################################################
# 1. EIKON CONNECTION
###############################################################

# >>> INSERT YOUR EIKON APP KEY HERE <<<
ek.set_app_key("YOUR_APP_KEY_HERE")


###############################################################
# 2. BLACK-SCHOLES GREEKS
###############################################################

def bs_gamma(S, K, T, vol, r=0.0):
    """Black–Scholes Gamma for FX options."""
    if K <= 0 or vol <= 0 or T <= 0:
        return 0.0

    d1 = (np.log(S/K) + (r + 0.5 * vol**2)*T) / (vol * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    return gamma


###############################################################
# 3. VOL SURFACE FETCHER (USDJPY)
###############################################################

def fetch_usdjpy_vol_surface():
    """
    Fetch ATM, RR, BF vols for USDJPY from Eikon.
    Tenors: 1W, 1M, 3M.
    """

    fields = [
        "TR.ATMVol", 
        "TR.RiskReversal",
        "TR.Butterfly"
    ]

    tenors = {
        "1W":  "JPY1WK=",
        "1M":  "JPY1MO=",
        "3M":  "JPY3MO="
    }

    surface = {}

    for tenor, ric in tenors.items():
        df, err = ek.get_data(ric, fields)
        if err:
            raise ValueError(f"Eikon error for {ric}: {err}")
        
        surface[tenor] = {
            "ATM": float(df["ATM Volatility"][0])/100,
            "RR":  float(df["Risk Reversal"][0])/100,
            "BF":  float(df["Butterfly"][0])/100
        }

    # Spot
    spot_df, err = ek.get_data("USDJPY=", ["TRDPRC_1"])
    spot = float(spot_df["TRDPRC_1"][0])

    return surface, spot


###############################################################
# 4. CME OPTION-CHAIN PULL (6J)
###############################################################

def fetch_6j_option_chain():
    """
    Fetch CME JPY/USD FX options (6J).
    Note: 6J options are on JPY per USD, inverse of USDJPY.
    We'll flip strikes accordingly.
    """
    chain = "0#6J+O"
    fields = ["TR.StrikePrice", "TR.PutCall", "TR.ExchangeOpenInterest"]

    df, err = ek.get_data(chain, fields)
    if err:
        raise ValueError(f"Eikon option chain error: {err}")

    df = df.dropna()
    return df


###############################################################
# 5. DEALER GAMMA + FLIP CALCULATION
###############################################################

def estimate_dealer_gamma_usdjpy(spot, vol_1m, option_chain):
    """
    Approximate dealer gamma exposure using CME 6J options.
    Convert 6J strikes (JPY per USD) to USDJPY strike levels.
    """

    gammas = []

    for _, row in option_chain.iterrows():
        K_6J = row["Strike Price"]
        if K_6J <= 0:
            continue

        # CME 6J options are quoted as JPY per USD => USDJPY strike = K_6J
        K = K_6J

        T = 30/365
        vol = vol_1m

        gamma = bs_gamma(spot, K, T, vol)
        exposure = gamma * row["Exchange Open Interest"]
        gammas.append(exposure)

    total_gamma = np.sum(gammas)
    return total_gamma


def find_gamma_flip_level(spot, vol_1m, option_chain, iterations=40):
    """
    Solve for the spot where net gamma exposure = 0.
    Simple binary search.
    """

    def g(S):
        return estimate_dealer_gamma_usdjpy(S, vol_1m, option_chain)

    lo, hi = spot * 0.9, spot * 1.1
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


###############################################################
# 6. CTA POSITIONING MODEL
###############################################################

def fetch_cta_inputs_usdjpy():
    """
    Fetch momentum & realized vol proxies for CTA model.
    """
    fields = [
        "TR.Momentum20D",
        "TR.Momentum60D",
        "TR.Momentum120D",
        "TR.RealizedVol"
    ]

    df, err = ek.get_data("USDJPY=", fields)
    if err:
        raise ValueError(f"Eikon CTA fetch error: {err}")

    data = df.iloc[0].astype(float)
    return data


def compute_cta_signal(momentum20, momentum60, momentum120, realized_vol, target_vol=0.10):
    """
    Simple weighted trend + vol-target leverage model.
    """
    trend = (
        0.4*momentum20 +
        0.3*momentum60 +
        0.3*momentum120
    )

    # Vol-targeting leverage (inverse of realized vol)
    leverage = target_vol / realized_vol if realized_vol > 0 else 1.0

    return trend * leverage


###############################################################
# 7. EVENT SHOCK MODEL (SCAFFOLD)
###############################################################

def dummy_event_model(dealer_gamma, cta_signal, rr_skew):
    """
    Placeholder logistic model showing how you would integrate
    dealer gamma + CTA + skew into predicted vol-surface reaction.
    Replace with a real trained model later.
    """

    # Toy “probability” formulas:
    atm_pop_prob = 1 / (1 + np.exp(-(0.7*cta_signal - 0.4*dealer_gamma + rr_skew)))
    rr_steep_prob = 1 / (1 + np.exp(-(0.5*cta_signal + 0.3*rr_skew)))
    gamma_squeeze = abs(dealer_gamma) / (abs(dealer_gamma) + 5_000)  # bounded 0–1

    return {
        "atm_pop_prob": float(atm_pop_prob),
        "rr_steep_prob": float(rr_steep_prob),
        "gamma_squeeze_risk": float(gamma_squeeze)
    }


###############################################################
# 8. INTEGRATED REFLEXIVITY MODEL
###############################################################

def run_reflexivity_model():
    print("\n--- USDJPY REFLEXIVITY MODEL ---\n")

    # -----------------------------
    # 1. Vol Surface + Spot
    # -----------------------------
    surface, spot = fetch_usdjpy_vol_surface()
    vol_1m = surface["1M"]["ATM"]
    skew_1m = surface["1M"]["RR"]

    # -----------------------------
    # 2. Option Chain
    # -----------------------------
    chain = fetch_6j_option_chain()

    # -----------------------------
    # 3. Dealer Gamma & Flip
    # -----------------------------
    dealer_gamma = estimate_dealer_gamma_usdjpy(spot, vol_1m, chain)
    flip = find_gamma_flip_level(spot, vol_1m, chain)

    # -----------------------------
    # 4. CTA Positioning
    # -----------------------------
    cta_data = fetch_cta_inputs_usdjpy()
    cta_sig = compute_cta_signal(
        cta_data["Momentum 20 Day"],
        cta_data["Momentum 60 Day"],
        cta_data["Momentum 120 Day"],
        cta_data["Realized Volatility"]
    )

    # -----------------------------
    # 5. Event Shock Model
    # -----------------------------
    shock = dummy_event_model(dealer_gamma, cta_sig, skew_1m)

    # -----------------------------
    # 6. Pretty Output
    # -----------------------------
    print(f"Spot: {spot:.2f}")
    print(f"1M ATM Vol: {vol_1m:.2%}")
    print(f"1M RR Skew: {skew_1m:.2%}\n")

    print("=== DEALER FLOW ===")
    print(f"Dealer Gamma Exposure: {dealer_gamma:,.0f}")
    print(f"Gamma Flip Level: {flip:.2f}")
    print(f"Distance to Flip: {spot - flip:.2f}\n")

    print("=== CTA POSITIONING ===")
    print(f"CTA Trend Signal: {cta_sig:.3f}\n")

    print("=== EVENT REACTION PROBABILITIES ===")
    print(f"ATM Vol Pop Probability: {shock['atm_pop_prob']:.1%}")
    print(f"RR Steepening Probability: {shock['rr_steep_prob']:.1%}")
    print(f"Gamma Squeeze Risk: {shock['gamma_squeeze_risk']:.1%}\n")

    print("--- END OF MODEL ---\n")


###############################################################
# 9. MAIN
###############################################################

if __name__ == "__main__":
    run_reflexivity_model()
