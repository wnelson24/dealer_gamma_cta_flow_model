"""
USDJPY Dealer-Gamma + CTA Positioning Reflexivity Model (RDP Version)
Author: William Nelson
Purpose: Provide a flow-aware prediction of vol-surface moves around macro events.

Requirements:
- Refinitiv Workspace (RDP API)
- refinitiv-dataplatform
- numpy, pandas, scipy, scikit-learn
"""

###############################################################
# 0. IMPORTS
###############################################################

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from refinitiv.dataplatform import Sessions, Delivery


###############################################################
# 1. RDP AUTHENTICATION
###############################################################

# Workspace AppKey (OAuth)
Sessions.get_session(
    session_type=Sessions.SessionTypeEnum.DESKTOP,
    app_key="YOUR_RDP_APP_KEY"
).open()


###############################################################
# 2. BLACK–SCHOLES GAMMA
###############################################################

def bs_gamma(S, K, T, vol, r=0.0):
    if K <= 0 or vol <= 0 or T <= 0:
        return 0.0

    d1 = (np.log(S/K) + (r + 0.5 * vol**2)*T) / (vol * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    return gamma


###############################################################
# 3. VOL SURFACE FETCHER (RDP VERSION)
###############################################################

def fetch_usdjpy_vol_surface():
    """
    Fetch ATM, RR, BF vols for USDJPY from RDP.
    Tenors: 1W, 1M, 3M.
    """

    fields = ["TR.ATMVol", "TR.RiskReversal", "TR.Butterfly"]

    rics = {
        "1W":  "JPY1WK=",
        "1M":  "JPY1MO=",
        "3M":  "JPY3MO="
    }

    surface = {}

    for tenor, ric in rics.items():
        df = Delivery.get_data(universe=[ric], fields=fields).data.df

        surface[tenor] = {
            "ATM": float(df["TR.ATMVol"].iloc[0]) / 100.0,
            "RR":  float(df["TR.RiskReversal"].iloc[0]) / 100.0,
            "BF":  float(df["TR.Butterfly"].iloc[0]) / 100.0
        }

    # Spot
    spot_df = Delivery.get_snapshot(universe=["USDJPY="]).data.df
    spot = float(spot_df["MID_PRICE"].iloc[0])

    return surface, spot


###############################################################
# 4. CME OPTION-CHAIN FETCH (RDP VERSION)
###############################################################

def fetch_6j_option_chain():
    chain = "0#6J+O"

    fields = [
        "TR.StrikePrice",
        "TR.PutCall",
        "TR.ExchangeOpenInterest"
    ]

    df = Delivery.get_data(universe=[chain], fields=fields).data.df
    df = df.dropna()

    return df


###############################################################
# 5. DEALER GAMMA & GAMMA FLIP
###############################################################

def estimate_dealer_gamma_usdjpy(spot, vol_1m, option_chain):
    gammas = []

    for _, row in option_chain.iterrows():
        K = row["TR.StrikePrice"]  # JPY per USD, same as USDJPY

        T = 30 / 365
        vol = vol_1m

        gamma = bs_gamma(spot, K, T, vol)
        oi = row["TR.ExchangeOpenInterest"]

        gammas.append(gamma * oi)

    return np.sum(gammas)


def find_gamma_flip_level(spot, vol_1m, option_chain, iterations=40):

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
# 6. CTA INPUTS & SIGNAL
###############################################################

def fetch_cta_inputs_usdjpy():

    fields = [
        "TR.Momentum20D",
        "TR.Momentum60D",
        "TR.Momentum120D",
        "TR.RealizedVol"
    ]

    df = Delivery.get_data(
        universe=["USDJPY="],
        fields=fields
    ).data.df

    return df.iloc[0].astype(float)


def compute_cta_signal(m20, m60, m120, realized_vol, target_vol=0.10):

    trend = (
        0.4 * m20 +
        0.3 * m60 +
        0.3 * m120
    )

    leverage = target_vol / realized_vol if realized_vol > 0 else 1.0

    return trend * leverage


###############################################################
# 7. EVENT SHOCK MODEL (PLACEHOLDER)
###############################################################

def dummy_event_model(dealer_gamma, cta_signal, rr_skew):

    atm_pop_prob = 1 / (1 + np.exp(-(0.7*cta_signal - 0.4*dealer_gamma + rr_skew)))
    rr_steep_prob = 1 / (1 + np.exp(-(0.5*cta_signal + 0.3*rr_skew)))
    gamma_squeeze = abs(dealer_gamma) / (abs(dealer_gamma) + 5000)

    return {
        "atm_pop_prob": float(atm_pop_prob),
        "rr_steep_prob": float(rr_steep_prob),
        "gamma_squeeze_risk": float(gamma_squeeze)
    }


###############################################################
# 8. INTEGRATED REFLEXIVITY MODEL
###############################################################

def run_reflexivity_model():
    print("\n--- USDJPY REFLEXIVITY MODEL (RDP VERSION) ---\n")

    # 1. Vol Surface
    surface, spot = fetch_usdjpy_vol_surface()
    vol_1m = surface["1M"]["ATM"]
    skew_1m = surface["1M"]["RR"]

    # 2. Option Chain
    chain = fetch_6j_option_chain()

    # 3. Dealer gamma and flip
    dealer_gamma = estimate_dealer_gamma_usdjpy(spot, vol_1m, chain)
    flip = find_gamma_flip_level(spot, vol_1m, chain)

    # 4. CTA Positioning
    cta = fetch_cta_inputs_usdjpy()
    cta_sig = compute_cta_signal(
        cta["TR.Momentum20D"],
        cta["TR.Momentum60D"],
        cta["TR.Momentum120D"],
        cta["TR.RealizedVol"]
    )

    # 5. Event Shock
    shock = dummy_event_model(dealer_gamma, cta_sig, skew_1m)

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
