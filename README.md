# dealer_gamma_cta_flow_model


# USDJPY Dealer-Gamma & CTA Flow Reflexivity Model

## Overview

This project implements a **flow-aware macro–options model** designed to quantify how  
**dealer gamma positioning**, **CTA/systematic flows**, and **vol-surface structure** interact to create  
**reflexive volatility dynamics** in USDJPY, especially around major macro events  
(CPI, NFP, FOMC, BOJ, BOE, ECB).

The model emulates the type of internal tooling used on FX Options and Macro-Rates desks  
to understand:

- Dealer gamma (long/short)  
- Gamma flip levels  
- Systematic/CTA trend & volatility-targeting behaviour  
- Event-driven vol pops and skew shifts  
- Short-dated convexity dislocations  
- Flow-driven mispricing across RR/BF/ATM tenors  

The script fetches **live market data** from the **Refinitiv Eikon API**, computes the flow regime,  
and outputs probabilities of **event-driven vol-surface reactions**.

---

## Why This Matters (Trader Perspective)

FX Options desks are dominated by **flows**, not just pricing models.

- **Short gamma → hedging amplifies spot moves → skew steepens**  
- **Long gamma → hedging dampens moves → vol compresses**  
- **Strong CTA trend → systematic leverage → convexity in vol**  
- **Macro events + negative gamma → reflexive volatility spikes**

This model quantifies exactly that interaction, providing insight into:

- Pre-event skew dislocations  
- Vol pops vs compressions  
- Gamma-squeeze risk  
- Flow-conditioned vol-RV opportunities  
- Surface behaviour around macro catalysts

The output resembles desk-style vol commentary used in morning meetings.

---

## Model Components

### 1. Vol Surface Fetcher (USDJPY)

Pulls from Eikon:

- ATM volatility  
- 25D Risk Reversal  
- 25D Butterfly  

Across tenors:
- **1W**, **1M**, **3M**

This establishes the **current vol-surface regime**.

---

### 2. Dealer Gamma Exposure Model

Uses CME **6J** FX options chain (strikes + open interest) to approximate:

- **Dealer net gamma exposure**  
- **Gamma flip level** (spot where dealer gamma switches sign)

Logic:

- Dealers **short gamma** → hedge *with* spot → reflexivity, overshoot, vol pop  
- Dealers **long gamma** → hedge *against* spot → dampening, realized vol compression  

Outputs:

- Total net gamma  
- Gamma flip spot  
- Distance of current spot to flip (reflexivity tension)

---

### 3. CTA / Systematic Positioning Model

Creates a synthetic CTA score from:

- 20D momentum  
- 60D momentum  
- 120D momentum  
- Realized volatility (vol-targeting leverage)

Formula:

CTA Score = Trend Signal × Vol-Target Leverage



Interpretation:

- Positive → systematic long USDJPY  
- Negative → systematic short  
- Large absolute → higher flow reflexivity  

---

### 4. Event Shock Model (Reflexivity Engine)

Estimates flow-conditioned probabilities of:

- **ATM vol pop**  
- **RR steepening**  
- **Gamma-squeeze risk**  
- **Convexity shock intensity**

Inputs:

- Dealer gamma  
- CTA score  
- RR skew regime  

A logistic-style function is used as a placeholder, ready to be replaced with  
a fully trained CPI/NFP/FOMC historical model.

---

## Integrated Output

Running the script prints something like:

