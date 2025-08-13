
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import math

st.set_page_config(page_title="LEAP + Monthly Covered Call Toolkit", layout="wide")

# ---------- numeric-safe helpers ----------
def to_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except Exception:
        return np.nan

def is_num(x):
    try:
        xf = float(x)
        return np.isfinite(xf)
    except Exception:
        return False

def get_scalar_last(series: pd.Series):
    try:
        return to_float(series.iloc[-1])
    except Exception:
        return np.nan

# ---------- indicators ----------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def black_scholes_delta(S, K, T, r, sigma, call=True):
    sigma = max(to_float(sigma) if is_num(sigma) else 1e-8, 1e-8)
    S = to_float(S); K = to_float(K); T = to_float(T); r = to_float(r)
    if not all(map(is_num, [S, K, T, r])):
        return np.nan
    if T <= 0:
        return 1.0 if (call and S > K) else ( -1.0 if (not call and S < K) else 0.0 )
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) if call else norm_cdf(d1) - 1.0

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def bollinger(series: pd.Series, window=20, num_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return ma, upper, lower

def iv_rank_proxy(close: pd.Series):
    if len(close) < 20:
        return np.nan
    mid, up, low = bollinger(close, 20, 2)
    width = (up - low) / mid
    width = width.dropna()
    if len(width) < 20:
        return np.nan
    latest = width.iloc[-1]
    pct_rank = (width <= latest).sum() / len(width) * 100.0
    return to_float(pct_rank)

# ---------- earnings ----------
def next_earnings_date(tkr: yf.Ticker):
    try:
        ed = tkr.get_earnings_dates(limit=6)
        if ed is not None and not ed.empty:
            date_col = None
            for c in ed.columns:
                if "Date" in c:
                    date_col = c; break
            if date_col:
                future = ed[ed[date_col] >= pd.Timestamp.utcnow()]
                if not future.empty:
                    return pd.to_datetime(future[date_col].iloc[0], errors="coerce")
    except Exception:
        pass
    try:
        cal = tkr.calendar
        if cal is not None and not cal.empty:
            for cand in ["Earnings Date", "EarningsDate", "Next Earnings Date"]:
                if cand in cal.index:
                    val = cal.loc[cand].values[0]
                    return pd.to_datetime(val, errors="coerce")
    except Exception:
        pass
    return None

def earnings_within(days: int, tkr: yf.Ticker) -> bool:
    nd = next_earnings_date(tkr)
    if nd is None or pd.isna(nd):
        return False
    delta = (nd - pd.Timestamp.utcnow()).days
    return (delta >= 0) and (delta <= int(days))

# ---------- options helpers ----------
def choose_option_strike_from_chain(chain_df, target_delta_low, target_delta_high, S=None, r=0.04, T_years=0.5):
    if chain_df is None or chain_df.empty:
        return None
    df = chain_df.copy()
    if "contractSymbol" in df.columns:
        df = df[df["contractSymbol"].astype(str).str.contains("C")]
    if df.empty:
        return None
    S = to_float(S); r = to_float(r); T_years = to_float(T_years)
    results = []
    tmid = to_float((target_delta_low + target_delta_high) / 2.0)
    for _, row in df.iterrows():
        K = to_float(row.get("strike"))
        iv = to_float(row.get("impliedVolatility"))
        if not (is_num(K) and is_num(iv) and iv > 0):
            continue
        exp = row.get("expiration", None)
        if pd.notna(exp):
            if not isinstance(exp, pd.Timestamp):
                exp = pd.to_datetime(exp, errors="coerce")
            T = max(((exp - pd.Timestamp.utcnow()).days) / 365.0, 1/365)
        else:
            T = T_years if is_num(T_years) else 0.5
        delta = black_scholes_delta(S, K, T, r, iv, call=True)
        if not is_num(delta):
            continue
        pref = 0 if (target_delta_low <= delta <= target_delta_high) else 1
        results.append((pref, abs(delta - tmid), delta, K, iv, exp))
    if not results:
        return None
    results.sort(key=lambda x: (x[0], x[1]))
    _, _, d_pick, k_pick, iv_pick, exp_pick = results[0]
    return {"strike": k_pick, "delta": d_pick, "iv": iv_pick, "expiration": exp_pick}

def pick_expiration(ticker_obj: yf.Ticker, min_days: int, max_days: int):
    try:
        exps = ticker_obj.options
        if not exps:
            return None
        today = pd.Timestamp.utcnow().normalize()
        cands = []
        for e in exps:
            et = pd.to_datetime(e, errors="coerce")
            if pd.isna(et): 
                continue
            dte = (et - today).days
            if min_days <= dte <= max_days:
                cands.append((dte, e))
        if not cands:
            beyond = []
            for e in exps:
                et = pd.to_datetime(e, errors="coerce")
                if pd.isna(et): 
                    continue
                dte = (et - today).days
                if dte >= min_days:
                    beyond.append((dte, e))
            if not beyond:
                return None
            cands = beyond
        cands.sort(key=lambda x: x[0])
        return cands[0][1]
    except Exception:
        return None

def get_chain(ticker_obj: yf.Ticker, expiration: str):
    try:
        ch = ticker_obj.option_chain(expiration)
        calls = ch.calls.copy(); calls["expiration"] = pd.to_datetime(expiration, errors="coerce")
        puts = ch.puts.copy();  puts["expiration"] = pd.to_datetime(expiration, errors="coerce")
        return pd.concat([calls, puts], ignore_index=True)
    except Exception:
        return None

# ---------- UI ----------
st.title("LEAP + Monthly Covered Call Scanner (Patched)")

st.sidebar.header("Universe & Filters")
default_tickers = ['AAPL','MSFT','JNJ','PFE','PG','KO','PEP','NVDA','AVGO','ADBE','AMD','CRM','NOW','UNH','MRK','ABT','ABBV','LLY','TMO','DHR','V','MA','HD','COST','WMT','XOM','CVX','JPM','BAC']
tickers_text = st.sidebar.text_area("Tickers", value=",".join(default_tickers), height=120)

pe_max = st.sidebar.number_input("Max P/E", 1.0, 200.0, 25.0, 0.5)
div_min = st.sidebar.number_input("Min Dividend Yield %", 0.0, 15.0, 2.0, 0.1)
rev_yoy_min = st.sidebar.number_input("Min Revenue Growth YoY %", -100.0, 200.0, 5.0, 0.5)

rsi_min = st.sidebar.number_input("RSI min", 0.0, 100.0, 40.0, 1.0)
rsi_max = st.sidebar.number_input("RSI max", 0.0, 100.0, 60.0, 1.0)

require_above_200dma = st.sidebar.checkbox("Require price above 200-DMA", True)
require_bb_mid_to_upper = st.sidebar.checkbox("Require price between middle & upper Bollinger band", True)

st.sidebar.markdown("---")
st.sidebar.subheader("Earnings Exclusion")
exclude_earnings = st.sidebar.checkbox("Exclude earnings within N days", True)
earnings_window = st.sidebar.number_input("N (days)", 1, 60, 10, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Trade Construction")
target_long_delta_low = st.sidebar.slider("LEAP target delta low", 0.50, 0.90, 0.70, 0.01)
target_long_delta_high = st.sidebar.slider("LEAP target delta high", 0.55, 0.95, 0.80, 0.01)
long_min_months = st.sidebar.slider("LEAP min months to expiry", 6, 18, 6, 1)
long_max_months = st.sidebar.slider("LEAP max months to expiry", 6, 24, 12, 1)
short_days_min = st.sidebar.slider("Short call min DTE", 14, 90, 30, 1)
short_days_max = st.sidebar.slider("Short call max DTE", 20, 120, 45, 1)
short_target_delta = st.sidebar.slider("Short call target delta", 0.15, 0.40, 0.28, 0.01)
risk_free_rate = st.sidebar.number_input("Risk-free rate (annual, %)", 0.0, 10.0, 4.0, 0.1) / 100.0

def parse_tickers(text: str):
    parts = [p.strip().upper() for p in text.replace("\n", ",").replace(" ", ",").split(",")]
    return sorted(list(set([p for p in parts if p])))

def get_price_history(ticker: str, period="2y"):
    data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    return data if (data is not None and not data.empty) else None

def fundamentals_snapshot(t: yf.Ticker, current_price: float):
    info = {"price": current_price}
    try:
        fast = t.fast_info
        info["pe"] = to_float(getattr(fast, "trailing_pe", None))
        dy = getattr(fast, "dividend_yield", None)
        if dy is not None and to_float(dy) < 1.0:
            info["dividend_yield"] = to_float(dy) * 100.0
        else:
            info["dividend_yield"] = to_float(dy)
    except Exception:
        info["pe"] = np.nan; info["dividend_yield"] = np.nan
    try:
        if not is_num(info["pe"]) or not is_num(info["dividend_yield"]):
            ii = t.info
            if not is_num(info["pe"]):
                info["pe"] = to_float(ii.get("trailingPE", np.nan))
            if not is_num(info["dividend_yield"]) and ii.get("dividendYield", None) is not None:
                info["dividend_yield"] = to_float(ii["dividendYield"]) * 100.0
    except Exception:
        pass
    rev = np.nan
    try:
        fin = t.financials
        if fin is not None and not fin.empty and 'Total Revenue' in fin.index:
            r = fin.loc['Total Revenue'].dropna()
            if len(r) >= 2 and to_float(r.iloc[1]) != 0:
                rev = (to_float(r.iloc[0]) - to_float(r.iloc[1])) / to_float(r.iloc[1]) * 100.0
    except Exception:
        pass
    if not is_num(rev):
        try:
            qfin = t.quarterly_financials
            if qfin is not None and not qfin.empty and 'Total Revenue' in qfin.index:
                rq = qfin.loc['Total Revenue'].dropna()
                if len(rq) >= 5 and to_float(rq.iloc[4]) != 0:
                    rev = (to_float(rq.iloc[0]) - to_float(rq.iloc[4])) / to_float(rq.iloc[4]) * 100.0
        except Exception:
            pass
    info["revenue_yoy_%"] = to_float(rev)
    return info

st.write("Set filters, then press **Run Scan**.")

tickers = parse_tickers(tickers_text)
if st.button("Run Scan", use_container_width=True):
    rows = []
    progress = st.progress(0.0)
    for i, tk in enumerate(tickers):
        progress.progress((i+1)/len(tickers))
        try:
            hist = get_price_history(tk, period="2y")
            if hist is None:
                st.write(f"Skipping {tk}: no price history."); continue
            close = hist["Close"].dropna()
            price = to_float(close.iloc[-1])
            if not is_num(price):
                st.write(f"Skipping {tk}: invalid price."); continue

            tkr = yf.Ticker(tk)

            # Earnings exclusion
            if exclude_earnings and earnings_within(earnings_window, tkr):
                earn_flag = True
            else:
                earn_flag = False

            # Fundamentals
            fund = fundamentals_snapshot(tkr, price)
            pe = to_float(fund.get("pe", np.nan))
            dy = to_float(fund.get("dividend_yield", np.nan))
            rev_yoy = to_float(fund.get("revenue_yoy_%", np.nan))

            # Technicals
            ma10 = close.rolling(10).mean()
            ma200 = close.rolling(200).mean()
            rsi14 = rsi(close, 14)
            bb_mid, bb_up, bb_low = bollinger(close, 20, 2)
            macd_line, signal_line, hist_line = macd(close)

            rsi_val = get_scalar_last(rsi14)
            ma200_last = get_scalar_last(ma200)
            bb_mid_last = get_scalar_last(bb_mid)
            bb_up_last  = get_scalar_last(bb_up)
            macd_hist_last = get_scalar_last(hist_line)

            # Conditions
            conds = []
            conds.append(is_num(pe) and pe <= pe_max)
            conds.append(is_num(dy) and dy >= div_min)
            conds.append(is_num(rev_yoy) and rev_yoy >= rev_yoy_min)
            if require_above_200dma:
                conds.append(is_num(ma200_last) and price > ma200_last)
            if require_bb_mid_to_upper:
                conds.append(is_num(bb_mid_last) and is_num(bb_up_last) and (price >= bb_mid_last) and (price <= bb_up_last))
            conds.append(is_num(rsi_val) and (rsi_val >= rsi_min) and (rsi_val <= rsi_max))

            passed = (not earn_flag) and all(bool(c) for c in conds)

            rows.append({
                "Ticker": tk,
                "Price": round(price, 2) if is_num(price) else np.nan,
                "P/E": round(pe, 2) if is_num(pe) else np.nan,
                "Dividend Yield %": round(dy, 2) if is_num(dy) else np.nan,
                "Revenue YoY %": round(rev_yoy, 2) if is_num(rev_yoy) else np.nan,
                "RSI(14)": round(rsi_val, 1) if is_num(rsi_val) else np.nan,
                "Above 200DMA": bool(is_num(ma200_last) and price > ma200_last),
                "Between BB mid & upper": bool(is_num(bb_mid_last) and is_num(bb_up_last) and (price >= bb_mid_last) and (price <= bb_up_last)),
                "Earnings ≤ N days": bool(earn_flag),
                "Passed Filters": bool(passed)
            })
        except Exception as e:
            st.write(f"Error processing {tk}: {e}")
            continue

    if rows:
        df = pd.DataFrame(rows)
        order = ["Ticker","Price","P/E","Dividend Yield %","Revenue YoY %","RSI(14)",
                 "Above 200DMA","Between BB mid & upper","Earnings ≤ N days","Passed Filters"]
        df = df[order]
        st.success(f"Scan complete. {int(df['Passed Filters'].sum())} out of {len(df)} passed.")
        st.dataframe(df, use_container_width=True)
        st.download_button("Download Results CSV", df.to_csv(index=False).encode("utf-8"),
                           file_name="leap_coveredcall_scan.csv", mime="text/csv", use_container_width=True)
    else:
        st.warning("No results to show.")
