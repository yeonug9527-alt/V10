import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import concurrent.futures
import uuid
import math


def normalize_universe_dates(df):
    """Normalize historical-universe intervals without recursion."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Symbol", "StartDate", "EndDate"])
    out = df.copy()
    aliases = {
        "Code": "Symbol", "Ticker": "Symbol", "종목코드": "Symbol",
        "시작일": "StartDate", "종료일": "EndDate",
        "Start": "StartDate", "End": "EndDate",
    }
    out = out.rename(columns={k: v for k, v in aliases.items() if k in out.columns})
    if "Symbol" not in out.columns:
        return pd.DataFrame(columns=["Symbol", "StartDate", "EndDate"])
    out["Symbol"] = out["Symbol"].astype(str).str.strip()
    out["StartDate"] = pd.to_datetime(out["StartDate"], errors="coerce") if "StartDate" in out.columns else pd.NaT
    out["EndDate"] = pd.to_datetime(out["EndDate"], errors="coerce") if "EndDate" in out.columns else pd.NaT
    out["StartDate"] = out["StartDate"].fillna(pd.Timestamp.min)
    out["EndDate"] = out["EndDate"].fillna(pd.Timestamp.max)
    out = out[(out["Symbol"] != "") & (out["StartDate"] <= out["EndDate"])]
    return out[["Symbol", "StartDate", "EndDate"]].drop_duplicates(
        ["Symbol", "StartDate", "EndDate"]
    ).reset_index(drop=True)


def symbol_active_on(universe_df, symbol, date):
    """Check whether a symbol is active on a given date."""
    if universe_df is None or universe_df.empty:
        return True
    u = normalize_universe_dates(universe_df)
    d = pd.Timestamp(date)
    rows = u[u["Symbol"].astype(str) == str(symbol)]
    return bool(((rows["StartDate"] <= d) & (d <= rows["EndDate"])).any())


def gap_bucket(gap_pct):
    if pd.isna(gap_pct): return "데이터부족"
    if gap_pct <= -3: return "≤ -3%"
    if gap_pct < 0: return "-3~0%"
    if gap_pct < 3: return "0~3%"
    if gap_pct < 7: return "3~7%"
    return "≥ 7%"

def expanded_trade_stats(trades: pd.DataFrame) -> dict:
    if trades is None or trades.empty:
        return {}
    x = trades.copy()
    pnl = pd.to_numeric(x.get("Net_PnL"), errors="coerce").dropna()
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) else np.inf
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    return {
        "trades": len(pnl),
        "win_rate": float((pnl > 0).mean()) if len(pnl) else np.nan,
        "profit_factor": float(pf),
        "avg_win": float(avg_win) if pd.notna(avg_win) else np.nan,
        "avg_loss": float(avg_loss) if pd.notna(avg_loss) else np.nan,
        "expectancy": float(pnl.mean()) if len(pnl) else np.nan,
        "total_cost": float(pd.to_numeric(x.get("Total_Cost"), errors="coerce").fillna(0).sum())
            if "Total_Cost" in x else np.nan,
    }

def add_stock_names(df, master):
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out=df.copy()
    if master is not None and not master.empty and "Symbol" in master.columns and "Name" in master.columns:
        mp=dict(zip(master["Symbol"].astype(str),master["Name"].astype(str)))
        out["Name"]=out["Symbol"].astype(str).map(mp).fillna(out["Symbol"].astype(str))
        cols=out.columns.tolist(); out=out[[c for c in ["Name","Symbol"] if c in cols]+[c for c in cols if c not in {"Name","Symbol"}]]
    return out

def korean_trade_table(df, master=None):
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    x=add_stock_names(df,master)
    mp={"Name":"종목명","Symbol":"종목코드","Position_ID":"거래번호","Entry_Date":"진입일","Exit_Date":"청산일","Initial_Qty":"초기수량","Action_Qty":"매도수량","Remaining_Qty":"남은수량","Holding_Days":"보유일수","Entry_Price":"진입가격","Exit_Price":"청산가격","Eval_Price":"현재평가가격","Gross_PnL":"매매손익","Buy_Fee_Allocated":"매수수수료","Sell_Fee":"매도수수료","Sell_Tax":"거래세","Net_PnL":"순손익","Realized_PnL":"실현손익","Realized_Total_PnL":"총실현손익","Unrealized_Net_PnL":"미실현손익","Net_PnL_Pct_On_Sold_Cost":"순수익률","Exit_Reason":"청산사유","Signal_Date":"신호일","Signal_ATR":"신호일 ATR","EntryGapPct":"진입갭","SignalRegime":"신호 당시 시장상태","SignalYear":"신호연도","Date":"날짜","Type":"거래구분","Price":"가격","Reason":"사유","ReservedReleased":"반환된 예약금"}
    return x.rename(columns={k:v for k,v in mp.items() if k in x.columns})

def recommendation_risk_levels(row):
    try:
        entry=float(row.get("Close",np.nan)); atr=float(row.get("ATR14",np.nan))
        if not (np.isfinite(entry) and entry>0 and np.isfinite(atr) and atr>0): return None
        stop=entry-2.0*atr; risk=entry-stop; tp1=entry+2.0*risk; tp2=entry+3.5*risk
        return {"기준 진입가":entry,"손절가":stop,"1차 익절가":tp1,"2차 익절가":tp2,"손절폭":(stop/entry-1)*100,"1차 익절폭":(tp1/entry-1)*100,"2차 익절폭":(tp2/entry-1)*100}
    except Exception:
        return None

def backtest_glossary():
    g=pd.DataFrame([["최종 자산","마지막 날 현금 + 보유주식 평가액"],["누적 수익률","초기자본 대비 최종 자산의 증가·감소율"],["연평균 수익률(CAGR)","전체 기간 수익을 연간 기준으로 환산한 값"],["최대 낙폭(MDD)","고점 대비 가장 크게 떨어진 폭. 작을수록 좋음"],["실현 승률","완전히 청산된 거래 중 수익 거래 비율"],["수익/손실 비율(PF)","전체 이익 ÷ 전체 손실. 1보다 크면 이익이 손실보다 큼"],["Sharpe","일별 변동성 대비 수익의 정도"],["Sortino","하락 변동성만 위험으로 본 수익 대비 위험 지표"],["예약금","다음날 매수를 위해 따로 확보한 현금"],["미실현 손익","아직 팔지 않은 주식의 평가손익. 실현 승률에는 제외"],["신호일","매수 조건을 판단한 날의 종가"],["진입일","실제 매수 체결일. 기본은 다음 거래일 시가"],["TP1 / TP2","1차 / 2차 익절 목표가"],["SL","손절가"]],columns=["용어","뜻"])
    st.dataframe(g,use_container_width=True,hide_index=True)

def calendar_walk_forward(signal_df: pd.DataFrame, train_years=2, test_years=1,
                          score_candidates=(60,65,70,75,80,85,90)):
    """Date-based WF; no splitting within a calendar day."""
    if signal_df is None or signal_df.empty or "SignalDate" not in signal_df:
        return pd.DataFrame(), None
    d = signal_df.copy()
    d["SignalDate"] = pd.to_datetime(d["SignalDate"]).dt.normalize()
    years = sorted(d["SignalDate"].dt.year.unique())
    rows = []
    selected = []
    for end_train in years:
        test_year = end_train + 1
        train_start = end_train - train_years + 1
        if train_start not in years or test_year not in years:
            continue
        tr = d[(d.SignalDate.dt.year >= train_start) & (d.SignalDate.dt.year <= end_train)]
        te = d[d.SignalDate.dt.year == test_year]
        if tr.empty or te.empty:
            continue
        best_score, best_val = None, -np.inf
        for sc in score_candidates:
            z = tr[tr["Score"] >= sc]
            val = pd.to_numeric(z.get("Fwd20_Net", z.get("Fwd20")), errors="coerce").mean()
            if pd.notna(val) and val > best_val:
                best_score, best_val = sc, val
        if best_score is not None:
            selected.append(best_score)
        for sc in score_candidates:
            z = te[te["Score"] >= sc]
            val = pd.to_numeric(z.get("Fwd20_Net", z.get("Fwd20")), errors="coerce").mean()
            rows.append({
                "Train": f"{train_start}-{end_train}",
                "Test": str(test_year),
                "Threshold": sc,
                "Test_Fwd20": val,
                "Test_N": len(z),
            })
    wf = pd.DataFrame(rows)
    chosen = int(round(np.median(selected))) if selected else None
    return wf, chosen

# ============================================================
# KRX CLOSE -> NEXT OPEN STRATEGY LAB V9.7
# Research-first version
# V7: date-based WF, historical validity, regime alignment, gap diagnostics
#
# IMPORTANT DESIGN PRINCIPLES
# 1) Signal is known only at t close.
# 2) Pending order executes at t+1 open.
# 3) ATR/stop/targets are frozen from SIGNAL-DAY information.
# 4) Buy fee is accounted exactly once in position P&L.
# 5) Open positions are excluded from realized win-rate/PF.
# 6) Pending capital is explicitly reserved/released.
# 7) Current KRX universe is clearly labeled as survivorship-biased.
# 8) Optional historical-universe CSV can reduce survivorship bias.
# 9) Market regime, factor score and forward-return research are separated.
# ============================================================

st.set_page_config(
    page_title="KRX 종가매매 전략 연구소 V9.7",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.small-note {font-size: 0.85rem; color: #666;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 0. HELPERS
# ============================================================

def safe_div(a, b, default=0.0):
    try:
        if b is None or pd.isna(b) or float(b) == 0:
            return default
        return float(a) / float(b)
    except Exception:
        return default


def wilder_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50)
    return rsi


def wilder_atr(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def ema_slope(series, lookback=5):
    return series - series.shift(lookback)


def zscore(series, window=60):
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def normalize_score(value, low, high):
    if pd.isna(value):
        return np.nan
    return np.clip((value - low) / (high - low) * 100, 0, 100)

# ============================================================
# 1. DATA
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def get_krx_stock_list():
    try:
        df = fdr.StockListing("KRX").copy()
        symbol_col = next((c for c in ["Symbol", "Code"] if c in df.columns), None)
        name_col = next((c for c in ["Name", "종목명", " 종목명"] if c in df.columns), None)
        if symbol_col is None or name_col is None:
            raise ValueError("Symbol/Name 컬럼을 찾지 못했습니다.")

        df = df.rename(columns={symbol_col: "Symbol", name_col: "Name"})
        df["Symbol"] = df["Symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
        df = df.dropna(subset=["Symbol"])
        if "Market" in df.columns:
            df = df[df["Market"].isin(["KOSPI", "KOSDAQ"])].copy()
        else:
            df["Market"] = "UNKNOWN"

        # ETF/ETN/SPAC/REIT 등 명백한 비일반주식 제거.
        exclude = [
            "스팩", "SPAC", "ETF", "ETN", "리츠", "REIT", "인프라",
            "신주인수권", "KODEX", "TIGER", "ARIRANG", "KBSTAR",
            "HANARO", "KOSEF", "TIMEFOLIO", "PLUS", "인버스", "레버리지",
        ]
        pattern = "|".join(map(lambda x: x.upper(), exclude))
        mask = ~df["Name"].astype(str).str.upper().str.contains(pattern, regex=True, na=False)
        df = df[mask].copy()

        # 명백한 우선주 표기만 제거. 코드 마지막 숫자 추정 금지.
        pref = (
            df["Name"].astype(str).str.endswith("우")
            | df["Name"].astype(str).str.contains(r"우B$|우C$|우\(전환\)$", regex=True, na=False)
        )
        df = df[~pref].copy()
        return df[["Symbol", "Name", "Market"]].drop_duplicates("Symbol").reset_index(drop=True)
    except Exception as e:
        st.error(f"KRX 목록 로드 실패: {e}")
        return pd.DataFrame(columns=["Symbol", "Name", "Market"])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv(symbol, start_date, end_date):
    try:
        df = fdr.DataReader(symbol, start_date, end_date)
        if df is None or df.empty:
            return None
        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required):
            return None
        df = df[required].copy().dropna()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)]
        df["Volume"] = df["Volume"].clip(lower=0)
        return df if len(df) >= 80 else None
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_index(market, start_date, end_date):
    try:
        ticker = "KS11" if market == "KOSPI" else "KQ11"
        df = fdr.DataReader(ticker, start_date, end_date)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return df["Close"].dropna().sort_index()
    except Exception:
        return None


def load_historical_universe(uploaded_file):
    """Optional CSV columns: Symbol,Name,Market,StartDate,EndDate.
    StartDate/EndDate are optional. If absent, the symbol is valid for the entire test range.
    """
    if uploaded_file is None:
        return None, "현재 KRX 목록 사용"
    try:
        u = pd.read_csv(uploaded_file)
        required = {"Symbol", "Market"}
        if not required.issubset(u.columns):
            return None, "CSV에는 최소 Symbol, Market 컬럼이 필요합니다."
        u["Symbol"] = u["Symbol"].astype(str).str.extract(r"(\d{6})", expand=False)
        u = u.dropna(subset=["Symbol"])
        if "Name" not in u.columns:
            u["Name"] = u["Symbol"]
        for c in ["StartDate", "EndDate"]:
            if c in u.columns:
                u[c] = pd.to_datetime(u[c], errors="coerce")
        return normalize_universe_dates(u).drop_duplicates("Symbol"), "사용자 제공 역사적 유니버스"
    except Exception as e:
        return None, f"역사적 유니버스 CSV 오류: {e}"

# ============================================================
# 2. INDICATORS / FACTORS
# ============================================================


def add_market_regime_column(df, market_data):
    """Attach a date-aligned market regime to a price DataFrame."""
    out = df.copy()
    if out.empty:
        out["Regime"] = "데이터부족"
        return out
    if isinstance(market_data, pd.DataFrame):
        mclose = market_data["Close"] if "Close" in market_data.columns else (
            market_data.iloc[:, 0] if market_data.shape[1] == 1 else pd.Series(dtype=float)
        )
    elif isinstance(market_data, pd.Series):
        mclose = market_data
    else:
        mclose = pd.Series(dtype=float)
    mclose = pd.to_numeric(mclose, errors="coerce").dropna()
    mclose.index = pd.to_datetime(mclose.index).normalize()
    mclose = mclose[~mclose.index.duplicated(keep="last")].sort_index()
    if mclose.empty:
        out["Regime"] = "데이터부족"
        return out
    ema60 = mclose.ewm(span=60, adjust=False, min_periods=60).mean()
    ema120 = mclose.ewm(span=120, adjust=False, min_periods=120).mean()
    slope20 = ema60.pct_change(20)
    regime = pd.Series("횡보장", index=mclose.index, dtype="object")
    regime[(mclose > ema60) & (ema60 > ema120) & (slope20 > 0.005)] = "상승장"
    regime[(mclose < ema60) & (ema60 < ema120) & (slope20 < -0.005)] = "하락장"
    dates = pd.to_datetime(out.index).normalize()
    out["Regime"] = dates.map(regime).fillna("데이터부족").to_numpy()
    return out



def calculate_indicators(raw, market_close=None, market_data=None):
    """t일 종가까지만 사용해 계산하는 기술지표 엔진."""
    df = raw.copy()
    c, h, l, o, v = [pd.to_numeric(df[x], errors="coerce") for x in ["Close","High","Low","Open","Volume"]]

    for n in [5, 10, 20, 60, 120, 200]:
        df[f"EMA{n}"] = c.ewm(span=n, adjust=False, min_periods=n).mean()
        df[f"SMA{n}"] = c.rolling(n, min_periods=n).mean()
    df["EMA20_Slope"] = ema_slope(df["EMA20"], 5)
    df["EMA60_Slope"] = ema_slope(df["EMA60"], 10)

    df["RSI14"] = wilder_rsi(c, 14)
    df["ATR14"] = wilder_atr(df, 14)
    df["ATR_Pct"] = df["ATR14"] / c.replace(0, np.nan)

    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # 영상에서 보이는 복합 신호를 재현하기 위한 보조 지표.
    # 원본 영상의 비공개 소스코드를 알 수 없으므로 표준 정의를 사용한다.
    median_price = (h + l) / 2.0
    df["AwesomeOscillator"] = (
        median_price.rolling(5, min_periods=5).mean()
        - median_price.rolling(34, min_periods=34).mean()
    )
    df["Momentum10"] = c - c.shift(10)
    df["ROC10"] = c.pct_change(10) * 100

    # 표준 SuperTrend(ATR 10, multiplier 3.0)
    st_period = 10
    st_mult = 3.0
    st_atr = wilder_atr(df, st_period)
    hl2 = (h + l) / 2.0
    basic_upper = hl2 + st_mult * st_atr
    basic_lower = hl2 - st_mult * st_atr
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    for j in range(1, len(df)):
        if pd.isna(st_atr.iloc[j]):
            continue
        prev_close = c.iloc[j - 1]
        prev_fu = final_upper.iloc[j - 1]
        prev_fl = final_lower.iloc[j - 1]
        final_upper.iloc[j] = (
            basic_upper.iloc[j]
            if pd.isna(prev_fu) or basic_upper.iloc[j] < prev_fu or prev_close > prev_fu
            else prev_fu
        )
        final_lower.iloc[j] = (
            basic_lower.iloc[j]
            if pd.isna(prev_fl) or basic_lower.iloc[j] > prev_fl or prev_close < prev_fl
            else prev_fl
        )
    supertrend = pd.Series(np.nan, index=df.index, dtype=float)
    supertrend_dir = pd.Series(0, index=df.index, dtype=int)
    for j in range(1, len(df)):
        if pd.isna(st_atr.iloc[j]):
            continue
        prev_st = supertrend.iloc[j - 1]
        if pd.isna(prev_st):
            supertrend.iloc[j] = final_upper.iloc[j] if c.iloc[j] <= final_upper.iloc[j] else final_lower.iloc[j]
            supertrend_dir.iloc[j] = -1 if c.iloc[j] <= final_upper.iloc[j] else 1
        elif prev_st == final_upper.iloc[j - 1]:
            if c.iloc[j] <= final_upper.iloc[j]:
                supertrend.iloc[j] = final_upper.iloc[j]
                supertrend_dir.iloc[j] = -1
            else:
                supertrend.iloc[j] = final_lower.iloc[j]
                supertrend_dir.iloc[j] = 1
        else:
            if c.iloc[j] >= final_lower.iloc[j]:
                supertrend.iloc[j] = final_lower.iloc[j]
                supertrend_dir.iloc[j] = 1
            else:
                supertrend.iloc[j] = final_upper.iloc[j]
                supertrend_dir.iloc[j] = -1
    df["SuperTrend"] = supertrend
    df["SuperTrendDir"] = supertrend_dir
    df["SuperTrendBull"] = supertrend_dir > 0

    up_move, down_move = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr_w = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_w.replace(0,np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr_w.replace(0,np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0,np.nan)
    df["PlusDI"], df["MinusDI"] = plus_di, minus_di
    df["ADX14"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    df["VideoLongScore"] = (
        (df["SuperTrendDir"] > 0).astype(int)
        + (df["PlusDI"] > df["MinusDI"]).astype(int)
        + (df["MACD"] > df["MACD_Signal"]).astype(int)
        + (df["AwesomeOscillator"] > 0).astype(int)
        + (df["Momentum10"] > 0).astype(int)
        + (df["ROC10"] > 0).astype(int)
    )
    df["VideoShortScore"] = (
        (df["SuperTrendDir"] < 0).astype(int)
        + (df["MinusDI"] > df["PlusDI"]).astype(int)
        + (df["MACD"] < df["MACD_Signal"]).astype(int)
        + (df["AwesomeOscillator"] < 0).astype(int)
        + (df["Momentum10"] < 0).astype(int)
        + (df["ROC10"] < 0).astype(int)
    )
    df["VideoLongSignal"] = (
        (df["VideoLongScore"] >= 5)
        & (df["VideoLongScore"].shift(1) < 5)
    )
    df["VideoShortSignal"] = (
        (df["VideoShortScore"] >= 5)
        & (df["VideoShortScore"].shift(1) < 5)
    )

    bb_mid = c.rolling(20, min_periods=20).mean()
    bb_std = c.rolling(20, min_periods=20).std()
    df["BB_Mid"] = bb_mid
    df["BB_Upper"], df["BB_Lower"] = bb_mid + 2*bb_std, bb_mid - 2*bb_std
    df["BB_Width"] = (df["BB_Upper"]-df["BB_Lower"]) / bb_mid.replace(0,np.nan)
    df["BB_Position"] = (c-df["BB_Lower"]) / (df["BB_Upper"]-df["BB_Lower"]).replace(0,np.nan)

    df["TradingValue"] = c*v
    df["TradingValue20"] = df["TradingValue"].rolling(20, min_periods=20).mean()
    df["VolumeMA20"] = v.rolling(20, min_periods=20).mean()
    df["VolumeRatio"] = v / df["VolumeMA20"].replace(0,np.nan)

    # OBV
    direction = np.sign(c.diff()).fillna(0)
    df["OBV"] = (direction*v).cumsum()
    df["OBV_MA20"] = df["OBV"].rolling(20, min_periods=20).mean()
    df["OBV_Slope"] = df["OBV"] - df["OBV"].shift(5)
    df["OBV_Trend"] = (df["OBV"] > df["OBV_MA20"]) & (df["OBV_Slope"] > 0)

    # 일봉 OHLCV로 계산하는 CVD 근사치. 실제 체결별 CVD가 아님을 명확히 표시한다.
    candle_pressure = ((2*c - h - l) / (h-l).replace(0,np.nan)).clip(-1,1).fillna(0)
    df["CVD"] = (candle_pressure * v).cumsum()
    df["CVD_MA20"] = df["CVD"].rolling(20, min_periods=20).mean()
    df["CVD_Slope"] = df["CVD"] - df["CVD"].shift(5)
    df["CVD_Trend"] = (df["CVD"] > df["CVD_MA20"]) & (df["CVD_Slope"] > 0)
    df["OBV_CVD_Bull"] = df["OBV_Trend"] & df["CVD_Trend"]

    mf_mult = ((c-l)-(h-c))/(h-l).replace(0,np.nan)
    mf_vol = mf_mult.fillna(0)*v
    df["CMF20"] = mf_vol.rolling(20,min_periods=20).sum()/v.rolling(20,min_periods=20).sum().replace(0,np.nan)

    typical=(h+l+c)/3
    money_flow=typical*v
    tp_delta=typical.diff()
    pos_flow=money_flow.where(tp_delta>0,0.0).rolling(14,min_periods=14).sum()
    neg_flow=money_flow.where(tp_delta<0,0.0).abs().rolling(14,min_periods=14).sum()
    mfr=pos_flow/neg_flow.replace(0,np.nan)
    df["MFI14"]=100-100/(1+mfr)
    df.loc[(neg_flow==0)&(pos_flow>0),"MFI14"]=100

    for n in [3,5,8,10,20,60,120]:
        df[f"Return{n}"] = c.pct_change(n)*100

    # 상승 종목을 더 직접적으로 찾기 위한 추가 특징
    df["EMA20_Accel"] = df["EMA20_Slope"] - df["EMA20_Slope"].shift(5)
    df["EMA60_Accel"] = df["EMA60_Slope"] - df["EMA60_Slope"].shift(10)
    df["TrendAlignment"] = (
        (c > df["EMA20"]).astype(int) +
        (df["EMA20"] > df["EMA60"]).astype(int) +
        (df["EMA60"] > df["EMA120"]).astype(int)
    )
    df["MACD_Hist_Slope"] = df["MACD_Hist"] - df["MACD_Hist"].shift(3)
    df["OBV_SlopePct"] = df["OBV"].pct_change(5) * 100
    df["CMF_Positive"] = df["CMF20"] > 0

    if market_close is not None:
        m = pd.Series(market_close).reindex(df.index).ffill()
        for n in [20,60]:
            df[f"MarketReturn{n}"]=m.pct_change(n)*100
            df[f"RelativeReturn{n}"]=df[f"Return{n}"]-df[f"MarketReturn{n}"]
        df["MarketEMA20"]=m.ewm(span=20,adjust=False,min_periods=20).mean()
        df["MarketEMA60"]=m.ewm(span=60,adjust=False,min_periods=60).mean()
        df["MarketEMA60Slope"]=df["MarketEMA60"]-df["MarketEMA60"].shift(10)
        df["MarketAboveEMA60"]=m>df["MarketEMA60"]
    else:
        for n in [20,60]:
            df[f"MarketReturn{n}"]=np.nan
            df[f"RelativeReturn{n}"]=np.nan
        df["MarketEMA20"]=df["MarketEMA60"]=df["MarketEMA60Slope"]=np.nan
        df["MarketAboveEMA60"]=False

    # 전일까지만 사용한 돌파 기준
    df["High20Prev"]=h.shift(1).rolling(20,min_periods=20).max()
    df["High60Prev"]=h.shift(1).rolling(60,min_periods=60).max()
    df["Low60Prev"]=l.shift(1).rolling(60,min_periods=60).min()
    df["Breakout20"]=c>df["High20Prev"]
    df["Breakout60"]=c>df["High60Prev"]
    df["Position60"]=(c-df["Low60Prev"])/(df["High60Prev"]-df["Low60Prev"]).replace(0,np.nan)

    candle_range=(h-l).replace(0,np.nan)
    df["BodyPct"]=(c-o).abs()/candle_range
    df["CloseLocation"]=(c-l)/candle_range
    df["UpperWickPct"]=(h-np.maximum(o,c))/candle_range
    df["LowerWickPct"]=(np.minimum(o,c)-l)/candle_range
    df["GapPct"]=(o/c.shift(1)-1)*100
    df["DisparityEMA20"]=(c/df["EMA20"]-1)*100

    # 과거 분포 기반. 현재값 자체는 포함되지만 미래는 포함하지 않음.
    df["ATRPercentile"]=df["ATR_Pct"].rolling(120,min_periods=40).rank(pct=True)
    df["BBWidthPercentile"]=df["BB_Width"].rolling(120,min_periods=40).rank(pct=True)
    df["GapHistory20"]=df["GapPct"].shift(1).rolling(20,min_periods=10).median()
    df["GapAbsHistory20"]=df["GapPct"].shift(1).abs().rolling(20,min_periods=10).median()
    df["High20Now"]=h.rolling(20,min_periods=20).max()
    df["Pullback20Pct"]=(df["High20Now"]-c)/df["High20Now"].replace(0,np.nan)

    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    if market_data is not None:
        # V8의 out 변수 버그 수정
        df = add_market_regime_column(df, market_data)
    else:
        df["Regime"] = df.apply(market_regime, axis=1)
    return df


def market_regime(row):
    if pd.isna(row.get("MarketEMA60")):
        return "데이터부족"
    above=bool(row.get("MarketAboveEMA60",False))
    slope=row.get("MarketEMA60Slope",np.nan)
    ret20=row.get("MarketReturn20",np.nan)
    if pd.isna(slope) or pd.isna(ret20):
        return "데이터부족"
    if above and slope>0 and ret20>3: return "강한 상승장"
    if above and slope>=0: return "상승장"
    if (not above) and slope<0 and ret20<-3: return "강한 하락장"
    if not above and slope<=0: return "하락장"
    return "횡보장"


def _rank_pct(series, higher_is_better=True):
    x=pd.to_numeric(series,errors="coerce")
    if higher_is_better:
        return x.rank(pct=True,method="average")*100
    return (1-x.rank(pct=True,method="average")+1/len(x))*100 if len(x) else x



def add_rise_pattern_features(df):
    """현재까지의 데이터만 사용해 상승 패턴을 분류한다.
    패턴은 '상승 시작/지속/돌파/눌림 재상승' 중 가장 가까운 형태를 선택하고,
    과거 미래값을 참조하지 않는다.
    """
    x=df.copy()
    out_type=pd.Series("관찰형", index=x.index, dtype="object")
    out_score=pd.Series(50.0, index=x.index, dtype=float)
    out_desc=pd.Series("뚜렷한 상승 패턴이 아직 약합니다.", index=x.index, dtype="object")

    c=x["Close"]; e20=x["EMA20"]; e60=x["EMA60"]; e120=x["EMA120"]
    r5=x["Return5"]; r20=x["Return20"]; rs=x.get("RelativeReturn20", pd.Series(np.nan,index=x.index))
    vol=x["VolumeRatio"]; adx=x["ADX14"]; rsi=x["RSI14"]
    disp=x["DisparityEMA20"]; pb=x["Pullback20Pct"]
    macd_s=x["MACD_Hist_Slope"]; obv=x["OBV_Slope"]
    cvd=x.get("CVD_Slope", pd.Series(np.nan,index=x.index))
    flow_bull=x.get("OBV_CVD_Bull", pd.Series(False,index=x.index))
    br20=x["Breakout20"]; br60=x["Breakout60"]
    close_loc=x["CloseLocation"]

    alignment=(c>e20)&(e20>e60)&(e60>e120)
    trend_up=(x["EMA20_Slope"]>0)&(x["EMA60_Slope"]>0)
    acceleration=(x["EMA20_Accel"]>0)
    breakout=(br20|br60)
    breakout_quality=(
        30*breakout.astype(float)
        + 25*(vol>=1.25).astype(float)
        + 20*(close_loc>=0.60).astype(float)
        + 15*(rsi<74).astype(float)
        + 10*(rs>0).astype(float)
    ).clip(0,100)

    # 상승 시작: 장기 추세가 막 살아나면서 단기 모멘텀이 붙는 구간
    start_score=(
        22*alignment.astype(float)+
        18*trend_up.astype(float)+
        15*acceleration.astype(float)+
        15*(r5.between(0.5,8)).astype(float)+
        12*(r20>3).astype(float)+
        8*(vol>=1.1).astype(float)+
        4*(macd_s>0).astype(float)+
        4*(obv>0).astype(float)+
        4*(cvd>0).astype(float)+
        3*flow_bull.astype(float)
    ).clip(0,100)

    # 상승 지속: 이미 상승한 종목 중 추세/강도/거래가 유지되는 구간
    cont_score=(
        25*alignment.astype(float)+
        20*(r20>5).astype(float)+
        15*(rs>2).astype(float)+
        15*(adx>=20).astype(float)+
        10*(macd_s>=0).astype(float)+
        10*(vol>=0.9).astype(float)+
        5*(disp.between(-1,9)).astype(float)
    ).clip(0,100)

    # 돌파: 전일 이전 고점을 넘어가면서 거래가 붙는 형태
    breakout_score=breakout_quality

    # 눌림 재상승: 상승 추세에서 과도하지 않은 조정 후 재가속
    pullback_score=(
        22*alignment.astype(float)+
        20*pb.between(2,10).astype(float)+
        15*(x["EMA20_Slope"]>0).astype(float)+
        12*(vol.between(0.6,1.7)).astype(float)+
        12*(macd_s>=0).astype(float)+
        10*(rsi.between(45,68)).astype(float)+
        9*(c>e20).astype(float)
    ).clip(0,100)

    scores=pd.DataFrame({
        "상승 시작형":start_score,
        "상승 지속형":cont_score,
        "돌파형":breakout_score,
        "눌림 재상승형":pullback_score,
    }, index=x.index)

    best_name=scores.idxmax(axis=1)
    best_score=scores.max(axis=1)

    # 너무 약한 패턴은 관찰형으로 낮춘다.
    valid=best_score>=58
    out_type.loc[valid]=best_name.loc[valid]
    out_score=best_score.where(valid, 50.0)

    descriptions={
        "상승 시작형":"추세가 살아나면서 단기 상승 힘이 붙는 구간입니다.",
        "상승 지속형":"상승 추세와 시장 대비 강도가 유지되는 구간입니다.",
        "돌파형":"최근 고점을 넘고 거래가 동반되는 형태입니다.",
        "눌림 재상승형":"상승 추세 안에서 조정 후 다시 올라갈 조건을 찾는 형태입니다.",
        "관찰형":"상승 조건이 일부 있지만 뚜렷한 대표 패턴은 아닙니다.",
    }
    out_desc=out_type.map(descriptions).fillna(descriptions["관찰형"])

    x["RisePattern"]=out_type
    x["PatternScore"]=out_score
    x["PatternDescription"]=out_desc
    x["BreakoutQuality"]=breakout_quality
    x["RiseStructureScore"]=(
        0.35*start_score+0.25*cont_score+0.20*breakout_score+0.20*pullback_score
    ).clip(0,100)
    return x


def apply_cross_sectional_features(universe_data):
    """같은 날짜의 종목끼리 상대순위를 계산하고 상승 패턴을 반영한다."""
    if not universe_data:
        return universe_data

    prepared={}
    rows=[]
    for sym,df in universe_data.items():
        if df is None or df.empty:
            continue
        z=add_rise_pattern_features(df)
        prepared[sym]=z
        for dt,row in z.iterrows():
            rows.append({
                "Date":pd.Timestamp(dt),"Symbol":sym,
                "RelativeReturn20":row.get("RelativeReturn20",np.nan),
                "RelativeReturn60":row.get("RelativeReturn60",np.nan),
                "Return20":row.get("Return20",np.nan),"Return60":row.get("Return60",np.nan),
                "VolumeRatio":row.get("VolumeRatio",np.nan),
                "TradingValue20":row.get("TradingValue20",np.nan),
                "Breakout":float(bool(row.get("Breakout20",False)))+0.5*float(bool(row.get("Breakout60",False))),
                "Pullback":row.get("Pullback20Pct",np.nan),
                "ADX14":row.get("ADX14",np.nan),
                "DisparityEMA20":row.get("DisparityEMA20",np.nan),
                "RSI14":row.get("RSI14",np.nan),"ATR_Pct":row.get("ATR_Pct",np.nan),
                "OBVSlope":row.get("OBV_Slope",np.nan),
                "CVDSlope":row.get("CVD_Slope",np.nan),
                "OBV_CVD_Bull":float(bool(row.get("OBV_CVD_Bull",False))),
                "PatternScore":row.get("PatternScore",50),
                "RiseStructureScore":row.get("RiseStructureScore",50),
            })

    allr=pd.DataFrame(rows)
    if allr.empty:
        return universe_data

    breadth_up=allr.groupby("Date")["Return20"].apply(
        lambda s: float((pd.to_numeric(s,errors="coerce")>0).mean()*100)
    )
    allr["MarketParticipation"]=allr["Date"].map(breadth_up)

    def rank_col(col):
        return allr.groupby("Date")[col].rank(pct=True)*100

    allr["RS20Pct"]=rank_col("RelativeReturn20")
    allr["RS60Pct"]=rank_col("RelativeReturn60")
    allr["MomentumPct"]=(0.6*rank_col("Return20")+0.4*rank_col("Return60"))
    allr["VolumePct"]=rank_col("VolumeRatio")
    allr["LiquidityPct"]=rank_col("TradingValue20")
    allr["BreakoutPct"]=rank_col("Breakout")
    allr["OBVPct"]=rank_col("OBVSlope")
    allr["CVDPct"]=rank_col("CVDSlope")
    allr["FlowScore"]=(0.40*allr["OBVPct"]+0.40*allr["CVDPct"]+0.20*(allr["OBV_CVD_Bull"]*100)).clip(0,100)

    # 눌림은 2~10% 부근을 가장 선호한다.
    allr["PullbackQuality"]=(100-(allr["Pullback"]-5).abs()*8).clip(0,100)
    allr["PullbackPct"]=rank_col("PullbackQuality")

    allr["TrendRaw"]=(
        0.22*allr["RS20Pct"]+
        0.14*allr["RS60Pct"]+
        0.18*allr["MomentumPct"]+
        0.16*allr["BreakoutPct"]+
        0.10*allr["PullbackPct"]+
        0.10*allr["LiquidityPct"]+
        0.10*allr["PatternScore"]
    )
    allr["TrendPct"]=allr.groupby("Date")["TrendRaw"].rank(pct=True)*100

    participation_score=allr["MarketParticipation"].clip(0,100)
    allr["QualityScore"]=(
        0.18*allr["TrendPct"]+
        0.15*allr["MomentumPct"]+
        0.13*allr["RS20Pct"]+
        0.08*allr["RS60Pct"]+
        0.08*allr["VolumePct"]+
        0.06*allr["BreakoutPct"]+
        0.05*allr["LiquidityPct"]+
        0.05*participation_score+
        0.12*allr["PatternScore"]+
        0.10*allr["FlowScore"]
    ).clip(0,100)

    entry_rsi=100-(allr["RSI14"]-60).abs()*2.7
    entry_disp=100-(allr["DisparityEMA20"].clip(-12,18)-4).abs()*4.5
    entry_atr=100-(allr["ATR_Pct"]*100-3.5).abs()*11.0
    entry_vol=100-(allr["VolumeRatio"]-1.5).abs()*18.0
    entry_breakout=(0.65*allr["BreakoutPct"]+0.35*allr["VolumePct"]).clip(0,100)
    allr["EntryRaw"]=(
        0.22*entry_rsi.clip(0,100)+
        0.25*entry_disp.clip(0,100)+
        0.15*entry_atr.clip(0,100)+
        0.12*entry_vol.clip(0,100)+
        0.08*allr["PullbackPct"]+
        0.10*entry_breakout+
        0.08*allr["FlowScore"]
    )
    allr["EntryScore"]=allr["EntryRaw"].clip(0,100)

    # 최종점수는 '종목 자체', '내일 진입', '과거 실제 상승 결과'를 분리해서 합친다.
    # HistoricalSetupScore는 나중에 마지막 날짜에만 채워지며 그 시점의 미래를 참조하지 않는다.
    allr["FinalScore"]=(0.60*allr["QualityScore"]+0.40*allr["EntryScore"]).clip(0,100)
    allr["FinalRankPct"]=allr.groupby("Date")["FinalScore"].rank(pct=True,ascending=False,method="first")
    allr["TopBucket"]=np.select(
        [allr["FinalRankPct"]<=0.05,allr["FinalRankPct"]<=0.10,allr["FinalRankPct"]<=0.20],
        ["상위 5%","상위 10%","상위 20%"],default="하위 80%"
    )

    cols=[
        "RS20Pct","RS60Pct","MomentumPct","VolumePct","LiquidityPct","BreakoutPct",
        "PullbackPct","TrendPct","QualityScore","EntryScore","FinalScore",
        "FinalRankPct","TopBucket","MarketParticipation","PatternScore","RiseStructureScore",
        "OBVPct","CVDPct","FlowScore"
    ]
    for sym,df in prepared.items():
        idx=pd.MultiIndex.from_product([pd.to_datetime(df.index),[sym]],names=["Date","Symbol"])
        z=allr.set_index(["Date","Symbol"]).reindex(idx)
        z.index=z.index.get_level_values("Date")
        for col in cols:
            prepared[sym][col]=z[col].to_numpy()

    for sym in universe_data:
        if sym in prepared:
            universe_data[sym]=prepared[sym]
    return universe_data


def _normalize_feature_frame(x, cols):
    z=x[cols].apply(pd.to_numeric, errors="coerce").copy()
    med=z.median()
    mad=(z-med).abs().median().replace(0,np.nan)
    scale=(mad*1.4826).fillna(z.std()).replace(0,np.nan).fillna(1.0)
    return (z-med)/scale


def historical_setup_expectancy(df, lookback=320, neighbors=24):
    """현재 신호와 과거의 비슷한 상황을 비교해 실제 다음 수익을 참고한다.
    현재 시점 이후의 데이터는 절대 사용하지 않는다. 확률 보장이 아니라 과거 유사상황 통계다.
    """
    if df is None or len(df) < 90:
        return {"score":50.0,"win5":np.nan,"avg5":np.nan,"avg20":np.nan,"samples":0}
    x=df.copy().sort_index()
    feature_cols=["Return5","Return20","RelativeReturn20","VolumeRatio","RSI14","ADX14",
                  "DisparityEMA20","ATR_Pct","Pullback20Pct","Position60","CMF20"]
    feature_cols=[c for c in feature_cols if c in x.columns]
    if len(feature_cols)<7:
        return {"score":50.0,"win5":np.nan,"avg5":np.nan,"avg20":np.nan,"samples":0}
    # 신호일의 미래수익. 현재 마지막 행은 미래가 없으므로 자동 제외된다.
    x["_NextOpenRet"]=(x["Open"].shift(-1)/x["Close"]-1)*100
    x["_Fwd5"]=(x["Close"].shift(-5)/x["Open"].shift(-1)-1)*100
    x["_Fwd20"]=(x["Close"].shift(-20)/x["Open"].shift(-1)-1)*100
    hist=x.iloc[:-1].tail(lookback).copy()
    hist=hist.dropna(subset=feature_cols+["_Fwd5","_Fwd20"])
    cur=x.iloc[-1]
    if hist.empty or cur[feature_cols].isna().any():
        return {"score":50.0,"win5":np.nan,"avg5":np.nan,"avg20":np.nan,"samples":0}
    hz=_normalize_feature_frame(hist,feature_cols)
    cm=(pd.DataFrame([cur[feature_cols].to_dict()]))
    all_for_scale=pd.concat([hist[feature_cols],cm],ignore_index=True)
    med=all_for_scale.median(); mad=(all_for_scale-med).abs().median()
    scale=(mad*1.4826).replace(0,np.nan).fillna(all_for_scale.std()).replace(0,np.nan).fillna(1.0)
    cz=((cm-med)/scale).iloc[0]
    weights={
        "Return5":1.15,"Return20":1.20,"RelativeReturn20":1.35,"VolumeRatio":0.90,
        "RSI14":0.75,"ADX14":0.95,"DisparityEMA20":0.85,"ATR_Pct":0.65,
        "Pullback20Pct":0.90,"Position60":0.80,"CMF20":1.05,
    }
    w=np.array([weights.get(c,1.0) for c in feature_cols],dtype=float)
    dist=np.sqrt(((hz.sub(cz,axis=1).to_numpy()**2)*w).sum(axis=1)/w.sum())
    hist=hist.assign(_dist=dist).sort_values("_dist").head(neighbors)
    n=len(hist)
    if n<6:
        return {"score":50.0,"win5":np.nan,"avg5":np.nan,"avg20":np.nan,"samples":n}
    win5=float((hist["_Fwd5"]>0).mean()*100)
    avg5=float(hist["_Fwd5"].mean())
    avg20=float(hist["_Fwd20"].mean())
    s_win=win5
    s_5=float(np.clip(50+(avg5/5.0)*25,0,100))
    s_20=float(np.clip(50+(avg20/12.0)*25,0,100))
    raw=0.50*s_win+0.30*s_5+0.20*s_20
    confidence=min(1.0,n/20.0)
    score=50+(raw-50)*confidence
    return {"score":float(np.clip(score,0,100)),"win5":win5,"avg5":avg5,"avg20":avg20,"samples":n}


def attach_current_expectancy(results):
    for sym,df in results.items():
        if df is None or df.empty:
            continue
        e=historical_setup_expectancy(df)
        for k,v in {
            "HistoricalSetupScore":e["score"],"PastSimilarWin5":e["win5"],
            "PastSimilarAvg5":e["avg5"],"PastSimilarAvg20":e["avg20"],"PastSimilarSamples":e["samples"]
        }.items():
            results[sym].loc[results[sym].index[-1],k]=v
    return results


def score_row(row):
    required=["Close","EMA20","EMA60","EMA120","RSI14","ATR14","VolumeRatio",
              "TradingValue20","ADX14","FinalScore","QualityScore","EntryScore"]
    if any(c not in row or pd.isna(row[c]) for c in required):
        return None

    q=float(row["QualityScore"]); e=float(row["EntryScore"])
    base_final=float(row["FinalScore"])
    hist_raw=row.get("HistoricalSetupScore",50.0)
    hist=float(hist_raw) if pd.notna(hist_raw) else 50.0
    win5=row.get("PastSimilarWin5",np.nan)
    avg5=row.get("PastSimilarAvg5",np.nan)
    avg20=row.get("PastSimilarAvg20",np.nan)
    samples=row.get("PastSimilarSamples",0)

    # '상승 가능성 점수':
    # 과거 유사상황의 승률 + 실제 평균수익 + 표본 수를 함께 반영한다.
    win_component=float(np.clip(win5 if pd.notna(win5) else 50,0,100))
    avg5_component=float(np.clip(50+(float(avg5)/4.0)*25,0,100)) if pd.notna(avg5) else 50
    avg20_component=float(np.clip(50+(float(avg20)/10.0)*25,0,100)) if pd.notna(avg20) else 50
    confidence=min(1.0,float(samples or 0)/20.0)
    rise_potential_raw=0.45*hist+0.25*win_component+0.20*avg5_component+0.10*avg20_component
    rise_potential=50+(rise_potential_raw-50)*confidence

    # 현재 조건 75% + 과거 실제 유사상황 25%.
    # 종목 자체 점수와 진입 점수는 별도로 유지한다.
    final=float(np.clip(0.45*q + 0.30*e + 0.25*rise_potential,0,100))

    trend=float(row.get("TrendPct",q)); mom=float(row.get("MomentumPct",q))
    rs=float(row.get("RS20Pct",50))
    liq=float(row.get("LiquidityPct",50)); br=float(row.get("BreakoutPct",50))
    pattern_score=float(row.get("PatternScore",50))
    return {
        "TotalScore":final,
        "BaseFinalScore":base_final,
        "HistoricalSetupScore":hist,
        "RisePotentialScore":float(np.clip(rise_potential,0,100)),
        "QualityScore":q,"EntryScore":e,
        "PastSimilarWin5":float(win5) if pd.notna(win5) else np.nan,
        "PastSimilarAvg5":float(avg5) if pd.notna(avg5) else np.nan,
        "PastSimilarAvg20":float(avg20) if pd.notna(avg20) else np.nan,
        "PastSimilarSamples":int(samples) if pd.notna(samples) else 0,
        "Trend":trend,"Momentum":mom,"RelativeStrength":rs,
        "LiquidityVolume":liq,
        "Stability":float(np.clip(100-abs(float(row["ATR_Pct"])*100-3.5)*12,0,100)),
        "EntryQuality":e,"BreakoutCandle":br,
        "PatternScore":pattern_score,
        "OBVScore":float(row.get("OBVPct",50)),
        "CVDScore":float(row.get("CVDPct",50)),
        "FlowScore":float(row.get("FlowScore",50)),
        "OBVCVDBull":bool(row.get("OBV_CVD_Bull",False)),
        "PatternType":row.get("RisePattern","관찰형"),
        "PatternDescription":row.get("PatternDescription","뚜렷한 상승 패턴이 아직 약합니다."),
        "Regime":row.get("Regime",market_regime(row)),
        "RankPct":float(row.get("FinalRankPct",np.nan)),
        "TopBucket":row.get("TopBucket","분석불가")
    }


# ============================================================
# 5. GATE
# ============================================================


def entry_gate(row, settings):
    required=["Close","EMA20","EMA60","RSI14","ATR_Pct","TradingValue20","VolumeRatio","RelativeReturn20","FinalScore"]
    for c in required:
        if c not in row or pd.isna(row[c]):
            return False,[f"필수 지표 부족: {c}"]
    reasons=[]
    if row["TradingValue20"]<settings["min_trading_value"]:
        reasons.append("유동성 부족")
    if row["RSI14"]>settings["max_rsi"]:
        reasons.append("단기 과열")
    if row["DisparityEMA20"]>settings["max_disparity"]:
        reasons.append("20일선과 너무 멀리 이격")
    if row["ATR_Pct"]>settings["max_atr_ratio"]:
        reasons.append("변동성 과다")
    if row.get("VolumeRatio",0)>settings.get("max_volume_ratio",4.5):
        reasons.append("거래량 급증 과열")
    if settings["market_filter"] and row.get("Regime",market_regime(row)) in ["하락장","강한 하락장","데이터부족"]:
        reasons.append("시장 국면 불리")
    if settings["require_relative_strength"] and row.get("RelativeReturn20",-999)<=0:
        reasons.append("시장 대비 20일 약세")
    if settings["require_trend"] and not (row["Close"]>row["EMA20"]>row["EMA60"]):
        reasons.append("상승 추세 미충족")
    if row.get("ADX14",0)<settings.get("min_adx",15):
        reasons.append("추세 힘 부족")
    if row.get("FinalRankPct",1)>settings.get("max_rank_pct",0.20):
        reasons.append("당일 상대순위가 낮음")
    return len(reasons)==0,reasons


# ============================================================
# 6. EXECUTION / ACCOUNTING
# ============================================================

def sell_cost(gross_value, fee_rate, tax_rate):
    return gross_value * fee_rate + gross_value * tax_rate


def buy_cost(gross_value, fee_rate):
    return gross_value * fee_rate


def close_position(pos, exit_date, exit_price, reason, qty, fee_rate, tax_rate):
    qty = int(max(0, min(qty, pos["qty"])))
    if qty <= 0:
        return None
    gross_sell = qty * exit_price
    sell_fee = gross_sell * fee_rate
    sell_tax = gross_sell * tax_rate
    # Allocate the original buy fee exactly once across shares sold.
    allocated_buy_fee = pos["buy_fee_total"] * (qty / pos["initial_qty"])
    gross_pnl = (exit_price - pos["entry_price"]) * qty
    net_pnl = gross_pnl - allocated_buy_fee - sell_fee - sell_tax
    return {
        "Position_ID": pos["Position_ID"],
        "Symbol": pos["sym"],
        "Entry_Date": pos["entry_date"],
        "Exit_Date": exit_date,
        "Initial_Qty": pos["initial_qty"],
        "Action_Qty": qty,
        "Entry_Price": pos["entry_price"],
        "Exit_Price": exit_price,
        "Gross_PnL": gross_pnl,
        "Buy_Fee_Allocated": allocated_buy_fee,
        "Sell_Fee": sell_fee,
        "Sell_Tax": sell_tax,
        "Net_PnL": net_pnl,
        "Net_PnL_Pct_On_Sold_Cost": safe_div(net_pnl, pos["entry_price"] * qty, 0) * 100,
        "Exit_Reason": reason,
    }

# ============================================================
# 7. BACKTEST ENGINE
# ============================================================

def run_backtest(universe_data, settings, initial_capital, max_concurrent, max_holding_days,
                 partial_ratio, fee_rate, tax_rate, slip_rate, execution_mode, progress_callback=None):
    dates = sorted(set(d for df in universe_data.values() for d in df.index))
    if len(dates) < 2:
        return None

    cash = float(initial_capital)
    reserved_cash = 0.0
    pending = []
    positions = {}
    trade_events = []
    closed_positions = []
    rejected_orders = []
    portfolio = []

    total_dates = len(dates)
    last_reported = -1
    for i, current_date in enumerate(dates):
        is_last = i == len(dates) - 1
        # 진행률은 백테스트 날짜 처리 기준. 너무 잦은 Streamlit 갱신으로 느려지지 않도록
        # 최대 약 100회만 화면을 갱신한다.
        if progress_callback:
            pct = int((i + 1) / max(total_dates, 1) * 100)
            if pct != last_reported or is_last:
                progress_callback(pct, current_date, i + 1, total_dates)
                last_reported = pct

        # --------------------------------------------------------
        # A. Execute pending orders at current day's actual open.
        #    Stop/targets remain frozen from SIGNAL day.
        # --------------------------------------------------------
        if execution_mode == "next_open" and pending:
            pending.sort(key=lambda x: x["score"], reverse=True)
            new_pending = []
            slots = max(0, max_concurrent - len(positions))
            executed = 0
            for order in pending:
                sym = order["sym"]
                df = universe_data.get(sym)

                # 종목 자체의 다음 유효 거래일까지 예약을 유지한다.
                # 전 시장 공통 날짜 기준으로 '데이터 없음=실패' 처리하면
                # 휴장/거래정지/개별 종목 비거래일에서 잘못된 주문 취소가 발생할 수 있다.
                if df is None:
                    reserved_cash -= order["reserved_amount"]
                    rejected_orders.append({"Date": current_date, "Symbol": sym, "Reason": "종목 데이터 없음", "ReservedReleased": order["reserved_amount"]})
                    continue
                if current_date not in df.index:
                    new_pending.append(order)
                    continue

                # 슬롯이 없으면 해당 신호는 취소하고 예약금은 즉시 반환한다.
                if executed >= slots:
                    reserved_cash -= order["reserved_amount"]
                    rejected_orders.append({"Date": current_date, "Symbol": sym, "Reason": "동시보유 슬롯 부족으로 주문 취소", "ReservedReleased": order["reserved_amount"]})
                    continue

                reserved_cash -= order["reserved_amount"]

                row = df.loc[current_date]
                open_px = float(row["Open"]) * (1 + slip_rate)
                if not np.isfinite(open_px) or open_px <= 0:
                    rejected_orders.append({"Date": current_date, "Symbol": sym, "Reason": "비정상 시가", "ReservedReleased": order["reserved_amount"]})
                    continue

                alloc = order["reserved_amount"]
                qty = int((alloc * 0.98) / open_px)
                if qty <= 0:
                    rejected_orders.append({"Date": current_date, "Symbol": sym, "Reason": "매수 수량 0", "ReservedReleased": order["reserved_amount"]})
                    continue

                gross_buy = qty * open_px
                buy_fee = buy_cost(gross_buy, fee_rate)
                total_cost = gross_buy + buy_fee
                if cash < total_cost:
                    rejected_orders.append({"Date": current_date, "Symbol": sym, "Reason": "실제 현금 부족", "ReservedReleased": order["reserved_amount"]})
                    continue

                cash -= total_cost
                pid = str(uuid.uuid4())[:8]
                pos = {
                    "Position_ID": pid,
                    "sym": sym,
                    "entry_date": current_date,
                    "signal_date": order["signal_date"],
                    "entry_price": open_px,
                    "qty": qty,
                    "initial_qty": qty,
                    "buy_fee_total": buy_fee,
                    "signal_atr": order["signal_atr"],
                    "stop_loss": order["stop_loss"],
                    "t1": order["t1"],
                    "t2": order["t2"],
                    "holding_days": 0,
                    "partial_done": False,
                    "realized_pnl": 0.0,
                }
                positions[pid] = pos
                executed += 1
                trade_events.append({
                    "Position_ID": pid, "Symbol": sym, "Date": current_date,
                    "Type": "BUY", "Action_Qty": qty, "Price": open_px,
                    "Net_PnL": 0.0, "Reason": "t일 종가 신호 -> t+1일 시가 체결",
                    "Signal_Date": order["signal_date"], "Signal_ATR": order["signal_atr"],
                })

            pending = new_pending

        # --------------------------------------------------------
        # B. Manage existing positions.
        # --------------------------------------------------------
        for pid in list(positions.keys()):
            pos = positions[pid]
            df = universe_data[pos["sym"]]
            if current_date not in df.index:
                continue
            row = df.loc[current_date]
            if current_date != pos["entry_date"]:
                pos["holding_days"] += 1

            # Same-bar priority: stop -> T2 -> T1 -> trend -> time.
            exit_reason = None
            exit_px = None
            sell_qty = 0

            # If open gaps through stop, actual executable exit is the open, not the historical stop price.
            if row["Low"] <= pos["stop_loss"]:
                exit_reason = "손절"
                exit_px = min(float(row["Open"]), pos["stop_loss"]) * (1 - slip_rate) if float(row["Open"]) <= pos["stop_loss"] else pos["stop_loss"] * (1 - slip_rate)
                sell_qty = pos["qty"]
            elif row["High"] >= pos["t2"]:
                exit_reason = "TP2"
                exit_px = pos["t2"] * (1 - slip_rate)
                sell_qty = pos["qty"]
            elif row["High"] >= pos["t1"] and not pos["partial_done"]:
                if partial_ratio >= 1.0:
                    exit_reason = "TP1 전량"
                    exit_px = pos["t1"] * (1 - slip_rate)
                    sell_qty = pos["qty"]
                else:
                    exit_reason = "TP1 부분익절"
                    exit_px = pos["t1"] * (1 - slip_rate)
                    sell_qty = max(1, int(pos["initial_qty"] * partial_ratio))
                    sell_qty = min(sell_qty, pos["qty"])
            elif row["Close"] < row["EMA20"] and row["EMA20_Slope"] < 0:
                exit_reason = "EMA20 추세청산"
                exit_px = float(row["Close"]) * (1 - slip_rate)
                sell_qty = pos["qty"]
            elif pos["holding_days"] >= max_holding_days:
                exit_reason = "최대보유기간"
                exit_px = float(row["Close"]) * (1 - slip_rate)
                sell_qty = pos["qty"]

            if exit_reason and sell_qty > 0:
                event = close_position(pos, current_date, exit_px, exit_reason, sell_qty, fee_rate, tax_rate)
                if event:
                    proceeds = sell_qty * exit_px - sell_cost(sell_qty * exit_px, fee_rate, tax_rate)
                    cash += proceeds
                    pos["realized_pnl"] += event["Net_PnL"]
                    trade_events.append({
                        "Position_ID": pid, "Symbol": pos["sym"], "Date": current_date,
                        "Type": "SELL_PARTIAL" if sell_qty < pos["qty"] else "SELL",
                        "Action_Qty": sell_qty, "Price": exit_px,
                        "Net_PnL": event["Net_PnL"], "Reason": exit_reason,
                        "Signal_Date": pos["signal_date"], "Signal_ATR": pos["signal_atr"],
                    })
                    closed = sell_qty >= pos["qty"]
                    if closed:
                        closed_positions.append({
                            **event,
                            "Realized_Total_PnL": pos["realized_pnl"],
                            "Status": "CLOSED",
                        })
                        del positions[pid]
                    else:
                        pos["qty"] -= sell_qty
                        pos["partial_done"] = True

        # --------------------------------------------------------
        # C. Signal generation at current close.
        # --------------------------------------------------------
        if not is_last:
            active_symbols = {p["sym"] for p in positions.values()}
            pending_symbols = {p["sym"] for p in pending}
            candidates = []
            for sym, df in universe_data.items():
                if sym in active_symbols or sym in pending_symbols or current_date not in df.index:
                    continue
                row = df.loc[current_date]
                passed, reasons = entry_gate(row, settings)
                score = score_row(row)
                if passed and score is not None and score["TotalScore"] >= settings["min_score"]:
                    candidates.append((score["TotalScore"], sym, row, score))

            candidates.sort(key=lambda x: (x[0], x[3]["RelativeStrength"] if "RelativeStrength" in x[3] else 0), reverse=True)

            if execution_mode == "close":
                slots = max(0, max_concurrent - len(positions))
                for score_val, sym, row, score in candidates[:slots]:
                    entry_px = float(row["Close"]) * (1 + slip_rate)
                    atr = float(row["ATR14"])  # SIGNAL-DAY ATR only.
                    stop = entry_px - 2.0 * atr
                    risk = entry_px - stop
                    t1 = entry_px + 2.0 * risk
                    t2 = entry_px + 3.5 * risk
                    alloc = cash / max(1, max_concurrent - len(positions))
                    qty = int((alloc * 0.98) / entry_px)
                    if qty <= 0:
                        continue
                    gross_buy = qty * entry_px
                    fee = buy_cost(gross_buy, fee_rate)
                    total = gross_buy + fee
                    if total > cash:
                        continue
                    cash -= total
                    pid = str(uuid.uuid4())[:8]
                    positions[pid] = {
                        "Position_ID": pid, "sym": sym, "entry_date": current_date,
                        "signal_date": current_date, "entry_price": entry_px, "qty": qty,
                        "initial_qty": qty, "buy_fee_total": fee, "signal_atr": atr,
                        "stop_loss": stop, "t1": t1, "t2": t2, "holding_days": 0,
                        "partial_done": False, "realized_pnl": 0.0,
                    }
                    trade_events.append({
                        "Position_ID": pid, "Symbol": sym, "Date": current_date,
                        "Type": "BUY_CLOSE", "Action_Qty": qty, "Price": entry_px,
                        "Net_PnL": 0.0, "Reason": "당일 종가 체결", "Signal_Date": current_date,
                        "Signal_ATR": atr,
                    })
            else:
                # Reserve cash without removing it from cash. Reserved amount is released
                # on execution/cancellation, so it cannot become permanently trapped.
                free_cash = max(0.0, cash - reserved_cash)
                free_slots = max(0, max_concurrent - len(positions) - len(pending))
                if free_slots > 0 and free_cash > 0:
                    per_order = free_cash / free_slots
                    for score_val, sym, row, score in candidates[:free_slots]:
                        atr = float(row["ATR14"])  # FROZEN SIGNAL-DAY ATR.
                        estimated_entry = float(row["Close"])
                        stop = estimated_entry - 2.0 * atr
                        risk = estimated_entry - stop
                        t1 = estimated_entry + 2.0 * risk
                        t2 = estimated_entry + 3.5 * risk
                        reserve = min(per_order, max(0.0, cash - reserved_cash))
                        if reserve <= 0:
                            break
                        pending.append({
                            "sym": sym,
                            "score": score_val,
                            "signal_date": current_date,
                            "signal_atr": atr,
                            "stop_loss": stop,
                            "t1": t1,
                            "t2": t2,
                            "reserved_amount": reserve,
                        })
                        reserved_cash += reserve

        # --------------------------------------------------------
        # D. Equity curve. Pending cash remains cash because it is not spent.
        # --------------------------------------------------------
        stock_value = 0.0
        for pos in positions.values():
            df = universe_data[pos["sym"]]
            valid = df.index[df.index <= current_date]
            if len(valid):
                px = float(df.loc[valid[-1], "Close"])
                stock_value += pos["qty"] * px
        total_asset = cash + stock_value
        portfolio.append({
            "Date": current_date,
            "Cash": cash,
            "ReservedCash": reserved_cash,
            "InvestedValue": stock_value,
            "TotalAsset": total_asset,
            "OpenPositions": len(positions),
            "PendingOrders": len(pending),
        })

    # Unclosed positions are intentionally NOT included in realized win rate/PF.
    open_rows = []
    if portfolio:
        last_date = portfolio[-1]["Date"]
        for pid, pos in positions.items():
            df = universe_data[pos["sym"]]
            valid = df.index[df.index <= last_date]
            eval_px = pos["entry_price"]
            if len(valid):
                eval_px = float(df.loc[valid[-1], "Close"])
            est_sell = pos["qty"] * eval_px
            est_cost = sell_cost(est_sell, fee_rate, tax_rate)
            allocated_buy_fee = pos["buy_fee_total"] * (pos["qty"] / pos["initial_qty"])
            eval_net = (eval_px - pos["entry_price"]) * pos["qty"] - allocated_buy_fee - est_cost + pos["realized_pnl"]
            open_rows.append({
                "Position_ID": pid, "Symbol": pos["sym"], "Entry_Date": pos["entry_date"],
                "Holding_Days": pos["holding_days"], "Remaining_Qty": pos["qty"],
                "Entry_Price": pos["entry_price"], "Eval_Price": eval_px,
                "Realized_PnL": pos["realized_pnl"], "Unrealized_Net_PnL": eval_net,
            })

    port_df = pd.DataFrame(portfolio)
    events_df = pd.DataFrame(trade_events)
    closed_df = pd.DataFrame(closed_positions)
    open_df = pd.DataFrame(open_rows)
    rejected_df = pd.DataFrame(rejected_orders)

    # 체결 갭/신호 국면/연도는 신호일과 실제 체결일의 가격으로 사후 계산한다.
    # 신호일 이후의 데이터는 진입 결정에 사용하지 않는다.
    if not closed_df.empty:
        gap_vals=[]; regime_vals=[]; year_vals=[]
        for _, rr in closed_df.iterrows():
            sym=rr.get("Symbol"); sd=pd.Timestamp(rr.get("Signal_Date")); ed=pd.Timestamp(rr.get("Entry_Date"))
            gap=np.nan; reg="데이터부족"
            df=universe_data.get(sym)
            if df is not None:
                if sd in df.index and ed in df.index:
                    gap=(float(df.loc[ed,"Open"])/float(df.loc[sd,"Close"])-1)*100
                if sd in df.index:
                    reg=df.loc[sd].get("Regime","데이터부족")
            gap_vals.append(gap); regime_vals.append(reg); year_vals.append(sd.year if pd.notna(sd) else np.nan)
        closed_df["EntryGapPct"]=gap_vals
        closed_df["SignalRegime"]=regime_vals
        closed_df["SignalYear"]=year_vals

    if not port_df.empty:
        port_df["DailyReturn"] = port_df["TotalAsset"].pct_change()
        port_df["Peak"] = port_df["TotalAsset"].cummax()
        port_df["Drawdown"] = port_df["TotalAsset"] / port_df["Peak"] - 1
        max_dd = port_df["Drawdown"].min() * 100
        total_return = (port_df.iloc[-1]["TotalAsset"] / initial_capital - 1) * 100
        days = max(1, (port_df.iloc[-1]["Date"] - port_df.iloc[0]["Date"]).days)
        cagr = ((port_df.iloc[-1]["TotalAsset"] / initial_capital) ** (365 / days) - 1) * 100 if port_df.iloc[-1]["TotalAsset"] > 0 else -100
    else:
        max_dd, total_return, cagr = 0, 0, 0

    # Realized position stats only.
    if not closed_df.empty:
        # Each fully closed position may have multiple partial events. Reconstruct from event rows.
        closed_pnl = events_df[events_df["Type"].isin(["SELL", "SELL_PARTIAL"])].groupby("Position_ID")["Net_PnL"].sum()
        closed_pnl = closed_pnl[closed_pnl.index.isin(closed_df["Position_ID"])]
        wins = int((closed_pnl > 0).sum())
        losses = int((closed_pnl < 0).sum())
        pf_num = closed_pnl[closed_pnl > 0].sum()
        pf_den = abs(closed_pnl[closed_pnl < 0].sum())
        pf = pf_num / pf_den if pf_den > 0 else np.inf
        win_rate = wins / len(closed_pnl) * 100 if len(closed_pnl) else 0
        expectancy = closed_pnl.mean() if len(closed_pnl) else 0
        avg_win = closed_pnl[closed_pnl > 0].mean() if wins else 0
        avg_loss = closed_pnl[closed_pnl < 0].mean() if losses else 0
    else:
        closed_pnl = pd.Series(dtype=float)
        wins = losses = 0
        pf = 0
        win_rate = expectancy = avg_win = avg_loss = 0

    # 포트폴리오 위험조정 성과 및 연속손실.
    if not port_df.empty:
        daily=port_df["DailyReturn"].replace([np.inf,-np.inf],np.nan).dropna()
        sharpe=(daily.mean()/daily.std()*np.sqrt(252)) if len(daily)>1 and daily.std()>0 else np.nan
        downside=daily[daily<0]
        sortino=(daily.mean()/downside.std()*np.sqrt(252)) if len(downside)>1 and downside.std()>0 else np.nan
    else:
        sharpe=sortino=np.nan
    max_consecutive_losses=0
    cur=0
    if len(closed_pnl):
        for val in closed_pnl:
            if val<0:
                cur+=1; max_consecutive_losses=max(max_consecutive_losses,cur)
            else:
                cur=0

    def _group_perf(col):
        if closed_df.empty or col not in closed_df.columns: return pd.DataFrame()
        x=closed_df.copy()
        pnl=pd.to_numeric(x.get("Realized_Total_PnL",0),errors="coerce")
        x["_PnL"]=pnl
        return x.groupby(col,dropna=False).agg(
            거래수=("_PnL","size"), 평균손익=("_PnL","mean"),
            승률=("_PnL",lambda z:(z>0).mean()*100),
            총손익=("_PnL","sum")
        ).reset_index()

    gap_perf=_group_perf("EntryGapPct")
    if not gap_perf.empty:
        gap_perf["갭구간"]=pd.cut(gap_perf["EntryGapPct"],[-np.inf,-3,0,3,7,np.inf],labels=["≤-3%","-3~0%","0~3%","3~7%","≥7%"])
    return {
        "portfolio": port_df,
        "events": events_df,
        "closed": closed_df,
        "open": open_df,
        "rejected": rejected_df,
        "meta": {
            "final_asset": float(port_df.iloc[-1]["TotalAsset"]) if not port_df.empty else initial_capital,
            "total_return": total_return,
            "cagr": cagr,
            "mdd": max_dd,
            "closed_positions": len(closed_pnl),
            "win_rate": win_rate,
            "profit_factor": pf,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "open_positions": len(open_df),
            "rejected_orders": len(rejected_df),
            "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
            "sortino": float(sortino) if pd.notna(sortino) else np.nan,
            "max_consecutive_losses": int(max_consecutive_losses),
            "gap_performance": gap_perf,
            "regime_performance": _group_perf("SignalRegime"),
            "year_performance": _group_perf("SignalYear"),
            "max_reserved": float(port_df["ReservedCash"].max()) if not port_df.empty else 0,
        },
    }

# ============================================================
# 8. FORWARD RETURN RESEARCH
# ============================================================

def forward_return_research(universe_data, settings, horizons=(1,5,10,20,30)):
    """신호일 이후 수익만 사용. 후보 선별 자체는 신호일 종가 정보로만 결정."""
    rows=[]
    maxh=max(horizons)
    for sym,df in universe_data.items():
        for i in range(len(df)-maxh-1):
            date=df.index[i]
            row=df.iloc[i]
            passed,reasons=entry_gate(row,settings)
            score=score_row(row)
            if not passed or score is None:
                continue
            next_row=df.iloc[i+1]
            entry=float(next_row["Open"])
            if not np.isfinite(entry) or entry<=0: continue
            out={
                "Symbol":sym,"SignalDate":date,"Regime":score["Regime"],
                "QualityScore":score["QualityScore"],"EntryScore":score["EntryScore"],
                "FinalScore":score["TotalScore"],"RankPct":score.get("RankPct",np.nan),
                "TopBucket":score.get("TopBucket","분석불가"),
                "RSI":row["RSI14"],"ADX":row["ADX14"],"VolumeRatio":row["VolumeRatio"],
                "RelativeReturn20":row.get("RelativeReturn20",np.nan),
                "GapToEntryPct":(entry/row["Close"]-1)*100,
                "ATR_Pct":row.get("ATR_Pct",np.nan),"EMA20Disparity":row.get("DisparityEMA20",np.nan),
            }
            for h in horizons:
                j=i+1+h-1
                out[f"Fwd{h}D"]=(float(df.iloc[j]["Close"])/entry-1)*100 if j<len(df) else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def top_bucket_summary(fwd, horizon="Fwd20D"):
    if fwd is None or fwd.empty:return pd.DataFrame()
    order=["상위 5%","상위 10%","상위 20%","하위 80%"]
    x=fwd[fwd["TopBucket"].isin(order)].copy()
    if x.empty:return pd.DataFrame()
    return x.groupby("TopBucket",observed=False).agg(
        신호수=(horizon,"count"),평균수익=(horizon,"mean"),
        중앙수익=(horizon,"median"),승률=(horizon,lambda z:(z>0).mean()*100)
    ).reindex(order).reset_index()


def rolling_walk_forward(signal_df, train_days=504, test_days=126, step_days=126):
    """날짜를 기준으로 train/test를 순차 분리한다. test 날짜는 train에 포함하지 않는다."""
    if signal_df is None or signal_df.empty:return pd.DataFrame(),None
    d=signal_df.copy()
    d["SignalDate"]=pd.to_datetime(d["SignalDate"]).dt.normalize()
    dates=np.array(sorted(d["SignalDate"].dropna().unique()),dtype="datetime64[ns]")
    if len(dates)<train_days+test_days:return pd.DataFrame(),None
    rows=[]; selected=[]
    for start in range(0,len(dates)-train_days-test_days+1,step_days):
        tr_start,tr_end=dates[start],dates[start+train_days-1]
        te_start,te_end=dates[start+train_days],dates[start+train_days+test_days-1]
        tr=d[(d.SignalDate>=tr_start)&(d.SignalDate<=tr_end)]
        te=d[(d.SignalDate>=te_start)&(d.SignalDate<=te_end)]
        if tr.empty or te.empty:continue
        candidates=[50,55,60,65,70,75,80,85,90]
        table=[]
        for th in candidates:
            z=tr[tr["FinalScore"]>=th]["Fwd20D"].dropna()
            if len(z)<20:continue
            table.append((th,float(z.mean()),float((z>0).mean()*100),len(z)))
        if not table:continue
        # 학습에서는 기대수익과 안정성을 함께 본다.
        bt=max(table,key=lambda q:(q[1]+0.03*(q[2]-50),q[3]))
        th=bt[0]; selected.append(th)
        z=te[te["FinalScore"]>=th]["Fwd20D"].dropna()
        rows.append({
            "학습기간":f"{pd.Timestamp(tr_start).date()} ~ {pd.Timestamp(tr_end).date()}",
            "검증기간":f"{pd.Timestamp(te_start).date()} ~ {pd.Timestamp(te_end).date()}",
            "선택기준":th,"검증신호수":len(z),
            "검증평균20일수익":z.mean() if len(z) else np.nan,
            "검증승률":(z>0).mean()*100 if len(z) else np.nan
        })
    chosen=int(round(np.median(selected))) if selected else None
    return pd.DataFrame(rows),chosen


def choose_threshold_from_train(train_df,candidates=(50,55,60,65,70,75,80,85,90)):
    if train_df is None or train_df.empty or "Fwd20D" not in train_df:return 70,pd.DataFrame()
    result=[]
    score_col="FinalScore" if "FinalScore" in train_df else "Score"
    for threshold in candidates:
        x=train_df[train_df[score_col]>=threshold]["Fwd20D"].dropna()
        if len(x)<20:continue
        result.append({"최소점수":threshold,"신호수":len(x),"실현 승률":(x>0).mean()*100,
                       "평균20일수익":x.mean(),"중앙20일수익":x.median()})
    r=pd.DataFrame(result)
    if r.empty:return 70,r
    r=r.sort_values(["평균20일수익","실현 승률"],ascending=False)
    return int(r.iloc[0]["최소점수"]),r


# ============================================================
# 9. WALK-FORWARD MIN-SCORE SELECTION

# ============================================================

# ============================================================
# 10. LOAD UNIVERSE DATA
# ============================================================

def build_universe(market, start_date, end_date, sample_count, historical_universe=None, progress_callback=None):
    """유니버스 구축.
    진행률은 0~100 전체 파이프라인을 의미한다:
    0~5 준비/시장지수, 5~70 종목 데이터, 70~75 횡단면,
    75~98 과거 유사상황 분석, 98~100 완료.
    """
    if historical_universe is None:
        master = get_krx_stock_list().copy()
        universe_label = "현재 KRX 유니버스 (생존자 편향 가능)"
    else:
        master = historical_universe.copy()
        universe_label = "사용자 제공 역사적 유니버스"

    if market == "KOSPI+KOSDAQ":
        master = master[master["Market"].isin(["KOSPI","KOSDAQ"])].copy()
    else:
        master = master[master["Market"] == market].copy()

    if sample_count and sample_count > 0:
        master = master.head(sample_count).copy()

    names = dict(zip(master["Symbol"], master["Name"]))
    market_map = dict(zip(master["Symbol"], master["Market"]))
    results = {}
    failures = []

    def report(pct, msg):
        if progress_callback:
            progress_callback(float(pct), 100.0, msg)

    report(0, "종목 목록과 시장 데이터를 준비하는 중...")
    index_cache = {}
    markets=sorted(set(market_map.values()))
    for j,mkt in enumerate(markets, start=1):
        index_cache[mkt] = fetch_market_index(mkt, start_date, end_date)
        report(2 + 3*j/max(len(markets),1), f"{mkt} 시장지수 준비 완료")

    symbols = master["Symbol"].tolist()
    total = len(symbols)
    report(5, f"총 {total:,}개 종목의 가격·거래량 데이터를 검색합니다.")

    def worker(sym):
        raw = fetch_ohlcv(sym, start_date, end_date)
        if raw is None or len(raw) < 100:
            return sym, None, "데이터 부족"
        mkt = market_map.get(sym, "KOSPI")
        df = calculate_indicators(raw, index_cache.get(mkt))
        return sym, df, None

    workers = min(16, max(1, total))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, sym) for sym in symbols]
        done = 0
        for fut in concurrent.futures.as_completed(futs):
            sym, df, err = fut.result()
            done += 1
            if df is None:
                failures.append({"Symbol": sym, "Name": names.get(sym, sym), "Reason": err})
            else:
                results[sym] = df
            pct=5 + 65*(done/max(total,1))
            report(pct, f"가격 데이터 검색 중 · {done:,}/{total:,} · {names.get(sym, sym)}")

    report(72, f"총 {len(results):,}개 종목의 상승 패턴과 상대순위를 계산합니다.")
    results = apply_cross_sectional_features(results)

    items=list(results.items())
    n_items=len(items)
    for i,(sym,df) in enumerate(items, start=1):
        if df is not None and not df.empty:
            e=historical_setup_expectancy(df, lookback=320, neighbors=30)
            for k,v in {
                "HistoricalSetupScore":e["score"],"PastSimilarWin5":e["win5"],
                "PastSimilarAvg5":e["avg5"],"PastSimilarAvg20":e["avg20"],
                "PastSimilarSamples":e["samples"]
            }.items():
                results[sym].loc[results[sym].index[-1],k]=v
        pct=75 + 23*(i/max(n_items,1))
        report(pct, f"과거 유사상황 검증 중 · {i:,}/{n_items:,} · {names.get(sym, sym)}")

    report(100, f"검색 완료 · 분석 가능 {len(results):,}개 / 데이터 부족 {len(failures):,}개")
    return results, master, pd.DataFrame(failures), universe_label

# ============================================================
# 11. UI / SESSION
# ============================================================



def add_single_stock_score(df):
    """단일종목 화면용. 횡단면 대신 자기 과거 상대위치를 사용하되 상승 패턴도 동일하게 평가."""
    x=add_rise_pattern_features(df.copy())

    def hist_pct(col, higher=True):
        s=pd.to_numeric(x[col],errors="coerce")
        return s.rolling(120,min_periods=40).rank(pct=True,ascending=higher)*100

    x["RS20Pct"]=hist_pct("RelativeReturn20")
    x["RS60Pct"]=hist_pct("RelativeReturn60")
    x["MomentumPct"]=(0.6*hist_pct("Return20")+0.4*hist_pct("Return60"))
    x["VolumePct"]=hist_pct("VolumeRatio")
    x["LiquidityPct"]=hist_pct("TradingValue20")
    x["BreakoutPct"]=hist_pct("Breakout20")
    x["PullbackPct"]=(100-(x["Pullback20Pct"]-5).abs()*8).clip(0,100)
    x["TrendPct"]=(
        0.25*x["RS20Pct"]+0.15*x["RS60Pct"]+0.20*x["MomentumPct"]+
        0.15*x["BreakoutPct"]+0.10*x["PullbackPct"]+0.15*x["PatternScore"]
    ).clip(0,100)
    x["QualityScore"]=(
        0.22*x["TrendPct"]+0.20*x["MomentumPct"]+0.18*x["RS20Pct"]+
        0.10*x["RS60Pct"]+0.10*x["VolumePct"]+0.05*x["LiquidityPct"]+
        0.15*x["PatternScore"]
    ).clip(0,100)

    er=(100-(x["RSI14"]-58).abs()*3).clip(0,100)
    ed=(100-(x["DisparityEMA20"].clip(-10,20)-4).abs()*5).clip(0,100)
    ea=(100-(x["ATR_Pct"]*100-3.5).abs()*12).clip(0,100)
    ev=(100-(x["VolumeRatio"]-1.6).abs()*20).clip(0,100)
    x["EntryScore"]=(0.30*er+0.30*ed+0.20*ea+0.10*ev+0.10*x["PullbackPct"]).clip(0,100)
    x["FinalScore"]=(0.62*x["QualityScore"]+0.38*x["EntryScore"]).clip(0,100)
    x["FinalRankPct"]=np.nan
    x["TopBucket"]="단일종목"
    return x

def recommendation_summary(row, score, passed, reasons, min_score=65):
    """초보자가 바로 이해할 수 있는 최종 판단 + 추천 근거."""
    if score is None:
        return {
            "판단": "분석불가",
            "한줄판단": "필요한 데이터가 부족해서 추천 여부를 판단할 수 없습니다.",
            "추천이유": [],
            "주의점": ["지표 계산에 필요한 가격·거래량 데이터가 부족합니다."],
        }

    q = float(score.get("QualityScore", 0))
    e = float(score.get("EntryScore", 0))
    f = float(score.get("TotalScore", 0))
    rank = score.get("RankPct", np.nan)
    rs = row.get("RelativeReturn20", np.nan)
    rsi = row.get("RSI14", np.nan)
    disp = row.get("DisparityEMA20", np.nan)
    vol = row.get("VolumeRatio", np.nan)
    adx = row.get("ADX14", np.nan)
    atr = row.get("ATR_Pct", np.nan) * 100 if pd.notna(row.get("ATR_Pct", np.nan)) else np.nan
    regime = row.get("Regime", "데이터부족")
    hist_score = row.get("HistoricalSetupScore", np.nan)
    hist_win5 = row.get("PastSimilarWin5", np.nan)
    hist_avg5 = row.get("PastSimilarAvg5", np.nan)
    hist_n = row.get("PastSimilarSamples", 0)
    rise_potential = score.get("RisePotentialScore", np.nan)
    pattern = score.get("PatternType", row.get("RisePattern", "관찰형"))
    pattern_desc = score.get("PatternDescription", row.get("PatternDescription", ""))

    strengths = []
    cautions = []

    if row.get("Close", 0) > row.get("EMA20", np.inf) > row.get("EMA60", np.inf):
        strengths.append("상승 추세가 확인됩니다. 주가가 20일선과 60일선 위에 있습니다.")
    else:
        cautions.append("상승 추세 조건이 완전히 충족되지 않았습니다.")

    if pd.notna(rs) and rs > 0:
        strengths.append(f"시장보다 강합니다. 최근 20일 시장 대비 {rs:+.1f}% 앞섰습니다.")
    elif pd.notna(rs):
        cautions.append(f"시장보다 약합니다. 최근 20일 시장 대비 {rs:+.1f}%입니다.")

    if pd.notna(vol) and vol >= 1.2:
        strengths.append(f"거래가 붙고 있습니다. 평소보다 약 {vol:.1f}배 많은 거래량입니다.")
    elif pd.notna(vol):
        cautions.append(f"거래량이 평소 수준입니다. 현재 약 {vol:.1f}배입니다.")

    if pd.notna(rsi):
        if rsi <= 68:
            strengths.append(f"과열 부담이 크지 않습니다. 현재 RSI는 {rsi:.1f}입니다.")
        elif rsi <= 72:
            cautions.append(f"상승은 강하지만 과열에 가까워지고 있습니다. RSI {rsi:.1f}입니다.")
        else:
            cautions.append(f"단기 과열 신호가 있습니다. RSI {rsi:.1f}입니다.")

    if pd.notna(disp):
        if -2 <= disp <= 8:
            strengths.append(f"진입 가격이 비교적 편합니다. 20일선보다 {disp:+.1f}% 높습니다.")
        elif disp > 8:
            cautions.append(f"주가가 20일선보다 {disp:+.1f}% 높아 추격매수 부담이 있습니다.")
        else:
            cautions.append(f"20일선보다 {disp:+.1f}% 낮아 추세 확인이 더 필요합니다.")

    if pd.notna(adx):
        if adx >= 20:
            strengths.append(f"추세의 힘이 있습니다. 추세 강도 지표가 {adx:.1f}입니다.")
        else:
            cautions.append(f"추세의 힘이 약합니다. 추세 강도 지표가 {adx:.1f}입니다.")

    if pd.notna(atr):
        if atr <= 5:
            strengths.append(f"가격 변동 위험이 비교적 관리 가능한 수준입니다. 하루 평균 변동폭 약 {atr:.1f}%입니다.")
        elif atr > 7:
            cautions.append(f"변동성이 큽니다. 하루 평균 변동폭이 약 {atr:.1f}%라 손실 폭도 커질 수 있습니다.")

    if regime in ["강한 상승장", "상승장"]:
        strengths.append(f"시장 분위기가 {regime}이라 상승 추세 종목에 유리합니다.")
    elif regime in ["하락장", "강한 하락장"]:
        cautions.append(f"현재 시장 분위기가 {regime}이라 신규 진입에 불리합니다.")

    if pattern in ["상승 시작형","상승 지속형","돌파형","눌림 재상승형"]:
        strengths.append(f"현재 패턴은 **{pattern}**입니다. {pattern_desc}")
    else:
        cautions.append(pattern_desc)

    if pd.notna(rise_potential):
        if rise_potential >= 70:
            strengths.append(f"과거 결과를 포함한 상승 가능성 점수가 {rise_potential:.1f}점으로 높습니다.")
        elif rise_potential < 45:
            cautions.append(f"과거 결과를 포함한 상승 가능성 점수가 {rise_potential:.1f}점으로 낮습니다.")

    if pd.notna(hist_win5) and hist_n >= 6:
        if hist_win5 >= 60 and pd.notna(hist_avg5) and hist_avg5 > 0:
            strengths.append(f"과거 비슷한 상황 {int(hist_n)}번에서는 5일 뒤 상승한 비율이 {hist_win5:.0f}%였고 평균 수익은 {hist_avg5:+.1f}%였습니다.")
        elif hist_win5 < 45:
            cautions.append(f"과거 비슷한 상황에서는 5일 뒤 상승 비율이 {hist_win5:.0f}%로 낮았습니다.")
        else:
            cautions.append(f"과거 비슷한 상황의 5일 상승 비율은 {hist_win5:.0f}%로 뚜렷한 우위가 크지 않았습니다.")

    if pd.notna(rank):
        pct = rank * 100
        if pct <= 5:
            strengths.append("오늘 분석한 종목 중 최상위 5% 안에 들어갑니다.")
        elif pct <= 10:
            strengths.append("오늘 분석한 종목 중 상위 10% 안에 들어갑니다.")
        elif pct <= 20:
            strengths.append("오늘 분석한 종목 중 상위 20% 안에 들어갑니다.")
        elif pct > 20:
            cautions.append("좋은 요소가 있어도 오늘 전체 후보 중 상대순위가 낮습니다.")

    if f >= min_score and passed and (pd.isna(rise_potential) or rise_potential >= 60):
        decision = "추천"
        headline = "현재 조건과 과거 상승 결과가 함께 좋아 매수 검토 우선순위가 높은 후보입니다."
    elif f >= min_score and rise_potential >= 60:
        decision = "조건부 추천"
        headline = "상승 가능성은 높지만 현재 가격·시장·위험 조건 중 일부를 확인해야 합니다."
    elif q >= 70 and e < 60:
        decision = "관망"
        headline = "종목 자체는 좋은 편이지만 현재 가격은 진입하기에 아쉬운 상태입니다."
    elif q >= 70 and (pd.isna(rise_potential) or rise_potential >= 50):
        decision = "관망"
        headline = "좋은 요소는 있지만 과거 상승 결과나 현재 진입 조건이 충분히 강하지 않습니다."
    else:
        decision = "제외"
        headline = "현재 조건과 과거 상승 근거를 함께 봤을 때 우선순위가 낮습니다."

    # OBV/CVD 수급 흐름을 추천 이유/주의점에 반영
    flow_score=float(score.get("FlowScore",50))
    if score.get("OBVCVDBull",False):
        strengths.append("OBV와 CVD가 동시에 매수 우위라 거래량 흐름이 좋습니다.")
    elif flow_score >= 65:
        strengths.append("OBV/CVD 수급 흐름이 평균보다 강한 편입니다.")
    elif flow_score <= 35:
        cautions.append("OBV/CVD 수급 흐름이 약해 상승 지속 여부를 주의해야 합니다.")

    # 게이트 탈락 이유를 사람이 읽는 문장으로 변환
    reason_map = {
        "유동성 부족": "거래가 충분히 활발하지 않습니다.",
        "단기 과열": "단기적으로 너무 빠르게 올라 추격매수 위험이 있습니다.",
        "20일선과 너무 멀리 이격": "최근 상승폭이 커서 20일선 대비 가격 부담이 큽니다.",
        "변동성 과다": "가격 변동이 커서 손실 위험이 상대적으로 큽니다.",
        "거래량 급증 과열": "거래량이 지나치게 폭발해 단기 추격 위험이 있습니다.",
        "시장 국면 불리": "현재 시장 흐름이 신규 매수에 불리합니다.",
        "시장 대비 20일 약세": "최근 시장 전체보다 주가 흐름이 약합니다.",
        "상승 추세 미충족": "20일선과 60일선 기준 상승 추세가 충분히 확인되지 않습니다.",
        "추세 힘 부족": "방향성은 있어도 추세가 강하게 이어지는 상태는 아닙니다.",
        "당일 상대순위가 낮음": "오늘 후보 전체와 비교하면 우선순위가 낮습니다.",
    }
    gate_cautions = [reason_map.get(x, x) for x in reasons]
    cautions = gate_cautions + [x for x in cautions if x not in gate_cautions]

    return {
        "판단": decision,
        "한줄판단": headline,
        "추천이유": strengths[:6],
        "주의점": cautions[:6],
    }


def explain_stock(row, score, passed, reasons):
    """기존 코드와의 호환용: 한국어 설명 문자열을 반환."""
    rec = recommendation_summary(row, score, passed, reasons, DEFAULTS.get("min_score", 65) if "DEFAULTS" in globals() else 65)
    lines = [rec["한줄판단"]]
    lines += ["추천 이유: " + x for x in rec["추천이유"]]
    lines += ["주의점: " + x for x in rec["주의점"]]
    return lines[:10]


def korean_indicator_table(row, score):
    """내부 컬럼명을 사용자에게 그대로 노출하지 않기 위한 한국어 요약표."""
    def f(v, nd=1):
        return "-" if pd.isna(v) else f"{float(v):.{nd}f}"
    return pd.DataFrame([
        ["종목 자체 점수", f(score.get("QualityScore", np.nan)), "기업/종목의 추세·모멘텀·상대강도 등을 평가"],
        ["내일 진입 점수", f(score.get("EntryScore", np.nan)), "내일 시가에 들어가도 부담이 적은지 평가"],
        ["최종 점수", f(score.get("TotalScore", np.nan)), "현재 조건 72% + 과거 유사상황 통계 28%"],
        ["시장 대비 강도", f(row.get("RelativeReturn20", np.nan), 2) + "%", "최근 20일 시장보다 얼마나 강했는지"],
        ["추세 강도", f(row.get("ADX14", np.nan)), "추세가 얼마나 뚜렷한지"],
        ["과열 정도", f(row.get("RSI14", np.nan)), "높을수록 단기 과열 가능성이 커짐"],
        ["거래량 변화", f(row.get("VolumeRatio", np.nan), 2) + "배", "평소 거래량 대비 현재 거래량"],
        ["20일선과의 거리", f(row.get("DisparityEMA20", np.nan), 2) + "%", "너무 높으면 추격매수 위험"],
        ["가격 변동 위험", f(row.get("ATR_Pct", np.nan) * 100, 2) + "%", "하루 평균 가격 변동폭"],
        ["과거 유사상황 점수", f(row.get("HistoricalSetupScore", np.nan)), "현재와 비슷했던 과거 상황의 실제 결과를 반영"],
        ["과거 5일 상승 비율", f(row.get("PastSimilarWin5", np.nan)) + "%", "비슷한 상황 뒤 5일 상승 사례 비율"],
        ["과거 유사상황 수", f(row.get("PastSimilarSamples", np.nan),0) + "회", "통계에 사용한 과거 비슷한 사례 수"],
        ["상승 패턴", score.get("PatternType", row.get("RisePattern","관찰형")), score.get("PatternDescription", "")],
        ["패턴 강도", f(score.get("PatternScore", np.nan)), "현재 상승 패턴이 얼마나 뚜렷한지"],
        ["상승 가능성 점수", f(score.get("RisePotentialScore", np.nan)), "현재와 비슷했던 과거 결과까지 반영한 점수"],
        ["OBV 매수세", f(score.get("OBVScore", row.get("OBVPct", np.nan))), "거래량이 가격 상승 방향으로 누적되는지"],
        ["CVD 매수세(근사)", f(score.get("CVDScore", row.get("CVDPct", np.nan))), "일봉 OHLCV로 추정한 매수·매도 압력"],
        ["OBV+CVD 동시 강세", "예" if score.get("OBVCVDBull",False) else "아니오", "두 거래량 흐름이 동시에 매수 우위인지"],
        ["거래량 흐름 점수", f(score.get("FlowScore", row.get("FlowScore", np.nan))), "OBV와 CVD를 합친 수급 점수"],
        ["시장 분위기", row.get("Regime", "데이터부족"), "전체 시장이 상승/하락 중인지"],
    ], columns=["평가 항목", "현재 값", "쉽게 말하면"])


DEFAULTS={
    "min_trading_value":2e9,"max_rsi":72.0,"max_disparity":12.0,"max_atr_ratio":0.07,
    "min_score":65,"market_filter":True,"require_relative_strength":True,"require_trend":True,
    "max_volume_ratio":4.5,"min_adx":15.0,"max_rank_pct":0.20,
}
if "settings" not in st.session_state: st.session_state["settings"]=DEFAULTS.copy()
st.title("📈 KRX 종가매매 전략 연구소 V9.7")
st.caption("t일 종가에서만 판단 → 기본 체결은 t+1일 시가. 기존 상승 패턴·거래량에 OBV + CVD(일봉 근사 수급)를 추가해 다음날 상승 후보를 평가합니다.")

with st.sidebar:
    st.header("⚙️ 공통 전략 설정")
    st.caption("숫자만 보지 않고 ‘왜 사는지 / 왜 기다리는지 / 왜 제외하는지’를 함께 판단합니다.")
    s=st.session_state["settings"]
    s["min_score"]=st.slider("최종점수 기준",50,95,int(s["min_score"]),5,help="종목 자체 점수와 내일 진입 점수를 합친 최종 기준입니다.")
    s["max_rank_pct"]=st.slider("허용할 종목 순위",0.05,0.50,float(s["max_rank_pct"]),0.05,format="%.0f%%",help="예: 상위 20%로 설정하면 오늘 후보 중 상위 20%까지만 매수 검토합니다.")
    s["min_trading_value"]=st.number_input("20일 평균 거래대금(원)",0,100_000_000_000,int(s["min_trading_value"]),1_000_000_000,help="거래가 너무 적어 원하는 가격에 사고팔기 어려운 종목을 제외합니다.")
    s["max_rsi"]=st.slider("단기 과열 기준",60,85,int(s["max_rsi"]),1,help="이 숫자보다 높으면 최근 상승이 너무 빠른 것으로 보고 추격매수를 피합니다.")
    s["max_disparity"]=st.slider("20일 평균가격과 최대 거리",5,30,int(s["max_disparity"]),1,help="현재 가격이 최근 평균가격에서 너무 멀리 떨어진 종목을 피합니다.")
    s["max_atr_ratio"]=st.slider("가격 변동 위험 상한",3,15,int(s["max_atr_ratio"]*100),1)/100
    s["max_volume_ratio"]=st.slider("거래량 과열 상한",2.0,8.0,float(s["max_volume_ratio"]),0.5,help="거래량이 갑자기 폭발한 날에는 추격매수를 피합니다.")
    s["min_adx"]=st.slider("최소 추세 강도",10,30,int(s["min_adx"]),1,help="추세가 너무 약하면 매수 후보에서 제외합니다.")
    s["market_filter"]=st.checkbox("하락장에서는 신규 매수 막기",bool(s["market_filter"]))
    s["require_relative_strength"]=st.checkbox("시장보다 강한 종목만 보기",bool(s["require_relative_strength"]))
    s["require_trend"]=st.checkbox("상승 추세 종목만 보기",bool(s["require_trend"]))
    if "wf_threshold" in st.session_state:
        st.info(f"워크포워드가 고른 기준: {st.session_state['wf_threshold']}점")

tab1,tab2,tab3,tab4,tab5=st.tabs(["🔍 오늘의 종목","📊 추천 이유","📈 백테스트","🧪 기대값·검증","🛡️ 무결성"])

with tab1:
    st.header("🔍 오늘 실제로 오를 가능성이 높은 종목 찾기")
    st.write("단순히 점수가 높은 종목이 아니라 **현재 상승 패턴 + 시장 대비 강도 + 거래량 + 진입가격 + 과거 비슷한 상황의 실제 결과**를 함께 봅니다.")

    st.info(
        "💡 최종 순위는 ① 종목 자체의 상승 구조 ② 내일 시가 진입 적합성 ③ "
        "현재와 비슷했던 과거 상황의 실제 5일·20일 결과를 함께 사용합니다."
    )

    market=st.selectbox(
        "분석할 시장",
        ["KOSPI","KOSDAQ","KOSPI+KOSDAQ"],
        key="scr_market",
        format_func=lambda x: {"KOSPI":"🇰🇷 코스피","KOSDAQ":"🟦 코스닥","KOSPI+KOSDAQ":"🇰🇷 코스피 + 코스닥 전체"}.get(x,x),
        help="전체를 선택하면 코스피와 코스닥을 동시에 검색한 뒤 두 시장의 종목을 같은 기준으로 비교합니다."
    )
    sample=st.number_input(
        "분석할 종목 수 (0 = 전체)",0,3000,300,10,key="scr_sample",
        help="0이면 선택한 시장의 전체 종목을 검색합니다. 처음에는 300~500개로 테스트하면 빠릅니다."
    )

    st.subheader("🎯 이번 검색에서 특히 찾는 상승 형태")
    pattern_cols=st.columns(4)
    pattern_help=[
        ("🚀 상승 시작형","추세가 살아나며 상승 힘이 붙는 구간"),
        ("📈 상승 지속형","이미 오른 뒤에도 추세·강도가 유지되는 구간"),
        ("突破 돌파형","최근 고점을 넘고 거래량이 동반되는 구간"),
        ("🔄 눌림 재상승형","상승 추세 안에서 조정 후 다시 올라갈 구간"),
    ]
    for col,(title,desc) in zip(pattern_cols,pattern_help):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    if st.button("🚀 오늘의 종목 찾기",type="primary",use_container_width=True):
        start=(datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)-timedelta(days=300)).strftime("%Y-%m-%d")
        end=datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None).strftime("%Y-%m-%d")

        progress=st.progress(0, text="검색 준비 중...")
        status=st.empty()

        def update_progress(done,total,msg):
            pct=float(done)/max(float(total),1.0)
            progress.progress(min(max(pct,0),1), text=f"🔎 {msg}  ·  {pct*100:.0f}%")
            status.caption(msg)

        data,master,failures,label=build_universe(
            market,start,end,int(sample) if sample else 0,
            progress_callback=update_progress
        )

        # 데이터 수집 이후에도 실제 진행률을 계속 보여준다.
        scored_items=[]
        total_data=max(len(data),1)
        for idx,(sym,df) in enumerate(data.items(),start=1):
            if df.empty:
                continue
            row=df.iloc[-1]
            score=score_row(row)
            if score is None:
                continue
            passed,reasons=entry_gate(row,s)
            name=master.loc[master["Symbol"]==sym,"Name"].iloc[0] if not master.loc[master["Symbol"]==sym].empty else sym
            scored_items.append((sym,row,score,passed,reasons,name))

            pct=98.0+1.5*(idx/total_data)
            progress.progress(min(pct/100,0.995),text=f"📊 최종 상승 가능성 계산 중... {idx:,}/{len(data):,} · {pct:.1f}%")

        # 최종점수로 다시 순위를 계산한다.
        scored_items.sort(key=lambda x:x[2]["TotalScore"],reverse=True)
        total_scored=max(len(scored_items),1)
        for i,(sym,row,score,passed,reasons,name) in enumerate(scored_items,start=1):
            rank_pct=i/total_scored
            score["RankPct"]=rank_pct
            score["TopBucket"]=(
                "상위 5%" if rank_pct<=0.05 else
                "상위 10%" if rank_pct<=0.10 else
                "상위 20%" if rank_pct<=0.20 else "하위 80%"
            )

        progress.progress(1.0,text="✅ 종목 검색·상승 가능성 분석 완료")
        status.success(f"검색 완료 · 분석 가능 {len(scored_items):,}개 / 데이터 부족·제외 {len(failures):,}개")

        rows=[]
        for sym,row,score,passed,reasons,name in scored_items:
            rec=recommendation_summary(row,score,passed,reasons,s["min_score"])
            rank=score.get("RankPct",np.nan)
            rows.append({
                "당일 순위": int(round(rank*total_scored)) if pd.notna(rank) else np.nan,
                "상위 비율": score.get("TopBucket","분석불가"),
                "종목명":name,
                "종목코드":sym,
                "판단":rec["판단"],
                "상승 패턴":score.get("PatternType","관찰형"),
                "최종점수":round(score["TotalScore"],1),
                "종목 자체 점수":round(score["QualityScore"],1),
                "내일 진입 점수":round(score["EntryScore"],1),
                "상승 가능성":round(score.get("RisePotentialScore",50),1),
                "과거 5일 상승비율":round(score.get("PastSimilarWin5",np.nan),1),
                "과거 유사상황 수":int(score.get("PastSimilarSamples",0)),
                "시장 대비 강도":round(row.get("RelativeReturn20",np.nan),2),
                "과열 정도":round(row.get("RSI14",np.nan),1),
                "거래량":round(row.get("VolumeRatio",np.nan),2),
                "OBV 매수세":round(score.get("OBVScore",50),1),
                "CVD 매수세":round(score.get("CVDScore",50),1),
                "수급 흐름":round(score.get("FlowScore",50),1),
                "20일선과 거리":round(row.get("DisparityEMA20",np.nan),2),
                "가격 변동 위험":round(row.get("ATR_Pct",np.nan)*100,2),
                "시장 분위기":row.get("Regime","데이터부족"),
            })
            risk_levels=recommendation_risk_levels(row)
            if risk_levels:
                rows[-1].update({k:round(v,2) for k,v in risk_levels.items()})

        result_df=pd.DataFrame(rows)
        st.session_state["scan_result"]=result_df
        st.session_state["scan_details"]={sym:(row,score,passed,reasons,name) for sym,row,score,passed,reasons,name in scored_items}

        if result_df.empty:
            st.warning("분석 가능한 종목이 없습니다. 분석 종목 수를 늘리거나 조건을 완화해 보세요.")
        else:
            st.subheader("🏆 오늘의 상위 후보")
            st.caption("종목명·종목코드와 함께 기준 진입가·손절가·1차/2차 익절가를 보여줍니다. 기준 진입가는 신호일 종가이며 실제 매수는 기본적으로 다음 거래일 시가입니다.")

            top=result_df.head(20).copy()
            st.dataframe(
                top,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "최종점수":st.column_config.NumberColumn(format="%.1f"),
                    "상승 가능성":st.column_config.NumberColumn(format="%.1f"),
                    "과거 5일 상승비율":st.column_config.NumberColumn(format="%.1f%%"),
                    "거래량":st.column_config.NumberColumn(format="%.2f배"),
                    "기준 진입가":st.column_config.NumberColumn(format="%d원"),
                    "손절가":st.column_config.NumberColumn(format="%d원"),
                    "1차 익절가":st.column_config.NumberColumn(format="%d원"),
                    "2차 익절가":st.column_config.NumberColumn(format="%d원"),
                    "손절폭":st.column_config.NumberColumn(format="%+.2f%%"),
                    "1차 익절폭":st.column_config.NumberColumn(format="%+.2f%%"),
                    "2차 익절폭":st.column_config.NumberColumn(format="%+.2f%%"),
                }
            )

            st.subheader("🟢 왜 이 종목인가?")
            # DataFrame의 한글 컬럼명을 itertuples() 속성으로 접근하면
            # pandas 버전에 따라 필드명이 달라져 AttributeError가 날 수 있다.
            # Series의 명시적 컬럼명 접근으로 고정한다.
            for _, r in top.head(5).iterrows():
                sym_key = r["종목코드"]
                detail=st.session_state["scan_details"].get(sym_key)
                if not detail:
                    continue
                row,score,passed,reasons,name=detail
                rec=recommendation_summary(row,score,passed,reasons,s["min_score"])
                rank_value = r["당일 순위"]
                pattern_value = r["상승 패턴"]
                title=f"{rank_value}위 · {name} · {rec['판단']} · {pattern_value} · 최종 {score['TotalScore']:.1f}"
                with st.expander(title,expanded=(rank_value==1)):
                    c1,c2,c3=st.columns(3)
                    c1.metric("종목 자체",f"{score['QualityScore']:.1f}")
                    c2.metric("내일 진입",f"{score['EntryScore']:.1f}")
                    c3.metric("상승 가능성",f"{score['RisePotentialScore']:.1f}")
                    risk_levels=recommendation_risk_levels(row)
                    if risk_levels:
                        r1,r2,r3,r4=st.columns(4)
                        r1.metric("기준 진입가",f"{risk_levels['기준 진입가']:,.0f}원")
                        r2.metric("손절가",f"{risk_levels['손절가']:,.0f}원",f"{risk_levels['손절폭']:+.2f}%")
                        r3.metric("1차 익절가",f"{risk_levels['1차 익절가']:,.0f}원",f"{risk_levels['1차 익절폭']:+.2f}%")
                        r4.metric("2차 익절가",f"{risk_levels['2차 익절가']:,.0f}원",f"{risk_levels['2차 익절폭']:+.2f}%")
                        st.caption("※ 실제 다음날 시가가 기준 진입가와 달라도 백테스트는 신호일 기준 손절/익절 가격을 고정합니다.")
                    st.markdown("**추천 이유**")
                    for reason in rec["추천이유"]:
                        st.write("✅ "+reason)
                    st.markdown("**주의할 점**")
                    for caution in rec["주의점"]:
                        st.write("⚠️ "+caution)
                    if score.get("PastSimilarSamples",0):
                        st.caption(
                            f"과거 유사상황 {int(score['PastSimilarSamples'])}회 · "
                            f"5일 상승비율 {score.get('PastSimilarWin5',np.nan):.1f}% · "
                            f"평균 5일 수익 {score.get('PastSimilarAvg5',np.nan):+.2f}% · "
                            f"평균 20일 수익 {score.get('PastSimilarAvg20',np.nan):+.2f}%"
                        )


with tab2:
    st.header("📊 왜 이 종목을 추천하거나 기다리라고 하나요?")
    st.caption("종목 하나를 고르면 숫자보다 먼저 ‘추천 이유 / 주의점 / 최종 판단’을 한국어 문장으로 보여줍니다.")
    sym_input=st.text_input("종목코드 또는 종목명", "005930", key="single_sym")
    days=st.selectbox("분석 기간",[260,520,780],index=1)
    if st.button("🔎 이 종목 자세히 보기",type="primary"):
        master=get_krx_stock_list(); sym=sym_input.strip().zfill(6)
        match=master[master["Symbol"]==sym]
        if match.empty: match=master[master["Name"].astype(str).str.contains(sym_input.strip(),na=False)]
        if match.empty:
            st.error("종목을 찾지 못했습니다.")
        else:
            sym=match.iloc[0]["Symbol"]; name=match.iloc[0]["Name"]; market=match.iloc[0]["Market"]
            start=(datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)-timedelta(days=days+180)).strftime("%Y-%m-%d")
            end=datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None).strftime("%Y-%m-%d")
            raw=fetch_ohlcv(sym,start,end); mkt=fetch_market_index(market,start,end)
            if raw is None: st.error("가격 데이터를 가져오지 못했습니다.")
            else:
                df=calculate_indicators(raw,mkt); tmp=add_single_stock_score(df)
                hist=historical_setup_expectancy(tmp, lookback=320, neighbors=30)
                for k,v in {
                    "HistoricalSetupScore":hist["score"],"PastSimilarWin5":hist["win5"],
                    "PastSimilarAvg5":hist["avg5"],"PastSimilarAvg20":hist["avg20"],
                    "PastSimilarSamples":hist["samples"]
                }.items():
                    tmp.loc[tmp.index[-1],k]=v
                row=tmp.iloc[-1]
                score=score_row(row); single_s=dict(s); single_s["max_rank_pct"]=1.0
                passed,reasons=entry_gate(row,single_s)
                rec=recommendation_summary(row,score,passed,reasons,s["min_score"])
                st.subheader(f"{name} ({sym}) · {market} · {tmp.index[-1].date()}")
                if score:
                    if rec["판단"]=="추천": st.success("🟢 " + rec["한줄판단"])
                    elif rec["판단"]=="조건부 추천": st.warning("🟠 " + rec["한줄판단"])
                    elif rec["판단"]=="관망": st.warning("🟡 " + rec["한줄판단"])
                    else: st.error("🔴 " + rec["한줄판단"])
                    a,b,c,d=st.columns(4)
                    a.metric("종목 자체 점수",f"{score['QualityScore']:.1f} / 100")
                    b.metric("내일 진입 점수",f"{score['EntryScore']:.1f} / 100")
                    c.metric("상승 가능성",f"{score['RisePotentialScore']:.1f} / 100")
                    d.metric("최종 점수",f"{score['TotalScore']:.1f} / 100")
                    st.info(f"현재 상승 패턴: **{score.get('PatternType','관찰형')}** · {score.get('PatternDescription','')}")
                    st.markdown("### 👍 추천하는 이유")
                    if rec["추천이유"]:
                        for x in rec["추천이유"]: st.write("• " + x)
                    else:
                        st.write("• 현재 설정에서 뚜렷한 추천 근거가 충분하지 않습니다.")
                    st.markdown("### ⚠️ 주의할 점")
                    if rec["주의점"]:
                        for x in rec["주의점"]: st.write("• " + x)
                    else:
                        st.write("• 특별히 확인된 주의점이 없습니다.")
                    st.markdown("### 📌 지표를 쉽게 읽는 법")
                    st.dataframe(korean_indicator_table(row,score),use_container_width=True,hide_index=True)
                    st.line_chart(tmp[["Close","EMA20","EMA60","EMA120"]].tail(250))
                    st.subheader("🎬 영상형 복합 신호(표준 지표 기반)")
                    video_cols=[c for c in [
                        "SuperTrend","SuperTrendDir","AwesomeOscillator","Momentum10","ROC10",
                        "PlusDI","MinusDI","MACD","MACD_Signal","VideoLongScore","VideoShortScore"
                    ] if c in tmp.columns]
                    if video_cols:
                        vr=tmp.iloc[-1]
                        vc1,vc2,vc3=st.columns(3)
                        vc1.metric("LONG 점수",f"{int(vr.get('VideoLongScore',0))}/6")
                        vc2.metric("SHORT 점수",f"{int(vr.get('VideoShortScore',0))}/6")
                        vc3.metric("SuperTrend", "상승" if vr.get("SuperTrendDir",0)>0 else "하락")
                        st.dataframe(tmp[video_cols].tail(20),use_container_width=True,hide_index=False)
                        st.caption("영상의 실제 원본 수식은 확인할 수 없으므로, 화면에 표시된 DMI/AO/Momentum/ROC/MACD/SuperTrend 구조를 표준 수식으로 재구성한 연구용 신호입니다.")

with tab3:
    st.header("📈 실제 포트폴리오 백테스트")
    st.warning("역사적 유니버스 CSV를 넣지 않으면 현재 KRX 종목목록을 사용하므로 생존자 편향이 남을 수 있습니다.")
    c1,c2,c3=st.columns(3)
    with c1:
        bt_market=st.selectbox("시장",["KOSPI","KOSDAQ","KOSPI+KOSDAQ"],key="bt_market",format_func=lambda x: {"KOSPI":"🇰🇷 코스피","KOSDAQ":"🟦 코스닥","KOSPI+KOSDAQ":"🇰🇷 코스피 + 코스닥 통합"}.get(x,x))
        years=st.slider("백테스트 기간(년)",1,8,3)
        sample=st.number_input("종목 수 (0=전체)",0,3000,100,10,key="bt_sample")
    with c2:
        execution=st.selectbox("체결",["다음날 시가","당일 종가(연구용)"],key="bt_execution")
        max_concurrent=st.slider("최대 동시보유",1,20,5)
        max_hold=st.slider("최대 보유일",5,60,30,5)
    with c3:
        capital=st.number_input("초기자본",1_000_000,1_000_000_000,10_000_000,1_000_000)
        partial=st.selectbox("1차 익절 비중",["30%","50%","100%"])
        fee=st.number_input("매수/매도 수수료 %",0.0,0.2,0.015,0.005)/100
        tax=st.number_input("매도 거래세 %",0.0,1.0,0.20,0.05)/100
        slip=st.number_input("슬리피지 %",0.0,1.0,0.10,0.05)/100
    use_wf=st.checkbox("워크포워드가 선택한 기준 사용",True,help="신호 연구 탭에서 선택된 점수가 있으면 실제 포트폴리오 백테스트의 최소점수로 연결합니다.")
    hist_file=st.file_uploader("선택: 역사적 유니버스 CSV (Symbol,Name,Market,StartDate,EndDate)",type=["csv"],key="hist_u")
    if st.button("🚀 백테스트 실행",type="primary"):
        start=(datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)-timedelta(days=int(years*365+300))).strftime("%Y-%m-%d")
        end=datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None).strftime("%Y-%m-%d")
        hist_u,_=load_historical_universe(hist_file)
        bt_progress=st.progress(0, text="백테스트 준비 중...")
        bt_status=st.empty()

        def bt_data_progress(done,total,msg):
            pct=float(done)/max(float(total),1.0)
            bt_progress.progress(min(pct*0.40,0.40), text=f"📥 {msg} · {pct*100:.0f}%")
            bt_status.caption(msg)

        data,master,failures,universe_label=build_universe(
            bt_market,start,end,int(sample) if sample else 0,hist_u,
            progress_callback=bt_data_progress
        )
        if not data:
            bt_progress.progress(1.0,text="❌ 백테스트 데이터가 없습니다.")
            bt_status.error("백테스트 데이터가 없습니다.")
            st.error("백테스트 데이터가 없습니다.")
        else:
            bt_settings=dict(s)
            if use_wf and "wf_threshold" in st.session_state: bt_settings["min_score"]=st.session_state["wf_threshold"]
            partial_val=0.3 if partial=="30%" else 0.5 if partial=="50%" else 1.0

            def bt_engine_progress(pct,current_date,done,total):
                overall=0.40 + 0.60*(pct/100.0)
                bt_progress.progress(min(overall,1.0), text=f"📈 백테스트 계산 중 · {done:,}/{total:,}일 · {pct}%")
                bt_status.caption(f"처리 중: {pd.Timestamp(current_date).date()} · 보유 {len(st.session_state.get('_bt_preview_positions', [])) if False else 0}건")

            result=run_backtest(
                data,bt_settings,float(capital),max_concurrent,max_hold,partial_val,
                fee,tax,slip,"next_open" if execution=="다음날 시가" else "close",
                progress_callback=bt_engine_progress
            )
            bt_progress.progress(1.0,text="✅ 백테스트 완료")
            bt_status.success(f"백테스트 완료 · {len(data):,}개 종목 · {universe_label}")
            result["meta"].update({"universe_label":universe_label,"loaded_symbols":len(data),"requested_symbols":len(master),"failed_symbols":len(failures),"used_min_score":bt_settings["min_score"]})
            st.session_state["bt"]=result; st.session_state["bt_failures"]=failures
    if "bt" in st.session_state:
        r=st.session_state["bt"]; m=r["meta"]
        st.info(f"사용 최소점수 {m.get('used_min_score','-')} | 유니버스 {m['universe_label']} | 요청 {m['requested_symbols']} | 성공 {m['loaded_symbols']}")
        st.caption("읽는 순서: ① 최종 자산·누적 수익률 → ② 최대 낙폭(MDD) → ③ 실현 승률·수익/손실 비율 → ④ 거래내역의 종목명·종목코드·진입/청산 가격")
        cols=st.columns(10)
        vals=[("최종 자산",f"{m['final_asset']:,.0f}"),("누적 수익률",f"{m['total_return']:+.2f}%"),("연평균 수익률",f"{m['cagr']:+.2f}%"),
              ("최대 낙폭",f"{m['mdd']:.2f}%"),("실현 승률",f"{m['win_rate']:.1f}%"),("수익/손실 비율",f"{m['profit_factor']:.2f}"),
              ("위험 대비 수익",f"{m['sharpe']:.2f}"),("하락 위험 대비 수익",f"{m['sortino']:.2f}"),("최대 연속 손실",str(m["max_consecutive_losses"])),("아직 끝나지 않은 거래",str(m["open_positions"]))]
        for c,(lab,val) in zip(cols,vals):c.metric(lab,val)
        st.subheader("📖 결과 해석 방법")
        backtest_glossary()
        st.subheader("자산곡선"); st.line_chart(r["portfolio"].set_index("Date")[["TotalAsset"]].rename(columns={"TotalAsset":"총자산"}))
        st.subheader("현금 / 예약금 / 투자금"); st.line_chart(r["portfolio"].set_index("Date")[["Cash","ReservedCash","InvestedValue"]].rename(columns={"Cash":"현금","ReservedCash":"예약금","InvestedValue":"투자금"}))
        st.subheader("📋 거래가 끝난 종목"); st.caption("종목명과 종목코드를 같이 표시합니다."); st.dataframe(korean_trade_table(r["closed"],master),use_container_width=True,hide_index=True)
        st.subheader("📋 아직 끝나지 않은 거래 — 실현 성과에서 제외"); st.dataframe(korean_trade_table(r["open"],master),use_container_width=True,hide_index=True)
        if not r["meta"]["gap_performance"].empty:
            st.subheader("실제 다음날 시가가 얼마나 달랐는지"); st.dataframe(r["meta"]["gap_performance"],use_container_width=True,hide_index=True)
        if not r["meta"]["regime_performance"].empty:
            st.subheader("시장 분위기별 성과"); st.dataframe(r["meta"]["regime_performance"],use_container_width=True,hide_index=True)
        if not r["meta"]["year_performance"].empty:
            st.subheader("연도별 성과"); st.dataframe(r["meta"]["year_performance"],use_container_width=True,hide_index=True)
        st.subheader("예약금 반환 / 실행 실패"); st.dataframe(korean_trade_table(r["rejected"],master),use_container_width=True,hide_index=True)

with tab4:
    st.header("🧪 기대값 + 상위 5/10/20% 비교 + 진짜 워크포워드")
    st.caption("점수 70점 이상 같은 절대 컷 하나만 보는 대신, 당일 종목 중 상위 5%·10%·20%의 실제 선도수익을 비교합니다.")
    rw_market=st.selectbox(
        "연구 시장",
        ["KOSPI","KOSDAQ","KOSPI+KOSDAQ"],
        key="rw_market",
        format_func=lambda x: {"KOSPI":"🇰🇷 코스피","KOSDAQ":"🟦 코스닥","KOSPI+KOSDAQ":"🇰🇷 코스피 + 코스닥 통합"}.get(x,x)
    )
    rw_compare_all=st.checkbox("코스피·코스닥·통합 3가지를 한 번에 비교",False)
    rw_years=st.slider("연구기간(년)",1,8,3,key="rw_years")
    rw_sample=st.number_input("연구 종목 수",0,3000,100,10,key="rw_sample")
    if st.button("🔬 신호 연구 실행",type="primary"):
        start=(datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)-timedelta(days=int(rw_years*365+350))).strftime("%Y-%m-%d")
        end=datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None).strftime("%Y-%m-%d")
        markets_to_run=["KOSPI","KOSDAQ","KOSPI+KOSDAQ"] if rw_compare_all else [rw_market]
        rp=st.progress(0,text="신호 연구 준비 중...")
        rs=st.empty()
        fwd_results={}
        for mi,mkt in enumerate(markets_to_run,1):
            rs.info(f"{mkt} 데이터 수집·계산 중... ({mi}/{len(markets_to_run)})")
            data,master,failures,label=build_universe(mkt,start,end,int(rw_sample) if rw_sample else 0)
            fwd_results[mkt]=forward_return_research(data,s)
            rp.progress(mi/len(markets_to_run),text=f"🔬 {mkt} 신호 연구 완료 · {mi}/{len(markets_to_run)}")
        st.session_state["fwd_results"]=fwd_results
        st.session_state["fwd"]=fwd_results.get(rw_market,pd.DataFrame())
        st.session_state["fwd_label"]=rw_market
        rs.success("기대값·워크포워드 연구 완료")
    if "fwd_results" in st.session_state and st.session_state["fwd_results"]:
        fwd_results=st.session_state["fwd_results"]
        compare_rows=[]
        for mkt,fx in fwd_results.items():
            if fx is None or fx.empty:
                compare_rows.append({"시장":mkt,"신호수":0,"평균20일수익":np.nan,"20일승률":np.nan})
            else:
                z=fx["Fwd20D"].dropna()
                compare_rows.append({"시장":mkt,"신호수":len(z),"평균20일수익":z.mean(),"20일승률":(z>0).mean()*100})
        if len(compare_rows)>1:
            st.subheader("📊 시장별 기대값 비교")
            st.dataframe(pd.DataFrame(compare_rows),use_container_width=True,hide_index=True)
        fwd=st.session_state["fwd"]
        st.info(f"분석 신호 {len(fwd):,}개 | {st.session_state['fwd_label']}")
        if not fwd.empty:
            st.subheader("상위 5% / 10% / 20% 실제 기대값")
            st.dataframe(top_bucket_summary(fwd,"Fwd20D"),use_container_width=True,hide_index=True)
            st.caption("상위 5%는 상위 10%·20%와 중복됩니다. 따라서 각 행은 '그 순위 이내'의 성과를 의미합니다.")
            st.subheader("종목 자체 점수 vs 내일 진입점수")
            qbin=pd.qcut(fwd["QualityScore"],5,duplicates="drop")
            ebin=pd.qcut(fwd["EntryScore"],5,duplicates="drop")
            qe=fwd.assign(종목자체구간=qbin,진입구간=ebin)
            st.dataframe(qe.groupby("종목자체구간",observed=False).agg(신호수=("Fwd20D","count"),평균20일수익=("Fwd20D","mean"),승률=("Fwd20D",lambda z:(z>0).mean()*100)).reset_index(),use_container_width=True,hide_index=True)
            st.dataframe(qe.groupby("진입구간",observed=False).agg(신호수=("Fwd20D","count"),평균20일수익=("Fwd20D","mean"),승률=("Fwd20D",lambda z:(z>0).mean()*100)).reset_index(),use_container_width=True,hide_index=True)
            st.subheader("갭별 성과")
            gx=fwd.copy(); gx["갭구간"]=pd.cut(gx["GapToEntryPct"],[-np.inf,-3,0,3,7,np.inf],labels=["≤-3%","-3~0%","0~3%","3~7%","≥7%"])
            st.dataframe(gx.groupby("갭구간",observed=False).agg(신호수=("Fwd20D","count"),평균20일수익=("Fwd20D","mean"),승률=("Fwd20D",lambda z:(z>0).mean()*100)).reset_index(),use_container_width=True,hide_index=True)
            st.subheader("시장 분위기별 성과")
            st.dataframe(fwd.groupby("Regime").agg(신호수=("Fwd20D","count"),평균5일=("Fwd5D","mean"),평균20일=("Fwd20D","mean"),승률=("Fwd20D",lambda z:(z>0).mean()*100)).reset_index(),use_container_width=True,hide_index=True)
            st.subheader("날짜 기준 워크포워드")
            wf,chosen=rolling_walk_forward(fwd,train_days=504,test_days=126,step_days=126)
            if not wf.empty:
                st.dataframe(wf,use_container_width=True,hide_index=True)
                st.session_state["wf_threshold"]=chosen
                st.success(f"학습구간에서 선택된 기준의 중앙값: {chosen}점 → 실제 포트폴리오 백테스트에 연결 가능")
                st.caption("각 검증구간은 학습구간 이후의 날짜만 사용합니다. 검증구간 성과는 기준 선택에 사용하지 않습니다.")
            else: st.warning("최소 약 2년 학습 + 6개월 검증 데이터가 필요합니다.")

with tab5:
    st.header("🛡️ 백테스트 무결성 감사")
    audits=[
        ("미래정보 누수","PASS","신호는 t일 종가까지, 체결은 t+1일 시가. 다음날 갭은 연구/사후평가에만 사용."),
        ("상대순위 누수","PASS","각 날짜에 존재한 종목끼리만 상대순위를 계산하고 미래 날짜를 참조하지 않음."),
        ("점수 분리","PASS","종목 자체 점수(Quality)와 내일 진입점수(Entry)를 분리 후 최종점수로 결합."),
        ("상위 5/10/20%","PASS","절대 70점 컷 외에 당일 상대순위별 실제 기대값을 비교."),
        ("워크포워드","PASS","날짜 순 train/test 분리, 학습구간에서만 기준 선택."),
        ("실제 포트폴리오 연결","PASS","워크포워드 선택 기준을 백테스트 최소점수로 자동 연결 가능. 코스피/코스닥/통합 시장을 선택할 수 있음."),
        ("수수료 중복","PASS","매수/매도 비용은 거래 원가에 각각 1회 반영."),
        ("예약금 정산","PASS","체결·취소·슬롯 부족 시 예약금을 반환."),
        ("미청산 분리","PASS","미청산은 실현 승률/PF에서 제외하고 평가손익을 별도 표시."),
        ("역사적 유니버스","LIMITATION","CSV가 없으면 현재 KRX 목록을 사용하므로 생존자 편향 가능."),
        ("갭 성과","PASS","실제 t+1 시가와 신호일 종가의 갭을 구간별로 분석."),
        ("시장 국면/연도별","PASS","완료 포지션을 신호 국면과 연도별로 분해."),
        ("Sharpe/Sortino","PASS","일별 자산수익률로 연환산 위험조정 성과를 계산."),
        ("최대 연속 손실","PASS","완전 청산 포지션 기준 최대 연속 손실 횟수 계산."),
        ("실제 비용","PASS","수수료·거래세·슬리피지 입력값을 체결 가격과 현금흐름에 반영."),
    ]
    st.dataframe(pd.DataFrame(audits,columns=["검증항목","결과","설명"]),use_container_width=True,hide_index=True)
    st.info("무결성 PASS는 코드상 설계 기준입니다. 데이터 오류, 거래정지, 호가단위, 시장충격 등 실제 체결 환경까지 보증하는 것은 아닙니다.")

st.markdown("---")
st.caption("연구용 소프트웨어입니다. 백테스트 성과는 미래 수익을 보장하지 않습니다.")
