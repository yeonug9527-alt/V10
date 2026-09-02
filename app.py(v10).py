import concurrent.futures
from datetime import datetime, timedelta
import io
import math
import uuid
import zoneinfo

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# Streamlit 페이지 기본 설정
# ============================================================
st.set_page_config(
    page_title="KRX V10+ 급등 전조 스캐너 & 백테스트 연구소",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1rem; padding-bottom: 2rem;}
.stAlert {margin-top: 0.5rem; margin-bottom: 0.5rem;}
.metric-card {background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 12px; margin-bottom: 10px;}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 0. 유틸리티 및 시간 함수
# ============================================================
def get_kst_now():
    return datetime.now(zoneinfo.ZoneInfo("Asia/Seoul"))


def safe_div(a, b, default=np.nan):
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
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calculate_cmf(df, period=20):
    """CMF 계산. 계산 불가 구간은 NaN을 유지한다."""
    high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]
    price_range = (high - low).replace(0, np.nan)
    mf_multiplier = ((close - low) - (high - close)) / price_range
    mf_volume = mf_multiplier * volume

    # FIX: 초기 구간과 거래량 합계 0을 0으로 치환하지 않고 NaN으로 유지
    volume_sum = volume.rolling(period, min_periods=period).sum()
    mf_sum = mf_volume.rolling(period, min_periods=period).sum()
    cmf = mf_sum / volume_sum.replace(0, np.nan)
    return cmf


# ============================================================
# 1. 유니버스 및 가격 데이터 수집
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_krx_stock_list():
    """현재 KRX 종목 목록 수집 및 보통주 필터링"""
    try:
        df = fdr.StockListing("KRX").copy()
        symbol_col = next(
            (c for c in ["Symbol", "Code", "종목코드"] if c in df.columns), None
        )
        name_col = next(
            (c for c in ["Name", "종목명", " 종목명"] if c in df.columns), None
        )
        market_col = next((c for c in ["Market", "시장"] if c in df.columns), None)

        if not symbol_col or not name_col:
            return pd.DataFrame(columns=["Symbol", "Name", "Market"])

        rename_map = {symbol_col: "Symbol", name_col: "Name"}
        if market_col:
            rename_map[market_col] = "Market"
        df = df.rename(columns=rename_map)

        df["Symbol"] = df["Symbol"].astype(str).str.extract(r"(\d{6})")[0]
        df = df.dropna(subset=["Symbol"])

        if "Market" in df.columns:
            df = df[df["Market"].isin(["KOSPI", "KOSDAQ"])].copy()
        else:
            df["Market"] = "KOSPI/KOSDAQ"

        exclude_keywords = [
            "스팩", "SPAC", "ETF", "ETN", "리츠", "REIT", "인프라",
            "신주인수권", "KODEX", "TIGER", "ARIRANG", "KBSTAR", "HANARO",
            "KOSEF", "인버스", "레버리지",
        ]
        pattern = "|".join(exclude_keywords)
        mask = ~df["Name"].astype(str).str.upper().str.contains(
            pattern, regex=True, na=False
        )
        df = df[mask].copy()

        pref_mask = df["Name"].astype(str).str.contains(
            r"우$|우B$|우C$|우\(전환\)$", regex=True, na=False
        )
        df = df[~pref_mask].copy()

        return df[["Symbol", "Name", "Market"]].drop_duplicates("Symbol").reset_index(drop=True)
    except Exception as e:
        st.error(f"KRX 종목 목록 로드 중 오류 발생: {e}")
        return pd.DataFrame(columns=["Symbol", "Name", "Market"])


def parse_historical_universe_csv(uploaded_file):
    """역사적 유니버스 CSV 파싱"""
    if uploaded_file is None:
        return None, "현재 KRX 유니버스 사용 중"
    try:
        df = pd.read_csv(uploaded_file)
        alias = {
            "Code": "Symbol", "Ticker": "Symbol", "종목코드": "Symbol",
            "시작일": "StartDate", "종료일": "EndDate",
        }
        df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})

        if "Symbol" not in df.columns:
            return None, "CSV 파일에 Code 또는 Symbol 컬럼이 존재하지 않습니다."

        df["Symbol"] = df["Symbol"].astype(str).str.extract(r"(\d{6})")[0]
        df = df.dropna(subset=["Symbol"]).copy()

        if "Name" not in df.columns:
            df["Name"] = df["Symbol"]
        if "Market" not in df.columns:
            df["Market"] = "ALL"

        # FIX: 시작/종료일이 없으면 실제로 전체 기간 유효하게 설정
        if "StartDate" in df.columns:
            df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
        else:
            df["StartDate"] = pd.Timestamp.min

        if "EndDate" in df.columns:
            df["EndDate"] = pd.to_datetime(df["EndDate"], errors="coerce")
        else:
            df["EndDate"] = pd.Timestamp.max

        df["StartDate"] = df["StartDate"].fillna(pd.Timestamp.min)
        df["EndDate"] = df["EndDate"].fillna(pd.Timestamp.max)
        df["Market"] = df["Market"].astype(str).str.upper().str.strip()

        return (
            df[["Symbol", "Name", "Market", "StartDate", "EndDate"]].drop_duplicates(),
            "역사적 유니버스 CSV 적용 완료",
        )
    except Exception as e:
        return None, f"CSV 업로드 파싱 오류: {e}"


def filter_universe_by_market_and_date(df, market_choice, analysis_date=None):
    """현재/역사적 유니버스 공통 시장·기간 필터."""
    if df is None or df.empty:
        return df

    x = df.copy()
    if market_choice != "KOSPI + KOSDAQ":
        # FIX: CSV의 Market도 실제 선택 시장에 적용. ALL은 특정 시장을 알 수 없으므로 제외.
        x = x[x["Market"].astype(str).str.upper() == market_choice].copy()
    else:
        x = x[x["Market"].astype(str).str.upper().isin(["KOSPI", "KOSDAQ", "ALL"])].copy()

    if analysis_date is not None and {"StartDate", "EndDate"}.issubset(x.columns):
        dt = pd.Timestamp(analysis_date)
        x = x[(x["StartDate"] <= dt) & (dt <= x["EndDate"])].copy()
    return x


def build_historical_validity_map(hist_universe):
    """종목별 유효기간 목록 생성."""
    if hist_universe is None or hist_universe.empty:
        return {}
    result = {}
    for sym, g in hist_universe.groupby("Symbol"):
        result[sym] = list(zip(g["StartDate"], g["EndDate"]))
    return result


def is_symbol_valid_on_date(sym, dt, validity_map=None):
    if not validity_map:
        return True
    if sym not in validity_map:
        return False
    ts = pd.Timestamp(dt)
    return any(start <= ts <= end for start, end in validity_map[sym])


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv_data(symbol, start_date, end_date):
    """단일 종목 OHLCV + 원천 거래대금 보존."""
    try:
        raw = fdr.DataReader(symbol, start_date, end_date)
        if raw is None or raw.empty:
            return None, "데이터 없음", "없음"

        raw = raw.copy()
        amount_candidates = ["Amount", "amount", "거래대금", "TradingValue", "Value"]
        amount_col = next((c for c in amount_candidates if c in raw.columns), None)

        # FIX: OHLCV를 먼저 잘라내지 않고 원천 거래대금 컬럼을 먼저 확보
        if amount_col is not None:
            amount_source = f"원천 {amount_col}"
            raw["Amount"] = pd.to_numeric(raw[amount_col], errors="coerce")
        else:
            amount_source = "Close×Volume 계산"
            raw["Amount"] = (
                pd.to_numeric(raw["Close"], errors="coerce")
                * pd.to_numeric(raw["Volume"], errors="coerce")
            )

        required = ["Open", "High", "Low", "Close", "Volume", "Amount"]
        if not all(col in raw.columns for col in required):
            return None, "필수 OHLCV/거래대금 컬럼 누락", amount_source

        df = raw[required].copy()
        df = df[~df.index.duplicated(keep="last")].sort_index()
        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # FIX: 거래대금 결측을 0으로 치환하지 않음
        if df["Amount"].isna().any():
            return None, "거래대금 결측 포함 - 분석불가", amount_source

        if len(df) < 30:
            return None, f"데이터 행 수 부족 ({len(df)}행)", amount_source

        return df, f"성공 ({amount_source})", amount_source
    except Exception as e:
        return None, f"수집 에러: {str(e)}", "오류"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_index(market_type, start_date, end_date):
    ticker = "KS11" if market_type == "KOSPI" else "KQ11"
    try:
        df = fdr.DataReader(ticker, start_date, end_date)
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return df["Close"].dropna().sort_index()
    except Exception:
        return None


# ============================================================
# 2. 기술 지표 및 급등 전조 특징 계산
# ============================================================
def calculate_technical_features(df, market_index_series=None):
    if df is None or len(df) < 20:
        return None

    x = df.copy()
    close, high, low, open_px, volume, amount = (
        x["Close"], x["High"], x["Low"], x["Open"], x["Volume"], x["Amount"]
    )

    # FIX: Return60 누락으로 발생하던 KeyError 수정
    for d in [1, 3, 5, 10, 20, 60]:
        x[f"Return{d}"] = close.pct_change(d) * 100

    x["SMA20"] = close.rolling(20, min_periods=20).mean()
    x["SMA60"] = close.rolling(60, min_periods=60).mean()
    x["EMA20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    x["EMA60"] = close.ewm(span=60, adjust=False, min_periods=60).mean()
    x["EMA120"] = close.ewm(span=120, adjust=False, min_periods=120).mean()
    x["EMA20_Slope"] = x["EMA20"] - x["EMA20"].shift(5)
    x["EMA60_Slope"] = x["EMA60"] - x["EMA60"].shift(10)
    x["DisparityEMA20"] = (close / x["EMA20"] - 1) * 100

    x["High20"] = high.rolling(20, min_periods=20).max()
    x["Low20"] = low.rolling(20, min_periods=20).min()
    x["High60"] = high.rolling(60, min_periods=60).max()
    x["BoxWidth20"] = (x["High20"] - x["Low20"]) / x["Low20"].replace(0, np.nan) * 100
    x["BoxWidth5"] = (
        (high.rolling(5).max() - low.rolling(5).min())
        / low.rolling(5).min().replace(0, np.nan) * 100
    )
    x["BoxPosition20"] = (close - x["Low20"]) / (x["High20"] - x["Low20"]).replace(0, np.nan)
    x["BoxWidthRatio5_20"] = x["BoxWidth5"] / x["BoxWidth20"].replace(0, np.nan)

    x["HigherLow5"] = low.rolling(5).min() > low.shift(5).rolling(5).min()
    x["HigherHigh5"] = high.rolling(5).max() > high.shift(5).rolling(5).max()
    x["DistToHigh20"] = (x["High20"] - close) / close * 100
    x["DistToHigh60"] = (x["High60"] - close) / close * 100

    low5 = low.rolling(5).min()
    x["LowSlope5"] = (low5 - low5.shift(5)) / low5.shift(5).replace(0, np.nan) * 100
    high5 = high.rolling(5).max()
    x["HighSlope5"] = (high5 - high5.shift(5)) / high5.shift(5).replace(0, np.nan) * 100

    x["ATR14"] = wilder_atr(x, 14)
    x["ATR_Ratio"] = x["ATR14"] / close * 100
    x["Vol5"] = close.pct_change().rolling(5).std()
    x["Vol20"] = close.pct_change().rolling(20).std()
    x["VolRatio5_20"] = x["Vol5"] / x["Vol20"].replace(0, np.nan)

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    x["BB_Width"] = (2 * bb_std * 2) / bb_mid.replace(0, np.nan)

    x["VolumeMA20"] = volume.rolling(20, min_periods=20).mean()
    x["VolumeMA60"] = volume.rolling(60, min_periods=60).mean()
    x["VolumeRatio20"] = volume / x["VolumeMA20"].replace(0, np.nan)
    x["VolumeRatio60"] = volume / x["VolumeMA60"].replace(0, np.nan)
    x["AmountMA20"] = amount.rolling(20, min_periods=20).mean()
    x["AmountRatio20"] = amount / x["AmountMA20"].replace(0, np.nan)
    x["AmountChange5"] = (amount - amount.shift(5)) / amount.shift(5).replace(0, np.nan) * 100

    obv_dir = np.sign(close.diff()).fillna(0)
    x["OBV"] = (obv_dir * volume).cumsum()
    x["OBV_MA20"] = x["OBV"].rolling(20, min_periods=20).mean()
    x["OBV_Slope5"] = x["OBV"] - x["OBV"].shift(5)
    x["CMF20"] = calculate_cmf(x, 20)

    x["RSI14"] = wilder_rsi(close, 14)
    candle_range = (high - low).replace(0, np.nan)
    x["CloseLocation"] = (close - low) / candle_range
    x["UpperWickRatio"] = (high - np.maximum(open_px, close)) / candle_range

    if market_index_series is not None and not market_index_series.empty:
        m_series = market_index_series.reindex(x.index).ffill()
        x["MarketReturn5"] = m_series.pct_change(5) * 100
        x["MarketReturn20"] = m_series.pct_change(20) * 100
        x["MarketReturn60"] = m_series.pct_change(60) * 100
        x["RelReturn5"] = x["Return5"] - x["MarketReturn5"]
        x["RelReturn20"] = x["Return20"] - x["MarketReturn20"]
        x["RelReturn60"] = x["Return60"] - x["MarketReturn60"]
    else:
        for d in [5, 20, 60]:
            x[f"MarketReturn{d}"] = np.nan
            x[f"RelReturn{d}"] = np.nan

    return x


# ============================================================
# 3. 데이터 품질 검증 및 통계
# ============================================================
def audit_stock_data_quality(raw_df, sym_name):
    reasons = []
    if raw_df is None or raw_df.empty:
        return False, ["데이터 없음"]

    if len(raw_df) < 40:
        reasons.append(f"데이터 부족 ({len(raw_df)}행)")
    if raw_df.index.duplicated().any():
        reasons.append("중복 거래일 존재")
    if not raw_df.index.is_monotonic_increasing:
        reasons.append("날짜 정렬 오류")

    kst_today = get_kst_now().date()
    if (raw_df.index > pd.Timestamp(kst_today)).any():
        reasons.append("미래 날짜 포함 오류")

    null_cols = raw_df[["Open", "High", "Low", "Close", "Volume"]].isnull().sum()
    if null_cols.sum() > 0:
        reasons.append("OHLCV 결측값 포함")

    # FIX: Amount 결측은 0으로 대체하지 않고 품질 오류로 처리
    if "Amount" not in raw_df.columns or raw_df["Amount"].isna().any():
        reasons.append("거래대금 결측 - 분석불가")

    if (raw_df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        reasons.append("가격 0 이하 비정상 데이터")
    if (raw_df["High"] < raw_df["Low"]).any():
        reasons.append("High < Low 비정상 데이터")
    if (raw_df["Volume"] < 0).any() or (raw_df["Amount"] < 0).any():
        reasons.append("음수 거래량/거래대금 오류")

    recent_v = raw_df["Volume"].tail(5)
    if (recent_v == 0).all():
        reasons.append("최근 5일 연속 거래량 0 (거래정지 추정)")

    return len(reasons) == 0, reasons


# ============================================================
# 4. 가격 전조 점수 설계
# ============================================================
def calculate_precursor_subscores(row, missing_handling="지표 제외 후 나머지로 계산"):
    def eval_score(condition_dict):
        scores, weights = [], []
        for val, (score_val, w) in condition_dict.items():
            if pd.isna(val):
                if missing_handling in ["해당 종목 제외", "분석불가 표시"]:
                    return np.nan
                continue
            scores.append(score_val)
            weights.append(w)
        if not scores:
            return np.nan
        return float(np.sum(np.array(scores) * np.array(weights)) / np.sum(weights))

    comp_dict = {
        row.get("VolRatio5_20"): (100 if row.get("VolRatio5_20", 1) < 0.7 else 50, 0.3),
        row.get("BoxWidthRatio5_20"): (100 if row.get("BoxWidthRatio5_20", 1) < 0.6 else 50, 0.3),
        row.get("ATR_Ratio"): (100 if row.get("ATR_Ratio", 5) < 3.5 else 40, 0.2),
        row.get("BB_Width"): (100 if row.get("BB_Width", 0.2) < 0.1 else 50, 0.2),
    }
    score_A = eval_score(comp_dict)

    prep_dict = {
        row.get("DistToHigh20"): (100 if row.get("DistToHigh20", 10) < 3.0 else 40, 0.25),
        row.get("HigherLow5"): (100 if bool(row.get("HigherLow5", False)) else 20, 0.25),
        row.get("HigherHigh5"): (100 if bool(row.get("HigherHigh5", False)) else 30, 0.2),
        row.get("CloseLocation"): (100 if row.get("CloseLocation", 0.5) > 0.7 else 40, 0.3),
    }
    score_B = eval_score(prep_dict)

    flow_dict = {
        row.get("VolumeRatio20"): (100 if row.get("VolumeRatio20", 1) >= 1.5 else 40, 0.25),
        row.get("AmountRatio20"): (100 if row.get("AmountRatio20", 1) >= 1.5 else 40, 0.25),
        row.get("OBV_Slope5"): (100 if row.get("OBV_Slope5", 0) > 0 else 30, 0.25),
        row.get("CMF20"): (100 if row.get("CMF20", 0) > 0.05 else 30, 0.25),
    }
    score_C = eval_score(flow_dict)

    risk_dict = {
        row.get("Return1"): (100 if row.get("Return1", 0) < 5.0 else 30, 0.25),
        row.get("Return5"): (100 if row.get("Return5", 0) < 12.0 else 30, 0.25),
        row.get("DisparityEMA20"): (100 if row.get("DisparityEMA20", 0) < 8.0 else 20, 0.25),
        row.get("RSI14"): (100 if row.get("RSI14", 50) < 68.0 else 20, 0.25),
    }
    score_D = eval_score(risk_dict)

    if any(pd.isna(v) for v in [score_A, score_B, score_C, score_D]):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    total_score = score_A * 0.25 + score_B * 0.30 + score_C * 0.25 + score_D * 0.20
    return total_score, score_A, score_B, score_C, score_D


def classify_surge_pattern(row):
    if row.get("BoxWidthRatio5_20", 1) < 0.6 and row.get("DistToHigh20", 10) < 4:
        return "압축 후 상단 접근"
    elif row.get("AmountRatio20", 1) >= 2.0:
        return "거래대금 증가형"
    elif bool(row.get("HigherLow5", False)) and row.get("LowSlope5", 0) > 1.5:
        return "저점 상승형"
    elif row.get("DistToHigh20", 10) < 2.0:
        return "돌파 준비형"
    elif row.get("Return5", 0) < 0 and row.get("DisparityEMA20", 0) > 0:
        return "눌림 후 재상승 준비형"
    elif row.get("Close", 0) > row.get("EMA20", 0) and row.get("EMA20", 0) > row.get("EMA60", 0):
        return "추세 유지형"
    return "관찰형"


# ============================================================
# 5. 필수 게이트
# ============================================================
def evaluate_essential_gates(row, settings, data_last_date, ref_date):
    reasons = []

    min_amount_krw = settings["min_amount_100m"] * 100_000_000
    amount_ma20 = row.get("AmountMA20", np.nan)

    # FIX: 거래대금 결측을 비교식에서 조용히 통과시키지 않음
    if pd.isna(amount_ma20):
        reasons.append("거래대금 계산 불가")
    elif amount_ma20 < min_amount_krw:
        reasons.append("최소 거래대금 미달")

    if data_last_date != ref_date:
        reasons.append("데이터 기준일 불일치")

    if pd.isna(row.get("Return1", np.nan)):
        reasons.append("당일 수익률 계산 불가")
    elif row.get("Return1", 0) > settings["max_today_return"]:
        reasons.append("당일 급등 과열 제외")

    if row.get("Close", 0) <= 0 or pd.isna(row.get("Close")):
        reasons.append("비정상 종가 데이터")

    return len(reasons) == 0, reasons


# ============================================================
# 6. V9 백테스트 엔진 통합
# ============================================================
def _release_reservation(order, reserved_cash, trade_logs, current_date, reason):
    """예약금을 정확히 반환하고 로그에 기록."""
    amount = float(order.get("reserved_amount", 0.0))
    reserved_cash = max(0.0, reserved_cash - amount)
    trade_logs.append(
        {
            "Date": current_date,
            "Symbol": order.get("sym"),
            "Type": "RESERVATION_RELEASE",
            "Reserved_Amount": amount,
            "Reason": reason,
        }
    )
    return reserved_cash


def _next_symbol_date(df, current_date):
    """종목 데이터에서 실제 다음 행의 거래일을 반환."""
    try:
        idx = df.index.get_loc(current_date)
        if isinstance(idx, slice):
            return None
        if idx + 1 < len(df.index):
            return df.index[idx + 1]
    except Exception:
        pass
    return None


def run_v9_backtest_engine(
    universe_dict,
    settings,
    initial_capital=10_000_000,
    max_concurrent=5,
    fee_rate=0.00015,
    tax_rate=0.0020,
    slippage=0.001,
    validity_map=None,
):
    """t일 종가 신호 -> 해당 종목의 실제 다음 거래일 시가 체결."""
    all_dates = sorted(set(d for df in universe_dict.values() for d in df.index))
    if len(all_dates) < 2:
        return None, None, None, None

    cash = float(initial_capital)
    reserved_cash = 0.0
    pending_orders = []
    positions = {}
    closed_trades = []
    trade_logs = []
    equity_curve = []

    for i, current_date in enumerate(all_dates):
        is_last_day = i == len(all_dates) - 1

        # ----------------------------------------------------
        # A. 실제 다음 거래일 시가 체결
        # ----------------------------------------------------
        if pending_orders:
            pending_orders.sort(key=lambda x: x["score"], reverse=True)
            new_pending = []
            available_slots = max(0, max_concurrent - len(positions))

            for order in pending_orders:
                sym = order["sym"]
                df = universe_dict.get(sym)
                expected_date = order.get("expected_entry_date")

                # FIX: 예약 주문은 미리 계산한 '해당 종목의 실제 다음 행'에서만 체결
                if expected_date is None:
                    reserved_cash = _release_reservation(
                        order, reserved_cash, trade_logs, current_date, "실제 다음 거래일 확인 불가"
                    )
                    continue

                if current_date < expected_date:
                    new_pending.append(order)
                    continue

                if current_date > expected_date:
                    reserved_cash = _release_reservation(
                        order, reserved_cash, trade_logs, current_date, "예정 체결일 경과로 주문 취소"
                    )
                    continue

                if df is None or current_date not in df.index or not is_symbol_valid_on_date(sym, current_date, validity_map):
                    reserved_cash = _release_reservation(
                        order, reserved_cash, trade_logs, current_date, "종목 데이터/유효기간 오류"
                    )
                    continue

                if available_slots <= 0:
                    # FIX: 슬롯 초과 시 예약금 즉시 반환
                    reserved_cash = _release_reservation(
                        order, reserved_cash, trade_logs, current_date, "동시보유 슬롯 초과"
                    )
                    continue

                # FIX: 체결 직전 예약금은 해제하고 실제 매수비용만 cash에서 차감
                reserved_cash = max(0.0, reserved_cash - float(order["reserved_amount"]))
                row = df.loc[current_date]
                open_px = pd.to_numeric(pd.Series([row["Open"]]), errors="coerce").iloc[0]

                if pd.isna(open_px) or open_px <= 0:
                    reserved_cash = max(0.0, reserved_cash)
                    trade_logs.append(
                        {
                            "Date": current_date,
                            "Symbol": sym,
                            "Type": "ORDER_FAIL",
                            "Reserved_Amount": order["reserved_amount"],
                            "Reason": "비정상 시가 - 예약금 반환",
                        }
                    )
                    continue

                exec_price = float(open_px) * (1 + slippage)
                alloc_cash = float(order["reserved_amount"])
                qty = int((alloc_cash * 0.98) / exec_price)

                if qty <= 0:
                    trade_logs.append(
                        {
                            "Date": current_date,
                            "Symbol": sym,
                            "Type": "ORDER_FAIL",
                            "Reserved_Amount": alloc_cash,
                            "Reason": "주문가능 수량 0 - 예약금 반환",
                        }
                    )
                    continue

                buy_gross = qty * exec_price
                buy_fee = buy_gross * fee_rate
                total_cost = buy_gross + buy_fee

                if cash < total_cost:
                    trade_logs.append(
                        {
                            "Date": current_date,
                            "Symbol": sym,
                            "Type": "ORDER_FAIL",
                            "Reserved_Amount": alloc_cash,
                            "Reason": "현금 부족 - 예약금 반환",
                        }
                    )
                    continue

                cash -= total_cost
                available_slots -= 1
                pid = str(uuid.uuid4())[:8]

                positions[pid] = {
                    "pid": pid,
                    "sym": sym,
                    "entry_date": current_date,
                    "signal_date": order["signal_date"],
                    "entry_price": exec_price,
                    "qty": qty,
                    "initial_qty": qty,
                    "buy_fee": buy_fee,
                    "atr": order["atr"],
                    "stop_loss": order["stop_loss"],
                    "tp1": order["tp1"],
                    "tp2": order["tp2"],
                    "holding_days": 0,
                    "partial_done": False,
                    "realized_pnl": 0.0,
                }

                trade_logs.append(
                    {
                        "Date": current_date,
                        "Symbol": sym,
                        "Type": "BUY",
                        "Price": exec_price,
                        "Qty": qty,
                        "Signal_Date": order["signal_date"],
                        "Reason": "t일 종가 신호 -> 실제 다음 거래일 시가 체결",
                    }
                )

            pending_orders = new_pending

        # ----------------------------------------------------
        # B. 포지션 청산
        # ----------------------------------------------------
        for pid in list(positions.keys()):
            pos = positions[pid]
            df = universe_dict[pos["sym"]]
            if current_date not in df.index or not is_symbol_valid_on_date(pos["sym"], current_date, validity_map):
                continue

            row = df.loc[current_date]
            if current_date != pos["entry_date"]:
                pos["holding_days"] += 1

            exit_reason = None
            exit_price = None
            sell_qty = 0
            raw_exit_price = None

            stop_loss = pos["stop_loss"]
            if pd.isna(stop_loss) or stop_loss <= 0:
                continue

            # FIX: 갭 하락이면 손절가가 아니라 실제 시가에 체결
            open_px = float(row["Open"]) if pd.notna(row["Open"]) else np.nan
            low_px = float(row["Low"]) if pd.notna(row["Low"]) else np.nan
            high_px = float(row["High"]) if pd.notna(row["High"]) else np.nan

            if pd.isna(open_px) or open_px <= 0 or pd.isna(low_px) or pd.isna(high_px):
                continue

            if open_px <= stop_loss:
                exit_reason = "1. 손절 갭하락(시가 체결)"
                raw_exit_price = open_px
                sell_qty = pos["qty"]
            elif low_px <= stop_loss:
                exit_reason = "1. 손절(손절가 체결)"
                raw_exit_price = stop_loss
                sell_qty = pos["qty"]
            elif high_px >= pos["tp2"]:
                exit_reason = "2. 2차 목표가"
                raw_exit_price = pos["tp2"]
                sell_qty = pos["qty"]
            elif high_px >= pos["tp1"] and not pos["partial_done"]:
                exit_reason = "3. 1차 목표가 (부분익절)"
                raw_exit_price = pos["tp1"]
                sell_qty = max(1, int(pos["initial_qty"] * 0.5))
                sell_qty = min(sell_qty, pos["qty"])
            elif row["Close"] < row["EMA20"] and row.get("EMA20_Slope", 0) < 0:
                exit_reason = "4. EMA20 추세 이탈"
                raw_exit_price = float(row["Close"])
                sell_qty = pos["qty"]
            elif pos["holding_days"] >= settings["max_holding_days"]:
                exit_reason = "5. 최대 보유기간 만료"
                raw_exit_price = float(row["Close"])
                sell_qty = pos["qty"]

            if exit_reason and sell_qty > 0 and raw_exit_price is not None and raw_exit_price > 0:
                # FIX: 모든 매도 체결가격에 동일하게 실제 체결가격 기준 슬리피지 적용
                exit_price = raw_exit_price * (1 - slippage)
                sell_gross = sell_qty * exit_price
                sell_fee = sell_gross * fee_rate
                sell_tax = sell_gross * tax_rate
                buy_fee_alloc = pos["buy_fee"] * (sell_qty / pos["initial_qty"])

                net_pnl = (
                    (exit_price - pos["entry_price"]) * sell_qty
                    - buy_fee_alloc - sell_fee - sell_tax
                )
                cash += sell_gross - sell_fee - sell_tax
                pos["realized_pnl"] += net_pnl

                trade_logs.append(
                    {
                        "Date": current_date,
                        "Symbol": pos["sym"],
                        "Type": "SELL",
                        "Price": exit_price,
                        "Raw_Trigger_Price": raw_exit_price,
                        "Qty": sell_qty,
                        "Reason": exit_reason,
                        "Net_PnL": net_pnl,
                    }
                )

                if sell_qty >= pos["qty"]:
                    invested = pos["entry_price"] * pos["initial_qty"]
                    closed_trades.append(
                        {
                            "Position_ID": pid,
                            "Symbol": pos["sym"],
                            "Entry_Date": pos["entry_date"],
                            "Signal_Date": pos["signal_date"],
                            "Exit_Date": current_date,
                            "Holding_Days": pos["holding_days"],
                            "Entry_Price": pos["entry_price"],
                            "Exit_Price": exit_price,
                            "Net_PnL": pos["realized_pnl"],
                            "Return_Pct": safe_div(pos["realized_pnl"], invested, 0.0) * 100,
                            "Exit_Reason": exit_reason,
                        }
                    )
                    del positions[pid]
                else:
                    pos["qty"] -= sell_qty
                    pos["partial_done"] = True

        # ----------------------------------------------------
        # C. t일 종가 신호
        # ----------------------------------------------------
        if not is_last_day:
            active_symbols = {p["sym"] for p in positions.values()}
            pending_symbols = {p["sym"] for p in pending_orders}
            candidates = []

            for sym, df in universe_dict.items():
                if sym in active_symbols or sym in pending_symbols or current_date not in df.index:
                    continue
                if not is_symbol_valid_on_date(sym, current_date, validity_map):
                    continue

                # FIX: 실제 다음 거래일이 존재해야만 신호를 예약
                expected_entry_date = _next_symbol_date(df, current_date)
                if expected_entry_date is None:
                    continue
                if not is_symbol_valid_on_date(sym, expected_entry_date, validity_map):
                    continue

                row = df.loc[current_date]
                passed, _ = evaluate_essential_gates(row, settings, current_date, current_date)
                tot_score, _, _, _, _ = calculate_precursor_subscores(
                    row, settings["missing_handling"]
                )

                if passed and pd.notna(tot_score) and tot_score >= settings["min_score"]:
                    candidates.append((tot_score, sym, row, expected_entry_date))

            candidates.sort(key=lambda x: x[0], reverse=True)
            free_slots = max(0, max_concurrent - len(positions) - len(pending_orders))
            free_cash = max(0.0, cash - reserved_cash)

            if free_slots > 0 and free_cash > 0:
                per_order_cash = free_cash / free_slots
                for score_val, sym, row, expected_entry_date in candidates[:free_slots]:
                    atr = row.get("ATR14", np.nan)
                    close_px = row.get("Close", np.nan)
                    if pd.isna(atr) or pd.isna(close_px) or atr <= 0 or close_px <= 0:
                        continue

                    stop_px = close_px - 2.0 * atr
                    risk = close_px - stop_px
                    tp1_px = close_px + 2.0 * risk
                    tp2_px = close_px + 3.5 * risk
                    reserve_amt = min(per_order_cash, max(0.0, cash - reserved_cash))
                    if reserve_amt <= 0:
                        break

                    pending_orders.append(
                        {
                            "sym": sym,
                            "score": score_val,
                            "signal_date": current_date,
                            "expected_entry_date": expected_entry_date,
                            "atr": float(atr),
                            "stop_loss": float(stop_px),
                            "tp1": float(tp1_px),
                            "tp2": float(tp2_px),
                            "reserved_amount": float(reserve_amt),
                        }
                    )
                    reserved_cash += reserve_amt

        # ----------------------------------------------------
        # D. 자산 평가
        # ----------------------------------------------------
        stock_eval = 0.0
        for pos in positions.values():
            df = universe_dict[pos["sym"]]
            if current_date in df.index:
                stock_eval += pos["qty"] * float(df.loc[current_date, "Close"])

        equity_curve.append(
            {
                "Date": current_date,
                "Cash": cash,
                "ReservedCash": reserved_cash,
                "StockValue": stock_eval,
                "TotalAsset": cash + stock_eval,
                "OpenPositions": len(positions),
                "PendingOrders": len(pending_orders),
            }
        )

    # FIX: 백테스트 종료 시 남은 예약 주문은 전부 취소하고 예약금 반환
    if pending_orders:
        final_date = all_dates[-1]
        for order in pending_orders:
            reserved_cash = _release_reservation(
                order, reserved_cash, trade_logs, final_date, "백테스트 종료 - 미체결 예약 주문 취소"
            )
        pending_orders = []

    # FIX: 마지막에 예약금이 남지 않는지 방어 검증
    if reserved_cash > 1e-9:
        trade_logs.append(
            {
                "Date": all_dates[-1],
                "Type": "INTEGRITY_WARNING",
                "Reason": f"예약금 잔액 {reserved_cash:,.2f}원 - 강제 0 처리",
            }
        )
        reserved_cash = 0.0

    return (
        pd.DataFrame(equity_curve),
        pd.DataFrame(closed_trades),
        pd.DataFrame(trade_logs),
        positions,
    )


# ============================================================
# 7. 워크포워드 검증
# ============================================================
def run_walk_forward_validation(universe_dict, settings, train_years=1, test_years=1, validity_map=None):
    """시간순 Train/Test 분리. 실제 다음 거래일 수익률을 사용."""
    all_dates = sorted(set(d for df in universe_dict.values() for d in df.index))
    if not all_dates:
        return pd.DataFrame()

    min_year = all_dates[0].year
    max_year = all_dates[-1].year
    results = []

    # FIX: range 끝점을 포함하지 않아 마지막 검증 구간이 누락되는 문제 보정
    for start_y in range(min_year, max_year - train_years - test_years + 2):
        train_start = pd.Timestamp(f"{start_y}-01-01")
        train_end = pd.Timestamp(f"{start_y + train_years - 1}-12-31")
        test_start = pd.Timestamp(f"{start_y + train_years}-01-01")
        test_end = pd.Timestamp(f"{start_y + train_years + test_years - 1}-12-31")

        train_signals = []
        for sym, df in universe_dict.items():
            sub = df[(df.index >= train_start) & (df.index <= train_end)]
            for dt, row in sub.iterrows():
                if not is_symbol_valid_on_date(sym, dt, validity_map):
                    continue
                next_dt = _next_symbol_date(df, dt)
                if (
                    next_dt is None
                    or next_dt > train_end
                    or not is_symbol_valid_on_date(sym, next_dt, validity_map)
                ):
                    # FIX: Train 신호의 다음 봉이 Test/유효기간 밖이면 Train 성과에서 제외
                    continue
                score, _, _, _, _ = calculate_precursor_subscores(row, settings["missing_handling"])
                if pd.notna(score):
                    next_ret = (df.loc[next_dt, "Open"] / row["Close"] - 1) * 100
                    train_signals.append({"Score": score, "NextRet": next_ret})

        train_df = pd.DataFrame(train_signals)
        if train_df.empty:
            continue

        best_min_score = 65
        best_avg_ret = -np.inf
        for cand_score in [60, 65, 70, 75, 80]:
            filtered = train_df[train_df["Score"] >= cand_score]
            if len(filtered) >= 10:
                avg_r = filtered["NextRet"].mean()
                if avg_r > best_avg_ret:
                    best_avg_ret = avg_r
                    best_min_score = cand_score

        test_signals = []
        for sym, df in universe_dict.items():
            sub = df[(df.index >= test_start) & (df.index <= test_end)]
            for dt, row in sub.iterrows():
                if not is_symbol_valid_on_date(sym, dt, validity_map):
                    continue
                score, _, _, _, _ = calculate_precursor_subscores(row, settings["missing_handling"])
                if pd.notna(score) and score >= best_min_score:
                    next_dt = _next_symbol_date(df, dt)
                    # FIX: Test 기간 밖의 다음 봉 수익률을 Test 성과에 포함하지 않음
                    if (
                        next_dt is None
                        or next_dt > test_end
                        or not is_symbol_valid_on_date(sym, next_dt, validity_map)
                    ):
                        continue
                    next_ret = (df.loc[next_dt, "Open"] / row["Close"] - 1) * 100
                    test_signals.append({"Score": score, "NextRet": next_ret})

        test_df = pd.DataFrame(test_signals)
        results.append(
            {
                "학습기간": f"{train_start.date()}~{train_end.date()}",
                "검증기간": f"{test_start.date()}~{test_end.date()}",
                "선택된 최소점수": best_min_score,
                "학습 신호수": len(train_df),
                "검증 신호수": len(test_df),
                "검증 평균수익률(%)": test_df["NextRet"].mean() if not test_df.empty else np.nan,
                "검증 승률(%)": (test_df["NextRet"] > 0).mean() * 100 if not test_df.empty else np.nan,
            }
        )

    return pd.DataFrame(results)


# ============================================================
# 8. 무결성 감사
# ============================================================
def run_integrity_audit(csv_applied=False):
    """실제로 구현된 로직만 PASS. 자동 검증하지 못하는 항목은 CHECK/LIMITATION."""
    audit_data = [
        ("미래정보 누수 방지", "PASS", "t일 종가 지표로 신호를 만들고 실제 다음 거래일 시가에서 체결"),
        ("실제 다음 거래일 체결", "PASS", "종목별 DataFrame의 실제 다음 index를 expected_entry_date로 사용"),
        ("시장지수 날짜 정렬", "PASS", "종목 날짜에 시장지수를 reindex 후 forward-fill"),
        ("가격 데이터 기준일 명시", "PASS", "실행시각과 가격 데이터 최신 거래일을 분리 표시"),
        ("상대수익률 60일 계산", "PASS", "Return60을 생성한 뒤 RelReturn60 계산"),
        ("워크포워드 시간순 분리", "PASS", "Train에서 선택한 점수를 Test에 순서대로 적용"),
        ("수수료 중복 차감 방지", "PASS", "매수/매도 비용을 각 체결에서 한 번만 차감"),
        ("거래세 차감 적용", "PASS", "매도 시 설정된 tax_rate 적용"),
        ("슬리피지 반영", "PASS", "매수 +슬리피지, 매도 -슬리피지"),
        ("체결 주문 예약금 반환", "PASS", "체결 직전 예약금 해제 후 실제 비용만 현금 차감"),
        ("슬롯 초과 주문 예약금 반환", "PASS", "슬롯 부족 주문 즉시 반환"),
        ("실패 주문 예약금 반환", "PASS", "비정상 시가/수량/현금 부족 시 반환 로그 기록"),
        ("마지막 날 미체결 예약금 처리", "PASS", "백테스트 종료 후 모든 pending reservation 반환"),
        ("손절 갭 하락 체결", "PASS", "시가가 손절가 이하이면 손절가가 아니라 실제 시가 사용"),
        ("손절가 터치 체결", "PASS", "시가는 손절가 위이고 저가가 손절가 이하이면 손절가 사용"),
        ("미청산 포지션 처리", "CHECK", "미청산 포지션은 실현 거래 성과에서 제외하며 평가자산에는 포함"),
        ("역사적 유니버스 실제 적용", "PASS" if csv_applied else "CHECK", "CSV 업로드 시 스캔/백테스트/WF 공통 유효기간 필터에 적용"),
        ("거래대금 원 단위 자동 검증", "CHECK", "원천 공급원의 단위를 코드에서 독립적으로 검증하지 못함; 화면에 명시"),
        ("CMF 결측값 0 대체 방지", "PASS", "CMF 계산 불가 구간과 거래량 합계 0을 NaN 유지"),
        ("생존자 편향", "LIMITATION", "현재 KRX 유니버스 사용 시 과거 생존자 편향 가능"),
    ]
    return pd.DataFrame(audit_data, columns=["감사 항목", "상태", "상세 설명"])


# ============================================================
# 9. Streamlit 메인 UI
# ============================================================
def main():
    kst_now = get_kst_now()
    st.title("📈 KRX V10+ 급등 전조 스캐너 & 백테스트 연구소")
    st.caption(
        f"실행 시각 (KST): {kst_now.strftime('%Y-%m-%d %H:%M:%S')} | 가격·거래량·시장상대강도 전용 검증 도구"
    )

    st.sidebar.header("⚙️ 분석 및 스캔 설정")
    market_choice = st.sidebar.selectbox("시장 선택", ["KOSPI + KOSDAQ", "KOSPI", "KOSDAQ"])

    uploaded_csv = st.sidebar.file_uploader("역사적 유니버스 CSV (선택)", type=["csv"])
    hist_universe, univ_msg = parse_historical_universe_csv(uploaded_csv)
    if hist_universe is None:
        st.sidebar.info("ℹ️ " + univ_msg)
        st.sidebar.warning("⚠️ 현재 종목 목록을 사용하므로 과거 백테스트에는 생존자 편향이 남을 수 있습니다.")
    else:
        st.sidebar.success("✅ " + univ_msg)

    st.sidebar.subheader("필수 게이트 조건")
    min_amount_100m = st.sidebar.number_input("최소 20일 평균 거래대금 (억원)", value=10, min_value=1)
    max_today_return = st.sidebar.number_input("당일 상승률 제한 (%) - 추격 방지", value=12.0, min_value=0.0)
    min_score = st.sidebar.slider("최소 급등 전조 점수", 0, 100, 65)
    missing_handling = st.sidebar.selectbox(
        "결측 지표 처리 방식",
        ["지표 제외 후 나머지로 계산", "해당 종목 제외", "분석불가 표시"],
    )
    sample_scan_count = st.sidebar.number_input("스캔 종목 수 제한 (0: 전체)", value=100, min_value=0)

    settings = {
        "min_amount_100m": min_amount_100m,
        "max_today_return": max_today_return,
        "min_score": min_score,
        "missing_handling": missing_handling,
        "max_holding_days": 10,
    }

    # FIX: 스캔 실행 전에는 KRX/시장지수/종목 가격 데이터를 조회하지 않음.
    # FIX: 기존 자동 스캔 구조를 버튼 기반 수동 스캔으로 변경.
    end_dt = kst_now.date()
    start_dt = end_dt - timedelta(days=365 * 3)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🔍 급등 전조 후보 스캐너",
            "📊 후보 상세 분석",
            "⚖️ 전략 조건별 비교",
            "🔄 V9 통합 백테스트",
            "⏳ 워크포워드 검증",
            "🛡️ 무결성 & 데이터 품질",
        ]
    )

    # FIX: 스캔 결과를 session_state에 보존하여 버튼 클릭 이후 탭 전환 시에도 재수집하지 않음.
    if "v10_scan_universe_data" not in st.session_state:
        st.session_state.v10_scan_universe_data = {}
    if "v10_scan_quality_logs" not in st.session_state:
        st.session_state.v10_scan_quality_logs = []
    if "v10_scan_excluded_logs" not in st.session_state:
        st.session_state.v10_scan_excluded_logs = []
    if "v10_scan_stock_list" not in st.session_state:
        st.session_state.v10_scan_stock_list = pd.DataFrame()
    if "v10_scan_ref_date" not in st.session_state:
        st.session_state.v10_scan_ref_date = None
    if "v10_scan_using_historical" not in st.session_state:
        st.session_state.v10_scan_using_historical = False
    if "v10_scan_meta" not in st.session_state:
        st.session_state.v10_scan_meta = pd.DataFrame()
    if "v10_scan_signature" not in st.session_state:
        st.session_state.v10_scan_signature = None

    current_signature = (
        market_choice,
        int(min_amount_100m),
        float(max_today_return),
        int(min_score),
        missing_handling,
        int(sample_scan_count),
        uploaded_csv.name if uploaded_csv is not None else None,
        len(hist_universe) if hist_universe is not None else 0,
    )

    universe_data = st.session_state.v10_scan_universe_data
    data_quality_logs = st.session_state.v10_scan_quality_logs
    excluded_logs = st.session_state.v10_scan_excluded_logs
    stock_list = st.session_state.v10_scan_stock_list
    ref_date = st.session_state.v10_scan_ref_date
    using_historical_universe = st.session_state.v10_scan_using_historical
    validity_map = build_historical_validity_map(hist_universe) if using_historical_universe else None
    meta = stock_list.drop_duplicates("Symbol").set_index("Symbol") if not stock_list.empty else pd.DataFrame()

    # FIX: 후보 결과도 session_state에 저장하여 다른 탭으로 이동해도 유지.
    cand_df = st.session_state.get("v10_scan_candidates", pd.DataFrame())

    with tab1:
        st.subheader("🎯 급등 전조 선행 신호 스캔")
        st.info("📌 자동 스캔은 실행하지 않습니다. 아래 버튼을 눌렀을 때만 KRX 목록·시장지수·가격 데이터를 수집합니다.")

        scan_col1, scan_col2 = st.columns([1, 3])
        with scan_col1:
            scan_clicked = st.button("🔍 스캔 시작", type="primary", use_container_width=True, key="v10_manual_scan")
        with scan_col2:
            if st.session_state.v10_scan_signature is not None and st.session_state.v10_scan_signature != current_signature:
                st.warning("⚠️ 스캔 설정이 이전 결과와 달라졌습니다. 새 설정을 적용하려면 ‘스캔 시작’을 눌러주세요.")

        if scan_clicked:
            # FIX: 버튼 클릭 시에만 유니버스를 결정하고 실제 데이터 수집을 시작.
            with st.status("스캔 준비 중...", expanded=True) as scan_status:
                progress = st.progress(0, text="0% — 유니버스 준비 중")
                status_text = st.empty()

                try:
                    if hist_universe is not None:
                        # FIX: 역사적 CSV를 현재 KRX와 합산하지 않고 단독 분석 유니버스로 사용.
                        scan_stock_list = filter_universe_by_market_and_date(hist_universe, market_choice)
                        scan_using_historical = True
                    else:
                        scan_stock_list = get_krx_stock_list()
                        scan_stock_list = filter_universe_by_market_and_date(scan_stock_list, market_choice)
                        scan_stock_list = scan_stock_list.copy()
                        scan_stock_list["StartDate"] = pd.Timestamp.min
                        scan_stock_list["EndDate"] = pd.Timestamp.max
                        scan_using_historical = False

                    if sample_scan_count > 0:
                        scan_stock_list = scan_stock_list.head(int(sample_scan_count)).copy()

                    if scan_stock_list is None or scan_stock_list.empty:
                        raise RuntimeError("분석 대상 유니버스가 없습니다.")

                    status_text.write(
                        f"유니버스 준비 완료: {len(scan_stock_list):,}개 종목 | "
                        f"기준 기간: {start_dt} ~ {end_dt}"
                    )
                    progress.progress(0.02, text=f"2% — 유니버스 {len(scan_stock_list):,}개 준비 완료")

                    # 시장지수도 스캔 버튼을 눌렀을 때만 조회.
                    status_text.write("시장지수(KOSPI/KOSDAQ) 수집 중...")
                    kospi_idx = fetch_market_index("KOSPI", start_dt, end_dt)
                    kosdaq_idx = fetch_market_index("KOSDAQ", start_dt, end_dt)
                    if kospi_idx is None or kosdaq_idx is None:
                        st.warning("⚠️ 시장지수 데이터 부족: 상대강도 지표 일부가 결측 처리될 수 있습니다.")

                    scan_universe_data = {}
                    scan_quality_logs = []
                    scan_excluded_logs = []
                    total = len(scan_stock_list)
                    done = 0

                    # FIX: 진행률이 실제 완료 종목 수를 반영하도록 개별 종목 완료마다 갱신.
                    # FIX: 기존 concurrent.futures를 활용하되 결과 저장은 메인 스레드에서 수행.
                    def fetch_one(item):
                        _, r = item
                        sym, name, mkt = r["Symbol"], r["Name"], r["Market"]
                        m_idx = kospi_idx if mkt == "KOSPI" else kosdaq_idx
                        raw_df, msg, amount_source = fetch_ohlcv_data(
                            sym,
                            start_dt.strftime("%Y-%m-%d"),
                            end_dt.strftime("%Y-%m-%d"),
                        )
                        return sym, name, mkt, raw_df, msg, amount_source

                    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                        futures = [executor.submit(fetch_one, item) for item in scan_stock_list.iterrows()]
                        for future in concurrent.futures.as_completed(futures):
                            sym, name, mkt, raw_df, msg, amount_source = future.result()
                            done += 1
                            pct = 0.02 + 0.88 * (done / max(total, 1))
                            progress.progress(
                                min(pct, 0.90),
                                text=f"{done:,}/{total:,} 종목 처리 완료 ({done / max(total, 1) * 100:.1f}%)",
                            )
                            status_text.write(f"최근 처리: {sym} {name} | {msg}")

                            is_valid, audit_reasons = audit_stock_data_quality(raw_df, name)
                            if not is_valid:
                                scan_quality_logs.append(
                                    {
                                        "Symbol": sym,
                                        "Name": name,
                                        "Status": "오류/부족",
                                        "Reasons": ", ".join(audit_reasons),
                                        "거래대금 소스": amount_source,
                                    }
                                )
                                continue

                            m_idx = kospi_idx if mkt == "KOSPI" else kosdaq_idx
                            feat_df = calculate_technical_features(raw_df, m_idx)
                            if feat_df is None:
                                scan_quality_logs.append(
                                    {
                                        "Symbol": sym,
                                        "Name": name,
                                        "Status": "오류/부족",
                                        "Reasons": "기술지표 계산 불가",
                                        "거래대금 소스": amount_source,
                                    }
                                )
                                continue

                            # FIX: 역사적 CSV의 유효기간을 실제 분석 데이터에 반영.
                            if scan_using_historical:
                                intervals = build_historical_validity_map(
                                    scan_stock_list[scan_stock_list["Symbol"] == sym]
                                )
                                valid_mask = pd.Series(False, index=feat_df.index)
                                for start_v, end_v in intervals.get(sym, []):
                                    valid_mask |= (feat_df.index >= start_v) & (feat_df.index <= end_v)
                                feat_df = feat_df.loc[valid_mask].copy()

                            if feat_df.empty:
                                scan_quality_logs.append(
                                    {
                                        "Symbol": sym,
                                        "Name": name,
                                        "Status": "오류/부족",
                                        "Reasons": "역사적 유효기간 내 가격 데이터 없음",
                                        "거래대금 소스": amount_source,
                                    }
                                )
                                continue

                            scan_universe_data[sym] = feat_df
                            scan_quality_logs.append(
                                {
                                    "Symbol": sym,
                                    "Name": name,
                                    "Status": "정상",
                                    "Reasons": "합격",
                                    "거래대금 소스": amount_source,
                                }
                            )

                    if not scan_universe_data:
                        progress.progress(1.0, text="100% — 분석 가능한 종목 없음")
                        scan_status.update(label="스캔 완료 — 분석 가능한 종목 없음", state="error")
                        st.error("분석 가능한 종목 데이터가 없습니다. 데이터 품질 탭에서 원인을 확인하세요.")
                    else:
                        # FIX: 실제 사용 데이터의 최신 거래일을 기준일로 확정.
                        scan_ref_date = max(df.index[-1].date() for df in scan_universe_data.values())
                        scan_validity_map = build_historical_validity_map(hist_universe) if scan_using_historical else None
                        scan_meta = scan_stock_list.drop_duplicates("Symbol").set_index("Symbol")
                        scan_candidates = []
                        scan_excluded_logs = []

                        for sym, df in scan_universe_data.items():
                            row = df.iloc[-1]
                            last_dt = df.index[-1].date()
                            if sym not in scan_meta.index:
                                continue
                            name = scan_meta.loc[sym, "Name"]
                            mkt = scan_meta.loc[sym, "Market"]

                            passed, gate_reasons = evaluate_essential_gates(row, settings, last_dt, scan_ref_date)
                            tot_score, s_A, s_B, s_C, s_D = calculate_precursor_subscores(
                                row, missing_handling
                            )

                            if not passed:
                                scan_excluded_logs.append(
                                    {"Symbol": sym, "Name": name, "Reasons": ", ".join(gate_reasons)}
                                )
                                continue

                            if pd.isna(tot_score):
                                reason = (
                                    "CMF 포함 필수 점수 계산 불가"
                                    if missing_handling == "분석불가 표시"
                                    else "필수 지표 결측으로 점수 계산 불가"
                                )
                                scan_excluded_logs.append({"Symbol": sym, "Name": name, "Reasons": reason})
                                continue

                            if tot_score < min_score:
                                scan_excluded_logs.append(
                                    {"Symbol": sym, "Name": name, "Reasons": f"점수 미달 ({tot_score:.1f}점)"}
                                )
                                continue

                            next_dt = _next_symbol_date(df, df.index[-1])
                            next_text = str(next_dt.date()) if next_dt is not None else "다음 거래일 데이터 없음"
                            cmf_text = f"{row['CMF20']:.3f}" if pd.notna(row.get("CMF20")) else "CMF 계산 불가"

                            scan_candidates.append(
                                {
                                    "종목코드": sym,
                                    "종목명": name,
                                    "시장": mkt,
                                    "분석기준일": scan_ref_date,
                                    "실제 다음 거래일": next_text,
                                    "전조 점수": round(tot_score, 1),
                                    "압축 점수": round(s_A, 1) if pd.notna(s_A) else np.nan,
                                    "돌파 준비 점수": round(s_B, 1) if pd.notna(s_B) else np.nan,
                                    "자금 유입 점수": round(s_C, 1) if pd.notna(s_C) else np.nan,
                                    "추격 위험 점수": round(s_D, 1) if pd.notna(s_D) else np.nan,
                                    "오늘 수익률(%)": round(row.get("Return1", 0), 2),
                                    "5일 수익률(%)": round(row.get("Return5", 0), 2),
                                    "20일 수익률(%)": round(row.get("Return20", 0), 2),
                                    "20일 박스폭(%)": round(row.get("BoxWidth20", 0), 2),
                                    "거래량/20일평균": round(row.get("VolumeRatio20", 0), 2),
                                    "거래대금/20일평균": round(row.get("AmountRatio20", 0), 2),
                                    "CMF20": cmf_text,
                                    "RSI": round(row.get("RSI14", 0), 1),
                                    "전조 유형": classify_surge_pattern(row),
                                    "추천 근거": "가격·거래량 조건이 상대적으로 양호한 연구용 후보",
                                    "주의사항": "상승을 보장하지 않으며 다음 거래일 시가 체결 관찰 필요",
                                }
                            )

                        scan_cand_df = pd.DataFrame(scan_candidates)
                        if not scan_cand_df.empty:
                            scan_cand_df = scan_cand_df.sort_values(by="전조 점수", ascending=False)
                            scan_cand_df.insert(0, "순위", range(1, len(scan_cand_df) + 1))

                        # FIX: 스캔 결과를 세션에 저장. 이후 탭 이동은 재스캔하지 않음.
                        st.session_state.v10_scan_universe_data = scan_universe_data
                        st.session_state.v10_scan_quality_logs = scan_quality_logs
                        st.session_state.v10_scan_excluded_logs = scan_excluded_logs
                        st.session_state.v10_scan_stock_list = scan_stock_list
                        st.session_state.v10_scan_ref_date = scan_ref_date
                        st.session_state.v10_scan_using_historical = scan_using_historical
                        st.session_state.v10_scan_meta = scan_meta.reset_index()
                        st.session_state.v10_scan_candidates = scan_cand_df
                        st.session_state.v10_scan_signature = current_signature

                        progress.progress(1.0, text=f"100% — 스캔 완료 ({len(scan_universe_data):,}개 분석 성공)")
                        scan_status.update(label=f"스캔 완료 — {len(scan_universe_data):,}개 종목 분석", state="complete")

                        universe_data = scan_universe_data
                        data_quality_logs = scan_quality_logs
                        excluded_logs = scan_excluded_logs
                        stock_list = scan_stock_list
                        ref_date = scan_ref_date
                        using_historical_universe = scan_using_historical
                        validity_map = scan_validity_map
                        meta = scan_meta
                        cand_df = scan_cand_df

                except Exception as e:
                    scan_status.update(label="스캔 중 오류 발생", state="error")
                    st.exception(e)

        # 저장된 결과 표시
        cand_df = st.session_state.get("v10_scan_candidates", pd.DataFrame())
        universe_data = st.session_state.get("v10_scan_universe_data", {})
        ref_date = st.session_state.get("v10_scan_ref_date")
        using_historical_universe = st.session_state.get("v10_scan_using_historical", False)

        if ref_date is None or not universe_data:
            st.warning("📌 아직 스캔하지 않았습니다. ‘🔍 스캔 시작’을 눌러야 종목 검색이 시작됩니다.")
        else:
            st.info(
                f"📅 **분석 기준일**: {ref_date} | **예측 대상일**: 각 종목 가격 데이터의 실제 다음 거래일 시가 | "
                f"**실제 사용 유니버스**: {'역사적 유니버스 CSV' if using_historical_universe else '현재 KRX'} | "
                f"**거래대금 단위**: 원(KRW)로 취급"
            )
            st.caption("※ 거래대금 단위는 원(KRW)로 취급하며 공급원 단위를 코드에서 독립적으로 검증하지는 않습니다.")
            if not cand_df.empty:
                st.dataframe(cand_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "📥 후보 결과 CSV 다운로드",
                    cand_df.to_csv(index=False).encode("utf-8-sig"),
                    f"candidates_{ref_date}.csv",
                    "text/csv",
                )
            else:
                st.warning("조건을 충족하는 급등 전조 후보 종목이 없습니다.")
            st.caption("💡 OBV·CMF는 가격과 거래량으로 계산한 보조지표이며 실제 투자자별 수급 데이터가 아닙니다.")

    # 공통으로 저장된 스캔 결과 사용
    universe_data = st.session_state.get("v10_scan_universe_data", {})
    data_quality_logs = st.session_state.get("v10_scan_quality_logs", [])
    excluded_logs = st.session_state.get("v10_scan_excluded_logs", [])
    stock_list = st.session_state.get("v10_scan_stock_list", pd.DataFrame())
    ref_date = st.session_state.get("v10_scan_ref_date")
    using_historical_universe = st.session_state.get("v10_scan_using_historical", False)
    cand_df = st.session_state.get("v10_scan_candidates", pd.DataFrame())
    validity_map = build_historical_validity_map(hist_universe) if using_historical_universe else None
    meta = stock_list.drop_duplicates("Symbol").set_index("Symbol") if not stock_list.empty else pd.DataFrame()

    # ----------------------------------------------------
    # Tab 2
    # ----------------------------------------------------
    with tab2:
        st.subheader("🔍 후보 종목 상세 분석 및 기술적 차트")
        if not universe_data or cand_df.empty:
            st.info("먼저 ‘🔍 스캔 시작’을 실행하고 후보가 생성되면 상세 분석이 가능합니다.")
        else:
            selected_sym = st.selectbox("분석할 후보 종목 선택", cand_df["종목코드"] + " | " + cand_df["종목명"])
            sym_code = selected_sym.split(" | ")[0]
            target_df = universe_data[sym_code]
            last_r = target_df.iloc[-1]

            st.write(f"### {selected_sym} 지표 요약")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("종가", f"{int(last_r['Close']):,} 원")
            col_b.metric("20일 평균 거래대금", f"{int(last_r['AmountMA20']/100_000_000):,} 억원")
            col_c.metric("RSI (14)", f"{last_r['RSI14']:.1f}")
            col_d.metric("ATR (14)", f"{int(last_r['ATR14']):,} 원")

            cmf_display = f"{last_r['CMF20']:.3f}" if pd.notna(last_r.get("CMF20")) else "CMF 계산 불가"
            st.metric("CMF (20)", cmf_display)

            entry_ref = last_r["Close"]
            atr_val = last_r["ATR14"]
            stop_ref = entry_ref - 2.0 * atr_val
            tp1_ref = entry_ref + 2.0 * (entry_ref - stop_ref)
            tp2_ref = entry_ref + 3.5 * (entry_ref - stop_ref)
            st.info(
                f"🔬 **연구용 참고 가격 수준 (다음날 시가 기준 참고용)**:\n"
                f"- 신호일 종가: {int(entry_ref):,}원 | ATR 기준 손절 참고선: {int(stop_ref):,}원 (-{((entry_ref-entry_ref+2.0*atr_val)/entry_ref*100):.1f}%)\n"
                f"- 1차 목표 참고선: {int(tp1_ref):,}원 | 2차 목표 참고선: {int(tp2_ref):,}원"
            )

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25])
            fig.add_trace(go.Candlestick(x=target_df.index, open=target_df["Open"], high=target_df["High"], low=target_df["Low"], close=target_df["Close"], name="OHLC"), row=1, col=1)
            fig.add_trace(go.Scatter(x=target_df.index, y=target_df["EMA20"], name="EMA20", line=dict(color="orange")), row=1, col=1)
            fig.add_trace(go.Scatter(x=target_df.index, y=target_df["High20"], name="20일 박스상단", line=dict(color="red", dash="dash")), row=1, col=1)
            fig.add_trace(go.Bar(x=target_df.index, y=target_df["Volume"], name="거래량", marker_color="blue"), row=2, col=1)
            fig.add_trace(go.Scatter(x=target_df.index, y=target_df["RSI14"], name="RSI", line=dict(color="purple")), row=3, col=1)
            fig.update_layout(height=650, title_text=f"{selected_sym} 기술적 분석 차트", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # Tab 3
    # ----------------------------------------------------
    with tab3:
        st.subheader("⚖️ 단계별 전략 조건 비교 (A ~ H)")
        if not universe_data:
            st.info("먼저 ‘🔍 스캔 시작’을 실행하세요. 데이터 수집 후 실제 다음 거래일 수익률을 계산합니다.")
        else:
            st.write("동일 기간·동일 유니버스에서 각 조건을 실제로 적용하여 다음 거래일 시가 수익률을 계산합니다.")
            strategy_rows = []
            for label, mask_fn in [
                ("A. 전체 대상 종목", lambda r: True),
                ("B. 박스권 조건", lambda r: pd.notna(r.get("BoxWidth20")) and r["BoxWidth20"] <= 10),
                ("C. 박스권 + 변동성 압축", lambda r: pd.notna(r.get("BoxWidthRatio5_20")) and r["BoxWidthRatio5_20"] < 0.6),
                ("D. C + 거래대금 증가", lambda r: pd.notna(r.get("BoxWidthRatio5_20")) and r["BoxWidthRatio5_20"] < 0.6 and pd.notna(r.get("AmountRatio20")) and r["AmountRatio20"] >= 1.5),
                ("E. D + 상대강도", lambda r: pd.notna(r.get("BoxWidthRatio5_20")) and r["BoxWidthRatio5_20"] < 0.6 and pd.notna(r.get("AmountRatio20")) and r["AmountRatio20"] >= 1.5 and pd.notna(r.get("RelReturn20")) and r["RelReturn20"] > 0),
                ("F. E + 오늘 급등 제외", lambda r: pd.notna(r.get("Return1")) and r["Return1"] <= settings["max_today_return"] and pd.notna(r.get("BoxWidthRatio5_20")) and r["BoxWidthRatio5_20"] < 0.6 and pd.notna(r.get("AmountRatio20")) and r["AmountRatio20"] >= 1.5 and pd.notna(r.get("RelReturn20")) and r["RelReturn20"] > 0),
                ("G. F + OBV·CMF", lambda r: pd.notna(r.get("OBV_Slope5")) and r["OBV_Slope5"] > 0 and pd.notna(r.get("CMF20")) and r["CMF20"] > 0.05 and pd.notna(r.get("Return1")) and r["Return1"] <= settings["max_today_return"] and pd.notna(r.get("BoxWidthRatio5_20")) and r["BoxWidthRatio5_20"] < 0.6 and pd.notna(r.get("AmountRatio20")) and r["AmountRatio20"] >= 1.5 and pd.notna(r.get("RelReturn20")) and r["RelReturn20"] > 0),
                ("H. 최종 가격 전조 전략", lambda r: pd.notna(calculate_precursor_subscores(r, missing_handling)[0]) and calculate_precursor_subscores(r, missing_handling)[0] >= min_score),
            ]:
                rets = []
                for sym, df in universe_data.items():
                    for dt, r in df.iterrows():
                        if not is_symbol_valid_on_date(sym, dt, validity_map):
                            continue
                        next_dt = _next_symbol_date(df, dt)
                        if next_dt is None or not is_symbol_valid_on_date(sym, next_dt, validity_map):
                            continue
                        try:
                            if mask_fn(r):
                                rets.append((df.loc[next_dt, "Open"] / r["Close"] - 1) * 100)
                        except Exception:
                            continue
                s = pd.Series(rets, dtype=float)
                strategy_rows.append(
                    {
                        "전략": label,
                        "신호 수": len(s),
                        "1일후 평균(%)": s.mean() if not s.empty else np.nan,
                        "1일후 승률(%)": (s > 0).mean() * 100 if not s.empty else np.nan,
                    }
                )
            st.dataframe(pd.DataFrame(strategy_rows), use_container_width=True, hide_index=True)
            st.caption("※ 과거 성과를 보장하지 않습니다. 모든 수치는 현재 선택한 데이터/유니버스에서 실제 계산된 값입니다.")

    # ----------------------------------------------------
    # Tab 4
    # ----------------------------------------------------
    with tab4:
        st.subheader("🔄 V9 백테스트 엔진 시뮬레이션")
        if not universe_data:
            st.info("먼저 ‘🔍 스캔 시작’을 실행하세요.")
        else:
            st.markdown("**청산 우선순위**: 1. 손절 ➡️ 2. 2차 목표가 ➡️ 3. 1차 부분익절 ➡️ 4. EMA20 추세 이탈 ➡️ 5. 최대 보유기간")
            init_cap = st.number_input("초기 자본금 (원)", value=10_000_000, step=1_000_000)
            max_pos = st.slider("최대 동시보유 종목 수", 1, 10, 5)

            if st.button("🚀 V9 백테스트 실행"):
                with st.spinner("백테스트 시뮬레이션 진행 중..."):
                    eq_df, closed_df, logs_df, open_pos = run_v9_backtest_engine(
                        universe_data,
                        settings,
                        initial_capital=init_cap,
                        max_concurrent=max_pos,
                        validity_map=validity_map,
                    )

                if eq_df is not None and not eq_df.empty:
                    final_asset = eq_df.iloc[-1]["TotalAsset"]
                    total_ret = (final_asset / init_cap - 1) * 100
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("최종 평가자산", f"{int(final_asset):,} 원")
                    m2.metric("총 수익률", f"{total_ret:.2f} %")
                    m3.metric("완전 청산 거래 수", f"{len(closed_df) if closed_df is not None else 0} 건")
                    win_r = ((closed_df["Net_PnL"] > 0).mean() * 100 if closed_df is not None and not closed_df.empty else np.nan)
                    m4.metric("실현 승률", f"{win_r:.1f} %" if pd.notna(win_r) else "N/A")
                    st.line_chart(eq_df.set_index("Date")["TotalAsset"])

                    if open_pos:
                        st.warning(f"백테스트 종료 시 미청산 포지션 {len(open_pos)}건은 실현 승률/청산 거래 수에 포함하지 않았습니다. 평가자산에는 마지막 가격으로 반영됩니다.")
                    if closed_df is not None and not closed_df.empty:
                        st.write("### 청산 완료 거래 목록 (실현 성과)")
                        st.dataframe(closed_df, use_container_width=True)
                    if logs_df is not None and not logs_df.empty:
                        st.write("### 주문/예약금 로그")
                        st.dataframe(logs_df, use_container_width=True)

    # ----------------------------------------------------
    # Tab 5
    # ----------------------------------------------------
    with tab5:
        st.subheader("⏳ 시간순 워크포워드(Walk-Forward) 검증")
        if not universe_data:
            st.info("먼저 ‘🔍 스캔 시작’을 실행하세요.")
        else:
            st.write("과거 Train 구간에서 선택한 임계값을 이후 Test 구간에 적용합니다.")
            if st.button("🧪 워크포워드 검증 실행"):
                with st.spinner("워크포워드 검증 계산 중..."):
                    wf_df = run_walk_forward_validation(
                        universe_data, settings, train_years=1, test_years=1, validity_map=validity_map
                    )
                if not wf_df.empty:
                    st.dataframe(wf_df, use_container_width=True)
                else:
                    st.warning("검증 가능한 연도별 데이터 구간이 부족합니다. 3년 수집을 기본으로 사용합니다.")

    # ----------------------------------------------------
    # Tab 6
    # ----------------------------------------------------
    with tab6:
        st.subheader("🛡️ 전략 및 데이터 무결성 감사 (Integrity Audit)")
        audit_res = run_integrity_audit(csv_applied=using_historical_universe)
        st.dataframe(audit_res, use_container_width=True, hide_index=True)

        st.write("### 📋 종목별 데이터 수집 및 품질 통계")
        col_q1, col_q2 = st.columns(2)
        dq_df = pd.DataFrame(data_quality_logs)
        col_q1.metric("전체 사용 유니버스 종목 수", len(stock_list))
        col_q1.metric("분석 성공 종목 수", len(dq_df[dq_df["Status"] == "정상"]) if not dq_df.empty else 0)
        col_q2.metric("데이터 오류/부족 종목 수", len(dq_df[dq_df["Status"] != "정상"]) if not dq_df.empty else 0)
        col_q2.metric("최종 후보 선정 종목 수", len(cand_df) if not cand_df.empty else 0)

        if not dq_df.empty:
            st.dataframe(dq_df, use_container_width=True)

        if not universe_data:
            st.info("아직 스캔하지 않았습니다. 데이터 품질 로그는 ‘스캔 시작’ 이후 생성됩니다.")
        else:
            st.info(
                f"현재 실제 계산 유니버스: {'역사적 유니버스 CSV' if using_historical_universe else '현재 KRX'} | "
                f"거래대금: 원(KRW) 취급 | CMF 결측값 0 대체 안 함"
            )


if __name__ == "__main__":
    main()
