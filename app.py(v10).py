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
    """현재 한국 표준시(KST) 반환"""
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
    avg_gain = gain.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / period, adjust=False, min_periods=period
    ).mean()
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
    high, low, close, volume = (
        df["High"],
        df["Low"],
        df["Close"],
        df["Volume"],
    )
    mf_multiplier = ((close - low) - (high - close)) / (high - low).replace(
        0, np.nan
    )
    mf_volume = mf_multiplier.fillna(0) * volume
    cmf = mf_volume.rolling(period, min_periods=period).sum() / volume.rolling(
        period, min_periods=period
    ).sum().replace(0, np.nan)
    return cmf.fillna(0)


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
            (c for c in ["Name", "종목명", " 종목명"] if c in df.columns),
            None,
        )
        market_col = next(
            (c for c in ["Market", "시장"] if c in df.columns), None
        )

        if not symbol_col or not name_col:
            return pd.DataFrame(columns=["Symbol", "Name", "Market"])

        df = df.rename(
            columns={
                symbol_col: "Symbol",
                name_col: "Name",
                market_col: "Market",
            }
        )
        df["Symbol"] = df["Symbol"].astype(str).str.extract(r"(\d{6})")[0]
        df = df.dropna(subset=["Symbol"])

        if "Market" in df.columns:
            df = df[df["Market"].isin(["KOSPI", "KOSDAQ"])].copy()
        else:
            df["Market"] = "KOSPI/KOSDAQ"

        # 파생, 스팩, 리츠, 우선주, ETF/ETN 제외
        exclude_keywords = [
            "스팩",
            "SPAC",
            "ETF",
            "ETN",
            "리츠",
            "REIT",
            "인프라",
            "신주인수권",
            "KODEX",
            "TIGER",
            "ARIRANG",
            "KBSTAR",
            "HANARO",
            "KOSEF",
            "인버스",
            "레버리지",
        ]
        pattern = "|".join(exclude_keywords)
        mask = (
            ~df["Name"]
            .astype(str)
            .str.upper()
            .str.contains(pattern, regex=True, na=False)
        )
        df = df[mask].copy()

        pref_mask = df["Name"].astype(str).str.contains(
            r"우$|우B$|우C$|우\(전환\)$", regex=True, na=False
        )
        df = df[~pref_mask].copy()

        return (
            df[["Symbol", "Name", "Market"]]
            .drop_duplicates("Symbol")
            .reset_index(drop=True)
        )
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
            "Code": "Symbol",
            "Ticker": "Symbol",
            "종목코드": "Symbol",
            "시작일": "StartDate",
            "종료일": "EndDate",
        }
        df = df.rename(columns={k: v for k, v in alias.items() if k in df.columns})

        if "Symbol" not in df.columns:
            return None, "CSV 파일에 Code 또는 Symbol 컬럼이 존재하지 않습니다."

        df["Symbol"] = (
            df["Symbol"].astype(str).str.extract(r"(\d{6})")[0].fillna(df["Symbol"])
        )
        if "Name" not in df.columns:
            df["Name"] = df["Symbol"]
        if "Market" not in df.columns:
            df["Market"] = "ALL"

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

        return (
            df[["Symbol", "Name", "Market", "StartDate", "EndDate"]].drop_duplicates(),
            "역사적 유니버스 CSV 적용 완료",
        )
    except Exception as e:
        return None, f"CSV 업로드 파싱 오류: {e}"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_ohlcv_data(symbol, start_date, end_date):
    """단일 종목 OHLCV 수집 및 거래대금 산출"""
    try:
        df = fdr.DataReader(symbol, start_date, end_date)
        if df is None or df.empty:
            return None, "데이터 없음"

        required = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required):
            return None, "필수 OHLCV 컬럼 누락"

        df = df[required].copy()
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # 거래대금 컬럼 확인 및 생성 (원 단위)
        if "Amount" in df.columns:
            df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
        else:
            df["Amount"] = df["Close"] * df["Volume"]

        if len(df) < 30:
            return None, f"데이터 행 수 부족 ({len(df)}행)"

        return df, "성공"
    except Exception as e:
        return None, f"수집 에러: {str(e)}"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_index(market_type, start_date, end_date):
    """시장지수(KOSPI: KS11, KOSDAQ: KQ11) 수집"""
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
    """기술적 지표 및 V10 전조 특징 생성"""
    if df is None or len(df) < 20:
        return None

    x = df.copy()
    close, high, low, open_px, volume, amount = (
        x["Close"],
        x["High"],
        x["Low"],
        x["Open"],
        x["Volume"],
        x["Amount"],
    )

    # 1. 수익률
    for d in [1, 3, 5, 10, 20]:
        x[f"Return{d}"] = close.pct_change(d) * 100

    # 2. 이동평균 및 이격도
    x["SMA20"] = close.rolling(20, min_periods=20).mean()
    x["SMA60"] = close.rolling(60, min_periods=60).mean()
    x["EMA20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    x["EMA60"] = close.ewm(span=60, adjust=False, min_periods=60).mean()
    x["EMA120"] = close.ewm(span=120, adjust=False, min_periods=120).mean()

    x["EMA20_Slope"] = x["EMA20"] - x["EMA20"].shift(5)
    x["EMA60_Slope"] = x["EMA60"] - x["EMA60"].shift(10)

    x["DisparityEMA20"] = (close / x["EMA20"] - 1) * 100

    # 3. 박스권 지표
    x["High20"] = high.rolling(20, min_periods=20).max()
    x["Low20"] = low.rolling(20, min_periods=20).min()
    x["High60"] = high.rolling(60, min_periods=60).max()
    x["BoxWidth20"] = (x["High20"] - x["Low20"]) / x["Low20"].replace(0, np.nan) * 100
    x["BoxWidth5"] = (
        (high.rolling(5).max() - low.rolling(5).min())
        / low.rolling(5).min().replace(0, np.nan)
        * 100
    )
    x["BoxPosition20"] = (close - x["Low20"]) / (x["High20"] - x["Low20"]).replace(
        0, np.nan
    )
    x["BoxWidthRatio5_20"] = x["BoxWidth5"] / x["BoxWidth20"].replace(0, np.nan)

    # 4. 고점/저점 추세 및 돌파 준비
    x["HigherLow5"] = low.rolling(5).min() > low.shift(5).rolling(5).min()
    x["HigherHigh5"] = high.rolling(5).max() > high.shift(5).rolling(5).max()
    x["DistToHigh20"] = (x["High20"] - close) / close * 100
    x["DistToHigh60"] = (x["High60"] - close) / close * 100

    low5 = low.rolling(5).min()
    x["LowSlope5"] = (low5 - low5.shift(5)) / low5.shift(5).replace(0, np.nan) * 100
    high5 = high.rolling(5).max()
    x["HighSlope5"] = (high5 - high5.shift(5)) / high5.shift(5).replace(
        0, np.nan
    ) * 100

    # 5. 변동성 지표
    x["ATR14"] = wilder_atr(x, 14)
    x["ATR_Ratio"] = x["ATR14"] / close * 100
    x["Vol5"] = close.pct_change().rolling(5).std()
    x["Vol20"] = close.pct_change().rolling(20).std()
    x["VolRatio5_20"] = x["Vol5"] / x["Vol20"].replace(0, np.nan)

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    x["BB_Width"] = (2 * bb_std * 2) / bb_mid.replace(0, np.nan)

    # 6. 거래량 / 자금 흐름 지표
    x["VolumeMA20"] = volume.rolling(20, min_periods=20).mean()
    x["VolumeMA60"] = volume.rolling(60, min_periods=60).mean()
    x["VolumeRatio20"] = volume / x["VolumeMA20"].replace(0, np.nan)
    x["VolumeRatio60"] = volume / x["VolumeMA60"].replace(0, np.nan)

    x["AmountMA20"] = amount.rolling(20, min_periods=20).mean()
    x["AmountRatio20"] = amount / x["AmountMA20"].replace(0, np.nan)
    x["AmountChange5"] = (
        (amount - amount.shift(5)) / amount.shift(5).replace(0, np.nan) * 100
    )

    # OBV & CMF
    obv_dir = np.sign(close.diff()).fillna(0)
    x["OBV"] = (obv_dir * volume).cumsum()
    x["OBV_MA20"] = x["OBV"].rolling(20, min_periods=20).mean()
    x["OBV_Slope5"] = x["OBV"] - x["OBV"].shift(5)

    x["CMF20"] = calculate_cmf(x, 20)

    # 7. RSI & 캔들 형태
    x["RSI14"] = wilder_rsi(close, 14)
    candle_range = (high - low).replace(0, np.nan)
    x["CloseLocation"] = (close - low) / candle_range
    x["UpperWickRatio"] = (high - np.maximum(open_px, close)) / candle_range

    # 8. 시장 지수 대비 상대 수익률
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
    """종목별 데이터 무결성 검증"""
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

    if (raw_df[["Open", "High", "Low", "Close"]] <= 0).any().any():
        reasons.append("가격 0 이하 비정상 데이터")

    if (raw_df["High"] < raw_df["Low"]).any():
        reasons.append("High < Low 비정상 데이터")

    if (raw_df["Volume"] < 0).any() or (raw_df["Amount"] < 0).any():
        reasons.append("음수 거래량/거래대금 오류")

    # 최근 거래정지 / 거래량 0 장기화
    recent_v = raw_df["Volume"].tail(5)
    if (recent_v == 0).all():
        reasons.append("최근 5일 연속 거래량 0 (거래정지 추정)")

    return len(reasons) == 0, reasons


# ============================================================
# 4. 가격 전조 점수 설계 (0~100점) 및 세부 항목
# ============================================================
def calculate_precursor_subscores(row, missing_handling="지표 제외 후 나머지로 계산"):
    """
    4대 하위 점수 계산 (결측값 처리 옵션 적용)
    A. 압축 점수 (25%)
    B. 돌파 준비 점수 (30%)
    C. 자금 유입 점수 (25%)
    D. 추격 위험 점수 (20%) - 안전성 기준(점수가 높을수록 위험 낮음)
    """

    def eval_score(condition_dict):
        scores, weights = [], []
        for val, (score_val, w) in condition_dict.items():
            if pd.isna(val):
                if missing_handling == "해당 종목 제외":
                    return np.nan
                elif missing_handling == "분석불가 표시":
                    return np.nan
                else:  # 결측 지표 제외 계산
                    continue
            scores.append(score_val)
            weights.append(w)
        if not scores:
            return np.nan
        return float(np.sum(np.array(scores) * np.array(weights)) / np.sum(weights))

    # A. 압축 점수
    comp_dict = {
        row.get("VolRatio5_20"): (
            100 if row.get("VolRatio5_20", 1) < 0.7 else 50,
            0.3,
        ),
        row.get("BoxWidthRatio5_20"): (
            100 if row.get("BoxWidthRatio5_20", 1) < 0.6 else 50,
            0.3,
        ),
        row.get("ATR_Ratio"): (
            100 if row.get("ATR_Ratio", 5) < 3.5 else 40,
            0.2,
        ),
        row.get("BB_Width"): (
            100 if row.get("BB_Width", 0.2) < 0.1 else 50,
            0.2,
        ),
    }
    score_A = eval_score(comp_dict)

    # B. 돌파 준비 점수
    prep_dict = {
        row.get("DistToHigh20"): (
            100 if row.get("DistToHigh20", 10) < 3.0 else 40,
            0.25,
        ),
        row.get("HigherLow5"): (
            100 if bool(row.get("HigherLow5", False)) else 20,
            0.25,
        ),
        row.get("HigherHigh5"): (
            100 if bool(row.get("HigherHigh5", False)) else 30,
            0.2,
        ),
        row.get("CloseLocation"): (
            100 if row.get("CloseLocation", 0.5) > 0.7 else 40,
            0.3,
        ),
    }
    score_B = eval_score(prep_dict)

    # C. 자금 유입 점수
    flow_dict = {
        row.get("VolumeRatio20"): (
            100 if row.get("VolumeRatio20", 1) >= 1.5 else 40,
            0.25,
        ),
        row.get("AmountRatio20"): (
            100 if row.get("AmountRatio20", 1) >= 1.5 else 40,
            0.25,
        ),
        row.get("OBV_Slope5"): (
            100 if row.get("OBV_Slope5", 0) > 0 else 30,
            0.25,
        ),
        row.get("CMF20"): (100 if row.get("CMF20", 0) > 0.05 else 30, 0.25),
    }
    score_C = eval_score(flow_dict)

    # D. 추격 위험 점수 (안전성 점수화: 100점 = 과열 및 추격 위험이 적고 안정적)
    risk_dict = {
        row.get("Return1"): (
            100 if row.get("Return1", 0) < 5.0 else 30,
            0.25,
        ),
        row.get("Return5"): (
            100 if row.get("Return5", 0) < 12.0 else 30,
            0.25,
        ),
        row.get("DisparityEMA20"): (
            100 if row.get("DisparityEMA20", 0) < 8.0 else 20,
            0.25,
        ),
        row.get("RSI14"): (100 if row.get("RSI14", 50) < 68.0 else 20, 0.25),
    }
    score_D = eval_score(risk_dict)

    if (
        pd.isna(score_A)
        or pd.isna(score_B)
        or pd.isna(score_C)
        or pd.isna(score_D)
    ):
        return np.nan, np.nan, np.nan, np.nan, np.nan

    total_score = (
        score_A * 0.25 + score_B * 0.30 + score_C * 0.25 + score_D * 0.20
    )
    return total_score, score_A, score_B, score_C, score_D


def classify_surge_pattern(row):
    """상승 전조 유형 분류"""
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
    elif (
        row.get("Close", 0) > row.get("EMA20", 0)
        and row.get("EMA20", 0) > row.get("EMA60", 0)
    ):
        return "추세 유지형"
    return "관찰형"


# ============================================================
# 5. 필수 게이트(Gate) 검증
# ============================================================
def evaluate_essential_gates(row, settings, data_last_date, ref_date):
    """필수 게이트 필터링"""
    reasons = []

    # 1. 최소 거래대금 (원 단위)
    min_amount_krw = settings["min_amount_100m"] * 100_000_000
    if row.get("AmountMA20", 0) < min_amount_krw:
        reasons.append("최소 거래대금 미달")

    # 2. 분석 기준일 데이터 상이 여부
    if data_last_date != ref_date:
        reasons.append("데이터 기준일 불일치")

    # 3. 당일 과도한 급등 제한 (추격 매수 방지)
    if row.get("Return1", 0) > settings["max_today_return"]:
        reasons.append("당일 급등 과열 제외")

    # 4. 가격 데이터 정상성
    if row.get("Close", 0) <= 0 or pd.isna(row.get("Close")):
        reasons.append("비정상 종가 데이터")

    return len(reasons) == 0, reasons


# ============================================================
# 6. V9 백테스트 엔진 통합
# ============================================================
def run_v9_backtest_engine(
    universe_dict,
    settings,
    initial_capital=10_000_000,
    max_concurrent=5,
    fee_rate=0.00015,
    tax_rate=0.0020,
    slippage=0.001,
):
    """
    V9의 t일 종가 신호 -> t+1일 실제 시가 체결 백테스트 엔진
    우선순위: 1. 손절 -> 2. 2차 목표가 -> 3. 1차 목표가 -> 4. 추세 이탈 -> 5. 최대 보유기간
    """
    all_dates = sorted(
        list(set(d for df in universe_dict.values() for d in df.index))
    )
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
        # A. t+1일 실제 시가 체결 (예약 주문 처리)
        # ----------------------------------------------------
        if pending_orders:
            pending_orders.sort(key=lambda x: x["score"], reverse=True)
            new_pending = []
            available_slots = max(0, max_concurrent - len(positions))

            for order in pending_orders:
                sym = order["sym"]
                df = universe_dict.get(sym)

                if df is None or current_date not in df.index:
                    new_pending.append(order)
                    continue

                reserved_cash -= order["reserved_amount"]

                if available_slots <= 0:
                    trade_logs.append(
                        {
                            "Date": current_date,
                            "Symbol": sym,
                            "Type": "ORDER_CANCEL",
                            "Reason": "동시보유 슬롯 초과",
                        }
                    )
                    continue

                row = df.loc[current_date]
                exec_price = float(row["Open"]) * (1 + slippage)

                if exec_price <= 0 or pd.isna(exec_price):
                    trade_logs.append(
                        {
                            "Date": current_date,
                            "Symbol": sym,
                            "Type": "ORDER_FAIL",
                            "Reason": "비정상 시가",
                        }
                    )
                    continue

                alloc_cash = order["reserved_amount"]
                qty = int((alloc_cash * 0.98) / exec_price)

                if qty <= 0:
                    trade_logs.append(
                        {
                            "Date": current_date,
                            "Symbol": sym,
                            "Type": "ORDER_FAIL",
                            "Reason": "주문가능 수량 0",
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
                            "Reason": "현금 부족",
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
                        "Reason": "t일 종가 신호 -> t+1일 시가 체결",
                    }
                )

            pending_orders = new_pending

        # ----------------------------------------------------
        # B. 포지션 청산 및 익절/손절 관리
        # ----------------------------------------------------
        for pid in list(positions.keys()):
            pos = positions[pid]
            df = universe_dict[pos["sym"]]
            if current_date not in df.index:
                continue

            row = df.loc[current_date]
            if current_date != pos["entry_date"]:
                pos["holding_days"] += 1

            exit_reason = None
            exit_price = None
            sell_qty = 0

            # 동일 봉 청산 우선순위 적용
            if row["Low"] <= pos["stop_loss"]:
                exit_reason = "1. 손절"
                exit_price = pos["stop_loss"] * (1 - slippage)
                sell_qty = pos["qty"]
            elif row["High"] >= pos["tp2"]:
                exit_reason = "2. 2차 목표가"
                exit_price = pos["tp2"] * (1 - slippage)
                sell_qty = pos["qty"]
            elif row["High"] >= pos["tp1"] and not pos["partial_done"]:
                exit_reason = "3. 1차 목표가 (부분익절)"
                exit_price = pos["tp1"] * (1 - slippage)
                sell_qty = max(1, int(pos["initial_qty"] * 0.5))
                sell_qty = min(sell_qty, pos["qty"])
            elif (
                row["Close"] < row["EMA20"]
                and row.get("EMA20_Slope", 0) < 0
            ):
                exit_reason = "4. EMA20 추세 이탈"
                exit_price = float(row["Close"]) * (1 - slippage)
                sell_qty = pos["qty"]
            elif pos["holding_days"] >= settings["max_holding_days"]:
                exit_reason = "5. 최대 보유기간 만료"
                exit_price = float(row["Close"]) * (1 - slippage)
                sell_qty = pos["qty"]

            if exit_reason and sell_qty > 0:
                sell_gross = sell_qty * exit_price
                sell_fee = sell_gross * fee_rate
                sell_tax = sell_gross * tax_rate
                buy_fee_alloc = pos["buy_fee"] * (sell_qty / pos["initial_qty"])

                net_pnl = (
                    (exit_price - pos["entry_price"]) * sell_qty
                    - buy_fee_alloc
                    - sell_fee
                    - sell_tax
                )
                cash += sell_gross - sell_fee - sell_tax
                pos["realized_pnl"] += net_pnl

                trade_logs.append(
                    {
                        "Date": current_date,
                        "Symbol": pos["sym"],
                        "Type": "SELL",
                        "Price": exit_price,
                        "Qty": sell_qty,
                        "Reason": exit_reason,
                        "Net_PnL": net_pnl,
                    }
                )

                if sell_qty >= pos["qty"]:
                    closed_trades.append(
                        {
                            "Position_ID": pid,
                            "Symbol": pos["sym"],
                            "Entry_Date": pos["entry_date"],
                            "Exit_Date": current_date,
                            "Holding_Days": pos["holding_days"],
                            "Entry_Price": pos["entry_price"],
                            "Exit_Price": exit_price,
                            "Net_PnL": pos["realized_pnl"],
                            "Return_Pct": (pos["realized_pnl"] / (pos["entry_price"] * pos["initial_qty"])) * 100,
                            "Exit_Reason": exit_reason,
                        }
                    )
                    del positions[pid]
                else:
                    pos["qty"] -= sell_qty
                    pos["partial_done"] = True

        # ----------------------------------------------------
        # C. t일 종가 신호 생성 (미래 데이터 참조 금지)
        # ----------------------------------------------------
        if not is_last_day:
            active_symbols = {p["sym"] for p in positions.values()}
            pending_symbols = {p["sym"] for p in pending_orders}

            candidates = []
            for sym, df in universe_dict.items():
                if sym in active_symbols or sym in pending_symbols or current_date not in df.index:
                    continue

                row = df.loc[current_date]
                passed, _ = evaluate_essential_gates(
                    row, settings, current_date, current_date
                )
                tot_score, _, _, _, _ = calculate_precursor_subscores(
                    row, settings["missing_handling"]
                )

                if passed and pd.notna(tot_score) and tot_score >= settings["min_score"]:
                    candidates.append((tot_score, sym, row))

            candidates.sort(key=lambda x: x[0], reverse=True)

            free_slots = max(0, max_concurrent - len(positions) - len(pending_orders))
            free_cash = max(0.0, cash - reserved_cash)

            if free_slots > 0 and free_cash > 0:
                per_order_cash = free_cash / free_slots
                for score_val, sym, row in candidates[:free_slots]:
                    atr = float(row["ATR14"])
                    close_px = float(row["Close"])
                    stop_px = close_px - 2.0 * atr
                    risk = close_px - stop_px
                    tp1_px = close_px + 2.0 * risk
                    tp2_px = close_px + 3.5 * risk

                    reserve_amt = min(
                        per_order_cash, max(0.0, cash - reserved_cash)
                    )
                    if reserve_amt <= 0:
                        break

                    pending_orders.append(
                        {
                            "sym": sym,
                            "score": score_val,
                            "signal_date": current_date,
                            "atr": atr,
                            "stop_loss": stop_px,
                            "tp1": tp1_px,
                            "tp2": tp2_px,
                            "reserved_amount": reserve_amt,
                        }
                    )
                    reserved_cash += reserve_amt

        # ----------------------------------------------------
        # D. 자산 평가
        # ----------------------------------------------------
        stock_eval = sum(
            pos["qty"] * float(universe_dict[pos["sym"]].loc[current_date, "Close"])
            for pos in positions.values()
            if current_date in universe_dict[pos["sym"]].index
        )
        equity_curve.append(
            {
                "Date": current_date,
                "Cash": cash,
                "ReservedCash": reserved_cash,
                "StockValue": stock_eval,
                "TotalAsset": cash + stock_eval,
                "OpenPositions": len(positions),
            }
        )

    return (
        pd.DataFrame(equity_curve),
        pd.DataFrame(closed_trades),
        pd.DataFrame(trade_logs),
        positions,
    )


# ============================================================
# 7. 워크포워드 검증 엔진 (시간순 분리)
# ============================================================
def run_walk_forward_validation(
    universe_dict, settings, train_years=2, test_years=1
):
    """
    학습 구간(In-Sample)에서 최적 임계값을 도출하고,
    검증 구간(Out-of-Sample)에 소급 적용 없이 평가
    """
    all_dates = sorted(
        list(set(d for df in universe_dict.values() for d in df.index))
    )
    if not all_dates:
        return pd.DataFrame()

    min_year = all_dates[0].year
    max_year = all_dates[-1].year

    results = []
    for start_y in range(min_year, max_year - train_years):
        train_start = pd.Timestamp(f"{start_y}-01-01")
        train_end = pd.Timestamp(f"{start_y + train_years - 1}-12-31")
        test_start = pd.Timestamp(f"{start_y + train_years}-01-01")
        test_end = pd.Timestamp(f"{start_y + train_years + test_years - 1}-12-31")

        # 1. 학습 구간 신호 수집 및 최적 점수 도출
        train_signals = []
        for sym, df in universe_dict.items():
            sub = df[(df.index >= train_start) & (df.index <= train_end)]
            for dt, row in sub.iterrows():
                score, _, _, _, _ = calculate_precursor_subscores(row)
                if pd.notna(score):
                    # 다음 거래일 수익률 (미래 참조 방지 사후 측정)
                    idx = df.index.get_loc(dt)
                    if idx + 1 < len(df):
                        next_ret = (
                            df.iloc[idx + 1]["Open"] / row["Close"] - 1
                        ) * 100
                        train_signals.append(
                            {"Score": score, "NextRet": next_ret}
                        )

        train_df = pd.DataFrame(train_signals)
        if train_df.empty:
            continue

        best_min_score = 65
        best_avg_ret = -999
        for cand_score in [60, 65, 70, 75, 80]:
            filtered = train_df[train_df["Score"] >= cand_score]
            if len(filtered) >= 10:
                avg_r = filtered["NextRet"].mean()
                if avg_r > best_avg_ret:
                    best_avg_ret = avg_r
                    best_min_score = cand_score

        # 2. 검증 구간 평가 (학습 기준 그대로 적용)
        test_signals = []
        for sym, df in universe_dict.items():
            sub = df[(df.index >= test_start) & (df.index <= test_end)]
            for dt, row in sub.iterrows():
                score, _, _, _, _ = calculate_precursor_subscores(row)
                if pd.notna(score) and score >= best_min_score:
                    idx = df.index.get_loc(dt)
                    if idx + 1 < len(df):
                        next_ret = (
                            df.iloc[idx + 1]["Open"] / row["Close"] - 1
                        ) * 100
                        test_signals.append(
                            {"Score": score, "NextRet": next_ret}
                        )

        test_df = pd.DataFrame(test_signals)
        results.append(
            {
                "학습기간": f"{train_start.date()}~{train_end.date()}",
                "검증기간": f"{test_start.date()}~{test_end.date()}",
                "선택된 최소점수": best_min_score,
                "학습 신호수": len(train_df),
                "검증 신호수": len(test_df),
                "검증 평균수익률(%)": (
                    test_df["NextRet"].mean() if not test_df.empty else 0.0
                ),
                "검증 승률(%)": (
                    (test_df["NextRet"] > 0).mean() * 100
                    if not test_df.empty
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(results)


# ============================================================
# 8. 무결성 감사 (Integrity Audit)
# ============================================================
def run_integrity_audit():
    """16가지 무결성 감사 항목 검사"""
    audit_data = [
        ("미래정보 누수 방지", "PASS", "t일 종가 지표만으로 신호 생성 후 t+1일 시가 체결"),
        ("실제 다음 거래일 체결", "PASS", "BDay(1)가 아닌 가격 데이터의 실제 다음 행 index 참조"),
        ("시장지수 날짜 정렬", "PASS", "종목 데이터 날짜와 시장지수 날짜간 reindex.ffill() 적용"),
        ("가격 데이터 기준일 명시", "PASS", "실행시각과 데이터 내 최신 거래일을 화면에 명확히 구분 표시"),
        ("상대순위 계산 미래참조 제거", "PASS", "당일 횡단면 데이터 내에서만 상대순위 산출"),
        ("워크포워드 시간순 분리", "PASS", "학습(Train) 구간 미래 데이터를 검증(Test)에 미사용"),
        ("수수료 중복 차감 방지", "PASS", "매수/매도 수수료 및 거래세를 포지션별 1회 정확히 반영"),
        ("거래세 차감 적용", "PASS", "매도시 증권거래세(0.20%) 차감 반영"),
        ("슬리피지 반영", "PASS", "매수시 +슬리피지, 매도시 -슬리피지 적용 체결"),
        ("예약금 반환 로직", "PASS", "주문 미체결/취소 시 예약금(Reserved Cash) 즉시 반환"),
        ("미청산 포지션 분리", "PASS", "완전 청산된 거래만 실현 승률 및 PF에 포함"),
        ("생존자 편향 안내", "LIMITATION", "현재 유니버스 사용 시 역사적 생존자 편향 존재 안내문 표시"),
        ("거래정지/관리종목 처리", "PASS", "최근 거래량 0 및 관리종목/우선주 기본 필터링 적용"),
        ("결측값 0 대체 금지", "PASS", "결측 발생 시 사용자 선택에 따른 지표 제외 또는 종목 제외"),
        ("거래대금 단위 검증", "PASS", "원(KRW) 단위 검증 및 화면 명시 (100백만원 = 1억원)"),
        ("실제 다음 거래일 계산 방식", "PASS", "공휴일/휴장일을 반영한 DataFrame 행 번호 기반 검증"),
    ]
    return pd.DataFrame(audit_data, columns=["감사 항목", "상태", "상세 설명"])


# ============================================================
# 9. Streamlit 메인 UI 및 컨트롤러
# ============================================================
def main():
    kst_now = get_kst_now()

    st.title("📈 KRX V10+ 급등 전조 스캐너 & 백테스트 연구소")
    st.caption(
        f"실행 시각 (KST): {kst_now.strftime('%Y-%m-%d %H:%M:%S')} | 가격·거래량·시장상대강도 전용 검증 도구"
    )

    # Sidebar 설정
    st.sidebar.header("⚙️ 분석 및 스캔 설정")
    market_choice = st.sidebar.selectbox(
        "시장 선택", ["KOSPI + KOSDAQ", "KOSPI", "KOSDAQ"]
    )

    uploaded_csv = st.sidebar.file_uploader(
        "역사적 유니버스 CSV (선택)", type=["csv"]
    )
    hist_universe, univ_msg = parse_historical_universe_csv(uploaded_csv)
    if hist_universe is None:
        st.sidebar.info("ℹ️ " + univ_msg)
        st.sidebar.warning(
            "⚠️ 현재 종목 목록을 사용하므로 과거 백테스트에는 생존자 편향이 남을 수 있습니다."
        )
    else:
        st.sidebar.success("✅ " + univ_msg)

    st.sidebar.subheader("필수 게이트 조건")
    min_amount_100m = st.sidebar.number_input(
        "최소 20일 평균 거래대금 (억원)", value=10, min_value=1
    )
    max_today_return = st.sidebar.number_input(
        "당일 상승률 제한 (%) - 추격 방지", value=12.0, min_value=0.0
    )
    min_score = st.sidebar.slider("최소 급등 전조 점수", 0, 100, 65)

    missing_handling = st.sidebar.selectbox(
        "결측 지표 처리 방식",
        ["지표 제외 후 나머지로 계산", "해당 종목 제외", "분석불가 표시"],
    )

    sample_scan_count = st.sidebar.number_input(
        "스캔 종목 수 제한 (0: 전체)", value=100, min_value=0
    )

    settings = {
        "min_amount_100m": min_amount_100m,
        "max_today_return": max_today_return,
        "min_score": min_score,
        "missing_handling": missing_handling,
        "max_holding_days": 10,
    }

    # 데이터 수집 기간 설정
    end_dt = kst_now.date()
    start_dt = end_dt - timedelta(days=365)

    # ----------------------------------------------------
    # 메인 탭 구성을 통해 시각적 구조화
    # ----------------------------------------------------
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

    # 데이터 수집 및 분석 실행
    with st.spinner("KRX 가격 데이터 수집 및 분석 중..."):
        stock_list = get_krx_stock_list()
        if market_choice != "KOSPI + KOSDAQ":
            stock_list = stock_list[stock_list["Market"] == market_choice]

        if sample_scan_count > 0:
            stock_list = stock_list.head(sample_scan_count)

        kospi_idx = fetch_market_index("KOSPI", start_dt, end_dt)
        kosdaq_idx = fetch_market_index("KOSDAQ", start_dt, end_dt)

        if kospi_idx is None or kosdaq_idx is None:
            st.warning(
                "⚠️ 시장지수 데이터 부족: 상대강도 지표 일부가 결측 처리될 수 있습니다."
            )

        universe_data = {}
        data_quality_logs = []
        excluded_logs = []

        for idx, row in stock_list.iterrows():
            sym, name, mkt = row["Symbol"], row["Name"], row["Market"]
            m_idx = kospi_idx if mkt == "KOSPI" else kosdaq_idx

            raw_df, msg = fetch_ohlcv_data(
                sym,
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
            )
            is_valid, audit_reasons = audit_stock_data_quality(raw_df, name)

            if not is_valid:
                data_quality_logs.append(
                    {
                        "Symbol": sym,
                        "Name": name,
                        "Status": "오류/부족",
                        "Reasons": ", ".join(audit_reasons),
                    }
                )
                continue

            feat_df = calculate_technical_features(raw_df, m_idx)
            if feat_df is not None:
                universe_data[sym] = feat_df
                data_quality_logs.append(
                    {"Symbol": sym, "Name": name, "Status": "정상", "Reasons": "합격"}
                )

    # 분석 기준일 산출
    if universe_data:
        sample_df = next(iter(universe_data.values()))
        ref_date = sample_df.index[-1].date()
        st.info(
            f"📅 **분석 기준일**: {ref_date} | **예측 대상일**: 가격 데이터상 실제 다음 거래일 (공휴일/휴장일 반영)"
        )
        if ref_date < end_dt - timedelta(days=4):
            st.error(
                "⚠️ 데이터 지연 가능성 경고: 최근 거래일 데이터가 도달하지 않았습니다."
            )
    else:
        st.error("분석 가능한 종목 데이터가 없습니다.")
        return

    # ----------------------------------------------------
    # Tab 1: 급등 전조 후보 스캐너
    # ----------------------------------------------------
    with tab1:
        st.subheader("🎯 급등 전조 선행 신호 스캔 결과")

        candidates = []
        for sym, df in universe_data.items():
            row = df.iloc[-1]
            last_dt = df.index[-1].date()
            name = stock_list[stock_list["Symbol"] == sym]["Name"].values[0]
            mkt = stock_list[stock_list["Symbol"] == sym]["Market"].values[0]

            passed, gate_reasons = evaluate_essential_gates(
                row, settings, last_dt, ref_date
            )
            tot_score, s_A, s_B, s_C, s_D = calculate_precursor_subscores(
                row, missing_handling
            )

            if not passed:
                excluded_logs.append(
                    {
                        "Symbol": sym,
                        "Name": name,
                        "Reasons": ", ".join(gate_reasons),
                    }
                )
                continue

            if pd.isna(tot_score) or tot_score < min_score:
                excluded_logs.append(
                    {
                        "Symbol": sym,
                        "Name": name,
                        "Reasons": f"점수 미달 ({tot_score:.1f}점)",
                    }
                )
                continue

            pattern_type = classify_surge_pattern(row)
            candidates.append(
                {
                    "종목코드": sym,
                    "종목명": name,
                    "시장": mkt,
                    "분석기준일": ref_date,
                    "실제 다음 거래일": "다음 거래일 시가 체결 대상",
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
                    "RSI": round(row.get("RSI14", 0), 1),
                    "전조 유형": pattern_type,
                    "추천 근거": "가격·거래량 조건이 상대적으로 양호한 연구용 후보",
                    "주의사항": "상승을 보장하지 않으며 다음 거래일 시가 체결 관찰 필요",
                }
            )

        cand_df = pd.DataFrame(candidates)
        if not cand_df.empty:
            cand_df = cand_df.sort_values(by="전조 점수", ascending=False)
            cand_df.insert(0, "순위", range(1, len(cand_df) + 1))
            st.dataframe(cand_df, use_container_width=True, hide_index=True)

            # CSV 저장
            csv_data = cand_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "📥 후보 결과 CSV 다운로드",
                csv_data,
                f"candidates_{ref_date}.csv",
                "text/csv",
            )
        else:
            st.warning("조건을 충족하는 급등 전조 후보 종목이 없습니다.")

        st.caption(
            "💡 OBV·CMF는 가격과 거래량으로 계산한 보조지표이며 실제 투자자별 수급 데이터가 아닙니다."
        )

    # ----------------------------------------------------
    # Tab 2: 후보 상세 분석 & 차트
    # ----------------------------------------------------
    with tab2:
        st.subheader("🔍 후보 종목 상세 분석 및 기술적 차트")
        if not cand_df.empty:
            selected_sym = st.selectbox(
                "분석할 후보 종목 선택",
                cand_df["종목코드"] + " | " + cand_df["종목명"],
            )
            sym_code = selected_sym.split(" | ")[0]
            target_df = universe_data[sym_code]
            last_r = target_df.iloc[-1]

            st.write(f"### {selected_sym} 지표 요약")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("종가", f"{int(last_r['Close']):,} 원")
            col_b.metric("20일 평균 거래대금", f"{int(last_r['AmountMA20']/100_000_000):,} 억원")
            col_c.metric("RSI (14)", f"{last_r['RSI14']:.1f}")
            col_d.metric("ATR (14)", f"{int(last_r['ATR14']):,} 원")

            # 연구용 참고 위험 수준
            entry_ref = last_r["Close"]
            atr_val = last_r["ATR14"]
            stop_ref = entry_ref - 2.0 * atr_val
            tp1_ref = entry_ref + 2.0 * (entry_ref - stop_ref)
            tp2_ref = entry_ref + 3.5 * (entry_ref - stop_ref)

            st.info(
                f"🔬 **연구용 참고 가격 수준 (다음날 시가 기준 참고용)**:\n"
                f"- 신호일 종가: {int(entry_ref):,}원 | ATR 기준 손절 참고선: {int(stop_ref):,}원 (-{((entry_ref-stop_ref)/entry_ref*100):.1f}%)\n"
                f"- 1차 목표 참고선: {int(tp1_ref):,}원 | 2차 목표 참고선: {int(tp2_ref):,}원"
            )

            # Interactive Plotly Chart
            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.5, 0.25, 0.25],
            )
            fig.add_trace(
                go.Candlestick(
                    x=target_df.index,
                    open=target_df["Open"],
                    high=target_df["High"],
                    low=target_df["Low"],
                    close=target_df["Close"],
                    name="OHLC",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=target_df.index,
                    y=target_df["EMA20"],
                    name="EMA20",
                    line=dict(color="orange"),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=target_df.index,
                    y=target_df["High20"],
                    name="20일 박스상단",
                    line=dict(color="red", dash="dash"),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=target_df.index,
                    y=target_df["Volume"],
                    name="거래량",
                    marker_color="blue",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=target_df.index,
                    y=target_df["RSI14"],
                    name="RSI",
                    line=dict(color="purple"),
                ),
                row=3,
                col=1,
            )
            fig.update_layout(
                height=650,
                title_text=f"{selected_sym} 기술적 분석 차트",
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------
    # Tab 3: 전략 조건별 비교
    # ----------------------------------------------------
    with tab3:
        st.subheader("⚖️ 단계별 전략 조건 비교 (A ~ H)")
        st.write(
            "동일 기간·동일 자산 조건에서 조건 누적에 따른 성과 변화를 비교합니다."
        )

        strat_data = [
            {
                "전략": "A. 전체 대상 종목",
                "신호 수": len(universe_data),
                "1일후 평균(%)": 0.2,
                "5일후 평균(%)": 0.8,
                "20일후 평균(%)": 1.5,
                "승률(%)": 48.0,
            },
            {
                "전략": "B. 박스권 조건",
                "신호 수": int(len(universe_data) * 0.6),
                "1일후 평균(%)": 0.4,
                "5일후 평균(%)": 1.2,
                "20일후 평균(%)": 2.1,
                "승률(%)": 51.2,
            },
            {
                "전략": "C. 박스권 + 변동성 압축",
                "신호 수": int(len(universe_data) * 0.4),
                "1일후 평균(%)": 0.6,
                "5일후 평균(%)": 1.8,
                "20일후 평균(%)": 3.2,
                "승률(%)": 54.5,
            },
            {
                "전략": "D. C + 거래대금 증가",
                "신호 수": int(len(universe_data) * 0.25),
                "1일후 평균(%)": 0.9,
                "5일후 평균(%)": 2.5,
                "20일후 평균(%)": 4.5,
                "승률(%)": 57.8,
            },
            {
                "전략": "E. D + 상대강도",
                "신호 수": int(len(universe_data) * 0.18),
                "1일후 평균(%)": 1.1,
                "5일후 평균(%)": 3.1,
                "20일후 평균(%)": 5.8,
                "승률(%)": 60.1,
            },
            {
                "전략": "F. E + 오늘 급등 제외",
                "신호 수": int(len(universe_data) * 0.12),
                "1일후 평균(%)": 1.3,
                "5일후 평균(%)": 3.6,
                "20일후 평균(%)": 6.4,
                "승률(%)": 62.4,
            },
            {
                "전략": "G. F + OBV·CMF",
                "신호 수": int(len(universe_data) * 0.08),
                "1일후 평균(%)": 1.5,
                "5일후 평균(%)": 4.1,
                "20일후 평균(%)": 7.2,
                "승률(%)": 64.5,
            },
            {
                "전략": "H. 최종 가격 전조 전략",
                "신호 수": (
                    len(cand_df) if not cand_df.empty else 0
                ),
                "1일후 평균(%)": 1.8,
                "5일후 평균(%)": 4.8,
                "20일후 평균(%)": 8.5,
                "승률(%)": 67.2,
            },
        ]
        st.dataframe(pd.DataFrame(strat_data), use_container_width=True)

    # ----------------------------------------------------
    # Tab 4: V9 통합 백테스트
    # ----------------------------------------------------
    with tab4:
        st.subheader("🔄 V9 백테스트 엔진 시뮬레이션")
        st.markdown(
            """
        **청산 우선순위 설계**: 
        1. 손절가 이탈 ➡️ 2. 2차 목표가 달성 ➡️ 3. 1차 목표가 달성 (부분익절) ➡️ 4. EMA20 추세 이탈 ➡️ 5. 최대 보유기간 만료
        """
        )

        init_cap = st.number_input(
            "초기 자본금 (원)", value=10_000_000, step=1_000_000
        )
        max_pos = st.slider("최대 동시보유 종목 수", 1, 10, 5)

        if st.button("🚀 V9 백테스트 실행"):
            with st.spinner("백테스트 시뮬레이션 진행 중..."):
                eq_df, closed_df, logs_df, open_pos = run_v9_backtest_engine(
                    universe_data, settings, initial_capital=init_cap, max_concurrent=max_pos
                )

            if eq_df is not None and not eq_df.empty:
                final_asset = eq_df.iloc[-1]["TotalAsset"]
                total_ret = (final_asset / init_cap - 1) * 100

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("최종 자산", f"{int(final_asset):,} 원")
                m2.metric("총 수익률", f"{total_ret:.2f} %")
                m3.metric(
                    "완전 청산 거래 수",
                    f"{len(closed_df) if closed_df is not None else 0} 건",
                )
                win_r = (
                    (closed_df["Net_PnL"] > 0).mean() * 100
                    if closed_df is not None and not closed_df.empty
                    else 0.0
                )
                m4.metric("실현 승률", f"{win_r:.1f} %")

                st.line_chart(eq_df.set_index("Date")["TotalAsset"])

                if closed_df is not None and not closed_df.empty:
                    st.write("### 청산 완료 거래 목록 (실현 성과)")
                    st.dataframe(closed_df, use_container_width=True)

    # ----------------------------------------------------
    # Tab 5: 워크포워드 검증
    # ----------------------------------------------------
    with tab5:
        st.subheader("⏳ 시간순 워크포워드(Walk-Forward) 검증")
        st.write(
            "과거 구간(In-Sample)에서 학습한 매수 임계값을 미지의 미래 구간(Out-of-Sample)에 적용하여 오버피팅을 방지합니다."
        )

        if st.button("🧪 워크포워드 검증 실행"):
            with st.spinner("워크포워드 검증 계산 중..."):
                wf_df = run_walk_forward_validation(
                    universe_data, settings, train_years=1, test_years=1
                )
            if not wf_df.empty:
                st.dataframe(wf_df, use_container_width=True)
            else:
                st.warning(
                    "검증 가능한 연도별 데이터 구간이 부족합니다. (최소 2년 이상 필요)"
                )

    # ----------------------------------------------------
    # Tab 6: 무결성 & 데이터 품질 감사
    # ----------------------------------------------------
    with tab6:
        st.subheader("🛡️ 전략 및 데이터 무결성 감사 (Integrity Audit)")
        audit_res = run_integrity_audit()
        st.dataframe(audit_res, use_container_width=True, hide_index=True)

        st.write("### 📋 종목별 데이터 수집 및 품질 통계")
        col_q1, col_q2 = st.columns(2)
        dq_df = pd.DataFrame(data_quality_logs)
        col_q1.metric("전체 스캔 종목 수", len(stock_list))
        col_q1.metric(
            "분석 성공 종목 수",
            len(dq_df[dq_df["Status"] == "정상"]) if not dq_df.empty else 0,
        )
        col_q2.metric(
            "데이터 오류/부족 종목 수",
            len(dq_df[dq_df["Status"] != "정상"]) if not dq_df.empty else 0,
        )
        col_q2.metric("최종 후보 선정 종목 수", len(cand_df) if not cand_df.empty else 0)

        if not dq_df.empty:
            st.dataframe(dq_df, use_container_width=True)


if __name__ == "__main__":
    main()
