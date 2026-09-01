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
.metric-card {  
    background-color: #f8f9fa;  
    border: 1px solid #e9ecef;  
    border-radius: 8px;  
    padding: 12px;  
    margin-bottom: 10px;  
}  
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
        alpha=1 / period,  
        adjust=False,  
        min_periods=period,  
    ).mean()  
  
    avg_loss = loss.ewm(  
        alpha=1 / period,  
        adjust=False,  
        min_periods=period,  
    ).mean()  
  
    rs = avg_gain / avg_loss.replace(0, np.nan)  
    rsi = 100 - (100 / (1 + rs))  
  
    rsi = rsi.where(  
        ~((avg_loss == 0) & (avg_gain > 0)),  
        100,  
    )  
    rsi = rsi.where(  
        ~((avg_loss == 0) & (avg_gain == 0)),  
        50,  
    )  
  
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
  
    return tr.ewm(  
        alpha=1 / period,  
        adjust=False,  
        min_periods=period,  
    ).mean()  
  
  
# ============================================================  
# FIX: CMF 결측값을 0으로 대체하지 않음  
# ============================================================  
def calculate_cmf(df, period=20):  
    high = df["High"]  
    low = df["Low"]  
    close = df["Close"]  
    volume = df["Volume"]  
  
    denominator = (high - low).replace(0, np.nan)  
  
    mf_multiplier = (  
        ((close - low) - (high - close))  
        / denominator  
    )  
  
    mf_volume = mf_multiplier * volume  
  
    volume_sum = volume.rolling(  
        period,  
        min_periods=period,  
    ).sum()  
  
    mf_volume_sum = mf_volume.rolling(  
        period,  
        min_periods=period,  
    ).sum()  
  
    # FIX:  
    # 계산 불가능한 초기 구간 및 거래량 합계 0 구간은 NaN 유지.  
    cmf = mf_volume_sum / volume_sum.replace(0, np.nan)  
  
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
            (  
                c  
                for c in ["Symbol", "Code", "종목코드"]  
                if c in df.columns  
            ),  
            None,  
        )  
  
        name_col = next(  
            (  
                c  
                for c in ["Name", "종목명", " 종목명"]  
                if c in df.columns  
            ),  
            None,  
        )  
  
        market_col = next(  
            (  
                c  
                for c in ["Market", "시장"]  
                if c in df.columns  
            ),  
            None,  
        )  
  
        if not symbol_col or not name_col:  
            return pd.DataFrame(  
                columns=["Symbol", "Name", "Market"]  
            )  
  
        rename_map = {  
            symbol_col: "Symbol",  
            name_col: "Name",  
        }  
  
        if market_col:  
            rename_map[market_col] = "Market"  
  
        df = df.rename(columns=rename_map)  
  
        df["Symbol"] = (  
            df["Symbol"]  
            .astype(str)  
            .str.extract(r"(\d{6})")[0]  
        )  
  
        df = df.dropna(subset=["Symbol"])  
  
        if "Market" in df.columns:  
            df = df[  
                df["Market"].isin(["KOSPI", "KOSDAQ"])  
            ].copy()  
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
  
        mask = ~(  
            df["Name"]  
            .astype(str)  
            .str.upper()  
            .str.contains(  
                pattern,  
                regex=True,  
                na=False,  
            )  
        )  
  
        df = df[mask].copy()  
  
        pref_mask = df["Name"].astype(str).str.contains(  
            r"우$|우B$|우C$|우\(전환\)$",  
            regex=True,  
            na=False,  
        )  
  
        df = df[~pref_mask].copy()  
  
        return (  
            df[  
                ["Symbol", "Name", "Market"]  
            ]  
            .drop_duplicates("Symbol")  
            .reset_index(drop=True)  
        )  
  
    except Exception as e:  
        st.error(  
            f"KRX 종목 목록 로드 중 오류 발생: {e}"  
        )  
  
        return pd.DataFrame(  
            columns=["Symbol", "Name", "Market"]  
        )  
  
  
# ============================================================  
# FIX: 역사적 유니버스 CSV 실제 적용용 파서  
# ============================================================  
def parse_historical_universe_csv(uploaded_file):  
    """  
    역사적 유니버스 CSV 파싱.  
  
    지원:  
    Symbol / Code / Ticker / 종목코드  
    Name  
    Market  
    StartDate / 시작일  
    EndDate / 종료일  
    """  
  
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
            "종목명": "Name",  
            "시장": "Market",  
        }  
  
        df = df.rename(  
            columns={  
                k: v  
                for k, v in alias.items()  
                if k in df.columns  
            }  
        )  
  
        if "Symbol" not in df.columns:  
            return (  
                None,  
                "CSV 파일에 Code 또는 Symbol 컬럼이 존재하지 않습니다.",  
            )  
  
        df["Symbol"] = (  
            df["Symbol"]  
            .astype(str)  
            .str.extract(r"(\d{6})")[0]  
        )  
  
        df = df.dropna(subset=["Symbol"]).copy()  
  
        if "Name" not in df.columns:  
            df["Name"] = df["Symbol"]  
  
        if "Market" not in df.columns:  
            df["Market"] = "ALL"  
  
        df["Market"] = (  
            df["Market"]  
            .astype(str)  
            .str.upper()  
            .str.strip()  
        )  
  
        # FIX:  
        # 날짜가 없으면 전체 기간 유효.  
        if "StartDate" in df.columns:  
            df["StartDate"] = pd.to_datetime(  
                df["StartDate"],  
                errors="coerce",  
            )  
        else:  
            df["StartDate"] = pd.Timestamp("1900-01-01")  
  
        if "EndDate" in df.columns:  
            df["EndDate"] = pd.to_datetime(  
                df["EndDate"],  
                errors="coerce",  
            )  
        else:  
            df["EndDate"] = pd.Timestamp("2100-12-31")  
  
        df["StartDate"] = df["StartDate"].fillna(  
            pd.Timestamp("1900-01-01")  
        )  
  
        df["EndDate"] = df["EndDate"].fillna(  
            pd.Timestamp("2100-12-31")  
        )  
  
        # 잘못된 기간 제거  
        df = df[  
            df["StartDate"] <= df["EndDate"]  
        ].copy()  
  
        df = df[  
            [  
                "Symbol",  
                "Name",  
                "Market",  
                "StartDate",  
                "EndDate",  
            ]  
        ].drop_duplicates()  
  
        if df.empty:  
            return None, "유효한 CSV 종목이 없습니다."  
  
        return (  
            df.reset_index(drop=True),  
            "역사적 유니버스 CSV 적용 완료",  
        )  
  
    except Exception as e:  
        return (  
            None,  
            f"CSV 업로드 파싱 오류: {e}",  
        )  
  
  
# ============================================================  
# FIX: 원천 거래대금 컬럼 보존  
# ============================================================  
@st.cache_data(ttl=1800, show_spinner=False)  
def fetch_ohlcv_data(  
    symbol,  
    start_date,  
    end_date,  
):  
    """  
    단일 종목 OHLCV 수집.  
  
    원천 Amount 계열을 먼저 보존하고,  
    없을 때만 Close × Volume으로 계산한다.  
    """  
  
    try:  
        raw = fdr.DataReader(  
            symbol,  
            start_date,  
            end_date,  
        )  
  
        if raw is None or raw.empty:  
            return None, "데이터 없음", {}  
  
        raw = raw.copy()  
  
        required = [  
            "Open",  
            "High",  
            "Low",  
            "Close",  
            "Volume",  
        ]  
  
        if not all(  
            col in raw.columns  
            for col in required  
        ):  
            return (  
                None,  
                "필수 OHLCV 컬럼 누락",  
                {},  
            )  
  
        # ====================================================  
        # FIX:  
        # Amount 원천 컬럼을 OHLCV 선택 전에 탐색  
        # ====================================================  
        amount_candidates = [  
            "Amount",  
            "amount",  
            "거래대금",  
            "TradingValue",  
            "Value",  
        ]  
  
        amount_source = None  
  
        for col in amount_candidates:  
            if col in raw.columns:  
                amount_source = col  
                break  
  
        df = raw[  
            required  
        ].copy()  
  
        df = df[  
            ~df.index.duplicated(  
                keep="last"  
            )  
        ].sort_index()  
  
        # ====================================================  
        # FIX:  
        # 원천 Amount 우선 사용  
        # ====================================================  
        if amount_source is not None:  
            amount_series = pd.to_numeric(  
                raw[amount_source],  
                errors="coerce",  
            )  
  
            amount_series = amount_series[  
                ~amount_series.index.duplicated(  
                    keep="last"  
                )  
            ].sort_index()  
  
            amount_series = amount_series.reindex(  
                df.index  
            )  
  
            # FIX:  
            # 결측을 0으로 바꾸지 않는다.  
            if amount_series.isna().any():  
                return (  
                    None,  
                    "원천 거래대금 결측",  
                    {  
                        "amount_source": amount_source,  
                        "amount_unit_status": "CHECK",  
                    },  
                )  
  
            if (  
                np.isinf(  
                    amount_series.to_numpy(  
                        dtype=float  
                    )  
                ).any()  
            ):  
                return (  
                    None,  
                    "원천 거래대금 무한값",  
                    {  
                        "amount_source": amount_source,  
                        "amount_unit_status": "CHECK",  
                    },  
                )  
  
            df["Amount"] = amount_series  
  
            amount_source_label = (  
                f"원천 {amount_source}"  
            )  
  
        else:  
            # =================================================  
            # FIX:  
            # 원천 거래대금이 없을 때만 계산  
            # =================================================  
            close_numeric = pd.to_numeric(  
                df["Close"],  
                errors="coerce",  
            )  
  
            volume_numeric = pd.to_numeric(  
                df["Volume"],  
                errors="coerce",  
            )  
  
            if (  
                close_numeric.isna().any()  
                or volume_numeric.isna().any()  
            ):  
                return (  
                    None,  
                    "거래대금 계산용 가격/거래량 결측",  
                    {  
                        "amount_source": "계산 불가",  
                        "amount_unit_status": "CHECK",  
                    },  
                )  
  
            df["Amount"] = (  
                close_numeric  
                * volume_numeric  
            )  
  
            amount_source_label = (  
                "Close × Volume 계산"  
            )  
  
        # ====================================================  
        # FIX:  
        # 거래대금은 음수/무한값을 허용하지 않음  
        # ====================================================  
        if (  
            df["Amount"].isna().any()  
            or np.isinf(  
                df["Amount"].to_numpy(  
                    dtype=float  
                )  
            ).any()  
        ):  
            return (  
                None,  
                "거래대금 결측/무한값",  
                {  
                    "amount_source": amount_source_label,  
                    "amount_unit_status": "CHECK",  
                },  
            )  
  
        if (df["Amount"] < 0).any():  
            return (  
                None,  
                "음수 거래대금",  
                {  
                    "amount_source": amount_source_label,  
                    "amount_unit_status": "CHECK",  
                },  
            )  
  
        if len(df) < 30:  
            return (  
                None,  
                f"데이터 행 수 부족 ({len(df)}행)",  
                {  
                    "amount_source": amount_source_label,  
                    "amount_unit_status": "CHECK",  
                },  
            )  
  
        meta = {  
            "amount_source": amount_source_label,  
  
            # FIX:  
            # 현재 데이터 공급원의 거래대금 단위를 자동 검증할  
            # 수 없으므로 CHECK.  
            "amount_unit_status": (  
                "CHECK - 원(KRW) 단위로 취급하나 "  
                "원천 단위 자동 검증 불가"  
            ),  
        }  
  
        return df, "성공", meta  
  
    except Exception as e:  
        return (  
            None,  
            f"수집 에러: {str(e)}",  
            {},  
        )  
  
  
@st.cache_data(ttl=1800, show_spinner=False)  
def fetch_market_index(  
    market_type,  
    start_date,  
    end_date,  
):  
    """시장지수 수집"""  
  
    ticker = (  
        "KS11"  
        if market_type == "KOSPI"  
        else "KQ11"  
    )  
  
    try:  
        df = fdr.DataReader(  
            ticker,  
            start_date,  
            end_date,  
        )  
  
        if (  
            df is None  
            or df.empty  
            or "Close" not in df.columns  
        ):  
            return None  
  
        return (  
            df["Close"]  
            .dropna()  
            .sort_index()  
        )  
  
    except Exception:  
        return None  
  
  
# ============================================================  
# FIX: 역사적 유니버스 유효기간 필터  
# ============================================================  
def apply_historical_universe_validity(  
    feat_df,  
    symbol,  
    historical_universe,  
):  
    """  
    각 날짜마다 CSV의 StartDate <= 날짜 <= EndDate인 경우만  
    해당 종목의 유효 데이터로 남긴다.  
    """  
  
    if (  
        historical_universe is None  
        or historical_universe.empty  
    ):  
        return feat_df  
  
    rules = historical_universe[  
        historical_universe["Symbol"].astype(str)  
        == str(symbol)  
    ]  
  
    if rules.empty:  
        return feat_df.iloc[0:0].copy()  
  
    idx = pd.DatetimeIndex(  
        pd.to_datetime(feat_df.index)  
    )  
  
    valid_mask = np.zeros(  
        len(feat_df),  
        dtype=bool,  
    )  
  
    for _, rule in rules.iterrows():  
        start_date = pd.Timestamp(  
            rule["StartDate"]  
        ).normalize()  
  
        end_date = pd.Timestamp(  
            rule["EndDate"]  
        ).normalize()  
  
        valid_mask |= (  
            (idx.normalize() >= start_date)  
            & (idx.normalize() <= end_date)  
        )  
  
    return feat_df.loc[valid_mask].copy()  
  
  
# ============================================================  
# 2. 기술 지표 및 급등 전조 특징 계산  
# ============================================================  
def calculate_technical_features(  
    df,  
    market_index_series=None,  
):  
    """기술적 지표 및 V10 전조 특징 생성"""  
  
    if df is None or len(df) < 20:  
        return None  
  
    x = df.copy()  
  
    close = x["Close"]  
    high = x["High"]  
    low = x["Low"]  
    open_px = x["Open"]  
    volume = x["Volume"]  
    amount = x["Amount"]  
  
    # ========================================================  
    # FIX:  
    # Return60 추가.  
    # 기존 Return60 누락으로 KeyError 발생.  
    # ========================================================  
    for d in [  
        1,  
        3,  
        5,  
        10,  
        20,  
        60,  
    ]:  
        x[f"Return{d}"] = (  
            close.pct_change(d) * 100  
        )  
  
    # 이동평균  
    x["SMA20"] = close.rolling(  
        20,  
        min_periods=20,  
    ).mean()  
  
    x["SMA60"] = close.rolling(  
        60,  
        min_periods=60,  
    ).mean()  
  
    x["EMA20"] = close.ewm(  
        span=20,  
        adjust=False,  
        min_periods=20,  
    ).mean()  
  
    x["EMA60"] = close.ewm(  
        span=60,  
        adjust=False,  
        min_periods=60,  
    ).mean()  
  
    x["EMA120"] = close.ewm(  
        span=120,  
        adjust=False,  
        min_periods=120,  
    ).mean()  
  
    x["EMA20_Slope"] = (  
        x["EMA20"] - x["EMA20"].shift(5)  
    )  
  
    x["EMA60_Slope"] = (  
        x["EMA60"] - x["EMA60"].shift(10)  
    )  
  
    x["DisparityEMA20"] = (  
        close / x["EMA20"] - 1  
    ) * 100  
  
    # 박스권  
    x["High20"] = high.rolling(  
        20,  
        min_periods=20,  
    ).max()  
  
    x["Low20"] = low.rolling(  
        20,  
        min_periods=20,  
    ).min()  
  
    x["High60"] = high.rolling(  
        60,  
        min_periods=60,  
    ).max()  
  
    x["BoxWidth20"] = (  
        (  
            x["High20"]  
            - x["Low20"]  
        )  
        / x["Low20"].replace(  
            0,  
            np.nan,  
        )  
        * 100  
    )  
  
    x["BoxWidth5"] = (  
        (  
            high.rolling(5).max()  
            - low.rolling(5).min()  
        )  
        / low.rolling(5)  
        .min()  
        .replace(0, np.nan)  
        * 100  
    )  
  
    x["BoxPosition20"] = (  
        (close - x["Low20"])  
        / (  
            x["High20"]  
            - x["Low20"]  
        ).replace(0, np.nan)  
    )  
  
    x["BoxWidthRatio5_20"] = (  
        x["BoxWidth5"]  
        / x["BoxWidth20"].replace(  
            0,  
            np.nan,  
        )  
    )  
  
    # 고점/저점  
    x["HigherLow5"] = (  
        low.rolling(5).min()  
        > low.shift(5).rolling(5).min()  
    )  
  
    x["HigherHigh5"] = (  
        high.rolling(5).max()  
        > high.shift(5).rolling(5).max()  
    )  
  
    x["DistToHigh20"] = (  
        (x["High20"] - close)  
        / close  
        * 100  
    )  
  
    x["DistToHigh60"] = (  
        (x["High60"] - close)  
        / close  
        * 100  
    )  
  
    low5 = low.rolling(5).min()  
  
    x["LowSlope5"] = (  
        (low5 - low5.shift(5))  
        / low5.shift(5).replace(  
            0,  
            np.nan,  
        )  
        * 100  
    )  
  
    high5 = high.rolling(5).max()  
  
    x["HighSlope5"] = (  
        (high5 - high5.shift(5))  
        / high5.shift(5).replace(  
            0,  
            np.nan,  
        )  
        * 100  
    )  
  
    # 변동성  
    x["ATR14"] = wilder_atr(  
        x,  
        14,  
    )  
  
    x["ATR_Ratio"] = (  
        x["ATR14"]  
        / close  
        * 100  
    )  
  
    x["Vol5"] = (  
        close.pct_change()  
        .rolling(5)  
        .std()  
    )  
  
    x["Vol20"] = (  
        close.pct_change()  
        .rolling(20)  
        .std()  
    )  
  
    x["VolRatio5_20"] = (  
        x["Vol5"]  
        / x["Vol20"].replace(  
            0,  
            np.nan,  
        )  
    )  
  
    bb_mid = close.rolling(  
        20,  
        min_periods=20,  
    ).mean()  
  
    bb_std = close.rolling(  
        20,  
        min_periods=20,  
    ).std()  
  
    x["BB_Width"] = (  
        4 * bb_std  
        / bb_mid.replace(  
            0,  
            np.nan,  
        )  
    )  
  
    # 거래량  
    x["VolumeMA20"] = volume.rolling(  
        20,  
        min_periods=20,  
    ).mean()  
  
    x["VolumeMA60"] = volume.rolling(  
        60,  
        min_periods=60,  
    ).mean()  
  
    x["VolumeRatio20"] = (  
        volume  
        / x["VolumeMA20"].replace(  
            0,  
            np.nan,  
        )  
    )  
  
    x["VolumeRatio60"] = (  
        volume  
        / x["VolumeMA60"].replace(  
            0,  
            np.nan,  
        )  
    )  
  
    # 거래대금  
    x["AmountMA20"] = amount.rolling(  
        20,  
        min_periods=20,  
    ).mean()  
  
    x["AmountRatio20"] = (  
        amount  
        / x["AmountMA20"].replace(  
            0,  
            np.nan,  
        )  
    )  
  
    x["AmountChange5"] = (  
        (  
            amount  
            - amount.shift(5)  
        )  
        / amount.shift(5).replace(  
            0,  
            np.nan,  
        )  
        * 100  
    )  
  
    # OBV  
    obv_dir = (  
        np.sign(close.diff())  
        .fillna(0)  
    )  
  
    x["OBV"] = (  
        obv_dir * volume  
    ).cumsum()  
  
    x["OBV_MA20"] = x["OBV"].rolling(  
        20,  
        min_periods=20,  
    ).mean()  
  
    x["OBV_Slope5"] = (  
        x["OBV"]  
        - x["OBV"].shift(5)  
    )  
  
    # ========================================================  
    # FIX:  
    # CMF 계산 불가 구간은 NaN 유지  
    # ========================================================  
    x["CMF20"] = calculate_cmf(  
        x,  
        20,  
    )  
  
    # RSI / 캔들  
    x["RSI14"] = wilder_rsi(  
        close,  
        14,  
    )  
  
    candle_range = (  
        high - low  
    ).replace(  
        0,  
        np.nan,  
    )  
  
    x["CloseLocation"] = (  
        close - low  
    ) / candle_range  
  
    x["UpperWickRatio"] = (  
        high  
        - np.maximum(  
            open_px,  
            close,  
        )  
    ) / candle_range  
  
    # 시장 상대수익률  
    if (  
        market_index_series is not None  
        and not market_index_series.empty  
    ):  
        m_series = (  
            market_index_series  
            .reindex(x.index)  
            .ffill()  
        )  
  
        x["MarketReturn5"] = (  
            m_series.pct_change(5)  
            * 100  
        )  
  
        x["MarketReturn20"] = (  
            m_series.pct_change(20)  
            * 100  
        )  
  
        x["MarketReturn60"] = (  
            m_series.pct_change(60)  
            * 100  
        )  
  
        x["RelReturn5"] = (  
            x["Return5"]  
            - x["MarketReturn5"]  
        )  
  
        x["RelReturn20"] = (  
            x["Return20"]  
            - x["MarketReturn20"]  
        )  
  
        x["RelReturn60"] = (  
            x["Return60"]  
            - x["MarketReturn60"]  
        )  
  
    else:  
        for d in [  
            5,  
            20,  
            60,  
        ]:  
            x[f"MarketReturn{d}"] = np.nan  
            x[f"RelReturn{d}"] = np.nan  
  
    return x  
  
  
# ============================================================  
# 3. 데이터 품질 검증 및 통계  
# ============================================================  
def audit_stock_data_quality(  
    raw_df,  
    sym_name,  
    data_meta=None,  
):  
    """종목별 데이터 무결성 검증"""  
  
    reasons = []  
  
    if raw_df is None or raw_df.empty:  
        return False, ["데이터 없음"]  
  
    if len(raw_df) < 40:  
        reasons.append(  
            f"데이터 부족 ({len(raw_df)}행)"  
        )  
  
    if raw_df.index.duplicated().any():  
        reasons.append("중복 거래일 존재")  
  
    if not raw_df.index.is_monotonic_increasing:  
        reasons.append("날짜 정렬 오류")  
  
    kst_today = get_kst_now().date()  
  
    if (  
        raw_df.index  
        > pd.Timestamp(kst_today)  
    ).any():  
        reasons.append(  
            "미래 날짜 포함 오류"  
        )  
  
    required_cols = [  
        "Open",  
        "High",  
        "Low",  
        "Close",  
        "Volume",  
        "Amount",  
    ]  
  
    null_cols = (  
        raw_df[required_cols]  
        .isnull()  
        .sum()  
    )  
  
    if null_cols.sum() > 0:  
        reasons.append(  
            "OHLCV/거래대금 결측값 포함"  
        )  
  
    if (  
        raw_df[  
            [  
                "Open",  
                "High",  
                "Low",  
                "Close",  
            ]  
        ]  
        <= 0  
    ).any().any():  
        reasons.append(  
            "가격 0 이하 비정상 데이터"  
        )  
  
    if (  
        raw_df["High"]  
        < raw_df["Low"]  
    ).any():  
        reasons.append(  
            "High < Low 비정상 데이터"  
        )  
  
    if (  
        raw_df["Volume"] < 0  
    ).any() or (  
        raw_df["Amount"] < 0  
    ).any():  
        reasons.append(  
            "음수 거래량/거래대금 오류"  
        )  
  
    recent_v = raw_df[  
        "Volume"  
    ].tail(5)  
  
    if (  
        recent_v == 0  
    ).all():  
        reasons.append(  
            "최근 5일 연속 거래량 0 "  
            "(거래정지 추정)"  
        )  
  
    return (  
        len(reasons) == 0,  
        reasons,  
    )  
  
  
# ============================================================  
# FIX: 전조 지표 결측 현황  
# ============================================================  
def get_missing_precursor_indicators(row):  
    cols = [  
        "VolRatio5_20",  
        "BoxWidthRatio5_20",  
        "ATR_Ratio",  
        "BB_Width",  
        "DistToHigh20",  
        "HigherLow5",  
        "HigherHigh5",  
        "CloseLocation",  
        "VolumeRatio20",  
        "AmountRatio20",  
        "OBV_Slope5",  
        "CMF20",  
        "Return1",  
        "Return5",  
        "DisparityEMA20",  
        "RSI14",  
    ]  
  
    missing = []  
  
    for col in cols:  
        if col not in row.index:  
            missing.append(col)  
        elif pd.isna(row[col]):  
            missing.append(col)  
  
    return missing  
  
  
# ============================================================  
# 4. 가격 전조 점수 설계  
# ============================================================  
def calculate_precursor_subscores(  
    row,  
    missing_handling="지표 제외 후 나머지로 계산",  
):  
    """  
    A. 압축 25%  
    B. 돌파 준비 30%  
    C. 자금 유입 25%  
    D. 추격 위험 20%  
    """  
  
    def eval_score(condition_dict):  
        scores = []  
        weights = []  
  
        for val, (  
            score_val,  
            weight,  
        ) in condition_dict.items():  
  
            if pd.isna(val):  
                if missing_handling in [  
                    "해당 종목 제외",  
                    "분석불가 표시",  
                ]:  
                    return np.nan  
  
                # 지표 제외 후 나머지로 계산  
                continue  
  
            scores.append(  
                score_val  
            )  
            weights.append(  
                weight  
            )  
  
        if not scores:  
            return np.nan  
  
        return float(  
            np.sum(  
                np.array(scores)  
                * np.array(weights)  
            )  
            / np.sum(weights)  
        )  
  
    # A  
    comp_dict = {  
        row.get(  
            "VolRatio5_20"  
        ): (  
            100  
            if row.get(  
                "VolRatio5_20",  
                1,  
            )  
            < 0.7  
            else 50,  
            0.3,  
        ),  
        row.get(  
            "BoxWidthRatio5_20"  
        ): (  
            100  
            if row.get(  
                "BoxWidthRatio5_20",  
                1,  
            )  
            < 0.6  
            else 50,  
            0.3,  
        ),  
        row.get(  
            "ATR_Ratio"  
        ): (  
            100  
            if row.get(  
                "ATR_Ratio",  
                5,  
            )  
            < 3.5  
            else 40,  
            0.2,  
        ),  
        row.get(  
            "BB_Width"  
        ): (  
            100  
            if row.get(  
                "BB_Width",  
                0.2,  
            )  
            < 0.1  
            else 50,  
            0.2,  
        ),  
    }  
  
    score_A = eval_score(  
        comp_dict  
    )  
  
    # B  
    prep_dict = {  
        row.get(  
            "DistToHigh20"  
        ): (  
            100  
            if row.get(  
                "DistToHigh20",  
                10,  
            )  
            < 3.0  
            else 40,  
            0.25,  
        ),  
        row.get(  
            "HigherLow5"  
        ): (  
            100  
            if bool(  
                row.get(  
                    "HigherLow5",  
                    False,  
                )  
            )  
            else 20,  
            0.25,  
        ),  
        row.get(  
            "HigherHigh5"  
        ): (  
            100  
            if bool(  
                row.get(  
                    "HigherHigh5",  
                    False,  
                )  
            )  
            else 30,  
            0.2,  
        ),  
        row.get(  
            "CloseLocation"  
        ): (  
            100  
            if row.get(  
                "CloseLocation",  
                0.5,  
            )  
            > 0.7  
            else 40,  
            0.3,  
        ),  
    }  
  
    score_B = eval_score(  
        prep_dict  
    )  
  
    # C  
    flow_dict = {  
        row.get(  
            "VolumeRatio20"  
        ): (  
            100  
            if row.get(  
                "VolumeRatio20",  
                1,  
            )  
            >= 1.5  
            else 40,  
            0.25,  
        ),  
        row.get(  
            "AmountRatio20"  
        ): (  
            100  
            if row.get(  
                "AmountRatio20",  
                1,  
            )  
            >= 1.5  
            else 40,  
            0.25,  
        ),  
        row.get(  
            "OBV_Slope5"  
        ): (  
            100  
            if row.get(  
                "OBV_Slope5",  
                0,  
            )  
            > 0  
            else 30,  
            0.25,  
        ),  
        row.get(  
            "CMF20"  
        ): (  
            100  
            if row.get(  
                "CMF20",  
                0,  
            )  
            > 0.05  
            else 30,  
            0.25,  
        ),  
    }  
  
    score_C = eval_score(  
        flow_dict  
    )  
  
    # D  
    risk_dict = {  
        row.get(  
            "Return1"  
        ): (  
            100  
            if row.get(  
                "Return1",  
                0,  
            )  
            < 5.0  
            else 30,  
            0.25,  
        ),  
        row.get(  
            "Return5"  
        ): (  
            100  
            if row.get(  
                "Return5",  
                0,  
            )  
            < 12.0  
            else 30,  
            0.25,  
        ),  
        row.get(  
            "DisparityEMA20"  
        ): (  
            100  
            if row.get(  
                "DisparityEMA20",  
                0,  
            )  
            < 8.0  
            else 20,  
            0.25,  
        ),  
        row.get(  
            "RSI14"  
        ): (  
            100  
            if row.get(  
                "RSI14",  
                50,  
            )  
            < 68.0  
            else 20,  
            0.25,  
        ),  
    }  
  
    score_D = eval_score(  
        risk_dict  
    )  
  
    if any(  
        pd.isna(x)  
        for x in [  
            score_A,  
            score_B,  
            score_C,  
            score_D,  
        ]  
    ):  
        return (  
            np.nan,  
            np.nan,  
            np.nan,  
            np.nan,  
            np.nan,  
        )  
  
    total_score = (  
        score_A * 0.25  
        + score_B * 0.30  
        + score_C * 0.25  
        + score_D * 0.20  
    )  
  
    return (  
        total_score,  
        score_A,  
        score_B,  
        score_C,  
        score_D,  
    )  
  
  
def classify_surge_pattern(row):  
    """상승 전조 유형 분류"""  
  
    if (  
        row.get(  
            "BoxWidthRatio5_20",  
            1,  
        )  
        < 0.6  
        and row.get(  
            "DistToHigh20",  
            10,  
        )  
        < 4  
    ):  
        return "압축 후 상단 접근"  
  
    elif row.get(  
        "AmountRatio20",  
        1,  
    ) >= 2.0:  
        return "거래대금 증가형"  
  
    elif (  
        bool(  
            row.get(  
                "HigherLow5",  
                False,  
            )  
        )  
        and row.get(  
            "LowSlope5",  
            0,  
        )  
        > 1.5  
    ):  
        return "저점 상승형"  
  
    elif row.get(  
        "DistToHigh20",  
        10,  
    ) < 2.0:  
        return "돌파 준비형"  
  
    elif (  
        row.get(  
            "Return5",  
            0,  
        )  
        < 0  
        and row.get(  
            "DisparityEMA20",  
            0,  
        )  
        > 0  
    ):  
        return "눌림 후 재상승 준비형"  
  
    elif (  
        row.get(  
            "Close",  
            0,  
        )  
        > row.get(  
            "EMA20",  
            0,  
        )  
        and row.get(  
            "EMA20",  
            0,  
        )  
        > row.get(  
            "EMA60",  
            0,  
        )  
    ):  
        return "추세 유지형"  
  
    return "관찰형"  
  
  
# ============================================================  
# 5. 필수 게이트  
# ============================================================  
def evaluate_essential_gates(  
    row,  
    settings,  
    data_last_date,  
    ref_date,  
):  
    """필수 게이트 필터링"""  
  
    reasons = []  
  
    min_amount_krw = (  
        settings["min_amount_100m"]  
        * 100_000_000  
    )  
  
    amount_ma20 = row.get(  
        "AmountMA20",  
        np.nan,  
    )  
  
    # FIX:  
    # 거래대금 결측은 0으로 보지 않고 분석 불가.  
    if pd.isna(amount_ma20):  
        reasons.append(  
            "20일 평균 거래대금 계산 불가"  
        )  
    elif amount_ma20 < min_amount_krw:  
        reasons.append(  
            "최소 거래대금 미달"  
        )  
  
    if data_last_date != ref_date:  
        reasons.append(  
            "데이터 기준일 불일치"  
        )  
  
    today_return = row.get(  
        "Return1",  
        np.nan,  
    )  
  
    if pd.isna(today_return):  
        reasons.append(  
            "당일 수익률 계산 불가"  
        )  
    elif today_return > settings[  
        "max_today_return"  
    ]:  
        reasons.append(  
            "당일 급등 과열 제외"  
        )  
  
    close = row.get(  
        "Close",  
        np.nan,  
    )  
  
    if pd.isna(close) or close <= 0:  
        reasons.append(  
            "비정상 종가 데이터"  
        )  
  
    return (  
        len(reasons) == 0,  
        reasons,  
    )  
  
  
# ============================================================  
# FIX: 실제 다음 거래일 반환  
# ============================================================  
def get_next_trading_date(  
    df,  
    current_date,  
):  
    """  
    BDay(1)을 사용하지 않고 해당 종목 DataFrame의  
    실제 다음 거래일을 반환.  
    """  
  
    if df is None or df.empty:  
        return None  
  
    try:  
        idx = df.index  
  
        loc = idx.get_loc(  
            current_date  
        )  
  
        if isinstance(  
            loc,  
            slice,  
        ):  
            return None  
  
        next_pos = loc + 1  
  
        if next_pos >= len(idx):  
            return None  
  
        return idx[next_pos]  
  
    except Exception:  
        return None  
  
  
# ============================================================  
# 6. V9 백테스트 엔진  
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
    t일 종가 신호  
    ->  
    해당 종목의 실제 다음 거래일 시가 체결  
  
    청산 우선순위:  
    1. 손절  
    2. 2차 목표  
    3. 1차 부분익절  
    4. EMA20 추세 이탈  
    5. 최대 보유기간  
    """  
  
    all_dates = sorted(  
        list(  
            set(  
                d  
                for df in universe_dict.values()  
                for d in df.index  
            )  
        )  
    )  
  
    if len(all_dates) < 2:  
        return (  
            None,  
            None,  
            None,  
            {},  
        )  
  
    cash = float(  
        initial_capital  
    )  
  
    reserved_cash = 0.0  
  
    pending_orders = []  
    positions = {}  
  
    closed_trades = []  
    trade_logs = []  
    equity_curve = []  
  
    for i, current_date in enumerate(  
        all_dates  
    ):  
  
        is_last_global_day = (  
            i == len(all_dates) - 1  
        )  
  
        # ====================================================  
        # A. 예약 주문 처리  
        # ====================================================  
        if pending_orders:  
  
            pending_orders.sort(  
                key=lambda x: x["score"],  
                reverse=True,  
            )  
  
            new_pending = []  
  
            available_slots = max(  
                0,  
                max_concurrent  
                - len(positions),  
            )  
  
            for order in pending_orders:  
  
                sym = order["sym"]  
  
                df = universe_dict.get(  
                    sym  
                )  
  
                reservation = float(  
                    order.get(  
                        "reserved_amount",  
                        0.0,  
                    )  
                )  
  
                target_date = order.get(  
                    "target_entry_date"  
                )  
  
                # =================================================  
                # FIX:  
                # 주문 대상 종목의 실제 다음 거래일이 아니면  
                # 아직 보관한다.  
                # =================================================  
                if (  
                    target_date is not None  
                    and current_date  
                    < target_date  
                ):  
                    new_pending.append(  
                        order  
                    )  
                    continue  
  
                # 데이터 자체가 없으면 예약금 즉시 반환  
                if (  
                    df is None  
                    or df.empty  
                    or current_date  
                    not in df.index  
                ):  
  
                    reserved_cash = max(  
                        0.0,  
                        reserved_cash  
                        - reservation,  
                    )  
  
                    trade_logs.append(  
                        {  
                            "Date": current_date,  
                            "Symbol": sym,  
                            "Type": "ORDER_CANCEL",  
                            "Reason": (  
                                "종목 데이터 오류/없음"  
                            ),  
                            "Reserved_Returned": reservation,  
                        }  
                    )  
  
                    continue  
  
                # 목표 날짜를 지나갔으면 취소  
                if (  
                    target_date is not None  
                    and current_date  
                    > target_date  
                ):  
  
                    reserved_cash = max(  
                        0.0,  
                        reserved_cash  
                        - reservation,  
                    )  
  
                    trade_logs.append(  
                        {  
                            "Date": current_date,  
                            "Symbol": sym,  
                            "Type": "ORDER_CANCEL",  
                            "Reason": (  
                                "실제 다음 거래일 체결기회 경과"  
                            ),  
                            "Reserved_Returned": reservation,  
                        }  
                    )  
  
                    continue  
  
                # =================================================  
                # FIX:  
                # 주문 처리 시 자신의 예약금을 먼저 해제한다.  
                # =================================================  
                reserved_cash = max(  
                    0.0,  
                    reserved_cash  
                    - reservation,  
                )  
  
                # 슬롯 부족  
                if available_slots <= 0:  
  
                    trade_logs.append(  
                        {  
                            "Date": current_date,  
                            "Symbol": sym,  
                            "Type": "ORDER_CANCEL",  
                            "Reason": (  
                                "동시보유 슬롯 초과"  
                            ),  
                            "Reserved_Returned": reservation,  
                        }  
                    )  
  
                    continue  
  
                row = df.loc[  
                    current_date  
                ]  
  
                open_px = pd.to_numeric(  
                    row["Open"],  
                    errors="coerce",  
                )  
  
                if (  
                    pd.isna(open_px)  
                    or open_px <= 0  
                ):  
  
                    trade_logs.append(  
                        {  
                            "Date": current_date,  
                            "Symbol": sym,  
                            "Type": "ORDER_FAIL",  
                            "Reason": (  
                                "비정상 시가"  
                            ),  
                            "Reserved_Returned": reservation,  
                        }  
                    )  
  
                    continue  
  
                exec_price = float(  
                    open_px  
                ) * (  
                    1 + slippage  
                )  
  
                if (  
                    not np.isfinite(  
                        exec_price  
                    )  
                    or exec_price <= 0  
                ):  
  
                    trade_logs.append(  
                        {  
                            "Date": current_date,  
                            "Symbol": sym,  
                            "Type": "ORDER_FAIL",  
                            "Reason": (  
                                "비정상 체결가격"  
                            ),  
                            "Reserved_Returned": reservation,  
                        }  
                    )  
  
                    continue  
  
                # =================================================  
                # FIX:  
                # 예약금은 전체 현금이 아니라 실제 사용 가능한  
                # 현금과 비교한다.  
                # =================================================  
                available_cash = max(  
                    0.0,  
                    cash  
                    - reserved_cash,  
                )  
  
                alloc_cash = min(  
                    reservation,  
                    available_cash,  
                )  
  
                qty = int(  
                    (  
                        alloc_cash  
                        * 0.98  
                    )  
                    / exec_price  
                )  
  
                if qty <= 0:  
  
                    trade_logs.append(  
                        {  
                            "Date": current_date,  
                            "Symbol": sym,  
                            "Type": "ORDER_FAIL",  
                            "Reason": (  
                                "주문가능 수량 0"  
                            ),  
                            "Reserved_Returned": reservation,  
                        }  
                    )  
  
                    continue  
  
                buy_gross = (  
                    qty  
                    * exec_price  
                )  
  
                buy_fee = (  
                    buy_gross  
                    * fee_rate  
                )  
  
                total_cost = (  
                    buy_gross  
                    + buy_fee  
                )  
  
                if (  
                    total_cost  
                    > available_cash  
                    + 1e-9  
                ):  
  
                    trade_logs.append(  
                        {  
                            "Date": current_date,  
                            "Symbol": sym,  
                            "Type": "ORDER_FAIL",  
                            "Reason": (  
                                "예약금 해제 후 실제 가용 현금 부족"  
                            ),  
                            "Reserved_Returned": reservation,  
                        }  
                    )  
  
                    continue  
  
                cash -= total_cost  
  
                available_slots -= 1  
  
                pid = str(  
                    uuid.uuid4()  
                )[:8]  
  
                stop_loss = float(  
                    order["stop_loss"]  
                )  
  
                tp1 = float(  
                    order["tp1"]  
                )  
  
                tp2 = float(  
                    order["tp2"]  
                )  
  
                atr_value = float(  
                    order["atr"]  
                )  
  
                # =================================================  
                # FIX:  
                # 손절/ATR 값 자체가 비정상인 경우 포지션 생성 금지  
                # =================================================  
                if (  
                    not np.isfinite(  
                        stop_loss  
                    )  
                    or not np.isfinite(  
                        tp1  
                    )  
                    or not np.isfinite(  
                        tp2  
                    )  
                    or not np.isfinite(  
                        atr_value  
                    )  
                    or stop_loss <= 0  
                    or tp1 <= 0  
                    or tp2 <= 0  
                    or atr_value <= 0  
                ):  
  
                    cash += total_cost  
  
                    trade_logs.append(  
                        {  
                            "Date": current_date,  
                            "Symbol": sym,  
                            "Type": "ORDER_FAIL",  
                            "Reason": (  
                                "손절/목표가/ATR 비정상"  
                            ),  
                            "Reserved_Returned": reservation,  
                        }  
                    )  
  
                    continue  
  
                positions[pid] = {  
                    "pid": pid,  
                    "sym": sym,  
                    "entry_date": current_date,  
                    "signal_date": order[  
                        "signal_date"  
                    ],  
                    "entry_price": exec_price,  
                    "qty": qty,  
                    "initial_qty": qty,  
                    "buy_fee": buy_fee,  
                    "atr": atr_value,  
                    "stop_loss": stop_loss,  
                    "tp1": tp1,  
                    "tp2": tp2,  
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
                        "Reason": (  
                            "t일 종가 신호 -> "  
                            "해당 종목 실제 다음 거래일 시가 체결"  
                        ),  
                        "Signal_Date": order[  
                            "signal_date"  
                        ],  
                        "Target_Entry_Date": target_date,  
                        "Reserved_Amount": reservation,  
                    }  
                )  
  
            pending_orders = new_pending  
  
        # ====================================================  
        # B. 포지션 청산  
        # ====================================================  
        for pid in list(  
            positions.keys()  
        ):  
  
            pos = positions[pid]  
  
            df = universe_dict.get(  
                pos["sym"]  
            )  
  
            if (  
                df is None  
                or current_date  
                not in df.index  
            ):  
                trade_logs.append(  
                    {  
                        "Date": current_date,  
                        "Symbol": pos["sym"],  
                        "Type": "POSITION_DATA_ERROR",  
                        "Reason": (  
                            "포지션 평가용 데이터 없음"  
                        ),  
                    }  
                )  
                continue  
  
            row = df.loc[  
                current_date  
            ]  
  
            if (  
                current_date  
                != pos["entry_date"]  
            ):  
                pos[  
                    "holding_days"  
                ] += 1  
  
            open_px = pd.to_numeric(  
                row["Open"],  
                errors="coerce",  
            )  
  
            low_px = pd.to_numeric(  
                row["Low"],  
                errors="coerce",  
            )  
  
            high_px = pd.to_numeric(  
                row["High"],  
                errors="coerce",  
            )  
  
            close_px = pd.to_numeric(  
                row["Close"],  
                errors="coerce",  
            )  
  
            if (  
                pd.isna(open_px)  
                or pd.isna(low_px)  
                or pd.isna(high_px)  
                or pd.isna(close_px)  
                or open_px <= 0  
                or low_px <= 0  
                or high_px <= 0  
                or close_px <= 0  
            ):  
                trade_logs.append(  
                    {  
                        "Date": current_date,  
                        "Symbol": pos["sym"],  
                        "Type": "POSITION_DATA_ERROR",  
                        "Reason": (  
                            "OHLC 데이터 비정상"  
                        ),  
                    }  
                )  
                continue  
  
            exit_reason = None  
            exit_price_before_slippage = None  
            sell_qty = 0  
  
            stop_loss = pos[  
                "stop_loss"  
            ]  
  
            tp1 = pos["tp1"]  
            tp2 = pos["tp2"]  
  
            # =================================================  
            # FIX:  
            # 손절 갭 하락 처리.  
            #  
            # Open <= Stop:  
            #     실제 시가 체결  
            #  
            # Open > Stop && Low <= Stop:  
            #     손절가 체결  
            # =================================================  
            if open_px <= stop_loss:  
  
                exit_reason = (  
                    "1. 손절 - 갭 하락 시가 체결"  
                )  
  
                exit_price_before_slippage = float(  
                    open_px  
                )  
  
                sell_qty = pos["qty"]  
  
            elif low_px <= stop_loss:  
  
                exit_reason = (  
                    "1. 손절 - 손절가 체결"  
                )  
  
                exit_price_before_slippage = float(  
                    stop_loss  
                )  
  
                sell_qty = pos["qty"]  
  
            elif high_px >= tp2:  
  
                exit_reason = (  
                    "2. 2차 목표가"  
                )  
  
                exit_price_before_slippage = float(  
                    tp2  
                )  
  
                sell_qty = pos["qty"]  
  
            elif (  
                high_px >= tp1  
                and not pos[  
                    "partial_done"  
                ]  
            ):  
  
                exit_reason = (  
                    "3. 1차 목표가 "  
                    "(부분익절)"  
                )  
  
                exit_price_before_slippage = float(  
                    tp1  
                )  
  
                sell_qty = max(  
                    1,  
                    int(  
                        pos[  
                            "initial_qty"  
                        ]  
                        * 0.5  
                    ),  
                )  
  
                sell_qty = min(  
                    sell_qty,  
                    pos["qty"],  
                )  
  
            elif (  
                close_px  
                < row.get(  
                    "EMA20",  
                    np.inf,  
                )  
                and row.get(  
                    "EMA20_Slope",  
                    0,  
                )  
                < 0  
            ):  
  
                exit_reason = (  
                    "4. EMA20 추세 이탈"  
                )  
  
                exit_price_before_slippage = float(  
                    close_px  
                )  
  
                sell_qty = pos["qty"]  
  
            elif (  
                pos["holding_days"]  
                >= settings[  
                    "max_holding_days"  
                ]  
            ):  
  
                exit_reason = (  
                    "5. 최대 보유기간 만료"  
                )  
  
                exit_price_before_slippage = float(  
                    close_px  
                )  
  
                sell_qty = pos["qty"]  
  
            if (  
                exit_reason  
                and sell_qty > 0  
            ):  
  
                if (  
                    exit_price_before_slippage  
                    is None  
                    or not np.isfinite(  
                        exit_price_before_slippage  
                    )  
                    or exit_price_before_slippage  
                    <= 0  
                ):  
                    trade_logs.append(  
                        {  
                            "Date": current_date,  
                            "Symbol": pos["sym"],  
                            "Type": "SELL_FAIL",  
                            "Reason": (  
                                "비정상 손절/청산 가격"  
                            ),  
                        }  
                    )  
                    continue  
  
                # 매도 슬리피지 적용  
                exit_price = (  
                    exit_price_before_slippage  
                    * (1 - slippage)  
                )  
  
                if (  
                    not np.isfinite(  
                        exit_price  
                    )  
                    or exit_price <= 0  
                ):  
                    trade_logs.append(  
                        {  
                            "Date": current_date,  
                            "Symbol": pos["sym"],  
                            "Type": "SELL_FAIL",  
                            "Reason": (  
                                "슬리피지 적용 후 가격 비정상"  
                            ),  
                        }  
                    )  
                    continue  
  
                sell_gross = (  
                    sell_qty  
                    * exit_price  
                )  
  
                sell_fee = (  
                    sell_gross  
                    * fee_rate  
                )  
  
                sell_tax = (  
                    sell_gross  
                    * tax_rate  
                )  
  
                buy_fee_alloc = (  
                    pos["buy_fee"]  
                    * (  
                        sell_qty  
                        / pos[  
                            "initial_qty"  
                        ]  
                    )  
                )  
  
                net_pnl = (  
                    (  
                        exit_price  
                        - pos[  
                            "entry_price"  
                        ]  
                    )  
                    * sell_qty  
                    - buy_fee_alloc  
                    - sell_fee  
                    - sell_tax  
                )  
  
                cash += (  
                    sell_gross  
                    - sell_fee  
                    - sell_tax  
                )  
  
                pos[  
                    "realized_pnl"  
                ] += net_pnl  
  
                trade_logs.append(  
                    {  
                        "Date": current_date,  
                        "Symbol": pos["sym"],  
                        "Type": "SELL",  
                        "Price": exit_price,  
                        "Raw_Trigger_Price": exit_price_before_slippage,  
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
                            "Entry_Date": pos[  
                                "entry_date"  
                            ],  
                            "Exit_Date": current_date,  
                            "Holding_Days": pos[  
                                "holding_days"  
                            ],  
                            "Entry_Price": pos[  
                                "entry_price"  
                            ],  
                            "Exit_Price": exit_price,  
                            "Net_PnL": pos[  
                                "realized_pnl"  
                            ],  
                            "Return_Pct": (  
                                pos[  
                                    "realized_pnl"  
                                ]  
                                / (  
                                    pos[  
                                        "entry_price"  
                                    ]  
                                    * pos[  
                                        "initial_qty"  
                                    ]  
                                )  
                                * 100  
                            ),  
                            "Exit_Reason": exit_reason,  
                        }  
                    )  
  
                    del positions[pid]  
  
                else:  
                    pos["qty"] -= sell_qty  
                    pos[  
                        "partial_done"  
                    ] = True  
  
        # ====================================================  
        # C. t일 종가 신호  
        # ====================================================  
        #  
        # FIX:  
        # 해당 종목의 실제 다음 거래일이 존재하는 경우에만  
        # 예약 주문 생성.  
        # ====================================================  
        if not is_last_global_day:  
  
            active_symbols = {  
                p["sym"]  
                for p in positions.values()  
            }  
  
            pending_symbols = {  
                p["sym"]  
                for p in pending_orders  
            }  
  
            candidates = []  
  
            for sym, df in universe_dict.items():  
  
                if (  
                    sym in active_symbols  
                    or sym in pending_symbols  
                    or current_date  
                    not in df.index  
                ):  
                    continue  
  
                next_trade_date = (  
                    get_next_trading_date(  
                        df,  
                        current_date,  
                    )  
                )  
  
                # FIX:  
                # 다음 거래일이 없는 종목은 신호를 만들지 않는다.  
                if next_trade_date is None:  
                    continue  
  
                row = df.loc[  
                    current_date  
                ]  
  
                passed, _ = (  
                    evaluate_essential_gates(  
                        row,  
                        settings,  
                        current_date,  
                        current_date,  
                    )  
                )  
  
                (  
                    tot_score,  
                    _,  
                    _,  
                    _,  
                    _,  
                ) = calculate_precursor_subscores(  
                    row,  
                    settings[  
                        "missing_handling"  
                    ],  
                )  
  
                if (  
                    passed  
                    and pd.notna(  
                        tot_score  
                    )  
                    and tot_score  
                    >= settings[  
                        "min_score"  
                    ]  
                ):  
                    candidates.append(  
                        (  
                            tot_score,  
                            sym,  
                            row,  
                            next_trade_date,  
                        )  
                    )  
  
            candidates.sort(  
                key=lambda x: x[0],  
                reverse=True,  
            )  
  
            free_slots = max(  
                0,  
                max_concurrent  
                - len(positions)  
                - len(pending_orders),  
            )  
  
            free_cash = max(  
                0.0,  
                cash  
                - reserved_cash,  
            )  
  
            if (  
                free_slots > 0  
                and free_cash > 0  
            ):  
  
                per_order_cash = (  
                    free_cash  
                    / free_slots  
                )  
  
                for (  
                    score_val,  
                    sym,  
                    row,  
                    next_trade_date,  
                ) in candidates[  
                    :free_slots  
                ]:  
  
                    atr = pd.to_numeric(  
                        row.get(  
                            "ATR14"  
                        ),  
                        errors="coerce",  
                    )  
  
                    close_px = pd.to_numeric(  
                        row.get(  
                            "Close"  
                        ),  
                        errors="coerce",  
                    )  
  
                    if (  
                        pd.isna(atr)  
                        or pd.isna(  
                            close_px  
                        )  
                        or atr <= 0  
                        or close_px <= 0  
                    ):  
                        continue  
  
                    stop_px = (  
                        close_px  
                        - 2.0 * atr  
                    )  
  
                    risk = (  
                        close_px  
                        - stop_px  
                    )  
  
                    tp1_px = (  
                        close_px  
                        + 2.0 * risk  
                    )  
  
                    tp2_px = (  
                        close_px  
                        + 3.5 * risk  
                    )  
  
                    if (  
                        stop_px <= 0  
                        or risk <= 0  
                        or tp1_px <= 0  
                        or tp2_px <= 0  
                    ):  
                        continue  
  
                    reserve_amt = min(  
                        per_order_cash,  
                        max(  
                            0.0,  
                            cash  
                            - reserved_cash,  
                        ),  
                    )  
  
                    if reserve_amt <= 0:  
                        break  
  
                    pending_orders.append(  
                        {  
                            "sym": sym,  
                            "score": score_val,  
                            "signal_date": current_date,  
                            "target_entry_date": next_trade_date,  
                            "atr": float(atr),  
                            "stop_loss": float(  
                                stop_px  
                            ),  
                            "tp1": float(  
                                tp1_px  
                            ),  
                            "tp2": float(  
                                tp2_px  
                            ),  
                            "reserved_amount": float(  
                                reserve_amt  
                            ),  
                        }  
                    )  
  
                    reserved_cash += (  
                        reserve_amt  
                    )  
  
        # ====================================================  
        # D. 자산 평가  
        # ====================================================  
        stock_eval = 0.0  
  
        for pos in positions.values():  
  
            df = universe_dict.get(  
                pos["sym"]  
            )  
  
            if (  
                df is not None  
                and current_date  
                in df.index  
            ):  
  
                close_value = pd.to_numeric(  
                    df.loc[  
                        current_date,  
                        "Close",  
                    ],  
                    errors="coerce",  
                )  
  
                if (  
                    pd.notna(  
                        close_value  
                    )  
                    and close_value > 0  
                ):  
                    stock_eval += (  
                        pos["qty"]  
                        * float(  
                            close_value  
                        )  
                    )  
  
        equity_curve.append(  
            {  
                "Date": current_date,  
                "Cash": cash,  
                "ReservedCash": reserved_cash,  
                "AvailableCash": max(  
                    0.0,  
                    cash  
                    - reserved_cash,  
                ),  
                "StockValue": stock_eval,  
                "TotalAsset": (  
                    cash  
                    + stock_eval  
                ),  
                "OpenPositions": len(  
                    positions  
                ),  
                "PendingOrders": len(  
                    pending_orders  
                ),  
            }  
        )  
  
    # ========================================================  
    # FIX:  
    # 백테스트 마지막 날 미체결 예약 주문 취소.  
    # 예약금은 반드시 반환.  
    # ========================================================  
    if pending_orders:  
  
        last_date = (  
            all_dates[-1]  
            if all_dates  
            else pd.Timestamp.now()  
        )  
  
        for order in pending_orders:  
  
            reservation = float(  
                order.get(  
                    "reserved_amount",  
                    0.0,  
                )  
            )  
  
            reserved_cash = max(  
                0.0,  
                reserved_cash  
                - reservation,  
            )  
  
            trade_logs.append(  
                {  
                    "Date": last_date,  
                    "Symbol": order["sym"],  
                    "Type": "ORDER_CANCEL",  
                    "Reason": (  
                        "백테스트 종료일 미체결 예약 주문"  
                    ),  
                    "Reserved_Returned": reservation,  
                }  
            )  
  
        pending_orders = []  
  
    # ========================================================  
    # FIX:  
    # 미청산 포지션은 closed_trades에 넣지 않는다.  
    # 별도 평가 대상으로 유지한다.  
    # ========================================================  
  
    return (  
        pd.DataFrame(  
            equity_curve  
        ),  
        pd.DataFrame(  
            closed_trades  
        ),  
        pd.DataFrame(  
            trade_logs  
        ),  
        positions,  
    )  
  
  
# ============================================================  
# FIX: 실제 미청산 포지션 평가표  
# ============================================================  
def build_open_positions_df(  
    positions,  
    universe_dict,  
    final_date,  
):  
    rows = []  
  
    for pid, pos in positions.items():  
  
        df = universe_dict.get(  
            pos["sym"]  
        )  
  
        if (  
            df is None  
            or df.empty  
        ):  
            continue  
  
        available_dates = df.index[  
            df.index <= final_date  
        ]  
  
        if len(  
            available_dates  
        ) == 0:  
            continue  
  
        valuation_date = (  
            available_dates[-1]  
        )  
  
        close_px = pd.to_numeric(  
            df.loc[  
                valuation_date,  
                "Close",  
            ],  
            errors="coerce",  
        )  
  
        if (  
            pd.isna(close_px)  
            or close_px <= 0  
        ):  
            continue  
  
        market_value = (  
            pos["qty"]  
            * float(close_px)  
        )  
  
        unrealized = (  
            (  
                float(close_px)  
                - pos[  
                    "entry_price"  
                ]  
            )  
            * pos["qty"]  
        )  
  
        rows.append(  
            {  
                "Position_ID": pid,  
                "Symbol": pos["sym"],  
                "Entry_Date": pos[  
                    "entry_date"  
                ],  
                "평가일": valuation_date,  
                "보유수량": pos["qty"],  
                "진입가격": pos[  
                    "entry_price"  
                ],  
                "현재가격": float(  
                    close_px  
                ),  
                "평가금액": market_value,  
                "미실현손익": unrealized,  
                "실현손익": pos[  
                    "realized_pnl"  
                ],  
                "총손익": (  
                    unrealized  
                    + pos[  
                        "realized_pnl"  
                    ]  
                ),  
                "손절가": pos[  
                    "stop_loss"  
                ],  
                "1차목표": pos[  
                    "tp1"  
                ],  
                "2차목표": pos[  
                    "tp2"  
                ],  
                "보유일수": pos[  
                    "holding_days"  
                ],  
            }  
        )  
  
    return pd.DataFrame(  
        rows  
    )  
  
  
# ============================================================  
# 7. 워크포워드 검증  
# ============================================================  
def run_walk_forward_validation(  
    universe_dict,  
    settings,  
    train_years=1,  
    test_years=1,  
):  
    """  
    학습 구간에서 최소점수를 선택하고,  
    이후 검증 구간에 동일 기준 적용.  
    """  
  
    all_dates = sorted(  
        list(  
            set(  
                d  
                for df in universe_dict.values()  
                for d in df.index  
            )  
        )  
    )  
  
    if not all_dates:  
        return pd.DataFrame()  
  
    min_year = all_dates[0].year  
    max_year = all_dates[-1].year  
  
    # ========================================================  
    # FIX:  
    # Train + Test가 실제로 모두 존재하는 구간만 순회.  
    # ========================================================  
    last_start_year = (  
        max_year  
        - train_years  
        - test_years  
        + 1  
    )  
  
    results = []  
  
    for start_y in range(  
        min_year,  
        last_start_year + 1,  
    ):  
  
        train_start = pd.Timestamp(  
            f"{start_y}-01-01"  
        )  
  
        train_end = pd.Timestamp(  
            f"{start_y + train_years - 1}-12-31"  
        )  
  
        test_start = pd.Timestamp(  
            f"{start_y + train_years}-01-01"  
        )  
  
        test_end = pd.Timestamp(  
            f"{start_y + train_years + test_years - 1}-12-31"  
        )  
  
        # ====================================================  
        # 1. Train  
        # ====================================================  
        train_signals = []  
  
        for sym, df in universe_dict.items():  
  
            sub = df[  
                (df.index >= train_start)  
                & (df.index <= train_end)  
            ]  
  
            for dt, row in sub.iterrows():  
  
                passed, _ = (  
                    evaluate_essential_gates(  
                        row,  
                        settings,  
                        dt,  
                        dt,  
                    )  
                )  
  
                if not passed:  
                    continue  
  
                score, _, _, _, _ = (  
                    calculate_precursor_subscores(  
                        row,  
                        settings[  
                            "missing_handling"  
                        ],  
                    )  
                )  
  
                if pd.isna(score):  
                    continue  
  
                next_date = (  
                    get_next_trading_date(  
                        df,  
                        dt,  
                    )  
                )  
  
                if next_date is None:  
                    continue  
  
                next_open = pd.to_numeric(  
                    df.loc[  
                        next_date,  
                        "Open",  
                    ],  
                    errors="coerce",  
                )  
  
                close = pd.to_numeric(  
                    row["Close"],  
                    errors="coerce",  
                )  
  
                if (  
                    pd.isna(  
                        next_open  
                    )  
                    or pd.isna(close)  
                    or close <= 0  
                    or next_open <= 0  
                ):  
                    continue  
  
                next_ret = (  
                    next_open  
                    / close  
                    - 1  
                ) * 100  
  
                train_signals.append(  
                    {  
                        "Score": score,  
                        "NextRet": next_ret,  
                    }  
                )  
  
        train_df = pd.DataFrame(  
            train_signals  
        )  
  
        if train_df.empty:  
            continue  
  
        # ====================================================  
        # 2. Train에서 threshold 선택  
        # ====================================================  
        best_min_score = 65  
        best_avg_ret = -np.inf  
  
        for cand_score in [  
            60,  
            65,  
            70,  
            75,  
            80,  
        ]:  
  
            filtered = train_df[  
                train_df["Score"]  
                >= cand_score  
            ]  
  
            if len(  
                filtered  
            ) < 10:  
                continue  
  
            avg_r = filtered[  
                "NextRet"  
            ].mean()  
  
            if avg_r > best_avg_ret:  
                best_avg_ret = avg_r  
                best_min_score = (  
                    cand_score  
                )  
  
        # ====================================================  
        # 3. Test  
        # ====================================================  
        test_signals = []  
  
        for sym, df in universe_dict.items():  
  
            sub = df[  
                (df.index >= test_start)  
                & (df.index <= test_end)  
            ]  
  
            for dt, row in sub.iterrows():  
  
                passed, _ = (  
                    evaluate_essential_gates(  
                        row,  
                        settings,  
                        dt,  
                        dt,  
                    )  
                )  
  
                if not passed:  
                    continue  
  
                score, _, _, _, _ = (  
                    calculate_precursor_subscores(  
                        row,  
                        settings[  
                            "missing_handling"  
                        ],  
                    )  
                )  
  
                if (  
                    pd.isna(score)  
                    or score  
                    < best_min_score  
                ):  
                    continue  
  
                next_date = (  
                    get_next_trading_date(  
                        df,  
                        dt,  
                    )  
                )  
  
                if next_date is None:  
                    continue  
  
                next_open = pd.to_numeric(  
                    df.loc[  
                        next_date,  
                        "Open",  
                    ],  
                    errors="coerce",  
                )  
  
                close = pd.to_numeric(  
                    row["Close"],  
                    errors="coerce",  
                )  
  
                if (  
                    pd.isna(  
                        next_open  
                    )  
                    or pd.isna(close)  
                    or close <= 0  
                    or next_open <= 0  
                ):  
                    continue  
  
                next_ret = (  
                    next_open  
                    / close  
                    - 1  
                ) * 100  
  
                test_signals.append(  
                    {  
                        "Score": score,  
                        "NextRet": next_ret,  
                    }  
                )  
  
        test_df = pd.DataFrame(  
            test_signals  
        )  
  
        results.append(  
            {  
                "학습기간": (  
                    f"{train_start.date()}"  
                    f"~{train_end.date()}"  
                ),  
                "검증기간": (  
                    f"{test_start.date()}"  
                    f"~{test_end.date()}"  
                ),  
                "선택된 최소점수": best_min_score,  
                "학습 신호수": len(  
                    train_df  
                ),  
                "검증 신호수": len(  
                    test_df  
                ),  
                "검증 평균수익률(%)": (  
                    test_df[  
                        "NextRet"  
                    ].mean()  
                    if not test_df.empty  
                    else np.nan  
                ),  
                "검증 승률(%)": (  
                    (  
                        test_df[  
                            "NextRet"  
                        ]  
                        > 0  
                    ).mean()  
                    * 100  
                    if not test_df.empty  
                    else np.nan  
                ),  
            }  
        )  
  
    return pd.DataFrame(  
        results  
    )  
  
  
# ============================================================  
# FIX: 전략 비교를 실제 데이터로 계산  
# ============================================================  
def run_strategy_comparison(  
    universe_dict,  
    settings,  
):  
    """  
    기존 A~H 구조를 유지하되,  
    임의 성과 숫자를 사용하지 않고 실제 과거 데이터로 계산.  
  
    수익률 기준:  
    - 1일후: 신호일 종가 -> 다음 거래일 시가  
    - 5일후: 다음 거래일 시가 -> 그 후 5번째 거래일 종가  
    - 20일후: 다음 거래일 시가 -> 그 후 20번째 거래일 종가  
    """  
  
    strategies = [  
        "A. 전체 대상 종목",  
        "B. 박스권 조건",  
        "C. 박스권 + 변동성 압축",  
        "D. C + 거래대금 증가",  
        "E. D + 상대강도",  
        "F. E + 오늘 급등 제외",  
        "G. F + OBV·CMF",  
        "H. 최종 가격 전조 전략",  
    ]  
  
    buckets = {  
        name: []  
        for name in strategies  
    }  
  
    for sym, df in universe_dict.items():  
  
        for i in range(  
            len(df)  
        ):  
  
            if i >= len(df) - 1:  
                continue  
  
            dt = df.index[i]  
            row = df.iloc[i]  
  
            close = pd.to_numeric(  
                row.get("Close"),  
                errors="coerce",  
            )  
  
            next_open = pd.to_numeric(  
                df.iloc[i + 1].get(  
                    "Open"  
                ),  
                errors="coerce",  
            )  
  
            if (  
                pd.isna(close)  
                or pd.isna(next_open)  
                or close <= 0  
                or next_open <= 0  
            ):  
                continue  
  
            # A  
            conditions = []  
  
            conditions.append(  
                True  
            )  
  
            # B  
            b = (  
                row.get(  
                    "BoxPosition20",  
                    np.nan,  
                )  
                >= 0.60  
                and row.get(  
                    "DistToHigh20",  
                    np.nan,  
                )  
                <= 10  
            )  
  
            # C  
            c = (  
                b  
                and row.get(  
                    "VolRatio5_20",  
                    np.nan,  
                )  
                < 0.8  
                and row.get(  
                    "BoxWidthRatio5_20",  
                    np.nan,  
                )  
                < 0.8  
            )  
  
            # D  
            d = (  
                c  
                and row.get(  
                    "AmountRatio20",  
                    np.nan,  
                )  
                >= 1.5  
            )  
  
            # E  
            e = (  
                d  
                and row.get(  
                    "RelReturn20",  
                    np.nan,  
                )  
                > 0  
            )  
  
            # F  
            f = (  
                e  
                and row.get(  
                    "Return1",  
                    np.nan,  
                )  
                <= settings[  
                    "max_today_return"  
                ]  
            )  
  
            # G  
            g = (  
                f  
                and row.get(  
                    "OBV_Slope5",  
                    np.nan,  
                )  
                > 0  
                and row.get(  
                    "CMF20",  
                    np.nan,  
                )  
                > 0  
            )  
  
            # H  
            score, _, _, _, _ = (  
                calculate_precursor_subscores(  
                    row,  
                    settings[  
                        "missing_handling"  
                    ],  
                )  
            )  
  
            h = (  
                g  
                and pd.notna(score)  
                and score  
                >= settings[  
                    "min_score"  
                ]  
            )  
  
            passes = [  
                True,  
                b,  
                c,  
                d,  
                e,  
                f,  
                g,  
                h,  
            ]  
  
            # 미래 성과 계산  
            next_ret = (  
                next_open  
                / close  
                - 1  
            ) * 100  
  
            ret5 = np.nan  
            ret20 = np.nan  
  
            if i + 5 < len(df):  
                future_close5 = pd.to_numeric(  
                    df.iloc[  
                        i + 5  
                    ]["Close"],  
                    errors="coerce",  
                )  
  
                if (  
                    pd.notna(  
                        future_close5  
                    )  
                    and future_close5 > 0  
                ):  
                    ret5 = (  
                        future_close5  
                        / next_open  
                        - 1  
                    ) * 100  
  
            if i + 20 < len(df):  
                future_close20 = pd.to_numeric(  
                    df.iloc[  
                        i + 20  
                    ]["Close"],  
                    errors="coerce",  
                )  
  
                if (  
                    pd.notna(  
                        future_close20  
                    )  
                    and future_close20 > 0  
                ):  
                    ret20 = (  
                        future_close20  
                        / next_open  
                        - 1  
                    ) * 100  
  
            for idx, passed in enumerate(  
                passes  
            ):  
  
                if passed:  
                    buckets[  
                        strategies[idx]  
                    ].append(  
                        {  
                            "Date": dt,  
                            "Symbol": sym,  
                            "NextRet": next_ret,  
                            "Ret5": ret5,  
                            "Ret20": ret20,  
                        }  
                    )  
  
    rows = []  
  
    for strategy in strategies:  
  
        df_s = pd.DataFrame(  
            buckets[strategy]  
        )  
  
        if df_s.empty:  
            rows.append(  
                {  
                    "전략": strategy,  
                    "신호 수": 0,  
                    "1일후 평균(%)": np.nan,  
                    "5일후 평균(%)": np.nan,  
                    "20일후 평균(%)": np.nan,  
                    "승률(%)": np.nan,  
                }  
            )  
            continue  
  
        rows.append(  
            {  
                "전략": strategy,  
                "신호 수": len(df_s),  
                "1일후 평균(%)": df_s[  
                    "NextRet"  
                ].mean(),  
                "5일후 평균(%)": df_s[  
                    "Ret5"  
                ].mean(),  
                "20일후 평균(%)": df_s[  
                    "Ret20"  
                ].mean(),  
                "승률(%)": (  
                    df_s[  
                        "NextRet"  
                    ]  
                    > 0  
                ).mean()  
                * 100,  
            }  
        )  
  
    return pd.DataFrame(  
        rows  
    )  
  
  
# ============================================================  
# 8. 무결성 감사  
# ============================================================  
def run_integrity_audit():  
    """  
    실제 구현 여부에 맞게 PASS / CHECK / LIMITATION 표시.  
    """  
  
    audit_data = [  
        (  
            "미래정보 누수 방지",  
            "PASS",  
            "신호는 t일 데이터만 사용하고 체결은 이후 거래일로 분리",  
        ),  
        (  
            "실제 다음 거래일 체결",  
            "PASS",  
            "종목별 DataFrame의 실제 다음 행을 체결일로 사용",  
        ),  
        (  
            "시장지수 날짜 정렬",  
            "PASS",  
            "시장지수 reindex + ffill 적용",  
        ),  
        (  
            "가격 데이터 기준일 명시",  
            "PASS",  
            "실행시각과 분석 기준일을 별도 표시",  
        ),  
        (  
            "Return60 계산",  
            "PASS",  
            "Return60 생성 후 RelReturn60 계산",  
        ),  
        (  
            "원천 거래대금 보존",  
            "PASS",  
            "Amount 계열 원천 컬럼을 OHLCV 선택 전에 탐색",  
        ),  
        (  
            "거래대금 결측값 0 대체 방지",  
            "PASS",  
            "결측 거래대금은 분석 오류로 처리",  
        ),  
        (  
            "거래대금 단위 자동 검증",  
            "CHECK",  
            "원(KRW) 단위로 취급하지만 데이터 공급원의 원천 단위를 자동 검증하지 않음",  
        ),  
        (  
            "CMF 초기 결측 유지",  
            "PASS",  
            "CMF 계산 불가 구간은 NaN 유지",  
        ),  
        (  
            "CMF 결측 0 대체 방지",  
            "PASS",  
            "fillna(0) 제거",  
        ),  
        (  
            "역사적 유니버스 CSV 실제 적용",  
            "PASS",  
            "CSV 종목과 StartDate/EndDate를 실제 universe_data 생성에 적용",  
        ),  
        (  
            "현재 KRX와 역사적 유니버스 중복 방지",  
            "PASS",  
            "CSV가 있으면 CSV 유니버스를 사용하고 현재 KRX와 합산하지 않음",  
        ),  
        (  
            "예약금 반환 - 슬롯 초과",  
            "PASS",  
            "취소 시 reserved_cash 즉시 반환",  
        ),  
        (  
            "예약금 반환 - 주문 실패",  
            "PASS",  
            "시가/수량/현금/가격 오류 시 예약금 반환",  
        ),  
        (  
            "예약금 반환 - 데이터 오류",  
            "PASS",  
            "주문 대상 데이터가 없으면 예약금 반환",  
        ),  
        (  
            "예약금 반환 - 마지막 날",  
            "PASS",  
            "백테스트 종료 시 미체결 예약 주문 모두 취소",  
        ),  
        (  
            "예약금/현금 이중 차감 방지",  
            "PASS",  
            "예약은 가상 잠금이며 실제 현금은 체결 시에만 차감",  
        ),  
        (  
            "손절 갭 하락 체결",  
            "PASS",  
            "시가가 손절가 이하이면 실제 시가 기준 손절",  
        ),  
        (  
            "미청산 포지션 실현승률 포함 방지",  
            "PASS",  
            "완전 청산 거래만 closed_trades에 포함",  
        ),  
        (  
            "미청산 포지션 평가",  
            "PASS",  
            "마지막 평가가격 기준 미실현손익 별도 표시",  
        ),  
        (  
            "워크포워드 시간순 분리",  
            "PASS",  
            "Train에서 선택한 점수를 이후 Test에 적용",  
        ),  
        (  
            "전략 비교 실제 계산",  
            "PASS",  
            "A~H 성과를 과거 OHLCV 데이터에서 직접 계산",  
        ),  
        (  
            "생존자 편향",  
            "LIMITATION",  
            "현재 KRX 목록을 사용할 경우 과거 상장폐지 종목이 포함되지 않을 수 있음",  
        ),  
        (  
            "외국인/기관 수급 검증",  
            "LIMITATION",  
            "별도 수급 API를 사용하지 않음",  
        ),  
    ]  
  
    return pd.DataFrame(  
        audit_data,  
        columns=[  
            "감사 항목",  
            "상태",  
            "상세 설명",  
        ],  
    )  
  
  
# ============================================================  
# 9. Streamlit 메인 UI  
# ============================================================  
def main():  
  
    kst_now = get_kst_now()  
  
    st.title(  
        "📈 KRX V10+ 급등 전조 스캐너 & 백테스트 연구소"  
    )  
  
    st.caption(  
        f"실행 시각 (KST): "  
        f"{kst_now.strftime('%Y-%m-%d %H:%M:%S')} "  
        f"| 가격·거래량·시장상대강도 전용 검증 도구"  
    )  
  
    # ========================================================  
    # Sidebar  
    # ========================================================  
    st.sidebar.header(  
        "⚙️ 분석 및 스캔 설정"  
    )  
  
    market_choice = st.sidebar.selectbox(  
        "시장 선택",  
        [  
            "KOSPI + KOSDAQ",  
            "KOSPI",  
            "KOSDAQ",  
        ],  
    )  
  
    uploaded_csv = st.sidebar.file_uploader(  
        "역사적 유니버스 CSV (선택)",  
        type=["csv"],  
    )  
  
    hist_universe, univ_msg = (  
        parse_historical_universe_csv(  
            uploaded_csv  
        )  
    )  
  
    if hist_universe is None:  
  
        st.sidebar.info(  
            "ℹ️ " + univ_msg  
        )  
  
        st.sidebar.warning(  
            "⚠️ 현재 종목 목록을 사용하므로 "  
            "과거 백테스트에는 생존자 편향이 남을 수 있습니다."  
        )  
  
    else:  
  
        st.sidebar.success(  
            "✅ " + univ_msg  
        )  
  
        st.sidebar.caption(  
            f"CSV 종목 행 수: "  
            f"{len(hist_universe):,}"  
        )  
  
    st.sidebar.subheader(  
        "필수 게이트 조건"  
    )  
  
    min_amount_100m = st.sidebar.number_input(  
        "최소 20일 평균 거래대금 (억원)",  
        value=10,  
        min_value=1,  
    )  
  
    st.sidebar.caption(  
        "거래대금은 원(KRW) 단위로 취급합니다. "  
        "원천 데이터 단위 자동 검증은 CHECK 상태입니다."  
    )  
  
    max_today_return = st.sidebar.number_input(  
        "당일 상승률 제한 (%) - 추격 방지",  
        value=12.0,  
        min_value=0.0,  
    )  
  
    min_score = st.sidebar.slider(  
        "최소 급등 전조 점수",  
        0,  
        100,  
        65,  
    )  
  
    missing_handling = st.sidebar.selectbox(  
        "결측 지표 처리 방식",  
        [  
            "지표 제외 후 나머지로 계산",  
            "해당 종목 제외",  
            "분석불가 표시",  
        ],  
    )  
  
    sample_scan_count = st.sidebar.number_input(  
        "스캔 종목 수 제한 (0: 전체)",  
        value=100,  
        min_value=0,  
    )  
  
    settings = {  
        "min_amount_100m": min_amount_100m,  
        "max_today_return": max_today_return,  
        "min_score": min_score,  
        "missing_handling": missing_handling,  
        "max_holding_days": 10,  
    }  
  
    # ========================================================  
    # FIX:  
    # 워크포워드와 60일 지표를 실제로 사용할 수 있도록  
    # 기본 데이터 기간을 3년으로 확대.  
    # ========================================================  
    end_dt = kst_now.date()  
  
    start_dt = (  
        end_dt  
        - timedelta(days=365 * 3)  
    )  
  
    # ========================================================  
    # 탭  
    # ========================================================  
    (  
        tab1,  
        tab2,  
        tab3,  
        tab4,  
        tab5,  
        tab6,  
    ) = st.tabs(  
        [  
            "🔍 급등 전조 후보 스캐너",  
            "📊 후보 상세 분석",  
            "⚖️ 전략 조건별 비교",  
            "🔄 V9 통합 백테스트",  
            "⏳ 워크포워드 검증",  
            "🛡️ 무결성 & 데이터 품질",  
        ]  
    )  
  
    # ========================================================
    # FIX: 스캔은 앱 시작 시 자동 실행하지 않고 버튼 클릭 시에만 실행.
    # ========================================================
    csv_signature = None
    if uploaded_csv is not None:
        csv_signature = (
            getattr(uploaded_csv, "name", ""),
            getattr(uploaded_csv, "size", None),
        )

    scan_signature = (
        market_choice,
        int(sample_scan_count),
        csv_signature,
    )

    if "v10_scan_cache" not in st.session_state:
        st.session_state["v10_scan_cache"] = None

    scan_clicked = st.button(
        "🔍 스캔 시작" if st.session_state["v10_scan_cache"] is None else "🔄 다시 스캔",
        type="primary",
    )

    cached = st.session_state.get("v10_scan_cache")
    if cached is not None and cached.get("signature") != scan_signature:
        st.warning(
            "⚠️ 시장/유니버스/스캔 종목 수 설정이 변경되었습니다. "
            "현재 설정으로 다시 스캔하려면 **🔄 다시 스캔**을 눌러주세요."
        )
        cached = None

    if scan_clicked:
        # ========================================================  
        # 유니버스 구성  
        # ========================================================  
        with st.spinner(  
            "KRX 가격 데이터 수집 및 분석 중..."  
        ):  
  
            # ====================================================  
            # FIX:  
            # CSV가 있으면 현재 KRX 목록을 사용하지 않는다.  
            # ====================================================  
            if hist_universe is not None:  
  
                stock_list = (  
                    hist_universe.copy()  
                )  
  
                # 시장 선택 적용  
                if market_choice == "KOSPI":  
                    stock_list = stock_list[  
                        stock_list["Market"]  
                        == "KOSPI"  
                    ].copy()  
  
                elif market_choice == "KOSDAQ":  
                    stock_list = stock_list[  
                        stock_list["Market"]  
                        == "KOSDAQ"  
                    ].copy()  
  
                else:  
                    stock_list = stock_list[  
                        stock_list["Market"].isin(  
                            [  
                                "KOSPI",  
                                "KOSDAQ",  
                                "ALL",  
                            ]  
                        )  
                    ].copy()  
  
                universe_source = (  
                    "역사적 유니버스 CSV"  
                )  
  
            else:  
  
                stock_list = (  
                    get_krx_stock_list()  
                )  
  
                if (  
                    market_choice  
                    != "KOSPI + KOSDAQ"  
                ):  
                    stock_list = stock_list[  
                        stock_list["Market"]  
                        == market_choice  
                    ].copy()  
  
                universe_source = (  
                    "현재 KRX 유니버스"  
                )  
  
            if sample_scan_count > 0:  
                stock_list = (  
                    stock_list  
                    .head(  
                        int(  
                            sample_scan_count  
                        )  
                    )  
                    .copy()  
                )  
  
            st.info(  
                f"사용 유니버스: "  
                f"**{universe_source}** | "  
                f"분석 대상: **{len(stock_list):,}개**"  
            )  
  
            # 시장지수  
            kospi_idx = fetch_market_index(  
                "KOSPI",  
                start_dt,  
                end_dt,  
            )  
  
            kosdaq_idx = fetch_market_index(  
                "KOSDAQ",  
                start_dt,  
                end_dt,  
            )  
  
            if (  
                kospi_idx is None  
                or kosdaq_idx is None  
            ):  
                st.warning(  
                    "⚠️ 시장지수 데이터 부족: "  
                    "상대강도 지표 일부가 결측 처리될 수 있습니다."  
                )  
  
            universe_data = {}  
  
            data_quality_logs = []  
            excluded_logs = []  
  
            amount_meta_by_symbol = {}  
  
            # ====================================================  
            # 종목 데이터  
            # ====================================================  
            # FIX: 실제 반복 처리 횟수를 기준으로 진행률 표시.  
            scan_total = len(stock_list)  
            scan_progress = st.progress(0.0)  
            scan_status = st.empty()  
  
            for scan_no, (_, stock_row) in enumerate(  
                stock_list.iterrows(), start=1  
            ):  
                # FIX: 각 종목 처리 시작 시 실제 처리 개수를 갱신.  
                scan_progress.progress(  
                    scan_no / scan_total if scan_total else 1.0  
                )  
                scan_status.caption(  
                    f"🔎 스캔 진행: {scan_no:,} / {scan_total:,} | "  
                    f"현재 종목: {stock_row['Name']} ({stock_row['Symbol']})"  
                )  
  
                sym = str(  
                    stock_row[  
                        "Symbol"  
                    ]  
                )  
  
                name = str(  
                    stock_row[  
                        "Name"  
                    ]  
                )  
  
                mkt = str(  
                    stock_row[  
                        "Market"  
                    ]  
                )  
  
                if mkt == "KOSPI":  
                    m_idx = kospi_idx  
                elif mkt == "KOSDAQ":  
                    m_idx = kosdaq_idx  
                else:  
                    m_idx = None  
  
                # =================================================  
                # FIX:  
                # CSV의 유효기간을 고려해서 필요한 범위만 가져옴.  
                # =================================================  
                fetch_start = start_dt  
                fetch_end = end_dt  
  
                if hist_universe is not None:  
  
                    rules = hist_universe[  
                        hist_universe[  
                            "Symbol"  
                        ].astype(str)  
                        == sym  
                    ]  
  
                    if not rules.empty:  
  
                        rule_start = (  
                            pd.Timestamp(  
                                rules[  
                                    "StartDate"  
                                ].min()  
                            ).date()  
                        )  
  
                        rule_end = (  
                            pd.Timestamp(  
                                rules[  
                                    "EndDate"  
                                ].max()  
                            ).date()  
                        )  
  
                        fetch_start = max(  
                            fetch_start,  
                            rule_start,  
                        )  
  
                        fetch_end = min(  
                            fetch_end,  
                            rule_end,  
                        )  
  
                        if (  
                            fetch_start  
                            > fetch_end  
                        ):  
                            data_quality_logs.append(  
                                {  
                                    "Symbol": sym,  
                                    "Name": name,  
                                    "Status": "기간 밖",  
                                    "Reasons": (  
                                        "CSV 유효기간이 분석기간과 겹치지 않음"  
                                    ),  
                                    "AmountSource": "",  
                                    "AmountUnitStatus": "",  
                                    "CMFStatus": "",  
                                }  
                            )  
                            continue  
  
                    else:  
                        data_quality_logs.append(  
                            {  
                                "Symbol": sym,  
                                "Name": name,  
                                "Status": "기간 밖",  
                                "Reasons": (  
                                    "CSV 유효기간 정보 없음"  
                                ),  
                                "AmountSource": "",  
                                "AmountUnitStatus": "",  
                                "CMFStatus": "",  
                            }  
                        )  
                        continue  
  
                raw_df, msg, data_meta = (  
                    fetch_ohlcv_data(  
                        sym,  
                        fetch_start.strftime(  
                            "%Y-%m-%d"  
                        ),  
                        fetch_end.strftime(  
                            "%Y-%m-%d"  
                        ),  
                    )  
                )  
  
                amount_meta_by_symbol[  
                    sym  
                ] = data_meta  
  
                is_valid, audit_reasons = (  
                    audit_stock_data_quality(  
                        raw_df,  
                        name,  
                        data_meta,  
                    )  
                )  
  
                if not is_valid:  
  
                    data_quality_logs.append(  
                        {  
                            "Symbol": sym,  
                            "Name": name,  
                            "Status": "오류/부족",  
                            "Reasons": ", ".join(  
                                audit_reasons  
                            ),  
                            "AmountSource": data_meta.get(  
                                "amount_source",  
                                "",  
                            ),  
                            "AmountUnitStatus": data_meta.get(  
                                "amount_unit_status",  
                                "",  
                            ),  
                            "CMFStatus": "",  
                        }  
                    )  
  
                    continue  
  
                feat_df = (  
                    calculate_technical_features(  
                        raw_df,  
                        m_idx,  
                    )  
                )  
  
                if feat_df is None:  
                    data_quality_logs.append(  
                        {  
                            "Symbol": sym,  
                            "Name": name,  
                            "Status": "분석불가",  
                            "Reasons": (  
                                "기술지표 계산 실패"  
                            ),  
                            "AmountSource": data_meta.get(  
                                "amount_source",  
                                "",  
                            ),  
                            "AmountUnitStatus": data_meta.get(  
                                "amount_unit_status",  
                                "",  
                            ),  
                            "CMFStatus": "",  
                        }  
                    )  
                    continue  
  
                # =================================================  
                # FIX:  
                # CSV 유효기간을 실제 DataFrame 날짜에 적용.  
                # =================================================  
                if hist_universe is not None:  
                    feat_df = (  
                        apply_historical_universe_validity(  
                            feat_df,  
                            sym,  
                            hist_universe,  
                        )  
                    )  
  
                if feat_df.empty:  
                    data_quality_logs.append(  
                        {  
                            "Symbol": sym,  
                            "Name": name,  
                            "Status": "기간 밖",  
                            "Reasons": (  
                                "역사적 유니버스 유효기간 내 가격 데이터 없음"  
                            ),  
                            "AmountSource": data_meta.get(  
                                "amount_source",  
                                "",  
                            ),  
                            "AmountUnitStatus": data_meta.get(  
                                "amount_unit_status",  
                                "",  
                            ),  
                            "CMFStatus": "",  
                        }  
                    )  
                    continue  
  
                cmf_last = feat_df[  
                    "CMF20"  
                ].iloc[-1]  
  
                if pd.isna(cmf_last):  
                    cmf_status = (  
                        "CMF 계산 불가"  
                    )  
                else:  
                    cmf_status = "정상"  
  
                universe_data[  
                    sym  
                ] = feat_df  
  
                data_quality_logs.append(  
                    {  
                        "Symbol": sym,  
                        "Name": name,  
                        "Status": "정상",  
                        "Reasons": "합격",  
                        "AmountSource": data_meta.get(  
                            "amount_source",  
                            "",  
                        ),  
                        "AmountUnitStatus": data_meta.get(  
                            "amount_unit_status",  
                            "",  
                        ),  
                        "CMFStatus": cmf_status,  
                    }  
                )  
  


        # FIX: 스캔 완료 상태를 실제 처리 결과로 표시.
        scan_progress.progress(1.0)
        scan_status.success(
            f"✅ 스캔 완료: {len(stock_list):,}개 종목 처리 | "
            f"분석 성공: {len(universe_data):,}개"
        )

        st.session_state["v10_scan_cache"] = {
            "signature": scan_signature,
            "stock_list": stock_list,
            "universe_source": universe_source,
            "universe_data": universe_data,
            "data_quality_logs": data_quality_logs,
            "excluded_logs": excluded_logs,
            "amount_meta_by_symbol": amount_meta_by_symbol,
        }
    elif cached is not None:
        stock_list = cached["stock_list"]
        universe_source = cached["universe_source"]
        universe_data = cached["universe_data"]
        data_quality_logs = cached["data_quality_logs"]
        excluded_logs = cached["excluded_logs"]
        amount_meta_by_symbol = cached["amount_meta_by_symbol"]
    else:
        st.info(
            "📌 자동 스캔은 실행하지 않습니다. **🔍 스캔 시작** 버튼을 눌러주세요."
        )
        return
    # ========================================================  
    # 기준일  
    # ========================================================  
    if not universe_data:  
  
        st.error(  
            "분석 가능한 종목 데이터가 없습니다."  
        )  
        return  
  
    # ========================================================  
    # FIX:  
    # 전체 유니버스에서 가장 최근 거래일을 분석 기준일로 사용.  
    # 각 종목의 마지막 날짜가 다르면 Gate에서 제외된다.  
    # ========================================================  
    ref_date = max(  
        df.index[-1].date()  
        for df in universe_data.values()  
    )  
  
    st.info(  
        f"📅 **분석 기준일**: {ref_date} | "  
        f"**예측 대상일**: 각 종목의 실제 다음 거래일 시가 | "  
        f"**유니버스**: {universe_source}"  
    )  
  
    if ref_date < (  
        end_dt  
        - timedelta(days=4)  
    ):  
        st.error(  
            "⚠️ 데이터 지연 가능성 경고: "  
            "최근 거래일 데이터가 도달하지 않았습니다."  
        )  
  
    # ========================================================  
    # Tab 1  
    # ========================================================  
    with tab1:  
  
        st.subheader(  
            "🎯 급등 전조 선행 신호 스캔 결과"  
        )  
  
        candidates = []  
  
        for sym, df in universe_data.items():  
  
            row = df.iloc[-1]  
  
            last_dt = (  
                df.index[-1].date()  
            )  
  
            stock_match = stock_list[  
                stock_list[  
                    "Symbol"  
                ].astype(str)  
                == str(sym)  
            ]  
  
            if stock_match.empty:  
                continue  
  
            name = str(  
                stock_match.iloc[0][  
                    "Name"  
                ]  
            )  
  
            mkt = str(  
                stock_match.iloc[0][  
                    "Market"  
                ]  
            )  
  
            passed, gate_reasons = (  
                evaluate_essential_gates(  
                    row,  
                    settings,  
                    last_dt,  
                    ref_date,  
                )  
            )  
  
            (  
                tot_score,  
                s_A,  
                s_B,  
                s_C,  
                s_D,  
            ) = calculate_precursor_subscores(  
                row,  
                missing_handling,  
            )  
  
            if not passed:  
  
                excluded_logs.append(  
                    {  
                        "Symbol": sym,  
                        "Name": name,  
                        "Reasons": ", ".join(  
                            gate_reasons  
                        ),  
                    }  
                )  
  
                continue  
  
            if pd.isna(  
                tot_score  
            ):  
  
                missing = (  
                    get_missing_precursor_indicators(  
                        row  
                    )  
                )  
  
                reason = (  
                    "전조 지표 계산 불가"  
                )  
  
                if "CMF20" in missing:  
                    reason += (  
                        " / CMF 계산 불가"  
                    )  
  
                if missing:  
                    reason += (  
                        " / 결측: "  
                        + ", ".join(  
                            missing  
                        )  
                    )  
  
                excluded_logs.append(  
                    {  
                        "Symbol": sym,  
                        "Name": name,  
                        "Reasons": reason,  
                    }  
                )  
  
                continue  
  
            if (  
                tot_score  
                < min_score  
            ):  
  
                excluded_logs.append(  
                    {  
                        "Symbol": sym,  
                        "Name": name,  
                        "Reasons": (  
                            f"점수 미달 "  
                            f"({tot_score:.1f}점)"  
                        ),  
                    }  
                )  
  
                continue  
  
            pattern_type = (  
                classify_surge_pattern(  
                    row  
                )  
            )  
  
            next_trade_date = (  
                get_next_trading_date(  
                    df,  
                    df.index[-1],  
                )  
            )  
  
            if next_trade_date is None:  
                next_trade_label = (  
                    "다음 거래일 데이터 없음"  
                )  
            else:  
                next_trade_label = (  
                    str(  
                        next_trade_date.date()  
                    )  
                    + " 시가"  
                )  
  
            candidates.append(  
                {  
                    "종목코드": sym,  
                    "종목명": name,  
                    "시장": mkt,  
                    "분석기준일": ref_date,  
                    "실제 다음 거래일": next_trade_label,  
                    "전조 점수": round(  
                        tot_score,  
                        1,  
                    ),  
                    "압축 점수": round(  
                        s_A,  
                        1,  
                    ),  
                    "돌파 준비 점수": round(  
                        s_B,  
                        1,  
                    ),  
                    "자금 유입 점수": round(  
                        s_C,  
                        1,  
                    ),  
                    "추격 위험 점수": round(  
                        s_D,  
                        1,  
                    ),  
                    "오늘 수익률(%)": round(  
                        row.get(  
                            "Return1",  
                            np.nan,  
                        ),  
                        2,  
                    ),  
                    "5일 수익률(%)": round(  
                        row.get(  
                            "Return5",  
                            np.nan,  
                        ),  
                        2,  
                    ),  
                    "20일 수익률(%)": round(  
                        row.get(  
                            "Return20",  
                            np.nan,  
                        ),  
                        2,  
                    ),  
                    "20일 박스폭(%)": round(  
                        row.get(  
                            "BoxWidth20",  
                            np.nan,  
                        ),  
                        2,  
                    ),  
                    "거래량/20일평균": round(  
                        row.get(  
                            "VolumeRatio20",  
                            np.nan,  
                        ),  
                        2,  
                    ),  
                    "거래대금/20일평균": round(  
                        row.get(  
                            "AmountRatio20",  
                            np.nan,  
                        ),  
                        2,  
                    ),  
                    "RSI": round(  
                        row.get(  
                            "RSI14",  
                            np.nan,  
                        ),  
                        1,  
                    ),  
                    "CMF": (  
                        "계산 불가"  
                        if pd.isna(  
                            row.get(  
                                "CMF20",  
                                np.nan,  
                            )  
                        )  
                        else round(  
                            row[  
                                "CMF20"  
                            ],  
                            3,  
                        )  
                    ),  
                    "전조 유형": pattern_type,  
                    "추천 근거": (  
                        "가격·거래량 조건을 "  
                        "실제 계산한 연구용 후보"  
                    ),  
                    "주의사항": (  
                        "상승을 보장하지 않으며 "  
                        "다음 거래일 시가 체결 결과를 "  
                        "별도 검증해야 함"  
                    ),  
                }  
            )  
  
        cand_df = pd.DataFrame(  
            candidates  
        )  
  
        if not cand_df.empty:  
  
            cand_df = cand_df.sort_values(  
                by="전조 점수",  
                ascending=False,  
            )  
  
            cand_df.insert(  
                0,  
                "순위",  
                range(  
                    1,  
                    len(cand_df) + 1,  
                ),  
            )  
  
            st.dataframe(  
                cand_df,  
                use_container_width=True,  
                hide_index=True,  
            )  
  
            csv_data = (  
                cand_df  
                .to_csv(  
                    index=False  
                )  
                .encode(  
                    "utf-8-sig"  
                )  
            )  
  
            st.download_button(  
                "📥 후보 결과 CSV 다운로드",  
                csv_data,  
                f"candidates_{ref_date}.csv",  
                "text/csv",  
            )  
  
        else:  
  
            st.warning(  
                "조건을 충족하는 급등 전조 후보 종목이 없습니다."  
            )  
  
        st.caption(  
            "거래대금은 원(KRW) 단위로 취급합니다. "  
            "원천 데이터의 실제 단위는 자동 검증하지 못하므로 "  
            "무결성 감사에서 CHECK로 표시합니다."  
        )  
  
        st.caption(  
            "CMF가 계산 불가능한 종목은 설정한 결측 처리 방식에 따라 "  
            "제외되거나 해당 지표를 제외하고 점수를 계산합니다."  
        )  
  
        st.caption(  
            "OBV·CMF는 가격과 거래량으로 계산한 보조지표이며 "  
            "실제 투자자별 수급 데이터가 아닙니다."  
        )  
  
    # ========================================================  
    # Tab 2  
    # ========================================================  
    with tab2:  
  
        st.subheader(  
            "🔍 후보 종목 상세 분석 및 기술적 차트"  
        )  
  
        if not cand_df.empty:  
  
            selected_sym = st.selectbox(  
                "분석할 후보 종목 선택",  
                cand_df[  
                    "종목코드"  
                ]  
                + " | "  
                + cand_df[  
                    "종목명"  
                ],  
            )  
  
            sym_code = (  
                selected_sym.split(  
                    " | "  
                )[0]  
            )  
  
            target_df = universe_data[  
                sym_code  
            ]  
  
            last_r = target_df.iloc[  
                -1  
            ]  
  
            st.write(  
                f"### {selected_sym} 지표 요약"  
            )  
  
            col_a, col_b, col_c, col_d = (  
                st.columns(4)  
            )  
  
            col_a.metric(  
                "종가",  
                f"{int(last_r['Close']):,} 원",  
            )  
  
            col_b.metric(  
                "20일 평균 거래대금",  
                f"{int(last_r['AmountMA20']/100_000_000):,} 억원",  
            )  
  
            col_c.metric(  
                "RSI (14)",  
                f"{last_r['RSI14']:.1f}",  
            )  
  
            col_d.metric(  
                "ATR (14)",  
                f"{int(last_r['ATR14']):,} 원",  
            )  
  
            cmf_value = last_r.get(  
                "CMF20",  
                np.nan,  
            )  
  
            if pd.isna(  
                cmf_value  
            ):  
                st.warning(  
                    "CMF 계산 불가"  
                )  
            else:  
                st.metric(  
                    "CMF (20)",  
                    f"{cmf_value:.3f}",  
                )  
  
            st.caption(  
                "거래대금: 원(KRW) 단위로 취급 / "  
                "원천 단위 자동 검증: CHECK"  
            )  
  
            entry_ref = last_r[  
                "Close"  
            ]  
  
            atr_val = last_r[  
                "ATR14"  
            ]  
  
            stop_ref = (  
                entry_ref  
                - 2.0 * atr_val  
            )  
  
            tp1_ref = (  
                entry_ref  
                + 2.0  
                * (  
                    entry_ref  
                    - stop_ref  
                )  
            )  
  
            tp2_ref = (  
                entry_ref  
                + 3.5  
                * (  
                    entry_ref  
                    - stop_ref  
                )  
            )  
  
            st.info(  
                f"🔬 **연구용 참고 가격 수준 "  
                f"(다음날 시가 기준 참고용)**:\n"  
                f"- 신호일 종가: "  
                f"{int(entry_ref):,}원\n"  
                f"- ATR 기준 손절 참고선: "  
                f"{int(stop_ref):,}원 "  
                f"(-{((entry_ref-stop_ref)/entry_ref*100):.1f}%)\n"  
                f"- 1차 목표 참고선: "  
                f"{int(tp1_ref):,}원\n"  
                f"- 2차 목표 참고선: "  
                f"{int(tp2_ref):,}원"  
            )  
  
            fig = make_subplots(  
                rows=3,  
                cols=1,  
                shared_xaxes=True,  
                vertical_spacing=0.03,  
                row_heights=[  
                    0.5,  
                    0.25,  
                    0.25,  
                ],  
            )  
  
            fig.add_trace(  
                go.Candlestick(  
                    x=target_df.index,  
                    open=target_df[  
                        "Open"  
                    ],  
                    high=target_df[  
                        "High"  
                    ],  
                    low=target_df[  
                        "Low"  
                    ],  
                    close=target_df[  
                        "Close"  
                    ],  
                    name="OHLC",  
                ),  
                row=1,  
                col=1,  
            )  
  
            fig.add_trace(  
                go.Scatter(  
                    x=target_df.index,  
                    y=target_df[  
                        "EMA20"  
                    ],  
                    name="EMA20",  
                    line=dict(  
                        color="orange"  
                    ),  
                ),  
                row=1,  
                col=1,  
            )  
  
            fig.add_trace(  
                go.Scatter(  
                    x=target_df.index,  
                    y=target_df[  
                        "High20"  
                    ],  
                    name="20일 박스상단",  
                    line=dict(  
                        color="red",  
                        dash="dash",  
                    ),  
                ),  
                row=1,  
                col=1,  
            )  
  
            fig.add_trace(  
                go.Bar(  
                    x=target_df.index,  
                    y=target_df[  
                        "Volume"  
                    ],  
                    name="거래량",  
                    marker_color="blue",  
                ),  
                row=2,  
                col=1,  
            )  
  
            fig.add_trace(  
                go.Scatter(  
                    x=target_df.index,  
                    y=target_df[  
                        "RSI14"  
                    ],  
                    name="RSI",  
                    line=dict(  
                        color="purple"  
                    ),  
                ),  
                row=3,  
                col=1,  
            )  
  
            fig.update_layout(  
                height=650,  
                title_text=(  
                    f"{selected_sym} 기술적 분석 차트"  
                ),  
                xaxis_rangeslider_visible=False,  
            )  
  
            st.plotly_chart(  
                fig,  
                use_container_width=True,  
            )  
  
        else:  
  
            st.info(  
                "현재 조건을 충족하는 후보가 없어 상세 분석을 표시할 수 없습니다."  
            )  
  
    # ========================================================  
    # Tab 3  
    # ========================================================  
    with tab3:  
  
        st.subheader(  
            "⚖️ 단계별 전략 조건 비교 (A ~ H)"  
        )  
  
        st.write(  
            "동일 기간·동일 유니버스에서 "  
            "각 조건을 실제 과거 가격 데이터에 적용합니다."  
        )  
  
        st.caption(  
            "1일후 수익률은 신호일 종가 → 실제 다음 거래일 시가 기준이며, "  
            "5/20일 수익률은 다음 거래일 시가 진입 후 해당 거래일수의 종가 기준입니다."  
        )  
  
        comparison_df = (  
            run_strategy_comparison(  
                universe_data,  
                settings,  
            )  
        )  
  
        if not comparison_df.empty:  
  
            st.dataframe(  
                comparison_df,  
                use_container_width=True,  
                hide_index=True,  
            )  
  
        else:  
  
            st.warning(  
                "실제 비교 가능한 과거 신호가 없습니다."  
            )  
  
    # ========================================================  
    # Tab 4  
    # ========================================================  
    with tab4:  
  
        st.subheader(  
            "🔄 V9 백테스트 엔진 시뮬레이션"  
        )  
  
        st.markdown(  
            """  
**청산 우선순위**  
  
1. 손절  
2. 2차 목표가  
3. 1차 목표가 부분익절  
4. EMA20 추세 이탈  
5. 최대 보유기간 만료  
  
**중요:** 다음날 시가가 손절가보다 낮으면 손절가가 아니라 실제 시가로 손절됩니다.  
"""  
        )  
  
        init_cap = st.number_input(  
            "초기 자본금 (원)",  
            value=10_000_000,  
            step=1_000_000,  
        )  
  
        max_pos = st.slider(  
            "최대 동시보유 종목 수",  
            1,  
            10,  
            5,  
        )  
  
        if st.button(  
            "🚀 V9 백테스트 실행"  
        ):  
  
            with st.spinner(  
                "백테스트 시뮬레이션 진행 중..."  
            ):  
  
                (  
                    eq_df,  
                    closed_df,  
                    logs_df,  
                    open_pos,  
                ) = run_v9_backtest_engine(  
                    universe_data,  
                    settings,  
                    initial_capital=init_cap,  
                    max_concurrent=max_pos,  
                )  
  
            if (  
                eq_df is not None  
                and not eq_df.empty  
            ):  
  
                final_asset = float(  
                    eq_df.iloc[-1][  
                        "TotalAsset"  
                    ]  
                )  
  
                total_ret = (  
                    final_asset  
                    / init_cap  
                    - 1  
                ) * 100  
  
                m1, m2, m3, m4 = (  
                    st.columns(4)  
                )  
  
                m1.metric(  
                    "최종 자산",  
                    f"{int(final_asset):,} 원",  
                )  
  
                m2.metric(  
                    "총 수익률",  
                    f"{total_ret:.2f} %",  
                )  
  
                m3.metric(  
                    "완전 청산 거래 수",  
                    f"{len(closed_df) if closed_df is not None else 0} 건",  
                )  
  
                win_r = (  
                    (  
                        closed_df[  
                            "Net_PnL"  
                        ]  
                        > 0  
                    ).mean()  
                    * 100  
                    if (  
                        closed_df is not None  
                        and not closed_df.empty  
                    )  
                    else 0.0  
                )  
  
                m4.metric(  
                    "실현 승률",  
                    f"{win_r:.1f} %",  
                )  
  
                st.line_chart(  
                    eq_df.set_index(  
                        "Date"  
                    )[  
                        "TotalAsset"  
                    ]  
                )  
  
                # =================================================  
                # FIX:  
                # 미청산 포지션을 실현 거래와 분리.  
                # =================================================  
                st.write(  
                    "### 미청산 포지션"  
                )  
  
                final_date = eq_df.iloc[  
                    -1  
                ]["Date"]  
  
                open_df = (  
                    build_open_positions_df(  
                        open_pos,  
                        universe_data,  
                        final_date,  
                    )  
                )  
  
                if (  
                    open_df is not None  
                    and not open_df.empty  
                ):  
  
                    unrealized_total = (  
                        open_df[  
                            "미실현손익"  
                        ].sum()  
                    )  
  
                    realized_open_total = (  
                        open_df[  
                            "실현손익"  
                        ].sum()  
                    )  
  
                    c1, c2, c3 = (  
                        st.columns(3)  
                    )  
  
                    c1.metric(  
                        "미청산 포지션 수",  
                        len(open_df),  
                    )  
  
                    c2.metric(  
                        "미실현 평가손익",  
                        f"{int(unrealized_total):,} 원",  
                    )  
  
                    c3.metric(  
                        "미청산 포지션 실현손익",  
                        f"{int(realized_open_total):,} 원",  
                    )  
  
                    st.dataframe(  
                        open_df,  
                        use_container_width=True,  
                        hide_index=True,  
                    )  
  
                    st.caption(  
                        "미청산 포지션은 실현 승률에 포함하지 않습니다. "  
                        "최종 자산에는 마지막 거래일 종가 기준 평가금액으로 반영됩니다."  
                    )  
  
                else:  
  
                    st.info(  
                        "백테스트 종료 시 미청산 포지션이 없습니다."  
                    )  
  
                if (  
                    closed_df is not None  
                    and not closed_df.empty  
                ):  
  
                    st.write(  
                        "### 청산 완료 거래 목록 (실현 성과)"  
                    )  
  
                    st.dataframe(  
                        closed_df,  
                        use_container_width=True,  
                    )  
  
                if (  
                    logs_df is not None  
                    and not logs_df.empty  
                ):  
  
                    with st.expander(  
                        "거래/예약금 처리 로그"  
                    ):  
                        st.dataframe(  
                            logs_df,  
                            use_container_width=True,  
                            hide_index=True,  
                        )  
  
    # ========================================================  
    # Tab 5  
    # ========================================================  
    with tab5:  
  
        st.subheader(  
            "⏳ 시간순 워크포워드(Walk-Forward) 검증"  
        )  
  
        st.write(  
            "과거 Train 구간에서 선택한 최소 점수를 "  
            "이후 Test 구간에 그대로 적용합니다."  
        )  
  
        st.caption(  
            "워크포워드 결과가 없으면 현재 데이터 기간이 Train + Test 기간을 충족하지 않는 것입니다."  
        )  
  
        if st.button(  
            "🧪 워크포워드 검증 실행"  
        ):  
  
            with st.spinner(  
                "워크포워드 검증 계산 중..."  
            ):  
  
                wf_df = (  
                    run_walk_forward_validation(  
                        universe_data,  
                        settings,  
                        train_years=1,  
                        test_years=1,  
                    )  
                )  
  
            if (  
                wf_df is not None  
                and not wf_df.empty  
            ):  
  
                st.dataframe(  
                    wf_df,  
                    use_container_width=True,  
                    hide_index=True,  
                )  
  
            else:  
  
                st.warning(  
                    "검증 가능한 연도별 데이터 구간이 부족합니다."  
                )  
  
    # ========================================================  
    # Tab 6  
    # ========================================================  
    with tab6:  
  
        st.subheader(  
            "🛡️ 전략 및 데이터 무결성 감사"  
        )  
  
        audit_res = (  
            run_integrity_audit()  
        )  
  
        st.dataframe(  
            audit_res,  
            use_container_width=True,  
            hide_index=True,  
        )  
  
        st.write(  
            "### 📋 실제 사용 유니버스"  
        )  
  
        st.info(  
            f"유니버스 출처: **{universe_source}** | "  
            f"실제 분석 데이터 종목 수: **{len(universe_data):,}개**"  
        )  
  
        if hist_universe is not None:  
  
            st.caption(  
                "역사적 유니버스 CSV가 제공되었으므로 "  
                "현재 KRX 목록과 합산하지 않았습니다."  
            )  
  
        st.write(  
            "### 📋 종목별 데이터 수집 및 품질 통계"  
        )  
  
        col_q1, col_q2 = (  
            st.columns(2)  
        )  
  
        dq_df = pd.DataFrame(  
            data_quality_logs  
        )  
  
        col_q1.metric(  
            "유니버스 입력 종목 수",  
            len(stock_list),  
        )  
  
        col_q1.metric(  
            "분석 성공 종목 수",  
            (  
                len(  
                    dq_df[  
                        dq_df[  
                            "Status"  
                        ]  
                        == "정상"  
                    ]  
                )  
                if not dq_df.empty  
                else 0  
            ),  
        )  
  
        col_q2.metric(  
            "데이터 오류/부족 종목 수",  
            (  
                len(  
                    dq_df[  
                        dq_df[  
                            "Status"  
                        ]  
                        != "정상"  
                    ]  
                )  
                if not dq_df.empty  
                else 0  
            ),  
        )  
  
        col_q2.metric(  
            "최종 후보 선정 종목 수",  
            (  
                len(cand_df)  
                if not cand_df.empty  
                else 0  
            ),  
        )  
  
        if not dq_df.empty:  
  
            st.dataframe(  
                dq_df,  
                use_container_width=True,  
                hide_index=True,  
            )  
  
        if excluded_logs:  
  
            st.write(  
                "### 후보 제외 사유"  
            )  
  
            excluded_df = pd.DataFrame(  
                excluded_logs  
            )  
  
            st.dataframe(  
                excluded_df,  
                use_container_width=True,  
                hide_index=True,  
            )  
  
  
# ============================================================  
# 실행  
# ============================================================  
if __name__ == "__main__":  
    main()  
