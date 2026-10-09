# -*- coding: utf-8 -*-
"""
ABD ve BIST Çok Zaman Dilimli Hisse Tarayıcı (1s / 2s / 4s / 1G, yalnızca long adaylar)

Eğitim ve araştırma amaçlı bir teknik tarama aracıdır; yatırım tavsiyesi değildir.
Eşikler kanıtlanmış bir üstünlük değil, kullanıcı tarafından değiştirilebilen başlangıç
varsayımlarıdır. Backtest yapılmamıştır. Otomatik emir / aracı kurum entegrasyonu yoktur.

Bölümler (tek dosya):
  1. Yapılandırma        5. İndikatörler
  2. Evren (TradingView) 6. Strateji, skor, risk
  3. Veri (yfinance)     7. Tarama akışı
  4. Seans / mum         8. Arayüz
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

# ======================================================================================
# 1. YAPILANDIRMA
# ======================================================================================
APP_VERSION = "1.0"
UA = {"User-Agent": "mtf-hisse-tarayici/1.0 (personal use; low-volume requests)"}

MARKETS: Dict[str, Dict[str, Any]] = {
    "ABD": dict(
        tv_market="america", tv_exchanges=["NASDAQ", "NYSE", "AMEX"],  # TV'de NYSE American = "AMEX" (doğrulanmadı)
        yf_suffix="", currency="USD", tz="America/New_York", cal="XNYS",
        fallback_session=(dtime(9, 30), dtime(16, 0)), benchmark="SPY",
        default_tol=5, min_price=5.0, min_value=5_000_000.0, max_atr_pct=8.0, portfolio=10_000.0,
    ),
    "BIST": dict(
        tv_market="turkey", tv_exchanges=["BIST"],
        yf_suffix=".IS", currency="TRY", tz="Europe/Istanbul", cal="XIST",
        fallback_session=(dtime(10, 0), dtime(18, 0)), benchmark="XU100.IS",
        default_tol=20, min_price=5.0, min_value=10_000_000.0, max_atr_pct=8.0, portfolio=100_000.0,
    ),
}
TF_ORDER = ["1d", "4h", "2h", "1h"]
TF_LABEL = {"1d": "1G", "4h": "4s", "2h": "2s", "1h": "1s"}
MIN_BARS = {"1d": 210, "4h": 100, "2h": 100, "1h": 100}   # ısınma + yakınsama için asgari tamamlanmış mum
CLS_CAND, CLS_WATCH, CLS_NO, CLS_DATA = "Alım adayı", "İzleme", "Uygun değil", "Veri yetersiz"
HOUR = pd.Timedelta(hours=1)
EPS = pd.Timedelta(seconds=1)


@dataclass
class Cfg:
    market: str = "ABD"
    tol_min: int = 5                      # veri geliş toleransı (dk)
    include_short: bool = True            # seans sonu kısa mumları dahil et
    reject_stale: bool = True             # güncel olmayan 1s veriyi reddet
    regime_mode: str = "Skor bileşeni"    # Kapalı | Zorunlu filtre | Skor bileşeni
    benchmark: str = "SPY"
    trigger_mode: str = "Kırılım ve geri çekilme"
    adx_min: float = 20.0
    max_ext_atr: float = 2.0
    rsi2_lo: float = 50.0
    rsi2_hi: float = 70.0
    rv_thr: float = 1.5
    rv_sessions: int = 20
    max_age: int = 3
    min_price: float = 5.0
    min_value: float = 5_000_000.0
    max_atr_pct: float = 8.0
    stop_method: str = "ATR (4s)"         # ATR (4s) | Yapı (1s dip)
    atr_mult: float = 1.5
    min_rr: float = 2.0
    need_res: bool = False                # direnç bulunmazsa adayı İzleme'ye al
    min_score: int = 60
    portfolio: float = 10_000.0
    cash: float = 10_000.0
    risk_pct: float = 0.5
    max_weight: float = 20.0


# ======================================================================================
# Yardımcılar
# ======================================================================================
def isnan(x) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x)) or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def fmt(x, nd: int = 2) -> str:
    return "—" if isnan(x) else f"{float(x):,.{nd}f}"


def backoff_sleep(attempt: int, base: float = 1.5, cap: float = 30.0) -> None:
    """Exponential backoff + jitter."""
    time.sleep(min(cap, base * (2 ** attempt)) * (0.5 + random.random()))


# ======================================================================================
# 2. EVREN (TradingView tarayıcı uç noktası: resmî/garantili API DEĞİL)
# ======================================================================================
class TVError(Exception):
    def __init__(self, status: Optional[int], msg: str):
        super().__init__(f"TradingView hatası (HTTP {status}): {msg}")
        self.status = status


TV_REQUIRED = ["name", "description", "exchange", "type", "subtype"]
TV_OPTIONAL = ["close", "volume", "market_cap_basic", "sector"]   # alan adları doğrulanamadı; çıkarılabilir


def tv_request(market_key: str, payload: dict, retries: int = 3, timeout: int = 20) -> dict:
    url = f"https://scanner.tradingview.com/{MARKETS[market_key]['tv_market']}/scan"
    last: Exception = TVError(None, "bilinmeyen")
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, json=payload, headers=UA, timeout=timeout)
            if r.status_code == 200:
                try:
                    j = r.json()
                except ValueError:
                    raise TVError(200, "geçersiz JSON")
                if not isinstance(j, dict) or not isinstance(j.get("data"), list):
                    raise TVError(200, "beklenmeyen yanıt yapısı ('data' listesi yok)")
                return j
            if r.status_code in (429, 500, 502, 503, 504):
                last = TVError(r.status_code, r.text[:200])
                backoff_sleep(attempt)
                continue
            raise TVError(r.status_code, r.text[:300])
        except (requests.Timeout, requests.ConnectionError) as e:
            last = TVError(None, f"ağ/timeout: {e}")
            backoff_sleep(attempt)
    raise last


def tv_payload(market_key: str, columns: List[str], start: int, end: int, include_adr: bool) -> dict:
    m = MARKETS[market_key]
    flt = [{"left": "type", "operation": "in_range", "right": ["stock", "dr"] if include_adr else ["stock"]}]
    if m["tv_exchanges"]:
        flt.append({"left": "exchange", "operation": "in_range", "right": m["tv_exchanges"]})
    p = {"filter": flt, "options": {"lang": "en"}, "markets": [m["tv_market"]],
         "symbols": {"query": {"types": []}, "tickers": []}, "columns": columns, "range": [start, end]}
    if "market_cap_basic" in columns:
        p["sort"] = {"sortBy": "market_cap_basic", "sortOrder": "desc"}
    return p


def tv_fetch_universe(market_key: str, include_adr: bool, page: int = 500, max_rows: int = 20000,
                      request_fn: Callable = tv_request) -> Tuple[pd.DataFrame, dict]:
    """Sayfalı çekim. Desteklenmeyen isteğe bağlı sütunları çıkarıp yeniden dener."""
    columns = TV_REQUIRED + TV_OPTIONAL
    removed: List[str] = []
    rows: List[dict] = []
    bad_rows = pages = 0
    total = None
    start = 0
    while start < max_rows:
        try:
            j = request_fn(market_key, tv_payload(market_key, columns, start, start + page, include_adr))
        except TVError as e:
            opt_left = [c for c in columns if c in TV_OPTIONAL]
            if e.status in (400, 422) and opt_left:
                named = [c for c in opt_left if c in str(e)]
                drop = named or opt_left
                columns = [c for c in columns if c not in drop]
                removed += drop
                continue
            raise
        data = j["data"]
        total = j.get("totalCount", total)
        for it in data:
            s, d = (it.get("s"), it.get("d")) if isinstance(it, dict) else (None, None)
            if not isinstance(s, str) or not isinstance(d, list) or len(d) != len(columns):
                bad_rows += 1
                continue
            rec = dict(zip(columns, d))
            rec["tv_symbol"] = s
            rows.append(rec)
        pages += 1
        if len(data) < page:
            break
        start += page
        time.sleep(0.3)
    if not rows:
        raise TVError(None, "Boş yanıt: hiç sembol alınamadı")
    df = pd.DataFrame(rows)
    for c in TV_REQUIRED + TV_OPTIONAL:
        if c not in df.columns:
            df[c] = np.nan
    info = dict(source="TradingView tarayıcı uç noktası", total_reported=total, rows_fetched=len(rows),
                pages=pages, bad_rows=bad_rows, removed_columns=removed)
    return df, info


def classify_instrument(t: Any, sub: Any) -> str:
    t = str(t or "").lower()
    s = str(sub or "").lower()
    if t == "stock":
        if s == "common":
            return "Adi hisse"
        if s in ("", "nan", "none"):
            return "Adi hisse (alt tür doğrulanamadı)"
        if s == "preferred":
            return "İmtiyazlı hisse"
        return f"Diğer hisse ({s})"
    if t == "dr":
        return "ADR/DR"
    if t == "fund":
        return "Fon/ETF"
    return f"Diğer ({t or '?'})"


def map_yf_symbol(market_key: str, ticker: str) -> Tuple[Optional[str], str]:
    code = str(ticker or "").strip().upper()
    if not code or code == "NAN":
        return None, "boş sembol"
    if market_key == "BIST":
        code = re.sub(r"\.IS$", "", code)
        if not re.fullmatch(r"[A-Z0-9]{2,8}", code):
            return None, "BIST sembol biçimi doğrulanamadı"
        return code + ".IS", ""
    yf = code.replace(".", "-").replace("/", "-").replace(" ", "-")
    if not re.fullmatch(r"[A-Z0-9]{1,6}(-[A-Z0-9]{1,3})?", yf):
        return None, "ABD sembol biçimi doğrulanamadı"
    return yf, ""


def finalize_universe(df: pd.DataFrame, market_key: str, include_adr: bool, info: dict) -> Tuple[pd.DataFrame, dict]:
    d = df.copy()
    n0 = len(d)
    d = d.drop_duplicates("tv_symbol", keep="first").reset_index(drop=True)
    info["dup_tv_removed"] = n0 - len(d)
    d["ticker"] = d["name"].astype(str)
    d["company"] = d["description"].astype(str)
    d["instrument_class"] = [classify_instrument(t, s) for t, s in zip(d["type"], d["subtype"])]
    maps = [map_yf_symbol(market_key, t) for t in d["ticker"]]
    d["yf_symbol"] = [m[0] for m in maps]
    d["map_reason"] = [m[1] for m in maps]
    scope = d["instrument_class"].str.startswith("Adi hisse")
    if include_adr:
        scope = scope | (d["instrument_class"] == "ADR/DR")
    d["in_scope"] = scope
    n1 = len(d)
    ok = d["yf_symbol"].notna()
    dup_yf = ok & d.duplicated("yf_symbol", keep="first")
    d.loc[dup_yf, "map_reason"] = "yinelenen yfinance sembolü"
    d.loc[dup_yf, "yf_symbol"] = None
    info["dup_yf_removed"] = int(dup_yf.sum())
    info["class_counts"] = d["instrument_class"].value_counts().to_dict()
    info["in_scope"] = int(d["in_scope"].sum())
    info["map_failed"] = int((d["in_scope"] & d["yf_symbol"].isna()).sum())
    info["total_after_dedupe"] = n1
    info["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    info["market"] = market_key
    return d, info


def universe_from_csv(raw: bytes, market_key: str) -> Tuple[pd.DataFrame, dict]:
    import io
    df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", dtype=str)
    cols = {c.strip().lower(): c for c in df.columns}
    sym = next((cols[k] for k in ("symbol", "sembol", "ticker", "kod") if k in cols), None)
    if sym is None:
        raise ValueError("CSV'de 'symbol' (veya sembol/ticker/kod) sütunu bulunamadı.")
    nm = next((cols[k] for k in ("name", "company", "şirket", "sirket", "description") if k in cols), None)
    ex = next((cols[k] for k in ("exchange", "borsa") if k in cols), None)
    out = pd.DataFrame({
        "tv_symbol": df[sym].astype(str),
        "name": df[sym].astype(str).str.replace(r"^.*:", "", regex=True),
        "description": df[nm].astype(str) if nm else df[sym].astype(str),
        "exchange": df[ex].astype(str) if ex else ("BIST" if market_key == "BIST" else ""),
        "type": "stock", "subtype": "common", "close": np.nan, "volume": np.nan,
        "market_cap_basic": np.nan, "sector": np.nan,
    })
    info = dict(source="Kullanıcı CSV", total_reported=len(out), rows_fetched=len(out), pages=1,
                bad_rows=0, removed_columns=[])
    return out, info


def _universe_path(market_key: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"mtf_scanner_universe_{market_key}.json")


def save_universe(df: pd.DataFrame, info: dict, market_key: str) -> None:
    try:
        with open(_universe_path(market_key), "w", encoding="utf-8") as f:
            json.dump({"info": info, "rows": json.loads(df.to_json(orient="records"))}, f, ensure_ascii=False)
    except Exception:
        pass   # kalıcılık garanti değil


def load_saved_universe(market_key: str) -> Optional[Tuple[pd.DataFrame, dict]]:
    try:
        with open(_universe_path(market_key), "r", encoding="utf-8") as f:
            j = json.load(f)
        return pd.DataFrame(j["rows"]), j["info"]
    except Exception:
        return None


def load_universe(market_key: str, source: str, include_adr: bool, csv_bytes: Optional[bytes]) -> Tuple[pd.DataFrame, dict]:
    notes = []
    if source == "TradingView (otomatik)":
        try:
            raw, info = tv_fetch_universe(market_key, include_adr)
            df, info = finalize_universe(raw, market_key, include_adr, info)
            save_universe(df, info, market_key)
            return df, info
        except Exception as e:
            notes.append(f"TradingView başarısız: {e}")
            saved = load_saved_universe(market_key)
            if saved is None:
                raise RuntimeError("; ".join(notes) + " — Kayıtlı evren yok. Sembol CSV'si yükleyin.")
            df, info = saved
            info["note"] = "; ".join(notes) + f" — kayıtlı evren kullanıldı (kayıt zamanı {info.get('fetched_at')})"
            # kayıtlı evrende kapsam, güncel ADR seçimine göre yeniden hesaplanır
            sc = df["instrument_class"].str.startswith("Adi hisse")
            df["in_scope"] = sc | ((df["instrument_class"] == "ADR/DR") if include_adr else False)
            info["in_scope"] = int(df["in_scope"].sum())
            return df, info
    if source == "CSV yükle":
        if not csv_bytes:
            raise RuntimeError("Önce bir CSV dosyası yükleyin.")
        raw, info = universe_from_csv(csv_bytes, market_key)
        return finalize_universe(raw, market_key, include_adr, info)
    saved = load_saved_universe(market_key)
    if saved is None:
        raise RuntimeError("Kayıtlı evren bulunamadı.")
    df, info = saved
    info["note"] = f"Kayıtlı evren (kayıt zamanı {info.get('fetched_at')}); güncel olmayabilir."
    return df, info


# ======================================================================================
# 3. VERİ (yfinance)
# ======================================================================================
def split_download(raw: Optional[pd.DataFrame], symbols: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
    """Tek sembol / çoklu sembol / MultiIndex (her iki düzey sırası) çıktılarını sembol->DataFrame'e çevirir."""
    out: Dict[str, Optional[pd.DataFrame]] = {s: None for s in symbols}
    if raw is None or len(raw) == 0:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        lvl = None
        for L in range(raw.columns.nlevels):
            if set(map(str, symbols)) & set(map(str, raw.columns.get_level_values(L))):
                lvl = L
                break
        if lvl is None:
            return out
        present = set(raw.columns.get_level_values(lvl))
        for s in symbols:
            if s in present:
                sub = raw.xs(s, axis=1, level=lvl).dropna(how="all")
                out[s] = sub if len(sub) else None
    elif len(symbols) == 1:
        sub = raw.dropna(how="all")
        out[symbols[0]] = sub if len(sub) else None
    return out


def _download_core(symbols: tuple, interval: str, period: str, threads: int):
    import yfinance as yf
    last: Optional[Exception] = None
    syms = list(symbols)
    for attempt in range(3):
        try:
            # auto_adjust=False: bölünme-düzeltmeli ham OHLC (temettü düzeltmesi yok); tüm zaman dilimlerinde aynı politika.
            raw = yf.download(syms, period=period, interval=interval, auto_adjust=False, group_by="ticker",
                              progress=False, threads=(threads if threads > 1 else False), timeout=30)
            frames = split_download(raw, syms)
            if all(v is None for v in frames.values()):
                raise RuntimeError("yfinance boş yanıt döndürdü (rate limit olabilir)")
            errs: Dict[str, str] = {}
            try:
                errs = {k: str(v) for k, v in getattr(yf.shared, "_ERRORS", {}).items() if k in symbols}
            except Exception:
                pass
            return frames, errs
        except Exception as e:  # noqa: BLE001
            last = e
            backoff_sleep(attempt, base=2.0)
    raise RuntimeError(f"yfinance başarısız: {last}")


@st.cache_data(ttl=3600, show_spinner=False)
def dl_daily(symbols: tuple, threads: int = 1):
    return _download_core(symbols, "1d", "2y", threads)


@st.cache_data(ttl=900, show_spinner=False)
def dl_hourly(symbols: tuple, threads: int = 1):
    return _download_core(symbols, "1h", "6mo", threads)


def clean_ohlcv(df: Optional[pd.DataFrame], tz: str, daily: bool) -> Tuple[pd.DataFrame, dict]:
    q = dict(raw=0, dup=0, nan=0, bad_price=0, zero_vol=0)
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]), q
    q["raw"] = len(df)
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in cols):
        return pd.DataFrame(columns=cols), q
    d = df[cols].apply(pd.to_numeric, errors="coerce")
    idx = pd.DatetimeIndex(d.index)
    if daily:
        if idx.tz is not None:
            idx = idx.tz_convert(tz).tz_localize(None)
        idx = idx.normalize()
    else:
        idx = idx.tz_localize("UTC") if idx.tz is None else idx
        idx = idx.tz_convert(tz)
    d.index = idx
    n = len(d)
    d = d.dropna()
    q["nan"] = n - len(d)
    d = d.sort_index()
    n = len(d)
    d = d[~d.index.duplicated(keep="last")]
    q["dup"] = n - len(d)
    tol = 1e-6
    bad = ((d[["Open", "High", "Low", "Close"]] <= 0).any(axis=1) | (d["High"] < d["Low"]) |
           (d["Close"] > d["High"] * (1 + tol)) | (d["Close"] < d["Low"] * (1 - tol)) |
           (d["Open"] > d["High"] * (1 + tol)) | (d["Open"] < d["Low"] * (1 - tol)) | (d["Volume"] < 0))
    q["bad_price"] = int(bad.sum())
    d = d[~bad]
    q["zero_vol"] = int((d["Volume"] == 0).sum())
    return d, q


# ======================================================================================
# 4. SEANS / MUM KAPANIŞI
# ======================================================================================
def build_schedule(mkt: dict) -> Tuple[pd.DataFrame, str]:
    """Seans takvimi: index = seans tarihi (tz'siz), open/close = UTC. exchange_calendars; yoksa sabit hafta içi yedek."""
    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar(mkt["cal"])
        sch = cal.schedule[["open", "close"]].copy()
        sch.index = pd.DatetimeIndex(sch.index).tz_localize(None) if pd.DatetimeIndex(sch.index).tz is not None \
            else pd.DatetimeIndex(sch.index)
        for c in ("open", "close"):
            sch[c] = pd.to_datetime(sch[c], utc=True)
        return sch, f"exchange_calendars ({mkt['cal']}): tatiller ve erken kapanışlar takvimden"
    except Exception as e:  # noqa: BLE001
        o, c = mkt["fallback_session"]
        days = pd.bdate_range(pd.Timestamp.now().normalize() - pd.Timedelta(days=1100),
                              pd.Timestamp.now().normalize() + pd.Timedelta(days=10))
        opens = [pd.Timestamp(datetime.combine(d.date(), o)).tz_localize(mkt["tz"]).tz_convert("UTC") for d in days]
        closes = [pd.Timestamp(datetime.combine(d.date(), c)).tz_localize(mkt["tz"]).tz_convert("UTC") for d in days]
        sch = pd.DataFrame({"open": opens, "close": closes}, index=days)
        return sch, (f"UYARI: takvim doğrulanamadı ({e}); hafta içi sabit seans varsayıldı. "
                     "Resmî tatiller ve erken kapanışlar BİLİNMİYOR.")


@st.cache_resource(show_spinner=False)
def get_schedule(market_key: str):
    return build_schedule(MARKETS[market_key])


def annotate_intraday(df: pd.DataFrame, sch: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """1s mumlara seans, dilim (slot), bitiş zamanı ekler. Seans dışı / hizasız mumları atar ve sayar."""
    stats = dict(inp=len(df), non_session=0, misaligned=0)
    if df.empty:
        return df.assign(slot=pd.Series(dtype=int)), stats
    sess = df.index.normalize().tz_localize(None)
    op = pd.to_datetime(sch["open"].reindex(sess).to_numpy(), utc=True)
    cl = pd.to_datetime(sch["close"].reindex(sess).to_numpy(), utc=True)
    st_utc = df.index.tz_convert("UTC")
    with np.errstate(invalid="ignore"):
        off = np.asarray((st_utc - op) / HOUR, dtype=float)
        length = np.asarray((cl - op) / HOUR, dtype=float)
        slot = np.rint(off)
        nonsess = np.isnan(off)
        aligned = np.abs(off - slot) < 1e-6
        inrange = (slot >= 0) & (slot < np.ceil(length - 1e-9))
    ok = (~nonsess) & aligned & inrange
    stats["non_session"] = int(nonsess.sum())
    stats["misaligned"] = int(((~nonsess) & ~(aligned & inrange)).sum())
    out = df[ok].copy()
    if out.empty:
        out["slot"] = pd.Series(dtype=int)
        return out, stats
    e1 = st_utc[ok] + HOUR
    end = e1.where(e1 <= cl[ok], cl[ok])
    out["slot"] = slot[ok].astype(int)
    out["session"] = sess[ok]
    out["open_ts"] = pd.Series(op[ok], index=out.index)
    out["close_ts"] = pd.Series(cl[ok], index=out.index)
    out["end"] = pd.Series(end, index=out.index)
    return out, stats


def annotate_daily(df: pd.DataFrame, sch: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    stats = dict(inp=len(df), non_session=0)
    if df.empty:
        return df.assign(end=pd.Series(dtype="datetime64[ns, UTC]")), stats
    cl = pd.to_datetime(sch["close"].reindex(df.index).to_numpy(), utc=True)
    ok = ~cl.isna()
    stats["non_session"] = int((~ok).sum())
    out = df[ok].copy()
    out["end"] = pd.Series(cl[ok], index=out.index)
    return out, stats


def only_completed(df: pd.DataFrame, now: pd.Timestamp, tol_min: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["end"] + pd.Timedelta(minutes=tol_min) <= now]


def build_htf(ann: pd.DataFrame, n: int, include_short: bool) -> pd.DataFrame:
    """1s mumlardan n saatlik mum: seans açılışına sabitli, gün/seans sınırı birleştirmez, eksik alt mum => geçersiz."""
    cols = ["Open", "High", "Low", "Close", "Volume", "end", "short", "slot"]
    if ann.empty:
        return pd.DataFrame(columns=cols)
    a = ann.copy()
    a["bucket"] = a["slot"] // n
    a["_start"] = a.index
    a["nslots"] = np.ceil(np.asarray((a["close_ts"] - a["open_ts"]) / HOUR, dtype=float) - 1e-9).astype(int)
    g = a.groupby(["session", "bucket"], sort=True)
    agg = g.agg(Open=("Open", "first"), High=("High", "max"), Low=("Low", "min"), Close=("Close", "last"),
                Volume=("Volume", "sum"), cnt=("slot", "size"), start=("_start", "min"),
                nslots=("nslots", "first"), open_ts=("open_ts", "first"), close_ts=("close_ts", "first"))
    b = agg.index.get_level_values("bucket").to_numpy()
    exp = np.minimum((b + 1) * n, agg["nslots"].to_numpy()) - b * n
    agg = agg[agg["cnt"].to_numpy() == exp].copy()
    b = agg.index.get_level_values("bucket").to_numpy()
    if agg.empty:
        return pd.DataFrame(columns=cols)
    nominal_end = agg["open_ts"] + pd.to_timedelta((b + 1) * n, unit="h")
    end = nominal_end.where(nominal_end <= agg["close_ts"], agg["close_ts"])
    start_utc = agg["start"].dt.tz_convert("UTC") if hasattr(agg["start"], "dt") else agg["start"]
    short = (end - start_utc) < (pd.Timedelta(hours=n) - EPS)
    res = agg[["Open", "High", "Low", "Close", "Volume"]].copy()
    res["end"] = end
    res["short"] = short
    res["slot"] = b
    res.index = pd.DatetimeIndex(agg["start"])
    if not include_short:
        res = res[~res["short"]]
    return res[cols]


def expected_last_end(sch: pd.DataFrame, now: pd.Timestamp, tol_min: int) -> Optional[pd.Timestamp]:
    """Takvime göre, şu ana kadar kapanmış olması gereken son 1s mumun bitişi."""
    cutoff = now - pd.Timedelta(minutes=tol_min)
    past = sch[sch["open"] <= cutoff]
    for sess in list(past.index[::-1][:3]):
        o, c = sch.at[sess, "open"], sch.at[sess, "close"]
        n = int(math.ceil((c - o) / HOUR - 1e-9))
        ends = [min(o + pd.Timedelta(hours=k + 1), c) for k in range(n)]
        okk = [e for e in ends if e <= cutoff]
        if okk:
            return max(okk)
    return None


def market_status(sch: pd.DataFrame, now: pd.Timestamp):
    cur = sch[(sch["open"] <= now) & (sch["close"] > now)]
    if len(cur):
        return "Seans açık", cur["close"].iloc[0]
    nxt = sch[sch["open"] > now]
    return "Seans kapalı", (nxt["open"].iloc[0] if len(nxt) else None)


# ======================================================================================
# 5. İNDİKATÖRLER
# ======================================================================================
def wilder(s: pd.Series, n: int) -> pd.Series:
    """Wilder yumuşatması: ilk n geçerli değerin SMA'sı ile tohumlanır, sonra alpha=1/n özyinelemesi."""
    v = s.astype(float).to_numpy()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    valid = ~np.isnan(v)
    if valid.sum() < n:
        return out
    first = int(np.argmax(valid))
    if first + n > len(v) or np.isnan(v[first:first + n]).any():
        return out
    seg = pd.Series(v[first + n - 1:], dtype=float)
    seg.iloc[0] = v[first:first + n].mean()
    out.iloc[first + n - 1:] = seg.ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()
    return out


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    dlt = close.diff()
    ag = wilder(dlt.clip(lower=0), n)
    al = wilder((-dlt).clip(lower=0), n)
    rs = ag / al.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    out = out.where(~((al == 0) & (ag > 0)), 100.0)
    out = out.where(~((al == 0) & (ag == 0)), 50.0)
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l, v = d["Close"], d["High"], d["Low"], d["Volume"]
    d["sma20"], d["sma50"], d["sma200"] = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    d["ema20"], d["ema50"] = ema(c, 20), ema(c, 50)
    d["rsi14"] = rsi(c, 14)
    d["macd"] = ema(c, 12) - ema(c, 26)
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1).where(pc.notna())
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index).where(pc.notna())
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index).where(pc.notna())
    atr = wilder(tr, 14)
    d["atr14"] = atr
    d["atr_pct"] = 100 * atr / c
    d["pdi"] = 100 * wilder(pdm, 14) / atr
    d["mdi"] = 100 * wilder(mdm, 14) / atr
    dx = 100 * (d["pdi"] - d["mdi"]).abs() / (d["pdi"] + d["mdi"]).replace(0, np.nan)
    d["adx"] = wilder(dx, 14)
    m, sd = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    d["bb_mid"], d["bb_up"], d["bb_lo"] = m, m + 2 * sd, m - 2 * sd
    d["bb_width"] = (d["bb_up"] - d["bb_lo"]) / m
    d["vol_avg20"] = v.rolling(20).mean().shift(1)             # değerlendirilen mum ortalamadan çıkarılır
    d["relvol_simple"] = (v / d["vol_avg20"]).replace([np.inf, -np.inf], np.nan)
    d["hh20"] = h.shift(1).rolling(20).max()                   # önceki 20 tamamlanmış mum (mevcut hariç)
    d["ll20"] = l.shift(1).rolling(20).min()
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    d["obv"] = obv
    d["obv_slope"] = (obv - obv.shift(10)) / v.rolling(10).sum().replace(0, np.nan)   # [-1,1]: 10 mumluk OBV değişimi / hacim toplamı
    return d


def add_relvol_slot(d: pd.DataFrame, sessions: int = 20, min_sessions: int = 5) -> pd.DataFrame:
    """Aynı saat dilimindeki (slot) önceki seans hacimlerinin ortalamasına oran."""
    out = d.copy()
    if "slot" not in out.columns or out.empty:
        out["relvol_slot"] = np.nan
        return out
    base = out.groupby("slot")["Volume"].transform(
        lambda s: s.shift(1).rolling(sessions, min_periods=min_sessions).mean())
    out["relvol_slot"] = (out["Volume"] / base).replace([np.inf, -np.inf], np.nan)
    return out


def prepare_frames(daily_raw, hourly_raw, sch, mkt: dict, cfg: Cfg, now: pd.Timestamp):
    tz = mkt["tz"]
    dclean, qd = clean_ohlcv(daily_raw, tz, True)
    hclean, qh = clean_ohlcv(hourly_raw, tz, False)
    dd, sd = annotate_daily(dclean, sch)
    dd = only_completed(dd, now, cfg.tol_min)
    ann, sh = annotate_intraday(hclean, sch)
    f1 = only_completed(ann, now, cfg.tol_min).copy()
    if len(f1):
        f1["short"] = (f1["end"] - f1.index.tz_convert("UTC")) < (HOUR - EPS)
        if not cfg.include_short:
            f1 = f1[~f1["short"]]
    f2 = only_completed(build_htf(ann, 2, cfg.include_short), now, cfg.tol_min)
    f4 = only_completed(build_htf(ann, 4, cfg.include_short), now, cfg.tol_min)
    ref_end = f1["end"].max() if len(f1) else None
    if ref_end is not None:   # üst zaman dilimi, 1s referans anından sonra kapanamaz (look-ahead yok)
        dd, f2, f4 = dd[dd["end"] <= ref_end], f2[f2["end"] <= ref_end], f4[f4["end"] <= ref_end]
    fr = {"1d": add_indicators(dd) if len(dd) else dd, "4h": add_indicators(f4) if len(f4) else f4,
          "2h": add_indicators(f2) if len(f2) else f2}
    if len(f1):
        f1 = add_relvol_slot(add_indicators(f1), 20)
    fr["1h"] = f1
    quality = dict(daily=qd, hourly=qh, daily_ann=sd, hourly_ann=sh, ref_end=ref_end)
    return fr, quality


# ======================================================================================
# 6. STRATEJİ, SKOR, RİSK
# ======================================================================================
def position_size(portfolio: float, cash: float, risk_pct: float, max_weight_pct: float,
                  entry: float, stop: float) -> Optional[dict]:
    rps = entry - stop
    if not (entry > 0 and stop > 0 and rps > 0):
        return None
    by_risk = math.floor(portfolio * risk_pct / 100.0 / rps)
    by_weight = math.floor(portfolio * max_weight_pct / 100.0 / entry)
    by_cash = math.floor(cash / entry)
    cands = {"risk bütçesi": by_risk, "maks. pozisyon ağırlığı": by_weight, "kullanılabilir nakit": by_cash}
    bind = min(cands, key=cands.get)
    shares = max(0, cands[bind])
    return dict(risk_per_share=rps, shares=shares, binding=bind, position_value=shares * entry,
                risk_amount=shares * rps, weight_pct=(shares * entry / portfolio * 100.0) if portfolio > 0 else float("nan"))


def trailing_run(b: pd.Series) -> int:
    """Sondaki ardışık True sayısı (1 = tetikleyici son tamamlanmış mumda oluştu)."""
    n = 0
    for x in reversed(b.fillna(False).astype(bool).tolist()):
        if x:
            n += 1
        else:
            break
    return n


def _cmp(a, b, op):
    if isnan(a) or isnan(b):
        return None
    return bool(op(a, b))


def _all(*xs):
    if any(x is None for x in xs):
        return None
    return all(xs)


def find_resistance(entry: float, f4: pd.DataFrame, fd: pd.DataFrame) -> Tuple[Optional[float], str]:
    cands = []
    if len(f4) >= 20:
        cands.append((float(f4["High"].tail(20).max()), "4s son 20 mum tepesi"))
    if len(fd) >= 60:
        cands.append((float(fd["High"].tail(60).max()), "1G son 60 gün tepesi"))
    above = [c for c in cands if c[0] > entry]
    if not above:
        return None, ""
    return min(above, key=lambda x: x[0])


def base_row(meta: dict, cfg: Cfg) -> dict:
    mkt = MARKETS[cfg.market]
    return dict(Sembol=meta.get("ticker", meta["yf_symbol"]), yf_symbol=meta["yf_symbol"], Şirket=meta.get("company", ""),
                Piyasa=cfg.market, ParaBirimi=mkt["currency"], Sektör=meta.get("sector") if not isnan(meta.get("sector")) else "Bilinmiyor",
                Sınıf=CLS_DATA, Kurulum="—", Skor=np.nan, SkorMaks=np.nan, Teyitler="", SonKapanış=np.nan, KapanışZamanı="",
                RSI_2s=np.nan, MACDHist_2s=np.nan, ADX_4s=np.nan, GöreliHacim_1s=np.nan, HacimTürü="", ATRpct_1s=np.nan,
                ATRpct_1G=np.nan, Giriş=np.nan, Stop=np.nan, Hedef_2R=np.nan, Hedef_3R=np.nan, Direnç=np.nan,
                RR_Direnç=np.nan, Adet=np.nan, PozisyonDeğeri=np.nan, SinyalYaşı=np.nan, Gerekçe="", detay={})


def evaluate(meta: dict, fr: Dict[str, pd.DataFrame], quality: dict, ctx: dict, cfg: Cfg) -> dict:
    row = base_row(meta, cfg)
    mkt = MARKETS[cfg.market]
    row["VeriÇekimi"] = ctx.get("fetched_at", "")
    detail: Dict[str, Any] = dict(quality={k: v for k, v in quality.items() if k != "ref_end"}, pos=[], neg=[], risks=[],
                                  comps=[], gaps=[], plan=None)
    row["detay"] = detail
    # ---- veri yeterliliği
    gaps = []
    for tf in TF_ORDER:
        n = len(fr[tf])
        if n < MIN_BARS[tf]:
            gaps.append(f"{TF_LABEL[tf]}: {n} tamamlanmış mum var, en az {MIN_BARS[tf]} gerekli (yetersiz geçmiş; kısa veriyle uzun gösterge taklit edilmez)")
    if len(fr["1h"]):
        last_end = fr["1h"]["end"].iloc[-1]
        row["SonKapanış"] = float(fr["1h"]["Close"].iloc[-1])
        row["KapanışZamanı"] = last_end.tz_convert(mkt["tz"]).strftime("%Y-%m-%d %H:%M")
        exp = ctx.get("exp_end")
        if cfg.reject_stale and exp is not None and last_end < exp:
            gaps.append(f"1s veri güncel değil: son kapanmış mum {last_end.tz_convert(mkt['tz']):%Y-%m-%d %H:%M}, "
                        f"beklenen {exp.tz_convert(mkt['tz']):%Y-%m-%d %H:%M} (sağlayıcı gecikmesi veya eksik mum)")
    if gaps:
        detail["gaps"] = gaps
        row["Gerekçe"] = "Veri yetersiz: " + "; ".join(gaps)[:300]
        return row
    D, H4, H2, H1 = fr["1d"], fr["4h"], fr["2h"], fr["1h"]
    d, h4, h2, h1 = D.iloc[-1], H4.iloc[-1], H2.iloc[-1], H1.iloc[-1]
    comps: List[dict] = []

    def add(tf, name, ok, pts, text, grp, enabled=True):
        comps.append(dict(tf=tf, name=name, ok=ok, pts=pts, text=text, grp=grp, enabled=enabled))

    # ---- B. günlük ana trend (25)
    add("1d", "Kapanış > SMA50", _cmp(d.Close, d.sma50, lambda a, b: a > b), 10, f"{fmt(d.Close)} / SMA50 {fmt(d.sma50)}", "setup")
    add("1d", "SMA50 > SMA200", _cmp(d.sma50, d.sma200, lambda a, b: a > b), 10, f"SMA50 {fmt(d.sma50)} / SMA200 {fmt(d.sma200)}", "setup")
    add("1d", "RSI14 > 50", _cmp(d.rsi14, 50.0, lambda a, b: a > b), 5, f"RSI {fmt(d.rsi14, 1)}", "setup")
    # ---- C. 4s kurulum (20)
    add("4h", "Kapanış > EMA20 > EMA50", _all(_cmp(h4.Close, h4.ema20, lambda a, b: a > b), _cmp(h4.ema20, h4.ema50, lambda a, b: a > b)),
        8, f"{fmt(h4.Close)} / EMA20 {fmt(h4.ema20)} / EMA50 {fmt(h4.ema50)}", "setup")
    add("4h", f"ADX14 ≥ {cfg.adx_min:g} ve +DI > -DI", _all(_cmp(h4.adx, cfg.adx_min, lambda a, b: a >= b), _cmp(h4.pdi, h4.mdi, lambda a, b: a > b)),
        8, f"ADX {fmt(h4.adx, 1)} / +DI {fmt(h4.pdi, 1)} / -DI {fmt(h4.mdi, 1)}", "setup")
    ext = (h4.Close - h4.ema20) / h4.atr14 if not (isnan(h4.atr14) or h4.atr14 == 0) else float("nan")
    add("4h", f"EMA20'den uzaklık ≤ {cfg.max_ext_atr:g} ATR", _cmp(ext, cfg.max_ext_atr, lambda a, b: a <= b), 4, f"{fmt(ext)} ATR", "entry")
    # ---- D. 2s momentum (15)
    add("2h", "MACD > sinyal", _cmp(h2.macd, h2.macd_sig, lambda a, b: a > b), 6, f"MACD {fmt(h2.macd, 3)} / sinyal {fmt(h2.macd_sig, 3)}", "entry")
    hs = H2["macd_hist"].tail(3).to_numpy()
    imp = None if (len(hs) < 3 or np.isnan(hs).any()) else bool(hs[2] > hs[1] > hs[0])
    add("2h", "Histogram son 3 mumda yükseliyor", imp, 5, "son 3 hist: " + ", ".join(fmt(x, 3) for x in hs), "score")
    add("2h", f"RSI14 {cfg.rsi2_lo:g}–{cfg.rsi2_hi:g} aralığında",
        _all(_cmp(h2.rsi14, cfg.rsi2_lo, lambda a, b: a >= b), _cmp(h2.rsi14, cfg.rsi2_hi, lambda a, b: a <= b)), 4, f"RSI {fmt(h2.rsi14, 1)}", "entry")
    # ---- E. 1s tetikleyici (15)
    rv = H1["relvol_slot"].fillna(H1["relvol_simple"])
    rv_kind = "aynı saat dilimi" if not isnan(h1.relvol_slot) else "basit oran (yeterli seans geçmişi yok)"
    brk = (H1["Close"] > H1["hh20"]) & (rv >= cfg.rv_thr)
    touch3 = (H1["Low"] <= H1["ema20"]).astype(float).rolling(3).max() > 0
    pb = touch3 & (H1["Close"] > H1["ema20"]) & (H1["Close"] > H1["Open"]) & (H1["ema20"] > H1["ema50"])
    allow_b = cfg.trigger_mode in ("Kırılım ve geri çekilme", "Yalnızca kırılım")
    allow_p = cfg.trigger_mode in ("Kırılım ve geri çekilme", "Yalnızca geri çekilme")
    age_b, age_p = trailing_run(brk), trailing_run(pb)
    data_ok_b = not (isnan(h1.hh20) or isnan(rv.iloc[-1]))
    data_ok_p = not (isnan(h1.ema20) or isnan(h1.ema50))
    active, trig_names, age = [], [], 0
    if allow_b and age_b > 0:
        active.append(("Kırılım", age_b))
    if allow_p and age_p > 0:
        active.append(("Geri çekilme", age_p))
    if active:
        age = min(a for _, a in active)
        trig_names = [n for n, _ in active]
    fresh = bool(active) and age <= cfg.max_age
    miss = (allow_b and not data_ok_b and not active) or (allow_p and not data_ok_p and not active)
    trig_ok = None if (miss and not active) else fresh
    add("1h", "Tetikleyici: " + (" + ".join(trig_names) if trig_names else "yok"), trig_ok, 15,
        f"Kırılım: kapanış>{fmt(h1.hh20)} (önceki 20 mum tepesi) ve hacim≥{cfg.rv_thr:g}× [{fmt(h1.Close)}, hacim {fmt(rv.iloc[-1])}× {rv_kind}] yaş={age_b or '—'}; "
        f"Geri çekilme: son 3 mumda EMA20 teması + EMA20 üstü kapanış + yeşil mum + EMA20>EMA50 [EMA20 {fmt(h1.ema20)}] yaş={age_p or '—'}", "entry")
    if active and not fresh:
        comps[-1]["pts"] = 15
        comps[-1]["partial"] = 7   # eski tetikleyici kısmi puan
    # ---- F. hacim ve likidite (15)
    val20 = float((D["Close"] * D["Volume"]).tail(20).mean()) if len(D) >= 20 else float("nan")
    price = float(h1.Close)
    add("1d", f"Likidite derinliği (≥ 3× asgari işlem tutarı; yaklaşık)", _cmp(val20, 3 * cfg.min_value, lambda a, b: a >= b), 5,
        f"~{fmt(val20, 0)} {mkt['currency']}/gün (OHLCV'den tahmin)", "score")
    add("1d", "OBV eğimi (10 mum) > 0", _cmp(d.obv_slope, 0.0, lambda a, b: a > b), 5, f"eğim {fmt(d.obv_slope, 3)}", "score")
    add("1d", "Günlük göreli hacim ≥ 1.0 (basit oran)", _cmp(d.relvol_simple, 1.0, lambda a, b: a >= b), 5, f"{fmt(d.relvol_simple)}×", "score")
    # ---- A. piyasa rejimi ve göreli güç (10)
    reg, rs = ctx.get("regime"), None
    on = cfg.regime_mode != "Kapalı"
    add("1d", f"Piyasa rejimi olumlu ({cfg.benchmark})", reg, 5, ctx.get("regime_text", "referans verisi yok"), "regime", enabled=on)
    rs63 = float("nan")
    bench_c = ctx.get("bench_close")
    if bench_c is not None and len(D) > 64:
        common = D.index.intersection(bench_c.index)
        if len(common) >= 64:
            sc, bc = D["Close"].loc[common], bench_c.loc[common]
            rs63 = float((sc.iloc[-1] / sc.iloc[-64] - 1) - (bc.iloc[-1] / bc.iloc[-64] - 1))
    add("1d", f"63G göreli güç > 0 ({cfg.benchmark}'e göre getiri farkı)", _cmp(rs63, 0.0, lambda a, b: a > b), 5,
        f"{fmt(rs63 * 100 if not isnan(rs63) else rs63)} puan", "rs", enabled=on)
    detail["comps"] = comps

    # ---- skor
    def earned(c):
        if not c["enabled"] or c["ok"] is None:
            return 0
        return c["pts"] if c["ok"] else c.get("partial", 0) if c["name"].startswith("Tetikleyici") else 0
    score = sum(earned(c) for c in comps)
    smax = sum(c["pts"] for c in comps if c["enabled"] and c["ok"] is not None)
    row.update(Skor=score, SkorMaks=smax)
    def tfok(tf, grp=("setup", "entry")):
        xs = [c["ok"] for c in comps if c["tf"] == tf and c["grp"] in grp]
        return _all(*xs) if xs else None
    flags = {"1d": tfok("1d", ("setup",)), "4h": tfok("4h"), "2h": tfok("2h", ("entry",)), "1h": trig_ok}
    row["Teyitler"] = " ".join(f"{TF_LABEL[t]}{'✓' if flags[t] else ('?' if flags[t] is None else '✗')}" for t in TF_ORDER)

    row.update(RSI_2s=float(h2.rsi14), MACDHist_2s=float(h2.macd_hist), ADX_4s=float(h4.adx), GöreliHacim_1s=float(rv.iloc[-1]) if not isnan(rv.iloc[-1]) else np.nan,
               HacimTürü=rv_kind, ATRpct_1s=float(h1.atr_pct), ATRpct_1G=float(d.atr_pct), SinyalYaşı=age if age else np.nan)
    row["Kurulum"] = " + ".join(trig_names) if trig_names else "—"

    # ---- likidite / risk filtresi (kapı)
    liq = [("Fiyat ≥ asgari", _cmp(price, cfg.min_price, lambda a, b: a >= b), f"{fmt(price)} (asgari {cfg.min_price:g})"),
           ("Ort. günlük işlem tutarı ≥ asgari (yaklaşık)", _cmp(val20, cfg.min_value, lambda a, b: a >= b), f"~{fmt(val20, 0)} (asgari {cfg.min_value:,.0f} {mkt['currency']})"),
           ("Günlük ATR% ≤ azami", _cmp(d.atr_pct, cfg.max_atr_pct, lambda a, b: a <= b), f"{fmt(d.atr_pct)}% (azami {cfg.max_atr_pct:g}%)")]
    pos, neg, risks, gap2 = detail["pos"], detail["neg"], detail["risks"], []
    for n_, ok_, t_ in liq:
        (pos if ok_ else neg if ok_ is False else gap2).append(f"{n_}: {t_}")
    crit_gaps = [f"{c['tf']} · {c['name']}" for c in comps if c["grp"] in ("setup", "entry") and c["ok"] is None]
    if cfg.regime_mode == "Zorunlu filtre" and reg is None:
        crit_gaps.append("piyasa rejimi (referans verisi yok)")
    crit_gaps += gap2
    for c in comps:
        if c["enabled"] and c["ok"] is None and c["grp"] in ("regime", "rs", "score"):
            risks.append(f"Hesaplanamadı (puan 0 verildi): {c['name']}")
    if crit_gaps:
        detail["gaps"] = crit_gaps
        row["Sınıf"] = CLS_DATA
        row["Gerekçe"] = "Veri yetersiz: hesaplanamayan değerlendirme(ler): " + "; ".join(crit_gaps)[:300]
        return row
    for c in comps:
        if not c["enabled"] or c["grp"] in ("regime", "rs") and c["ok"] is None:
            continue
        line = f"[{TF_LABEL[c['tf']]}] {c['name']} — {c['text']}"
        if c["ok"]:
            pos.append(line)
        elif c["ok"] is False:
            neg.append(line)
    liq_ok = all(x[1] for x in liq)
    setup_ok = _all(*[c["ok"] for c in comps if c["grp"] == "setup"])
    regime_ok = True if cfg.regime_mode != "Zorunlu filtre" else bool(reg)
    entry_ok = _all(*[c["ok"] for c in comps if c["grp"] == "entry"])
    if cfg.regime_mode != "Kapalı" and reg is False:
        risks.append("Piyasa rejimi olumsuz: referans endeks SMA50/SMA200 altında.")
    if not liq_ok:
        row["Sınıf"] = CLS_NO
        row["Gerekçe"] = "Likidite/risk filtresi: " + "; ".join(n for n in neg if n.startswith(("Fiyat", "Ort.", "Günlük ATR")))[:250]
        return row
    if not regime_ok:
        row["Sınıf"] = CLS_NO
        row["Gerekçe"] = "Piyasa rejimi olumsuz (zorunlu filtre açık)."
        return row
    if not setup_ok:
        row["Sınıf"] = CLS_NO
        row["Gerekçe"] = "Günlük/4s trend koşulları sağlanmadı: " + "; ".join(n.split(" — ")[0] for n in neg if n.startswith(("[1G]", "[4s]")))[:250]
        return row

    # ---- varsayımsal işlem planı
    entry = price
    atr4 = float(h4.atr14)
    if cfg.stop_method == "ATR (4s)":
        stop = entry - cfg.atr_mult * atr4
        stop_txt = f"giriş − {cfg.atr_mult:g}×ATR14(4s) = {fmt(entry)} − {cfg.atr_mult:g}×{fmt(atr4)}"
    else:
        lo10 = float(H1["Low"].tail(10).min())
        stop = lo10 - 0.1 * float(h1.atr14)
        stop_txt = f"son 10 adet 1s mum dibi ({fmt(lo10)}) − 0.1×ATR14(1s)"
    plan = None
    res, res_src = find_resistance(entry, H4, D)
    plan_ok = False
    if stop > 0 and entry > stop:
        risk_ps = entry - stop
        ps = position_size(cfg.portfolio, cfg.cash, cfg.risk_pct, cfg.max_weight, entry, stop)
        plan = dict(entry=entry, entry_txt="Referans: son tamamlanmış 1s kapanışı (gerçekleşmiş fiyat DEĞİL)", stop=stop, stop_txt=stop_txt,
                    risk_ps=risk_ps, risk_pct_price=100 * risk_ps / entry, t2=entry + 2 * risk_ps, t3=entry + 3 * risk_ps,
                    resistance=res, res_src=res_src, rr_res=((res - entry) / risk_ps) if res else float("nan"), size=ps)
        if res is None:
            rr_ok = not cfg.need_res
            risks.append("Bağımsız direnç bulunamadı (pencere içinde üstte tepe yok): R/R bağımsız olarak doğrulanamadı; 2R/3R yalnızca varsayımsal hedeftir.")
            if cfg.need_res:
                neg.append("Bağımsız dirence göre R/R doğrulanamadı (ayar: direnç zorunlu).")
        else:
            rr_ok = plan["rr_res"] >= cfg.min_rr
            (pos if rr_ok else neg).append(f"Dirence göre R/R {fmt(plan['rr_res'])} (direnç {fmt(res)}, kaynak: {res_src}; asgari {cfg.min_rr:g})")
        plan_ok = rr_ok
        row.update(Giriş=entry, Stop=stop, Hedef_2R=plan["t2"], Hedef_3R=plan["t3"], Direnç=res if res else np.nan,
                   RR_Direnç=plan["rr_res"], Adet=ps["shares"], PozisyonDeğeri=ps["position_value"])
        if ps["shares"] < 1:
            risks.append(f"Hesaplanan adet 0 ({ps['binding']} sınırlıyor): portföy/risk ayarlarıyla bu stop mesafesi uyumsuz.")
    else:
        neg.append("Stop ≥ giriş veya stop ≤ 0: işlem planı oluşturulamadı.")
    detail["plan"] = plan
    # ---- risk notları
    risks.append("Stop emirleri gap/likidite nedeniyle belirtilen fiyattan gerçekleşmeyebilir.")
    risks.append("Spread/komisyon verisi yok; işlem maliyeti hesaba katılmadı. Backtest yapılmadı.")
    if float(d.atr_pct) > 0.75 * cfg.max_atr_pct:
        risks.append(f"Günlük ATR% yüksek ({fmt(d.atr_pct)}%).")
    if val20 < 2 * cfg.min_value:
        risks.append("Ortalama işlem tutarı asgari eşiğe yakın (düşük likidite marjı).")
    if bool(h1.get("short", False)):
        risks.append("Son 1s mum seans sonu kısa mumudur.")
    if "basit" in rv_kind:
        risks.append("Göreli hacim aynı saat dilimine göre hesaplanamadı; basit oran kullanıldı.")
    if active and not fresh:
        risks.append(f"Tetikleyici {age} mumdur sürüyor (azami {cfg.max_age}); giriş gecikmiş olabilir.")

    if entry_ok and plan_ok and plan is not None and score >= cfg.min_score and plan["size"] is not None:
        row["Sınıf"] = CLS_CAND
    else:
        row["Sınıf"] = CLS_WATCH
        if entry_ok and plan_ok and score < cfg.min_score:
            neg.append(f"Teknik skor {score} < asgari {cfg.min_score}")
    row["Gerekçe"] = ("; ".join(c["name"].split(" (")[0] for c in comps if c["ok"] and c["grp"] in ("setup", "entry"))[:140]
                      or "—") + ((" | Eksik: " + "; ".join(n.split(" — ")[0] for n in neg)[:110]) if row["Sınıf"] == CLS_WATCH else "")
    if row["Sınıf"] != CLS_CAND and plan is not None and not entry_ok:
        row["Kurulum"] = "Tetikleyici bekleniyor" if not trig_names else row["Kurulum"]
    return row


# ======================================================================================
# 7. TARAMA AKIŞI
# ======================================================================================
def build_ctx(scan: dict, cfg: Cfg) -> dict:
    mkt = MARKETS[cfg.market]
    sch, note = get_schedule(cfg.market)
    now = pd.Timestamp(scan["now"])
    ctx = dict(sch=sch, cal_note=note, now=now, exp_end=expected_last_end(sch, now, cfg.tol_min),
               regime=None, regime_text="referans verisi alınamadı", bench_close=None, fetched_at="")
    try:
        frames, errs = dl_daily((cfg.benchmark,), 1)
        raw = frames.get(cfg.benchmark)
        dcl, _ = clean_ohlcv(raw, mkt["tz"], True)
        dd, _ = annotate_daily(dcl, sch)
        dd = only_completed(dd, now, cfg.tol_min)
        if len(dd) >= 210:
            sma50, sma200 = dd["Close"].rolling(50).mean().iloc[-1], dd["Close"].rolling(200).mean().iloc[-1]
            c = dd["Close"].iloc[-1]
            ctx["regime"] = bool(c > sma50 and c > sma200)
            ctx["regime_text"] = (f"{cfg.benchmark} kapanış {fmt(c)} | SMA50 {fmt(sma50)} | SMA200 {fmt(sma200)} "
                                  f"(son günlük kapanış {dd.index[-1]:%Y-%m-%d})")
            ctx["bench_close"] = dd["Close"]
        else:
            ctx["regime_text"] = f"{cfg.benchmark}: yetersiz günlük geçmiş ({len(dd)} mum)"
    except Exception as e:  # noqa: BLE001
        ctx["regime_text"] = f"{cfg.benchmark} verisi alınamadı: {str(e)[:120]}"
    return ctx


def make_scan(universe: pd.DataFrame, cfg: Cfg, mode: str, fast_n: int, batch_size: int, threads: int, pause: float) -> dict:
    u = universe[universe["in_scope"] & universe["yf_symbol"].notna()].copy()
    note = ""
    if mode == "Hızlı tarama":
        has_tv = u["close"].notna().any()
        if has_tv:
            u = u[(u["close"].isna()) | (u["close"] >= cfg.min_price)]
            if u["market_cap_basic"].notna().any():
                u = u.sort_values("market_cap_basic", ascending=False, na_position="last")
            u = u.head(int(fast_n))
            note = f"Ön filtre: TradingView fiyatı ≥ {cfg.min_price:g}, piyasa değerine göre ilk {int(fast_n)} sembol."
        else:
            note = "Ön filtre uygulanamadı (evrende TradingView fiyat/piyasa değeri yok): evrenin tamamı taranacak."
    metas = u.to_dict(orient="records")
    return dict(cfg=cfg, mode=mode, symbols=metas, pos=0, results={}, failures={}, running=True, finished=False,
                started=time.time(), elapsed=0.0, now=str(pd.Timestamp.now(tz="UTC")), batch_size=batch_size, threads=threads,
                pause=pause, ctx=None, note=note, errors=[], total_in_scope=int(universe["in_scope"].sum()),
                map_failed=int((universe["in_scope"] & universe["yf_symbol"].isna()).sum()))


def run_batch(scan: dict) -> None:
    cfg: Cfg = scan["cfg"]
    t0 = time.time()
    if scan["ctx"] is None:
        scan["ctx"] = build_ctx(scan, cfg)
    ctx = scan["ctx"]
    i0 = scan["pos"]
    batch = scan["symbols"][i0:i0 + scan["batch_size"]]
    syms = tuple(m["yf_symbol"] for m in batch)
    try:
        dframes, derr = dl_daily(syms, scan["threads"])
        hframes, herr = dl_hourly(syms, scan["threads"])
    except Exception as e:  # noqa: BLE001
        for m in batch:
            scan["failures"][m["yf_symbol"]] = f"indirme başarısız: {str(e)[:150]}"
        scan["errors"].append(f"Grup {i0}-{i0 + len(batch)}: {str(e)[:200]}")
        scan["pos"] = i0 + len(batch)
        scan["pause_extra"] = min(30.0, scan.get("pause_extra", 0) * 2 + 2.0)   # hız azaltma
        return
    ctx["fetched_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    for m in batch:
        s = m["yf_symbol"]
        try:
            if dframes.get(s) is None and hframes.get(s) is None:
                scan["failures"][s] = "yfinance: veri yok (" + (derr.get(s) or herr.get(s) or "sembol eşlemesi/borsa kapsamı doğrulanamadı")[:100] + ")"
                continue
            fr, q = prepare_frames(dframes.get(s), hframes.get(s), ctx["sch"], MARKETS[cfg.market], cfg, pd.Timestamp(ctx["now"]))
            scan["results"][s] = evaluate(m, fr, q, ctx, cfg)
        except Exception as e:  # noqa: BLE001
            scan["failures"][s] = f"analiz hatası: {type(e).__name__}: {str(e)[:120]}"
            scan["errors"].append(f"{s}: {traceback.format_exc(limit=2)[-300:]}")
    scan["pos"] = i0 + len(batch)
    scan["pause_extra"] = max(0.0, scan.get("pause_extra", 0) / 2)
    scan["elapsed"] += time.time() - t0
    time.sleep(scan["pause"] + scan.get("pause_extra", 0))
    if scan["pos"] >= len(scan["symbols"]):
        scan["running"], scan["finished"] = False, True


def get_symbol_frames(meta: dict, scan: dict):
    cfg: Cfg = scan["cfg"]
    s = meta["yf_symbol"]
    d, _ = dl_daily((s,), 1)
    h, _ = dl_hourly((s,), 1)
    return prepare_frames(d.get(s), h.get(s), ctx_of(scan)["sch"], MARKETS[cfg.market], cfg, pd.Timestamp(ctx_of(scan)["now"]))


def ctx_of(scan: dict) -> dict:
    if scan["ctx"] is None:
        scan["ctx"] = build_ctx(scan, scan["cfg"])
    return scan["ctx"]


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_fundamentals(yf_symbol: str) -> dict:
    import yfinance as yf
    info = yf.Ticker(yf_symbol).info
    if not isinstance(info, dict) or not info:
        raise RuntimeError("Yahoo temel veri döndürmedi")
    keys = dict(marketCap="Piyasa değeri", trailingPE="F/K (trailing)", forwardPE="F/K (forward)", revenueGrowth="Gelir büyümesi (YoY, Yahoo)",
                earningsGrowth="Kâr büyümesi (YoY, Yahoo)", profitMargins="Net kâr marjı", debtToEquity="Borç/Özsermaye (Yahoo ham değer)",
                freeCashflow="Serbest nakit akışı", mostRecentQuarter="Son raporlanan çeyrek (Yahoo)", sector="Sektör (Yahoo)",
                industry="Alt sektör (Yahoo)", financialCurrency="Finansal tablo para birimi")
    out = {v: info.get(k) for k, v in keys.items()}
    out["_fetched"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return out


# ======================================================================================
# 8. ARAYÜZ
# ======================================================================================
def make_chart(df: pd.DataFrame, title: str, intraday: bool, plan: Optional[dict] = None, n: int = 140) -> go.Figure:
    d = df.tail(n)
    fm = "%d.%m %H:%M" if intraday else "%d.%m.%Y"
    x = [t.strftime(fm) for t in d.index]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.52, 0.12, 0.16, 0.20], vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=x, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"], name="Fiyat"), row=1, col=1)
    for col, nm in (("ema20", "EMA20"), ("ema50", "EMA50"), ("sma50", "SMA50"), ("sma200", "SMA200")):
        if col in d and d[col].notna().any():
            fig.add_trace(go.Scatter(x=x, y=d[col], mode="lines", name=nm, line=dict(width=1.2)), row=1, col=1)
    fig.add_trace(go.Bar(x=x, y=d["Volume"], name="Hacim", marker_color="#8899aa"), row=2, col=1)
    if "rsi14" in d:
        fig.add_trace(go.Scatter(x=x, y=d["rsi14"], name="RSI14", line=dict(width=1.2)), row=3, col=1)
        for lv in (30, 50, 70):
            fig.add_hline(y=lv, line_width=0.5, line_dash="dot", row=3, col=1)
        fig.add_trace(go.Bar(x=x, y=d["macd_hist"], name="MACD hist", marker_color="#bbbbbb"), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["macd"], name="MACD", line=dict(width=1.2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=x, y=d["macd_sig"], name="Sinyal", line=dict(width=1.2)), row=4, col=1)
    if plan:
        for k, lab in (("entry", "Giriş (ref.)"), ("stop", "Stop"), ("t2", "2R"), ("t3", "3R")):
            fig.add_hline(y=plan[k], line_width=0.8, line_dash="dash", annotation_text=lab, row=1, col=1)
    fig.update_xaxes(type="category", nticks=8, rangeslider_visible=False)
    fig.update_layout(height=780, title=title, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h"))
    return fig


def sidebar_controls() -> dict:
    sb = st.sidebar
    sb.header("Tarama ayarları")
    market = sb.selectbox("Piyasa", list(MARKETS), key="market")
    mkt = MARKETS[market]
    sb.subheader("Evren")
    source = sb.radio("Evren kaynağı", ["TradingView (otomatik)", "CSV yükle", "Kayıtlı evren"], key="uni_source")
    csv_file = sb.file_uploader("Sembol CSV (sütun: symbol[, name, exchange])", type=["csv"]) if source == "CSV yükle" else None
    adr = sb.checkbox("ADR/DR dahil et", value=False, disabled=(market != "ABD"))
    mode = sb.radio("Tarama modu", ["Hızlı tarama", "Tam tarama"], key="scan_mode",
                    help="Hızlı: ön filtreden geçen ilk N sembol. Tam: kapsamdaki tüm erişilebilir semboller (uzun sürebilir).")
    fast_n = sb.number_input("Hızlı modda sembol sayısı", 20, 1000, 150, 10)
    sb.subheader("Strateji")
    sb.selectbox("Strateji", ["Trend yönünde swing (1G rejim/trend, 4s kurulum, 2s momentum, 1s tetikleyici)"])
    trig = sb.selectbox("Giriş tetikleyici türü", ["Kırılım ve geri çekilme", "Yalnızca kırılım", "Yalnızca geri çekilme"])
    regime = sb.selectbox("Piyasa rejimi", ["Skor bileşeni", "Zorunlu filtre", "Kapalı"])
    bench = sb.text_input("Referans sembol (yfinance)", mkt["benchmark"], key=f"bench_{market}")
    with sb.expander("Teknik eşikler (başlangıç varsayımı)"):
        adx = st.number_input("4s ADX asgari", 5.0, 50.0, 20.0, 1.0)
        ext = st.number_input("4s EMA20'den azami uzaklık (ATR)", 0.5, 6.0, 2.0, 0.25)
        r_lo = st.number_input("2s RSI alt", 30.0, 70.0, 50.0, 1.0)
        r_hi = st.number_input("2s RSI üst (aşırı alım sınırı)", 55.0, 90.0, 70.0, 1.0)
        rv = st.number_input("Kırılımda göreli hacim asgari (×)", 1.0, 5.0, 1.5, 0.1)
        age = st.number_input("Taze tetikleyici azami yaşı (1s mum)", 1, 10, 3, 1)
        ms = st.number_input("Alım adayı için asgari teknik skor (/100)", 0, 100, 60, 5)
        short = st.checkbox("Seans sonu kısa mumları dahil et", True)
    with sb.expander("Likidite ve risk filtreleri"):
        mp = st.number_input(f"Asgari fiyat ({mkt['currency']})", 0.0, 10000.0, float(mkt["min_price"]), 0.5, key=f"mp_{market}")
        mv = st.number_input(f"Asgari ort. günlük işlem tutarı ({mkt['currency']}, yaklaşık)", 0.0, 1e11, float(mkt["min_value"]), 500_000.0, key=f"mv_{market}")
        ma = st.number_input("Azami günlük ATR%", 1.0, 30.0, float(mkt["max_atr_pct"]), 0.5, key=f"ma_{market}")
        tol = st.number_input("Veri geliş toleransı (dk)", 0, 120, int(mkt["default_tol"]), 1, key=f"tol_{market}",
                              help="Mum bitişinden bu kadar süre geçmeden mum 'kapanmış' sayılmaz. BIST için Yahoo gecikmesi doğrulanmadı.")
        stale = st.checkbox("Güncel olmayan 1s veriyi reddet", True)
    with sb.expander("Işlem planı ve portföy"):
        sm = st.selectbox("Stop yöntemi", ["ATR (4s)", "Yapı (1s dip)"])
        am = st.number_input("ATR çarpanı", 0.5, 5.0, 1.5, 0.25)
        rr = st.number_input("Dirence göre asgari R/R", 1.0, 5.0, 2.0, 0.25)
        nres = st.checkbox("Direnç bulunmazsa adayı İzleme'ye al", False)
        pf = st.number_input(f"Portföy büyüklüğü ({mkt['currency']})", 100.0, 1e10, float(mkt["portfolio"]), 1000.0, key=f"pf_{market}")
        cash = st.number_input(f"Kullanılabilir nakit ({mkt['currency']})", 0.0, 1e10, float(mkt["portfolio"]), 1000.0, key=f"cash_{market}")
        rp = st.number_input("İşlem başına risk %", 0.05, 5.0, 0.5, 0.05)
        mw = st.number_input("Azami pozisyon ağırlığı %", 1.0, 100.0, 20.0, 1.0)
    with sb.expander("Cloud / performans"):
        bs = st.number_input("Grup büyüklüğü", 5, 60, 20, 5)
        th = st.number_input("yfinance iş parçacığı", 1, 4, 1, 1)
        pa = st.number_input("Gruplar arası bekleme (sn)", 0.0, 10.0, 1.0, 0.5)
    cfg = Cfg(market=market, tol_min=int(tol), include_short=short, reject_stale=stale, regime_mode=regime, benchmark=bench.strip() or mkt["benchmark"],
              trigger_mode=trig, adx_min=adx, max_ext_atr=ext, rsi2_lo=r_lo, rsi2_hi=r_hi, rv_thr=rv, max_age=int(age),
              min_price=mp, min_value=mv, max_atr_pct=ma, stop_method=sm, atr_mult=am, min_rr=rr, need_res=nres, min_score=int(ms),
              portfolio=pf, cash=cash, risk_pct=rp, max_weight=mw)
    return dict(cfg=cfg, source=source, csv=csv_file, adr=adr, mode=mode, fast_n=fast_n, bs=int(bs), th=int(th), pause=float(pa))


def results_df(scan: dict) -> pd.DataFrame:
    rows = [{k: v for k, v in r.items() if k != "detay"} for r in scan["results"].values()]
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="ABD & BIST Çok Zaman Dilimli Tarayıcı", layout="wide")
    st.session_state.setdefault("scan", None)
    st.session_state.setdefault("universes", {})
    st.title("ABD ve BIST Çok Zaman Dilimli Hisse Tarayıcı")
    st.caption("Eğitim/araştırma amaçlıdır; yatırım tavsiyesi değildir. Eşikler kanıtlanmış üstünlük değil, değiştirilebilir başlangıç varsayımlarıdır. "
               "Backtest yapılmamıştır; kâr veya başarı oranı iddiası yoktur. Veriler gecikmeli/düzeltmeli olabilir; gerçek zamanlı veri garantisi yoktur.")
    ui = sidebar_controls()
    cfg: Cfg = ui["cfg"]
    mkt = MARKETS[cfg.market]
    sb = st.sidebar
    sb.divider()
    uni = st.session_state["universes"].get(cfg.market)
    scan = st.session_state["scan"]

    if sb.button("Evreni yükle / yenile"):
        with st.spinner("Evren toplanıyor..."):
            try:
                csv_bytes = ui["csv"].getvalue() if ui["csv"] is not None else None
                df, info = load_universe(cfg.market, ui["source"], ui["adr"], csv_bytes)
                st.session_state["universes"][cfg.market] = dict(df=df, info=info)
                uni = st.session_state["universes"][cfg.market]
            except Exception as e:  # noqa: BLE001
                st.error(f"Evren yüklenemedi: {e}")
    c1, c2, c3, c4 = sb.columns(2) + sb.columns(2)
    if c1.button("Taramayı başlat"):
        if uni is None:
            try:
                csv_bytes = ui["csv"].getvalue() if ui["csv"] is not None else None
                df, info = load_universe(cfg.market, ui["source"], ui["adr"], csv_bytes)
                uni = st.session_state["universes"][cfg.market] = dict(df=df, info=info)
            except Exception as e:  # noqa: BLE001
                st.error(f"Evren yüklenemedi: {e}")
        if uni is not None:
            cfg_run = cfg
            st.session_state["scan"] = make_scan(uni["df"], cfg_run, ui["mode"], ui["fast_n"], ui["bs"], ui["th"], ui["pause"])
            scan = st.session_state["scan"]
    if c2.button("Durdur") and scan:
        scan["running"] = False
    if c3.button("Devam et") and scan and not scan["finished"]:
        scan["running"] = True
    if c4.button("Sonuçları temizle"):
        st.session_state["scan"] = None
        scan = None
        st.cache_data.clear()

    # ---------------- veri ve piyasa durumu
    sch, cal_note = get_schedule(cfg.market)
    now = pd.Timestamp.now(tz="UTC")
    stt, nxt = market_status(sch, now)
    st.subheader("Veri ve piyasa durumu")
    a, b, c = st.columns(3)
    a.metric(f"{cfg.market} seansı", stt)
    b.metric("Yerel saat", now.tz_convert(mkt["tz"]).strftime("%Y-%m-%d %H:%M"))
    c.metric("Seans kapanışı" if stt == "Seans açık" else "Sonraki açılış", nxt.tz_convert(mkt["tz"]).strftime("%Y-%m-%d %H:%M") if nxt is not None else "—")
    (st.warning if cal_note.startswith("UYARI") else st.caption)(f"Takvim: {cal_note}")
    st.caption("Kaynaklar: Yahoo Finance (yfinance) OHLCV, TradingView tarayıcı uç noktası (yalnızca sembol keşfi/ön filtre; resmî garantili API değil). "
               "Fiyatlar bölünme-düzeltmeli, temettü-düzeltmesiz ham OHLC'dir (auto_adjust=False). Veriler gecikmeli olabilir.")
    if uni is not None:
        info = uni["info"]
        with st.expander("Evren ve kapsam raporu", expanded=scan is None):
            st.write(f"**Kaynak:** {info.get('source')} · **Zaman:** {info.get('fetched_at', '—')} (UTC)")
            if info.get("note"):
                st.warning(info["note"])
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("TV'nin bildirdiği toplam", info.get("total_reported") if info.get("total_reported") is not None else "—")
            m2.metric("Elde edilen sembol", info.get("rows_fetched", "—"))
            m3.metric("Kapsamdaki (adi hisse" + (" + ADR" if ui["adr"] else "") + ")", info.get("in_scope", "—"))
            m4.metric("Eşlemesi başarısız", info.get("map_failed", "—"))
            st.write("Enstrüman sınıfı dağılımı:", info.get("class_counts"))
            if info.get("removed_columns"):
                st.info(f"TradingView tarafından desteklenmeyen isteğe bağlı sütunlar çıkarıldı: {info['removed_columns']}")
            st.caption("Not: Kapsam, TradingView'in döndürdüğü sembollerle sınırlıdır; 'tüm hisseler' değil, 'erişilebilir hisse evreni'dir. "
                       "Yinelenen TV sembolleri: %s, yinelenen yfinance eşlemeleri: %s." % (info.get("dup_tv_removed", 0), info.get("dup_yf_removed", 0)))
            dfu = uni["df"]
            bad = dfu[dfu["in_scope"] & dfu["yf_symbol"].isna()][["tv_symbol", "company", "map_reason"]]
            if len(bad):
                st.write("Eşlemesi başarısız semboller (silinmedi, taranmayacak):")
                st.dataframe(bad, hide_index=True)
    else:
        st.info("Kenar çubuğundan evreni yükleyin veya doğrudan taramayı başlatın.")

    # ---------------- tarama ilerleme
    if scan:
        total = len(scan["symbols"])
        res = scan["results"]
        counts = pd.Series([r["Sınıf"] for r in res.values()]).value_counts().to_dict() if res else {}
        st.subheader("Tarama kapsamı ve ilerlemesi")
        if scan.get("note"):
            st.caption(scan["note"])
        st.progress(min(1.0, scan["pos"] / total) if total else 1.0, text=f"{scan['pos']}/{total} sembol işlendi"
                    + (" · çalışıyor" if scan["running"] else " · tamamlandı" if scan["finished"] else " · durduruldu"))
        k = st.columns(6)
        k[0].metric("İşlenen", scan["pos"]); k[1].metric("Analiz edilen", len(res))
        k[2].metric("Elenen (uygun değil)", counts.get(CLS_NO, 0)); k[3].metric("Veri yetersiz", counts.get(CLS_DATA, 0))
        k[4].metric("İndirme/analiz başarısız", len(scan["failures"])); k[5].metric("Bekleyen", max(0, total - scan["pos"]))
        st.caption(f"Geçen süre: {scan['elapsed']:.0f} sn · Tarama anı (UTC): {scan['now'][:19]} · Sabitlenen tarama saatinden sonra kapanan mumlar bu taramaya dahil edilmez. "
                   "Streamlit Cloud yeniden başlarsa bellekteki tarama durumu kaybolabilir.")
        if scan["ctx"] is not None:
            st.caption(f"Piyasa rejimi ({scan['cfg'].benchmark}): "
                       + ("olumlu" if scan["ctx"]["regime"] else "olumsuz" if scan["ctx"]["regime"] is False else "hesaplanamadı")
                       + f" — {scan['ctx']['regime_text']}")
        tabs = st.tabs(["Adaylar", "Detay ve grafik", "Temel veriler", "Hatalar ve veri kalitesi", "Yöntem"])
        rdf = results_df(scan)
        with tabs[0]:
            if rdf.empty:
                st.info("Henüz sonuç yok.")
            else:
                f1, f2, f3 = st.columns([2, 2, 2])
                cls_sel = f1.multiselect("Sınıf", [CLS_CAND, CLS_WATCH, CLS_NO, CLS_DATA], default=[CLS_CAND, CLS_WATCH])
                q = f2.text_input("Ara (sembol / şirket)")
                mins = f3.slider("Asgari skor", 0, 100, 0)
                v = rdf[rdf["Sınıf"].isin(cls_sel) & (rdf["Skor"].fillna(0) >= mins)]
                if q:
                    v = v[v["Sembol"].str.contains(q, case=False, na=False) | v["Şirket"].str.contains(q, case=False, na=False)]
                v = v.sort_values(["Sınıf", "Skor"], key=lambda s: s.map({CLS_CAND: 0, CLS_WATCH: 1, CLS_NO: 2, CLS_DATA: 3}) if s.name == "Sınıf" else -s.fillna(-1))
                show = v[["Sembol", "Şirket", "Piyasa", "ParaBirimi", "SonKapanış", "KapanışZamanı", "Sınıf", "Kurulum", "Skor", "SkorMaks", "Teyitler",
                          "RSI_2s", "MACDHist_2s", "ADX_4s", "GöreliHacim_1s", "HacimTürü", "ATRpct_1G", "Giriş", "Stop", "Hedef_2R", "Direnç", "Adet", "SinyalYaşı", "Gerekçe"]]
                st.dataframe(show, hide_index=True)
                st.download_button("Sonuçları CSV indir", rdf.to_csv(index=False).encode("utf-8-sig"), "tarama_sonuclari.csv", "text/csv")
                cand = rdf[rdf["Sınıf"] == CLS_CAND]
                if len(cand) >= 3:
                    sec = cand["Sektör"].value_counts()
                    st.caption("Aday sektör dağılımı: " + ", ".join(f"{s}: {n}" for s, n in sec.items())
                               + (" — ⚠ yoğunlaşma" if sec.iloc[0] / len(cand) > 0.4 else ""))
                st.caption("Skor 0–100'dür; olasılık veya güven yüzdesi DEĞİLDİR. Eksik/kapalı bileşenler diğerlerine dağıtılmaz (SkorMaks = hesaplanabilen azami puan). "
                           "Giriş, son tamamlanmış 1s kapanışına dayalı VARSAYIMSAL referanstır. Tabloda USD ve TRY toplanmaz.")
        with tabs[1]:
            if rdf.empty:
                st.info("Sonuç yok.")
            else:
                names = rdf.sort_values("Skor", ascending=False, na_position="last")
                pick = st.selectbox("Hisse", names["yf_symbol"].tolist(),
                                    format_func=lambda s: f"{res[s]['Sembol']} — {res[s]['Şirket'][:30]} [{res[s]['Sınıf']}]")
                r = res[pick]
                det = r["detay"]
                st.markdown(f"### {r['Sembol']} · {r['Şirket']} — **{r['Sınıf']}**")
                st.write(f"Skor **{fmt(r['Skor'], 0)}** / mevcut azami {fmt(r['SkorMaks'], 0)} (tam ölçek 100) · Teyitler: {r['Teyitler']} · "
                         f"Kurulum: {r['Kurulum']} · Sinyal yaşı: {fmt(r['SinyalYaşı'], 0)} (1s mum) · Son kapanış zamanı: {r['KapanışZamanı']} · Veri çekimi: {r.get('VeriÇekimi', '')}")
                if det.get("gaps"):
                    st.error("Değerlendirme engellendi / eksik: " + " | ".join(det["gaps"]))
                cA, cB, cC = st.columns(3)
                with cA:
                    st.write("**Olumlu gerekçeler**")
                    for x in det["pos"] or ["—"]:
                        st.write("• " + x)
                with cB:
                    st.write("**Sağlanmayan koşullar**")
                    for x in det["neg"] or ["—"]:
                        st.write("• " + x)
                with cC:
                    st.write("**Riskler / uyarılar**")
                    for x in det["risks"] or ["—"]:
                        st.write("• " + x)
                if det["comps"]:
                    with st.expander("Teknik puan dökümü", expanded=False):
                        cdf = pd.DataFrame([dict(ZD=TF_LABEL[c["tf"]], Koşul=c["name"], Durum="kapalı" if not c["enabled"] else "hesaplanamadı" if c["ok"] is None else "✓" if c["ok"] else "✗",
                                                 Puan=(c["pts"] if (c["enabled"] and c["ok"]) else (c.get("partial", 0) if c["enabled"] and c["ok"] is False and c["name"].startswith("Tetikleyici") else 0)),
                                                 Azami=c["pts"] if c["enabled"] else 0, Değer=c["text"]) for c in det["comps"]])
                        st.dataframe(cdf, hide_index=True)
                        st.caption("Aynı trend bilgisi farklı göstergelerle tekrar ödüllendirilmez: günlük=trend yapısı, 4s=kurulum/trend gücü, 2s=momentum, 1s=tetikleyici, "
                                   "hacim=likidite/OBV/göreli hacim, rejim=endeks+göreli getiri. Eski (taze olmayan) tetikleyici kısmi 7 puan alır.")
                plan = det.get("plan")
                if plan:
                    st.markdown("**Varsayımsal işlem planı** (tavsiye değildir)")
                    cur = r["ParaBirimi"]
                    pdf = pd.DataFrame([
                        ("Referans giriş", fmt(plan["entry"]), plan["entry_txt"]),
                        ("Stop", fmt(plan["stop"]), plan["stop_txt"]),
                        ("Hisse başı risk", f"{fmt(plan['risk_ps'])} (%{fmt(plan['risk_pct_price'])})", ""),
                        ("Varsayımsal hedef 2R / 3R", f"{fmt(plan['t2'])} / {fmt(plan['t3'])}", "Sabit R katları; bağımsız direnç değildir"),
                        ("Bağımsız direnç", fmt(plan["resistance"]) if plan["resistance"] else "bulunamadı", plan["res_src"]),
                        ("Dirence göre R/R", fmt(plan["rr_res"]), f"asgari {cfg.min_rr:g}"),
                        ("Adet", str(plan["size"]["shares"]) if plan["size"] else "—", f"sınırlayan: {plan['size']['binding']}" if plan["size"] else ""),
                        ("Pozisyon değeri / portföy", f"{fmt(plan['size']['position_value'])} {cur} / %{fmt(plan['size']['weight_pct'])}" if plan["size"] else "—", ""),
                        ("Risk tutarı", f"{fmt(plan['size']['risk_amount'])} {cur}" if plan["size"] else "—", f"portföyün %{cfg.risk_pct:g}'i bütçe; stop gap'te gerçekleşmeyebilir")],
                        columns=["Kalem", "Değer", "Açıklama"])
                    st.dataframe(pdf, hide_index=True)
                tf_sel = st.radio("Grafik zaman dilimi", ["1h", "2h", "4h", "1d"], horizontal=True, index=0)
                meta = next((m for m in scan["symbols"] if m["yf_symbol"] == pick), None)
                if meta is not None:
                    try:
                        fr, _ = get_symbol_frames(meta, scan)
                        if len(fr[tf_sel]):
                            st.plotly_chart(make_chart(fr[tf_sel], f"{r['Sembol']} · {TF_LABEL[tf_sel]} (yalnızca kapanmış mumlar)", tf_sel != "1d", plan if tf_sel == "1h" else None))
                        else:
                            st.info("Bu zaman dilimi için tamamlanmış mum yok.")
                    except Exception as e:  # noqa: BLE001
                        st.warning(f"Grafik üretilemedi: {e}")
        with tabs[2]:
            st.caption("Yatırımcı değerlendirmesi teknik skordan AYRIDIR; teknik sinyal tek başına uzun vadeli yatırım tezi değildir. "
                       "Kaynak: Yahoo Finance (yfinance .info); kapsam/güncellik değişkendir, BIST için eksiklik yaygın olabilir. Bankalarda aynı eşikler anlamlı olmayabilir.")
            if rdf.empty:
                st.info("Sonuç yok.")
            else:
                opts = rdf[rdf["Sınıf"].isin([CLS_CAND, CLS_WATCH])]["yf_symbol"].tolist() or rdf["yf_symbol"].tolist()
                sel = st.multiselect("Hisse(ler) (en çok 15)", opts, default=opts[:3], max_selections=15)
                if st.button("Temel verileri getir") and sel:
                    rows = []
                    for s in sel:
                        try:
                            f = fetch_fundamentals(s)
                            rows.append({"Sembol": s, **{k: ("Veri yok" if isnan(v) else v) for k, v in f.items()}})
                        except Exception as e:  # noqa: BLE001
                            rows.append({"Sembol": s, "Piyasa değeri": f"alınamadı: {str(e)[:60]}"})
                        time.sleep(0.5)
                    fdf = pd.DataFrame(rows)
                    if "Son raporlanan çeyrek (Yahoo)" in fdf:
                        fdf["Son raporlanan çeyrek (Yahoo)"] = fdf["Son raporlanan çeyrek (Yahoo)"].map(
                            lambda x: datetime.fromtimestamp(x, tz=timezone.utc).strftime("%Y-%m-%d") if isinstance(x, (int, float)) and not isnan(x) else x)
                    st.dataframe(fdf.astype(str), hide_index=True)
                    st.caption("Eksik değerler sıfır/olumlu değerle doldurulmaz. 'Borç/Özsermaye' Yahoo'nun ham değeridir (birimi doğrulanmadı).")
        with tabs[3]:
            st.write(f"**Başarısız semboller ({len(scan['failures'])})**")
            if scan["failures"]:
                st.dataframe(pd.DataFrame([{"yfinance sembolü": k, "neden": v} for k, v in scan["failures"].items()]), hide_index=True)
            else:
                st.caption("Başarısız sembol yok.")
            vy = [(r["Sembol"], "; ".join(r["detay"].get("gaps", []))) for r in res.values() if r["Sınıf"] == CLS_DATA]
            st.write(f"**Veri yetersiz sonuçlar ({len(vy)})**")
            if vy:
                st.dataframe(pd.DataFrame(vy, columns=["Sembol", "engellenen değerlendirme"]), hide_index=True)
            qrows = []
            for r in res.values():
                q = r["detay"].get("quality")
                if q:
                    qrows.append(dict(Sembol=r["Sembol"], gün_ham=q["daily"]["raw"], saat_ham=q["hourly"]["raw"], saat_tekrar=q["hourly"]["dup"], saat_NaN=q["hourly"]["nan"],
                                      saat_hatalı_fiyat=q["hourly"]["bad_price"], saat_sıfır_hacim=q["hourly"]["zero_vol"],
                                      seans_dışı=q["hourly_ann"]["non_session"], hizasız=q["hourly_ann"]["misaligned"], gün_seans_dışı=q["daily_ann"]["non_session"]))
            if qrows:
                qd = pd.DataFrame(qrows)
                st.write("**Veri kalitesi (sembol başına)**")
                st.dataframe(qd, hide_index=True)
                if (qd["hizasız"] > 0.3 * qd["saat_ham"].clip(lower=1)).any():
                    st.warning("Bazı sembollerde saatlik mumların çoğu seans açılışına hizalı değil: sağlayıcı hizalaması bu piyasa için doğrulanamadı.")
            if scan["errors"]:
                with st.expander("Hata ayrıntıları"):
                    st.code("\n".join(scan["errors"][-20:]))
        with tabs[4]:
            st.markdown(METHOD_MD)
    else:
        st.markdown(METHOD_MD)

    # ---------------- bir sonraki grup (durdurma/devam için her grup ayrı çalışma)
    if scan and scan["running"] and not scan["finished"]:
        run_batch(scan)
        st.rerun()


METHOD_MD = """
### Yöntem özeti
**Zaman dilimleri:** 1G (rejim/ana trend) → 4s (kurulum) → 2s (momentum) → 1s (tetikleyici). Analizde yalnızca **tamamlanmış** mumlar kullanılır.
**Mum kapanışı:** 1s mum bitişi = min(başlangıç+1s, seans kapanışı); bitiş + tolerans ≤ tarama anı ise kapanmış sayılır. Günlük mum seans kapanışından sonra kullanılır.
2s/4s mumlar yfinance'tan istenmez; 1s mumlardan **seans açılışına sabitli** üretilir (gün/seans arası birleştirilmez; eksik alt mum varsa üst mum geçersiz; seans sonu kısa mumlar `kısa` işaretlidir).
**Günlük trend (25):** Kapanış>SMA50 (10), SMA50>SMA200 (10), RSI14>50 (5) — hepsi zorunlu. **4s (20):** Kapanış>EMA20>EMA50 (8), ADX≥eşik ve +DI>-DI (8), EMA20'den uzaklık ≤ eşik×ATR (4).
**2s (15):** MACD>sinyal (6, zorunlu), histogram son 3 mumda yükseliyor (5), RSI aralıkta (4, zorunlu). **1s (15):** *Kırılım*: kapanış > önceki 20 tamamlanmış mumun tepesi **ve** göreli hacim ≥ eşik; *Geri çekilme*: son 3 mumda EMA20 teması + EMA20 üstü kapanış + yeşil mum + EMA20>EMA50. Tetikleyici yaşı (1 = son mumda oluştu) gösterilir; azami yaştan eskiyse yalnızca İzleme.
**Hacim/likidite (15):** işlem tutarı derinliği (5), OBV eğimi (5), günlük basit göreli hacim (5). **Rejim/göreli güç (10):** referans endeks SMA50 ve SMA200 üzerinde (5), 63 günlük göreli getiri > 0 (5).
**Kapılar (puandan bağımsız):** asgari fiyat, ortalama işlem tutarı (OHLCV'den *yaklaşık*), azami günlük ATR%. Yüksek skor bu filtreleri geçersiz kılmaz. Eksik veri olumlu sinyal üretmez; skor eksik bileşenlere dağıtılmaz.
**Sınıflar:** Alım adayı (tüm kapılar + taze tetikleyici + geçerli plan + dirence göre R/R + asgari skor), İzleme (trend uygun, giriş koşulu/plan eksik), Uygun değil, Veri yetersiz.
**Plan:** referans giriş = son 1s kapanışı (gerçekleşmiş fiyat değil); stop = ATR(4s) tabanlı veya 1s yapı tabanlı; 2R/3R yalnızca varsayımsal hedeftir, R/R filtresi ise bağımsız dirence (4s 20 mum tepesi, 1G 60 gün tepesi) göre hesaplanır.
Pozisyon: min(risk bütçesi, azami ağırlık, nakit). USD ve TRY ayrı tutulur, kur dönüşümü yapılmaz.
**Sınırlar:** backtest yok; spread/komisyon/kayma yok; yfinance ve TradingView verileri gecikmeli/farklı olabilir; TradingView uç noktası resmî API değildir.
"""

if __name__ == "__main__":
    main()
