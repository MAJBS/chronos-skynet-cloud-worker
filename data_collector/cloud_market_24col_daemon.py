# data_collector/cloud_market_24col_daemon.py
import os
import sys
import json
import time
import urllib.request
import numpy as np
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIVE_DIR = os.path.join(ROOT_DIR, "live")

COIN_MAP = {
    "btc": {"bin": "BTCUSDT", "okx_swap": "BTC-USDT-SWAP", "cb": "BTC-USD"},
    "eth": {"bin": "ETHUSDT", "okx_swap": "ETH-USDT-SWAP", "cb": "ETH-USD"},
    "sol": {"bin": "SOLUSDT", "okx_swap": "SOL-USDT-SWAP", "cb": "SOL-USD"},
}


def _get_json(url: str, timeout: int = 6):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ChronosCloud/3.3"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def fetch_single_coin_24cols(coin: str, now_ms: int, prev_cvd: dict) -> dict:
    cfg = COIN_MAP[coin]
    sym_b = cfg["bin"]
    sym_okx_sw = cfg["okx_swap"]

    kl = _get_json(f"https://data-api.binance.vision/api/v3/klines?symbol={sym_b}&interval=1m&limit=2")[-1]
    s_open, s_high, s_low, s_close = float(kl[1]), float(kl[2]), float(kl[3]), float(kl[4])
    s_vol       = float(kl[5])
    s_trades    = float(kl[8])
    taker_buy_s = float(kl[9])
    delta_s     = 2.0 * taker_buy_s - s_vol
    prev_cvd[f"{coin}_s"] = prev_cvd.get(f"{coin}_s", 0.0) + delta_s

    dp = _get_json(f"https://data-api.binance.vision/api/v3/depth?symbol={sym_b}&limit=100")
    bids = [(float(p), float(q)) for p, q in dp.get("bids", [])]
    asks = [(float(p), float(q)) for p, q in dp.get("asks", [])]
    if bids and asks:
        b0_p, b0_q = bids[0]
        a0_p, a0_q = asks[0]
        spread_s = max(0.0001, a0_p - b0_p)
        obi_s    = (b0_q - a0_q) / max(1e-8, b0_q + a0_q)
        micro_p  = (b0_q * a0_p + a0_q * b0_p) / max(1e-8, b0_q + a0_q)
        depth_b1 = sum(p * q for p, q in bids if p >= s_close * 0.99)
        depth_a1 = sum(p * q for p, q in asks if p <= s_close * 1.01)
    else:
        spread_s, obi_s, micro_p, depth_b1, depth_a1 = 0.01, 0.0, s_close, 10000.0, 10000.0

    perp_close, perp_vol, obi_p, oi_usd, fund_r = s_close, s_vol, obi_s, 0.0, 0.0
    try:
        tk_sw = _get_json(f"https://www.okx.com/api/v5/market/ticker?instId={sym_okx_sw}").get("data", [{}])[0]
        perp_close = float(tk_sw.get("last", s_close))
        perp_vol   = float(tk_sw.get("volCcy24h", s_vol)) / 1440.0
        bsz = float(tk_sw.get("bidSz", 1.0))
        asz = float(tk_sw.get("askSz", 1.0))
        obi_p = (bsz - asz) / max(1e-8, bsz + asz)
    except Exception:
        pass

    try:
        oi_d = _get_json(f"https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId={sym_okx_sw}").get("data", [{}])[0]
        oi_usd = float(oi_d.get("oiUsd", 0.0))
    except Exception:
        pass

    try:
        fr_d = _get_json(f"https://www.okx.com/api/v5/public/funding-rate?instId={sym_okx_sw}").get("data", [{}])[0]
        fund_r = float(fr_d.get("fundingRate", 0.0))
    except Exception:
        pass

    prev_cvd[f"{coin}_p"] = prev_cvd.get(f"{coin}_p", 0.0) + obi_p * perp_vol

    cb_prem = 0.0
    try:
        cb_tk = _get_json(f"https://api.exchange.coinbase.com/products/{cfg['cb']}/ticker")
        cb_price = float(cb_tk.get("price", s_close))
        cb_prem  = cb_price - s_close
    except Exception:
        pass

    return {
        "timestamp_ms": int(now_ms),
        "spot_open": s_open, "spot_high": s_high, "spot_low": s_low, "spot_close": s_close,
        "spot_volume": s_vol, "spot_trades_count": s_trades,
        "perp_close": perp_close, "perp_volume": perp_vol,
        "cvd_spot": round(prev_cvd[f"{coin}_s"], 4), "cvd_perp": round(prev_cvd[f"{coin}_p"], 4),
        "obi_spot": round(obi_s, 6), "obi_perp": round(obi_p, 6),
        "micro_price_spot": round(micro_p, 4), "spread_spot": round(spread_s, 4),
        "depth_bid_1pct": round(depth_b1, 2), "depth_ask_1pct": round(depth_a1, 2),
        "liq_long_vol": 0.0, "liq_short_vol": 0.0,
        "open_interest": round(oi_usd, 2), "funding_rate": round(fund_r, 8),
        "coinbase_premium": round(cb_prem, 4), "dvol_index": 50.0,
        "top_trader_ls_ratio": 1.0, "spy_1m_return": 0.0,
    }


def main():
    os.makedirs(LIVE_DIR, exist_ok=True)
    prev_cvd = {}
    last_min = -1
    while True:
        now_ms = int(time.time() * 1000)
        curr_min = now_ms // 60000
        if curr_min > last_min:
            ts_bucket = curr_min * 60000
            for c in ("btc", "eth", "sol"):
                try:
                    row = fetch_single_coin_24cols(c, ts_bucket, prev_cvd)
                    wal_p = os.path.join(LIVE_DIR, f"wal_{c.upper()}_active.jsonl")
                    with open(wal_p, "a", encoding="utf-8") as wf:
                        wf.write(json.dumps(row) + "\n")
                except Exception as e:
                    print(f"[CLOUD MKT WARN {c}] {e}")
            last_min = curr_min
        time.sleep(5)


if __name__ == "__main__":
    main()
