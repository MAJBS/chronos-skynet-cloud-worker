# data_collector/belnap_collector_daemon.py
# Demonio Recolector Point-in-Time 24/7 Anti-Rate-Limit (v3.2)
#   - Incluye Google News CDN (when:1h), Sentimiento Taker de Binance Futures, Fallback Reddit JSON->RSS,
#     TTL de Hashes para evitar bloqueos tras >24h, y telemetria explicita `Raw=XX -> Ev=YY`.

import os
import sys
import re
import gzip
import json
import math
import time
import random
import hashlib
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CRIPTO_DIR = os.path.join(ROOT_DIR, "cripto_30q")
if CRIPTO_DIR not in sys.path:
    sys.path.insert(0, CRIPTO_DIR)

from belnap_kb_engine import BelnapKnowledgeBase


USER_AGENTS_POOL = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
]


def _enforce_single_instance_pid_lock(live_dir: str) -> None:
    os.makedirs(live_dir, exist_ok=True)
    pid_file = os.path.join(live_dir, "belnap_collector.pid")
    my_pid   = os.getpid()

    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r", encoding="utf-8") as pf:
                old_pid = int(pf.read().strip())
            if old_pid != my_pid and old_pid > 0:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/PID", str(old_pid)], capture_output=True, text=True)
                else:
                    import signal
                    os.kill(old_pid, signal.SIGTERM)
                time.sleep(0.5)
        except Exception:
            pass

    try:
        with open(pid_file, "w", encoding="utf-8") as pf:
            pf.write(str(my_pid))
    except Exception:
        pass


def _read_latest_live_prices(live_dir: str) -> dict[str, float]:
    prices = {"btc": 0.0, "eth": 0.0, "sol": 0.0}
    for coin in ("btc", "eth", "sol"):
        wal_p = os.path.join(live_dir, f"wal_{coin.upper()}_active.jsonl")
        if os.path.exists(wal_p):
            try:
                with open(wal_p, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    fsize = f.tell()
                    f.seek(max(0, fsize - 4096), os.SEEK_SET)
                    lines = f.read().decode("utf-8", errors="ignore").strip().splitlines()
                    if lines:
                        last_obj = json.loads(lines[-1])
                        p = float(last_obj.get("perp_close") or last_obj.get("spot_close") or 0.0)
                        if p > 0.0:
                            prices[coin] = p
            except Exception:
                pass
    return prices


def _http_get_text(url: str, timeout: int = 8, is_reddit: bool = False) -> str:
    if is_reddit:
        ua = f"windows:chronos.quant.belnap:v3.2.{random.randint(10,99)} (by /u/quant_research_node)"
    else:
        ua = random.choice(USER_AGENTS_POOL)

    headers = {
        "User-Agent": ua,
        "Accept": "application/json, application/xml, text/xml, text/html, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, identity",
        "Cache-Control": "no-cache",
    }

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw_bytes = resp.read()
        encoding = str(resp.info().get("Content-Encoding", "")).lower()
        if "gzip" in encoding or (len(raw_bytes) >= 2 and raw_bytes[:2] == b"\x1f\x8b"):
            raw_bytes = gzip.decompress(raw_bytes)
        return raw_bytes.decode("utf-8", errors="replace")


def _parse_x_syndication_timeline(html_text: str, handle: str, source_id: str) -> list[dict]:
    events = []
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html_text, re.DOTALL)
    if not m:
        return events

    try:
        data = json.loads(m.group(1))
        entries = (
            data.get("props", {})
            .get("pageProps", {})
            .get("timeline", {})
            .get("entries", [])
        )
        for ent in entries[:8]:
            tw = ent.get("content", {}).get("tweet", {})
            if not tw:
                continue
            full_text = str(tw.get("full_text") or tw.get("text") or "").strip()
            if not full_text:
                continue
            likes   = float(tw.get("favorite_count", 0))
            replies = float(tw.get("reply_count", 0))
            quotes  = float(tw.get("quote_count", 0))
            events.append({
                "source_id": source_id,
                "author": f"@{handle}",
                "text": f"@{handle}: {full_text}",
                "ts_ms": int(time.time() * 1000),
                "metrics": {
                    "likes": likes,
                    "replies": replies + quotes,
                    "bookmarks": likes * 0.15,
                }
            })
    except Exception:
        pass
    return events


class BelnapContinuousCollector:
    def __init__(self, root_dir: str):
        self.root_dir    = root_dir
        self.live_dir    = os.path.join(root_dir, "live")
        self.belnap_dir  = os.path.join(self.live_dir, "belnap")
        self.wal_path    = os.path.join(self.live_dir, "wal_BELNAP_active.jsonl")
        self.inbox_path  = os.path.join(self.belnap_dir, "browser_scraped_inbox.jsonl")
        self.config_path = os.path.join(SCRIPT_DIR, "belnap_sources_config.json")

        kb_dir  = os.path.join(root_dir, "cripto_30q", "belnap_kb")
        mem_dir = os.path.join(root_dir, "cripto_30q", "data", "belnap_memory")
        os.makedirs(self.belnap_dir, exist_ok=True)

        self.kb = BelnapKnowledgeBase(kb_dir, mem_dir)
        self.sources_cfg = self._load_and_register_sources_config()

        # Diccionario de hashes con timestamp para expirar hashes antiguos (TTL = 6 horas)
        self.seen_item_hashes_ts = {}
        self.last_polled_ts      = {}
        self.x_rotation_cursor   = 0
        self.prev_chain_tvl      = {}
        self.is_first_boot       = True

        self.e_fast = {c: np.zeros(3, dtype=np.float64) for c in ("btc", "eth", "sol")}
        self.e_slow = {c: np.zeros(3, dtype=np.float64) for c in ("btc", "eth", "sol")}
        self.euphoria_ema = {c: 0.0 for c in ("btc", "eth", "sol")}
        self.contagion_ema = 0.0
        self.credibility_signal_ema = 0.0
        self.last_decay_ts = time.time()
        self.wal_buffer_rows = []

        self._sanitize_and_restore_state_from_wal()

    def _sanitize_and_restore_state_from_wal(self) -> None:
        if not os.path.exists(self.wal_path):
            return

        try:
            clean_rows_by_ts = {}
            with open(self.wal_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    obj = json.loads(line)
                    if float(obj.get("btc_belnap_both_1m", 0.0)) >= 0.85 and float(obj.get("btc_belnap_true_1m", 0.0)) <= 0.015:
                        continue
                    ts = int(obj["timestamp_ms"])
                    clean_rows_by_ts[ts] = obj

            sorted_rows = [clean_rows_by_ts[k] for k in sorted(clean_rows_by_ts.keys())]
            tmp_w = f"{self.wal_path}.tmp.{os.getpid()}"
            with open(tmp_w, "w", encoding="utf-8") as wf:
                for r in sorted_rows:
                    wf.write(json.dumps(r) + "\n")
            os.replace(tmp_w, self.wal_path)

            if sorted_rows:
                last_r = sorted_rows[-1]
                age_sec = max(0.0, (time.time() * 1000.0 - float(last_r["timestamp_ms"])) / 1000.0)
                if age_sec <= 2700.0 and float(last_r.get("btc_dirichlet_s_1m", 4.0)) > 4.05:
                    d_1m = math.exp(-math.log(2.0) * age_sec / 240.0)
                    for c in ("btc", "eth", "sol"):
                        s_1m = float(last_r.get(f"{c}_dirichlet_s_1m", 4.0))
                        t_1m = float(last_r.get(f"{c}_belnap_true_1m", 0.0))
                        f_1m = float(last_r.get(f"{c}_belnap_false_1m", 0.0))
                        b_1m = float(last_r.get(f"{c}_belnap_both_1m", 0.0))
                        self.e_fast[c] = np.array([t_1m * s_1m, f_1m * s_1m, b_1m * s_1m], dtype=np.float64) * d_1m
                        self.e_slow[c] = self.e_fast[c].copy()
                    self.is_first_boot = False
        except Exception as e:
            print(f"[WARN] No se pudo restaurar estado previo de WAL: {e}")

    def _load_and_register_sources_config(self) -> dict:
        if not os.path.exists(self.config_path):
            return {"tier1_fast_1m_apis": [], "x_accounts_syndication_api": {"accounts": []}, "tier2_macro_60m_apis": []}

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        for ap in cfg.get("tier1_fast_1m_apis", []) + cfg.get("tier2_macro_60m_apis", []):
            sid = str(ap.get("source_id", ""))
            if sid and sid not in self.kb.sources_by_id:
                self.kb.sources_by_id[sid] = {
                    "id_source": sid,
                    "nombre_fuente": sid,
                    "transport_mode": ap.get("parser_type", "REST_JSON"),
                    "target": ap.get("url", ""),
                    "paths": {},
                    "prior_trust_weight": float(ap.get("prior_trust_weight", 1.8)),
                    "horizon_scope": ap.get("horizon_scope", ["1m", "60m"]),
                    "asset_affinity": ap.get("asset_affinity", ["btc", "eth", "sol"]),
                    "is_contrarian_retail_gauge": False,
                    "poll_interval_sec": int(ap.get("poll_interval_sec", 60)),
                }

        for acc in cfg.get("x_accounts_syndication_api", {}).get("accounts", []):
            sid = str(acc.get("source_id", ""))
            hnd = f"@{str(acc.get('handle', '')).lower().lstrip('@')}"
            entry = {
                "id_source": sid,
                "nombre_fuente": hnd,
                "transport_mode": "X_SYNDICATION_JSON",
                "target": hnd,
                "paths": {},
                "prior_trust_weight": float(acc.get("trust_weight", 2.0)),
                "horizon_scope": ["1m", "60m"],
                "asset_affinity": ["btc", "eth", "sol"],
                "is_contrarian_retail_gauge": bool(acc.get("contrarian", False)),
                "poll_interval_sec": 180,
            }
            if sid:
                self.kb.sources_by_id[sid] = entry
            self.kb.sources_by_handle[hnd] = entry

        return cfg

    def _apply_continuous_time_decay(self) -> None:
        now_s = time.time()
        dt_sec = max(0.0, now_s - self.last_decay_ts)
        self.last_decay_ts = now_s
        if dt_sec <= 0.0:
            return

        # Semividas calibradas: 6 minutos para carril rapido (1m) y 60 minutos para estructural (60m)
        decay_fast = math.exp(-math.log(2.0) * dt_sec / (6.0 * 60.0))
        decay_slow = math.exp(-math.log(2.0) * dt_sec / (60.0 * 60.0))

        for c in ("btc", "eth", "sol"):
            self.e_fast[c] *= decay_fast
            self.e_slow[c] *= decay_slow
            self.euphoria_ema[c] *= decay_slow

        self.contagion_ema *= decay_fast
        self.credibility_signal_ema *= decay_slow

    def _parse_api_response(self, api_spec: dict, raw_text: str) -> list[dict]:
        ptype = str(api_spec.get("parser_type", "")).upper()
        sid   = str(api_spec.get("source_id", "API"))
        now_ms = int(time.time() * 1000)
        bucket_5m = now_ms // 300_000
        events = []

        if ptype == "BINANCE_TAKER_RATIO_JSON":
            data = json.loads(raw_text)
            if data and isinstance(data, list):
                latest = data[-1]
                ratio = float(latest.get("buySellRatio", 1.0))
                sym_coin = "BTC" if "BTC" in sid else ("ETH" if "ETH" in sid else "SOL")
                if ratio >= 1.12:
                    txt = f"[{bucket_5m}] ${sym_coin} institutional net taker buy surge (ratio={ratio:.2f}) bullish breakout accumulation"
                elif ratio <= 0.89:
                    txt = f"[{bucket_5m}] ${sym_coin} aggressive net taker sell dump (ratio={ratio:.2f}) bearish outflow pressure"
                else:
                    txt = f"[{bucket_5m}] ${sym_coin} futures orderflow divided in conflict (ratio={ratio:.2f}) choppy volatility"
                events.append({
                    "source_id": sid,
                    "author": "Binance_Futures_Data",
                    "text": txt,
                    "ts_ms": now_ms,
                    "metrics": {"bookmarks": 8.0, "likes": 12.0}
                })

        elif ptype == "TREE_OF_ALPHA_JSON":
            items = json.loads(raw_text)
            for it in items[:25]:
                title  = str(it.get("title") or it.get("en") or "")
                body   = str(it.get("body") or "")
                author = str(it.get("source") or it.get("info", {}).get("twitterId") or "TreeOfAlpha")
                events.append({
                    "source_id": sid,
                    "author": author,
                    "text": f"{title} {body}".strip(),
                    "ts_ms": int(it.get("time", now_ms)),
                    "metrics": {
                        "likes": float(it.get("likes", 0)),
                        "replies": float(it.get("dislikes", 0)),
                    }
                })

        elif ptype == "BYBIT_V5_JSON":
            data = json.loads(raw_text)
            items = data.get("result", {}).get("list", [])
            for it in items[:15]:
                title = str(it.get("title", ""))
                desc  = str(it.get("description", ""))
                events.append({
                    "source_id": sid,
                    "author": "Bybit_Official",
                    "text": f"Bybit Announcement: {title}. {desc}",
                    "ts_ms": int(it.get("dateTimestamp", now_ms)),
                    "metrics": {"bookmarks": 10.0, "likes": 15.0}
                })

        elif ptype == "BINANCE_CMS_JSON":
            data = json.loads(raw_text)
            catalogs = data.get("data", {}).get("catalogs", [])
            for cat in catalogs:
                for art in cat.get("articles", [])[:10]:
                    title = str(art.get("title", ""))
                    events.append({
                        "source_id": sid,
                        "author": "Binance_Official",
                        "text": f"Binance Official Announcement: {title}",
                        "ts_ms": int(art.get("releaseDate", now_ms)),
                        "metrics": {"bookmarks": 15.0, "likes": 20.0}
                    })

        elif ptype == "OKX_V5_JSON":
            data = json.loads(raw_text)
            details = data.get("data", [{}])[0].get("details", []) if data.get("data") else []
            for it in details[:12]:
                title = str(it.get("title", ""))
                events.append({
                    "source_id": sid,
                    "author": "OKX_Official",
                    "text": f"OKX Announcement: {title}",
                    "ts_ms": int(it.get("pTime", now_ms)),
                    "metrics": {"bookmarks": 8.0, "likes": 12.0}
                })

        elif ptype == "REDDIT_JSON":
            data = json.loads(raw_text)
            children = data.get("data", {}).get("children", [])
            for ch in children:
                d = ch.get("data", {})
                title = str(d.get("title", ""))
                selftext = str(d.get("selftext", ""))[:260]
                ups = float(d.get("ups", 0))
                ratio = float(d.get("upvote_ratio", 0.5))
                down_est = ups * (1.0 - ratio) / max(0.05, ratio)
                events.append({
                    "source_id": sid,
                    "author": str(d.get("author", "")),
                    "text": f"{title}. {selftext}",
                    "ts_ms": int(float(d.get("created_utc", time.time())) * 1000),
                    "metrics": {
                        "likes": ups,
                        "replies": float(d.get("num_comments", 0)) + down_est,
                    }
                })

        elif ptype == "RSS_XML":
            root = ET.fromstring(raw_text)
            # Soporta tanto RSS (<item>) como Atom de Reddit (<entry>)
            items_xml = root.findall(".//item")
            if not items_xml:
                items_xml = root.findall(".//{http://www.w3.org/2005/Atom}entry")
            for item in items_xml[:18]:
                title = (
                    item.findtext("title")
                    or item.findtext("{http://www.w3.org/2005/Atom}title")
                    or ""
                )
                desc = (
                    item.findtext("description")
                    or item.findtext("{http://www.w3.org/2005/Atom}content")
                    or ""
                )
                desc_clean = re.sub(r"<[^>]+>", " ", desc)
                events.append({
                    "source_id": sid,
                    "author": sid,
                    "text": f"{title}. {desc_clean[:240]}",
                    "ts_ms": now_ms,
                    "metrics": {}
                })

        elif ptype == "DEFILLAMA_CHAINS_JSON":
            chains = json.loads(raw_text)
            for ch in chains:
                name = str(ch.get("name", "")).lower()
                if name in ("ethereum", "solana", "bitcoin"):
                    tvl = float(ch.get("tvl", 0.0))
                    prev = self.prev_chain_tvl.get(name, tvl)
                    self.prev_chain_tvl[name] = tvl
                    if prev > 0.0:
                        pct = ((tvl - prev) / prev) * 100.0
                        if abs(pct) >= 0.25:
                            dir_word = "record TVL inflow surge" if pct > 0 else "TVL outflow drain"
                            events.append({
                                "source_id": sid,
                                "author": "DeFiLlama_OnChain",
                                "text": f"[{bucket_5m}] {name.upper()} on-chain {dir_word} ({pct:+.2f}% change, TVL=${tvl/1e9:.2f}B)",
                                "ts_ms": now_ms,
                                "metrics": {"bookmarks": 10.0, "likes": 15.0}
                            })

        elif ptype == "FEAR_GREED_JSON":
            data = json.loads(raw_text).get("data", [])
            if data:
                val = int(data[0].get("value", 50))
                cls_lbl = str(data[0].get("value_classification", "Neutral"))
                if val >= 72:
                    txt = f"[{bucket_5m}] Crypto Fear and Greed Index in Greed ({val}/100 - {cls_lbl}): bullish momentum & retail euphoria"
                elif val <= 32:
                    txt = f"[{bucket_5m}] Crypto Fear and Greed Index in Fear ({val}/100 - {cls_lbl}): bearish panic selling pressure"
                else:
                    txt = f"[{bucket_5m}] Crypto Fear and Greed Index at {val}/100 ({cls_lbl}) divided market uncertainty"
                events.append({
                    "source_id": sid,
                    "author": "AlternativeMe_Macro",
                    "text": txt,
                    "ts_ms": now_ms,
                    "metrics": {}
                })

        elif ptype == "COINGECKO_TRENDING_JSON":
            coins = json.loads(raw_text).get("coins", [])
            trending_syms = [str(c.get("item", {}).get("symbol", "")).upper() for c in coins[:10]]
            for sym in ("BTC", "ETH", "SOL"):
                if sym in trending_syms:
                    events.append({
                        "source_id": sid,
                        "author": "CoinGecko_Trending",
                        "text": f"[{bucket_5m}] ${sym} surges into top retail trending searches on CoinGecko bullish rally",
                        "ts_ms": now_ms,
                        "metrics": {"likes": 25.0}
                    })

        return events

    def poll_due_api_sources(self) -> list[dict]:
        collected = []
        now_s = time.time()

        if os.path.exists(self.inbox_path):
            try:
                with open(self.inbox_path, "r", encoding="utf-8", errors="replace") as inf:
                    for line in inf:
                        line = line.strip()
                        if line.startswith("{"):
                            collected.append(json.loads(line))
                os.remove(self.inbox_path)
            except Exception:
                pass

        all_apis = (
            self.sources_cfg.get("tier1_fast_1m_apis", []) +
            self.sources_cfg.get("tier2_macro_60m_apis", [])
        )
        cp_token = str(self.sources_cfg.get("optional_api_keys", {}).get("cryptopanic_auth_token", "")).strip()
        if cp_token:
            all_apis.append({
                "source_id": "CRYPTOPANIC_API_V1",
                "url": f"https://cryptopanic.com/api/v1/posts/?auth_token={cp_token}&currencies=BTC,ETH,SOL&filter=important",
                "parser_type": "CRYPTOPANIC_JSON",
                "poll_interval_sec": 60,
            })

        for api_spec in all_apis:
            sid = str(api_spec.get("source_id", ""))
            interval = float(api_spec.get("poll_interval_sec", 60))
            if now_s - self.last_polled_ts.get(sid, 0.0) < interval:
                continue
            self.last_polled_ts[sid] = now_s
            is_red = "REDDIT" in sid.upper()
            try:
                raw_txt = _http_get_text(api_spec["url"], timeout=8, is_reddit=is_red)
                collected.extend(self._parse_api_response(api_spec, raw_txt))
            except Exception:
                # Si Reddit .json dio HTTP 429, usar automaticamente el feed .rss de ese mismo subreddit
                if is_red and ".json" in api_spec["url"]:
                    try:
                        rss_url = api_spec["url"].split(".json")[0] + ".rss"
                        raw_rss = _http_get_text(rss_url, timeout=8, is_reddit=True)
                        fallback_spec = dict(api_spec, parser_type="RSS_XML")
                        collected.extend(self._parse_api_response(fallback_spec, raw_rss))
                    except Exception:
                        pass
                continue

        x_cfg = self.sources_cfg.get("x_accounts_syndication_api", {})
        x_accounts = x_cfg.get("accounts", [])
        batch_sz = int(x_cfg.get("poll_batch_size_per_minute", 2))

        if x_accounts and (now_s - self.last_polled_ts.get("_X_BATCH_", 0.0) >= 60.0):
            self.last_polled_ts["_X_BATCH_"] = now_s
            for _ in range(min(batch_sz, len(x_accounts))):
                acc = x_accounts[self.x_rotation_cursor % len(x_accounts)]
                self.x_rotation_cursor += 1
                handle = str(acc.get("handle", "")).strip().lstrip("@")
                sid    = str(acc.get("source_id", f"X_{handle.upper()}"))
                if not handle:
                    continue
                url_synd = f"https://syndication.twitter.com/srv/timeline-profile/screen-name/{handle}"
                try:
                    html_s = _http_get_text(url_synd, timeout=6)
                    collected.extend(_parse_x_syndication_timeline(html_s, handle, sid))
                except Exception:
                    continue

        return collected

    def ingest_event_batch(self, raw_events: list[dict], current_prices: dict) -> tuple[int, int]:
        self._apply_continuous_time_decay()
        now_s = time.time()

        # Limpiar hashes con mas de 6 horas de antiguedad (TTL = 21,600s)
        if len(self.seen_item_hashes_ts) > 5000:
            self.seen_item_hashes_ts = {
                k: ts for k, ts in self.seen_item_hashes_ts.items() if (now_s - ts) <= 21600.0
            }

        boot_scale = 0.25 if self.is_first_boot else 1.00
        self.is_first_boot = False

        raw_new_count = 0
        ingested = 0
        win_bull = {c: 0.0 for c in ("btc", "eth", "sol")}
        win_bear = {c: 0.0 for c in ("btc", "eth", "sol")}

        for ev in raw_events:
            text = str(ev.get("text", "")).strip()
            if not text:
                continue
            h_key = hashlib.sha256(text[:180].lower().encode("utf-8")).hexdigest()[:16]
            if h_key in self.seen_item_hashes_ts and (now_s - self.seen_item_hashes_ts[h_key]) <= 21600.0:
                continue
            self.seen_item_hashes_ts[h_key] = now_s
            raw_new_count += 1

            res = self.kb.evaluate_message(
                text=text,
                source_id=str(ev.get("source_id", "")),
                author_handle=str(ev.get("author", "")),
                ts_ms=int(ev.get("ts_ms", time.time() * 1000)),
                metrics=ev.get("metrics", {}),
                current_prices=current_prices,
            )
            if not res.get("valid", False):
                continue

            ingested += 1
            for c in ("btc", "eth", "sol"):
                m_vec = res["coin_masses"][c] * boot_scale
                self.e_fast[c] += m_vec
                self.e_slow[c] += m_vec * 0.45
                win_bull[c] += float(m_vec[0])
                win_bear[c] += float(m_vec[1])
                self.euphoria_ema[c] = min(5.0, self.euphoria_ema[c] + float(res["euphoria_score"]) * 0.35 * boot_scale)

            self.contagion_ema = max(self.contagion_ema, float(res["contagion_gate_weight"]))
            net_dir = float(res["coin_masses"]["btc"][0] - res["coin_masses"]["btc"][1])
            if abs(net_dir) > 0.0:
                self.credibility_signal_ema = 0.85 * self.credibility_signal_ema + 0.15 * np.sign(net_dir) * float(res["trust_weight_used"])

        for c in ("btc", "eth", "sol"):
            if win_bull[c] > 0.25 and win_bear[c] > 0.25:
                inter_conflict = 0.85 * math.sqrt(win_bull[c] * win_bear[c])
                self.e_fast[c][2] += inter_conflict
                self.e_slow[c][2] += 0.40 * inter_conflict

        return raw_new_count, ingested

    def build_1m_point_in_time_snapshot(self, ts_ms: int) -> dict:
        self._apply_continuous_time_decay()
        row = {"timestamp_ms": int(ts_ms)}

        for c in ("btc", "eth", "sol"):
            ef = np.maximum(0.0, self.e_fast[c])
            es = np.maximum(0.0, self.e_slow[c])

            S_1m = float(ef[0] + ef[1] + ef[2] + 4.0)
            t_1m = float(ef[0] / S_1m)
            f_1m = float(ef[1] / S_1m)
            b_1m = float(ef[2] / S_1m)
            n_1m = float(4.0 / S_1m)

            S_60m = float(es[0] + es[1] + es[2] + 4.0)
            t_60m = float(es[0] / S_60m)
            f_60m = float(es[1] / S_60m)
            b_60m = float(es[2] / S_60m)

            row[f"{c}_belnap_true_1m"]      = round(t_1m, 6)
            row[f"{c}_belnap_false_1m"]     = round(f_1m, 6)
            row[f"{c}_belnap_both_1m"]      = round(b_1m, 6)
            row[f"{c}_belnap_neither_1m"]   = round(n_1m, 6)
            row[f"{c}_dirichlet_s_1m"]      = round(S_1m, 4)
            row[f"{c}_belnap_net_60m"]      = round(t_60m - f_60m, 6)
            row[f"{c}_belnap_both_60m"]     = round(b_60m, 6)
            row[f"{c}_dirichlet_s_60m"]     = round(S_60m, 4)
            row[f"{c}_social_euphoria_anom"] = round(float(self.euphoria_ema[c]), 4)

        row["triad_contagion_gate_emitted"] = round(float(self.contagion_ema), 4)
        row["memory_credibility_signal_z"]  = round(float(np.clip(self.credibility_signal_ema, -4.0, 4.0)), 4)
        return row

    def append_snapshot_and_flush_parquet_if_needed(self, row: dict) -> None:
        with open(self.wal_path, "a", encoding="utf-8") as wf:
            wf.write(json.dumps(row) + "\n")

        self.wal_buffer_rows.append(row)
        if len(self.wal_buffer_rows) >= 60:
            try:
                df_chunk = pd.DataFrame(self.wal_buffer_rows)
                day_str  = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
                ts_str   = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
                day_dir  = os.path.join(self.belnap_dir, day_str)
                os.makedirs(day_dir, exist_ok=True)
                pq_path  = os.path.join(day_dir, f"belnap_{ts_str}.parquet")
                df_chunk.to_parquet(pq_path, index=False)

                if os.path.exists(self.wal_path):
                    df_wal = pd.read_json(self.wal_path, lines=True, convert_dates=False).tail(120)
                    tmp_w = f"{self.wal_path}.tmp.{os.getpid()}"
                    df_wal.to_json(tmp_w, orient="records", lines=True)
                    os.replace(tmp_w, self.wal_path)
                self.wal_buffer_rows.clear()
            except Exception as e:
                print(f"[WARN] Error al volcar Parquet de Belnap: {e}")


def main():
    _enforce_single_instance_pid_lock(os.path.join(ROOT_DIR, "live"))

    print("================================================================================================================")
    print("=== SKYNET BELNAP 4D COLLECTOR v3.2 (ZERO RATE-LIMIT CDN + PROXIMITY MATCH — BTC / ETH / SOL)                ===")
    print("================================================================================================================")

    collector = BelnapContinuousCollector(ROOT_DIR)
    kb_counts = collector.kb.reload_all_knowledge()
    collector._load_and_register_sources_config()

    print(f"[*] Base de Conocimiento (7 .jsonl) compilada en RAM: {kb_counts}")
    print(f"[*] Grabando serie Point-in-Time en    : '{collector.wal_path}'")

    last_minute_bucket = -1

    try:
        while True:
            try:
                now_ms = int(time.time() * 1000)
                curr_minute_bucket = now_ms // 60000

                if curr_minute_bucket > last_minute_bucket:
                    live_prices = _read_latest_live_prices(collector.live_dir)
                    raw_events  = collector.poll_due_api_sources()
                    n_raw, n_ev = collector.ingest_event_batch(raw_events, live_prices)
                    n_resolved  = collector.kb.resolve_pending_claims_causal(now_ms, live_prices)

                    snap = collector.build_1m_point_in_time_snapshot(curr_minute_bucket * 60000)
                    collector.append_snapshot_and_flush_parquet_if_needed(snap)

                    ts_utc = str(pd.to_datetime(snap["timestamp_ms"], unit="ms", utc=True))
                    print(
                        f"[{ts_utc}] Raw={n_raw:2d}->Ev={n_ev:2d} | "
                        f"BTC(T={snap['btc_belnap_true_1m']:.2f},F={snap['btc_belnap_false_1m']:.2f},B={snap['btc_belnap_both_1m']:.2f},N={snap['btc_belnap_neither_1m']:.2f},S={snap['btc_dirichlet_s_1m']:.1f}) | "
                        f"ETH(T={snap['eth_belnap_true_1m']:.2f},F={snap['eth_belnap_false_1m']:.2f},B={snap['eth_belnap_both_1m']:.2f},N={snap['eth_belnap_neither_1m']:.2f},S={snap['eth_dirichlet_s_1m']:.1f}) | "
                        f"SOL(T={snap['sol_belnap_true_1m']:.2f},F={snap['sol_belnap_false_1m']:.2f},B={snap['sol_belnap_both_1m']:.2f},N={snap['sol_belnap_neither_1m']:.2f},S={snap['sol_dirichlet_s_1m']:.1f})"
                    )
                    last_minute_bucket = curr_minute_bucket
            except Exception as loop_err:
                print(f"[WARN] Error recuperable en ciclo de 1m: {loop_err}")

            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[*] Colector Belnap 4D detenido manualmente.")


if __name__ == "__main__":
    main()