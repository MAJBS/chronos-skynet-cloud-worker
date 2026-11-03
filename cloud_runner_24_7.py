# cloud_runner_24_7.py
import os
import sys
import json
import time
import threading
import subprocess
import urllib.request
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_DIR = os.path.join(ROOT_DIR, "live")
os.makedirs(os.path.join(LIVE_DIR, "belnap"), exist_ok=True)
os.makedirs(os.path.join(LIVE_DIR, "consolidated"), exist_ok=True)

HF_TOKEN     = os.environ.get("HF_WRITE_TOKEN", "")
GH_TOKEN     = os.environ.get("GH_PAT_TOKEN", "")
DATASET_REPO = os.environ.get("HF_DATASET_REPO", "Maycol127/chronos-skynet-vault")
GH_REPO_FULL = "MAJBS/chronos-skynet-cloud-worker"
MAX_RUN_SEC  = float(os.environ.get("MAX_JOB_SECONDS", "19200"))


def restore_from_vault():
    if not HF_TOKEN:
        return
    for fname, wal_name in [
        ("btc_clean_f64.parquet", "wal_BTC_active.jsonl"),
        ("eth_clean_f64.parquet", "wal_ETH_active.jsonl"),
        ("sol_clean_f64.parquet", "wal_SOL_active.jsonl"),
        ("belnap_master_f64.parquet", "wal_BELNAP_active.jsonl"),
    ]:
        try:
            p = hf_hub_download(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                filename=f"consolidated/{fname}",
                local_dir=LIVE_DIR,
                token=HF_TOKEN
            )
            df = pd.read_parquet(p)
            wal_p = os.path.join(LIVE_DIR, wal_name)
            df.tail(120).to_json(wal_p, orient="records", lines=True)
            print(f"[CLOUD RESTORE] {fname}: {len(df):,} barras restauradas desde {DATASET_REPO}")
        except Exception as e:
            print(f"[CLOUD INFO] {fname}: {e}")


def flush_all_to_vault():
    if not HF_TOKEN:
        return
    api = HfApi(token=HF_TOKEN)
    for fname, wal_name in [
        ("btc_clean_f64.parquet", "wal_BTC_active.jsonl"),
        ("eth_clean_f64.parquet", "wal_ETH_active.jsonl"),
        ("sol_clean_f64.parquet", "wal_SOL_active.jsonl"),
        ("belnap_master_f64.parquet", "wal_BELNAP_active.jsonl"),
    ]:
        try:
            wal_p    = os.path.join(LIVE_DIR, wal_name)
            master_p = os.path.join(LIVE_DIR, "consolidated", fname)
            dfs = []
            if os.path.exists(master_p):
                dfs.append(pd.read_parquet(master_p))
            if os.path.exists(wal_p):
                dfs.append(pd.read_json(wal_p, lines=True, convert_dates=False))
            if dfs:
                merged = (
                    pd.concat(dfs, ignore_index=True)
                    .sort_values("timestamp_ms", kind="mergesort")
                    .drop_duplicates(subset=["timestamp_ms"], keep="last")
                    .reset_index(drop=True)
                )
                merged.to_parquet(master_p, index=False)
                api.upload_file(
                    path_or_fileobj=master_p,
                    path_in_repo=f"consolidated/{fname}",
                    repo_id=DATASET_REPO,
                    repo_type="dataset",
                    commit_message=f"Cloud Auto-Backup {fname} ({len(merged)} bars)",
                    token=HF_TOKEN
                )
                merged.tail(120).to_json(wal_p, orient="records", lines=True)
                print(f"[CLOUD BACKUP OK] {fname}: {len(merged):,} barras guardadas en {DATASET_REPO}")
        except Exception as e:
            print(f"[CLOUD BACKUP WARN {fname}] {e}")


def trigger_next_relay_shift():
    if not GH_TOKEN:
        return
    url = f"https://api.github.com/repos/{GH_REPO_FULL}/actions/workflows/chronos_24_7.yml/dispatches"
    req = urllib.request.Request(
        url,
        data=json.dumps({"ref": "main"}).encode("utf-8"),
        headers={
            "Authorization": f"token {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "ChronosRelay/3.3",
            "Content-Type": "application/json",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as _:
            print("[CLOUD RELAY] Siguiente turno de 5.5h disparado automaticamente.")
    except Exception as e:
        print(f"[CLOUD RELAY INFO] {e}")


def run_daemon_forever(script_rel_path: str):
    while True:
        proc = subprocess.Popen([sys.executable, script_rel_path])
        proc.wait()
        time.sleep(5)


if __name__ == "__main__":
    t_start = time.time()
    restore_from_vault()
    threading.Thread(target=run_daemon_forever, args=("data_collector/cloud_market_24col_daemon.py",), daemon=True).start()
    threading.Thread(target=run_daemon_forever, args=("data_collector/belnap_collector_daemon.py",), daemon=True).start()

    time.sleep(125)
    print("[CLOUD] Ejecutando primer volcado rapido de verificacion a Hugging Face Vault...")
    flush_all_to_vault()

    last_backup = time.time()
    while time.time() - t_start < MAX_RUN_SEC:
        time.sleep(25)
        if time.time() - last_backup >= 600:
            flush_all_to_vault()
            last_backup = time.time()

    print("[CLOUD] Cerrando turno, guardando en Boveda y encendiendo el siguiente relevo 24/7...")
    flush_all_to_vault()
    trigger_next_relay_shift()
