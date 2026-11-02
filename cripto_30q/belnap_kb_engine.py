# cripto_30q/belnap_kb_engine.py
# Motor Simbolico-Vectorial de Baja Latencia (<0.5 ms) para los 7 Archivos .jsonl (Belnap 4D + Dirichlet EDL)
# Incluye:
#   1. Coincidencia Exacta sobre `03_belnap_dirichlet_evidential_lexicon.jsonl` (Peso 1.00x).
#   2. Coincidencia por Proximidad de Palabras Clave del Lexico 03 cuando hay cifras/verbos en medio (Peso 0.75x).
#   3. Diccionario de Raices Financieras Cripto EN/ES de Respaldo para titulares reales de prensa/exchanges.

import os
import re
import json
import math
import time
import hashlib
import numpy as np
import pandas as pd


KB_FILENAMES = {
    "sources":    "01_sources_and_endpoints_registry.jsonl",
    "entities":   "02_entity_and_spillover_router.jsonl",
    "lexicon":    "03_belnap_dirichlet_evidential_lexicon.jsonl",
    "syntax":     "04_paraconsistent_syntax_and_sarcasm_rules.jsonl",
    "heuristics": "05_engagement_and_disagreement_heuristics.jsonl",
    "archetypes": "06_historical_market_shock_archetypes.jsonl",
    "audit":      "07_claims_and_credibility_audit_rules.jsonl",
}

STOPWORDS_EN_ES = {
    "the", "and", "for", "with", "from", "into", "over", "under", "after", "before",
    "that", "this", "have", "has", "had", "will", "would", "could", "should", "about",
    "amid", "near", "upon", "while", "where", "when", "what", "which", "their", "there",
    "para", "como", "pero", "sobre", "entre", "hasta", "desde", "cuando", "donde", "este",
    "esta", "estos", "estas", "todo", "toda", "todos", "tras", "ante", "bajo", "contra",
    "crypto", "token", "tokens", "coin", "coins", "market", "markets", "price", "prices"
}

# Raices financieras universales (EN/ES) para asegurar captura de titulares reales cuando usan sinonimos
CORE_FINANCIAL_STEMS = [
    (re.compile(r"\b(inflow\w*|accumulat\w*|buyback\w*|bullish\w*|breakout\w*|surging|surge[ds]?|rallie[ds]|rally\w*|soar\w*|skyrocket\w*|all-time high|ath\b|approv\w*|adopt\w*|burn\w*|upgrade\w*|institutional buy\w*|net taker buy\w*|top traders long)\b", re.I),
     np.array([1.15, 0.0, 0.10], dtype=np.float64)),
    (re.compile(r"\b(outflow\w*|dump\w*|bearish\w*|crash\w*|plung\w*|plummet\w*|slump\w*|tumble[ds]?|hack\w*|exploit\w*|drain\w*|stolen|suspend\w*|halt\w*|insolven\w*|bankrupt\w*|lawsuit\w*|sued|reject\w*|delist\w*|unlock\w*|sell-off|selloff|net taker sell\w*|top traders short|extreme fear)\b", re.I),
     np.array([0.0, 1.20, 0.15], dtype=np.float64)),
    (re.compile(r"\b(volatile|volatility|choppy|uncertain\w*|divided|disagree\w*|conflict\w*|rumor\w*|unconfirmed|debate\w*|mixed signals|war\b|tariff\w*|squeeze|liquidat\w*|whipsaw|divergence|crowded)\b", re.I),
     np.array([0.15, 0.15, 0.95], dtype=np.float64)),
]


def _safe_load_jsonl(filepath: str, id_keys: tuple[str, ...]) -> list[dict]:
    if not os.path.exists(filepath):
        return []

    records_by_id = {}
    unnamed_records = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("```") or line.startswith("//") or line.startswith("#"):
                continue
            if not (line.startswith("{") and line.endswith("}")):
                start_brace = line.find("{")
                end_brace   = line.rfind("}")
                if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                    line = line[start_brace : end_brace + 1]
                else:
                    continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                rec_id = None
                for k in id_keys:
                    if k in obj and obj[k]:
                        rec_id = str(obj[k]).strip()
                        break
                if rec_id:
                    records_by_id[rec_id] = obj
                else:
                    unnamed_records.append(obj)
            except Exception:
                continue

    return list(records_by_id.values()) + unnamed_records


def _upgrade_cashtag_regex(raw_regex: str, fallback_aliases: list[str]) -> re.Pattern:
    if raw_regex and isinstance(raw_regex, str):
        cleaned = raw_regex.strip()
        if cleaned.startswith("\\b(") and cleaned.endswith(")\\b"):
            inner = cleaned[3:-3]
            upgraded = f"(?<![a-zA-Z0-9_])({inner})(?![a-zA-Z0-9_])"
        else:
            upgraded = cleaned.replace("\\b", "(?:(?<![a-zA-Z0-9_])|(?![a-zA-Z0-9_]))")
        try:
            return re.compile(upgraded, re.IGNORECASE)
        except Exception:
            pass

    escaped = [re.escape(str(a).strip()) for a in fallback_aliases if str(a).strip()]
    if not escaped:
        return re.compile(r"^\b$", re.IGNORECASE)
    escaped.sort(key=len, reverse=True)
    pat = f"(?<![a-zA-Z0-9_])({'|'.join(escaped)})(?![a-zA-Z0-9_])"
    return re.compile(pat, re.IGNORECASE)


def _extract_keyword_sets_from_patterns(patterns_lower: list[str]) -> list[tuple[str, ...]]:
    """
    Extrae tuplas de palabras significativas (>= 2 tokens de >= 4 letras) de cada frase
    del Lexico 03 para permitir coincidencias por proximidad aunque haya cifras en medio.
    """
    kw_sets = []
    seen = set()
    for p in patterns_lower:
        tokens = [
            t[:6] if len(t) >= 7 else t
            for t in re.findall(r"[a-zA-Z0-9_]{3,}", p)
            if t not in STOPWORDS_EN_ES and len(t) >= 3
        ]
        if len(tokens) >= 2:
            key_tup = tuple(sorted(set(tokens[:3])))
            if len(key_tup) >= 2 and key_tup not in seen:
                seen.add(key_tup)
                kw_sets.append(key_tup)
    return kw_sets


class BelnapKnowledgeBase:
    def __init__(self, kb_dir: str, memory_dir: str):
        self.kb_dir     = kb_dir
        self.memory_dir = memory_dir
        os.makedirs(self.kb_dir, exist_ok=True)
        os.makedirs(self.memory_dir, exist_ok=True)

        self.ledger_08_path      = os.path.join(self.memory_dir, "08_infinite_memory_pit_ledger.jsonl")
        self.credibility_09_path = os.path.join(self.memory_dir, "09_dynamic_accounts_credibility_state.jsonl")

        self.sources_by_id     = {}
        self.sources_by_handle = {}
        self.entities_compiled = []
        self.lexicon_compiled  = []
        self.syntax_compiled   = []
        self.heuristics_list   = []
        self.archetypes_list   = []
        self.audit_extractors  = []
        self.audit_updaters    = []

        self.credibility_state = {}
        self.pending_claims    = []

        self.reload_all_knowledge()

    def reload_all_knowledge(self) -> dict[str, int]:
        # 1. Cargar 01_sources_and_endpoints_registry.jsonl
        raw_sources = _safe_load_jsonl(
            os.path.join(self.kb_dir, KB_FILENAMES["sources"]),
            ("id_source", "source_id")
        )
        self.sources_by_id.clear()
        self.sources_by_handle.clear()
        for s in raw_sources:
            sid = str(s.get("id_source") or s.get("source_id") or f"SRC_{len(self.sources_by_id)}")
            handle = str(s.get("target_endpoint_or_handle") or s.get("target_url_or_handle") or "").strip().lower()
            entry = {
                "id_source": sid,
                "nombre_fuente": str(s.get("nombre_fuente", sid)),
                "transport_mode": str(s.get("transport_mode", "REST_JSON")),
                "target": handle,
                "paths": s.get("json_or_dom_paths") or s.get("extraction_spec") or {},
                "prior_trust_weight": float(np.clip(s.get("prior_trust_weight", 1.5), 0.05, 5.0)),
                "horizon_scope": list(s.get("horizon_scope", ["1m", "60m"])),
                "asset_affinity": [str(a).lower() for a in s.get("asset_affinity", ["btc", "eth", "sol"])],
                "is_contrarian_retail_gauge": bool(s.get("is_contrarian_retail_gauge", False)),
                "poll_interval_sec": int(max(5, s.get("poll_interval_sec", 30))),
            }
            self.sources_by_id[sid] = entry
            if handle:
                self.sources_by_handle[handle] = entry

        # 2. Cargar 02_entity_and_spillover_router.jsonl
        raw_entities = _safe_load_jsonl(
            os.path.join(self.kb_dir, KB_FILENAMES["entities"]),
            ("id_entity", "entity_id")
        )
        self.entities_compiled.clear()
        for e in raw_entities:
            eid = str(e.get("id_entity") or e.get("entity_id") or f"ENT_{len(self.entities_compiled)}")
            aliases = [str(x).strip() for x in e.get("match_aliases_and_cashtags", []) if str(x).strip()]
            rx = _upgrade_cashtag_regex(str(e.get("regex_word_boundary", "")), aliases)
            betas_raw = e.get("spillover_betas", [0.5, 0.5, 0.5])
            if isinstance(betas_raw, dict):
                betas = np.array([
                    float(betas_raw.get("btc", 0.5)),
                    float(betas_raw.get("eth", 0.5)),
                    float(betas_raw.get("sol", 0.5)),
                ], dtype=np.float64)
            else:
                b_list = list(betas_raw) + [0.0, 0.0, 0.0]
                betas = np.array(b_list[:3], dtype=np.float64)

            self.entities_compiled.append({
                "id_entity": eid,
                "canonical_name": str(e.get("canonical_name", eid)),
                "primary_asset": str(e.get("primary_asset", "macro_all")).lower(),
                "target_action": str(e.get("target_action", "ROUTE_TO_TENSORS")).upper(),
                "regex": rx,
                "aliases_lower": [a.lower() for a in aliases],
                "spillover_betas": np.clip(betas, 0.0, 2.0),
                "downside_asymmetry_mult": float(np.clip(e.get("downside_asymmetry_mult", 1.25), 0.5, 3.0)),
                "contagion_gate_weight": float(np.clip(e.get("contagion_gate_weight", 0.50), 0.0, 1.0)),
            })

        # 3. Cargar 03_belnap_dirichlet_evidential_lexicon.jsonl
        raw_lexicon = _safe_load_jsonl(
            os.path.join(self.kb_dir, KB_FILENAMES["lexicon"]),
            ("id_lexico", "term_id", "concept_id")
        )
        self.lexicon_compiled.clear()
        for lx in raw_lexicon:
            lid = str(lx.get("id_lexico") or lx.get("term_id") or f"LEX_{len(self.lexicon_compiled)}")
            pats = lx.get("patterns_en_es") or lx.get("patterns") or lx.get("phrases_en_es") or []
            pats_clean = [str(p).strip().lower() for p in pats if str(p).strip()]
            if not pats_clean:
                continue

            masses_raw = lx.get("evidential_masses", [0.0, 0.0, 0.0])
            if isinstance(masses_raw, (list, tuple)) and len(masses_raw) >= 3:
                masses = np.maximum(0.0, np.array(masses_raw[:3], dtype=np.float64))
            else:
                masses = np.maximum(0.0, np.array([
                    float(lx.get("de_T", lx.get("de_true", 0.0))),
                    float(lx.get("de_F", lx.get("de_false", 0.0))),
                    float(lx.get("de_B", lx.get("de_both", 0.0))),
                ], dtype=np.float64))

            hl_fast = float(np.clip(lx.get("half_life_fast_1m_min", 5.0), 0.5, 120.0))
            hl_slow = float(np.clip(lx.get("half_life_slow_60m_min", 60.0), 5.0, 720.0))

            self.lexicon_compiled.append({
                "id_lexico": lid,
                "familia": str(lx.get("familia_catalizador", "GENERAL")),
                "patterns_lower": pats_clean,
                "keyword_tuples": _extract_keyword_sets_from_patterns(pats_clean),
                "regex": _upgrade_cashtag_regex("", pats_clean),
                "masses": masses,
                "hl_fast_min": hl_fast,
                "hl_slow_min": hl_slow,
                "bot_shill_score": float(np.clip(lx.get("bot_shill_score", 0.0), 0.0, 1.0)),
                "euphoria_exhaustion_score": float(np.clip(lx.get("euphoria_exhaustion_score", 0.0), 0.0, 1.0)),
                "requires_order0_guard": bool(lx.get("requires_order0_bilattice_guard", True)),
                "synergy_scenario": str(lx.get("target_synergy_scenario", "NEUTRAL")),
            })

        # 4. Cargar 04_paraconsistent_syntax_and_sarcasm_rules.jsonl
        raw_syntax = _safe_load_jsonl(
            os.path.join(self.kb_dir, KB_FILENAMES["syntax"]),
            ("id_syntax_rule", "rule_id")
        )
        self.syntax_compiled.clear()
        for sr in raw_syntax:
            s_id = str(sr.get("id_syntax_rule") or sr.get("rule_id") or f"SYN_{len(self.syntax_compiled)}")
            trigs = [str(t).strip().lower() for t in sr.get("trigger_phrases_en_es", []) if str(t).strip()]
            if not trigs:
                continue
            mat_raw = sr.get("transform_matrix_3x3", [1, 0, 0, 0, 1, 0, 0, 0, 1])
            if len(mat_raw) == 9:
                mat = np.array(mat_raw, dtype=np.float64).reshape(3, 3)
            else:
                mat = np.eye(3, dtype=np.float64)

            self.syntax_compiled.append({
                "id_syntax_rule": s_id,
                "tipo_operador": str(sr.get("tipo_operador", "MODIFIER")),
                "triggers_lower": trigs,
                "scope_direction": str(sr.get("scope_direction", "BIDIRECTIONAL")).upper(),
                "max_char_distance": int(max(15, int(sr.get("max_token_distance", 6)) * 12)),
                "matrix_3x3": mat,
                "additive_both_boost": float(np.clip(sr.get("additive_both_boost", 0.0), 0.0, 1.0)),
                "override": str(sr.get("belnap_target_state_override", "NONE")).upper(),
            })

        # 5. Cargar 05_engagement_and_disagreement_heuristics.jsonl
        self.heuristics_list = _safe_load_jsonl(
            os.path.join(self.kb_dir, KB_FILENAMES["heuristics"]),
            ("id_heuristic", "heuristic_id")
        )

        # 6. Cargar 06_historical_market_shock_archetypes.jsonl
        self.archetypes_list = _safe_load_jsonl(
            os.path.join(self.kb_dir, KB_FILENAMES["archetypes"]),
            ("id_archetype", "archetype_id")
        )

        # 7. Cargar 07_claims_and_credibility_audit_rules.jsonl
        raw_audit = _safe_load_jsonl(
            os.path.join(self.kb_dir, KB_FILENAMES["audit"]),
            ("id_audit_rule", "rule_id")
        )
        self.audit_extractors.clear()
        self.audit_updaters.clear()
        for ar in raw_audit:
            rtype = str(ar.get("tipo_regla", "DIRECTIONAL_CALL_EXTRACTOR")).upper()
            pats = [str(p).strip().lower() for p in ar.get("match_patterns_en_es", []) if str(p).strip()]
            ar_clean = {
                "id_audit_rule": str(ar.get("id_audit_rule", f"AUD_{len(self.audit_extractors)}")),
                "tipo_regla": rtype,
                "patterns_lower": pats,
                "claimed_polarity": int(ar.get("claimed_polarity", 1)),
                "horizons_min": list(ar.get("evaluation_horizons_min", [15, 60, 1440])),
                "min_ortho_bps": float(ar.get("min_ortho_return_bps_for_win", 20.0)),
                "alpha_boost": float(ar.get("bayesian_alpha_win_boost", 0.15)),
                "beta_penalty": float(ar.get("bayesian_beta_loss_penalty", 0.20)),
                "contrarian_flip_th": float(ar.get("auto_flip_to_contrarian_hit_rate_below", 0.38)),
                "alert_template": str(ar.get("memory_alert_template", "")),
            }
            if "UPDATER" in rtype:
                self.audit_updaters.append(ar_clean)
            elif pats:
                self.audit_extractors.append(ar_clean)

        self._load_dynamic_memory_state()

        return {
            "sources": len(self.sources_by_id),
            "entities": len(self.entities_compiled),
            "lexicon": len(self.lexicon_compiled),
            "syntax": len(self.syntax_compiled),
            "heuristics": len(self.heuristics_list),
            "archetypes": len(self.archetypes_list),
            "audit_rules": len(self.audit_extractors) + len(self.audit_updaters),
        }

    def _load_dynamic_memory_state(self) -> None:
        self.credibility_state.clear()
        if os.path.exists(self.credibility_09_path):
            rows = _safe_load_jsonl(self.credibility_09_path, ("account_key",))
            for r in rows:
                k = str(r.get("account_key", "")).strip().lower()
                if k:
                    self.credibility_state[k] = r

        self.pending_claims.clear()
        if os.path.exists(self.ledger_08_path):
            rows = _safe_load_jsonl(self.ledger_08_path, ("claim_id",))
            for r in rows:
                if r.get("status") == "PENDING_EVAL":
                    self.pending_claims.append(r)

    def get_source_trust_and_contrarian(self, source_id: str, author_handle: str) -> tuple[float, bool]:
        key = str(author_handle or source_id or "unknown").strip().lower()
        base_w = 1.25
        is_contra = False

        if source_id in self.sources_by_id:
            s_cfg = self.sources_by_id[source_id]
            base_w    = s_cfg["prior_trust_weight"]
            is_contra = s_cfg["is_contrarian_retail_gauge"]
        elif key in self.sources_by_handle:
            s_cfg = self.sources_by_handle[key]
            base_w    = s_cfg["prior_trust_weight"]
            is_contra = s_cfg["is_contrarian_retail_gauge"]

        if key in self.credibility_state:
            st = self.credibility_state[key]
            base_w    = float(np.clip(st.get("dynamic_trust_weight", base_w), 0.10, 5.00))
            is_contra = bool(st.get("is_contrarian_retail_gauge", is_contra))

        return base_w, is_contra

    def _apply_syntax_modifiers(self, vec: np.ndarray, text_lower: str, span_start: int, span_end: int) -> np.ndarray:
        out_vec = vec.copy()
        for sr in self.syntax_compiled:
            d_char = sr["max_char_distance"]
            if sr["scope_direction"] == "PRECEDING":
                ctx_sub = text_lower[max(0, span_start - d_char) : span_start]
            elif sr["scope_direction"] == "FOLLOWING":
                ctx_sub = text_lower[span_end : min(len(text_lower), span_end + d_char)]
            elif sr["scope_direction"] == "FULL_MESSAGE":
                ctx_sub = text_lower
            else:
                ctx_sub = text_lower[max(0, span_start - d_char) : min(len(text_lower), span_end + d_char)]

            if any(trig in ctx_sub for trig in sr["triggers_lower"]):
                out_vec = np.maximum(0.0, sr["matrix_3x3"] @ out_vec)
                out_vec[2] += min(0.35, sr["additive_both_boost"])
                if sr["override"] == "FORCE_BOTH":
                    out_vec[2] += 0.65 * (out_vec[0] + out_vec[1])
                    out_vec[0] *= 0.35
                    out_vec[1] *= 0.35
                elif sr["override"] == "FORCE_NEITHER":
                    out_vec[:] = 0.0
                break
        return out_vec

    def evaluate_message(
        self,
        text: str,
        source_id: str = "",
        author_handle: str = "",
        ts_ms: int = None,
        metrics: dict = None,
        current_prices: dict = None
    ) -> dict:
        if not text or not isinstance(text, str):
            return {"valid": False, "reason": "EMPTY_TEXT"}

        if ts_ms is None:
            ts_ms = int(time.time() * 1000)
        if metrics is None:
            metrics = {}

        text_clean = text.strip()
        text_lower = text_clean.lower()

        # 1. Enrutamiento de Entidades y Derrame Beta (JSONL 02)
        matched_entities = []
        has_drop_only = False

        for ent in self.entities_compiled:
            if ent["regex"].search(text_clean):
                if ent["target_action"] == "DROP_IRRELEVANT":
                    has_drop_only = True
                else:
                    matched_entities.append(ent)

        if not matched_entities:
            if re.search(r"(?<![a-zA-Z0-9_])(\$?btc|bitcoin)(?![a-zA-Z0-9_])", text_lower):
                matched_entities.append({
                    "id_entity": "FALLBACK_BTC",
                    "primary_asset": "btc",
                    "spillover_betas": np.array([1.0, 0.65, 0.70], dtype=np.float64),
                    "downside_asymmetry_mult": 1.35,
                    "contagion_gate_weight": 0.60,
                })
            if re.search(r"(?<![a-zA-Z0-9_])(\$?eth|ethereum)(?![a-zA-Z0-9_])", text_lower):
                matched_entities.append({
                    "id_entity": "FALLBACK_ETH",
                    "primary_asset": "eth",
                    "spillover_betas": np.array([0.20, 1.0, 0.40], dtype=np.float64),
                    "downside_asymmetry_mult": 1.25,
                    "contagion_gate_weight": 0.45,
                })
            if re.search(r"(?<![a-zA-Z0-9_])(\$?sol|solana)(?![a-zA-Z0-9_])", text_lower):
                matched_entities.append({
                    "id_entity": "FALLBACK_SOL",
                    "primary_asset": "sol",
                    "spillover_betas": np.array([0.15, 0.30, 1.0], dtype=np.float64),
                    "downside_asymmetry_mult": 1.35,
                    "contagion_gate_weight": 0.45,
                })

        if not matched_entities:
            if has_drop_only:
                return {"valid": False, "reason": "DROPPED_IRRELEVANT_ALT"}
            matched_entities.append({
                "id_entity": "GENERAL_MARKET_MACRO",
                "primary_asset": "macro_all",
                "spillover_betas": np.array([0.75, 0.65, 0.70], dtype=np.float64),
                "downside_asymmetry_mult": 1.30,
                "contagion_gate_weight": 0.50,
            })

        spillover_vec = np.max([e["spillover_betas"] for e in matched_entities], axis=0)
        downside_mult = float(np.max([e["downside_asymmetry_mult"] for e in matched_entities]))
        contagion_w   = float(np.max([e["contagion_gate_weight"] for e in matched_entities]))

        # 2A. Nivel 1: Escaneo Lexico Exacto sobre JSONL 03
        total_masses = np.zeros(3, dtype=np.float64)
        bot_scores   = []
        euph_scores  = []
        hl_fast_list = []
        hl_slow_list = []
        order0_guard = False
        matched_lex_ids = set()

        for lx in self.lexicon_compiled:
            m_iter = list(lx["regex"].finditer(text_clean))
            if not m_iter:
                continue
            matched_lex_ids.add(lx["id_lexico"])
            for m_obj in m_iter:
                span_start, span_end = m_obj.span()
                vec = self._apply_syntax_modifiers(lx["masses"], text_lower, span_start, span_end)
                total_masses += vec
                bot_scores.append(lx["bot_shill_score"])
                euph_scores.append(lx["euphoria_exhaustion_score"])
                hl_fast_list.append(lx["hl_fast_min"])
                hl_slow_list.append(lx["hl_slow_min"])
                if lx["requires_order0_guard"]:
                    order0_guard = True

        # 2B. Nivel 2: Coincidencia por Proximidad de Palabras Clave del JSONL 03 (cuando hay cifras en medio)
        if len(matched_lex_ids) < 2:
            for lx in self.lexicon_compiled:
                if lx["id_lexico"] in matched_lex_ids or not lx["keyword_tuples"]:
                    continue
                for kw_tup in lx["keyword_tuples"]:
                    if all(kw in text_lower for kw in kw_tup):
                        pos_first = text_lower.find(kw_tup[0])
                        pos_last  = text_lower.find(kw_tup[-1])
                        if abs(pos_last - pos_first) <= 85:
                            vec = self._apply_syntax_modifiers(
                                lx["masses"] * 0.75, text_lower, min(pos_first, pos_last), max(pos_first, pos_last) + 10
                            )
                            total_masses += vec
                            bot_scores.append(lx["bot_shill_score"])
                            euph_scores.append(lx["euphoria_exhaustion_score"])
                            hl_fast_list.append(lx["hl_fast_min"])
                            hl_slow_list.append(lx["hl_slow_min"])
                            if lx["requires_order0_guard"]:
                                order0_guard = True
                            matched_lex_ids.add(lx["id_lexico"])
                            break

        # 2C. Nivel 3: Raices Financieras Cripto EN/ES (si el titular uso sinonimos financieros directos)
        if float(np.sum(total_masses)) <= 1e-6:
            for stem_rx, stem_vec in CORE_FINANCIAL_STEMS:
                for m_stem in stem_rx.finditer(text_clean):
                    s_ini, s_fin = m_stem.span()
                    vec = self._apply_syntax_modifiers(stem_vec * 0.65, text_lower, s_ini, s_fin)
                    total_masses += vec
                    order0_guard = True

        # GUARDIA DE VACUIDAD EPISTEMICA (Pag. 4 del PDF):
        if float(np.sum(total_masses)) <= 1e-6:
            return {"valid": False, "reason": "NO_LEXICON_CATALYST_HIT"}

        mean_bot  = float(np.mean(bot_scores)) if bot_scores else 0.0
        mean_euph = float(np.mean(euph_scores)) if euph_scores else 0.0

        if mean_bot > 0.0:
            total_masses[0] *= max(0.0, 1.0 - mean_bot)

        # Guardia de Colision Paraconsistente de Orden-0 (Belnap-Dunn / SVNS, Pag. 3 del PDF)
        if order0_guard and total_masses[0] > 0.05 and total_masses[1] > 0.05:
            overlap = min(total_masses[0], total_masses[1])
            total_masses[2] += 1.25 * overlap

        # 3. Modulacion por Engagement y Desacuerdo Inter-Inversor (JSONL 05) acotada a la masa lexica
        pos_likes = float(metrics.get("likes", metrics.get("votes_pos", 0.0)))
        neg_repl  = float(metrics.get("replies", metrics.get("votes_neg", 0.0)))
        bookmarks = float(metrics.get("bookmarks", metrics.get("votes_important", 0.0)))
        lol_memes = float(metrics.get("votes_lol", 0.0))
        tot_votes = pos_likes + neg_repl + bookmarks + lol_memes

        if tot_votes > 0.0 and (lol_memes / tot_votes) >= 0.35:
            return {"valid": False, "reason": "MEME_LOL_FILTERED"}

        eng_mult = 1.0
        if pos_likes >= 5.0 and (bookmarks / pos_likes) >= 0.30:
            eng_mult *= 1.40

        directional_mass = total_masses[0] + total_masses[1]
        if (pos_likes + neg_repl) >= 6.0 and directional_mass > 0.0:
            imb_ratio = abs(pos_likes - neg_repl) / (pos_likes + neg_repl)
            agreement_idx = 1.0 - math.sqrt(max(0.0, 1.0 - imb_ratio * imb_ratio))
            raw_dis = (1.0 - agreement_idx) * 0.30 * directional_mass
            total_masses[2] += min(0.35 * directional_mass, raw_dis)

        total_masses[:2] *= eng_mult

        # 4. Ponderacion Bayesiana de la Fuente (JSONL 01, 07 y 09)
        trust_w, is_contrarian = self.get_source_trust_and_contrarian(source_id, author_handle)
        if is_contrarian and total_masses[0] > 0.0:
            mean_euph = max(mean_euph, min(1.0, 0.50 + 0.35 * total_masses[0]))
            total_masses[2] += 0.30 * total_masses[0]
            total_masses[0] *= 0.20

        # 5. Proyeccion por Moneda (BTC, ETH, SOL) con Asimetria Bajista
        is_bearish_dominant = total_masses[1] > total_masses[0]
        coin_masses = {}
        for idx_c, c_name in enumerate(("btc", "eth", "sol")):
            beta_c = float(spillover_vec[idx_c])
            if is_bearish_dominant and c_name in ("eth", "sol"):
                beta_c *= downside_mult
            coin_masses[c_name] = trust_w * beta_c * total_masses

        # 6. Registro de Llamadas en la Memoria Infinita (JSONL 07 -> 08)
        claims_logged = 0
        if current_prices:
            for ar in self.audit_extractors:
                if any(pat in text_lower for pat in ar["patterns_lower"]):
                    primary_c = matched_entities[0]["primary_asset"]
                    if primary_c not in ("btc", "eth", "sol"):
                        primary_c = "btc"
                    claim_id = hashlib.sha256(f"{ts_ms}_{author_handle}_{text_clean[:64]}".encode("utf-8")).hexdigest()[:16]
                    claim_rec = {
                        "claim_id": claim_id,
                        "ts_ms": int(ts_ms),
                        "utc": str(pd.to_datetime(ts_ms, unit="ms", utc=True)),
                        "source_id": str(source_id),
                        "account_key": str(author_handle or source_id).strip().lower(),
                        "target_coin": primary_c,
                        "claimed_polarity": int(ar["claimed_polarity"]),
                        "rule_id": ar["id_audit_rule"],
                        "entry_prices": {
                            "btc": float(current_prices.get("btc", 0.0)),
                            "eth": float(current_prices.get("eth", 0.0)),
                            "sol": float(current_prices.get("sol", 0.0)),
                        },
                        "horizons_min": ar["horizons_min"],
                        "min_ortho_bps": ar["min_ortho_bps"],
                        "alpha_boost": ar["alpha_boost"],
                        "beta_penalty": ar["beta_penalty"],
                        "contrarian_flip_th": ar["contrarian_flip_th"],
                        "status": "PENDING_EVAL",
                        "excerpt": text_clean[:160],
                    }
                    self.pending_claims.append(claim_rec)
                    try:
                        with open(self.ledger_08_path, "a", encoding="utf-8") as lf:
                            lf.write(json.dumps(claim_rec) + "\n")
                        claims_logged += 1
                    except Exception:
                        pass
                    break

        return {
            "valid": True,
            "ts_ms": int(ts_ms),
            "trust_weight_used": trust_w,
            "is_contrarian_source": is_contrarian,
            "coin_masses": coin_masses,
            "hl_fast_min": float(np.median(hl_fast_list)) if hl_fast_list else 5.0,
            "hl_slow_min": float(np.median(hl_slow_list)) if hl_slow_list else 60.0,
            "euphoria_score": mean_euph,
            "bot_shill_score": mean_bot,
            "contagion_gate_weight": contagion_w if is_bearish_dominant or total_masses[2] > 0.8 else 0.0,
            "claims_logged": claims_logged,
        }

    def resolve_pending_claims_causal(self, now_ts_ms: int, current_prices: dict) -> int:
        if not self.pending_claims or not current_prices:
            return 0

        p_btc_now = float(current_prices.get("btc", 0.0))
        if p_btc_now <= 0.0:
            return 0

        resolved_count = 0
        still_pending  = []

        for cl in self.pending_claims:
            eval_h_min = int(min(cl.get("horizons_min", [60])))
            if now_ts_ms < int(cl["ts_ms"]) + eval_h_min * 60 * 1000:
                still_pending.append(cl)
                continue

            coin = cl.get("target_coin", "btc")
            p_in_c   = float(cl["entry_prices"].get(coin, 0.0))
            p_in_btc = float(cl["entry_prices"].get("btc", 0.0))
            p_now_c  = float(current_prices.get(coin, 0.0))

            if p_in_c <= 0.0 or p_in_btc <= 0.0 or p_now_c <= 0.0:
                continue

            ret_c_bps   = math.log(p_now_c / p_in_c) * 10000.0
            ret_btc_bps = math.log(p_btc_now / p_in_btc) * 10000.0
            ortho_bps   = ret_c_bps if coin == "btc" else (ret_c_bps - 1.05 * ret_btc_bps)

            signed_gain_bps = float(cl["claimed_polarity"]) * ortho_bps
            is_win = signed_gain_bps >= float(cl.get("min_ortho_bps", 20.0))

            acc_key = cl["account_key"]
            st = self.credibility_state.get(acc_key, {
                "account_key": acc_key,
                "wins": 0,
                "losses": 0,
                "total_evaluated": 0,
                "hit_rate": 0.50,
                "dynamic_trust_weight": 1.25,
                "is_contrarian_retail_gauge": False,
            })

            if is_win:
                st["wins"] += 1
                st["dynamic_trust_weight"] = float(np.clip(
                    st["dynamic_trust_weight"] + float(cl.get("alpha_boost", 0.15)), 0.10, 4.50
                ))
            else:
                st["losses"] += 1
                st["dynamic_trust_weight"] = float(np.clip(
                    st["dynamic_trust_weight"] - float(cl.get("beta_penalty", 0.20)), 0.10, 4.50
                ))

            st["total_evaluated"] = st["wins"] + st["losses"]
            st["hit_rate"]        = round(float(st["wins"]) / float(max(1, st["total_evaluated"])), 4)
            if st["total_evaluated"] >= 8 and st["hit_rate"] <= float(cl.get("contrarian_flip_th", 0.38)):
                st["is_contrarian_retail_gauge"] = True

            st["updated_utc"] = str(pd.to_datetime(now_ts_ms, unit="ms", utc=True))
            self.credibility_state[acc_key] = st
            resolved_count += 1

        self.pending_claims = still_pending

        if resolved_count > 0:
            try:
                tmp_09 = f"{self.credibility_09_path}.tmp.{os.getpid()}"
                with open(tmp_09, "w", encoding="utf-8") as f9:
                    for st_row in self.credibility_state.values():
                        f9.write(json.dumps(st_row) + "\n")
                os.replace(tmp_09, self.credibility_09_path)
            except Exception:
                pass

        return resolved_count