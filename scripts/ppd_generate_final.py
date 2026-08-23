#!/usr/bin/env python3
"""Canonical four-stage PPD synthetic-narrative generator.

Author: Sara Sezavar

Pipeline
--------
1. De-identify the source post and extract generalized clinical factors.
2. Assign an EPDS-aligned narrative severity bucket from those factors.
3. Generate an original narrative conditioned on severity and (optionally) timing.
4. Blindly validate the narrative; regenerate on severity mismatch, never relabel.

The safe defaults are intended for an overnight laptop pilot: the first 100 rows,
one worker, and batches of two. Pass ``--limit-rows 0`` only for a deliberate full
run. Ollama must already be running and the selected model must already be pulled.

Only Stage 1 receives raw source text. Failed generation never falls back to raw
text. Every completed batch is written atomically and can be resumed safely.

Requires only Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SCRIPT_VERSION = "ppd_generate_final v2.7"
OLLAMA_HOST_DEFAULT = "http://localhost:11434"
FAILED_SENTINEL = "[GENERATION_FAILED]"

EPDS_LABELS = ("Minimal", "Mild", "Moderate", "Severe")
EPDS_SET = set(EPDS_LABELS)
TIMING_BUCKETS = (
    "very early (0-2 weeks)",
    "early (>2-6 weeks)",
    "intermediate (>6-12 weeks)",
    "later (>12 weeks)",
    "unknown",
)
TIMING_SET = set(TIMING_BUCKETS)
INTENSITIES = {"low", "moderate", "high"}

LOG_LOCK = threading.Lock()


def normalize_timing(value: Any) -> str | None:
    """Normalize harmless model shorthand without inferring unstated timing."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    normalized = normalized.replace("–", "-").replace("—", "-").replace("−", "-")
    normalized = re.sub(r"\s+", " ", normalized)
    aliases = {
        "very early": TIMING_BUCKETS[0],
        "0-2 weeks": TIMING_BUCKETS[0],
        "0 to 2 weeks": TIMING_BUCKETS[0],
        "early": TIMING_BUCKETS[1],
        ">2-6 weeks": TIMING_BUCKETS[1],
        "2-6 weeks": TIMING_BUCKETS[1],
        "2 to 6 weeks": TIMING_BUCKETS[1],
        "intermediate": TIMING_BUCKETS[2],
        ">6-12 weeks": TIMING_BUCKETS[2],
        "6-12 weeks": TIMING_BUCKETS[2],
        "6 to 12 weeks": TIMING_BUCKETS[2],
        "later": TIMING_BUCKETS[3],
        ">12 weeks": TIMING_BUCKETS[3],
        "12+ weeks": TIMING_BUCKETS[3],
        "more than 12 weeks": TIMING_BUCKETS[3],
        "unknown": TIMING_BUCKETS[4],
        "not stated": TIMING_BUCKETS[4],
        "not specified": TIMING_BUCKETS[4],
        "unspecified": TIMING_BUCKETS[4],
        "not mentioned": TIMING_BUCKETS[4],
    }
    canonical = {bucket.lower(): bucket for bucket in TIMING_BUCKETS}
    return canonical.get(normalized) or aliases.get(normalized)


_TIMING_NARRATIVE_CUES = {
    TIMING_BUCKETS[0]: re.compile(
        r"\b(?:within (?:the )?first (?:two|2) weeks|in (?:the )?first (?:two|2) weeks|"
        r"(?:one|two|\d+) days (?:after|since) (?:birth|delivery|giving birth)|"
        r"just (?:gave birth|delivered))\b",
        re.I,
    ),
    TIMING_BUCKETS[1]: re.compile(
        r"\b(?:(?:between )?(?:two|2) (?:and|to) (?:six|6) weeks|"
        r"(?:three|3|four|4|five|5|six|6) weeks (?:after|since)|a few weeks (?:after|since))\b",
        re.I,
    ),
    TIMING_BUCKETS[2]: re.compile(
        r"\b(?:(?:between )?(?:six|6) (?:and|to) (?:twelve|12) weeks|"
        r"(?:seven|7|eight|8|nine|9|ten|10|eleven|11|twelve|12) weeks (?:after|since)|"
        r"(?:about|around|nearly) (?:two|2|three|3) months (?:after|since))\b",
        re.I,
    ),
    TIMING_BUCKETS[3]: re.compile(
        r"\b(?:(?:more than|over) (?:three|3) months|beyond (?:twelve|12) weeks|"
        r"several months|(?:almost|about|around|over|more than)? ?"
        r"(?:four|4|five|5|six|6|seven|7|eight|8|nine|9|ten|10|eleven|11|twelve|12|1[3-9]|2[0-4]) "
        r"months)(?: (?:postpartum|after|since)(?: (?:birth|delivery|giving birth|having (?:a|my) baby))?)?\b",
        re.I,
    ),
}
_IMPLIED_TIMING_CUE = re.compile(
    r"\b(?:newborn|early postpartum|early days|first weeks|"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a few|several) "
    r"(?:days|weeks|months) (?:after|since) (?:birth|delivery|giving birth)|"
    r"just (?:gave birth|delivered))\b",
    re.I,
)
_CANONICAL_TIMING_PREFIXES = {
    TIMING_BUCKETS[0]: "I am within the first two weeks after birth.",
    TIMING_BUCKETS[1]: "I am between two and six weeks after birth.",
    TIMING_BUCKETS[2]: "I am between six and twelve weeks after birth.",
    TIMING_BUCKETS[3]: "I am more than three months after birth.",
}


def validate_narrative_timing(text: str, expected_timing: str) -> tuple[bool, str]:
    """Require an explicit bucket cue, or no cue when timing is unknown."""
    if expected_timing == "unknown":
        if _IMPLIED_TIMING_CUE.search(text or ""):
            return False, "narrative implies timing although expected timing is unknown"
        return True, ""
    pattern = _TIMING_NARRATIVE_CUES.get(expected_timing)
    conflicting = [
        bucket
        for bucket, cue_pattern in _TIMING_NARRATIVE_CUES.items()
        if bucket != expected_timing and cue_pattern.search(text or "")
    ]
    if conflicting:
        return False, "narrative contains a conflicting timing cue: " + ",".join(conflicting)
    if pattern is None or not pattern.search(text or ""):
        return False, f"narrative lacks an explicit cue for {expected_timing!r}"
    return True, ""


@dataclass
class LLMResponse:
    text: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    error: str = ""


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def add(self, result: LLMResponse) -> None:
        self.calls += 1
        self.prompt_tokens += result.prompt_tokens
        self.output_tokens += result.output_tokens
        self.seconds += result.seconds
        if result.error:
            self.errors.append(result.error)


@dataclass
class ItemResult:
    value: dict[str, Any] | None = None
    last_candidate: dict[str, Any] | None = None
    status: str = "failed"
    attempts: int = 0
    errors: list[str] = field(default_factory=list)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str, log_file: Path | None = None) -> None:
    line = f"[{timestamp()}] {message}"
    with LOG_LOCK:
        print(line, flush=True)
        if log_file is not None:
            try:
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with log_file.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError:
                pass


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, content)


def load_jsonl(path: Path) -> list[dict[str, Any]] | None:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not an object")
                rows.append(value)
        return rows
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def batch_path(outdir: Path, batch_id: int) -> Path:
    return outdir / "batches" / f"batch_{batch_id:06d}.jsonl"


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or "", flags=re.UNICODE))


def shared_ngram_count(source: str, candidate: str, n: int) -> int:
    source_tokens = normalized_tokens(source)
    candidate_tokens = normalized_tokens(candidate)
    if n < 1 or len(source_tokens) < n or len(candidate_tokens) < n:
        return 0
    source_ngrams = {
        tuple(source_tokens[index : index + n])
        for index in range(len(source_tokens) - n + 1)
    }
    return sum(
        tuple(candidate_tokens[index : index + n]) in source_ngrams
        for index in range(len(candidate_tokens) - n + 1)
    )


_IDENTIFIER_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "url": re.compile(r"https?://\S+|www\.\S+", re.I),
    "handle": re.compile(r"(?<!\w)(?:@[A-Za-z0-9_]{3,}|/?u/[A-Za-z0-9_-]{3,})"),
    "phone": re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    "exact_date": re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        re.I,
    ),
}


def identifier_leaks(text: str) -> list[str]:
    return [name for name, pattern in _IDENTIFIER_PATTERNS.items() if pattern.search(text or "")]


def text_values(value: Any) -> str:
    """Join only string values from nested model output (not schema key names)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(text_values(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(text_values(item) for item in value)
    return ""


_SUPPORTED_DETAIL_PATTERNS = {
    "relationship/support person": re.compile(
        r"\b(?:partner|husband|wife|spouse|boyfriend|girlfriend|friend|relative|family)\b", re.I
    ),
    "infant sex": re.compile(r"\b(?:daughter|son|baby girl|baby boy)\b", re.I),
    "feeding method": re.compile(
        r"\b(?:feed(?:ing|s)?|breastfeed\w*|breast ?milk|milk supply|nurs(?:e|ing)|"
        r"formula|bottle[- ]?feed\w*)\b",
        re.I,
    ),
    "high-risk/bonding detail": re.compile(
        r"\b(?:suicid\w*|self[- ]?harm|hopeless\w*|want to die|not worth living|"
        r"end my life|better off without me|harm(?:ing)? (?:my |the )?baby|"
        r"unable to (?:care for|bond with|connect with) (?:my |the )?(?:baby|infant|child)|"
        r"cannot (?:care for|bond with|connect with)|can't (?:care for|bond with|connect with)|"
        r"bonding feels|feel(?:ing)? detached|feel(?:ing)? disconnected)\b",
        re.I,
    ),
    "treatment/medical event": re.compile(
        r"\b(?:therap\w*|counsel\w*|medicat\w*|antidepressant\w*|doctor|hospital|diagnos\w*)\b",
        re.I,
    ),
}


def unsupported_detail_categories(text: str, factors: dict[str, Any]) -> list[str]:
    """Flag sensitive concrete details in a draft unless Stage-1 factors support them."""
    support_text = text_values(factors)
    return [
        category
        for category, pattern in _SUPPORTED_DETAIL_PATTERNS.items()
        if pattern.search(text or "") and not pattern.search(support_text)
    ]


def strip_code_fence(raw: str) -> str:
    value = (raw or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def extract_json(raw: str, expected_type: type) -> Any | None:
    """Extract the first complete JSON value of the requested type."""
    stripped = strip_code_fence(raw)
    for candidate in (raw.strip() if raw else "", stripped):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, expected_type):
                return value
        except (ValueError, json.JSONDecodeError):
            pass

    decoder = json.JSONDecoder()
    opening = "[" if expected_type is list else "{"
    for index, character in enumerate(stripped):
        if character != opening:
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            if isinstance(value, expected_type):
                return value
        except (ValueError, json.JSONDecodeError):
            continue
    return None


def call_ollama(
    *,
    cfg: dict[str, Any],
    prompt: str,
    temperature: float,
    max_tokens: int,
    seed: int,
) -> LLMResponse:
    url = f"{cfg['ollama_host'].rstrip('/')}/api/generate"
    payload = {
        "model": cfg["model"],
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "top_p": cfg["top_p"],
            "top_k": cfg["top_k"],
            "repeat_penalty": cfg["repeat_penalty"],
            "num_predict": max_tokens,
            "num_ctx": cfg["num_ctx"],
            "seed": seed,
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error = ""
    started = time.monotonic()
    for attempt in range(1, cfg["transport_retries"] + 1):
        try:
            request = Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=cfg["request_timeout"]) as response:
                data = json.loads(response.read().decode("utf-8"))
            output = str(data.get("response") or "").strip()
            if not output:
                raise ValueError("Ollama returned an empty response")
            return LLMResponse(
                text=output,
                prompt_tokens=int(data.get("prompt_eval_count") or 0),
                output_tokens=int(data.get("eval_count") or 0),
                seconds=(float(data.get("total_duration") or 0) / 1e9)
                or (time.monotonic() - started),
            )
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < cfg["transport_retries"]:
                time.sleep(min(30.0, 2.0 * attempt))
    return LLMResponse(seconds=time.monotonic() - started, error=last_error)


def model_digest(host: str, model: str, timeout: int = 30) -> str | None:
    try:
        request = Request(
            f"{host.rstrip('/')}/api/show",
            data=json.dumps({"name": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("digest") or data.get("details", {}).get("digest")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return None


DEID_RULES = """De-identification rules:
- Remove or generalize names, usernames, handles, contact information, exact dates,
  locations, institutions, employers, and rare or uniquely identifying events.
- Do not quote or closely paraphrase the source. Use generalized clinical language.
- Do not invent facts that are not supported by the source.
"""

TIMING_DEFS = """Postpartum timing buckets (use one exact value):
- "very early (0-2 weeks)"
- "early (>2-6 weeks)"
- "intermediate (>6-12 weeks)"
- "later (>12 weeks)"
- "unknown"
Use "unknown" when timing is not stated or clearly implied. Never guess timing.
"""

SEVERITY_DEFS = """EPDS-aligned narrative severity buckets (use one exact value):
- "Minimal": transient adjustment stress, mild fatigue, or situational worry with
  little or no functional impairment. Never use Minimal for high-intensity or
  persistent/ongoing symptoms.
- "Mild": noticeable mood disturbance, guilt, anxiety, or intermittent distress
  with limited impairment and preserved basic functioning.
- "Moderate": persistent low mood, rumination, meaningful functional impairment,
  or bonding-difficulty indicators, without pervasive major impairment.
- "Severe": pervasive hopelessness, sustained major impairment, intense anxiety,
  significant bonding disruption, or urgent distress cues. A word such as "severe"
  or "depression" alone is insufficient without high intensity plus major impact,
  negative bonding indicators, hopelessness, inability to care, or urgent risk.
Timing is context, not a rule. These are narrative buckets, not reconstructed EPDS scores.
"""

SEVERITY_BOUNDARY_RULES = """Boundary rules:
- Minimal: symptoms must be low/transient/resolved and functioning essentially intact.
- Mild: noticeable symptoms but no more than limited functional impact.
- Moderate: persistent or high-intensity symptoms with meaningful, non-major impact.
- Severe: pervasive high-intensity distress plus major impairment, serious bonding
  disruption, inability to manage basic care, hopelessness, or urgent safety cues.
- Use the highest bucket directly supported by the factors, but do not infer missing
  impairment or risk and do not classify from a severity word alone.
"""

JSON_ONLY = "Return only valid JSON. Do not use markdown fences or add commentary."


def prompt_stage1(items: Sequence[dict[str, Any]], use_timing: bool) -> str:
    timing = TIMING_DEFS + "\n" if use_timing else ""
    timing_field = '"postpartum_timing": "one timing bucket", ' if use_timing else ""
    return f"""You are performing privacy-preserving abstraction of postpartum
mental-health text.

{DEID_RULES}
{timing}
For every input item, return one object with the same integer "id" and these fields:
{{"id": 0, "deidentified_summary": "1-2 generalized sentences", {timing_field}
 "symptoms": ["..."], "symptom_intensity": "low|moderate|high",
 "symptom_persistence": "...", "functional_impact": ["..."],
 "sleep_context": "...", "feeding_or_infant_care_stressors": ["..."],
 "perceived_support": "...", "bonding_indicators": ["..."]}}

Return a JSON list with exactly one output per input ID. {JSON_ONLY}
INPUT:
{json.dumps(items, ensure_ascii=False)}"""


def prompt_stage2(items: Sequence[dict[str, Any]], use_timing: bool) -> str:
    timing_note = (
        "The factors include postpartum_timing. Treat it only as context.\n" if use_timing else ""
    )
    return f"""Assign an EPDS-aligned narrative severity bucket using only the
generalized factors below; you will not see raw source text.

{SEVERITY_DEFS}
{SEVERITY_BOUNDARY_RULES}
{timing_note}
For each input, return the same integer "id", one exact severity label, and a
one-sentence rationale grounded in the factors:
{{"id": 0, "severity": "Minimal|Mild|Moderate|Severe", "rationale": "..."}}

Each input may contain "disallowed_severity_labels", computed from explicit
factor contradictions. Never return a label listed there; choose the best-supported
remaining label and explain the boundary decision.

Return a JSON list with exactly one output per input ID. {JSON_ONLY}
INPUT:
{json.dumps(items, ensure_ascii=False)}"""


def prompt_stage3(
    items: Sequence[dict[str, Any]],
    use_timing: bool,
    min_words: int,
    max_words: int,
    regeneration_attempt: int = 0,
    correction_attempt: int = 0,
) -> str:
    target_low = min(max_words, max(min_words, 200))
    target_high = min(max_words, max(target_low, 230))
    timing = TIMING_DEFS + "\n" if use_timing else ""
    timing_instruction = (
        'Copy "postpartum_timing" exactly into "timing" and make the narrative '
        "consistent with that context. Use exactly one compatible timing cue: "
        '"within the first two weeks after birth" for very early; '
        '"between two and six weeks after birth" for early; '
        '"between six and twelve weeks after birth" for intermediate; or '
        '"more than three months after birth" for later. For unknown timing, do not '
        "mention or imply days, weeks, months, newborn status, or early/later timing.\n"
        if use_timing
        else ""
    )
    timing_field = ', "timing": "exact input postpartum_timing"' if use_timing else ""
    regeneration = ""
    if regeneration_attempt:
        regeneration = (
            f"This is regeneration attempt {regeneration_attempt}. Produce a materially "
            "different composition while preserving only the supplied factors and target. "
            "Each input may contain validator_feedback. Correct the stated severity/timing "
            "mismatch directly: reduce or strengthen severity cues toward intended_severity "
            "and use the required timing rule for intended_timing.\n"
        )
    correction = ""
    if correction_attempt:
        correction = (
            "A prior response failed strict validation, commonly because it was too short. "
            f"This time write at least {target_low} words and aim for {target_low}-{target_high} "
            "words. Use 14-18 substantive "
            "sentences, and verify the word count before returning the JSON. Do not stop early.\n"
        )
    return f"""Generate original first-person postpartum diary narratives for
research. Stage 3 receives generalized factors only, never source posts.

{SEVERITY_DEFS}
{SEVERITY_BOUNDARY_RULES}
{timing}
{timing_instruction}{regeneration}{correction}
Constraints:
- Each synthetic_text must contain {min_words}-{max_words} words.
- Treat the minimum word count as a hard requirement; aim for {target_low}-{target_high} words.
- Match the supplied severity in expressed intensity, persistence, and impairment.
- Minimal must show transient/low distress and intact functioning; Mild must show
  noticeable but limited impairment; Moderate must show persistent meaningful
  difficulty; Severe must clearly show pervasive distress plus major impairment,
  inability to manage basic care, serious bonding disruption, or urgent risk.
- Do not copy or closely paraphrase an input field.
- Use only the supplied factors. Do not invent a partner, infant sex, occupation,
  feeding method, medical event, treatment, or support person unless supplied.
- Keep the narrative introspective. Do not invent phone calls, visits, appointments,
  conversations, or other concrete events merely to add length.
- Do not add names, handles, contact details, exact dates, locations, or institutions.
- Copy the supplied severity exactly into "target".

For each input, return:
{{"id": 0, "synthetic_text": "...", "target": "exact input severity"{timing_field},
 "style": "first-person postpartum diary"}}

Return a JSON list with exactly one output per input ID. {JSON_ONLY}
INPUT:
{json.dumps(items, ensure_ascii=False)}"""


def prompt_stage3_length_repair(
    item: dict[str, Any], min_words: int, max_words: int, use_timing: bool
) -> str:
    target_low = min(max_words, max(min_words, 235))
    target_high = min(max_words, max(target_low, 245))
    item_id = int(item["id"])
    target = str(item["target"])
    expected_timing = str(item.get("expected_timing", "unknown"))
    timing_field = f', "timing": {json.dumps(expected_timing)}' if use_timing else ""
    timing_phrases = {
        TIMING_BUCKETS[0]: 'include "within the first two weeks after birth"',
        TIMING_BUCKETS[1]: 'include "between two and six weeks after birth"',
        TIMING_BUCKETS[2]: 'include "between six and twelve weeks after birth"',
        TIMING_BUCKETS[3]: 'include "more than three months after birth"',
        TIMING_BUCKETS[4]: (
            "do not mention or imply days, weeks, months, newborn status, or early/later timing"
        ),
    }
    timing_requirement = timing_phrases.get(expected_timing, "") if use_timing else ""
    severity_requirements = {
        "Minimal": "show transient or low distress with intact everyday functioning",
        "Mild": "show noticeable distress but only limited impairment and preserved basic functioning",
        "Moderate": "show persistent distress with meaningful but non-major impairment",
        "Severe": (
            "show pervasive high-intensity distress plus major impairment, inability to manage "
            "basic care, serious bonding disruption, or urgent risk"
        ),
    }
    severity_requirement = severity_requirements.get(target, "match the supplied target")
    return f"""Rewrite and expand one under-length synthetic postpartum diary draft.
The draft and factors are synthetic/generalized; no raw source post is provided.

Requirements:
- Return {target_low}-{target_high} words in synthetic_text; count before responding.
- Preserve the supplied target and expected timing exactly.
- For severity: {severity_requirement}.
- For timing: {timing_requirement}.
- Preserve supported meaning, but rewrite and expand the composition.
- Use only supplied factors. Do not invent a partner, infant sex, occupation,
  feeding method, medical event, treatment, or support person unless supplied.
- Keep the narrative introspective; do not invent conversations or concrete events.
- Do not add names, handles, contact details, exact dates, locations, or institutions.

Return one JSON object:
{{"id": {item_id}, "synthetic_text": "...", "target": {json.dumps(target)}{timing_field},
 "style": "first-person postpartum diary"}}

{JSON_ONLY}
INPUT:
{json.dumps(item, ensure_ascii=False)}"""


def prompt_stage3_content_repair(
    item: dict[str, Any], categories: Sequence[str], min_words: int, max_words: int,
    use_timing: bool,
) -> str:
    target = str(item["target"])
    expected_timing = str(item.get("expected_timing", "unknown"))
    item_id = int(item["id"])
    timing_field = f', "timing": {json.dumps(expected_timing)}' if use_timing else ""
    category_rules = {
        "relationship/support person": (
            "Remove unsupported partner, spouse, friend, family, or relative details. "
            "Use only the generalized support wording present in factors."
        ),
        "infant sex": "Use only neutral terms such as baby, infant, or child; do not state infant sex.",
        "feeding method": "Remove feeding-method details unless the factors explicitly contain them.",
        "treatment/medical event": (
            "Remove medication, therapy, clinician, diagnosis, or medical-event details unless explicit in factors."
        ),
        "high-risk/bonding detail": (
            "Remove unsupported suicidality, self-harm, inability-to-care, or bonding-disruption "
            "claims. Do not soften or replace them with other unsupported clinical facts."
        ),
    }
    rules = "\n".join(f"- {category_rules[category]}" for category in categories)
    return f"""Rewrite one synthetic postpartum diary draft to remove unsupported details.
The supplied factors are the complete factual boundary. Do not add replacement facts.

Required corrections:
{rules}

Additional requirements:
- Keep synthetic_text within {min_words}-{max_words} words; aim for 180-220 words.
- Preserve target {json.dumps(target)} and timing {json.dumps(expected_timing)} exactly.
- Preserve supported symptom intensity, persistence, impairment, and timing.
- Keep the narrative first-person and natural, but prefer introspection over invented events.
- Do not add identifiers, exact dates, locations, institutions, or source-like wording.

Return one JSON object:
{{"id": {item_id}, "synthetic_text": "...", "target": {json.dumps(target)}{timing_field},
 "style": "first-person postpartum diary"}}

{JSON_ONLY}
INPUT:
{json.dumps(item, ensure_ascii=False)}"""


def prompt_stage4(narrative: str, use_timing: bool) -> str:
    timing = TIMING_DEFS + "\n" if use_timing else ""
    timing_field = ', "predicted_timing": "one timing bucket"' if use_timing else ""
    timing_decision = (
        "For timing, prioritize explicit relative phrases: first two weeks = very early; "
        "two-to-six weeks = early; six-to-twelve weeks = intermediate; more than three "
        "months = later. Return unknown when none is stated; do not assume early merely "
        "because the writer has an infant.\n"
        if use_timing
        else ""
    )
    return f"""Act as an independent blind rater. Assess the diary entry on its
own. You are intentionally not shown the intended severity or timing.

{SEVERITY_DEFS}
{SEVERITY_BOUNDARY_RULES}
{timing}
Choose Severe rather than Moderate when pervasive distress is paired with major
impairment, inability to manage basic care, serious bonding disruption, or urgent
risk. Choose Minimal only for transient/low distress with intact functioning.
{timing_decision}
Return one object:
{{"predicted_severity": "Minimal|Mild|Moderate|Severe"{timing_field},
 "confidence": 0.0, "evidence": "one brief sentence"}}

{JSON_ONLY}
DIARY ENTRY:
{json.dumps(narrative, ensure_ascii=False)}"""


def validate_stage1(
    item: Any,
    *,
    expected_id: int,
    source_text: str,
    use_timing: bool,
    copy_ngram_size: int,
    max_copy_ngrams: int,
) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not an object"
    if item.get("id") != expected_id:
        return False, f"id mismatch: {item.get('id')!r}"
    summary = item.get("deidentified_summary")
    if not isinstance(summary, str) or not summary.strip():
        return False, "missing deidentified_summary"
    for key in ("symptoms", "functional_impact", "feeding_or_infant_care_stressors", "bonding_indicators"):
        if not isinstance(item.get(key), list):
            return False, f"{key} must be a list"
    for key in ("symptom_persistence", "sleep_context", "perceived_support"):
        if not isinstance(item.get(key), str):
            return False, f"{key} must be a string"
    if item.get("symptom_intensity") not in INTENSITIES:
        return False, f"invalid symptom_intensity: {item.get('symptom_intensity')!r}"
    if use_timing:
        normalized_timing = normalize_timing(item.get("postpartum_timing"))
        if normalized_timing is None:
            return False, f"invalid postpartum_timing: {item.get('postpartum_timing')!r}"
        item["postpartum_timing"] = normalized_timing
    privacy_text = text_values({key: value for key, value in item.items() if key != "id"})
    leaks = identifier_leaks(privacy_text)
    if leaks:
        return False, "identifier leak in structured factors: " + ",".join(leaks)
    overlap = shared_ngram_count(source_text, privacy_text, copy_ngram_size)
    if overlap > max_copy_ngrams:
        return False, f"source-copy overlap: {overlap} shared {copy_ngram_size}-grams"
    return True, ""


def validate_stage2(
    item: Any, *, expected_id: int, factors: dict[str, Any]
) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not an object"
    if item.get("id") != expected_id:
        return False, f"id mismatch: {item.get('id')!r}"
    if item.get("severity") not in EPDS_SET:
        return False, f"invalid severity: {item.get('severity')!r}"
    if not isinstance(item.get("rationale"), str) or not item["rationale"].strip():
        return False, "missing rationale"
    severity = str(item["severity"])
    intensity = str(factors.get("symptom_intensity", "")).strip().lower()
    persistence = str(factors.get("symptom_persistence", "")).strip().lower()
    impacts = [
        str(value).strip()
        for value in factors.get("functional_impact", [])
        if str(value).strip() and str(value).strip().lower() not in {"none", "unknown", "not specified"}
    ]
    factor_text = text_values(factors).lower()
    persistent = bool(
        re.search(r"\b(?:persistent|ongoing|constant|continuous|sustained|daily|frequent)\b", persistence)
    )
    negative_bonding = bool(
        re.search(
            r"\b(?:bonding difficult|difficulty bonding|detached|disconnected|avoid\w* infant|"
            r"unable to (?:bond|connect|care)|no bond)\b",
            factor_text,
        )
    )
    urgent_or_hopeless = bool(
        re.search(
            r"\b(?:suicid\w*|self[- ]?harm|hopeless\w*|unable to care|cannot care|"
            r"can't care|urgent|emergency|psychosis|hallucinat\w*)\b",
            factor_text,
        )
    )
    if severity == "Minimal" and (intensity == "high" or persistent or impacts):
        return False, "Minimal conflicts with high/persistent symptoms or functional impact"
    if severity == "Severe":
        severe_support = intensity == "high" and bool(impacts or negative_bonding or urgent_or_hopeless)
        if not severe_support:
            return False, "Severe lacks high intensity plus major impact/risk/bonding support"
    return True, ""


def disallowed_severity_labels(factors: dict[str, Any]) -> list[str]:
    """Return boundary labels contradicted by the structured factors."""
    intensity = str(factors.get("symptom_intensity", "")).strip().lower()
    persistence = str(factors.get("symptom_persistence", "")).strip().lower()
    impacts = [
        str(value).strip()
        for value in factors.get("functional_impact", [])
        if str(value).strip() and str(value).strip().lower() not in {"none", "unknown", "not specified"}
    ]
    factor_text = text_values(factors).lower()
    persistent = bool(
        re.search(r"\b(?:persistent|ongoing|constant|continuous|sustained|daily|frequent)\b", persistence)
    )
    negative_bonding = bool(
        re.search(
            r"\b(?:bonding difficult|difficulty bonding|detached|disconnected|avoid\w* infant|"
            r"unable to (?:bond|connect|care)|no bond)\b",
            factor_text,
        )
    )
    urgent_or_hopeless = bool(
        re.search(
            r"\b(?:suicid\w*|self[- ]?harm|hopeless\w*|unable to care|cannot care|"
            r"can't care|urgent|emergency|psychosis|hallucinat\w*)\b",
            factor_text,
        )
    )
    disallowed = []
    if intensity == "high" or persistent or impacts:
        disallowed.append("Minimal")
    if not (intensity == "high" and bool(impacts or negative_bonding or urgent_or_hopeless)):
        disallowed.append("Severe")
    return disallowed


def validate_stage3(
    item: Any,
    *,
    expected_id: int,
    expected_severity: str,
    expected_timing: str,
    source_text: str,
    use_timing: bool,
    min_words: int,
    max_words: int,
    copy_ngram_size: int,
    max_copy_ngrams: int,
) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not an object"
    if item.get("id") != expected_id:
        return False, f"id mismatch: {item.get('id')!r}"
    text = item.get("synthetic_text")
    if not isinstance(text, str) or not text.strip():
        return False, "missing synthetic_text"
    if item.get("target") != expected_severity:
        return False, f"target mismatch: {item.get('target')!r}"
    if use_timing:
        normalized_timing = normalize_timing(item.get("timing"))
        if normalized_timing != expected_timing:
            return False, f"timing mismatch: {item.get('timing')!r}"
        item["timing"] = normalized_timing
        timing_ok, timing_error = validate_narrative_timing(text, expected_timing)
        if (
            not timing_ok
            and expected_timing != "unknown"
            and timing_error.startswith("narrative lacks an explicit cue")
        ):
            prefix = _CANONICAL_TIMING_PREFIXES[expected_timing]
            text = f"{prefix} {text.strip()}"
            item["synthetic_text"] = text
            item["timing_cue_injected"] = True
            timing_ok, timing_error = validate_narrative_timing(text, expected_timing)
        if not timing_ok:
            return False, timing_error
    count = word_count(text)
    if not min_words <= count <= max_words:
        return False, f"word count {count} outside [{min_words}, {max_words}]"
    leaks = identifier_leaks(text)
    if leaks:
        return False, "identifier leak: " + ",".join(leaks)
    overlap = shared_ngram_count(source_text, text, copy_ngram_size)
    if overlap > max_copy_ngrams:
        return False, f"source-copy overlap: {overlap} shared {copy_ngram_size}-grams"
    return True, ""


def validate_stage4(item: Any, *, use_timing: bool) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not an object"
    if item.get("predicted_severity") not in EPDS_SET:
        return False, f"invalid predicted_severity: {item.get('predicted_severity')!r}"
    confidence = item.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False, f"invalid confidence: {confidence!r}"
    if not 0.0 <= float(confidence) <= 1.0:
        return False, f"confidence outside [0,1]: {confidence!r}"
    if use_timing:
        normalized_timing = normalize_timing(item.get("predicted_timing"))
        if normalized_timing is None:
            return False, f"invalid predicted_timing: {item.get('predicted_timing')!r}"
        item["predicted_timing"] = normalized_timing
    if not isinstance(item.get("evidence"), str) or not item["evidence"].strip():
        return False, "missing evidence"
    return True, ""


def parse_group_by_id(raw: str, expected_ids: set[int]) -> tuple[dict[int, dict[str, Any]], str]:
    root: Any = None
    stripped = strip_code_fence(raw)
    try:
        root = json.loads(stripped)
    except (ValueError, json.JSONDecodeError):
        pass

    parsed = root if isinstance(root, list) else None
    if parsed is None:
        wrapper = root if isinstance(root, dict) else extract_json(raw, dict)
        if wrapper is not None:
            for key in ("items", "results", "outputs", "responses", "data"):
                if isinstance(wrapper.get(key), list):
                    parsed = wrapper[key]
                    break
            if parsed is None and len(expected_ids) == 1:
                # JSON mode may return the sole requested item as the top-level object.
                parsed = [wrapper]
        if parsed is None:
            return {}, "response did not contain a JSON list or recognized list wrapper"
    mapped: dict[int, dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id.strip().isdigit():
            item_id = int(item_id.strip())
            item["id"] = item_id
        if isinstance(item_id, bool) or not isinstance(item_id, int):
            continue
        if item_id in expected_ids and item_id not in mapped:
            mapped[item_id] = item
    if not mapped and len(expected_ids) == 1 and len(parsed) == 1 and isinstance(parsed[0], dict):
        # A single-item retry has only one possible ID, so recovery is unambiguous.
        item_id = next(iter(expected_ids))
        parsed[0]["id"] = item_id
        mapped[item_id] = parsed[0]
    if not mapped:
        return {}, "no expected IDs were returned"
    return mapped, ""


def run_group_stage(
    *,
    ids: Sequence[int],
    make_prompt: Callable[[Sequence[int], int], str],
    validator: Callable[[dict[str, Any], int], tuple[bool, str]],
    cfg: dict[str, Any],
    temperature: float,
    tokens_per_item: int,
    stage_number: int,
    usage: Usage,
) -> dict[int, ItemResult]:
    results = {item_id: ItemResult() for item_id in ids}

    def attempt_group(group_ids: Sequence[int], attempt_number: int) -> None:
        seed = cfg["seed"] + stage_number * 1_000_003 + min(group_ids) * 101 + attempt_number
        response = call_ollama(
            cfg=cfg,
            prompt=make_prompt(group_ids, attempt_number),
            temperature=temperature,
            max_tokens=tokens_per_item * len(group_ids),
            seed=seed,
        )
        usage.add(response)
        for item_id in group_ids:
            results[item_id].attempts += 1
        if not response.text:
            error = response.error or "empty LLM response"
            for item_id in group_ids:
                results[item_id].errors.append(error)
            return
        mapped, parse_error = parse_group_by_id(response.text, set(group_ids))
        if parse_error:
            for item_id in group_ids:
                results[item_id].errors.append(parse_error)
        for item_id in group_ids:
            item = mapped.get(item_id)
            if item is None:
                results[item_id].errors.append("output missing expected ID")
                continue
            ok, error = validator(item, item_id)
            if ok:
                results[item_id].value = item
                results[item_id].status = "ok"
            else:
                results[item_id].last_candidate = item
                results[item_id].errors.append(error)

    for attempt_number in range(1, cfg["schema_retries"] + 1):
        pending = [item_id for item_id in ids if results[item_id].status != "ok"]
        if not pending:
            break
        attempt_group(pending, attempt_number)

    pending = [item_id for item_id in ids if results[item_id].status != "ok"]
    for item_id in pending:
        for retry in range(1, cfg["schema_retries"] + 1):
            attempt_group([item_id], cfg["schema_retries"] + retry)
            if results[item_id].status == "ok":
                break
    return results


def stage1(
    records: dict[int, dict[str, Any]], cfg: dict[str, Any], usage: Usage
) -> dict[int, ItemResult]:
    ids = list(records)

    def make_prompt(group_ids: Sequence[int], _: int) -> str:
        return prompt_stage1(
            [{"id": item_id, "post": records[item_id]["source_text"]} for item_id in group_ids],
            cfg["use_timing"],
        )

    def validator(item: dict[str, Any], item_id: int) -> tuple[bool, str]:
        return validate_stage1(
            item,
            expected_id=item_id,
            source_text=records[item_id]["source_text"],
            use_timing=cfg["use_timing"],
            copy_ngram_size=cfg["copy_ngram_size"],
            max_copy_ngrams=cfg["max_copy_ngrams"],
        )

    return run_group_stage(
        ids=ids,
        make_prompt=make_prompt,
        validator=validator,
        cfg=cfg,
        temperature=cfg["temp_s1"],
        tokens_per_item=cfg["tokens_s1_per_item"],
        stage_number=1,
        usage=usage,
    )

def factor_payload(item_id: int, factors: dict[str, Any], use_timing: bool) -> dict[str, Any]:
    keys = (
        "deidentified_summary",
        "symptoms",
        "symptom_intensity",
        "symptom_persistence",
        "functional_impact",
        "sleep_context",
        "feeding_or_infant_care_stressors",
        "perceived_support",
        "bonding_indicators",
    )
    payload = {"id": item_id, **{key: factors.get(key) for key in keys}}
    if use_timing:
        payload["postpartum_timing"] = factors.get("postpartum_timing", "unknown")
    return payload


def stage2(
    stage1_results: dict[int, ItemResult], cfg: dict[str, Any], usage: Usage
) -> dict[int, ItemResult]:
    eligible = {
        item_id: result.value
        for item_id, result in stage1_results.items()
        if result.status == "ok" and result.value is not None
    }
    output = {item_id: ItemResult(errors=["Stage 1 failed"]) for item_id in stage1_results}
    if not eligible:
        return output

    def make_prompt(group_ids: Sequence[int], _: int) -> str:
        payloads = []
        for item_id in group_ids:
            payload = factor_payload(item_id, eligible[item_id], cfg["use_timing"])
            payload["disallowed_severity_labels"] = disallowed_severity_labels(
                eligible[item_id]
            )
            payloads.append(payload)
        return prompt_stage2(
            payloads,
            cfg["use_timing"],
        )

    generated = run_group_stage(
        ids=list(eligible),
        make_prompt=make_prompt,
        validator=lambda item, item_id: validate_stage2(
            item, expected_id=item_id, factors=eligible[item_id]
        ),
        cfg=cfg,
        temperature=cfg["temp_s2"],
        tokens_per_item=cfg["tokens_s2_per_item"],
        stage_number=2,
        usage=usage,
    )
    output.update(generated)
    return output


def generation_payload(
    item_id: int,
    factors: dict[str, Any],
    severity: dict[str, Any],
    use_timing: bool,
) -> dict[str, Any]:
    payload = factor_payload(item_id, factors, use_timing)
    payload["severity"] = severity["severity"]
    return payload


def generate_stage3(
    *,
    ids: Sequence[int],
    records: dict[int, dict[str, Any]],
    stage1_results: dict[int, ItemResult],
    stage2_results: dict[int, ItemResult],
    cfg: dict[str, Any],
    usage: Usage,
    regeneration_attempt: int = 0,
    validator_feedback: dict[int, dict[str, Any]] | None = None,
) -> dict[int, ItemResult]:
    def make_prompt(group_ids: Sequence[int], schema_attempt: int) -> str:
        payloads = []
        for item_id in group_ids:
            payload = generation_payload(
                item_id,
                stage1_results[item_id].value or {},
                stage2_results[item_id].value or {},
                cfg["use_timing"],
            )
            if validator_feedback and item_id in validator_feedback:
                payload["validator_feedback"] = validator_feedback[item_id]
            payloads.append(payload)
        return prompt_stage3(
            payloads,
            cfg["use_timing"],
            cfg["min_words"],
            cfg["max_words"],
            regeneration_attempt=regeneration_attempt,
            correction_attempt=max(0, schema_attempt - 1),
        )

    def validator(item: dict[str, Any], item_id: int) -> tuple[bool, str]:
        factors = stage1_results[item_id].value or {}
        severity = stage2_results[item_id].value or {}
        return validate_stage3(
            item,
            expected_id=item_id,
            expected_severity=str(severity.get("severity", "")),
            expected_timing=str(factors.get("postpartum_timing", "unknown")),
            source_text=records[item_id]["source_text"],
            use_timing=cfg["use_timing"],
            min_words=cfg["min_words"],
            max_words=cfg["max_words"],
            copy_ngram_size=cfg["copy_ngram_size"],
            max_copy_ngrams=cfg["max_copy_ngrams"],
        )

    generated = run_group_stage(
        ids=ids,
        make_prompt=make_prompt,
        validator=validator,
        cfg=cfg,
        temperature=cfg["temp_s3"],
        tokens_per_item=cfg["tokens_s3_per_item"],
        stage_number=3 + regeneration_attempt * 10,
        usage=usage,
    )

    repair_ids = []
    for item_id, result in generated.items():
        candidate = result.last_candidate or {}
        candidate_text = candidate.get("synthetic_text")
        if (
            result.status != "ok"
            and isinstance(candidate_text, str)
            and word_count(candidate_text) < cfg["min_words"]
        ):
            repair_ids.append(item_id)

    def repair_prompt(group_ids: Sequence[int], _: int) -> str:
        item_id = group_ids[0]
        factors = stage1_results[item_id].value or {}
        severity = stage2_results[item_id].value or {}
        payload = {
            "id": item_id,
            "factors": generation_payload(item_id, factors, severity, cfg["use_timing"]),
            "target": severity.get("severity"),
            "expected_timing": factors.get("postpartum_timing", "unknown"),
            "under_length_draft": (generated[item_id].last_candidate or {}).get(
                "synthetic_text", ""
            ),
        }
        return prompt_stage3_length_repair(
            payload, cfg["min_words"], cfg["max_words"], cfg["use_timing"]
        )

    repair_cfg = {**cfg, "schema_retries": 1}
    for item_id in repair_ids:
        repaired = run_group_stage(
            ids=[item_id],
            make_prompt=repair_prompt,
            validator=validator,
            cfg=repair_cfg,
            temperature=min(cfg["temp_s3"], 0.8),
            tokens_per_item=cfg["tokens_s3_per_item"],
            stage_number=30 + regeneration_attempt * 10,
            usage=usage,
        )[item_id]
        repaired.attempts += generated[item_id].attempts
        repaired.errors = generated[item_id].errors + repaired.errors
        if repaired.status == "ok":
            generated[item_id] = repaired
        else:
            generated[item_id].attempts = repaired.attempts
            generated[item_id].errors = repaired.errors
            generated[item_id].last_candidate = repaired.last_candidate

    content_repair_ids = []
    content_categories: dict[int, list[str]] = {}
    for item_id, result in generated.items():
        if result.status != "ok" or result.value is None:
            continue
        factors = stage1_results[item_id].value or {}
        categories = unsupported_detail_categories(
            str(result.value.get("synthetic_text", "")), factors
        )
        if categories:
            content_repair_ids.append(item_id)
            content_categories[item_id] = categories

    def content_prompt(group_ids: Sequence[int], _: int) -> str:
        item_id = group_ids[0]
        factors = stage1_results[item_id].value or {}
        severity = stage2_results[item_id].value or {}
        payload = {
            "id": item_id,
            "factors": generation_payload(item_id, factors, severity, cfg["use_timing"]),
            "target": severity.get("severity"),
            "expected_timing": factors.get("postpartum_timing", "unknown"),
            "draft_to_rewrite": (generated[item_id].value or {}).get("synthetic_text", ""),
        }
        return prompt_stage3_content_repair(
            payload,
            content_categories[item_id],
            cfg["min_words"],
            cfg["max_words"],
            cfg["use_timing"],
        )

    def content_validator(item: dict[str, Any], item_id: int) -> tuple[bool, str]:
        ok, error = validator(item, item_id)
        if not ok:
            return ok, error
        categories = unsupported_detail_categories(
            str(item.get("synthetic_text", "")), stage1_results[item_id].value or {}
        )
        if categories:
            return False, "unsupported detail after targeted repair: " + ",".join(categories)
        return True, ""

    content_cfg = {**cfg, "schema_retries": 1}
    for item_id in content_repair_ids:
        repaired = run_group_stage(
            ids=[item_id],
            make_prompt=content_prompt,
            validator=content_validator,
            cfg=content_cfg,
            temperature=min(cfg["temp_s3"], 0.7),
            tokens_per_item=cfg["tokens_s3_per_item"],
            stage_number=40 + regeneration_attempt * 10,
            usage=usage,
        )[item_id]
        repaired.attempts += generated[item_id].attempts
        repaired.errors = generated[item_id].errors + repaired.errors
        generated[item_id] = repaired

    return generated


def stage4_single(
    item_id: int,
    narrative: str,
    cfg: dict[str, Any],
    usage: Usage,
    validation_round: int,
) -> ItemResult:
    result = ItemResult()
    for attempt in range(1, cfg["schema_retries"] + 1):
        seed = cfg["seed"] + 4_000_003 + item_id * 101 + validation_round * 17 + attempt
        response = call_ollama(
            cfg=cfg,
            prompt=prompt_stage4(narrative, cfg["use_timing"]),
            temperature=cfg["temp_s4"],
            max_tokens=cfg["tokens_s4_per_item"],
            seed=seed,
        )
        usage.add(response)
        result.attempts += 1
        if not response.text:
            result.errors.append(response.error or "empty validator response")
            continue
        parsed = extract_json(response.text, dict)
        if parsed is None:
            result.errors.append("validator response did not contain a JSON object")
            continue
        ok, error = validate_stage4(parsed, use_timing=cfg["use_timing"])
        if ok:
            result.value = parsed
            result.status = "ok"
            return result
        result.errors.append(error)
    return result


def stages3_and4(
    *,
    records: dict[int, dict[str, Any]],
    stage1_results: dict[int, ItemResult],
    stage2_results: dict[int, ItemResult],
    cfg: dict[str, Any],
    usage: Usage,
) -> tuple[dict[int, ItemResult], dict[int, ItemResult], dict[int, dict[str, Any]]]:
    all_ids = list(records)
    eligible = [
        item_id
        for item_id in all_ids
        if stage1_results[item_id].status == "ok" and stage2_results[item_id].status == "ok"
    ]
    stage3_results = {item_id: ItemResult(errors=["Earlier stage failed"]) for item_id in all_ids}
    stage4_results = {item_id: ItemResult(status="not_run") for item_id in all_ids}
    outcomes = {
        item_id: {
            "row_status": "failed",
            "regen_count": 0,
            "validator_history": [],
        }
        for item_id in all_ids
    }
    if eligible:
        stage3_results.update(
            generate_stage3(
                ids=eligible,
                records=records,
                stage1_results=stage1_results,
                stage2_results=stage2_results,
                cfg=cfg,
                usage=usage,
            )
        )

    for item_id in eligible:
        if stage3_results[item_id].status != "ok" or stage3_results[item_id].value is None:
            outcomes[item_id]["row_status"] = "stage3_failed"
            continue
        if not cfg["use_validator"]:
            stage4_results[item_id] = ItemResult(status="skipped")
            outcomes[item_id]["row_status"] = "accepted"
            continue

        intended = str((stage2_results[item_id].value or {}).get("severity", ""))
        intended_timing = str(
            (stage1_results[item_id].value or {}).get("postpartum_timing", "unknown")
        )
        for validation_round in range(cfg["max_regen"] + 1):
            narrative = str((stage3_results[item_id].value or {}).get("synthetic_text", ""))
            validation = stage4_single(item_id, narrative, cfg, usage, validation_round)
            stage4_results[item_id] = validation
            history_entry = {
                "round": validation_round,
                "status": validation.status,
                "prediction": (validation.value or {}).get("predicted_severity", ""),
                "predicted_timing": (validation.value or {}).get("predicted_timing", ""),
                "confidence": (validation.value or {}).get("confidence", ""),
                "errors": validation.errors,
            }
            outcomes[item_id]["validator_history"].append(history_entry)
            if validation.status != "ok" or validation.value is None:
                outcomes[item_id]["row_status"] = "validator_failed"
                break
            severity_matches = validation.value.get("predicted_severity") == intended
            timing_matches = (
                not cfg["use_timing"]
                or intended_timing == "unknown"
                or validation.value.get("predicted_timing") == intended_timing
            )
            if severity_matches and timing_matches:
                outcomes[item_id]["row_status"] = "accepted"
                break
            if validation_round >= cfg["max_regen"]:
                if not severity_matches and not timing_matches:
                    outcomes[item_id]["row_status"] = "rejected_severity_and_timing_mismatch"
                elif not severity_matches:
                    outcomes[item_id]["row_status"] = "rejected_severity_mismatch"
                else:
                    outcomes[item_id]["row_status"] = "rejected_timing_mismatch"
                break

            outcomes[item_id]["regen_count"] += 1
            regenerated = generate_stage3(
                ids=[item_id],
                records=records,
                stage1_results=stage1_results,
                stage2_results=stage2_results,
                cfg=cfg,
                usage=usage,
                regeneration_attempt=validation_round + 1,
                validator_feedback={
                    item_id: {
                        "predicted_severity": validation.value.get("predicted_severity"),
                        "intended_severity": intended,
                        "predicted_timing": validation.value.get("predicted_timing"),
                        "intended_timing": intended_timing,
                    }
                },
            )[item_id]
            stage3_results[item_id] = regenerated
            if regenerated.status != "ok":
                outcomes[item_id]["row_status"] = "regen_failed"
                break

    return stage3_results, stage4_results, outcomes


def last_error(result: ItemResult) -> str:
    if result.status in {"ok", "skipped"}:
        return ""
    return result.errors[-1] if result.errors else ""


def process_batch(task: tuple[int, list[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    batch_id, input_rows, cfg = task
    log_file = Path(cfg["log_file"])
    log(f"Batch {batch_id}: starting {len(input_rows)} row(s)", log_file)
    started = time.monotonic()
    usage = Usage()
    records = {
        int(row["_source_index"]): {
            "source_text": str(row.get(cfg["text_column"], "")),
            "input": row,
        }
        for row in input_rows
    }
    s1 = stage1(records, cfg, usage)
    s2 = stage2(s1, cfg, usage)
    s3, s4, outcomes = stages3_and4(
        records=records,
        stage1_results=s1,
        stage2_results=s2,
        cfg=cfg,
        usage=usage,
    )
    elapsed = time.monotonic() - started
    rows: list[dict[str, Any]] = []
    for item_id, record in records.items():
        source = record["source_text"]
        factors = s1[item_id].value or {}
        severity = s2[item_id].value or {}
        generated = s3[item_id].value or {}
        validation = s4[item_id].value or {}
        text = str(generated.get("synthetic_text", ""))
        intended = str(severity.get("severity", ""))
        predicted = str(validation.get("predicted_severity", ""))
        timing = str(factors.get("postpartum_timing", ""))
        predicted_timing = str(validation.get("predicted_timing", ""))
        row = record["input"]
        rows.append(
            {
                "source_index": item_id,
                "batch_id": batch_id,
                "source_post": source,
                "source_label": row.get(cfg["label_column"], "") if cfg["label_column"] else "",
                "source_category": row.get(cfg["category_column"], "") if cfg["category_column"] else "",
                "source_sentiment": row.get(cfg["sentiment_column"], "") if cfg["sentiment_column"] else "",
                "s1_status": s1[item_id].status,
                "s1_attempts": s1[item_id].attempts,
                "s1_error": last_error(s1[item_id]),
                "s1_deidentified_summary": factors.get("deidentified_summary", FAILED_SENTINEL),
                "s1_postpartum_timing": timing,
                "s1_symptoms": factors.get("symptoms", []),
                "s1_symptom_intensity": factors.get("symptom_intensity", ""),
                "s1_symptom_persistence": factors.get("symptom_persistence", ""),
                "s1_functional_impact": factors.get("functional_impact", []),
                "s1_sleep_context": factors.get("sleep_context", ""),
                "s1_feeding_or_infant_care_stressors": factors.get("feeding_or_infant_care_stressors", []),
                "s1_perceived_support": factors.get("perceived_support", ""),
                "s1_bonding_indicators": factors.get("bonding_indicators", []),
                "s1_identifier_leaks": identifier_leaks(text_values(factors)),
                "s1_shared_source_ngrams": shared_ngram_count(
                    source, text_values(factors), cfg["copy_ngram_size"]
                ),
                "s2_status": s2[item_id].status,
                "s2_attempts": s2[item_id].attempts,
                "s2_error": last_error(s2[item_id]),
                "s2_severity": intended,
                "s2_rationale": severity.get("rationale", FAILED_SENTINEL),
                "s3_status": s3[item_id].status,
                "s3_attempts": s3[item_id].attempts,
                "s3_error": last_error(s3[item_id]),
                "s3_text": text or FAILED_SENTINEL,
                "s3_target": generated.get("target", ""),
                "s3_timing": generated.get("timing", ""),
                "s3_style": generated.get("style", ""),
                "s3_word_count": word_count(text),
                "s3_identifier_leaks": identifier_leaks(text),
                "s3_shared_source_ngrams": shared_ngram_count(source, text, cfg["copy_ngram_size"]),
                "s3_unsupported_detail_flags": unsupported_detail_categories(text, factors),
                "s3_timing_cue_injected": bool(generated.get("timing_cue_injected", False)),
                "s3_regen_count": outcomes[item_id]["regen_count"],
                "s4_status": s4[item_id].status,
                "s4_attempts": s4[item_id].attempts,
                "s4_error": last_error(s4[item_id]),
                "s4_predicted_severity": predicted,
                "s4_predicted_timing": predicted_timing,
                "s4_confidence": validation.get("confidence", ""),
                "s4_evidence": validation.get("evidence", ""),
                "s4_history": outcomes[item_id]["validator_history"],
                "severity_agreement": bool(predicted and predicted == intended),
                "timing_agreement": bool(
                    cfg["use_timing"] and predicted_timing and predicted_timing == timing
                ),
                "row_status": outcomes[item_id]["row_status"],
                "batch_elapsed_seconds": round(elapsed, 3),
                "batch_llm_calls": usage.calls,
                "batch_prompt_tokens": usage.prompt_tokens,
                "batch_output_tokens": usage.output_tokens,
                "batch_ollama_seconds": round(usage.seconds, 3),
            }
        )
    atomic_write_jsonl(batch_path(Path(cfg["outdir"]), batch_id), rows)
    accepted = sum(row["row_status"] == "accepted" for row in rows)
    log(
        f"Batch {batch_id}: completed in {elapsed:.1f}s; {accepted}/{len(rows)} accepted; "
        f"{usage.calls} LLM call(s)",
        log_file,
    )
    return {"batch_id": batch_id, "row_count": len(rows)}


def valid_batch(path: Path, expected_source_indices: Sequence[int]) -> bool:
    rows = load_jsonl(path)
    if rows is None or len(rows) != len(expected_source_indices):
        return False
    required = {"source_index", "batch_id", "row_status", "s3_text"}
    if not all(required.issubset(row) for row in rows):
        return False
    try:
        actual = [int(row["source_index"]) for row in rows]
    except (TypeError, ValueError):
        return False
    return actual == list(expected_source_indices)


def refresh_checkpoint(outdir: Path, batches: Sequence[tuple[int, list[dict[str, Any]]]]) -> None:
    completed = []
    for batch_id, rows in batches:
        indices = [int(row["_source_index"]) for row in rows]
        if valid_batch(batch_path(outdir, batch_id), indices):
            completed.append(batch_id)
    atomic_write_text(
        outdir / "checkpoint_done.txt",
        "".join(f"{batch_id}\n" for batch_id in completed),
    )


def manifest_config(args: argparse.Namespace, selected_count: int) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    return {
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(script_path),
        "input_path": str(Path(args.input).resolve()),
        "input_sha256": sha256_file(Path(args.input)),
        "start_row": args.start_row,
        "limit_rows": args.limit_rows,
        "selected_rows": selected_count,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "category_column": args.category_column,
        "sentiment_column": args.sentiment_column,
        "ollama_host": args.ollama_host,
        "ollama_model": args.ollama_model,
        "workers": args.workers,
        "batch_size": args.batch_size,
        "schema_retries": args.schema_retries,
        "transport_retries": args.transport_retries,
        "request_timeout": args.request_timeout,
        "num_ctx": args.num_ctx,
        "seed": args.seed,
        "use_timing": not args.no_timing,
        "use_validator": not args.no_validator,
        "max_regen": args.max_regen,
        "temp_s1": args.temp_s1,
        "temp_s2": args.temp_s2,
        "temp_s3": args.temp_s3,
        "temp_s4": args.temp_s4,
        "tokens_s1_per_item": args.tokens_s1_per_item,
        "tokens_s2_per_item": args.tokens_s2_per_item,
        "tokens_s3_per_item": args.tokens_s3_per_item,
        "tokens_s4_per_item": args.tokens_s4_per_item,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "copy_ngram_size": args.copy_ngram_size,
        "max_copy_ngrams": args.max_copy_ngrams,
        "timing_buckets": list(TIMING_BUCKETS),
        "epds_labels": list(EPDS_LABELS),
    }


def validate_or_create_manifest(path: Path, requested: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        manifest = {**requested, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        atomic_write_json(path, manifest)
        return manifest
    with path.open(encoding="utf-8") as handle:
        existing = json.load(handle)
    ignored = {"created_at", "model_digest", "last_completed_at"}
    mismatches = [
        f"{key}: existing={existing.get(key)!r}, requested={value!r}"
        for key, value in requested.items()
        if key not in ignored and existing.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "The output directory belongs to a different run configuration. "
            "Use a new --outdir.\n  - " + "\n  - ".join(mismatches)
        )
    return existing


def update_manifest_runtime(path: Path, manifest: dict[str, Any], digest: str | None) -> None:
    updated = dict(manifest)
    updated["model_digest"] = digest
    updated["last_completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write_json(path, updated)


def stringify(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return value


def merge_batches(
    outdir: Path,
    batches: Sequence[tuple[int, list[dict[str, Any]]]],
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_id, input_rows in batches:
        path = batch_path(outdir, batch_id)
        expected = [int(row["_source_index"]) for row in input_rows]
        if not valid_batch(path, expected):
            raise RuntimeError(f"Cannot merge missing or invalid checkpoint: {path}")
        loaded = load_jsonl(path)
        assert loaded is not None
        rows.extend(loaded)
    rows.sort(key=lambda row: int(row["source_index"]))

    seen: set[str] = set()
    for row in rows:
        if row.get("row_status") != "accepted":
            continue
        normalized = " ".join(normalized_tokens(str(row.get("s3_text", ""))))
        if not normalized or normalized in seen:
            row["row_status"] = "rejected_global_duplicate"
        else:
            seen.add(normalized)

    audit_path = outdir / "combined_private_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: stringify(value) for key, value in row.items()})

    public_fields = (
        "synthetic_id",
        "synthetic_text",
        "epds_bucket",
        "postpartum_timing",
        "word_count",
        "validator_predicted_severity",
        "validator_confidence",
        "regeneration_count",
    )
    public_path = outdir / "synthetic_dataset.csv"
    accepted_rows = [row for row in rows if row.get("row_status") == "accepted"]
    with public_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_fields)
        writer.writeheader()
        for number, row in enumerate(accepted_rows, start=1):
            writer.writerow(
                {
                    "synthetic_id": f"S{number:06d}",
                    "synthetic_text": row["s3_text"],
                    "epds_bucket": row["s2_severity"],
                    "postpartum_timing": row["s1_postpartum_timing"],
                    "word_count": row["s3_word_count"],
                    "validator_predicted_severity": row["s4_predicted_severity"],
                    "validator_confidence": row["s4_confidence"],
                    "regeneration_count": row["s3_regen_count"],
                }
            )

    batch_first_rows: dict[int, dict[str, Any]] = {}
    for row in rows:
        batch_first_rows.setdefault(int(row["batch_id"]), row)
    validated = [row for row in rows if row.get("s4_status") == "ok"]
    timing_validated = [
        row for row in validated if row.get("s1_postpartum_timing") != "unknown"
    ]
    total = len(rows)
    status_counts = Counter(str(row.get("row_status", "unknown")) for row in rows)

    def rate(count: int, denominator: int = total) -> float | None:
        return round(100.0 * count / denominator, 2) if denominator else None

    summary = {
        "total_rows": total,
        "accepted_rows": len(accepted_rows),
        "acceptance_rate_pct": rate(len(accepted_rows)),
        "status_counts": dict(status_counts),
        "stage_failure_rates_pct": {
            "stage1": rate(sum(row["s1_status"] != "ok" for row in rows)),
            "stage2": rate(sum(row["s2_status"] != "ok" for row in rows)),
            "stage3": rate(sum(row["s3_status"] != "ok" for row in rows)),
            "stage4": rate(sum(row["s4_status"] == "failed" for row in rows)),
        },
        "assigned_severity_counts": dict(Counter(row["s2_severity"] for row in rows if row["s2_severity"])),
        "accepted_severity_counts": dict(Counter(row["s2_severity"] for row in accepted_rows)),
        "timing_counts": dict(Counter(row["s1_postpartum_timing"] for row in rows if row["s1_postpartum_timing"])),
        "severity_agreement_rate_pct": rate(
            sum(bool(row["severity_agreement"]) for row in validated), len(validated)
        ),
        "timing_agreement_rate_pct": rate(
            sum(bool(row["timing_agreement"]) for row in timing_validated),
            len(timing_validated),
        ) if cfg["use_timing"] else None,
        "timing_agreement_evaluable_rows": len(timing_validated) if cfg["use_timing"] else None,
        "rows_regenerated": sum(int(row["s3_regen_count"]) > 0 for row in rows),
        "total_regenerations": sum(int(row["s3_regen_count"]) for row in rows),
        "source_copy_violations_in_saved_rows": sum(
            int(row["s3_shared_source_ngrams"]) > cfg["max_copy_ngrams"] for row in rows
        ),
        "identifier_violations_in_saved_rows": sum(bool(row["s3_identifier_leaks"]) for row in rows),
        "rows_flagged_for_unsupported_detail_review": sum(
            bool(row.get("s3_unsupported_detail_flags")) for row in rows
        ),
        "llm_usage": {
            "calls": sum(int(row["batch_llm_calls"]) for row in batch_first_rows.values()),
            "prompt_tokens": sum(int(row["batch_prompt_tokens"]) for row in batch_first_rows.values()),
            "output_tokens": sum(int(row["batch_output_tokens"]) for row in batch_first_rows.values()),
            "ollama_seconds": round(
                sum(float(row["batch_ollama_seconds"]) for row in batch_first_rows.values()), 3
            ),
        },
        "private_audit_file": str(audit_path),
        "public_synthetic_file": str(public_path),
    }
    atomic_write_json(outdir / "pilot_diagnostics.json", summary)
    return rows, summary


def load_input(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    path = Path(args.input)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if args.text_column not in columns:
            raise ValueError(
                f"Input is missing --text-column {args.text_column!r}. Available: {columns}"
            )
        rows = []
        for source_index, row in enumerate(reader):
            if str(row.get(args.text_column, "")).strip():
                row["_source_index"] = source_index
                rows.append(row)
    if args.start_row:
        rows = rows[args.start_row :]
    if args.limit_rows > 0:
        rows = rows[: args.limit_rows]
    return rows, columns


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive = {
        "--workers": args.workers,
        "--batch-size": args.batch_size,
        "--schema-retries": args.schema_retries,
        "--transport-retries": args.transport_retries,
        "--request-timeout": args.request_timeout,
        "--num-ctx": args.num_ctx,
        "--tokens-s1-per-item": args.tokens_s1_per_item,
        "--tokens-s2-per-item": args.tokens_s2_per_item,
        "--tokens-s3-per-item": args.tokens_s3_per_item,
        "--tokens-s4-per-item": args.tokens_s4_per_item,
        "--copy-ngram-size": args.copy_ngram_size,
    }
    for name, value in positive.items():
        if value < 1:
            parser.error(f"{name} must be at least 1")
    if args.start_row < 0 or args.limit_rows < 0 or args.max_regen < 0:
        parser.error("--start-row, --limit-rows, and --max-regen cannot be negative")
    if args.max_copy_ngrams < 0:
        parser.error("--max-copy-ngrams cannot be negative")
    if not 0.0 <= args.max_rejection_rate <= 1.0:
        parser.error("--max-rejection-rate must be between 0 and 1")
    if not 0.0 <= args.top_p <= 1.0:
        parser.error("--top-p must be between 0 and 1")
    if args.min_words < 1 or args.max_words < args.min_words:
        parser.error("word limits must satisfy 1 <= --min-words <= --max-words")
    if args.workers > 1 and args.limit_rows and args.limit_rows <= 100:
        print("NOTE: --workers 1 is usually fastest and safest for CPU-only Ollama.", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Canonical four-stage PPD generator. Safe defaults process 100 rows on "
            "one CPU-friendly worker; use --limit-rows 0 deliberately for all rows."
        ),
    )
    parser.add_argument("--input", required=True, help="Source CSV")
    parser.add_argument("--outdir", required=True, help="Fresh run directory, or same directory to resume")
    parser.add_argument("--ollama-model", required=True, help="Exact pulled Ollama model tag")
    parser.add_argument("--ollama-host", default=OLLAMA_HOST_DEFAULT)
    parser.add_argument("--text-column", default="Post")
    parser.add_argument("--label-column", default="Label")
    parser.add_argument("--category-column", default="Category")
    parser.add_argument("--sentiment-column", default="Sentiment")
    parser.add_argument("--start-row", type=int, default=0, help="Offset after blank-text rows are removed")
    parser.add_argument("--limit-rows", type=int, default=100, help="Rows to process; 0 means all rows")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent Ollama requests; keep 1 on a laptop")
    parser.add_argument("--batch-size", type=int, default=2, help="Items per Stage 1-3 request")
    parser.add_argument("--schema-retries", type=int, default=2)
    parser.add_argument("--transport-retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=int, default=1800, help="Read timeout per CPU inference request")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7930)
    parser.add_argument("--no-timing", action="store_true", help="Ablation: disable timing extraction/conditioning")
    parser.add_argument("--no-validator", action="store_true", help="Ablation: skip blind validation/regeneration")
    parser.add_argument("--max-regen", type=int, default=2)
    parser.add_argument("--temp-s1", type=float, default=0.3)
    parser.add_argument("--temp-s2", type=float, default=0.2)
    parser.add_argument("--temp-s3", type=float, default=1.1)
    parser.add_argument("--temp-s4", type=float, default=0.2)
    parser.add_argument("--tokens-s1-per-item", type=int, default=320)
    parser.add_argument("--tokens-s2-per-item", type=int, default=120)
    parser.add_argument("--tokens-s3-per-item", type=int, default=520)
    parser.add_argument("--tokens-s4-per-item", type=int, default=220)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repeat-penalty", type=float, default=1.15)
    parser.add_argument("--min-words", type=int, default=120)
    parser.add_argument("--max-words", type=int, default=250)
    parser.add_argument("--copy-ngram-size", type=int, default=8)
    parser.add_argument("--max-copy-ngrams", type=int, default=0)
    parser.add_argument("--max-rejection-rate", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration/input without contacting Ollama")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    input_path = Path(args.input).resolve()
    if not input_path.is_file():
        parser.error(f"--input is not a file: {input_path}")
    args.input = str(input_path)
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    log_file = outdir / "batch_log.txt"

    try:
        rows, columns = load_input(args)
    except (OSError, ValueError) as exc:
        log(f"ERROR: {exc}", log_file)
        return 2
    if not rows:
        log("ERROR: no non-empty source rows were selected", log_file)
        return 2

    for optional_name in ("label_column", "category_column", "sentiment_column"):
        column = getattr(args, optional_name)
        if column and column not in columns:
            log(f"Optional column {column!r} not found; it will be blank.", log_file)
            setattr(args, optional_name, "")

    manifest_path = outdir / "run_manifest.json"
    requested_manifest = manifest_config(args, len(rows))
    try:
        manifest = validate_or_create_manifest(manifest_path, requested_manifest)
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as exc:
        log(f"ERROR: {exc}", log_file)
        return 2

    batches = [
        (batch_id, rows[offset : offset + args.batch_size])
        for batch_id, offset in enumerate(range(0, len(rows), args.batch_size))
    ]
    completed = []
    pending = []
    for batch_id, batch_rows in batches:
        expected = [int(row["_source_index"]) for row in batch_rows]
        if valid_batch(batch_path(outdir, batch_id), expected):
            completed.append(batch_id)
        else:
            pending.append((batch_id, batch_rows))
    refresh_checkpoint(outdir, batches)

    log("=" * 72, log_file)
    log(f"{SCRIPT_VERSION}", log_file)
    log(
        f"Selected rows={len(rows)}; batches={len(batches)}; completed={len(completed)}; "
        f"pending={len(pending)}",
        log_file,
    )
    log(
        f"Laptop-safe settings: workers={args.workers}, batch_size={args.batch_size}, "
        f"request_timeout={args.request_timeout}s",
        log_file,
    )
    log(f"Private output directory: {outdir}", log_file)
    log("=" * 72, log_file)

    minimum_calls = len(pending) * 3 + (len(rows) if not args.no_validator else 0)
    log(
        f"Estimated minimum remaining LLM calls: about {minimum_calls} "
        "(retries/regeneration can increase this).",
        log_file,
    )
    if args.dry_run:
        log("Dry run complete; Ollama was not contacted.", log_file)
        return 0

    cfg = {
        "model": args.ollama_model,
        "ollama_host": args.ollama_host,
        "workers": args.workers,
        "schema_retries": args.schema_retries,
        "transport_retries": args.transport_retries,
        "request_timeout": args.request_timeout,
        "num_ctx": args.num_ctx,
        "seed": args.seed,
        "use_timing": not args.no_timing,
        "use_validator": not args.no_validator,
        "max_regen": args.max_regen,
        "temp_s1": args.temp_s1,
        "temp_s2": args.temp_s2,
        "temp_s3": args.temp_s3,
        "temp_s4": args.temp_s4,
        "tokens_s1_per_item": args.tokens_s1_per_item,
        "tokens_s2_per_item": args.tokens_s2_per_item,
        "tokens_s3_per_item": args.tokens_s3_per_item,
        "tokens_s4_per_item": args.tokens_s4_per_item,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "copy_ngram_size": args.copy_ngram_size,
        "max_copy_ngrams": args.max_copy_ngrams,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "category_column": args.category_column,
        "sentiment_column": args.sentiment_column,
        "outdir": str(outdir),
        "log_file": str(log_file),
    }

    digest = model_digest(args.ollama_host, args.ollama_model)
    if pending:
        warmup = call_ollama(
            cfg=cfg,
            prompt="Reply with only the word ready.",
            temperature=0.1,
            max_tokens=8,
            seed=args.seed,
        )
        if not warmup.text:
            log(
                "ERROR: Ollama preflight failed. Verify that Ollama is running and "
                f"model {args.ollama_model!r} is pulled. Detail: {warmup.error}",
                log_file,
            )
            return 3
        log("Ollama preflight succeeded.", log_file)

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_batch, (batch_id, batch_rows, cfg)): batch_id
                    for batch_id, batch_rows in pending
                }
                finished = 0
                for future in as_completed(futures):
                    batch_id = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        log(f"ERROR: batch {batch_id} crashed: {type(exc).__name__}: {exc}", log_file)
                        return 4
                    finished += 1
                    refresh_checkpoint(outdir, batches)
                    log(f"Progress: {finished}/{len(pending)} pending batches completed", log_file)
        except KeyboardInterrupt:
            refresh_checkpoint(outdir, batches)
            log("Interrupted. Durable completed batches are safe; rerun the same command to resume.", log_file)
            return 130
    else:
        log("All batches already have valid checkpoints; rebuilding final outputs.", log_file)

    refresh_checkpoint(outdir, batches)
    try:
        _, summary = merge_batches(outdir, batches, cfg)
    except (OSError, ValueError, RuntimeError) as exc:
        log(f"ERROR during merge: {exc}", log_file)
        return 5
    update_manifest_runtime(manifest_path, manifest, digest)

    rejection_rate = 1.0 - (summary["accepted_rows"] / summary["total_rows"])
    log("=" * 72, log_file)
    log(
        f"Finished: {summary['accepted_rows']}/{summary['total_rows']} accepted "
        f"({summary['acceptance_rate_pct']}%).",
        log_file,
    )
    log(f"Status counts: {summary['status_counts']}", log_file)
    log(f"LLM usage: {summary['llm_usage']}", log_file)
    log("PRIVATE (contains source posts): combined_private_audit.csv", log_file)
    log("SYNTHETIC-ONLY output (review before release): synthetic_dataset.csv", log_file)
    if rejection_rate > args.max_rejection_rate:
        log(
            f"WARNING: rejection rate {rejection_rate:.1%} exceeds the configured "
            f"{args.max_rejection_rate:.1%}. Inspect pilot_diagnostics.json before scaling.",
            log_file,
        )
    log("Done. Automatic laptop shutdown is intentionally not implemented.", log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
