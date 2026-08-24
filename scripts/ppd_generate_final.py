#!/usr/bin/env python3
"""Canonical four-stage PPD synthetic-narrative generator.

Author: Sara Sezavar

Pipeline
--------
An eligibility check excludes clearly non-postpartum or anticipatory seeds.
1. De-identify the source post and extract generalized clinical factors.
2. Assign an EPDS-aligned narrative severity bucket from those factors.
3. Generate an original narrative conditioned on severity and (optionally) timing.
4. Blindly validate the narrative; regenerate on severity mismatch, never relabel.

The safe defaults are intended for an overnight laptop pilot: the first 100 rows,
one worker, and batches of two. Pass ``--limit-rows 0`` only for a deliberate full
run. Ollama must already be running and the selected model must already be pulled.

Only the eligibility check and Stage 1 receive raw source text. Failed generation
never falls back to raw text. Every completed batch is written atomically and can
be resumed safely.

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


SCRIPT_VERSION = "ppd_generate_final v4.14"
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
STAGE3_SENTENCE_COUNT = 4

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


def source_timing_bucket(source_text: str) -> str:
    """Extract timing only from an explicit infant-age or post-birth expression."""
    number_words = {
        "a": 1.0, "one": 1.0, "two": 2.0, "couple": 2.0, "three": 3.0,
        "few": 3.0, "four": 4.0, "several": 4.0, "five": 5.0, "six": 6.0,
        "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
        "eleven": 11.0, "twelve": 12.0,
    }
    number = r"(?:\d+(?:\.\d+)?|a|one|two|couple|three|few|four|several|five|six|seven|eight|nine|ten|eleven|twelve)"
    patterns = (
        re.compile(
            rf"\b(?:baby|infant|child)(?:\s+is|'s)?\s+(?P<n>{number})\s*"
            r"(?P<u>days?|weeks?|months?)(?:\s*[- ]?old)?\b",
            re.I,
        ),
        re.compile(
            rf"\b(?P<n>{number})\s*(?P<u>days?|weeks?|months?)\s*"
            r"(?:postpartum\b|(?:after|since)\s*(?:birth|delivery|giving birth|having (?:a|my) baby)\b)",
            re.I,
        ),
        re.compile(
            rf"\b(?:gave birth|delivered|had (?:a|my(?: (?:\d+(?:st|nd|rd|th)|first|second|third|fourth))?) baby)\s+"
            rf"(?:almost|about|around)?\s*(?P<n>{number})\s*"
            r"(?P<u>days?|weeks?|months?)\s+ago\b",
            re.I,
        ),
    )
    later_match = re.search(
        rf"\b(?P<n>{number})\s*(?P<u>days?|weeks?|months?)\s+later\b",
        source_text or "",
        re.I,
    )
    has_birth_context = bool(re.search(
        r"\b(?:baby|birth|gave birth|delivered|postpartum)\b", source_text or "", re.I
    ))
    match = later_match if later_match and has_birth_context else next(
        (pattern.search(source_text or "") for pattern in patterns if pattern.search(source_text or "")),
        None,
    )
    if match is None:
        return "unknown"
    raw_number = match.group("n").lower()
    value = float(raw_number) if re.fullmatch(r"\d+(?:\.\d+)?", raw_number) else number_words[raw_number]
    unit = match.group("u").lower()
    weeks = value / 7.0 if unit.startswith("day") else value * 4.0 if unit.startswith("month") else value
    if weeks <= 2:
        return TIMING_BUCKETS[0]
    if weeks <= 6:
        return TIMING_BUCKETS[1]
    if weeks <= 12:
        return TIMING_BUCKETS[2]
    return TIMING_BUCKETS[3]


def source_persistence(source_text: str) -> str:
    """Keep duration conservative and tied to explicit source wording."""
    source = re.sub(r"\s+", " ", source_text or "").lower()
    if re.search(
        r"\b(?:no longer|recovered|resolved|feel more like myself|feeling more like myself|"
        r"doing better now|improved over time)\b",
        source,
    ):
        return "resolved"
    if re.search(
        r"\b(?:been (?:going|feeling|struggling|dealing|battling)|still (?:feel|feeling|have|"
        r"struggl\w*|suffer\w*|experienc\w*)|i am suffer\w*|i'm suffer\w*|ongoing|"
        r"every day|daily|constantly|keeps? (?:happening|coming back))\b",
        source,
    ):
        return "ongoing"
    if re.search(
        r"\b(?:i was suffer\w*|i went through|i experienced|after suffer\w*|"
        r"i had (?:postpartum )?(?:depress\w*|anxi\w*|intrusive thoughts?)|"
        r"when i tried to seek help|(?:caught|identified|recognized) my depress\w*|"
        r"years? ago|months? ago)\b",
        source,
    ):
        return "past, current status unknown"
    return "unknown"


def source_has_affirmed_risk(source_text: str, pattern: re.Pattern[str]) -> bool:
    """Ignore explicit denials while retaining affirmative risk mentions."""
    source = re.sub(r"\s+", " ", source_text or "").lower()
    for match in pattern.finditer(source):
        prefix = source[max(0, match.start() - 80):match.start()]
        if re.search(
            r"(?:\bnever(?:\s+\w+){0,6}|"
            r"\b(?:do|does|did|would|could|have|has|had|was|were) not(?:\s+\w+){0,6}|"
            r"\b(?:don't|doesn't|didn't|wouldn't|couldn't|haven't|hasn't|hadn't)(?:\s+\w+){0,6}|"
            r"\bno (?:thoughts?|ideas?|plans?|intentions?)(?:\s+\w+){0,4}|"
            r"\bwithout (?:any )?(?:thoughts?|ideas?|plans?|intentions?)(?:\s+\w+){0,4}|"
            r"\bden(?:y|ied|ies|ying)(?:\s+\w+){0,6})\s*$",
            prefix,
            re.I,
        ):
            continue
        return True
    return False


def source_eligibility(source_text: str) -> tuple[str, str]:
    """Screen seeds that do not describe a postpartum experience."""
    source = re.sub(r"\s+", " ", source_text or "").strip().lower()
    if re.search(r"\b(?:babysit|babysitting)\b|\bminus the baby\b", source):
        return "excluded_non_postpartum", "No postpartum experience is described."
    if re.search(
        r"\b(?:men|man|father\w*)\b.{0,90}\bdid(?:n't| not) have to "
        r"(?:produce|have|deliver|give birth to) (?:a |the )?baby\b",
        source,
    ):
        return "excluded_non_postpartum", "The text discusses depression outside a postpartum experience."
    if re.search(
        r"\b(?:when|if) i (?:have|get|deliver|give birth to) (?:my |a |the )?baby\b",
        source,
    ) and re.search(r"\b(?:will|ill|i'll|might|may|scared|afraid|worried)\b", source):
        return "excluded_anticipatory", "The text is anticipatory rather than a current or past postpartum account."

    postpartum_evidence = re.search(
        r"\b(?:postpartum|post-partum|ppd|baby blues|maternity leave|motherhood|"
        r"breast ?milk|breastfeed\w*|gave birth|giving birth|delivered|pushed a whole human|"
        r"after (?:having )?(?:a|my|the) baby|first had (?:a|my|the) baby|"
        r"had (?:a|my|the|his|her|our) baby|had my\b.{0,20}\bbaby|"
        r"new baby|my baby|our baby|the baby|this baby|"
        r"baby(?:'s)? first year|baby duty|hold my baby|baby cried|pediatrician)\b",
        source,
    )
    if postpartum_evidence:
        return "eligible", "Postpartum context is explicit in the source."
    return "excluded_non_postpartum", "Postpartum context is not explicit in the source."


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


def blind_timing_bucket(text: str) -> str:
    """Read timing from the narrative alone; never infer it from infant context."""
    matches = [
        bucket for bucket, pattern in _TIMING_NARRATIVE_CUES.items()
        if pattern.search(text or "")
    ]
    return matches[0] if len(matches) == 1 else "unknown"


def calibrate_blind_severity(text: str, predicted: str) -> str:
    """Enforce the documented severity gates on the blind narrative rating."""
    value = text or ""
    intensity_match = re.search(r"\b(low|moderate|high)[- ]intensity\b|\bat (low|moderate|high) intensity\b", value, re.I)
    intensity = ""
    if intensity_match:
        intensity = next(group for group in intensity_match.groups() if group).lower()
    persistent = bool(re.search(
        r"\b(?:persistent|persisted|ongoing|constant|continuous|sustained|daily|"
        r"every day|keeps? returning|does not go away|doesn't go away)\b",
        value,
        re.I,
    ))
    impairment = bool(re.search(
        r"\b(?:impair\w*|unable to function|cannot function|can't function|"
        r"hard to (?:complete|manage|handle|do) (?:basic |daily |everyday )?(?:tasks|activities)|"
        r"struggl\w* to (?:complete|manage|handle) (?:basic |daily |everyday )?(?:tasks|activities)|"
        r"disrupt\w* (?:my )?(?:daily life|functioning)|affect\w* (?:my )?daily life)\b",
        value,
        re.I,
    ))
    bonding_or_risk = bool(_SUPPORTED_DETAIL_PATTERNS["high-risk/bonding detail"].search(value))
    resolved = bool(re.search(r"\b(?:resolved|recovered|improved|no longer present)\b", value, re.I))
    moderate_gate = (persistent and intensity in {"moderate", "high"}) or impairment or bonding_or_risk
    severe_gate = intensity == "high" and (impairment or bonding_or_risk)
    if severe_gate:
        return "Severe"
    if moderate_gate:
        return "Moderate"
    if predicted in {"Moderate", "Severe"}:
        return "Mild"
    if predicted == "Minimal" and resolved and intensity in {"moderate", "high"}:
        return "Mild"
    return predicted


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


def materialize_stage3_text(item: dict[str, Any]) -> str:
    """Join structured sentence or paragraph slots into the narrative field."""
    sentences = item.get("sentences")
    sentence_values: list[Any] = []
    if isinstance(sentences, dict):
        sentence_values = [sentences[key] for key in sorted(sentences)]
    elif isinstance(sentences, list):
        sentence_values = sentences
    cleaned_sentences = [str(value).strip() for value in sentence_values if str(value).strip()]
    if cleaned_sentences:
        group_size = 2 if len(cleaned_sentences) <= STAGE3_SENTENCE_COUNT else 4
        paragraphs = [
            " ".join(cleaned_sentences[index : index + group_size])
            for index in range(0, len(cleaned_sentences), group_size)
        ]
        item["synthetic_text"] = "\n\n".join(paragraphs)
        return str(item["synthetic_text"])

    paragraphs = item.get("paragraphs")
    if isinstance(paragraphs, list):
        cleaned = [str(value).strip() for value in paragraphs if str(value).strip()]
        if cleaned:
            item["synthetic_text"] = "\n\n".join(cleaned)
    return str(item.get("synthetic_text") or "")


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


def missing_list_value(value: Any) -> bool:
    """Recognize empty model placeholders before they become factual support."""
    normalized = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not normalized:
        return True
    return normalized in {
        "none", "unknown", "n/a", "na", "not stated", "not specified",
        "not mentioned", "none stated", "none specified", "none mentioned",
    } or normalized.endswith(" not specified")


def preserve_explicit_source_factors(item: dict[str, Any], source_text: str) -> None:
    """Recover a few explicit cues that small models sometimes omit."""
    source = re.sub(r"\s+", " ", source_text or "").strip().lower()
    persistence = str(item.get("symptom_persistence") or "").strip().lower()
    if persistence in {"", "unknown", "not stated", "not specified"} and re.search(
        r"\b(?:been (?:going|feeling|struggling|dealing|battling)|"
        r"still (?:feel|feeling|struggl\w*|have|suffer\w*|experienc\w*)|"
        r"ongoing|every day|daily|constantly|keeps? (?:happening|coming back))\b",
        source,
    ):
        item["symptom_persistence"] = "ongoing"

    coping = str(item.get("coping_or_adjustment_context") or "").strip().lower()
    if coping in {"", "unknown", "not stated", "not specified"} and re.search(
        r"\b(?:i|we) (?:just )?need(?:ed)? (?:some )?time\b", source
    ):
        item["coping_or_adjustment_context"] = "needs time to adjust"

    explicit_intensity = re.search(
        r"\b(?P<level>mild|moderate|severe)\s+(?:postpartum\s+)?"
        r"(?:depress\w*|anxi\w*|symptoms?|distress)\b",
        source,
    )
    if explicit_intensity:
        item["symptom_intensity"] = {
            "mild": "low", "moderate": "moderate", "severe": "high",
        }[explicit_intensity.group("level")]


def generalize_precise_ages(item: dict[str, Any]) -> None:
    """Keep broad postpartum context without retaining exact child ages."""
    age_pattern = re.compile(
        r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve)[ -]?(?:day|week|month)s?[ -]old\b",
        re.I,
    )
    for key, value in list(item.items()):
        if key in {"id", "postpartum_timing"}:
            continue
        if isinstance(value, str):
            item[key] = age_pattern.sub("young", value)
        elif isinstance(value, list):
            item[key] = [
                age_pattern.sub("young", entry) if isinstance(entry, str) else entry
                for entry in value
            ]


def stage3_provenance(generated: dict[str, Any]) -> str:
    """Whether the accepted text came from the model or the deterministic fallback."""
    history = generated.get("grounding_history") or []
    for entry in history:
        if str(entry.get("round", "")).startswith("fallback"):
            return "fallback"
    return "llm" if history else ""


def conservative_factor_narrative(
    factors: dict[str, Any], severity: str, use_timing: bool,
    min_words: int, max_words: int, variant: int = 0,
) -> dict[str, Any]:
    """Build a grounded fallback directly from validated factor fields."""
    sentences: list[str] = []
    timing = str(factors.get("postpartum_timing") or "unknown")
    if use_timing and timing in _CANONICAL_TIMING_PREFIXES:
        sentences.append(_CANONICAL_TIMING_PREFIXES[timing])

    symptoms = [str(value).strip() for value in factors.get("symptoms", []) if str(value).strip()]
    intensity = str(factors.get("symptom_intensity") or "").strip().lower()
    persistence = str(factors.get("symptom_persistence") or "").strip().lower()
    past_experience = persistence.startswith("past,")
    if symptoms:
        symptom_text = ", ".join(symptoms[:-1])
        if len(symptoms) > 1:
            symptom_text += f" and {symptoms[-1]}"
        else:
            symptom_text = symptoms[0]
        resolved = persistence in {"resolved", "improved", "recovered", "no longer present"}
        if resolved or past_experience:
            openings = (
                f"I previously experienced {symptom_text}",
                f"My past experience included {symptom_text}",
                f"I describe a past experience of {symptom_text}",
            )
        elif persistence == "ongoing":
            openings = (
                f"I am experiencing {symptom_text}",
                f"My ongoing experience includes {symptom_text}",
                f"I continue to experience {symptom_text}",
            )
        else:
            openings = (
                f"I describe {symptom_text} as part of my experience",
                f"My account includes an experience of {symptom_text}",
                f"I identify an experience of {symptom_text}",
            )
        detail = openings[variant % len(openings)]
        if intensity in INTENSITIES:
            detail += f" at {intensity} intensity"
        if resolved:
            detail += ", and those symptoms have resolved"
        elif not past_experience and persistence not in {"", "unknown", "not stated", "not specified"}:
            detail += f", and it has been {persistence}"
        sentences.append(detail + ".")

    summary = str(factors.get("deidentified_summary") or "")
    if re.search(r"\b(?:unaware|did not (?:realize|recognize))\b", summary, re.I):
        sentences.append("I did not recognize this experience at first.")
    if re.search(r"\bemotional and physical changes\b", summary, re.I):
        sentences.append("I recognize emotional and physical changes following childbirth.")

    context_fields = (
        ("functional_impact", "I experience this functional impact: {}."),
        ("feeding_or_infant_care_stressors", "My stated infant-care stressor is {}."),
        ("bonding_indicators", "My bonding experience is {}."),
        ("risk_indicators", "My stated safety concern is {}."),
    )
    for key, template in context_fields:
        values = [str(value).strip() for value in factors.get(key, []) if str(value).strip()]
        if values:
            if key == "risk_indicators" and past_experience:
                sentences.append(f"My past safety concerns included {'; '.join(values)}.")
            elif key == "bonding_indicators" and values == ["positive bonding experience"]:
                sentences.append("I feel a positive bond with my baby.")
            else:
                sentences.append(template.format("; ".join(values)))

    for key, template in (
        ("sleep_context", "My stated sleep context is {}."),
        ("perceived_support", "My stated support experience is {}."),
    ):
        value = str(factors.get(key) or "").strip()
        if value.lower() not in {"", "unknown", "not stated", "not specified"}:
            if key == "perceived_support" and past_experience:
                sentences.append(f"At that time, my support experience was {value}.")
            else:
                sentences.append(template.format(value))

    coping = str(factors.get("coping_or_adjustment_context") or "").strip()
    if coping.lower() not in {"", "unknown", "not stated", "not specified"}:
        fragment = re.sub(r"^needs\b", "need", coping, flags=re.I)
        if re.match(r"^(?:need|want|am|feel|try|recognize|request|reached|sought|started)\b", fragment, re.I):
            sentences.append(f"I {fragment}.")
        else:
            sentences.append(f"My adjustment context is {fragment}.")

    # Keep sparse rows faithful by repeating only explicit categorical facts.
    if word_count(" ".join(sentences)) < min_words and symptoms:
        if resolved or past_experience:
            repeat_templates = (
                "I identify {} as part of my past experience.",
                "These reported symptoms belong to my past experience: {}.",
                "My account places {} in a past period of my life.",
            )
        elif persistence == "ongoing":
            repeat_templates = (
                "I identify {} as part of my ongoing experience.",
                "These symptoms remain part of my experience: {}.",
                "My ongoing account includes {}.",
            )
        else:
            repeat_templates = (
                "I identify {} among the symptoms I have described.",
                "The symptoms in my account include {}.",
                "My description of this experience includes {}.",
            )
        repeat_index = (variant // max(1, len(openings))) % len(repeat_templates)
        sentences.append(repeat_templates[repeat_index].format(symptom_text))
    if word_count(" ".join(sentences)) < min_words and intensity in INTENSITIES:
        sentences.append(f"I describe the intensity of this experience as {intensity}.")
    if (
        word_count(" ".join(sentences)) < min_words
        and persistence not in {"", "unknown", "not stated", "not specified"}
    ):
        sentences.append(f"I describe this experience as {persistence}.")
    if word_count(" ".join(sentences)) < min_words and symptoms:
        sentences.append(f"My stated symptom experience is {symptom_text}.")

    selected: list[str] = []
    for sentence in sentences:
        candidate = " ".join(selected + [sentence])
        if selected and word_count(candidate) > max_words:
            break
        selected.append(sentence)
    split_at = max(1, (len(selected) + 1) // 2)
    narrative = "\n\n".join(
        " ".join(part) for part in (selected[:split_at], selected[split_at:]) if part
    )
    return {
        "synthetic_text": narrative,
        "target": severity,
        "timing": timing if use_timing else "",
        "style": "first-person postpartum diary",
        "grounding_method": "deterministic_factor_fallback",
    }


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
    "functional impact": re.compile(
        r"\b(?:daily (?:activities|tasks|routine|life)|everyday (?:activities|tasks|life)|"
        r"household tasks|chores?|accomplish(?:ing)? tasks|complete tasks|manage tasks|"
        r"grocery shopping|cooking|unable to function|cannot function|can't function|"
        r"hard to get out of bed|basic tasks|functioning|work responsibilities|"
        r"get through (?:the )?day|responsibilities feel)\b",
        re.I,
    ),
    "social or assistance detail": re.compile(
        r"\b(?:ask(?:ing|ed)? for help|accept(?:ing|ed)? help|time alone|"
        r"connect with others|support from|friends?|people around me|isolat\w*|"
        r"relationships?|social(?:ly)?|withdraw(?:n|ing)?|avoid(?:ing)? (?:others|people))\b",
        re.I,
    ),
    "anxiety or worry": re.compile(r"\b(?:anxi\w*|worr(?:y|ied|ying)|panic\w*)\b", re.I),
    "guilt or self-worth": re.compile(
        r"\b(?:guilt\w*|worthless\w*|feel(?:ing)? like (?:a |i am )?fail\w*|"
        r"failing as|not (?:a )?good enough)\b",
        re.I,
    ),
    "sleep disturbance": re.compile(
        r"\b(?:insomnia|sleep(?:ing)? (?:poorly|badly|less)|can't sleep|cannot sleep|"
        r"waking (?:up )?(?:often|frequently)|restless nights?)\b",
        re.I,
    ),
    "fatigue or exhaustion": re.compile(
        r"\b(?:fatigu\w*|tired(?:ness)?|exhaust\w*|drained|no energy|low energy)\b", re.I
    ),
    "appetite detail": re.compile(
        r"\b(?:appetite|not eating|eat(?:ing)? (?:too much|too little|less)|loss of appetite)\b",
        re.I,
    ),
    "concentration detail": re.compile(
        r"\b(?:concentrat\w*|focus(?:ing)?|brain fog|forgetful\w*)\b", re.I
    ),
    "irritability or anger": re.compile(
        r"\b(?:irritab\w*|angry|anger|short[- ]tempered|snapp(?:y|ing))\b", re.I
    ),
    "overwhelm detail": re.compile(r"\b(?:overwhelm\w*|too much to handle)\b", re.I),
    "loss of interest": re.compile(
        r"\b(?:lost interest|loss of interest|no interest|nothing feels enjoyable|anhedoni\w*)\b",
        re.I,
    ),
    "physical discomfort": re.compile(
        r"\b(?:painful|in pain|aches?|sore(?:ness)?|nausea|dizz\w*|headaches?)\b", re.I
    ),
}

_EXTRA_TIMING_DETAIL = re.compile(
    r"\b(?:(?:for|about|around|nearly|over|more than)\s+)?"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a few|several)\s+"
    r"(?:days?|weeks?|months?)\b",
    re.I,
)


def has_extra_timing_detail(text: str, expected_timing: str) -> bool:
    """Allow the required bucket phrase, but no invented precise duration."""
    value = text or ""
    allowed = {
        TIMING_BUCKETS[0]: r"\bwithin the first two weeks after birth\b",
        TIMING_BUCKETS[1]: r"\bbetween two and six weeks after birth\b",
        TIMING_BUCKETS[2]: r"\bbetween six and twelve weeks after birth\b",
        TIMING_BUCKETS[3]: r"\bmore than three months after birth\b",
    }.get(expected_timing)
    if allowed:
        value = re.sub(allowed, "", value, count=1, flags=re.I)
    return bool(_EXTRA_TIMING_DETAIL.search(value))


def unsupported_detail_categories(text: str, factors: dict[str, Any]) -> list[str]:
    """Flag sensitive concrete details in a draft unless Stage-1 factors support them."""
    support_text = text_values(factors)
    categories = [
        category
        for category, pattern in _SUPPORTED_DETAIL_PATTERNS.items()
        if pattern.search(text or "") and not pattern.search(support_text)
    ]
    expected_timing = str(factors.get("postpartum_timing") or "unknown")
    if has_extra_timing_detail(text, expected_timing):
        categories.append("extra timing detail")
    return categories


def disallowed_content_categories(factors: dict[str, Any]) -> list[str]:
    """Tell the generator which sensitive categories have no factor support."""
    support_text = text_values(factors)
    categories = [
        category
        for category, pattern in _SUPPORTED_DETAIL_PATTERNS.items()
        if not pattern.search(support_text)
    ]
    categories.append("extra timing detail")
    return categories


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
- Preserve every supported emotional, cognitive, physical, coping, and functional
  factor. Do not compress a nuanced post into only a diagnosis or symptom word.
- Treat explicitly ongoing wording as ongoing. Use unknown only when duration is absent.
- For list fields, use an empty list when the source does not state a factor. Do not
  insert placeholders such as "not specified" or "none mentioned" into a list.
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
- "Moderate": persistent moderate/high mood symptoms, rumination, meaningful functional impairment,
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
- Moderate: persistent moderate/high symptoms or explicit meaningful, non-major functional difficulty.
  High intensity alone, without supported duration or impact, does not establish Moderate.
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
{{"id": 0, "deidentified_summary": "2-3 comprehensive generalized sentences", {timing_field}
 "symptoms": ["..."], "symptom_intensity": "low|moderate|high",
 "symptom_persistence": "...", "functional_impact": ["..."],
 "sleep_context": "...", "feeding_or_infant_care_stressors": ["..."],
 "perceived_support": "...", "bonding_indicators": ["..."],
 "risk_indicators": ["..."],
 "coping_or_adjustment_context": "..."}}

Extraction completeness. Inventing detail is still forbidden, but do not leave a field
empty when the source supports it. Work through the source and record everything it
actually states:
- List every distinct symptom, mood state, or emotional experience separately. Do not
  collapse several into one generic label.
- Treat self-directed beliefs as symptoms when stated, for example guilt, feeling like
  a bad parent, worthlessness, or believing the family would be better off without them.
- Record the content of intrusive or negative thoughts, not just the fact that they occurred.
- Record stated effects on daily functioning, work, or caregiving in functional_impact.
- Record any stated sleep, rest, or exhaustion detail in sleep_context.
- Record stated feeding, milk supply, or infant-care difficulties.
- Record stated help, isolation, partner or family involvement in perceived_support.
- Record stated treatment, exercise, medication, or self-management in
  coping_or_adjustment_context.
Leave a field empty only when the source genuinely says nothing about it. An empty field
means absent from the source, never merely unmentioned by you.

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
    sentence_schema = ", ".join(
        f'"s{index:02d}": "7-15 words"'
        for index in range(1, STAGE3_SENTENCE_COUNT + 1)
    )
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
            f"Fill all {STAGE3_SENTENCE_COUNT} required sentence slots this time. "
            "Each slot must contain 7-15 words. "
            "Do not omit slots or return a short summary.\n"
        )
    return f"""Generate original first-person postpartum diary narratives for
research. Stage 3 receives generalized factors only, never source posts.

{SEVERITY_DEFS}
{SEVERITY_BOUNDARY_RULES}
{timing}
{timing_instruction}{regeneration}{correction}
Constraints:
- The assembled narrative must contain {min_words}-{max_words} words.
- Fill exactly {STAGE3_SENTENCE_COUNT} ordered sentence slots, each containing
  7-15 words. The slots are joined into two paragraphs, producing 28-60 words.
- Ground every sentence in at least one supplied factor. With sparse factors,
  reflect on the same supported experience without adding symptoms, impairment,
  relationships, actions, causes, or duration.
- Write as the person, in the first person, using "I" and "my". Never write about her
  from the outside. Do not use "the individual", "the mother", "she", or "they".
- deidentified_summary is your main material. It usually carries the most concrete
  supported content, so build the entry from it rather than from the field labels.
- Never narrate the data itself. An empty field means stay silent about that topic, not
  describe it as missing. Do not write that something is unknown or unspecified, and do
  not mention severity, intensity or persistence as named quantities.
- Match the supplied severity in expressed intensity, persistence, and impairment.
- Minimal must show transient/low distress; Mild must show noticeable distress
  without adding persistence or impairment; Moderate must show persistent meaningful
  distress and include impairment only when supplied. Severe must use only supplied
  major-impairment, bonding-disruption, or urgent-risk factors.
- Do not copy or closely paraphrase an input field.
- Use only the supplied factors. Do not invent a partner, infant sex, occupation,
  feeding method, medical event, treatment, or support person unless supplied.
- Each input contains disallowed_content_categories. Do not mention or imply any
  detail from a listed category, even if it is commonly associated with the target.
- Use only the required timing phrase. Do not add another duration or numeric time cue.
- Keep the narrative introspective. Do not invent phone calls, visits, appointments,
  conversations, or other concrete events merely to add length.
- Do not add names, handles, contact details, exact dates, locations, or institutions.
- Copy the supplied severity exactly into "target".

For each input, return:
{{"id": 0, "sentences": {{{sentence_schema}}},
 "target": "exact input severity"{timing_field},
 "style": "first-person postpartum diary"}}

Return a JSON list with exactly one output per input ID. {JSON_ONLY}
INPUT:
{json.dumps(items, ensure_ascii=False)}"""


def prompt_stage3_length_repair(
    item: dict[str, Any], min_words: int, max_words: int, use_timing: bool
) -> str:
    target_low = min(max_words, max(min_words, 36))
    target_high = min(max_words, max(target_low, 60))
    sentence_low = max(
        7, (target_low + STAGE3_SENTENCE_COUNT - 1) // STAGE3_SENTENCE_COUNT
    )
    sentence_high = max(sentence_low, target_high // STAGE3_SENTENCE_COUNT)
    sentence_schema = ", ".join(
        f'"s{index:02d}": "{sentence_low}-{sentence_high} words"'
        for index in range(1, STAGE3_SENTENCE_COUNT + 1)
    )
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
        "Minimal": "show transient or low distress without inventing functional claims",
        "Mild": "show noticeable distress without inventing persistence or functional impact",
        "Moderate": (
            "show persistent distress; mention functional impairment only when it is explicit "
            "in the supplied factors"
        ),
        "Severe": (
            "show pervasive high-intensity distress plus major impairment, inability to manage "
            "basic care, serious bonding disruption, or urgent risk"
        ),
    }
    severity_requirement = severity_requirements.get(target, "match the supplied target")
    return f"""Rewrite and expand one under-length synthetic postpartum diary draft.
The draft and factors are synthetic/generalized; no raw source post is provided.

Requirements:
- Fill exactly {STAGE3_SENTENCE_COUNT} ordered fields in the "sentences" object.
- Each field must be a complete sentence containing {sentence_low}-{sentence_high}
  words, for {target_low}-{target_high} words total.
- Do not omit fields or return a short summary.
- Preserve the supplied target and expected timing exactly.
- For severity: {severity_requirement}.
- For timing: {timing_requirement}.
- Preserve supported meaning, but rewrite and expand the composition.
- Ground every sentence in a supplied factor. Revisit supported internal experience
  when facts are sparse instead of adding symptoms, impairment, relationships, or events.
- Use only supplied factors. Do not invent a partner, infant sex, occupation,
  feeding method, medical event, treatment, or support person unless supplied.
- Keep the narrative introspective; do not invent conversations or concrete events.
- Do not add names, handles, contact details, exact dates, locations, or institutions.

Return one JSON object:
{{"id": {item_id}, "sentences": {{{sentence_schema}}},
 "target": {json.dumps(target)}{timing_field},
 "style": "first-person postpartum diary"}}

{JSON_ONLY}
INPUT:
{json.dumps(item, ensure_ascii=False)}"""


def prompt_stage3_content_repair(
    item: dict[str, Any], categories: Sequence[str], min_words: int, max_words: int,
    use_timing: bool,
) -> str:
    target_low = min(max_words, max(min_words, 36))
    target_high = min(max_words, max(target_low, 60))
    sentence_low = max(
        7, (target_low + STAGE3_SENTENCE_COUNT - 1) // STAGE3_SENTENCE_COUNT
    )
    sentence_high = max(sentence_low, target_high // STAGE3_SENTENCE_COUNT)
    sentence_schema = ", ".join(
        f'"s{index:02d}": "{sentence_low}-{sentence_high} words"'
        for index in range(1, STAGE3_SENTENCE_COUNT + 1)
    )
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
        "functional impact": (
            "Remove unsupported chores, daily-task, work, or functioning claims. "
            "Describe only the supported internal experience."
        ),
        "social or assistance detail": (
            "Remove unsupported help-seeking, isolation, friendship, or social-connection claims."
        ),
        "anxiety or worry": "Remove anxiety, worry, or panic unless explicit in the factors.",
        "guilt or self-worth": (
            "Remove guilt, failure, worthlessness, or inadequacy unless explicit in the factors."
        ),
        "sleep disturbance": "Remove sleep problems unless explicit in the factors.",
        "fatigue or exhaustion": "Remove fatigue, tiredness, or exhaustion unless explicit in the factors.",
        "appetite detail": "Remove appetite or eating changes unless explicit in the factors.",
        "concentration detail": "Remove concentration, focus, or memory problems unless explicit in the factors.",
        "irritability or anger": "Remove irritability or anger unless explicit in the factors.",
        "overwhelm detail": "Remove feeling overwhelmed unless explicit in the factors.",
        "loss of interest": "Remove loss-of-interest claims unless explicit in the factors.",
        "physical discomfort": "Remove specific pain or physical symptoms unless explicit in the factors.",
        "extra timing detail": (
            "Keep only the required bucket timing phrase. Remove every other duration or numeric time cue."
        ),
    }
    rules = "\n".join(
        f"- {category_rules.get(category, f'Remove unsupported {category}.')}"
        for category in categories
    )
    return f"""Rewrite one synthetic postpartum diary draft to remove unsupported details.
The supplied factors are the complete factual boundary. Do not add replacement facts.

Required corrections:
{rules}

Additional requirements:
- Fill exactly {STAGE3_SENTENCE_COUNT} ordered fields in the "sentences" object.
- Each field must be a complete sentence containing {sentence_low}-{sentence_high}
  words, producing {target_low}-{target_high} words total.
- Do not omit fields or return a short summary.
- Preserve target {json.dumps(target)} and timing {json.dumps(expected_timing)} exactly.
- Preserve supported symptom intensity, persistence, impairment, and timing.
- Ground every sentence in a supplied factor; do not replace removed facts with new ones.
- Keep the narrative first-person and natural, but prefer introspection over invented events.
- Do not add identifiers, exact dates, locations, institutions, or source-like wording.

Return one JSON object:
{{"id": {item_id}, "sentences": {{{sentence_schema}}},
 "target": {json.dumps(target)}{timing_field},
 "style": "first-person postpartum diary"}}

{JSON_ONLY}
INPUT:
{json.dumps(item, ensure_ascii=False)}"""


def prompt_stage3_grounding_audit(items: Sequence[dict[str, Any]]) -> str:
    return f"""Act as a factual-grounding auditor for synthetic postpartum narratives.
Decide whether the narrative stays inside the facts it was given. Audit every sentence.

THE FACTUAL BOUNDARY is the entire "factors" object. That includes deidentified_summary
together with every structured field. The summary is a supplied fact with exactly the
same standing as the lists. A claim drawn from the summary is supported.

The narrative is SUPPOSED to reword the facts. It is written in the first person and it
will not match the field wording. Do not treat rewording as an addition. The following
are all supported and must NOT be flagged:
- a paraphrase or first-person version of the summary or any field
- a plainer or more natural phrasing of the same fact
- combining two supplied facts in one sentence
- restating a supplied intensity, persistence or timing value in ordinary words
- reflective phrasing that carries no new fact, such as "this has been hard for me"

Flag a claim ONLY when it introduces information that is genuinely absent from the whole
factors object: a new symptom, a new functional effect, a claim that something is absent
or fine, a relationship or other person, a social action, a coping action, a cause, a
physical condition, a concrete event, or a duration more precise than the supplied timing.
The severity target alone never licenses an extra symptom or impairment. A timing bucket
supports only its required broad timing phrase.

Worked examples. Suppose the summary says the person is experiencing depression they did
not initially recognize, and that their emotional state changed since giving birth, with
persistence "ongoing" and coping "needs time to adjust".
- "My emotional state has changed significantly since giving birth." SUPPORTED, it
  restates the summary.
- "I did not realise at first that I was depressed." SUPPORTED, it paraphrases the summary.
- "The depression is still with me." SUPPORTED, it restates persistence.
- "I am giving myself time to adjust." SUPPORTED, it restates coping.
- "My husband helps me in the evenings." NOT SUPPORTED, no partner in the factors.
- "I have barely slept in weeks." NOT SUPPORTED, no sleep information in the factors.
- "I cannot manage to feed the baby." NOT SUPPORTED, no functional impact supplied.

EMPTY FIELDS ARE SILENCE, NOT PERMISSION. An empty list or empty string means the source
said nothing on that topic, so any claim about that topic is unsupported unless the
summary states it. Apply this directly:
- sleep_context empty: any mention of sleep, rest, tiredness or nights is unsupported
- functional_impact empty: any claim about what they can or cannot manage is unsupported
- perceived_support empty: any mention of help, family, partner, or being alone is unsupported
- feeding_or_infant_care_stressors empty: any feeding or infant-care difficulty is unsupported
- bonding_indicators empty: any claim about closeness or distance from the baby is unsupported
- risk_indicators empty: any mention of self-harm or harm to the baby is unsupported
This holds no matter how plausible the claim is for someone with the supplied symptoms.

Before flagging a claim, search the whole factors object, including the summary, for
anything it could be restating. If you find a match, it is supported. Only flag what you
could not match anywhere.

Return one object per input ID:
{{"id": 0, "grounded": true, "unsupported_claims": [],
 "reason": "brief sentence-level assessment"}}

unsupported_claims must contain ONLY claims you are actually rejecting. Never list a
claim there that you judged supported. If a claim is supported, leave it out entirely.
Set grounded to false if and only if unsupported_claims is non-empty. For each entry,
name the new information it introduces in "reason". Return a JSON list with one object
per input ID. {JSON_ONLY}
INPUT:
{json.dumps(items, ensure_ascii=False)}"""


def prompt_stage3_semantic_repair(
    item: dict[str, Any], min_words: int, max_words: int, use_timing: bool
) -> str:
    target_low = min(max_words, max(min_words, 36))
    target_high = min(max_words, max(target_low, 60))
    sentence_low = max(
        7, (target_low + STAGE3_SENTENCE_COUNT - 1) // STAGE3_SENTENCE_COUNT
    )
    sentence_high = max(sentence_low, target_high // STAGE3_SENTENCE_COUNT)
    sentence_schema = ", ".join(
        f'"s{index:02d}": "{sentence_low}-{sentence_high} words"'
        for index in range(1, STAGE3_SENTENCE_COUNT + 1)
    )
    item_id = int(item["id"])
    target = str(item["target"])
    expected_timing = str(item.get("expected_timing", "unknown"))
    timing_field = f', "timing": {json.dumps(expected_timing)}' if use_timing else ""
    return f"""Rewrite one synthetic postpartum narrative after a strict grounding audit.
The factors are the entire factual boundary. Remove every unsupported claim listed by
the auditor. Do not replace it with another fact.

Requirements:
- Fill exactly {STAGE3_SENTENCE_COUNT} ordered sentence fields, each with
  {sentence_low}-{sentence_high} words, for {target_low}-{target_high} words total.
- Every sentence must be directly supported by at least one supplied factor.
- When factors are sparse, revisit the supported internal experience in plain language.
- Do not add symptoms, impairment, normal functioning, relationships, actions, causes,
  physical conditions, events, or duration.
- Preserve the target and required broad timing phrase exactly.
- Keep only the required timing phrase; add no second time or duration cue.

Return one JSON object:
{{"id": {item_id}, "sentences": {{{sentence_schema}}},
 "target": {json.dumps(target)}{timing_field},
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
    for key in (
        "symptoms", "functional_impact", "feeding_or_infant_care_stressors",
        "bonding_indicators", "risk_indicators",
    ):
        if not isinstance(item.get(key), list):
            return False, f"{key} must be a list"
        item[key] = [value for value in item[key] if not missing_list_value(value)]
    for key in (
        "symptom_persistence", "sleep_context", "perceived_support",
        "coping_or_adjustment_context",
    ):
        if not isinstance(item.get(key), str):
            return False, f"{key} must be a string"
    # 2026-08-23: small models occasionally drop plainly stated persistence or coping cues.
    preserve_explicit_source_factors(item, source_text)
    generalize_precise_ages(item)
    item["symptom_persistence"] = source_persistence(source_text)
    source_lower = (source_text or "").lower()
    self_harm_pattern = re.compile(
        r"\b(?:take my life|end my life|kill myself|suicid\w*|self[- ]?harm|"
        r"want(?:ed)? to die)\b",
        re.I,
    )
    infant_harm_pattern = re.compile(
        r"\b(?:take|end) (?:my )?(?:baby|infant|child)(?:'s)? life\b|"
        r"\bharm(?:ing)? (?:my |the )?(?:baby|infant|child)\b|"
        r"\b(?:take|end|contemplat\w*).{0,80}\b(?:baby|infant|child)(?:'s)? life\b",
        re.I,
    )
    # Risk fields must come from affirmative source wording, not model inference.
    item["risk_indicators"] = []
    if source_has_affirmed_risk(source_lower, self_harm_pattern):
        item["risk_indicators"].append("thoughts of self-harm")
    if source_has_affirmed_risk(source_lower, infant_harm_pattern):
        item["risk_indicators"].append("thoughts of harming my baby")
    if re.search(r"\breached out to (?:a )?(?:psychologist|therapist|counselor)\b", source_lower):
        item["coping_or_adjustment_context"] = "reached out to a mental-health professional"
    negative_bond = re.search(
        r"\b(?:difficulty bonding|cannot bond|can't bond|detached|disconnected|no bond)\b",
        source_lower,
    )
    positive_bond = re.search(
        r"\b(?:bond\w*|connect\w*|attach\w*|love\w*)\b.{0,30}\b(?:baby|infant|child)\b",
        source_lower,
    )
    if negative_bond:
        item["bonding_indicators"] = ["difficulty bonding"]
    elif positive_bond:
        item["bonding_indicators"] = ["positive bonding experience"]
    else:
        item["bonding_indicators"] = []
    if re.search(r"\b(?:cut|shut)\b.{0,35}\bfriend\w*\b|\bisolat\w*\b|\bno one\b", source_text, re.I):
        item["perceived_support"] = "limited"
    elif not re.search(
        r"\b(?:support\w*|help(?:ed|ing)? (?:me|us)|supportive|there for me|"
        r"alone|no one|partner|spouse|family)\b",
        source_text,
        re.I,
    ):
        item["perceived_support"] = ""
    if item.get("symptom_intensity") not in INTENSITIES:
        return False, f"invalid symptom_intensity: {item.get('symptom_intensity')!r}"
    if use_timing:
        item["postpartum_timing"] = source_timing_bucket(source_text)
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
    moderate_persistence = persistent and intensity in {"moderate", "high"}
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
    if severity == "Mild" and moderate_persistence:
        return False, "Mild conflicts with persistent moderate/high symptoms"
    if severity == "Moderate" and not (
        moderate_persistence or impacts or negative_bonding or urgent_or_hopeless
    ):
        return False, "Moderate lacks moderate/high persistence, impact, risk, or bonding difficulty"
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
    moderate_persistence = persistent and intensity in {"moderate", "high"}
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
    if moderate_persistence:
        disallowed.append("Mild")
    if not (moderate_persistence or impacts or negative_bonding or urgent_or_hopeless):
        disallowed.append("Moderate")
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
    text = materialize_stage3_text(item)
    if not text.strip():
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


def validate_grounding_audit(item: Any, *, expected_id: int) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not an object"
    if item.get("id") != expected_id:
        return False, f"id mismatch: {item.get('id')!r}"
    grounded = item.get("grounded")
    if isinstance(grounded, str) and grounded.strip().lower() in {
        "true", "yes", "grounded", "supported", "false", "no", "ungrounded",
        "unsupported", "partially grounded",
    }:
        grounded = grounded.strip().lower() in {"true", "yes", "grounded", "supported"}
        item["grounded"] = grounded
    elif isinstance(grounded, int) and grounded in {0, 1}:
        grounded = bool(grounded)
        item["grounded"] = grounded
    if not isinstance(grounded, bool):
        return False, "grounded must be boolean"
    claims = item.get("unsupported_claims")
    if isinstance(claims, str):
        claims = [claims]
    elif isinstance(claims, dict):
        claims = [claims]
    if not isinstance(claims, list):
        return False, "unsupported_claims must be a list"
    normalized_claims = []
    for value in claims:
        if isinstance(value, str):
            claim = value.strip()
        elif isinstance(value, dict):
            claim = str(
                value.get("claim") or value.get("text") or value.get("span")
                or text_values(value)
            ).strip()
        else:
            claim = str(value).strip()
        if claim:
            normalized_claims.append(claim)
    item["unsupported_claims"] = normalized_claims
    if grounded and normalized_claims:
        grounded = False
        item["grounded"] = False
    reason = item.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "; ".join(normalized_claims) or "No unsupported claim was identified."
        item["reason"] = reason
    if not grounded and not normalized_claims:
        item["unsupported_claims"] = [reason.strip()]
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
        "risk_indicators",
        "coping_or_adjustment_context",
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
    payload["disallowed_content_categories"] = disallowed_content_categories(factors)
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

    repair_cfg = {**cfg, "schema_retries": max(2, cfg["schema_retries"])}
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

    content_cfg = {**cfg, "schema_retries": max(2, cfg["schema_retries"])}
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

    for item_id, result in generated.items():
        if result.status == "ok" and result.value is not None:
            continue
        factors = stage1_results[item_id].value or {}
        severity = stage2_results[item_id].value or {}
        fallback = conservative_factor_narrative(
            factors,
            str(severity.get("severity", "")),
            cfg["use_timing"],
            cfg["min_words"],
            cfg["max_words"],
            item_id,
        )
        fallback["id"] = item_id
        fallback_ok, fallback_error = validator(fallback, item_id)
        fallback_categories = unsupported_detail_categories(
            str(fallback.get("synthetic_text", "")), factors
        )
        if fallback_ok and not fallback_categories:
            fallback["grounding_passed"] = True
            fallback["grounding_attempts"] = 0
            fallback["grounding_history"] = [
                {
                    "round": "fallback_after_stage3",
                    "grounded": True,
                    "unsupported_claims": [],
                    "reason": "Composed directly from validated factor fields.",
                }
            ]
            result.value = fallback
            result.last_candidate = fallback
            result.status = "ok"
        else:
            result.errors.append(
                "factor fallback failed: "
                + (fallback_error or ",".join(fallback_categories))
            )

    # A semantic pass catches unsupported claims that keyword rules cannot enumerate.
    grounding_history: dict[int, list[dict[str, Any]]] = {
        item_id: [] for item_id in generated
    }
    grounding_attempts: dict[int, int] = {item_id: 0 for item_id in generated}
    active_ids = [
        item_id
        for item_id, result in generated.items()
        if result.status == "ok" and result.value is not None
        and result.value.get("grounding_method") != "deterministic_factor_fallback"
    ]
    audit_cfg = {**cfg, "schema_retries": max(2, cfg["schema_retries"])}

    for audit_round in range(2):
        if not active_ids:
            break

        def grounding_prompt(group_ids: Sequence[int], _: int) -> str:
            payloads = []
            for item_id in group_ids:
                factors = stage1_results[item_id].value or {}
                severity = stage2_results[item_id].value or {}
                payloads.append(
                    {
                        "id": item_id,
                        "factors": generation_payload(
                            item_id, factors, severity, cfg["use_timing"]
                        ),
                        "required_timing": factors.get("postpartum_timing", "unknown"),
                        "narrative": (generated[item_id].value or {}).get(
                            "synthetic_text", ""
                        ),
                    }
                )
            return prompt_stage3_grounding_audit(payloads)

        audits = run_group_stage(
            ids=active_ids,
            make_prompt=grounding_prompt,
            validator=lambda item, item_id: validate_grounding_audit(
                item, expected_id=item_id
            ),
            cfg=audit_cfg,
            temperature=0.0,
            tokens_per_item=cfg["tokens_s4_per_item"],
            stage_number=50 + regeneration_attempt * 10 + audit_round * 2,
            usage=usage,
        )

        repair_ids: list[int] = []
        repair_claims: dict[int, list[str]] = {}
        for item_id in active_ids:
            audit = audits[item_id]
            grounding_attempts[item_id] += audit.attempts
            result = generated[item_id]
            if audit.status != "ok" or audit.value is None:
                result.errors.append("semantic grounding audit failed: " + last_error(audit))
                factors = stage1_results[item_id].value or {}
                severity = stage2_results[item_id].value or {}
                fallback = conservative_factor_narrative(
                    factors,
                    str(severity.get("severity", "")),
                    cfg["use_timing"],
                    cfg["min_words"],
                    cfg["max_words"],
                    item_id,
                )
                fallback["id"] = item_id
                fallback_ok, fallback_error = validator(fallback, item_id)
                fallback_categories = unsupported_detail_categories(
                    str(fallback.get("synthetic_text", "")), factors
                )
                if fallback_ok and not fallback_categories:
                    grounding_history[item_id].append(
                        {
                            "round": "fallback_after_audit_format",
                            "grounded": True,
                            "unsupported_claims": [],
                            "reason": "Composed directly from validated factor fields.",
                        }
                    )
                    fallback["grounding_passed"] = True
                    fallback["grounding_history"] = grounding_history[item_id]
                    fallback["grounding_attempts"] = grounding_attempts[item_id]
                    result.value = fallback
                    result.last_candidate = fallback
                    result.status = "ok"
                else:
                    fallback["grounding_passed"] = False
                    fallback["grounding_history"] = grounding_history[item_id]
                    fallback["grounding_attempts"] = grounding_attempts[item_id]
                    result.last_candidate = fallback
                    result.value = None
                    result.status = "failed"
                    result.errors.append(
                        "factor fallback failed: "
                        + (fallback_error or ",".join(fallback_categories))
                    )
                continue

            assessment = audit.value
            grounding_history[item_id].append(
                {
                    "round": audit_round,
                    "grounded": assessment["grounded"],
                    "unsupported_claims": assessment["unsupported_claims"],
                    "reason": assessment["reason"],
                }
            )
            if assessment["grounded"]:
                result.value["grounding_passed"] = True
                result.value["grounding_history"] = grounding_history[item_id]
                result.value["grounding_attempts"] = grounding_attempts[item_id]
                continue

            if audit_round == 1:
                factors = stage1_results[item_id].value or {}
                severity = stage2_results[item_id].value or {}
                fallback = conservative_factor_narrative(
                    factors,
                    str(severity.get("severity", "")),
                    cfg["use_timing"],
                    cfg["min_words"],
                    cfg["max_words"],
                    item_id,
                )
                fallback["id"] = item_id
                fallback_ok, fallback_error = validator(fallback, item_id)
                fallback_categories = unsupported_detail_categories(
                    str(fallback.get("synthetic_text", "")), factors
                )
                if fallback_ok and not fallback_categories:
                    grounding_history[item_id].append(
                        {
                            "round": "fallback",
                            "grounded": True,
                            "unsupported_claims": [],
                            "reason": "Composed directly from validated factor fields.",
                        }
                    )
                    fallback["grounding_passed"] = True
                    fallback["grounding_history"] = grounding_history[item_id]
                    fallback["grounding_attempts"] = grounding_attempts[item_id]
                    result.value = fallback
                    result.last_candidate = fallback
                    result.status = "ok"
                    continue

                result.errors.append(
                    "semantic grounding fallback failed: "
                    + (fallback_error or ",".join(fallback_categories))
                )
                fallback["grounding_passed"] = False
                fallback["grounding_history"] = grounding_history[item_id]
                fallback["grounding_attempts"] = grounding_attempts[item_id]
                result.last_candidate = fallback
                result.value = None
                result.status = "failed"
                continue

            repair_ids.append(item_id)
            repair_claims[item_id] = assessment["unsupported_claims"]

        next_active: list[int] = []
        for item_id in repair_ids:
            factors = stage1_results[item_id].value or {}
            severity = stage2_results[item_id].value or {}

            def semantic_prompt(_: Sequence[int], __: int) -> str:
                payload = {
                    "id": item_id,
                    "factors": generation_payload(
                        item_id, factors, severity, cfg["use_timing"]
                    ),
                    "target": severity.get("severity"),
                    "expected_timing": factors.get("postpartum_timing", "unknown"),
                    "draft_to_rewrite": (generated[item_id].value or {}).get(
                        "synthetic_text", ""
                    ),
                    "unsupported_claims": repair_claims[item_id],
                }
                return prompt_stage3_semantic_repair(
                    payload, cfg["min_words"], cfg["max_words"], cfg["use_timing"]
                )

            prior = generated[item_id]
            repaired = run_group_stage(
                ids=[item_id],
                make_prompt=semantic_prompt,
                validator=content_validator,
                cfg=audit_cfg,
                temperature=min(cfg["temp_s3"], 0.6),
                tokens_per_item=cfg["tokens_s3_per_item"],
                stage_number=51 + regeneration_attempt * 10 + audit_round * 2,
                usage=usage,
            )[item_id]
            repaired.attempts += prior.attempts
            repaired.errors = prior.errors + repaired.errors
            generated[item_id] = repaired
            if repaired.status == "ok" and repaired.value is not None:
                next_active.append(item_id)
            else:
                fallback = conservative_factor_narrative(
                    factors,
                    str(severity.get("severity", "")),
                    cfg["use_timing"],
                    cfg["min_words"],
                    cfg["max_words"],
                    item_id,
                )
                fallback["id"] = item_id
                fallback_ok, fallback_error = validator(fallback, item_id)
                fallback_categories = unsupported_detail_categories(
                    str(fallback.get("synthetic_text", "")), factors
                )
                if fallback_ok and not fallback_categories:
                    grounding_history[item_id].append(
                        {
                            "round": "fallback_after_repair",
                            "grounded": True,
                            "unsupported_claims": [],
                            "reason": "Composed directly from validated factor fields.",
                        }
                    )
                    fallback["grounding_passed"] = True
                    fallback["grounding_history"] = grounding_history[item_id]
                    fallback["grounding_attempts"] = grounding_attempts[item_id]
                    repaired.value = fallback
                    repaired.last_candidate = fallback
                    repaired.status = "ok"
                else:
                    repaired.errors.append(
                        "semantic grounding fallback failed: "
                        + (fallback_error or ",".join(fallback_categories))
                    )

        active_ids = next_active

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
            parsed["raw_predicted_severity"] = parsed["predicted_severity"]
            parsed["predicted_severity"] = calibrate_blind_severity(
                narrative, str(parsed["predicted_severity"])
            )
            if cfg["use_timing"]:
                parsed["raw_predicted_timing"] = parsed["predicted_timing"]
                parsed["predicted_timing"] = blind_timing_bucket(narrative)
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
    for record in records.values():
        if cfg["use_source_screen"]:
            status, reason = source_eligibility(record["source_text"])
        else:
            status, reason = "eligible", "Source screening disabled by configuration."
        record["source_eligibility"] = status
        record["source_eligibility_reason"] = reason
    eligible_records = {
        item_id: record
        for item_id, record in records.items()
        if record["source_eligibility"] == "eligible"
    }
    s1 = {
        item_id: ItemResult(status="not_run", errors=[record["source_eligibility_reason"]])
        for item_id, record in records.items()
    }
    if eligible_records:
        s1.update(stage1(eligible_records, cfg, usage))
    s2 = stage2(s1, cfg, usage)
    for item_id, record in records.items():
        if record["source_eligibility"] != "eligible":
            s2[item_id] = ItemResult(
                status="not_run", errors=[record["source_eligibility_reason"]]
            )
    s3, s4, outcomes = stages3_and4(
        records=records,
        stage1_results=s1,
        stage2_results=s2,
        cfg=cfg,
        usage=usage,
    )
    for item_id, record in records.items():
        if record["source_eligibility"] != "eligible":
            reason = record["source_eligibility_reason"]
            s3[item_id] = ItemResult(status="not_run", errors=[reason])
            s4[item_id] = ItemResult(status="not_run", errors=[reason])
            outcomes[item_id]["row_status"] = record["source_eligibility"]
    elapsed = time.monotonic() - started
    rows: list[dict[str, Any]] = []
    for item_id, record in records.items():
        source = record["source_text"]
        factors = s1[item_id].value or {}
        severity = s2[item_id].value or {}
        generated = s3[item_id].value or s3[item_id].last_candidate or {}
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
                "source_eligibility": record["source_eligibility"],
                "source_eligibility_reason": record["source_eligibility_reason"],
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
                "s1_risk_indicators": factors.get("risk_indicators", []),
                "s1_coping_or_adjustment_context": factors.get(
                    "coping_or_adjustment_context", ""
                ),
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
                "s3_grounding_passed": bool(generated.get("grounding_passed", False)),
                "s3_grounding_attempts": generated.get("grounding_attempts", 0),
                "s3_grounding_history": generated.get("grounding_history", []),
                # a 100% acceptance rate hid a 100% fallback rate in pilot20_v410
                "s3_provenance": stage3_provenance(generated),
                "s3_timing_cue_injected": bool(generated.get("timing_cue_injected", False)),
                "s3_regen_count": outcomes[item_id]["regen_count"],
                "s4_status": s4[item_id].status,
                "s4_attempts": s4[item_id].attempts,
                "s4_error": last_error(s4[item_id]),
                "s4_predicted_severity": predicted,
                "s4_raw_predicted_severity": validation.get("raw_predicted_severity", ""),
                "s4_predicted_timing": predicted_timing,
                "s4_raw_predicted_timing": validation.get("raw_predicted_timing", ""),
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
    eligible_count = sum(row["source_eligibility"] == "eligible" for row in rows)
    excluded_count = len(rows) - eligible_count
    log(
        f"Batch {batch_id}: completed in {elapsed:.1f}s; {accepted}/{eligible_count} eligible accepted; "
        f"{excluded_count} excluded; {usage.calls} LLM call(s)",
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
        "use_source_screen": not args.no_source_screen,
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
    eligible_rows = [row for row in rows if row.get("source_eligibility") == "eligible"]
    excluded_rows = [row for row in rows if row.get("source_eligibility") != "eligible"]
    validated = [row for row in eligible_rows if row.get("s4_status") == "ok"]
    timing_validated = [
        row for row in validated if row.get("s1_postpartum_timing") != "unknown"
    ]
    total = len(rows)
    eligible_total = len(eligible_rows)
    status_counts = Counter(str(row.get("row_status", "unknown")) for row in rows)

    def rate(count: int, denominator: int = eligible_total) -> float | None:
        return round(100.0 * count / denominator, 2) if denominator else None

    summary = {
        "total_rows": total,
        "eligible_rows": eligible_total,
        "excluded_source_rows": len(excluded_rows),
        "accepted_rows": len(accepted_rows),
        "acceptance_rate_pct": rate(len(accepted_rows)),
        "status_counts": dict(status_counts),
        "stage_failure_rates_pct": {
            "stage1": rate(sum(row["s1_status"] != "ok" for row in eligible_rows)),
            "stage2": rate(sum(row["s2_status"] != "ok" for row in eligible_rows)),
            "stage3": rate(sum(row["s3_status"] != "ok" for row in eligible_rows)),
            "stage4": rate(sum(row["s4_status"] == "failed" for row in eligible_rows)),
        },
        "assigned_severity_counts": dict(Counter(row["s2_severity"] for row in eligible_rows if row["s2_severity"])),
        "accepted_severity_counts": dict(Counter(row["s2_severity"] for row in accepted_rows)),
        "timing_counts": dict(Counter(row["s1_postpartum_timing"] for row in eligible_rows if row["s1_postpartum_timing"])),
        "severity_agreement_rate_pct": rate(
            sum(bool(row["severity_agreement"]) for row in validated), len(validated)
        ),
        "timing_agreement_rate_pct": rate(
            sum(bool(row["timing_agreement"]) for row in timing_validated),
            len(timing_validated),
        ) if cfg["use_timing"] else None,
        "timing_agreement_evaluable_rows": len(timing_validated) if cfg["use_timing"] else None,
        "rows_regenerated": sum(int(row["s3_regen_count"]) > 0 for row in eligible_rows),
        "total_regenerations": sum(int(row["s3_regen_count"]) for row in eligible_rows),
        "source_copy_violations_in_saved_rows": sum(
            int(row["s3_shared_source_ngrams"]) > cfg["max_copy_ngrams"] for row in eligible_rows
        ),
        "identifier_violations_in_saved_rows": sum(bool(row["s3_identifier_leaks"]) for row in eligible_rows),
        "rows_flagged_for_unsupported_detail_review": sum(
            bool(row.get("s3_unsupported_detail_flags")) for row in eligible_rows
        ),
        # report generated vs fallback separately; acceptance alone hides this
        "accepted_from_llm_generation": sum(
            row.get("s3_provenance") == "llm" for row in accepted_rows
        ),
        "accepted_from_deterministic_fallback": sum(
            row.get("s3_provenance") == "fallback" for row in accepted_rows
        ),
        "fallback_rate_among_accepted_pct": round(
            100.0
            * sum(row.get("s3_provenance") == "fallback" for row in accepted_rows)
            / max(1, len(accepted_rows)),
            2,
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
    parser.add_argument(
        "--no-source-screen",
        action="store_true",
        help="Ablation: process seeds without the postpartum-eligibility screen",
    )
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
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--max-words", type=int, default=100)
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

    pending_eligible_batches = 0
    pending_eligible_rows = 0
    for _, batch_rows in pending:
        eligible_in_batch = sum(
            args.no_source_screen
            or source_eligibility(str(row.get(args.text_column, "")))[0] == "eligible"
            for row in batch_rows
        )
        if eligible_in_batch:
            pending_eligible_batches += 1
            pending_eligible_rows += eligible_in_batch
    minimum_calls = pending_eligible_batches * 4 + (
        pending_eligible_rows if not args.no_validator else 0
    )
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
        "use_source_screen": not args.no_source_screen,
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

    eligible_total = int(summary["eligible_rows"])
    rejection_rate = 1.0 - (summary["accepted_rows"] / eligible_total) if eligible_total else 0.0
    log("=" * 72, log_file)
    log(
        f"Finished: {summary['accepted_rows']}/{eligible_total} eligible rows accepted "
        f"({summary['acceptance_rate_pct']}%); {summary['excluded_source_rows']}/"
        f"{summary['total_rows']} source rows excluded.",
        log_file,
    )
    log(f"Status counts: {summary['status_counts']}", log_file)
    log(
        f"Accepted text source: {summary['accepted_from_llm_generation']} generated, "
        f"{summary['accepted_from_deterministic_fallback']} deterministic fallback "
        f"({summary['fallback_rate_among_accepted_pct']}% fallback).",
        log_file,
    )
    if summary["fallback_rate_among_accepted_pct"] >= 50.0:
        log(
            "WARNING: most accepted rows are deterministic fallback text, not generated "
            "narratives. Acceptance and agreement rates do not describe generation "
            "quality in this run.",
            log_file,
        )
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
