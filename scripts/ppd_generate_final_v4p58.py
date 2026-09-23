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


SCRIPT_VERSION = "ppd_generate_final v4.58"
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
STAGE3_SENTENCE_COUNT = 3

_RISK_NEGATION_PREFIX = re.compile(
    r"(?:\bnever(?:\s+\w+){0,6}|"
    r"\b(?:do|does|did|would|could|have|has|had|was|were) not(?:\s+\w+){0,6}|"
    r"\b(?:don't|doesn't|didn't|wouldn't|couldn't|haven't|hasn't|hadn't)(?:\s+\w+){0,6}|"
    r"\bno (?:thoughts?|ideas?|plans?|intentions?)(?:\s+\w+){0,4}|"
    r"\bwithout (?:any )?(?:thoughts?|ideas?|plans?|intentions?)(?:\s+\w+){0,4}|"
    r"\bden(?:y|ied|ies|ying)(?:\s+\w+){0,6})\s*$",
    re.I,
)
_SELF_HARM_PATTERN = re.compile(
    r"\b(?:take my life|end my life|kill myself|suicid\w*|self[- ]?harm|"
    r"harm(?:ing|ed)? myself|hurt(?:ing|ed)? myself|want(?:ed)? to die)\b",
    re.I,
)
_THIRD_PERSON_NARRATIVE_PATTERN = re.compile(
    r"\b(?:the individual|the mother|she|they)\b",
    re.I,
)
_INFANT_HARM_PATTERN = re.compile(
    r"\b(?:take|end) (?:my )?(?:baby|infant|child)(?:'s)? life\b|"
    r"\bharm(?:ing)? (?:my |the )?(?:baby|infant|child)\b|"
    r"\b(?:take|end|contemplat\w*).{0,80}\b(?:baby|infant|child)(?:'s)? life\b",
    re.I,
)

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
    r"within (?:the )?first (?:few )?weeks|from the start|early on|"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a few|several) "
    r"(?:days|weeks|months) (?:after|since) (?:birth|delivery|giving birth)|"
    r"just (?:gave birth|delivered))\b",
    re.I,
)
_CANONICAL_TIMING_PREFIXES = {
    TIMING_BUCKETS[0]: "Within the first two weeks after birth,",
    TIMING_BUCKETS[1]: "Between two and six weeks after birth,",
    TIMING_BUCKETS[2]: "Between six and twelve weeks after birth,",
    TIMING_BUCKETS[3]: "More than three months after birth,",
}
_POSTPARTUM_TIMING_CUE = re.compile(
    r"\b(?:after giving birth|(?:during|in) (?:my|the) postpartum period|"
    r"within the first two weeks after birth|between two and six weeks after birth|"
    r"between six and twelve weeks after birth|more than three months after birth)\b",
    re.I,
)


def add_timing_opener(text: str, timing: str) -> str:
    """Integrate the required broad timing cue into the first sentence."""
    opener = _CANONICAL_TIMING_PREFIXES.get(timing)
    if not opener:
        return text.strip()
    value = re.sub(
        r"^(?:after giving birth|during (?:my|the) postpartum period|"
        r"in (?:my|the) postpartum period)\s*,?\s*",
        "",
        text.strip(),
        count=1,
        flags=re.I,
    )
    return f"{opener} {value}"


def source_timing_bucket(source_text: str) -> str:
    """Extract timing only from an explicit infant-age or post-birth expression."""
    number_words = {
        "a": 1.0, "one": 1.0, "two": 2.0, "couple": 2.0,
        "a couple": 2.0, "couple of": 2.0, "a couple of": 2.0,
        "three": 3.0, "few": 3.0, "a few": 3.0,
        "four": 4.0, "several": 4.0, "five": 5.0, "six": 6.0,
        "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
        "eleven": 11.0, "twelve": 12.0,
    }
    number = (
        r"(?:\d+(?:\.\d+)?|a couple of|couple of|a couple|a few|one|two|couple|three|few|"
        r"four|several|five|six|seven|eight|nine|ten|eleven|twelve|a)"
    )
    unit = r"(?:days?|d|weeks?|wks?|months?|mos?|years?|yrs?)"
    patterns = (
        re.compile(
            rf"\b(?:baby|infant|child)(?:\s+is|'s)?\s+(?P<n>{number})\s*"
            rf"(?P<u>{unit})(?:\s*[- ]?old)?\b",
            re.I,
        ),
        re.compile(
            rf"\b(?:i (?:have|had) (?:a |my )?|my )?(?P<n>{number})\s*[- ]?"
            rf"(?P<u>{unit})\s*[- ]old\s+(?:baby|infant|child)\b",
            re.I,
        ),
        re.compile(
            rf"\b(?P<n>{number})\s*(?P<u>{unit})\s*"
            r"(?:postpartum\b|(?:after|since)\s*(?:birth|delivery|giving birth|having (?:a|my) baby)\b)",
            re.I,
        ),
        re.compile(
            rf"\b(?:gave birth|delivered|had (?:a|my(?: (?:\d+(?:st|nd|rd|th)|first|second|third|fourth))?) baby)\s+"
            rf"(?:almost|about|around)?\s*(?P<n>{number})\s*"
            rf"(?P<u>{unit})\s+ago\b",
            re.I,
        ),
    )
    later_match = re.search(
        rf"\b(?P<n>{number})\s*(?P<u>{unit})\s+later\b",
        source_text or "",
        re.I,
    )
    has_birth_context = bool(re.search(
        r"\b(?:baby|birth|gave birth|delivered|postpartum|ppd|baby blues|pediatrician)\b",
        source_text or "",
        re.I,
    ))
    match = next(
        (pattern.search(source_text or "") for pattern in patterns if pattern.search(source_text or "")),
        None,
    )
    if match is None and later_match and has_birth_context:
        tail = (source_text or "")[later_match.end() : later_match.end() + 140]
        head = (source_text or "")[max(0, later_match.start() - 180) : later_match.start()]
        recovery_timing = bool(
            re.search(
                r"\b(?:now|currently)?\s*(?:i )?(?:feel|feeling) more like myself\b|"
                r"\b(?:recovered|recovery|doing better|symptoms? (?:eased|improved|resolved))\b",
                tail,
                re.I,
            )
            and re.search(r"\b(?:depress\w*|anxi\w*|ppd|baby blues)\b", head, re.I)
        )
        # 2026-09-07: a later recovery date is not symptom-onset timing.
        if not recovery_timing:
            match = later_match
    if match is None and has_birth_context:
        # 2026-09-04: short posts often state timing as a visit or checkup age.
        match = re.search(
            rf"\b(?:at\s+)?(?P<n>{number})\s*(?P<u>{unit})\s*"
            r"(?:check\s*up|for\s+(?:postpartum\s+)?depress\w*|when\s+i\s+sought\s+help)\b",
            source_text or "",
            re.I,
        )
    if match is None:
        return "unknown"
    raw_number = re.sub(r"\s+", " ", match.group("n").lower()).strip()
    value = float(raw_number) if re.fullmatch(r"\d+(?:\.\d+)?", raw_number) else number_words[raw_number]
    unit = match.group("u").lower()
    if unit.startswith("day") or unit == "d":
        weeks = value / 7.0
    elif unit.startswith("month") or unit.startswith("mo"):
        weeks = value * 4.0
    elif unit.startswith("year") or unit.startswith("yr"):
        weeks = value * 52.0
    else:
        weeks = value
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
        r"i had\s+[/_*]*\s*(?:(?:really|very|extremely|severe|bad)\s*[/_*]*\s*){0,2}"
        r"(?:postpartum\s+)?depress\w*|"
        r"(?:saw|visited) (?:a )?(?:therapist|psychologist|counselor)\b.{0,45}"
        r"\b(?:for|about)\b.{0,20}\bdepress\w*|"
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
        if _RISK_NEGATION_PREFIX.search(prefix):
            continue
        return True
    return False


def source_has_denied_risk(source_text: str, pattern: re.Pattern[str]) -> bool:
    """Retain an explicit safety denial without converting it into risk."""
    source = re.sub(r"\s+", " ", source_text or "").lower()
    for match in pattern.finditer(source):
        prefix = source[max(0, match.start() - 80):match.start()]
        if _RISK_NEGATION_PREFIX.search(prefix):
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
    if not intensity:
        natural_intensity = re.search(
            r"\b(mild|moderate|severe|intense)\s+"
            r"(?:depress\w*|anxi\w*|symptoms?|distress|feelings?)\b",
            value,
            re.I,
        )
        if natural_intensity:
            intensity = {
                "mild": "low", "moderate": "moderate", "severe": "high", "intense": "high",
            }[natural_intensity.group(1).lower()]
    persistent = bool(re.search(
        r"\b(?:persistent|persisted|ongoing|constant|continuous|sustained|daily|"
        r"every day|keeps? returning|does not go away|doesn't go away|"
        r"continue(?:s|d)? to|has continued|have continued|i have been|i've been|"
        r"all along|extended period|prolonged|lasted for (?:an )?extended period|"
        r"still (?:deal\w*|work\w*|feel\w*|experience\w*))\b",
        value,
        re.I,
    ))
    value_lower = value.lower()
    persistent = persistent or any(
        phrase in value_lower
        for phrase in ("each day", "every day", "still struggling", "continue to", "continues to")
    )
    natural_distress = bool(re.search(
        r"\b(?:depress\w*|distress\w*|struggl\w*|difficult|significant(?:ly)?|"
        r"hard to cope|battle\w*|emotionally heavy)\b",
        value,
        re.I,
    ))
    natural_distress = natural_distress or any(
        phrase in value_lower
        for phrase in ("struggling", "hard to cope", "battle", "significantly")
    )
    impairment = bool(re.search(
        r"\b(?:impair\w*|unable to function|cannot function|can't function|"
        r"hard to (?:complete|manage|handle|do) (?:basic |daily |everyday )?(?:tasks|activities)|"
        r"struggl\w* to (?:complete|manage|handle) (?:basic |daily |everyday )?(?:tasks|activities)|"
        r"disrupt\w* (?:my )?(?:daily life|functioning)|affect\w* (?:my )?daily life)\b",
        value,
        re.I,
    ))
    bonding_disruption = bool(re.search(
        r"\b(?:unable to (?:bond|connect)|cannot (?:bond|connect)|can't (?:bond|connect)|"
        r"bonding feels|feel(?:ing)? detached|feel(?:ing)? disconnected)\b",
        value,
        re.I,
    ))
    affirmed_risk = (
        source_has_affirmed_risk(value, _SELF_HARM_PATTERN)
        or source_has_affirmed_risk(value, _INFANT_HARM_PATTERN)
    )
    passive_self_negation = bool(re.search(
        r"\b(?:baby|infant|child|family|they|she|he) (?:will|would|might|may) "
        r"be better off without me\b",
        value,
        re.I,
    ))
    bonding_or_risk = bonding_disruption or affirmed_risk or passive_self_negation
    resolved = bool(re.search(r"\b(?:resolved|recovered|improved|no longer present)\b", value, re.I))
    moderate_gate = intensity == "high" or (
        persistent and (
            intensity in {"moderate", "high"} or (not intensity and natural_distress)
        )
    ) or impairment or bonding_or_risk
    severe_gate = affirmed_risk or passive_self_negation or (
        intensity == "high" and (impairment or bonding_disruption)
    )
    # 2026-09-04: explicit high intensity plus risk satisfies the Severe boundary.
    if severe_gate:
        return "Severe"
    if moderate_gate:
        return "Moderate"
    if predicted in {"Moderate", "Severe"}:
        return "Mild"
    if predicted == "Minimal" and resolved and intensity in {"moderate", "high"}:
        return "Mild"
    return predicted


def severity_from_blind_evidence(profile: dict[str, Any]) -> str:
    """Map target-blind narrative evidence through the predeclared rubric."""
    intensity = str(profile.get("symptom_intensity", "unknown")).lower()
    mood = bool(profile.get("mood_symptoms_present", False))
    persistent = bool(profile.get("persistent_or_extended", False))
    meaningful_impact = bool(profile.get("meaningful_impairment", False))
    major_impact = bool(profile.get("major_impairment_or_inability_basic_care", False))
    bonding = bool(profile.get("serious_bonding_disruption", False))
    affirmed_harm = bool(profile.get("affirmed_self_or_infant_harm", False))
    passive_or_hopeless = bool(
        profile.get("passive_self_negation_or_pervasive_hopelessness", False)
    )

    if affirmed_harm or passive_or_hopeless or (
        intensity == "high" and (major_impact or bonding)
    ):
        return "Severe"
    if (
        intensity == "high"
        or (persistent and (mood or intensity in {"moderate", "high"}))
        or meaningful_impact
        or major_impact
        or bonding
    ):
        return "Moderate"
    if mood or intensity in {"low", "moderate"}:
        return "Mild"
    return "Minimal"


def blind_narrative_evidence(text: str) -> dict[str, Any]:
    """Extract explicit validation evidence from the narrative alone."""
    value = text or ""
    affirmed_harm = (
        source_has_affirmed_risk(value, _SELF_HARM_PATTERN)
        or source_has_affirmed_risk(value, _INFANT_HARM_PATTERN)
    )
    passive_or_hopeless = bool(re.search(
        r"\b(?:pervasive hopelessness|hopeless\w*|"
        r"(?:baby|infant|child|family|they|she|he) (?:will|would|might|may) "
        r"be better off without me)\b",
        value,
        re.I,
    ))
    major_impact = bool(re.search(
        r"\b(?:unable to function|cannot function|can't function|"
        r"unable to (?:manage|perform) basic (?:care|tasks)|"
        r"cannot (?:manage|perform) basic (?:care|tasks)|"
        r"can't (?:manage|perform) basic (?:care|tasks))\b",
        value,
        re.I,
    ))
    bonding = bool(re.search(
        r"\b(?:unable to (?:bond|connect)|cannot (?:bond|connect)|"
        r"can't (?:bond|connect)|serious bonding (?:difficulty|disruption)|"
        r"feel(?:ing)? detached|feel(?:ing)? disconnected)\b",
        value,
        re.I,
    ))
    persistent = bool(re.search(
        r"\b(?:persistent|persisted|ongoing|constant|continuous|sustained|daily|"
        r"every day|continue(?:s|d)? to|has continued|have continued|"
        r"still (?:deal\w*|work\w*|feel\w*|experience\w*|struggl\w*)|"
        r"extended period|prolonged|lasted for (?:an )?extended period)\b",
        value,
        re.I,
    ))
    meaningful_impact = major_impact or bool(re.search(
        r"\b(?:withdr\w* from friends|cut (?:most|the majority) of my friends off|"
        r"daily (?:life|activities|tasks)|everyday (?:life|activities|tasks)|"
        r"work responsibilities|difficulty (?:functioning|managing|coping)|"
        r"hard to (?:function|manage|cope))\b",
        value,
        re.I,
    ))
    mood = bool(re.search(
        r"\b(?:depress\w*|anxi\w*|panic\w*|intrusive thoughts?|hopeless\w*|"
        r"worthless\w*|guilt\w*|baby blues|mood distress)\b",
        value,
        re.I,
    ))
    if affirmed_harm or passive_or_hopeless:
        intensity = "high"
    elif re.search(
        r"\b(?:severe|intense|extreme|overwhelming|debilitating|crippling|deep)\s+"
        r"(?:depress\w*|anxi\w*|symptoms?|distress|feelings?)\b",
        value,
        re.I,
    ):
        intensity = "high"
    elif re.search(
        r"\bmoderate\s+(?:depress\w*|anxi\w*|symptoms?|distress)\b|"
        r"\b(?:coping|experience|symptoms?|feelings?)\b.{0,25}\b"
        r"(?:difficult|significant|distressing|hard)\b|"
        r"\b(?:difficult|significant|distressing|hard)\b.{0,25}\b"
        r"(?:cope|experience|symptoms?|feelings?)\b",
        value,
        re.I,
    ):
        intensity = "moderate"
    elif re.search(
        r"\bmild\s+(?:depress\w*|anxi\w*|symptoms?|distress)\b",
        value,
        re.I,
    ):
        intensity = "low"
    else:
        intensity = "unknown"
    return {
        "mood_symptoms_present": mood,
        "symptom_intensity": intensity,
        "persistent_or_extended": persistent,
        "meaningful_impairment": meaningful_impact,
        "major_impairment_or_inability_basic_care": major_impact,
        "serious_bonding_disruption": bonding,
        "affirmed_self_or_infant_harm": affirmed_harm,
        "passive_self_negation_or_pervasive_hopelessness": passive_or_hopeless,
    }


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


def stage3_sentence_values(item: dict[str, Any]) -> list[str]:
    """Normalize harmless Stage-3 sentence layouts from small local models."""
    sentences = item.get("sentences")
    sentence_values: list[Any] = []
    if isinstance(sentences, dict):
        sentence_values = [sentences[key] for key in sorted(sentences)]
    elif isinstance(sentences, list):
        sentence_values = sentences
    elif isinstance(item.get("synthetic_text"), str):
        sentence_values = [item["synthetic_text"]]

    cleaned_sentences: list[str] = []
    for value in sentence_values:
        normalized = re.sub(r"\s+", " ", str(value or "")).strip()
        if not normalized:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[\"']?[A-ZI])", normalized)
        cleaned_sentences.extend(part.strip() for part in parts if part.strip())
    if cleaned_sentences:
        item["sentences"] = cleaned_sentences
    return cleaned_sentences


def normalize_stage3_language(
    item: dict[str, Any], factors: dict[str, Any] | None = None
) -> None:
    """Clean a few harmless local-model phrasing errors before validation."""
    sentences = stage3_sentence_values(item)
    supported_context = {
        str(value).strip().lower()
        for value in (factors or {}).get("supported_context", [])
        if str(value).strip()
    }
    replacements = (
        (r"\bI felt supported by strong support from\b", "I felt strongly supported by"),
        (r"\bI feel supported by strong support from\b", "I feel strongly supported by"),
        (r"\bI am supported by strong support from\b", "I am strongly supported by"),
        (r"\bAt the moment, I had\b", "At that time, I had"),
        (r"\bat after giving birth\b", "after giving birth"),
        (r"^Within my postpartum experience,?\s*", "After giving birth, "),
        (r",?\s*but now I see its impact\b", ""),
        (r"\s+and find a way forward\b", ""),
        (r",?\s+and (?:it|this) (?:has )?weigh(?:s|ed) heavily on me\b", ""),
        (
            r"^(?:Emotional and physical|Physical and emotional) changes "
            r"have been ongoing since giving birth",
            "I noticed emotional and physical changes after giving birth",
        ),
    )
    cleaned: list[str] = []
    changed = False
    for sentence in sentences:
        value = sentence
        for pattern, replacement in replacements:
            value = re.sub(pattern, replacement, value, flags=re.I)
        if "was on maternity leave" in supported_context:
            value = re.sub(
                r"^My coping mechanisms? (?:have|had) not improved since "
                r"(?:taking|starting|being on) maternity leave[.!?]?$",
                "I was on maternity leave during this period.",
                value,
                flags=re.I,
            )
        value = re.sub(
            r",?\s+which affect(?:s|ed) me deeply\b",
            "",
            value,
            flags=re.I,
        )
        if value and not re.search(r"[.!?][\"']?$", value):
            value += "."
        changed = changed or value != sentence
        cleaned.append(value)
    if changed:
        item["sentences"] = cleaned
        item["language_normalized"] = True


def normalize_stage3_timing(item: dict[str, Any], expected_timing: str) -> None:
    """Repair metadata and keep one source-controlled birth-timing cue."""
    normalized = normalize_timing(item.get("timing"))
    if normalized != expected_timing:
        item["raw_timing_metadata"] = item.get("timing", "")
        item["timing"] = expected_timing
        item["timing_metadata_normalized"] = True
    sentences = stage3_sentence_values(item)
    changed = False
    cleaned: list[str] = []
    birth_relative = re.compile(
        r"\b(?:(?:within|in) (?:the )?first (?:few|two|2) weeks|"
        r"between (?:two|2) and (?:six|6) weeks|"
        r"between (?:six|6) and (?:twelve|12) weeks|"
        r"more than (?:three|3) months|"
        r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve) "
        r"(?:days?|weeks?|months?) (?:after|since) (?:birth|delivery|giving birth))\b",
        re.I,
    )
    cue_seen = False
    for sentence in sentences:
        value = sentence
        if expected_timing == "unknown":
            value = birth_relative.sub("after giving birth", value)
            value = re.sub(r"\bnewborn\b", "baby", value, flags=re.I)
            value = re.sub(r"\bearly postpartum\b", "postpartum", value, flags=re.I)
            value = re.sub(
                r"\b(?:in the )?early days\b",
                "during the postpartum period",
                value,
                flags=re.I,
            )
            value = re.sub(r"\b(?:from the start|early on)\b,?\s*", "", value, flags=re.I)

            # 2026-09-07: Keep one broad cue; later copies add no information.
            def keep_first_cue(match: re.Match[str]) -> str:
                nonlocal cue_seen
                if cue_seen:
                    return ""
                cue_seen = True
                return match.group(0)

            value = _POSTPARTUM_TIMING_CUE.sub(keep_first_cue, value)
        else:
            value = _POSTPARTUM_TIMING_CUE.sub("", value)
        value = re.sub(r"^\s*[,;:]\s*", "", value)
        value = re.sub(r"\s+([,.!?])", r"\1", value)
        value = re.sub(r"\s+", " ", value).strip()
        if value and not re.search(r"[.!?][\"']?$", value):
            value += "."
        changed = changed or value != sentence
        if value:
            cleaned.append(value)
    if expected_timing in _CANONICAL_TIMING_PREFIXES and cleaned:
        cleaned[0] = add_timing_opener(cleaned[0], expected_timing)
    changed = cleaned != sentences
    if changed:
        item["sentences"] = cleaned
        item["timing_language_normalized"] = True


def materialize_stage3_text(item: dict[str, Any]) -> str:
    """Join structured sentence or paragraph slots into the narrative field."""
    cleaned_sentences = stage3_sentence_values(item)
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


def trusted_generation_factors(factors: dict[str, Any]) -> dict[str, Any]:
    """Exclude the free-form model summary from Stage-3 factual support."""
    return {
        key: value
        for key, value in factors.items()
        if key != "deidentified_summary"
    }


_GROUNDING_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "has", "have", "i", "in", "is", "it", "me", "my",
    "of", "on", "that", "the", "these", "this", "to", "was", "with",
    "after", "birth", "during", "experience", "experiencing", "experienced",
    "feel", "feeling", "feelings", "individual", "mother", "new", "parent",
    "person", "postpartum", "time", "times", "challenging", "difficult",
    "about", "able", "company", "despite", "giving", "had", "i'm", "i've", "it's", "made",
    "makes", "moments", "notice", "noticed", "quality", "some", "spend", "things", "think",
    "through", "together", "trouble", "twice", "went", "will",
}


def grounding_tokens(text: str) -> set[str]:
    aliases = {
        "realized": "recognize",
        "realize": "recognize", "recognized": "recognize", "recognise": "recognize",
        "changes": "change", "changed": "change", "changing": "change",
        "struggling": "difficult", "struggle": "difficult", "hard": "difficult",
        "persistent": "ongoing", "persisting": "ongoing",
        "high": "intense", "severe": "intense", "strong": "intense",
        "supported": "support", "supportive": "support",
        "friends": "friend",
        "depressed": "depression", "down": "depression",
        "anxious": "anxiety", "worried": "worry",
        "abilities": "ability", "children": "child", "infant": "child",
        "babies": "child", "baby": "child", "recovered": "recover",
        "overcame": "recover", "overcome": "recover",
        "enjoying": "enjoy", "lately": "ongoing", "needed": "need",
        "needs": "need", "still": "ongoing", "uncertainty": "uncertain",
        "intake": "consumption", "mood": "depression", "reduced": "reduce",
        "reducing": "reduce",
    }
    tokens = set()
    for token in normalized_tokens(text):
        if token.endswith("'s"):
            token = token[:-2]
        token = aliases.get(token, token)
        if token not in _GROUNDING_STOPWORDS and len(token) > 2:
            tokens.add(token)
    return tokens


def source_grounded_values(values: Sequence[Any], source_text: str) -> list[str]:
    """Keep a model-extracted phrase only when its content words occur in the source."""
    source_tokens = grounding_tokens(source_text)
    grounded: list[str] = []
    for raw_value in values:
        value = re.sub(r"\s+", " ", str(raw_value or "")).strip()
        candidate_tokens = grounding_tokens(value)
        if not value or not candidate_tokens:
            continue
        overlap = candidate_tokens & source_tokens
        if overlap and len(overlap) / len(candidate_tokens) >= 0.60:
            grounded.append(value)
    return grounded


def reconcile_grounding_assessment(
    assessment: dict[str, Any], factors: dict[str, Any]
) -> dict[str, Any]:
    """Remove auditor claims that clearly overlap the supplied factual boundary."""
    claims = [str(value).strip() for value in assessment.get("unsupported_claims", [])]
    factor_tokens = grounding_tokens(text_values(trusted_generation_factors(factors)))
    remaining = []
    reconciled = []
    for claim in claims:
        claim_tokens = grounding_tokens(claim)
        overlap = claim_tokens & factor_tokens
        coverage = len(overlap) / max(1, len(claim_tokens))
        has_sensitive_addition = bool(unsupported_detail_categories(claim, factors))
        has_meta_language = bool(
            _META_NARRATIVE_PATTERN.search(claim)
            or _MISSINGNESS_NARRATIVE_PATTERN.search(claim)
        )
        minimum_overlap = 1 if len(claim_tokens) <= 2 else 2
        if (
            (len(overlap) >= minimum_overlap and coverage >= (2 / 3))
            and not has_sensitive_addition
            and not has_meta_language
        ):
            reconciled.append(claim)
        else:
            remaining.append(claim)
    assessment["unsupported_claims"] = remaining
    assessment["reconciled_supported_claims"] = reconciled
    assessment["grounded"] = not remaining
    if reconciled and not remaining:
        assessment["reason"] = "Auditor claims were direct restatements of supplied factors."
    return assessment


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


def source_supported_intensity(source_text: str, risk_indicators: Sequence[str]) -> str:
    """Assign intensity from explicit wording instead of model inference."""
    source = re.sub(r"\s+", " ", source_text or "").lower()
    if risk_indicators:
        return "high"
    explicit = re.search(
        r"\b(?P<level>mild|moderate|severe)\s+(?:postpartum\s+)?"
        r"(?:depress\w*|anxi\w*|symptoms?|distress)\b",
        source,
    )
    if explicit:
        return {
            "mild": "low", "moderate": "moderate", "severe": "high",
        }[explicit.group("level")]
    if re.search(
        r"\b(?:really|very|extremely)\W{0,3}\s*bad\s+(?:depress\w*|anxi\w*)|"
        r"\b(?:intense|overwhelming|debilitating|crippling)\s+"
        r"(?:depress\w*|anxi\w*|symptoms?|distress)|"
        r"\b(?:would not|wouldn't|do not|don't) know if i (?:would have )?surviv\w*|"
        r"\bbetter off without me\b|\bunable to (?:function|care)\b|"
        r"\b(?:disappeared|withdrew)\b.{0,35}\bfriend\w*\b",
        source,
    ):
        return "high"
    if re.search(
        r"\b(?:depress\w*|anxi\w*|intrusive thoughts?|panic\w*|hopeless\w*|"
        r"worthless\w*|guilt\w*|baby blues)\b",
        source,
    ):
        return "moderate"
    return "low"


def source_supported_context(source_text: str) -> list[str]:
    """Recover salient context only when a source pattern explicitly supports it."""
    source = re.sub(r"\s+", " ", source_text or "").lower()
    facts: list[str] = []

    def add(value: str) -> None:
        if value not in facts:
            facts.append(value)

    if re.search(r"\b(?:did not|didn'?t) (?:even )?(?:realize|recognize)\b", source):
        add("did not recognize the depression at first")
    if (
        re.search(r"\b(?:emotion\w*|body|mind) change\w*\b", source)
        and re.search(r"\b(?:baby|birth|postpartum)\b", source)
    ):
        add("noticed emotional and physical changes after birth")
    if re.search(
        r"\b(?:tough|hard|difficult)\b.{0,30}\b(?:handle|manage)\b.{0,25}"
        r"\b(?:situation|period|things?)\b",
        source,
    ):
        add("had difficulty managing the postpartum period")
    if re.search(r"\bbaby duty\b", source):
        add("was caring for the baby")
    if re.search(r"\bmaternity leave\b", source):
        add("was on maternity leave")
    if re.search(
        r"\b(?:depress\w*|suffer\w*)\b.{0,35}\bfor\s+"
        r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"(?:weeks?|months?)\b",
        source,
    ):
        add("depression lasted for an extended period")
    if re.search(r"\b(?:cut|shut)\b.{0,35}\bfriend\w*\b|\bwithdrew\b.{0,35}\bfriend\w*\b", source):
        add("withdrew from friends")
    if re.search(r"\bfeel(?:ing)? more like myself\b", source):
        add("now feels more like oneself")
    if re.search(r"\b(?:angel|child|baby)\b.{0,30}\b(?:next to me|share it with|enjoy)\b", source):
        add("now enjoys time with the child")
    if re.search(r"\b(?:did not|didn'?t) ask for it\b|\bno way to prevent it\b", source):
        add("felt unable to prevent the depression")
    if re.search(r"\b(?:thankful|grateful)\b.{0,45}\b(?:hold|holding) my (?:baby|infant|child)\b", source):
        add("felt thankful about being able to hold the baby")
    if re.search(
        r"\b(?:play|interact) with (?:my |the )?(?:baby|infant|child)\b|"
        r"\b(?:play|interact) with (?:him|her|them)\b",
        source,
    ) and re.search(
        r"\b(?:told|said)\b.{0,45}\b(?:did not|didn'?t) have (?:ppd|postpartum depression)\b",
        source,
    ):
        add("ability to play with the baby was used to dismiss the depression")
    if re.search(
        r"\b(?:had it not been for|without)\b.{0,50}\b(?:would not|wouldn't|don't know if)\b"
        r".{0,25}\bsurviv\w*",
        source,
    ):
        add("the clinician's recognition made an important difference")
    if re.search(r"\b(?:doctor|clinician|dr\.?)\b.{0,35}\b(?:told|advised)\b", source) and re.search(
        r"\bexercis\w*\b", source
    ):
        add("exercise was suggested by a clinician")
    if re.search(r"\b(?:do not|don't|did not|didn't) have enough brea(?:st|ts) ?milk\b", source):
        add("had intrusive thoughts about not having enough breast milk")
    if re.search(r"\b(?:i am|i'm|felt like|feel like) (?:a )?bad (?:mom|mother|parent)\b", source):
        add("felt like a bad parent")
    if re.search(r"\b(?:baby|child|she|he) (?:will|would) be better off without me\b", source):
        add("had thoughts that the baby would be better off without oneself")
    return facts


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


def stage3_generalized_summary(summary: str) -> str:
    """Keep Stage-3 summaries from competing with the canonical timing bucket."""
    amount = (
        r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|a few|several)"
    )
    duration = rf"(?:about|around|approximately|almost|nearly|over)?\s*{amount}\s+(?:days?|weeks?|months?|years?)"
    value = re.sub(rf"\blasting\s+{duration}\b", "during the postpartum period", summary, flags=re.I)
    value = re.sub(rf"\bsince\s+{duration}(?:\s+ago)?\b", "during the postpartum period", value, flags=re.I)
    value = re.sub(rf"\buntil\s+{duration}\b", "until later", value, flags=re.I)
    # Intensity is carried separately so the surface model cannot overread one adjective.
    value = re.sub(
        r"\b(?:mild|moderate|severe|intense)\s+(?=(?:postpartum\s+)?"
        r"(?:depress\w*|anxi\w*|symptoms?|distress)\b)",
        "",
        value,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", value).strip()


def stage3_provenance(generated: dict[str, Any]) -> str:
    """Whether the accepted text came from the model or the deterministic fallback."""
    history = generated.get("grounding_history") or []
    for entry in history:
        if str(entry.get("round", "")).startswith("fallback"):
            return "fallback"
    return "llm" if history else ""


def narrative_word_bounds(
    factors: dict[str, Any], cfg: dict[str, Any]
) -> tuple[int, int]:
    """Match narrative length to the amount of supported source evidence."""
    if not cfg.get("use_adaptive_length", True):
        return int(cfg["min_words"]), int(cfg["max_words"])

    evidence_units = 0
    for key in (
        "symptoms", "functional_impact", "feeding_or_infant_care_stressors",
        "bonding_indicators", "risk_indicators", "risk_denials",
    ):
        evidence_units += int(any(
            bool(str(value).strip()) for value in factors.get(key, [])
        ))
    supported_context = {
        str(value).strip().lower()
        for value in factors.get("supported_context", [])
        if str(value).strip()
    }
    bonding = {
        str(value).strip().lower()
        for value in factors.get("bonding_indicators", [])
        if str(value).strip()
    }
    context_units = len(supported_context)
    if (
        "felt thankful about being able to hold the baby" in supported_context
        and "able to hold baby" in bonding
    ):
        context_units -= 1
    evidence_units += min(3, context_units)
    for key in (
        "symptom_persistence", "sleep_context", "perceived_support",
        "coping_or_adjustment_context",
    ):
        value = str(factors.get(key, "")).strip().lower()
        if value and value not in {"unknown", "not stated", "not specified"}:
            evidence_units += 1
    if str(factors.get("postpartum_timing", "unknown")) != "unknown":
        evidence_units += 1

    if evidence_units <= 5:
        return int(cfg["sparse_min_words"]), int(cfg["sparse_max_words"])
    return int(cfg["min_words"]), int(cfg["max_words"])


def narrative_prompt_bounds(
    factors: dict[str, Any], cfg: dict[str, Any]
) -> tuple[int, int]:
    """Leave a small margin between the requested and hard maximum."""
    low, high = narrative_word_bounds(factors, cfg)
    if cfg.get("use_adaptive_length", True) and high == int(cfg["sparse_max_words"]):
        high = max(low, int(cfg["sparse_target_max_words"]))
    return low, high


def conservative_factor_narrative(
    factors: dict[str, Any], severity: str, use_timing: bool,
    min_words: int, max_words: int, variant: int = 0,
) -> dict[str, Any]:
    """Build a source-bounded fallback from deterministic factors."""
    sentences: list[str] = []

    def add(sentence: str) -> None:
        value = re.sub(r"\s+", " ", sentence).strip()
        if value and value not in sentences:
            sentences.append(value)

    timing = str(factors.get("postpartum_timing") or "unknown")
    symptoms = [
        str(value).strip().lower()
        for value in factors.get("symptoms", [])
        if str(value).strip()
        and not re.search(
            r"\b(?:suicid\w*|self[- ]?harm|harm(?:ing)? (?:my |the )?"
            r"(?:baby|infant|child))\b",
            str(value),
            re.I,
        )
    ]
    symptom_text = ""
    if symptoms:
        symptom_text = (
            symptoms[0]
            if len(symptoms) == 1
            else ", ".join(symptoms[:-1]) + f" and {symptoms[-1]}"
        )
    intensity = str(factors.get("symptom_intensity") or "").strip().lower()
    persistence = str(factors.get("symptom_persistence") or "").strip().lower()
    past_experience = persistence.startswith("past,")
    resolved = persistence in {"resolved", "improved", "recovered", "no longer present"}

    if symptom_text:
        if re.search(r"\b(?:mild|moderate|severe|intense)\b", symptom_text, re.I):
            symptom_phrase = symptom_text
        elif intensity == "low":
            symptom_phrase = f"mild {symptom_text}"
        elif intensity == "high":
            symptom_phrase = f"intense {symptom_text}"
        else:
            symptom_phrase = symptom_text

        if resolved:
            sentence = (
                f"After giving birth, I went through {symptom_phrase}, but those feelings "
                "have since eased."
            )
        elif past_experience:
            sentence = f"After giving birth, I went through {symptom_phrase}."
        elif persistence == "ongoing":
            if intensity == "moderate":
                sentence = (
                    f"After giving birth, I continue to experience {symptom_text}, and "
                    "coping with it remains difficult."
                )
            else:
                sentence = f"After giving birth, I am still experiencing {symptom_phrase}."
        elif intensity == "moderate":
            sentence = (
                f"After giving birth, I experienced {symptom_text} that felt emotionally difficult."
            )
        else:
            sentence = f"After giving birth, I experienced {symptom_phrase}."
        if use_timing and timing in _CANONICAL_TIMING_PREFIXES:
            sentence = add_timing_opener(sentence, timing)
        add(sentence)

    context_sentences = {
        "did not recognize the depression at first":
            "I did not recognize the depression at first.",
        "noticed emotional and physical changes after birth":
            "I noticed emotional and physical changes during this experience.",
        "had difficulty managing the postpartum period":
            "Managing the postpartum period has been difficult for me."
            if persistence == "ongoing"
            else "Managing the postpartum period was difficult for me.",
        "was caring for the baby":
            "I was caring for my baby at the time.",
        "was on maternity leave":
            "I was also on maternity leave during this period.",
        "depression lasted for an extended period":
            "The depression lasted for an extended period.",
        "withdrew from friends":
            "At that time, I withdrew from friends.",
        "now feels more like oneself":
            "I now feel more like myself.",
        "now enjoys time with the child":
            "I now enjoy spending time with my child.",
        "felt unable to prevent the depression":
            "I felt unable to prevent the depression.",
        "the clinician's recognition made an important difference":
            "That recognition made an important difference for me.",
        "exercise was suggested by a clinician":
            "A clinician suggested that I exercise more consistently.",
        "felt thankful about being able to hold the baby":
            "I feel thankful that I can hold my baby now.",
        "ability to play with the baby was used to dismiss the depression":
            "My ability to play with my baby was used to dismiss my depression.",
        "had intrusive thoughts about not having enough breast milk":
            "I had intrusive thoughts that I did not have enough breast milk.",
        "felt like a bad parent":
            "I felt like a bad mother.",
        "had thoughts that the baby would be better off without oneself":
            "I had thoughts that my baby would be better off without me.",
    }
    supported_context = [
        str(value).strip().lower()
        for value in factors.get("supported_context", [])
        if str(value).strip()
    ]
    for context in supported_context:
        if context == "the clinician's recognition made an important difference":
            continue
        if context in context_sentences:
            add(context_sentences[context])

    impacts = [
        str(value).strip() for value in factors.get("functional_impact", [])
        if str(value).strip()
    ]
    for value in impacts:
        add(
            f"I had {value.lower()}."
            if re.match(r"^difficulty\b", value, re.I)
            else f"My experience affected {value.lower()}."
        )

    care = [
        str(value).strip()
        for value in factors.get("feeding_or_infant_care_stressors", [])
        if str(value).strip()
    ]
    for value in care:
        add(f"I also faced {value.lower()}.")

    bonding = {
        str(value).strip().lower()
        for value in factors.get("bonding_indicators", [])
        if str(value).strip()
    }
    if "positive bonding experience" in bonding:
        add("I felt a positive bond with my baby.")
    if "able to hold baby" in bonding:
        if "felt thankful about being able to hold the baby" not in supported_context:
            add("I was able to hold my baby.")
    if "able to interact with baby" in bonding:
        add("I was able to interact with my baby.")
    if "difficulty bonding" in bonding:
        add("I had difficulty bonding with my baby.")

    risk = [
        str(value).strip() for value in factors.get("risk_indicators", [])
        if str(value).strip()
    ]
    if risk:
        self_harm = any("self-harm" in value.lower() for value in risk)
        infant_harm = any("harming my baby" in value.lower() for value in risk)
        if self_harm and infant_harm:
            joined = "thoughts of harming myself and my baby"
        elif self_harm:
            joined = "thoughts of harming myself"
        elif infant_harm:
            joined = "thoughts of harming my baby"
        else:
            joined = " and ".join(risk).lower()
        if past_experience or resolved:
            add(f"At that time, I experienced {joined}.")
        elif persistence == "ongoing":
            add(f"I am currently experiencing {joined}.")
        else:
            add(f"After giving birth, I experienced {joined}.")

    denials = {
        str(value).strip().lower()
        for value in factors.get("risk_denials", [])
        if str(value).strip()
    }
    if "denied thoughts of self-harm" in denials:
        add("I did not have thoughts of harming myself.")
    if "denied thoughts of harming my baby" in denials:
        add("I did not have thoughts of harming my baby.")

    support = str(factors.get("perceived_support") or "").strip().lower()
    if re.search(r"strong support from friends and family", support, re.I):
        add("I felt supported by friends and family.")
    elif re.search(r"\b(?:lack of|limited|little|no|low) support\b|^limited$|^low$", support, re.I):
        add("At that time, I did not have enough support." if past_experience or resolved
            else "I do not have enough support.")

    coping = str(factors.get("coping_or_adjustment_context") or "").strip().lower()
    if re.search(r"needs? time to adjust", coping, re.I):
        add("I needed time to adjust to these changes." if past_experience or resolved
            else "I need time to adjust to these changes.")
    elif re.search(r"consistent exercis\w*", coping, re.I) and re.search(
        r"\b(?:improv\w*|help\w*|difference)\b", coping, re.I
    ):
        add("I exercised consistently and noticed some improvement.")
    elif re.search(r"exercis\w*(?: more)? consistently|consistent exercis\w*", coping, re.I):
        add("I exercised more consistently as a way of coping.")
    elif re.search(r"wanted to resume exercise", coping, re.I):
        add("I wanted to return to exercise as part of moving forward.")
    elif re.search(r"associated the depression with hormonal changes", coping, re.I):
        add("I associated the depression with hormonal changes.")
    elif re.search(r"sought help but was told to wait", coping, re.I):
        add("I sought help, but I was told to wait.")
    elif re.search(r"sought therapy and did not return", coping, re.I):
        if "ability to play with the baby was used to dismiss the depression" in supported_context:
            add("I sought therapy, but I did not return.")
        elif "dismissed" in coping:
            add("I sought therapy, but I did not return after my depression was dismissed.")
        else:
            add("I sought therapy, but I did not return.")
    elif re.search(r"reached out to a mental-health professional", coping, re.I):
        add("I reached out to a mental-health professional.")
    elif re.search(r"clinician recognized the depression during an infant checkup", coping, re.I):
        add("A clinician recognized my depression during my baby's checkup.")
    elif re.search(r"clinician recognized the depression", coping, re.I):
        add("A clinician recognized my depression.")
    elif re.search(r"reduced alcohol consumption", coping, re.I):
        add("I limited how much alcohol I drank.")
    elif re.search(r"baby cried.*(?:paused|eased).*care", coping, re.I):
        add("When my baby cried, the depression eased long enough for me to provide care.")
    elif re.search(r"(?:symptoms|depression) (?:paused|eased) long enough to care", coping, re.I):
        add("The depression eased long enough for me to care for my baby.")

    if "the clinician's recognition made an important difference" in supported_context:
        add(
            "That recognition made an important difference during a very difficult "
            "postpartum period."
        )

    # A sparse seed sometimes needs one natural restatement to meet the lower bound.
    if word_count(" ".join(sentences)) < min_words and symptom_text:
        add({
            "low": "The feelings were mild but still noticeable to me.",
            "moderate": "The experience was emotionally difficult for me.",
            "high": "The experience felt intense and deeply distressing to me.",
        }.get(intensity, "The experience was emotionally difficult for me."))

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
    "infant-care detail": re.compile(
        r"\b(?:care|caring) for (?:my |the )?(?:baby|infant|child)|"
        r"\blook(?:ing)? after (?:my |the )?(?:baby|infant|child)|"
        r"\bresponsib\w* for another life\b",
        re.I,
    ),
    "high-risk/bonding detail": re.compile(
        r"\b(?:suicid\w*|self[- ]?harm|hopeless\w*|want to die|not worth living|"
        r"end my life|better off without me|harm(?:ing|ed)? myself|"
        r"hurt(?:ing|ed)? myself|harm(?:ing)? (?:my |the )?baby|"
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
        r"get through (?:the )?day|responsibilities feel|functional impact)\b",
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
        r"\b(?:fatigu\w*|tired(?:ness)?|exhaust\w*|drained|no energy|low energy|"
        r"energy (?:level )?(?:change\w*|problem\w*|fluctuat\w*))\b", re.I
    ),
    "appetite detail": re.compile(
        r"\b(?:appetite|not eating|eat(?:ing)? (?:too much|too little|less)|loss of appetite)\b",
        re.I,
    ),
    "concentration detail": re.compile(
        r"\b(?:concentrat\w*|focus(?:ing)?|brain fog|forgetful\w*|"
        r"(?:mind|thinking|thoughts?) (?:feels? )?cloud\w*)\b", re.I
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
    "frequency or trajectory detail": re.compile(
        r"\b(?:throughout the day|some days|day to day|every day|daily|constantly|"
        r"frequently|often|unchanged|remains? stable|stayed the same|"
        r"better than (?:other|some) days|getting (?:better|worse)|"
        r"increasingly|worsen\w*|evolv\w* over time)\b",
        re.I,
    ),
    "coping or priority detail": re.compile(
        r"\b(?:top priority|trying to cope|working to cope|push(?:ing)? through|"
        r"holding me back|manage (?:this|my)(?: [a-z-]+)? condition|"
        r"to manage (?:this|my)|tak(?:e|ing) care of myself|determined to)\b",
        re.I,
    ),
    "causal or benefit detail": re.compile(
        r"\b(?:help(?:ed|ing|s)? (?:me|slightly|a little)|overshadow\w*|"
        r"made things? (?:better|worse)|because of this|as a result|"
        r"(?:crucial|essential|key) (?:for|to))\b",
        re.I,
    ),
    "new emotional appraisal": re.compile(
        r"\b(?:hopeful|grateful|proud|confident|optimistic|relieved|encouraged)\b",
        re.I,
    ),
    "expectation or comparison detail": re.compile(
        r"\b(?:more|less|harder|easier|better|worse)\b.{0,25}\bthan (?:i )?"
        r"(?:anticipated|expected)|\bnot what i (?:anticipated|expected)\b",
        re.I,
    ),
}

_EXTRA_TIMING_DETAIL = re.compile(
    r"\b(?:(?:for|about|around|nearly|over|more than)\s+)?"
    r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a few|few|several)\s+"
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
    support_text = text_values(trusted_generation_factors(factors))
    categories = [
        category
        for category, pattern in _SUPPORTED_DETAIL_PATTERNS.items()
        if pattern.search(text or "") and not pattern.search(support_text)
    ]
    expected_timing = str(factors.get("postpartum_timing") or "unknown")
    if has_extra_timing_detail(text, expected_timing):
        categories.append("extra timing detail")
    descriptor = re.search(
        r"\b(mild|moderate|severe)\s+(?:depress\w*|anxi\w*|symptoms?|distress)\b",
        text or "",
        re.I,
    )
    if descriptor:
        expected_intensity = {"mild": "low", "moderate": "moderate", "severe": "high"}[
            descriptor.group(1).lower()
        ]
        if str(factors.get("symptom_intensity") or "").strip().lower() != expected_intensity:
            categories.append("conflicting symptom descriptor")
    return categories


_META_NARRATIVE_PATTERN = re.compile(
    r"\b(?:severity|intensity|moderately|low[- ]grade|"
    r"(?:low|mild|moderate|high|severe) (?:level|degree)(?: of)?|"
    r"(?:mild|moderate|severe) symptoms?|"
    r"according to (?:the )?(?:assessment|audit|audited record)|audited record|"
    r"assigned (?:severity|label)|target (?:severity|label)|classified as|"
    r"low[- ]intensity|moderate[- ]intensity|high[- ]intensity|"
    r"at (?:low|moderate|high) intensity|stated symptom|stated support|"
    r"stated safety|adjustment context|reported symptoms?)\b",
    re.I,
)
_MISSINGNESS_NARRATIVE_PATTERN = re.compile(
    r"\b(?:current status (?:is )?unknown|details? (?:are|is|were|was) unknown|"
    r"no clarity|not specified|not provided|none (?:was|were) provided|"
    r"no specific details?|unknown factors?|the (?:record|account) (?:states|shows|places)|"
    r"no professional diagnosis or treatment has occurred)\b",
    re.I,
)
_STOCK_OR_CLINICAL_NARRATIVE_PATTERN = re.compile(
    r"\b(?:my postpartum experience (?:includes|has been|was|is)|"
    r"within my postpartum experience|"
    r"as i reflect on my postpartum journey|within this timeframe|navigate these changes|"
    r"i describe\b.{0,45}\b(?:experience|emotion)|positive bonding experience|"
    r"part of my emotional experience|coping mechanisms?|before intervention|the condition|"
    r"due to its negative impact|emotional symptoms|low support|noticeable distress|"
    r"feel emotionally difficult|symptoms paused|past impact on me|"
    r"weigh(?:s|ed|ing)? heavily on me|affect(?:s|ed|ing)? me deeply|"
    r"at this point,? i (?:did not|didn't) (?:recognize|realize)\b.{0,30}\bat first)\b",
    re.I,
)
_STANDALONE_TIMING_PATTERN = re.compile(
    r"(?:^|[.!?]\s+)(?:I am\s+)?(?:within the first two weeks|"
    r"between two and six weeks|between six and twelve weeks|more than three months)"
    r" after birth[.!?](?:\s|$)",
    re.I,
)


def narrative_quality_violations(
    item: dict[str, Any], text: str, factors: dict[str, Any]
) -> list[str]:
    """Catch human-review failures that schema and grounding checks can miss."""
    violations: list[str] = []
    if _META_NARRATIVE_PATTERN.search(text):
        violations.append("meta or label language")
    if _MISSINGNESS_NARRATIVE_PATTERN.search(text):
        violations.append("missingness or record language")
    if _STOCK_OR_CLINICAL_NARRATIVE_PATTERN.search(text):
        violations.append("stock or clinical narration")
    if _STANDALONE_TIMING_PATTERN.search(text):
        violations.append("standalone timing sentence")
    if _THIRD_PERSON_NARRATIVE_PATTERN.search(text):
        violations.append("third-person narration")
    postpartum_cues = re.findall(
        r"\b(?:after giving birth|during (?:my|the) postpartum period|"
        r"within the first two weeks after birth|between two and six weeks after birth|"
        r"between six and twelve weeks after birth|more than three months after birth)\b",
        text,
        re.I,
    )
    if len(postpartum_cues) > 1:
        violations.append("repeated postpartum timing cue")
    factor_text = text_values(trusted_generation_factors(factors)).lower()
    if re.search(r"\bquality time\b", text, re.I) and "quality time" not in factor_text:
        violations.append("unsupported bonding embellishment")
    if re.search(
        r"\b(?:emotional and physical|physical and emotional) changes\b.{0,25}"
        r"\b(?:ongoing|continued|persisted)\b",
        text,
        re.I,
    ):
        violations.append("unsupported trajectory applied to postpartum changes")
    factor_descriptor_text = re.sub(r"\bpostpartum\s+", "", factor_text)
    descriptors = re.finditer(
        r"\b(?P<level>mild|moderate|severe)\s+(?:postpartum\s+)?"
        r"(?:depress\w*|anxi\w*|symptoms?|distress)\b",
        text or "",
        re.I,
    )
    expected_intensity = str(factors.get("symptom_intensity") or "").strip().lower()
    intensity_map = {"mild": "low", "moderate": "moderate", "severe": "high"}
    if any(
        match.group(0).lower() not in factor_descriptor_text
        and intensity_map[match.group("level").lower()] != expected_intensity
        for match in descriptors
    ):
        violations.append("severity descriptor not source-grounded")
    if re.search(
        r"(?:^|[.!?]\s+)(?:Acknowledge|Experienced|Prioritizing|Trying)\b",
        text,
        re.I,
    ):
        violations.append("sentence fragment or non-diary command")

    slot_values = stage3_sentence_values(item)
    if slot_values:
        first_person_slots = sum(
            bool(re.search(r"\b(?:I|I'm|I've|I'd|my|me)\b", value, re.I))
            for value in slot_values
        )
        required_first_person = (
            1
            if item.get("grounding_method") == "deterministic_factor_fallback"
            else max(2, len(slot_values) - 1)
        )
        if first_person_slots < required_first_person:
            violations.append("insufficient first-person voice")

    persistence = str(factors.get("symptom_persistence") or "").strip().lower()
    symptom_terms = r"(?:depress\w*|anxi\w*|symptoms?|thoughts?|feelings?|emotions?|distress)"
    present_status = re.compile(
        rf"\b(?:(?:I am|I'm|I have been|I've been|I continue to|I still)\s+"
        rf"(?:currently\s+)?(?:experiencing|dealing with|struggling with|feeling|having)?"
        rf"\s*.{{0,25}}?{symptom_terms}|"
        rf"my {symptom_terms} (?:is|are|remain|affect|impact)|"
        rf"{symptom_terms} (?:is|are|remain|affect|impact))\b",
        re.I,
    )
    if (persistence.startswith("past,") or persistence in {"resolved", "improved", "recovered"}) \
            and (
                present_status.search(text)
                or re.search(r"\b(?:is|are) (?:causing|holding|affecting|impacting)\b", text, re.I)
                or re.search(r"\bstill (?:haunt|affect|impact|trouble|bother)\w* me\b", text, re.I)
            ):
        violations.append("past or resolved symptoms changed to current")
    if persistence == "ongoing" and re.search(
        r"\b(?:previously|past experience|symptoms? (?:had )?resolved|no longer present)\b",
        text,
        re.I,
    ):
        violations.append("ongoing symptoms changed to past or resolved")
    if persistence == "ongoing" and not re.search(
        r"\b(?:ongoing|persist\w*|continue(?:s|d)? to|has continued|have continued|"
        r"i have been|i've been|all along|still (?:deal\w*|struggl\w*|feel\w*|"
        r"experienc\w*|work\w*)|remain(?:s|ed)? difficult|has not eased|hasn't eased)\b",
        text,
        re.I,
    ):
        violations.append("ongoing symptom status omitted")
    if persistence in {"resolved", "improved", "recovered", "no longer present"} and not re.search(
        r"\b(?:eased|improved|recovered|resolved|no longer)\b", text, re.I
    ):
        violations.append("resolved symptom status omitted")
    if persistence in {"", "unknown", "not stated", "not specified"}:
        unsupported_status = re.search(
            r"\b(?:ongoing|persist\w*|still|currently|lately|every day|daily|previously|"
            r"resolved|recovered|i have been|i've been|right now)\b|"
            r"\b(?:i am|i'm) (?:currently )?(?:experienc\w*|feel\w*|dealing|struggl\w*|facing)\b|"
            r"\bmy (?:feelings?|emotions?) (?:feel|are|remain)\b",
            text,
            re.I,
        )
        if unsupported_status:
            violations.append("unsupported symptom status")
    if re.search(r"\b(?:had been ongoing|ongoing before|before intervention)\b", text, re.I) \
            and "depression lasted for an extended period" not in {
                str(value).strip().lower()
                for value in factors.get("supported_context", [])
            }:
        violations.append("unsupported duration or trajectory")
    if (
        "felt unable to prevent the depression" in {
            str(value).strip().lower()
            for value in factors.get("supported_context", [])
        }
        and re.search(
            r"\b(?:difficult\w*|symptoms?|feelings?)\b.{0,25}"
            r"\b(?:made|caused|led)\b.{0,20}\b(?:unable|could not|couldn't)\b"
            r".{0,25}\bprevent\w*\b",
            text,
            re.I,
        )
    ):
        violations.append("unsupported causal relation")
    intensity = str(factors.get("symptom_intensity") or "").strip().lower()
    if intensity == "high" and not re.search(
        r"\b(?:intense|severe|extreme|overwhelming|very strong|deep)\b", text, re.I
    ):
        violations.append("high symptom intensity omitted")

    coping = str(factors.get("coping_or_adjustment_context") or "").strip().lower()
    coping_requirements = (
        (r"needs? time to adjust", r"\bneed(?:ed)?\b.{0,25}\btime\b.{0,25}\badjust\w*\b", "adjustment context omitted"),
        (r"reduced alcohol", r"\b(?:reduc\w*|limit\w*|cut(?:ting)? back|stopp\w*)\b.{0,35}\b(?:alcohol|drinking)\b|\b(?:alcohol|drinking)\b.{0,35}\b(?:reduc\w*|limit\w*|cut(?:ting)? back|stopp\w*)\b", "alcohol reduction omitted"),
        (r"exercise.*improv|improv.*exercise", r"\bexercis\w*\b.*\b(?:help\w*|improv\w*|difference)\b|\b(?:help\w*|improv\w*|difference)\b.*\bexercis\w*\b", "exercise improvement omitted"),
        (r"(?:consistent|resume) exercise", r"\b(?:exercise\w*|gym)\b", "exercise context omitted"),
        (r"told to wait", r"\b(?:told|advised)\b.{0,25}\bwait\b", "help-seeking outcome omitted"),
        (r"sought therapy and did not return", r"\b(?:therapy|therapist|mental-health professional)\b.{0,65}\b(?:dismiss\w*|did not return|didn't return)\b|\b(?:dismiss\w*|did not return|didn't return)\b.{0,65}\b(?:therapy|therapist|mental-health professional)\b", "therapy outcome omitted"),
        (r"reached out to a mental-health professional", r"\b(?:mental-health professional|therapist|psychologist|counselor)\b", "professional help omitted"),
        (r"clinician recognized", r"\bclinician\b.{0,45}\b(?:recogniz\w*|identif\w*|notic\w*)\b|\b(?:recogniz\w*|identif\w*|notic\w*)\b.{0,45}\bclinician\b", "clinical recognition omitted"),
        (r"hormonal changes", r"\b(?:hormones?|hormonal)\b", "hormonal context omitted"),
        (r"(?:paused|eased) long enough to care", r"\b(?:paused|stopped|eased)\b.{0,55}\bcare\b|\bcare\b.{0,55}\b(?:paused|stopped|eased)\b", "caregiving context omitted"),
    )
    for factor_pattern, text_pattern, message in coping_requirements:
        if re.search(factor_pattern, coping, re.I) and not re.search(text_pattern, text, re.I):
            violations.append(message)

    bonding = {str(value).strip().lower() for value in factors.get("bonding_indicators", [])}
    if "positive bonding experience" in bonding and not re.search(
        r"\b(?:bond\w*|love\w*|connect\w*|close)\b.{0,35}\b(?:baby|infant|child)\b|"
        r"\b(?:baby|infant|child)\b.{0,35}\b(?:bond\w*|love\w*|connect\w*|close)\b",
        text,
        re.I,
    ):
        violations.append("positive bonding context omitted")
    if "able to hold baby" in bonding and not re.search(
        r"\b(?:hold|held)\b.{0,25}\b(?:baby|infant|child)\b",
        text,
        re.I,
    ):
        violations.append("ability to hold baby omitted")
    if "able to interact with baby" in bonding and not re.search(
        r"\b(?:interact\w*|play\w*)\b.{0,25}\b(?:baby|infant|child)\b",
        text,
        re.I,
    ):
        violations.append("infant interaction omitted")

    context_requirements = {
        "did not recognize the depression at first": r"\b(?:did not|didn't)\b.{0,25}\b(?:recogniz\w*|realiz\w*)\b",
        "noticed emotional and physical changes after birth": r"\bemotional\b.{0,45}\bphysical\b|\bphysical\b.{0,45}\bemotional\b",
        "had difficulty managing the postpartum period": r"\b(?:manag\w*|handl\w*)\b.{0,35}\b(?:difficult|hard|tough)\b|\b(?:difficult|hard|tough)\b.{0,35}\b(?:manag\w*|handl\w*)\b",
        "was caring for the baby": r"\b(?:care|caring)\b.{0,25}\b(?:baby|infant|child)\b|\bbaby duty\b",
        "was on maternity leave": r"\bmaternity leave\b",
        "depression lasted for an extended period": r"\b(?:last\w*|extended|prolonged)\b.{0,35}\b(?:depress\w*|period)\b|\bdepress\w*\b.{0,35}\b(?:last\w*|extended|prolonged)\b",
        "withdrew from friends": r"\b(?:withdr\w*|cut\w* off|pulled away)\b.{0,35}\bfriends?\b",
        "now feels more like oneself": r"\b(?:now|currently)\b.{0,30}\bmore like myself\b|\bfeel\w* more like myself\b",
        "now enjoys time with the child": r"\b(?:enjoy\w*|share\w*)\b.{0,35}\b(?:baby|infant|child)\b",
        "felt unable to prevent the depression": r"\b(?:unable|no way|could not|couldn't)\b.{0,35}\bprevent\w*\b|\bbeyond my control\b",
        "the clinician's recognition made an important difference": r"\b(?:recognition|recogniz\w*|identif\w*|caught)\b.{0,45}\b(?:difference|important|help\w*)\b",
        "exercise was suggested by a clinician": r"\b(?:clinician|doctor)\b.{0,50}\b(?:suggest\w*|recommend\w*|told)\b.{0,35}\bexercis\w*\b",
        "felt thankful about being able to hold the baby": r"\b(?:thankful|grateful)\b.{0,45}\bhold\w*\b.{0,20}\b(?:baby|infant|child)\b",
        "ability to play with the baby was used to dismiss the depression": r"\bplay\w*\b.{0,40}\b(?:baby|infant|child)\b.{0,65}\b(?:dismiss\w*|did not have|didn't have|ruled out)\b|\b(?:dismiss\w*|did not have|didn't have|ruled out)\b.{0,65}\bplay\w*\b",
        "had intrusive thoughts about not having enough breast milk": r"\bintrusive thoughts?\b.{0,55}\bnot (?:have|having) enough breast ?milk\b",
        "felt like a bad parent": r"\b(?:bad|inadequate) (?:mom|mother|parent)\b",
        "had thoughts that the baby would be better off without oneself": r"\b(?:baby|infant|child)\b.{0,35}\bbetter off without me\b",
    }
    for context in factors.get("supported_context", []):
        normalized_context = str(context).strip().lower()
        requirement = context_requirements.get(normalized_context)
        if requirement and not re.search(requirement, text, re.I):
            violations.append("supported context omitted: " + normalized_context)

    if item.get("grounding_method") != "deterministic_factor_fallback" and re.search(
        r"\bi (?:am )?describ(?:e|ing)\b.{0,35}\b(?:same|those)\b|"
        r"\bthe same emotional (?:difficulty|experience)\b",
        text,
        re.I,
    ):
        violations.append("formulaic repeated content")

    denials = {str(value).strip().lower() for value in factors.get("risk_denials", [])}
    if "denied thoughts of self-harm" in denials:
        if source_has_affirmed_risk(text, _SELF_HARM_PATTERN):
            violations.append("self-harm denial polarity changed")
        if not source_has_denied_risk(text, _SELF_HARM_PATTERN):
            violations.append("self-harm denial omitted")
    if "denied thoughts of harming my baby" in denials:
        if source_has_affirmed_risk(text, _INFANT_HARM_PATTERN):
            violations.append("infant-harm denial polarity changed")
        if not source_has_denied_risk(text, _INFANT_HARM_PATTERN):
            violations.append("infant-harm denial omitted")
    return violations


def remove_unsourced_severity_descriptors(
    item: dict[str, Any], factors: dict[str, Any]
) -> None:
    """Remove a bucket-like adjective unless the factor text explicitly contains it."""
    factor_text = re.sub(
        r"\bpostpartum\s+",
        "",
        text_values(trusted_generation_factors(factors)).lower(),
    )
    pattern = re.compile(
        r"\b(?P<level>mild|moderate|severe)\s+(?P<postpartum>postpartum\s+)?"
        r"(?P<noun>depress\w*|anxi\w*|symptoms?|distress)\b",
        re.I,
    )
    expected_intensity = str(factors.get("symptom_intensity") or "").strip().lower()
    intensity_map = {"mild": "low", "moderate": "moderate", "severe": "high"}
    changed = False

    def clean(value: str) -> str:
        nonlocal changed

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            normalized = re.sub(r"\bpostpartum\s+", "", match.group(0).lower())
            if (
                normalized in factor_text
                or intensity_map[match.group("level").lower()] == expected_intensity
            ):
                return match.group(0)
            changed = True
            return f"{match.group('postpartum') or ''}{match.group('noun')}"

        return pattern.sub(replace, value)

    slots = item.get("sentences")
    if isinstance(slots, dict):
        item["sentences"] = {key: clean(str(value)) for key, value in slots.items()}
    elif isinstance(slots, list):
        item["sentences"] = [clean(str(value)) for value in slots]
    elif isinstance(item.get("synthetic_text"), str):
        item["synthetic_text"] = clean(str(item["synthetic_text"]))
    if changed:
        item["severity_descriptor_removed"] = True


def normalize_target_severity_descriptors(
    item: dict[str, Any], expected_severity: str
) -> None:
    """Keep bucket words in the prose consistent with the assigned target."""
    pattern = re.compile(
        r"\b(?P<level>mild|moderate|severe)\s+(?P<postpartum>postpartum\s+)?"
        r"(?P<noun>depress\w*|anxi\w*|symptoms?|distress)\b",
        re.I,
    )
    changed = False

    def clean(value: str) -> str:
        nonlocal changed

        def replace(match: re.Match[str]) -> str:
            nonlocal changed
            if match.group("level").lower() == expected_severity.lower():
                return match.group(0)
            changed = True
            adjective = "intense " if match.group("level").lower() == "severe" else ""
            return f"{adjective}{match.group('postpartum') or ''}{match.group('noun')}"

        return re.sub(r"\s+", " ", pattern.sub(replace, value)).strip()

    slots = item.get("sentences")
    if isinstance(slots, dict):
        item["sentences"] = {key: clean(str(value)) for key, value in slots.items()}
    elif isinstance(slots, list):
        item["sentences"] = [clean(str(value)) for value in slots]
    elif isinstance(item.get("synthetic_text"), str):
        item["synthetic_text"] = clean(str(item["synthetic_text"]))
    if changed:
        item["target_descriptor_normalized"] = True


def conflicting_target_descriptors(text: str, expected_severity: str) -> list[str]:
    """Return explicit bucket adjectives that contradict the narrative target."""
    levels = {
        match.group(1).capitalize()
        for match in re.finditer(
            r"\b(mild|moderate|severe)\s+(?:postpartum\s+)?"
            r"(?:depress\w*|anxi\w*|symptoms?|distress)\b",
            text or "",
            re.I,
        )
        if match.group(1).lower() != expected_severity.lower()
    }
    return sorted(levels)


def disallowed_content_categories(factors: dict[str, Any]) -> list[str]:
    """Tell the generator which sensitive categories have no factor support."""
    support_text = text_values(trusted_generation_factors(factors))
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
    # 2026-08-27: Qwen3 needs thinking disabled for strict JSON responses.
    if cfg["ollama_thinking"] != "auto":
        payload["think"] = cfg["ollama_thinking"] == "on"
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
        digest = data.get("digest") or data.get("details", {}).get("digest")
        if digest:
            return str(digest)
        with urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout) as response:
            tags = json.loads(response.read().decode("utf-8"))
        for entry in tags.get("models", []):
            if entry.get("name") == model or entry.get("model") == model:
                return str(entry.get("digest") or "") or None
        return None
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
- "Mild": noticeable low/moderate mood disturbance, guilt, anxiety, or intermittent
  distress with limited impairment and preserved basic functioning.
- "Moderate": explicitly high-intensity symptoms, extended or persistent moderate/high
  mood symptoms, rumination, meaningful functional impairment, or bonding difficulty,
  without pervasive major impairment or affirmed urgent risk.
- "Severe": pervasive hopelessness, sustained major impairment, intense anxiety,
  significant bonding disruption, or urgent distress cues. A word such as "severe"
  or "depression" alone is insufficient without high intensity plus major impact,
  negative bonding indicators, pervasive hopelessness, inability to care, or affirmed
  thoughts of harming oneself or the baby. Treat a statement that the baby or family
  would be better off without the writer as safety-relevant passive self-negation.
Timing is context, not a rule. These are narrative buckets, not reconstructed EPDS scores.
"""

SEVERITY_BOUNDARY_RULES = """Boundary rules:
- Minimal: symptoms must be low/transient/resolved and functioning essentially intact.
- Mild: noticeable low/moderate symptoms but no more than limited functional impact.
- Moderate: explicitly high intensity, extended or persistent moderate/high symptoms,
  or explicit meaningful, non-major functional difficulty.
- Severe: pervasive high-intensity distress plus major impairment, serious bonding
  disruption, inability to manage basic care, pervasive hopelessness, or affirmed
  thoughts of harming oneself or the baby. Safety-relevant passive self-negation also
  belongs in Severe even when infant harm is explicitly denied.
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
 "risk_indicators": ["..."], "risk_denials": ["..."],
 "coping_or_adjustment_context": "..."}}

Extraction completeness. Inventing detail is still forbidden, but do not leave a field
empty when the source supports it. Work through the source and record everything it
actually states:
- List every distinct symptom, mood state, or emotional experience separately. Do not
  collapse several into one generic label.
- Assign high intensity only from explicit strong wording, urgent risk, or major stated
  impairment. A duration or the word depression alone does not establish high intensity.
- Treat self-directed beliefs as symptoms when stated, for example guilt, feeling like
  a bad parent, worthlessness, or believing the family would be better off without them.
- Record the content of intrusive or negative thoughts, not just the fact that they occurred.
- Record stated effects on daily functioning, work, or caregiving in functional_impact.
- Record any stated sleep, rest, or exhaustion detail in sleep_context.
- Record stated feeding, milk supply, or infant-care difficulties.
- Record stated help, isolation, partner or family involvement in perceived_support.
- Put affirmed self-harm or infant-harm thoughts only in risk_indicators. Put an
  explicit denial only in risk_denials; never reverse its polarity.
- Record stated treatment, exercise, medication, or self-management in
  coping_or_adjustment_context.
Leave a field empty only when the source genuinely says nothing about it. An empty field
means absent from the source, never merely unmentioned by you.

Return a JSON list with exactly one output per input ID. {JSON_ONLY}
INPUT:
{json.dumps(items, ensure_ascii=False)}"""


def prompt_direct_baseline(
    items: Sequence[dict[str, Any]],
    use_timing: bool,
    min_words: int,
    max_words: int,
    sparse_min_words: int,
    sparse_target_max_words: int,
    sparse_max_words: int,
    use_adaptive_length: bool,
    regeneration_attempt: int = 0,
    correction_attempt: int = 0,
) -> str:
    """One-call generation baseline used as an architectural ablation."""
    timing = TIMING_DEFS + "\n" if use_timing else ""
    factor_timing = ', "postpartum_timing": "one timing bucket"' if use_timing else ""
    narrative_timing = ', "timing": "same exact timing bucket"' if use_timing else ""
    sentence_schema = ", ".join(
        f'"s{index:02d}": "5-15 words"'
        for index in range(1, STAGE3_SENTENCE_COUNT + 1)
    )
    retry_note = ""
    if regeneration_attempt:
        retry_note = (
            f"This is regeneration attempt {regeneration_attempt}. Each affected input may "
            "contain validator_feedback and required_severity/required_timing. Preserve those "
            "required values and correct the blind-rater mismatch without adding facts.\n"
        )
    correction_note = ""
    if correction_attempt > 1:
        correction_note = (
            "A prior response failed strict validation. Recheck every required field, "
            "the exact required timing, the word limit, and the severity boundaries. "
            "Ongoing moderate/high symptoms cannot be Mild; high intensity alone cannot "
            "be Severe without supported major impact, bonding disruption, or risk.\n"
        )
    length_rule = (
        f"When the extracted factors contain at most four supported evidence units, "
        f"use {sparse_min_words}-{sparse_target_max_words} "
        f"assembled words. Otherwise use "
        f"{min_words}-{max_words} words."
        if use_adaptive_length
        else f"Use {min_words}-{max_words} assembled words."
    )
    return f"""Create privacy-preserving synthetic postpartum narratives directly from
source posts. This is a one-prompt baseline: perform abstraction, severity assignment,
and narrative generation in this single response.

{DEID_RULES}
{SEVERITY_DEFS}
{SEVERITY_BOUNDARY_RULES}
{timing}
{retry_note}{correction_note}
For each input, return one object with this exact nested structure:
{{"id": 0,
 "factors": {{"deidentified_summary": "2-3 generalized sentences"{factor_timing},
  "symptoms": ["..."], "symptom_intensity": "low|moderate|high",
  "symptom_persistence": "...", "functional_impact": ["..."],
  "sleep_context": "...", "feeding_or_infant_care_stressors": ["..."],
  "perceived_support": "...", "bonding_indicators": ["..."],
  "risk_indicators": ["..."], "risk_denials": ["..."],
  "coping_or_adjustment_context": "..."}},
 "classification": {{"severity": "Minimal|Mild|Moderate|Severe",
  "rationale": "one grounded sentence"}},
 "narrative": {{"sentences": {{{sentence_schema}}},
  "target": "same exact severity"{narrative_timing},
  "style": "first-person postpartum diary"}}}}

Requirements:
- Return exactly {STAGE3_SENTENCE_COUNT} narrative sentence slots. {length_rule}
  Write naturally in first person using I/my.
- Extract every supported factor, but never infer a missing symptom, impairment, risk,
  relationship, treatment, event, cause, or duration.
- Do not copy or closely paraphrase the source and do not retain identifiers.
- The classification must satisfy the stated severity boundary rules.
- Treat risk_denials only as explicit denials, never as urgent-risk evidence.
- Explicitly compute which labels conflict with the extracted factors before choosing.
  Persistent moderate/high symptoms require at least Moderate. Severe requires high
  intensity plus supported impact, bonding disruption, hopelessness, or urgent risk.
- The narrative must use only its factors and match its classification.
- Preserve symptom and risk tense exactly. Past or unresolved historical factors
  must not become current. Preserve every explicit risk denial as a denial.
- Never expose labels or pipeline metadata in prose: do not say severity, target,
  classification, assessment, audit, record, or named low/moderate/high intensity.
- For known timing, use exactly the compatible broad phrase: within the first two weeks
  after birth; between two and six weeks after birth; between six and twelve weeks after
  birth; or more than three months after birth. For unknown timing, include no time cue.
- If required_severity or required_timing is present, copy it exactly into the output.

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
An entry in risk_denials is evidence that the risk was denied, not an urgent-risk cue.
For each input, return the same integer "id", one exact severity label, and a
one-sentence rationale grounded in the factors:
{{"id": 0, "severity": "Minimal|Mild|Moderate|Severe", "rationale": "..."}}

Each input may contain "disallowed_severity_labels", computed from explicit
factor contradictions. Never return a label listed there; choose the best-supported
remaining label and explain the boundary decision.
Exactly one label remains allowed. Return that label; do not substitute a neighboring
bucket based only on the word "depression" or on help-seeking.

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
    timing = TIMING_DEFS + "\n" if use_timing else ""
    timing_instruction = (
        'Copy "postpartum_timing" exactly into "timing" and make the narrative '
        "consistent with that context. Use exactly one compatible timing cue: "
        '"within the first two weeks after birth" for very early; '
        '"between two and six weeks after birth" for early; '
        '"between six and twelve weeks after birth" for intermediate; or '
        '"more than three months after birth" for later. For unknown timing, do not '
        "mention or imply days, weeks, months, newborn status, 'from the start', "
        "'early on', or other early/later timing.\n"
        if use_timing
        else ""
    )
    timing_field = ', "timing": "exact input postpartum_timing"' if use_timing else ""
    regeneration = ""
    if regeneration_attempt:
        regeneration = (
            f"This is regeneration attempt {regeneration_attempt}. Produce a materially "
            "different composition while preserving only the supplied facts. Each input may "
            "contain validator_feedback. Follow target_expression_rule and correct the stated "
            "strength or timing mismatch without naming a label.\n"
        )
    correction = ""
    if correction_attempt:
        correction = (
            "A prior response failed strict validation. Return the requested sentence count, "
            "word range, metadata, tense, and factual content exactly this time.\n"
        )
    return f"""Rewrite grounded content plans as natural first-person postpartum diary
narratives for research. Stage 3 receives generalized factors only, never source posts.

{timing}
{timing_instruction}{regeneration}{correction}
Constraints:
- Each input contains required_min_words and required_max_words. Its assembled
  narrative must stay inside that exact range. The overall configured ceiling is
  {min_words}-{max_words} words for richer inputs.
- grounded_scaffold is the safe base. Rewrite it lightly for fluency. You may add a
  concrete detail only when it appears explicitly in supported_context or another
  non-empty structured factor. Never add information merely to sound realistic.
- required_fact_sentences is a mandatory content checklist. Represent every entry
  once before adding any restatement; do not drop the last item or a coping/context fact.
- Return "sentences" as a JSON list containing exactly required_sentence_count complete
  sentences. Two to four sentences are allowed. End every string with punctuation and
  begin at least two with "I" or "My".
- Every sentence must express only a fact in grounded_scaffold or another supplied
  non-empty factor. Prefer an unused fact from supported_context over repeating a fact.
  A sparse input may revisit one supported experience without repeating wording.
- Do not use filler such as "I am describing those same emotions," "the same emotional
  difficulty," or repeated uses of "part of my postpartum experience."
- Preserve symptom intensity, temporal status, functional impact, risk polarity, and
  coping/support facts from the scaffold. Do not soften or strengthen them.
- Keep a required timing phrase inside the first content sentence; never place it as a
  standalone sentence. Preserve explicit risk-denial sentences exactly.
- Write as the person, in the first person, using "I" and "my". Never write about her
  from the outside. Do not use "the individual", "the mother", "she", or "they".
- Never narrate the data itself. An empty field means stay silent about that topic, not
  describe it as missing. Do not write that something is unknown or unspecified, and do
  not mention severity, intensity or persistence as named quantities.
- Never expose a label or pipeline concept in prose. Do not use phrases such as
  "severity remains Mild", "classified as", "according to the assessment", "audited
  record", "high-intensity", "stated support", or "adjustment context".
- Avoid clinical or stock prose such as "coping mechanism", "before intervention",
  "the condition", "emotional symptoms", "positive bonding experience",
  "my postpartum experience includes/has been", "I describe", or "part of my
  emotional experience". State supported bonds and feelings directly.
- Keep independent source facts independent. Never say that symptoms or difficulties
  caused an inability to prevent depression unless that causal link is supplied.
- Follow required_status_rule exactly. Ongoing factors use present tense. Historical
  factors use past tense and must not imply current symptoms or risk. Resolved factors
  remain resolved. For unknown persistence, use a simple event statement and do not
  claim that symptoms continue, resolved, or lasted for a particular duration.
- Preserve risk_denials explicitly and only as denials. Never convert a denied thought
  into an affirmed safety concern.
- Follow target_expression_rule for clinical strength, but never quote that rule or
  describe a label, level, assessment, intensity, or severity in the narrative.
- Paraphrase the grounded scaffold enough to be natural, except for locked timing and
  risk-denial sentences. Never copy or closely paraphrase the unavailable source post.
- Use only the supplied factors. Do not invent a partner, infant sex, occupation,
  feeding method, medical event, treatment, or support person unless supplied.
- Each input contains disallowed_content_categories. Do not mention or imply any
  detail from a listed category, even if it is commonly associated with the target.
- Use only the required timing phrase. Do not add another duration or numeric time cue.
- When timing is unknown, the broad phrase "after giving birth" is allowed, but never
  guess early/later timing or write days, weeks, months, newborn, or "from the start".
- Keep the narrative introspective. Do not invent phone calls, visits, appointments,
  conversations, or other concrete events merely to add length.
- Do not add names, handles, contact details, exact dates, locations, or institutions.
- Do not write "Within my postpartum experience," "quality time," "now I see its
  impact," "find a way forward," or an unsupported recent recovery period.

For each input, return:
{{"id": 0, "sentences": ["sentence 1", "sentence 2"]{timing_field},
 "style": "first-person postpartum diary"}}

Return a JSON list with exactly one output per input ID. {JSON_ONLY}
INPUT:
{json.dumps(items, ensure_ascii=False)}"""


def prompt_stage3_length_repair(
    item: dict[str, Any], min_words: int, max_words: int, use_timing: bool
) -> str:
    target_low = min_words
    target_high = min(max_words, max(target_low, 60))
    factor_block = item.get("factors") if isinstance(item.get("factors"), dict) else {}
    sentence_count = min(4, max(2, int(factor_block.get("required_sentence_count", 3))))
    sentence_low = max(
        5, (target_low + sentence_count - 1) // sentence_count
    )
    sentence_high = max(sentence_low, target_high // sentence_count)
    sentence_schema = ", ".join(f'"sentence {index}"' for index in range(1, sentence_count + 1))
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
- Return exactly {sentence_count} ordered strings in the "sentences" list.
- Each field must be a complete sentence containing {sentence_low}-{sentence_high}
  words, for {target_low}-{target_high} words total.
- Do not omit fields or return a short summary.
- Treat grounded_scaffold inside factors as the complete content plan. Rewrite it
  lightly, without adding or omitting clinical facts. Begin at least two sentences
  with "I" or "My".
- Cover every entry in required_fact_sentences exactly once before adding any restatement.
- End every sentence string with punctuation. Use each supported fact before repeating
  one, and do not use "I am describing" or "the same emotional difficulty" as filler.
- Preserve the supplied target and expected timing exactly.
- For severity: {severity_requirement}.
- For timing: {timing_requirement}.
- If expected timing is unknown, "after giving birth" is allowed but no narrower
  birth-relative time, duration, newborn wording, or early/later wording is allowed.
- Preserve supported meaning, but rewrite and expand the composition.
- Follow required_status_rule exactly. Do not change historical, ongoing, resolved,
  or unknown symptom status, and preserve explicit risk denials as denials.
- Do not mention labels, severity, intensity levels, assessments, audits, records,
  missing fields, or unknown data in the narrative.
- Ground every sentence in a supplied factor. Revisit supported internal experience
  when facts are sparse instead of adding symptoms, impairment, relationships, or events.
- Use only supplied factors. Do not invent a partner, infant sex, occupation,
  feeding method, medical event, treatment, or support person unless supplied.
- Keep the narrative introspective; do not invent conversations or concrete events.
- Do not add names, handles, contact details, exact dates, locations, or institutions.

Return one JSON object:
{{"id": {item_id}, "sentences": [{sentence_schema}],
 "target": {json.dumps(target)}{timing_field},
 "style": "first-person postpartum diary"}}

{JSON_ONLY}
INPUT:
{json.dumps(item, ensure_ascii=False)}"""


def prompt_stage3_content_repair(
    item: dict[str, Any], categories: Sequence[str], min_words: int, max_words: int,
    use_timing: bool,
) -> str:
    target_low = min_words
    target_high = min(max_words, max(target_low, 60))
    factor_block = item.get("factors") if isinstance(item.get("factors"), dict) else {}
    sentence_count = min(4, max(2, int(factor_block.get("required_sentence_count", 3))))
    sentence_low = max(
        5, (target_low + sentence_count - 1) // sentence_count
    )
    sentence_high = max(sentence_low, target_high // sentence_count)
    sentence_schema = ", ".join(f'"sentence {index}"' for index in range(1, sentence_count + 1))
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
        "infant-care detail": "Remove infant-care responsibilities unless explicit in the factors.",
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
        "frequency or trajectory detail": (
            "Remove frequency, improvement, worsening, or day-to-day claims unless explicit in the factors."
        ),
        "coping or priority detail": "Remove coping actions or priorities unless explicit in the factors.",
        "causal or benefit detail": "Remove causal or benefit claims unless explicit in the factors.",
        "new emotional appraisal": (
            "Remove hope, gratitude, confidence, pride, relief, or optimism unless explicit in the factors."
        ),
        "expectation or comparison detail": (
            "Remove comparisons with prior expectations unless explicit in the factors."
        ),
        "conflicting symptom descriptor": (
            "Use ordinary emotional wording that matches the supplied symptom intensity."
        ),
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
- Return exactly {sentence_count} ordered strings in the "sentences" list.
- Each field must be a complete sentence containing {sentence_low}-{sentence_high}
  words, producing {target_low}-{target_high} words total.
- Do not omit fields or return a short summary.
- Use grounded_scaffold inside factors as the complete content plan. Begin at least two
  sentences with "I" or "My" and add no fact absent from that scaffold.
- Cover every entry in required_fact_sentences exactly once; do not omit context merely
  to shorten the revision.
- End every sentence string with punctuation and avoid repetitive summary filler.
- Preserve target {json.dumps(target)} and timing {json.dumps(expected_timing)} exactly.
- Preserve supported symptom intensity, persistence, impairment, and timing.
- Follow required_status_rule exactly and preserve every explicit risk denial as a denial.
- Remove label, assessment, audit, record, field-name, and missing-data language.
- Ground every sentence in a supplied factor; do not replace removed facts with new ones.
- Keep the narrative first-person and natural, but prefer introspection over invented events.
- Do not add identifiers, exact dates, locations, institutions, or source-like wording.

Return one JSON object:
{{"id": {item_id}, "sentences": [{sentence_schema}],
 "target": {json.dumps(target)}{timing_field},
 "style": "first-person postpartum diary"}}

{JSON_ONLY}
INPUT:
{json.dumps(item, ensure_ascii=False)}"""


def prompt_stage3_validation_repair(
    item: dict[str, Any], min_words: int, max_words: int, use_timing: bool
) -> str:
    """Give a rejected draft one focused chance to satisfy local checks."""
    target_low = min_words
    target_high = min(max_words, max(target_low, 60))
    factor_block = item.get("factors") if isinstance(item.get("factors"), dict) else {}
    sentence_count = min(4, max(2, int(factor_block.get("required_sentence_count", 3))))
    sentence_low = max(
        6, (target_low + sentence_count - 1) // sentence_count
    )
    sentence_high = max(sentence_low, target_high // sentence_count)
    sentence_schema = ", ".join(f'"sentence {index}"' for index in range(1, sentence_count + 1))
    item_id = int(item["id"])
    target = str(item["target"])
    expected_timing = str(item.get("expected_timing", "unknown"))
    timing_field = f', "timing": {json.dumps(expected_timing)}' if use_timing else ""
    return f"""Repair one rejected synthetic postpartum narrative.
The validation errors identify what must change. Do not repeat those messages in prose.

Requirements:
- Return exactly {sentence_count} ordered strings in the "sentences" list, each containing
  {sentence_low}-{sentence_high} words, for {target_low}-{target_high} words total.
- Treat grounded_scaffold inside factors as the complete content plan. Rewrite only
  that plan, and begin at least two sentences with "I" or "My".
- Cover every entry in required_fact_sentences exactly once. If timing is unknown, use
  only broad "after giving birth" wording and no days, weeks, months, newborn, or early/later cue.
- End every sentence string with punctuation. Use distinct supported facts before
  revisiting one, and avoid formulaic phrases about "describing" the experience.
- Use only the supplied factors. Every sentence must restate a supported symptom,
  status, timing, impact, care issue, bond, risk statement, support, or coping detail.
- Follow required_status_rule exactly and preserve risk denials only as denials.
- Remove unsupported details rather than replacing them with plausible new details.
- Preserve target {json.dumps(target)} and timing {json.dumps(expected_timing)} exactly.
- Do not mention labels, field names, validation, missing information, or records.
- Keep the writing natural, first-person, and free of identifiers or precise new timing.

Return one JSON object:
{{"id": {item_id}, "sentences": [{sentence_schema}],
 "target": {json.dumps(target)}{timing_field},
 "style": "first-person postpartum diary"}}

{JSON_ONLY}
INPUT:
{json.dumps(item, ensure_ascii=False)}"""


def prompt_stage3_grounding_audit(items: Sequence[dict[str, Any]]) -> str:
    return f"""Act as a factual-grounding auditor for synthetic postpartum narratives.
Decide whether the narrative stays inside the facts it was given. Audit every sentence.

THE FACTUAL BOUNDARY is the entire "factors" object, especially grounded_scaffold,
supported_context, and the structured fields. Only information in that object is
supported.

The narrative is SUPPOSED to reword the facts. It is written in the first person and it
will not match the field wording. Do not treat a faithful paraphrase, a plainer phrasing,
a combination of supplied facts, or ordinary wording of supplied intensity, persistence,
or timing as an addition.

Flag a claim ONLY when it introduces information that is genuinely absent from the whole
factors object: a new symptom, a new functional effect, a claim that something is absent
or fine, a relationship or other person, a social action, a coping action, a cause, a
physical condition, a concrete event, or a duration more precise than the supplied timing.
The severity target alone never licenses an extra symptom or impairment. A timing bucket
supports only its required broad timing phrase.
General wording that locates an experience after childbirth is already supported when
the factors explicitly describe a postpartum experience; it is not a new medical event.
A retrospective phrase does not claim that symptoms resolved or remain current unless
the narrative separately says so.
Also flag any change in temporal status or polarity. Historical symptoms or safety
concerns cannot become current; unknown persistence cannot become ongoing or resolved;
and an explicit risk denial cannot become an affirmed thought. Label, audit, assessment,
record, field-name, and missing-data language is never acceptable diary prose.

EMPTY FIELDS ARE SILENCE, NOT PERMISSION. An empty list or empty string means the source
said nothing on that topic, so any claim about that topic is unsupported. Apply this directly:
- sleep_context empty: any mention of sleep, rest, tiredness or nights is unsupported
- functional_impact empty: any claim about what they can or cannot manage is unsupported
- perceived_support empty: any mention of help, family, partner, or being alone is unsupported
- feeding_or_infant_care_stressors empty: any feeding or infant-care difficulty is unsupported
- bonding_indicators empty: any claim about closeness or distance from the baby is unsupported
- risk_indicators empty: any mention of self-harm or harm to the baby is unsupported
- risk_denials non-empty: preserve the denial; any affirmed version is unsupported
This holds no matter how plausible the claim is for someone with the supplied symptoms.

Before flagging a claim, search the whole factors object for anything it could be
restating. If you find a match, it is supported. Only flag what you could not match.

Return one object per input ID:
{{"id": 0, "grounded": true, "unsupported_claims": [],
 "reason": "brief sentence-level assessment"}}

unsupported_claims must contain ONLY exact verbatim sentences or minimal exact phrases
copied from the supplied narrative. Never paraphrase a claim and never quote an example
or wording from these instructions. If a claim is supported, leave it out entirely.
Set grounded to false if and only if unsupported_claims is non-empty. For each entry,
name the new information it introduces in "reason". Return a JSON list with one object
per input ID. {JSON_ONLY}
INPUT:
{json.dumps(items, ensure_ascii=False)}"""


def prompt_stage3_semantic_repair(
    item: dict[str, Any], min_words: int, max_words: int, use_timing: bool
) -> str:
    target_low = min_words
    target_high = min(max_words, max(target_low, 60))
    factor_block = item.get("factors") if isinstance(item.get("factors"), dict) else {}
    sentence_count = min(4, max(2, int(factor_block.get("required_sentence_count", 3))))
    sentence_low = max(
        5, (target_low + sentence_count - 1) // sentence_count
    )
    sentence_high = max(sentence_low, target_high // sentence_count)
    sentence_schema = ", ".join(f'"sentence {index}"' for index in range(1, sentence_count + 1))
    item_id = int(item["id"])
    target = str(item["target"])
    expected_timing = str(item.get("expected_timing", "unknown"))
    timing_field = f', "timing": {json.dumps(expected_timing)}' if use_timing else ""
    return f"""Rewrite one synthetic postpartum narrative after a strict grounding audit.
The factors are the entire factual boundary. Remove every unsupported claim listed by
the auditor. Do not replace it with another fact.

Requirements:
- Return exactly {sentence_count} ordered strings in the "sentences" list, each with
  {sentence_low}-{sentence_high} words, for {target_low}-{target_high} words total.
- grounded_scaffold inside factors is the complete replacement content plan. Rewrite
  only that plan and begin at least two sentences with "I" or "My".
- End every sentence string with punctuation and avoid formulaic repeated content.
- Every sentence must be directly supported by at least one supplied factor.
- When factors are sparse, revisit the supported internal experience in plain language.
- Do not add symptoms, impairment, normal functioning, relationships, actions, causes,
  physical conditions, events, or duration.
- Preserve required_status_rule exactly, including the tense and polarity of any risk
  statement. Never expose severity labels, assessments, audits, or field names in prose.
- Preserve the target and required broad timing phrase exactly.
- Keep only the required timing phrase; add no second time or duration cue.

Return one JSON object:
{{"id": {item_id}, "sentences": [{sentence_schema}],
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
        "because the writer has an infant or says only after giving birth/postpartum.\n"
        if use_timing
        else ""
    )
    return f"""Act as an independent blind rater. Assess the diary entry on its
own. You are intentionally not shown the intended severity or timing.

{SEVERITY_DEFS}
{SEVERITY_BOUNDARY_RULES}
{timing}
First extract only evidence explicitly present in the diary. Then give your own raw
severity judgment. Mark persistent_or_extended true for ongoing, continuing, still
present, or extended-duration symptoms. Mark mood_symptoms_present true when the text
names depression, anxiety, panic, intrusive thoughts, hopelessness, or comparable mood
distress. Mark affirmed harm only for a non-negated thought of self-harm or infant harm.
Mark passive_self_negation_or_pervasive_hopelessness true for pervasive hopelessness or
a statement that the baby, family, or others would be better off without the writer.
An infant-harm denial does not negate a separate self-directed safety concern.

Use high intensity for explicit intense/severe/extreme distress or a safety concern;
moderate for explicitly difficult/significant distress; low for explicit mild distress;
otherwise use unknown. Historical or resolved wording describes status, not lower
episode intensity.
{timing_decision}
Return one object:
{{"predicted_severity": "Minimal|Mild|Moderate|Severe"{timing_field},
 "evidence_profile": {{"mood_symptoms_present": true,
  "symptom_intensity": "low|moderate|high|unknown",
  "persistent_or_extended": false,
  "meaningful_impairment": false,
  "major_impairment_or_inability_basic_care": false,
  "serious_bonding_disruption": false,
  "affirmed_self_or_infant_harm": false,
  "passive_self_negation_or_pervasive_hopelessness": false}},
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
    item.setdefault("risk_denials", [])
    for key in (
        "symptoms", "functional_impact", "feeding_or_infant_care_stressors",
        "bonding_indicators", "risk_indicators", "risk_denials",
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
    item["supported_context"] = source_supported_context(source_text)
    item["symptoms"] = source_grounded_values(item.get("symptoms", []), source_text)
    item["symptoms"] = [
        value
        for value in item["symptoms"]
        if not re.search(
            r"\b(?:bad (?:mom|mother|parent)|not (?:having|have) enough breast ?milk|"
            r"better off without me)\b",
            value,
            re.I,
        )
    ]
    if re.search(r"\b(?:told|said|advised)\b.{0,45}\bbaby blues\b", source_lower):
        item["symptoms"] = [
            value for value in item["symptoms"]
            if not re.search(r"\bbaby blues\b", value, re.I)
        ]
    canonical_symptoms = (
        ("depression", r"\bdepress\w*\b"),
        ("anxiety", r"\banxi\w*\b"),
        ("intrusive thoughts", r"\bintrusive thoughts?\b"),
        ("panic attacks", r"\bpanic(?: attacks?)?\b"),
        ("hopelessness", r"\bhopeless\w*\b"),
        ("guilt", r"\bguilt\w*\b"),
        ("irritability", r"\birritab\w*\b"),
        ("sleep difficulty", r"\b(?:insomnia|cannot sleep|can't sleep)\b"),
    )
    symptom_text = " ".join(str(value).lower() for value in item["symptoms"])
    for label, pattern in canonical_symptoms:
        if re.search(pattern, source_lower, re.I) and not re.search(pattern, symptom_text, re.I):
            item["symptoms"].append(label)
            symptom_text += " " + label
    if item["symptom_persistence"].startswith("past,") or item["symptom_persistence"] == "resolved":
        cleaned_summary = str(item.get("deidentified_summary") or "")
        cleaned_summary = re.sub(r"\bis currently experiencing\b", "experienced", cleaned_summary, flags=re.I)
        cleaned_summary = re.sub(r"\bis experiencing\b", "experienced", cleaned_summary, flags=re.I)
        cleaned_summary = re.sub(r"\bis suffering from\b", "experienced", cleaned_summary, flags=re.I)
        cleaned_summary = re.sub(r"\bcurrently\b\s*", "", cleaned_summary, flags=re.I)
        item["deidentified_summary"] = re.sub(r"\s+", " ", cleaned_summary).strip()
    if not re.search(
        r"\b(?:function\w*|daily (?:life|activities|tasks)|work\w*|job|care(?:giving)?|"
        r"look after|handle|manage|unable|cannot|can't|hard to|tough (?:for me )?to|"
        r"get out of bed|chores?|responsibilit\w*)\b",
        source_lower,
    ):
        item["functional_impact"] = []
    else:
        item["functional_impact"] = source_grounded_values(
            item["functional_impact"], source_text
        )
    if not re.search(
        r"\b(?:feed\w*|breast ?milk|milk supply|nurs\w*|formula|bottle|baby duty|"
        r"care for (?:my |the )?(?:baby|infant|child))\b",
        source_lower,
    ):
        item["feeding_or_infant_care_stressors"] = []
    else:
        item["feeding_or_infant_care_stressors"] = source_grounded_values(
            item["feeding_or_infant_care_stressors"], source_text
        )
    if not re.search(r"\b(?:sleep\w*|awake|insomnia|rest\w*|tired\w*|exhaust\w*)\b", source_lower):
        item["sleep_context"] = ""
    elif not source_grounded_values([item["sleep_context"]], source_text):
        item["sleep_context"] = ""
    # Trust only source patterns below, never a free-form model coping inference.
    item["coping_or_adjustment_context"] = ""
    # Risk fields must come from affirmative source wording, not model inference.
    item["risk_indicators"] = []
    item["risk_denials"] = []
    if source_has_affirmed_risk(source_lower, _SELF_HARM_PATTERN):
        item["risk_indicators"].append("thoughts of self-harm")
    if source_has_affirmed_risk(source_lower, _INFANT_HARM_PATTERN):
        item["risk_indicators"].append("thoughts of harming my baby")
    if source_has_denied_risk(source_lower, _SELF_HARM_PATTERN):
        item["risk_denials"].append("denied thoughts of self-harm")
    if source_has_denied_risk(source_lower, _INFANT_HARM_PATTERN):
        item["risk_denials"].append("denied thoughts of harming my baby")
    item["symptom_intensity"] = source_supported_intensity(
        source_text, item["risk_indicators"]
    )
    item["symptoms"] = [
        re.sub(
            r"^\s*(?:mild|moderate|severe|intense)\s+(?=(?:postpartum\s+)?"
            r"(?:depress\w*|anxi\w*|symptoms?|distress)\b)",
            "",
            str(value),
            flags=re.I,
        ).strip()
        for value in item["symptoms"]
    ]
    if re.search(r"\b(?:i|we) (?:just )?need(?:ed)? (?:some )?time\b", source_lower):
        item["coping_or_adjustment_context"] = "needs time to adjust"
    elif re.search(
        r"\b(?:tried to seek|sought) help\b.{0,80}\b(?:told|advised)\b.{0,35}\bwait\b",
        source_lower,
    ):
        item["coping_or_adjustment_context"] = "sought help but was told to wait"
    elif re.search(
        r"\b(?:saw|visited) (?:a )?(?:therapist|psychologist|counselor)\b", source_lower
    ) and re.search(r"\b(?:did not|didn'?t) (?:make|schedule|attend|return)\b", source_lower):
        if re.search(
            r"\b(?:told|said)\b.{0,45}\b(?:did not|didn'?t) have (?:ppd|postpartum depression)\b",
            source_lower,
        ):
            item["coping_or_adjustment_context"] = (
                "sought therapy and did not return after the depression was dismissed"
            )
        else:
            item["coping_or_adjustment_context"] = "sought therapy and did not return"
    elif re.search(r"\breached out to (?:a )?(?:psychologist|therapist|counselor)\b", source_lower):
        item["coping_or_adjustment_context"] = "reached out to a mental-health professional"
    elif re.search(
        r"\b(?:pediatrician|clinician|doctor)\b.{0,45}\b"
        r"(?:caught|identified|recognized|noticed)\b.{0,25}\bdepress\w*",
        source_lower,
    ):
        if re.search(r"\bcheck\s*up\b", source_lower):
            item["coping_or_adjustment_context"] = (
                "a clinician recognized the depression during an infant checkup"
            )
        else:
            item["coping_or_adjustment_context"] = "a clinician recognized the depression"
    elif re.search(r"\bexercis\w*(?:\s+more)?\s+consistently\b", source_lower):
        if re.search(r"\b(?:help\w*|improv\w*|difference)\b", source_lower):
            item["coping_or_adjustment_context"] = (
                "used consistent exercise and noticed some improvement"
            )
        else:
            item["coping_or_adjustment_context"] = "used consistent exercise as part of coping"
    elif re.search(r"\b(?:get|getting) back (?:in|to) the gym\b", source_lower):
        item["coping_or_adjustment_context"] = "wanted to resume exercise"
    elif re.search(
        r"\b(?:hormones?|hormonal)\b.{0,80}\bdepress\w*|"
        r"\bdepress\w*\b.{0,80}\b(?:hormones?|hormonal)\b",
        source_lower,
    ):
        item["coping_or_adjustment_context"] = (
            "associated the depression with hormonal changes"
        )
    elif re.search(
        r"\b(?:cut(?:ting)? back on|reduc(?:e|ed|ing)|stopp(?:ed|ing)|limit(?:ed|ing)?)\b.{0,30}"
        r"\b(?:alcohol|drink(?:ing)?)\b",
        source_lower,
    ):
        item["coping_or_adjustment_context"] = "reduced alcohol consumption"
    elif re.search(
        r"\beverything stopped long enough\b.{0,45}\bcare (?:for|of) (?:her|him|them|"
        r"my (?:baby|infant|child))\b",
        source_lower,
    ):
        item["coping_or_adjustment_context"] = (
            "when the baby cried, depression paused long enough to care for the baby"
            if re.search(r"\bbaby cried\b", source_lower)
            else "depression paused long enough to care for the baby"
        )
    negative_bond = re.search(
        r"\b(?:difficulty bonding|cannot bond|can't bond|detached|disconnected|no bond)\b",
        source_lower,
    )
    positive_bond = re.search(
        r"\b(?:bond\w*|connect\w*|attach\w*|love\w*)\b.{0,30}\b(?:baby|infant|child)\b",
        source_lower,
    )
    positive_bond = positive_bond or re.search(
        r"\b(?:baby|infant|child)\b.{0,35}\b(?:best thing|love\w*|bond\w*)\b",
        source_lower,
    )
    held_baby = re.search(
        r"\b(?:able to|could)\s+hold\s+my\s+(?:baby|infant|child)\b",
        source_lower,
    )
    interacted_with_baby = re.search(
        r"\b(?:able to|could)\s+(?:interact with|play with)\s+my\s+"
        r"(?:baby|infant|child)\b",
        source_lower,
    )
    interaction_used_to_dismiss = bool(
        interacted_with_baby
        and re.search(r"\b(?:told|said)\b.{0,55}\b(?:did not|didn'?t) have\b", source_lower)
    )
    if negative_bond:
        item["bonding_indicators"] = ["difficulty bonding"]
    elif positive_bond:
        item["bonding_indicators"] = ["positive bonding experience"]
    elif held_baby:
        item["bonding_indicators"] = ["able to hold baby"]
    elif interacted_with_baby and not interaction_used_to_dismiss:
        item["bonding_indicators"] = ["able to interact with baby"]
    else:
        item["bonding_indicators"] = []
    item["perceived_support"] = ""
    if re.search(
        r"\b(?:did not|didn'?t|do not|don't|without) (?:have |having )?(?:enough |any )?support\b|"
        r"\b(?:lack(?:ed|ing)?|little|no) support\b",
        source_lower,
    ):
        item["perceived_support"] = "limited support"
    elif re.search(
        r"\b(?:friends? and family|family and friends?)\b.{0,35}\bsupport\w*|"
        r"\bsupport\w*\b.{0,35}\b(?:friends? and family|family and friends?)\b",
        source_lower,
    ):
        item["perceived_support"] = "strong support from friends and family"
    if item.get("symptom_intensity") not in INTENSITIES:
        return False, f"invalid symptom_intensity: {item.get('symptom_intensity')!r}"
    if use_timing:
        item["postpartum_timing"] = source_timing_bucket(source_text)
        if item["postpartum_timing"] == "unknown":
            summary = str(item.get("deidentified_summary", ""))
            summary = re.sub(
                r"\b(?:within|during|around|about|approximately|several|a few|"
                r"a couple of?|one|two|three|four|\d+)\s+"
                r"(?:days?|weeks?|months?)\s+(?:postpartum|after (?:giving )?birth)\b",
                "after giving birth",
                summary,
                flags=re.I,
            )
            item["deidentified_summary"] = summary
    privacy_text = text_values({key: value for key, value in item.items() if key != "id"})
    leaks = identifier_leaks(privacy_text)
    if leaks:
        return False, "identifier leak in structured factors: " + ",".join(leaks)
    overlap = shared_ngram_count(source_text, privacy_text, copy_ngram_size)
    if overlap > max_copy_ngrams:
        return False, f"source-copy overlap: {overlap} shared {copy_ngram_size}-grams"
    return True, ""


def expected_factor_severity(factors: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    """Apply one deterministic boundary order to the structured Stage-1 evidence."""
    intensity = str(factors.get("symptom_intensity", "")).strip().lower()
    persistence = str(factors.get("symptom_persistence", "")).strip().lower()
    impacts = [
        str(value).strip()
        for value in factors.get("functional_impact", [])
        if str(value).strip() and str(value).strip().lower() not in {"none", "unknown", "not specified"}
    ]
    severity_factors = {
        key: value
        for key, value in trusted_generation_factors(factors).items()
        if key != "risk_denials"
    }
    factor_text = text_values(severity_factors).lower()
    extended_duration = "depression lasted for an extended period" in {
        str(value).strip().lower()
        for value in factors.get("supported_context", [])
    }
    persistent = extended_duration or bool(
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
    affirmed_risk = any(
        bool(str(value).strip()) for value in factors.get("risk_indicators", [])
    )
    urgent_or_hopeless = affirmed_risk or bool(
        re.search(
            r"\b(?:suicid\w*|self[- ]?harm|hopeless\w*|unable to care|cannot care|"
            r"can't care|urgent|emergency|psychosis|hallucinat\w*|"
            r"(?:baby|infant|child|family) (?:will|would|might|may) be better off without)\b",
            factor_text,
        )
    )
    major_impact = bool(re.search(
        r"\b(?:unable|cannot|can't)\b.{0,30}\b(?:function|basic care|care for|daily tasks)|"
        r"\bmajor (?:functional )?impairment\b",
        " ".join(impacts),
        re.I,
    ))
    severe_required = intensity == "high" and bool(
        major_impact or negative_bonding or urgent_or_hopeless
    )
    moderate_required = not severe_required and bool(
        intensity == "high"
        or moderate_persistence
        or impacts
        or negative_bonding
        or urgent_or_hopeless
    )
    symptoms_present = any(
        bool(str(value).strip()) for value in factors.get("symptoms", [])
    )
    clinical_mood_symptoms = bool(re.search(
        r"\b(?:depress\w*|anxi\w*|panic\w*|intrusive thoughts?|hopeless\w*|"
        r"worthless\w*|guilt\w*|baby blues)\b",
        text_values(factors.get("symptoms", [])).lower(),
        re.I,
    ))
    expected = (
        "Severe" if severe_required
        else "Moderate" if moderate_required
        else "Mild" if clinical_mood_symptoms or intensity == "moderate"
        else "Minimal"
    )
    return expected, {
        "persistent": persistent,
        "extended_duration": extended_duration,
        "negative_bonding": negative_bonding,
        "affirmed_risk": affirmed_risk,
        "urgent_or_hopeless": urgent_or_hopeless,
        "major_impact": major_impact,
        "symptoms_present": symptoms_present,
        "clinical_mood_symptoms": clinical_mood_symptoms,
    }


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
    expected, _ = expected_factor_severity(factors)
    if item["severity"] != expected:
        return False, f"severity {item['severity']!r} conflicts with required boundary {expected!r}"
    return True, ""


def disallowed_severity_labels(factors: dict[str, Any]) -> list[str]:
    """Return boundary labels contradicted by the structured factors."""
    expected, _ = expected_factor_severity(factors)
    return [label for label in EPDS_LABELS if label != expected]


def validate_stage3(
    item: Any,
    *,
    factors: dict[str, Any],
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
    normalize_stage3_language(item, factors)
    if item.get("grounding_method") != "deterministic_factor_fallback":
        slot_values = stage3_sentence_values(item)
        if not 2 <= len(slot_values) <= 4:
            return False, "expected 2-4 complete sentence slots"
        remove_unsourced_severity_descriptors(item, factors)
        normalize_target_severity_descriptors(item, expected_severity)
    if use_timing:
        normalize_stage3_timing(item, expected_timing)
    text = materialize_stage3_text(item)
    if not text.strip():
        return False, "missing synthetic_text"
    if item.get("target") is None or item.get("target") == "":
        item["target"] = expected_severity
        item["target_metadata_injected"] = True
    if item.get("target") != expected_severity:
        return False, f"target mismatch: {item.get('target')!r}"
    if use_timing:
        normalized_timing = normalize_timing(item.get("timing"))
        if normalized_timing is None and expected_timing in TIMING_SET:
            # Metadata can be restored; narrative timing is still checked below.
            normalized_timing = expected_timing
            item["timing"] = expected_timing
            item["timing_metadata_injected"] = True
        if normalized_timing != expected_timing:
            return False, f"timing mismatch: {item.get('timing')!r}"
        item["timing"] = normalized_timing
        timing_ok, timing_error = validate_narrative_timing(text, expected_timing)
        if (
            not timing_ok
            and expected_timing != "unknown"
            and timing_error.startswith("narrative lacks an explicit cue")
        ):
            text = add_timing_opener(text, expected_timing)
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
    conflicts = conflicting_target_descriptors(text, expected_severity)
    if conflicts:
        return False, "narrative uses a conflicting severity descriptor: " + ",".join(conflicts)
    quality_violations = narrative_quality_violations(item, text, factors)
    if quality_violations:
        return False, "narrative quality: " + ",".join(quality_violations)
    return True, ""


def validate_stage4(item: Any, *, use_timing: bool) -> tuple[bool, str]:
    if not isinstance(item, dict):
        return False, "not an object"
    if item.get("predicted_severity") not in EPDS_SET:
        return False, f"invalid predicted_severity: {item.get('predicted_severity')!r}"
    profile = item.get("evidence_profile")
    if not isinstance(profile, dict):
        return False, "missing evidence_profile"
    if profile.get("symptom_intensity") not in {"low", "moderate", "high", "unknown"}:
        return False, f"invalid evidence symptom_intensity: {profile.get('symptom_intensity')!r}"
    for key in (
        "mood_symptoms_present",
        "persistent_or_extended",
        "meaningful_impairment",
        "major_impairment_or_inability_basic_care",
        "serious_bonding_disruption",
        "affirmed_self_or_infant_harm",
        "passive_self_negation_or_pervasive_hopelessness",
    ):
        if not isinstance(profile.get(key), bool):
            return False, f"evidence_profile.{key} must be boolean"
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
        active = [key for key, value in profile.items() if value is True]
        item["evidence"] = (
            f"Blind evidence: intensity={profile['symptom_intensity']}; "
            + (", ".join(active) if active else "no positive evidence flags")
        )
    return True, ""


def audit_claim_is_narrative_span(claim: str, narrative: str) -> bool:
    """Reject auditor claims copied from its prompt or invented during review."""
    def comparable(value: str) -> str:
        value = value.lower().replace("\u2019", "'").replace("\u2018", "'")
        value = re.sub(r"\s+", " ", value).strip()
        return value.strip(" \t\r\n\"'.,;:!?()[]{}")

    claim_value = comparable(claim)
    narrative_value = comparable(narrative)
    return word_count(claim_value) >= 3 and claim_value in narrative_value


def validate_grounding_audit(
    item: Any, *, expected_id: int, narrative: str
) -> tuple[bool, str]:
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
    unanchored = [
        claim for claim in normalized_claims
        if not audit_claim_is_narrative_span(claim, narrative)
    ]
    if unanchored:
        return False, "unsupported_claims must be exact spans from the narrative"
    if grounded and normalized_claims:
        grounded = False
        item["grounded"] = False
    if not grounded and not normalized_claims:
        return False, "ungrounded assessment requires an exact narrative span"
    reason = item.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "; ".join(normalized_claims) or "No unsupported claim was identified."
        item["reason"] = reason
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
    if cfg.get("use_single_retry_pass", True) and len(ids) > 1:
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

    def make_prompt(group_ids: Sequence[int], schema_attempt: int) -> str:
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
        "symptoms",
        "symptom_intensity",
        "symptom_persistence",
        "functional_impact",
        "sleep_context",
        "feeding_or_infant_care_stressors",
        "perceived_support",
        "bonding_indicators",
        "risk_indicators",
        "risk_denials",
        "coping_or_adjustment_context",
        "supported_context",
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

    def make_prompt(group_ids: Sequence[int], schema_attempt: int) -> str:
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


def validate_direct_baseline(
    item: Any,
    *,
    expected_id: int,
    source_text: str,
    cfg: dict[str, Any],
    required_severity: str = "",
    required_timing: str = "",
) -> tuple[bool, str]:
    if not isinstance(item, dict) or item.get("id") != expected_id:
        return False, "direct schema: missing or mismatched id"
    if not all(isinstance(item.get(key), dict) for key in ("factors", "classification", "narrative")):
        return False, "direct schema: factors, classification, and narrative must be objects"

    factors = dict(item["factors"])
    factors["id"] = expected_id
    ok, error = validate_stage1(
        factors,
        expected_id=expected_id,
        source_text=source_text,
        use_timing=cfg["use_timing"],
        copy_ngram_size=cfg["copy_ngram_size"],
        max_copy_ngrams=cfg["max_copy_ngrams"],
    )
    if not ok:
        return False, "stage1: " + error

    classification = dict(item["classification"])
    classification["id"] = expected_id
    ok, error = validate_stage2(
        classification, expected_id=expected_id, factors=factors
    )
    if not ok:
        allowed = [
            label for label in EPDS_LABELS
            if label not in disallowed_severity_labels(factors)
        ]
        if len(allowed) == 1:
            # Boundary rules can resolve an otherwise contradictory direct label.
            classification["severity"] = allowed[0]
            classification["rationale"] = (
                "Deterministic boundary correction from the extracted factors."
            )
            ok, error = validate_stage2(
                classification, expected_id=expected_id, factors=factors
            )
    if not ok:
        return False, "stage2: " + error
    if required_severity and classification.get("severity") != required_severity:
        return False, f"stage2: required severity mismatch: {classification.get('severity')!r}"

    narrative = dict(item["narrative"])
    narrative["id"] = expected_id
    narrative["target"] = classification["severity"]
    expected_timing = str(factors.get("postpartum_timing", "unknown"))
    if required_timing and expected_timing != required_timing:
        return False, f"stage1: required timing mismatch: {expected_timing!r}"
    item_min, item_max = narrative_word_bounds(factors, cfg)
    ok, error = validate_stage3(
        narrative,
        factors=factors,
        expected_id=expected_id,
        expected_severity=str(classification["severity"]),
        expected_timing=expected_timing,
        source_text=source_text,
        use_timing=cfg["use_timing"],
        min_words=item_min,
        max_words=item_max,
        copy_ngram_size=cfg["copy_ngram_size"],
        max_copy_ngrams=cfg["max_copy_ngrams"],
    )
    if not ok:
        return False, "stage3: " + error
    unsupported = unsupported_detail_categories(
        str(narrative.get("synthetic_text", "")), factors
    )
    if unsupported:
        return False, "stage3: unsupported detail: " + ",".join(unsupported)

    item["factors"] = factors
    item["classification"] = classification
    item["narrative"] = narrative
    return True, ""


def generate_direct_baseline(
    *,
    ids: Sequence[int],
    records: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
    usage: Usage,
    regeneration_attempt: int = 0,
    validator_feedback: dict[int, dict[str, Any]] | None = None,
    required_targets: dict[int, tuple[str, str]] | None = None,
) -> dict[int, ItemResult]:
    """Generate factors, severity, and narrative together in one model call."""
    def make_prompt(group_ids: Sequence[int], schema_attempt: int) -> str:
        payloads = []
        for item_id in group_ids:
            payload: dict[str, Any] = {
                "id": item_id,
                "post": records[item_id]["source_text"],
            }
            if cfg["use_timing"]:
                # 2026-08-29: anchor timing to explicit birth-relative wording.
                payload["required_timing"] = source_timing_bucket(
                    records[item_id]["source_text"]
                )
            if validator_feedback and item_id in validator_feedback:
                payload["validator_feedback"] = validator_feedback[item_id]
            if required_targets and item_id in required_targets:
                severity, timing = required_targets[item_id]
                payload["required_severity"] = severity
                if cfg["use_timing"]:
                    payload["required_timing"] = timing
            payloads.append(payload)
        return prompt_direct_baseline(
            payloads,
            cfg["use_timing"],
            cfg["min_words"],
            cfg["max_words"],
            cfg["sparse_min_words"],
            cfg["sparse_target_max_words"],
            cfg["sparse_max_words"],
            cfg["use_adaptive_length"],
            regeneration_attempt,
            schema_attempt,
        )

    def validator(item: dict[str, Any], item_id: int) -> tuple[bool, str]:
        required_severity = ""
        required_timing = (
            source_timing_bucket(records[item_id]["source_text"])
            if cfg["use_timing"] else ""
        )
        if required_targets and item_id in required_targets:
            required_severity, required_timing = required_targets[item_id]
        return validate_direct_baseline(
            item,
            expected_id=item_id,
            source_text=records[item_id]["source_text"],
            cfg=cfg,
            required_severity=required_severity,
            required_timing=required_timing,
        )

    direct_cfg = {
        **cfg,
        "schema_retries": 1,
        "use_single_retry_pass": False,
    }
    return run_group_stage(
        ids=ids,
        make_prompt=make_prompt,
        validator=validator,
        cfg=direct_cfg,
        temperature=cfg["temp_s3"],
        tokens_per_item=cfg["tokens_direct_per_item"],
        stage_number=100 + regeneration_attempt * 10,
        usage=usage,
    )


def split_direct_results(
    ids: Sequence[int], direct: dict[int, ItemResult]
) -> tuple[dict[int, ItemResult], dict[int, ItemResult], dict[int, ItemResult]]:
    s1: dict[int, ItemResult] = {}
    s2: dict[int, ItemResult] = {}
    s3: dict[int, ItemResult] = {}
    for item_id in ids:
        result = direct[item_id]
        if result.status == "ok" and result.value is not None:
            common = {"status": "ok", "attempts": result.attempts, "errors": list(result.errors)}
            s1[item_id] = ItemResult(value=result.value["factors"], **common)
            s2[item_id] = ItemResult(value=result.value["classification"], **common)
            s3[item_id] = ItemResult(value=result.value["narrative"], **common)
        else:
            error = "direct generation failed: " + last_error(result)
            s1[item_id] = ItemResult(
                last_candidate=result.last_candidate,
                status="failed",
                attempts=result.attempts,
                errors=[error],
            )
            s2[item_id] = ItemResult(status="failed", attempts=result.attempts, errors=[error])
            s3[item_id] = ItemResult(status="failed", attempts=result.attempts, errors=[error])
    return s1, s2, s3


def audit_direct_grounding(
    *,
    ids: Sequence[int],
    s1: dict[int, ItemResult],
    s2: dict[int, ItemResult],
    s3: dict[int, ItemResult],
    cfg: dict[str, Any],
    usage: Usage,
    regeneration_attempt: int = 0,
) -> None:
    active = [
        item_id for item_id in ids
        if s1[item_id].status == s2[item_id].status == s3[item_id].status == "ok"
    ]
    if not active:
        return

    def make_prompt(group_ids: Sequence[int], _: int) -> str:
        payloads = []
        for item_id in group_ids:
            payloads.append(
                {
                    "id": item_id,
                    "factors": generation_payload(
                        item_id,
                        s1[item_id].value or {},
                        s2[item_id].value or {},
                        cfg["use_timing"],
                    ),
                    "required_timing": (s1[item_id].value or {}).get(
                        "postpartum_timing", "unknown"
                    ),
                    "narrative": (s3[item_id].value or {}).get("synthetic_text", ""),
                }
            )
        return prompt_stage3_grounding_audit(payloads)

    audits = run_group_stage(
        ids=active,
        make_prompt=make_prompt,
        validator=lambda item, item_id: validate_grounding_audit(
            item,
            expected_id=item_id,
            narrative=str((s3[item_id].value or {}).get("synthetic_text", "")),
        ),
        cfg={**cfg, "schema_retries": max(2, cfg["schema_retries"])},
        temperature=0.0,
        tokens_per_item=cfg["tokens_s4_per_item"],
        stage_number=110 + regeneration_attempt * 10,
        usage=usage,
    )
    for item_id in active:
        audit = audits[item_id]
        generated = s3[item_id]
        if audit.status != "ok" or audit.value is None:
            generated.status = "failed"
            generated.errors.append("direct grounding audit failed: " + last_error(audit))
            continue
        assessment = reconcile_grounding_assessment(
            audit.value, s1[item_id].value or {}
        )
        history = [
            {
                "round": f"direct_{regeneration_attempt}",
                "grounded": assessment["grounded"],
                "unsupported_claims": assessment["unsupported_claims"],
                "reconciled_supported_claims": assessment.get(
                    "reconciled_supported_claims", []
                ),
                "reason": assessment["reason"],
            }
        ]
        if not assessment["grounded"]:
            generated.status = "failed"
            generated.errors.append(
                "direct grounding rejected: " + "; ".join(assessment["unsupported_claims"])
            )
            generated.last_candidate = generated.value
            generated.value = None
            continue
        generated.value["grounding_passed"] = True
        generated.value["grounding_attempts"] = audit.attempts
        generated.value["grounding_history"] = history


def generation_payload(
    item_id: int,
    factors: dict[str, Any],
    severity: dict[str, Any],
    use_timing: bool,
) -> dict[str, Any]:
    payload = factor_payload(item_id, factors, use_timing)
    payload["severity"] = severity["severity"]
    payload["disallowed_content_categories"] = disallowed_content_categories(factors)
    persistence = str(factors.get("symptom_persistence") or "").strip().lower()
    if persistence == "ongoing":
        status_rule = (
            "Use present/ongoing tense and explicitly write either 'I continue to' or "
            "'I am still'; do not imply that symptoms resolved."
        )
    elif persistence.startswith("past,"):
        status_rule = "Use past tense only; do not imply that symptoms or safety concerns are current or resolved."
    elif persistence in {"resolved", "improved", "recovered", "no longer present"}:
        status_rule = "Use past tense and preserve that symptoms improved or resolved; do not imply current symptoms."
    else:
        status_rule = (
            "Use simple event wording without claiming that symptoms continue, resolved, "
            "or lasted for a particular duration."
        )
    if factors.get("risk_denials"):
        status_rule += " Preserve every risk denial explicitly and only as a denial."
    payload["required_status_rule"] = status_rule
    return payload


def scaffolded_generation_payload(
    item_id: int,
    factors: dict[str, Any],
    severity: dict[str, Any],
    use_timing: bool,
    min_words: int,
    max_words: int,
) -> dict[str, Any]:
    """Give Stage 3 a grounded content plan before it performs surface rewriting."""
    payload = generation_payload(item_id, factors, severity, use_timing)
    target = str(severity.get("severity", ""))
    payload.pop("severity", None)
    payload["target_expression_rule"] = {
        "Minimal": "Use brief, limited distress wording and do not imply impairment.",
        "Mild": "Use low but evident emotional difficulty without adding persistence or impairment.",
        "Moderate": (
            "Use meaningful emotional difficulty; preserve persistence or impairment only "
            "when the grounded scaffold contains it."
        ),
        "Severe": (
            "Preserve the scaffold's intense distress and any explicit major impact, "
            "bonding disruption, or risk statement."
        ),
    }.get(target, "Follow the grounded scaffold without changing its clinical strength.")
    scaffold = conservative_factor_narrative(
        factors,
        str(severity.get("severity", "")),
        use_timing,
        min_words,
        max_words,
        item_id,
    )
    scaffold_text = str(scaffold.get("synthetic_text", "")).strip()
    scaffold_slots = stage3_sentence_values({"synthetic_text": scaffold_text})
    payload["grounded_scaffold"] = scaffold_text
    payload["required_fact_sentences"] = scaffold_slots
    payload["required_sentence_count"] = min(4, max(2, len(scaffold_slots)))
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
    def bounds(item_id: int) -> tuple[int, int]:
        return narrative_word_bounds(stage1_results[item_id].value or {}, cfg)

    def prompt_bounds(item_id: int) -> tuple[int, int]:
        return narrative_prompt_bounds(stage1_results[item_id].value or {}, cfg)

    def make_prompt(group_ids: Sequence[int], schema_attempt: int) -> str:
        payloads = []
        for item_id in group_ids:
            item_min, item_max = prompt_bounds(item_id)
            payload = scaffolded_generation_payload(
                item_id,
                stage1_results[item_id].value or {},
                stage2_results[item_id].value or {},
                cfg["use_timing"],
                item_min,
                item_max,
            )
            if validator_feedback and item_id in validator_feedback:
                payload["validator_feedback"] = validator_feedback[item_id]
            payload["required_min_words"] = item_min
            payload["required_max_words"] = item_max
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
        item_min, item_max = bounds(item_id)
        return validate_stage3(
            item,
            factors=factors,
            expected_id=item_id,
            expected_severity=str(severity.get("severity", "")),
            expected_timing=str(factors.get("postpartum_timing", "unknown")),
            source_text=records[item_id]["source_text"],
            use_timing=cfg["use_timing"],
            min_words=item_min,
            max_words=item_max,
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
        item_min, _ = bounds(item_id)
        candidate = result.last_candidate or {}
        candidate_text = candidate.get("synthetic_text")
        if (
            result.status != "ok"
            and isinstance(candidate_text, str)
            and word_count(candidate_text) < item_min
        ):
            repair_ids.append(item_id)

    def repair_prompt(group_ids: Sequence[int], _: int) -> str:
        item_id = group_ids[0]
        factors = stage1_results[item_id].value or {}
        severity = stage2_results[item_id].value or {}
        item_min, item_max = prompt_bounds(item_id)
        payload = {
            "id": item_id,
            "factors": scaffolded_generation_payload(
                item_id, factors, severity, cfg["use_timing"], item_min, item_max
            ),
            "target": severity.get("severity"),
            "expected_timing": factors.get("postpartum_timing", "unknown"),
            "under_length_draft": (generated[item_id].last_candidate or {}).get(
                "synthetic_text", ""
            ),
        }
        return prompt_stage3_length_repair(
            payload, item_min, item_max, cfg["use_timing"]
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
        item_min, item_max = prompt_bounds(item_id)
        payload = {
            "id": item_id,
            "factors": scaffolded_generation_payload(
                item_id, factors, severity, cfg["use_timing"], item_min, item_max
            ),
            "target": severity.get("severity"),
            "expected_timing": factors.get("postpartum_timing", "unknown"),
            "draft_to_rewrite": (generated[item_id].value or {}).get("synthetic_text", ""),
        }
        return prompt_stage3_content_repair(
            payload,
            content_categories[item_id],
            item_min,
            item_max,
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

    # 2026-09-04: repair validator failures before using deterministic prose.
    validation_repair_ids = [
        item_id
        for item_id, result in generated.items()
        if result.status != "ok" or result.value is None
    ]
    validation_cfg = {**cfg, "schema_retries": max(2, cfg["schema_retries"])}
    for item_id in validation_repair_ids:
        prior = generated[item_id]
        factors = stage1_results[item_id].value or {}
        severity = stage2_results[item_id].value or {}
        item_min, item_max = prompt_bounds(item_id)
        payload = {
            "id": item_id,
            "factors": scaffolded_generation_payload(
                item_id, factors, severity, cfg["use_timing"], item_min, item_max
            ),
            "target": severity.get("severity"),
            "expected_timing": factors.get("postpartum_timing", "unknown"),
            "rejected_draft": (prior.last_candidate or {}).get("synthetic_text", ""),
            "validation_errors": prior.errors[-4:],
        }

        def validation_prompt(_: Sequence[int], __: int) -> str:
            return prompt_stage3_validation_repair(
                payload, item_min, item_max, cfg["use_timing"]
            )

        repaired = run_group_stage(
            ids=[item_id],
            make_prompt=validation_prompt,
            validator=content_validator,
            cfg=validation_cfg,
            temperature=min(cfg["temp_s3"], 0.6),
            tokens_per_item=cfg["tokens_s3_per_item"],
            stage_number=45 + regeneration_attempt * 10,
            usage=usage,
        )[item_id]
        repaired.attempts += prior.attempts
        repaired.errors = prior.errors + repaired.errors
        generated[item_id] = repaired

    for item_id, result in generated.items():
        if result.status == "ok" and result.value is not None:
            continue
        factors = stage1_results[item_id].value or {}
        severity = stage2_results[item_id].value or {}
        item_min, item_max = bounds(item_id)
        fallback = conservative_factor_narrative(
            factors,
            str(severity.get("severity", "")),
            cfg["use_timing"],
            item_min,
            item_max,
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
                item,
                expected_id=item_id,
                narrative=str((generated[item_id].value or {}).get("synthetic_text", "")),
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
                item_min, item_max = bounds(item_id)
                fallback = conservative_factor_narrative(
                    factors,
                    str(severity.get("severity", "")),
                    cfg["use_timing"],
                    item_min,
                    item_max,
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

            assessment = reconcile_grounding_assessment(
                audit.value, stage1_results[item_id].value or {}
            )
            grounding_history[item_id].append(
                {
                    "round": audit_round,
                    "grounded": assessment["grounded"],
                    "unsupported_claims": assessment["unsupported_claims"],
                    "reconciled_supported_claims": assessment.get(
                        "reconciled_supported_claims", []
                    ),
                    "reason": assessment["reason"],
                }
            )
            if assessment["grounded"]:
                result.value["grounding_passed"] = True
                result.value["grounding_history"] = grounding_history[item_id]
                result.value["grounding_attempts"] = grounding_attempts[item_id]
                continue

            result.errors.append(
                "semantic grounding rejected: "
                + "; ".join(assessment["unsupported_claims"])
            )

            if audit_round == 1:
                factors = stage1_results[item_id].value or {}
                severity = stage2_results[item_id].value or {}
                item_min, item_max = bounds(item_id)
                fallback = conservative_factor_narrative(
                    factors,
                    str(severity.get("severity", "")),
                    cfg["use_timing"],
                    item_min,
                    item_max,
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
            prompt_min, prompt_max = prompt_bounds(item_id)

            def semantic_prompt(_: Sequence[int], __: int) -> str:
                payload = {
                    "id": item_id,
                    "factors": scaffolded_generation_payload(
                        item_id, factors, severity, cfg["use_timing"], prompt_min, prompt_max
                    ),
                    "target": severity.get("severity"),
                    "expected_timing": factors.get("postpartum_timing", "unknown"),
                    "draft_to_rewrite": (generated[item_id].value or {}).get(
                        "synthetic_text", ""
                    ),
                    "unsupported_claims": repair_claims[item_id],
                }
                return prompt_stage3_semantic_repair(
                    payload, prompt_min, prompt_max, cfg["use_timing"]
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
                item_min, item_max = bounds(item_id)
                fallback = conservative_factor_narrative(
                    factors,
                    str(severity.get("severity", "")),
                    cfg["use_timing"],
                    item_min,
                    item_max,
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
            raw_severity = str(parsed["predicted_severity"])
            parsed["raw_predicted_severity"] = raw_severity
            parsed["raw_evidence_profile"] = dict(parsed["evidence_profile"])
            parsed["evidence_profile"] = blind_narrative_evidence(narrative)
            structured_severity = severity_from_blind_evidence(parsed["evidence_profile"])
            parsed["structured_predicted_severity"] = structured_severity
            parsed["calibrated_predicted_severity"] = calibrate_blind_severity(
                narrative, raw_severity
            )
            # The target stays hidden; the fixed rubric maps only blind evidence.
            parsed["predicted_severity"] = structured_severity
            if cfg["use_timing"]:
                raw_timing = str(parsed["predicted_timing"])
                parsed["raw_predicted_timing"] = raw_timing
                parsed["deterministic_predicted_timing"] = blind_timing_bucket(narrative)
                parsed["predicted_timing"] = parsed["deterministic_predicted_timing"]
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
                "raw_prediction": (validation.value or {}).get(
                    "raw_predicted_severity", ""
                ),
                "evidence_profile": (validation.value or {}).get("evidence_profile", {}),
                "raw_evidence_profile": (validation.value or {}).get(
                    "raw_evidence_profile", {}
                ),
                "calibrated_prediction": (validation.value or {}).get(
                    "calibrated_predicted_severity", ""
                ),
                "predicted_timing": (validation.value or {}).get("predicted_timing", ""),
                "deterministic_timing": (validation.value or {}).get(
                    "deterministic_predicted_timing", ""
                ),
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


def direct_stages_and4(
    *,
    records: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
    usage: Usage,
) -> tuple[
    dict[int, ItemResult], dict[int, ItemResult], dict[int, ItemResult],
    dict[int, ItemResult], dict[int, dict[str, Any]],
]:
    """Run the one-prompt ablation under the standard acceptance gates."""
    ids = list(records)
    direct = generate_direct_baseline(
        ids=ids, records=records, cfg=cfg, usage=usage
    )
    s1, s2, s3 = split_direct_results(ids, direct)
    audit_direct_grounding(
        ids=ids, s1=s1, s2=s2, s3=s3, cfg=cfg, usage=usage
    )
    s4 = {item_id: ItemResult(status="not_run") for item_id in ids}
    outcomes = {
        item_id: {"row_status": "failed", "regen_count": 0, "validator_history": []}
        for item_id in ids
    }

    for item_id in ids:
        if s3[item_id].status != "ok" or s3[item_id].value is None:
            outcomes[item_id]["row_status"] = "direct_generation_failed"
            continue
        if not cfg["use_validator"]:
            s4[item_id] = ItemResult(status="skipped")
            outcomes[item_id]["row_status"] = "accepted"
            continue

        intended = str((s2[item_id].value or {}).get("severity", ""))
        intended_timing = str(
            (s1[item_id].value or {}).get("postpartum_timing", "unknown")
        )
        required = {item_id: (intended, intended_timing)}
        for validation_round in range(cfg["max_regen"] + 1):
            narrative = str((s3[item_id].value or {}).get("synthetic_text", ""))
            validation = stage4_single(item_id, narrative, cfg, usage, validation_round)
            s4[item_id] = validation
            outcomes[item_id]["validator_history"].append(
                {
                    "round": validation_round,
                    "status": validation.status,
                    "prediction": (validation.value or {}).get("predicted_severity", ""),
                    "raw_prediction": (validation.value or {}).get(
                        "raw_predicted_severity", ""
                    ),
                    "evidence_profile": (validation.value or {}).get(
                        "evidence_profile", {}
                    ),
                    "raw_evidence_profile": (validation.value or {}).get(
                        "raw_evidence_profile", {}
                    ),
                    "calibrated_prediction": (validation.value or {}).get(
                        "calibrated_predicted_severity", ""
                    ),
                    "predicted_timing": (validation.value or {}).get("predicted_timing", ""),
                    "deterministic_timing": (validation.value or {}).get(
                        "deterministic_predicted_timing", ""
                    ),
                    "confidence": (validation.value or {}).get("confidence", ""),
                    "errors": validation.errors,
                }
            )
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
            regenerated = generate_direct_baseline(
                ids=[item_id],
                records=records,
                cfg=cfg,
                usage=usage,
                regeneration_attempt=validation_round + 1,
                validator_feedback={
                    item_id: {
                        "predicted_severity": validation.value.get("predicted_severity"),
                        "required_severity": intended,
                        "predicted_timing": validation.value.get("predicted_timing"),
                        "required_timing": intended_timing,
                    }
                },
                required_targets=required,
            )
            new_s1, new_s2, new_s3 = split_direct_results([item_id], regenerated)
            audit_direct_grounding(
                ids=[item_id],
                s1=new_s1,
                s2=new_s2,
                s3=new_s3,
                cfg=cfg,
                usage=usage,
                regeneration_attempt=validation_round + 1,
            )
            for old, new in (
                (s1[item_id], new_s1[item_id]),
                (s2[item_id], new_s2[item_id]),
                (s3[item_id], new_s3[item_id]),
            ):
                new.attempts += old.attempts
                new.errors = old.errors + new.errors
            s1[item_id] = new_s1[item_id]
            s2[item_id] = new_s2[item_id]
            s3[item_id] = new_s3[item_id]
            if s3[item_id].status != "ok" or s3[item_id].value is None:
                outcomes[item_id]["row_status"] = "regen_failed"
                break

    return s1, s2, s3, s4, outcomes


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
    s2 = {
        item_id: ItemResult(status="not_run", errors=[record["source_eligibility_reason"]])
        for item_id, record in records.items()
    }
    s3 = {
        item_id: ItemResult(status="not_run", errors=[record["source_eligibility_reason"]])
        for item_id, record in records.items()
    }
    s4 = {
        item_id: ItemResult(status="not_run", errors=[record["source_eligibility_reason"]])
        for item_id, record in records.items()
    }
    outcomes = {
        item_id: {"row_status": "failed", "regen_count": 0, "validator_history": []}
        for item_id in records
    }
    if eligible_records and cfg["pipeline_mode"] == "direct":
        direct_s1, direct_s2, direct_s3, direct_s4, direct_outcomes = direct_stages_and4(
            records=eligible_records, cfg=cfg, usage=usage
        )
        s1.update(direct_s1)
        s2.update(direct_s2)
        s3.update(direct_s3)
        s4.update(direct_s4)
        outcomes.update(direct_outcomes)
    elif eligible_records:
        s1.update(stage1(eligible_records, cfg, usage))
        s2.update(stage2(s1, cfg, usage))
        staged_s3, staged_s4, staged_outcomes = stages3_and4(
            records=records,
            stage1_results=s1,
            stage2_results=s2,
            cfg=cfg,
            usage=usage,
        )
        s3.update(staged_s3)
        s4.update(staged_s4)
        outcomes.update(staged_outcomes)
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
        raw_predicted = str(validation.get("raw_predicted_severity", ""))
        calibrated_predicted = str(validation.get("calibrated_predicted_severity", ""))
        timing = str(factors.get("postpartum_timing", ""))
        predicted_timing = str(validation.get("predicted_timing", ""))
        deterministic_timing = str(validation.get("deterministic_predicted_timing", ""))
        required_min_words, required_max_words = narrative_word_bounds(factors, cfg)
        row = record["input"]
        rows.append(
            {
                "source_index": item_id,
                "batch_id": batch_id,
                "pipeline_mode": cfg["pipeline_mode"],
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
                "s1_risk_denials": factors.get("risk_denials", []),
                "s1_coping_or_adjustment_context": factors.get(
                    "coping_or_adjustment_context", ""
                ),
                "s1_supported_context": factors.get("supported_context", []),
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
                "s3_recovery_errors": s3[item_id].errors,
                "s3_text": text or FAILED_SENTINEL,
                "s3_target": generated.get("target", ""),
                "s3_timing": generated.get("timing", ""),
                "s3_style": generated.get("style", ""),
                "s3_word_count": word_count(text),
                "s3_required_min_words": required_min_words,
                "s3_required_max_words": required_max_words,
                "s3_identifier_leaks": identifier_leaks(text),
                "s3_shared_source_ngrams": shared_ngram_count(source, text, cfg["copy_ngram_size"]),
                "s3_unsupported_detail_flags": unsupported_detail_categories(text, factors),
                "s3_quality_flags": narrative_quality_violations(generated, text, factors),
                "s3_grounding_passed": bool(generated.get("grounding_passed", False)),
                "s3_grounding_attempts": generated.get("grounding_attempts", 0),
                "s3_grounding_history": generated.get("grounding_history", []),
                # a 100% acceptance rate hid a 100% fallback rate in pilot20_v410
                "s3_provenance": stage3_provenance(generated),
                "s3_timing_cue_injected": bool(generated.get("timing_cue_injected", False)),
                "s3_timing_language_normalized": bool(
                    generated.get("timing_language_normalized", False)
                ),
                "s3_timing_metadata_normalized": bool(
                    generated.get("timing_metadata_normalized", False)
                ),
                "s3_raw_timing_metadata": generated.get("raw_timing_metadata", ""),
                "s3_severity_descriptor_removed": bool(
                    generated.get("severity_descriptor_removed", False)
                ),
                "s3_target_descriptor_normalized": bool(
                    generated.get("target_descriptor_normalized", False)
                ),
                "s3_language_normalized": bool(generated.get("language_normalized", False)),
                "s3_regen_count": outcomes[item_id]["regen_count"],
                "s4_status": s4[item_id].status,
                "s4_attempts": s4[item_id].attempts,
                "s4_error": last_error(s4[item_id]),
                "s4_validation_method": "blind_evidence_fixed_rubric" if validation else "",
                "s4_predicted_severity": predicted,
                "s4_raw_predicted_severity": raw_predicted,
                "s4_evidence_profile": validation.get("evidence_profile", {}),
                "s4_raw_evidence_profile": validation.get("raw_evidence_profile", {}),
                "s4_calibrated_predicted_severity": calibrated_predicted,
                "s4_predicted_timing": predicted_timing,
                "s4_raw_predicted_timing": validation.get("raw_predicted_timing", ""),
                "s4_deterministic_predicted_timing": deterministic_timing,
                "s4_confidence": validation.get("confidence", ""),
                "s4_evidence": validation.get("evidence", ""),
                "s4_history": outcomes[item_id]["validator_history"],
                "severity_agreement": bool(predicted and predicted == intended),
                "raw_severity_agreement": bool(
                    raw_predicted and raw_predicted == intended
                ),
                "calibrated_severity_agreement": bool(
                    calibrated_predicted and calibrated_predicted == intended
                ),
                "timing_agreement": bool(
                    cfg["use_timing"] and predicted_timing and predicted_timing == timing
                ),
                "deterministic_timing_agreement": bool(
                    cfg["use_timing"] and deterministic_timing and deterministic_timing == timing
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
        "ollama_thinking": args.ollama_thinking,
        "pipeline_mode": args.pipeline_mode,
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
        "use_adaptive_length": not args.no_adaptive_length,
        "max_regen": args.max_regen,
        "temp_s1": args.temp_s1,
        "temp_s2": args.temp_s2,
        "temp_s3": args.temp_s3,
        "temp_s4": args.temp_s4,
        "tokens_s1_per_item": args.tokens_s1_per_item,
        "tokens_s2_per_item": args.tokens_s2_per_item,
        "tokens_s3_per_item": args.tokens_s3_per_item,
        "tokens_s4_per_item": args.tokens_s4_per_item,
        "tokens_direct_per_item": args.tokens_direct_per_item,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "sparse_min_words": args.sparse_min_words,
        "sparse_target_max_words": args.sparse_target_max_words,
        "sparse_max_words": args.sparse_max_words,
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
    if digest:
        updated["model_digest"] = digest
    updated["last_completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    atomic_write_json(path, updated)


def lock_manifest_model(
    path: Path, manifest: dict[str, Any], digest: str
) -> dict[str, Any]:
    """Pin the model before the first checkpoint is generated."""
    existing = str(manifest.get("model_digest") or "")
    if existing and existing != digest:
        raise RuntimeError(
            "Ollama model digest changed for this output directory; use a new --outdir"
        )
    updated = {**manifest, "model_digest": digest}
    atomic_write_json(path, updated)
    return updated


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
        "validator_raw_severity",
        "validator_evidence_profile",
        "validator_calibrated_severity",
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
                    "validator_raw_severity": row["s4_raw_predicted_severity"],
                    "validator_evidence_profile": row["s4_evidence_profile"],
                    "validator_calibrated_severity": row[
                        "s4_calibrated_predicted_severity"
                    ],
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
        "pipeline_mode": cfg["pipeline_mode"],
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
        "structured_blind_severity_agreement_rate_pct": rate(
            sum(bool(row["severity_agreement"]) for row in validated), len(validated)
        ),
        "raw_blind_severity_agreement_rate_pct": rate(
            sum(bool(row["raw_severity_agreement"]) for row in validated), len(validated)
        ),
        "calibrated_severity_agreement_rate_pct": rate(
            sum(bool(row["calibrated_severity_agreement"]) for row in validated),
            len(validated),
        ),
        "timing_agreement_rate_pct": rate(
            sum(bool(row["timing_agreement"]) for row in timing_validated),
            len(timing_validated),
        ) if cfg["use_timing"] else None,
        "raw_blind_timing_agreement_all_rows_pct": rate(
            sum(bool(row["timing_agreement"]) for row in validated), len(validated)
        ) if cfg["use_timing"] else None,
        "deterministic_timing_agreement_rate_pct": rate(
            sum(bool(row["deterministic_timing_agreement"]) for row in timing_validated),
            len(timing_validated),
        ) if cfg["use_timing"] else None,
        "deterministic_timing_agreement_all_rows_pct": rate(
            sum(bool(row["deterministic_timing_agreement"]) for row in validated),
            len(validated),
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
        "rows_flagged_for_quality_review": sum(
            bool(row.get("s3_quality_flags")) for row in eligible_rows
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
        "--tokens-direct-per-item": args.tokens_direct_per_item,
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
    if (
        args.sparse_min_words < 1
        or args.sparse_target_max_words < args.sparse_min_words
        or args.sparse_max_words < args.sparse_target_max_words
        or args.sparse_max_words < args.sparse_min_words
        or args.sparse_max_words > args.max_words
    ):
        parser.error(
            "sparse limits must satisfy 1 <= --sparse-min-words <= "
            "--sparse-target-max-words <= --sparse-max-words <= --max-words"
        )
    if args.workers > 1 and args.limit_rows and args.limit_rows <= 100:
        print("NOTE: --workers 1 is usually fastest and safest for CPU-only Ollama.", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "PPD generator with canonical four-stage and direct one-prompt modes. "
            "Safe defaults process 100 rows on one CPU-friendly worker."
        ),
    )
    parser.add_argument("--input", required=True, help="Source CSV")
    parser.add_argument("--outdir", required=True, help="Fresh run directory, or same directory to resume")
    parser.add_argument("--ollama-model", required=True, help="Exact pulled Ollama model tag")
    parser.add_argument("--ollama-host", default=OLLAMA_HOST_DEFAULT)
    parser.add_argument(
        "--ollama-thinking",
        choices=("auto", "on", "off"),
        default="auto",
        help="Control reasoning output for thinking-capable Ollama models",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=("four-stage", "direct"),
        default="four-stage",
        help="Generation architecture; direct is the one-prompt comparison baseline",
    )
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
    parser.add_argument("--temp-s3", type=float, default=0.45)
    parser.add_argument("--temp-s4", type=float, default=0.2)
    parser.add_argument("--tokens-s1-per-item", type=int, default=320)
    parser.add_argument("--tokens-s2-per-item", type=int, default=120)
    parser.add_argument("--tokens-s3-per-item", type=int, default=520)
    parser.add_argument("--tokens-s4-per-item", type=int, default=320)
    parser.add_argument("--tokens-direct-per-item", type=int, default=1000)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repeat-penalty", type=float, default=1.15)
    parser.add_argument("--min-words", type=int, default=30)
    parser.add_argument("--max-words", type=int, default=100)
    parser.add_argument(
        "--no-adaptive-length",
        action="store_true",
        help="Ablation: use the global word range even for sparse factors",
    )
    parser.add_argument("--sparse-min-words", type=int, default=18)
    parser.add_argument("--sparse-target-max-words", type=int, default=45)
    parser.add_argument("--sparse-max-words", type=int, default=65)
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
    log(f"Pipeline mode: {args.pipeline_mode}", log_file)
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
    grouped_calls = 2 if args.pipeline_mode == "direct" else 4
    minimum_calls = pending_eligible_batches * grouped_calls + (
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
        "ollama_thinking": args.ollama_thinking,
        "pipeline_mode": args.pipeline_mode,
        "workers": args.workers,
        "schema_retries": args.schema_retries,
        "transport_retries": args.transport_retries,
        "request_timeout": args.request_timeout,
        "num_ctx": args.num_ctx,
        "seed": args.seed,
        "use_timing": not args.no_timing,
        "use_validator": not args.no_validator,
        "use_source_screen": not args.no_source_screen,
        "use_adaptive_length": not args.no_adaptive_length,
        "max_regen": args.max_regen,
        "temp_s1": args.temp_s1,
        "temp_s2": args.temp_s2,
        "temp_s3": args.temp_s3,
        "temp_s4": args.temp_s4,
        "tokens_s1_per_item": args.tokens_s1_per_item,
        "tokens_s2_per_item": args.tokens_s2_per_item,
        "tokens_s3_per_item": args.tokens_s3_per_item,
        "tokens_s4_per_item": args.tokens_s4_per_item,
        "tokens_direct_per_item": args.tokens_direct_per_item,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repeat_penalty": args.repeat_penalty,
        "min_words": args.min_words,
        "max_words": args.max_words,
        "sparse_min_words": args.sparse_min_words,
        "sparse_target_max_words": args.sparse_target_max_words,
        "sparse_max_words": args.sparse_max_words,
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
        if not digest:
            digest = model_digest(args.ollama_host, args.ollama_model)
        if not digest:
            log("ERROR: Ollama model digest could not be recorded.", log_file)
            return 3
        try:
            manifest = lock_manifest_model(manifest_path, manifest, digest)
        except RuntimeError as exc:
            log(f"ERROR: {exc}", log_file)
            return 3

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
        if digest:
            try:
                manifest = lock_manifest_model(manifest_path, manifest, digest)
            except RuntimeError as exc:
                log(f"ERROR: {exc}", log_file)
                return 3
        log("All batches already have valid checkpoints; rebuilding final outputs.", log_file)

    refresh_checkpoint(outdir, batches)
    try:
        _, summary = merge_batches(outdir, batches, cfg)
    except (OSError, ValueError, RuntimeError) as exc:
        log(f"ERROR during merge: {exc}", log_file)
        return 5
    update_manifest_runtime(
        manifest_path, manifest, digest or str(manifest.get("model_digest") or "") or None
    )

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
    log(
        "Blind severity agreement: "
        f"{summary['structured_blind_severity_agreement_rate_pct']}% structured; "
        f"{summary['raw_blind_severity_agreement_rate_pct']}% raw LLM; "
        f"{summary['calibrated_severity_agreement_rate_pct']}% calibrated diagnostic.",
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
