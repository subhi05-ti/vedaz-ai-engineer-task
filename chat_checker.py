"""
chat_checker.py  —  Task 1 for Vedaz AI Engineer Stage-2
=========================================================
Reads a .jsonl OR .json file of chats, then:
  1. Validates structure (system → alternating user/assistant turns)
  2. Counts word length of each chat
  3. Finds duplicate / near-duplicate chats
  4. Splits into train (85%) and test (15%) sets
  5. Flags chats that break Vedaz safety rules

Run:
    python chat_checker.py --input vedaz_astrologer_finetune.jsonl
    python chat_checker.py --input vedaz_astrologer_finetune.json --llm-check
"""

# ── Standard-library imports ──────────────────────────────────────────────────
import json          # To read and write JSON / JSONL files
import os            # To check file existence and build file paths
import re            # Regular expressions — used for keyword safety patterns
import argparse      # To parse command-line arguments (--input, --llm-check, etc.)
import random        # To randomly shuffle chats before train/test split
import sys           # To exit the script cleanly on fatal errors
from difflib import SequenceMatcher   # Built-in fuzzy-string similarity (no install needed)
from collections import defaultdict   # Dictionary with automatic default values
from dataclasses import dataclass, field  # Clean data containers (no boilerplate __init__)
from typing import List, Tuple, Optional, Dict  # Type hints for readability

# ── Third-party imports ───────────────────────────────────────────────────────
from dotenv import load_dotenv   # Loads GEMINI_API_KEY from a .env file safely

# Load environment variables from .env file (keeps API keys out of source code)
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA CLASSES
# We use Python dataclasses so each result object is self-documenting and typed
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SafetyResult:
    """Holds the safety-check outcome for a single chat."""
    flagged: bool                    # True = this chat broke a safety rule
    reasons: List[str] = field(default_factory=list)  # Human-readable list of reasons
    layer: str = "none"              # Which detection layer caught it: "keyword", "llm", or "none"


@dataclass
class ChatReport:
    """Holds the full analysis result for a single chat."""
    index: int                        # Position in the input file (1-based)
    chat_id: str                      # ID field if present (e.g. "conv_001_career_govt")
    is_valid: bool                    # Did it pass structural validation?
    validation_error: str = ""        # If invalid, why?
    word_count: int = 0               # Total words across all messages
    is_duplicate: bool = False        # Is it a near-duplicate of another chat?
    duplicate_of: str = ""            # ID/index of the chat it duplicates
    safety: SafetyResult = field(default_factory=lambda: SafetyResult(flagged=False))
    split: str = ""                   # "train" or "test" after splitting


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SAFETY RULES (Layer 1: Keyword / Regex)
# ══════════════════════════════════════════════════════════════════════════════

# --- 2a. English keyword patterns -----------------------------------------
# Each entry is (rule_name, compiled_regex).
# We use re.IGNORECASE so "You Will Die" matches same as "you will die".
# We use word-boundary \b where possible to avoid false positives
#   (e.g. "predictable" should not match "predict death").
ENGLISH_SAFETY_PATTERNS: List[Tuple[str, re.Pattern]] = [

    # Rule: Predicting death or serious illness for the user
    ("death_prediction",
     re.compile(
         r"\b(you will die|your death|predict(s)? (your )?(death|dying)|"
         r"you (are|will be) dying|fatal|will not survive|"
         r"life (is|will be) in danger|serious (illness|disease) (is )?(coming|ahead)|"
         r"you (have|will get) cancer|terminal)\b",
         re.IGNORECASE
     )),

    # Rule: Medical guarantees (promising a cure or medical outcome)
    ("medical_guarantee",
     re.compile(
         r"\b(guaranteed (cure|treatment|recovery|healing)|"
         r"will definitely (cure|heal|fix|treat)|"
         r"100% (cure|recovery|healing)|"
         r"astrology (can|will) cure|planets (will|can) heal your (disease|illness|cancer))\b",
         re.IGNORECASE
     )),

    # Rule: Financial / money guarantees (promising profit or wealth)
    ("financial_guarantee",
     re.compile(
         r"\b(guaranteed (profit|wealth|income|money|returns?)|"
         r"will definitely (earn|make|get) (money|profit|crore|lakh)|"
         r"100% (profit|success|return)|"
         r"stars guarantee (profit|money|wealth)|"
         r"you (will|are going to) (become rich|get rich|make crore))\b",
         re.IGNORECASE
     )),

    # Rule: Fear-selling / pressure to buy remedy (the most common Vedaz violation)
    ("fear_pressure_remedy",
     re.compile(
         r"\b(pay (now|immediately|urgently|today) or|"
         r"(bad luck|misfortune|disaster|curse|ruined|destroyed) (will|is going to) (happen|come)|"
         r"without (this|the) (remedy|puja|gemstone|ritual) (you will|bad things)|"
         r"(your life|everything) will be (ruined|destroyed|over) (if|unless)|"
         r"must (pay|buy|purchase) (now|immediately)|"
         r"urgent (remedy|puja|ritual) required)\b",
         re.IGNORECASE
     )),

    # Rule: High-pressure urgency / scarcity (a classic dark-pattern phrase)
    ("high_pressure_urgency",
     re.compile(
         r"\b(act (now|immediately) or (suffer|face consequences)|"
         r"limited time (offer|remedy|opportunity)|"
         r"only (I|we) can (save|help|protect) you|"
         r"no one else can help you|"
         r"you must (pay|do this) (now|immediately|today) or)\b",
         re.IGNORECASE
     )),

    # Rule: Predicting misfortune with certainty
    ("misfortune_certainty",
     re.compile(
         r"\b(you will (definitely|certainly|surely) (fail|lose|suffer|be ruined)|"
         r"there is no (hope|way out|escape)|"
         r"nothing can (save|help) you)\b",
         re.IGNORECASE
     )),
]

# --- 2b. Hindi / Devanagari keyword patterns --------------------------------
# We check for Hindi-script phrases that commonly appear in fear-selling chats.
# Regex works fine on Unicode strings in Python 3.
HINDI_SAFETY_PATTERNS: List[Tuple[str, re.Pattern]] = [

    # Hindi: "you will die" / "death is certain"
    ("hindi_death_prediction",
     re.compile(
         r"(mrityu hogi|aapki mrityu|maut aayegi|aap marenge|mar jayenge)",
         re.IGNORECASE
     )),

    # Hindi: "your life will be ruined" (fear-selling phrase)
    ("hindi_life_ruined",
     re.compile(
         r"(jeevan barbaad|sab kuch barbaad ho jayega|jeevan tabah|barbaad ho jaega|tabah ho jaega)",
         re.IGNORECASE
     )),

    # Hindi: paying for remedy under pressure
    ("hindi_pay_now_pressure",
     re.compile(
         r"(abhi payment karo|turant paise do|nahi to bura hoga|puja karwani padegi warna)",
         re.IGNORECASE
     )),

    # Hindi: "guaranteed cure / guaranteed profit"
    ("hindi_guarantee",
     re.compile(
         r"(guaranteed ilaaj|pakka labh hoga|nishchit roop se theek|100 pratishat faayda)",
         re.IGNORECASE
     )),

    # Devanagari script: "life will be ruined" (direct Hindi script)
    ("devanagari_ruined",
     re.compile(
         r"(जीवन बर्बाद|सब कुछ बर्बाद|जीवन तबाह|बर्बाद हो जाएगा|तबाह हो जाएगा)",
         re.IGNORECASE
     )),

    # Devanagari script: death prediction
    ("devanagari_death",
     re.compile(
         r"(मृत्यु होगी|आपकी मृत्यु|मौत आएगी|आप मरेंगे|मर जाएंगे)",
         re.IGNORECASE
     )),
]


def check_safety_rules_layer1(full_text: str) -> SafetyResult:
    """
    Layer 1: Pure keyword/regex safety check.
    Fast, free, always runs.
    Returns SafetyResult with flagged=True if any pattern matches.
    """
    reasons = []   # Will collect every rule that fired

    # Check all English patterns one by one
    for rule_name, pattern in ENGLISH_SAFETY_PATTERNS:
        if pattern.search(full_text):       # .search() finds a match anywhere in the string
            reasons.append(f"[EN] {rule_name}")   # Record which rule matched

    # Check all Hindi / transliterated patterns one by one
    for rule_name, pattern in HINDI_SAFETY_PATTERNS:
        if pattern.search(full_text):
            reasons.append(f"[HI] {rule_name}")

    # If at least one reason was found, mark as flagged
    flagged = len(reasons) > 0
    layer = "keyword" if flagged else "none"   # Record which layer caught it

    return SafetyResult(flagged=flagged, reasons=reasons, layer=layer)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SAFETY RULES (Layer 2: LLM Judge via Gemini)
# This layer is OPTIONAL — only runs when --llm-check flag is passed.
# It sends the whole chat to Gemini and asks it to score safety.
# ══════════════════════════════════════════════════════════════════════════════

def check_safety_rules_layer2_llm(messages: list, chat_id: str) -> SafetyResult:
    """
    Layer 2: Ask Gemini to judge if this chat breaks Vedaz safety rules.
    More expensive (API call per chat) but catches paraphrased violations
    that keyword patterns miss.

    Returns SafetyResult. If the API call fails, falls back to SafetyResult(flagged=False).
    """
    try:
        # Import here so the script still works even if google-generativeai is not installed
        import google.generativeai as genai   # Gemini Python SDK
    except ImportError:
        # If the library isn't installed, skip LLM check gracefully
        print("  [WARN] google-generativeai not installed. Skipping LLM safety check.")
        return SafetyResult(flagged=False, layer="none")

    # Read the API key from the environment variable (never hard-code!)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # If the key is missing, warn and skip instead of crashing
        print("  [WARN] GEMINI_API_KEY not set. Skipping LLM safety check.")
        return SafetyResult(flagged=False, layer="none")

    # Configure the Gemini SDK with the API key
    genai.configure(api_key=api_key)

    # Choose gemini-2.5-flash — supported in current environment
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Convert the messages list to a readable transcript for the judge prompt
    transcript = "\n---\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in messages
    )

    # The judge prompt: give Gemini a strict rubric so scores are consistent
    judge_prompt = f"""You are a safety reviewer for Vedaz, an AI astrology platform.
Vedaz's rules are:
1. NEVER predict death, serious illness, or guaranteed misfortune.
2. NEVER promise a medical, financial, or money result.
3. NEVER pressure the user to pay for a remedy using fear.
4. NEVER use fear-based language like "your life will be ruined" or "act now or suffer".
5. Remedies must be framed as OPTIONAL supportive practices, not guarantees.

Read the following chat transcript and answer with ONLY a JSON object in this exact format:
{{
  "flagged": true or false,
  "reasons": ["reason 1", "reason 2"]
}}

CHAT (id: {chat_id}):
{transcript}

JSON response:"""

    try:
        # Send the prompt to Gemini and get the response
        response = model.generate_content(judge_prompt)
        raw_text = response.text.strip()   # Remove leading/trailing whitespace

        # Gemini sometimes wraps JSON in markdown code fences — strip those
        if raw_text.startswith("```"):
            start = raw_text.find("{")     # Find the opening brace
            end = raw_text.rfind("}") + 1  # Find the closing brace
            raw_text = raw_text[start:end] # Extract pure JSON

        # Parse the JSON response from the judge
        result = json.loads(raw_text)

        flagged = bool(result.get("flagged", False))
        reasons = result.get("reasons", [])

        return SafetyResult(
            flagged=flagged,
            reasons=[f"[LLM] {r}" for r in reasons],  # Prefix to distinguish from keyword hits
            layer="llm" if flagged else "none"
        )

    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        # If parsing fails log and skip gracefully
        print(f"  [WARN] LLM judge parse error for {chat_id}: {e}")
        return SafetyResult(flagged=False, layer="none")


def check_safety(messages: list, chat_id: str, use_llm: bool = False) -> SafetyResult:
    """
    Master safety check: always runs Layer 1 (keywords).
    If use_llm=True, also runs Layer 2 (Gemini) for a deeper semantic check.
    Merges results: if either layer flags it, the chat is flagged.

    IMPORTANT: We only scan ASSISTANT messages for safety violations.
    User messages often QUOTE scary things said by others (e.g. "the pandit told me
    my life will be ruined") — flagging those would be false positives. The violation
    is only a problem if the ASSISTANT says something unsafe, not the user.
    """
    # Build a single string from ASSISTANT messages only — this prevents
    # false positives where the user quotes a scary phrase said by someone else
    assistant_text = " ".join(
        m.get("content", "")
        for m in messages
        if m.get("role") == "assistant"   # Only include assistant turns
    )

    # Also build full text for LLM check (LLM can understand context better)
    full_text_for_llm = " ".join(
        f"{m.get('role', '')} {m.get('content', '')}" for m in messages
    )

    # Always run keyword check on assistant text only (fast and free)
    layer1 = check_safety_rules_layer1(assistant_text)

    if not use_llm:
        # LLM check is disabled — return keyword result only
        return layer1

    # LLM check is enabled — pass full conversation so judge has context
    layer2 = check_safety_rules_layer2_llm(messages, chat_id)

    # Merge: combine reasons from both layers
    combined_reasons = layer1.reasons + layer2.reasons
    combined_flagged = layer1.flagged or layer2.flagged

    # Determine which layers fired for the audit trail
    if layer1.flagged and layer2.flagged:
        combined_layer = "keyword+llm"
    elif layer1.flagged:
        combined_layer = "keyword"
    elif layer2.flagged:
        combined_layer = "llm"
    else:
        combined_layer = "none"

    return SafetyResult(
        flagged=combined_flagged,
        reasons=combined_reasons,
        layer=combined_layer
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — STRUCTURE VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_structure(messages: list) -> Tuple[bool, str]:
    """
    Checks that a chat's messages list follows Vedaz's required format:
      - First message must have role == "system"
      - After the system message, roles must alternate: user, assistant, user, assistant ...
      - Must have at least one user turn and one assistant turn

    Returns (is_valid: bool, error_message: str).
    """
    # Guard: messages must be a non-empty list
    if not isinstance(messages, list) or len(messages) == 0:
        return False, "messages field is empty or not a list"

    # Rule 1: First message must be the system prompt
    if messages[0].get("role") != "system":
        return False, f"First message role is '{messages[0].get('role')}', expected 'system'"

    # Rule 2: Must have at least 3 messages (system + 1 user + 1 assistant)
    if len(messages) < 3:
        return False, f"Only {len(messages)} messages — need at least system + user + assistant"

    # Rule 3: After system, roles must alternate user → assistant → user → ...
    expected_role = "user"   # After system, we always expect user first
    for i, msg in enumerate(messages[1:], start=1):   # Skip index 0 (system)
        actual_role = msg.get("role")
        if actual_role != expected_role:
            return False, (
                f"Message {i} has role '{actual_role}', "
                f"expected '{expected_role}' (alternating user/assistant)"
            )
        # Flip the expected role for the next iteration
        expected_role = "assistant" if expected_role == "user" else "user"

    # Rule 4: Every message must have a non-empty "content" field
    for i, msg in enumerate(messages):
        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            return False, f"Message {i} has empty or missing content"

    # All checks passed
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — WORD COUNT
# ══════════════════════════════════════════════════════════════════════════════

def count_words(messages: list) -> int:
    """
    Counts the total number of whitespace-separated words across all messages.
    This is a rough but fast approximation of chat length.
    split() handles both English and Hindi text since both use spaces.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")      # Get the content string
        total += len(content.split())          # split() splits on any whitespace
    return total


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DUPLICATE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def get_chat_text(messages: list) -> str:
    """
    Collapses all user+assistant message content into one string for
    similarity comparison. We exclude system prompts because many chats
    share the same system message — that would make unrelated chats
    appear similar.
    """
    return " ".join(
        m.get("content", "")
        for m in messages
        if m.get("role") in ("user", "assistant")   # Skip "system"
    )


def find_duplicates(chats: List[dict], threshold: float = 0.85) -> Dict[int, int]:
    """
    Finds near-duplicate chats using difflib.SequenceMatcher.

    SequenceMatcher gives a ratio from 0.0 (totally different) to 1.0 (identical).
    threshold=0.85 means 85% similar — catches chats that are copy-pastes
    with minor edits, but won't flag chats that are just on the same topic.

    We chose SequenceMatcher over hashing because:
    - Hash-based methods only catch EXACT duplicates.
    - SequenceMatcher catches near-duplicates (e.g. name changed, date changed).
    - No extra library needed (standard library).

    Limitation: O(n²) pairs — fine for hundreds of chats, not for 100,000+.

    Returns dict: {later_index: earlier_index} for each duplicate found.
    """
    # Pre-compute text for each chat once (avoids re-computing inside the loop)
    texts = [get_chat_text(c.get("messages", [])) for c in chats]

    duplicates: Dict[int, int] = {}   # Maps later → earlier index

    # Compare every pair (i, j) where i < j to avoid comparing a chat with itself
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if j in duplicates:
                # j is already marked as a duplicate — skip it
                continue

            # Compute similarity ratio between the two chat texts
            ratio = SequenceMatcher(None, texts[i], texts[j]).ratio()

            if ratio >= threshold:
                duplicates[j] = i   # j is a near-duplicate of i

    return duplicates


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FILE LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_chats(filepath: str) -> List[dict]:
    """
    Loads chats from either:
    - A .jsonl file (one JSON object per line — standard fine-tuning format)
    - A .json file  (a JSON array — Vedaz's original format)

    Returns a list of dicts. Each dict must have at least a "messages" key.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        sys.exit(1)

    chats = []

    if filepath.endswith(".jsonl"):
        # JSONL: read line by line, parse each line as JSON
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()      # Remove newlines / spaces
                if not line:
                    continue             # Skip blank lines (common at end of file)
                try:
                    chats.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[WARN] Line {line_num} is not valid JSON: {e}")

    elif filepath.endswith(".json"):
        # JSON array: read the whole file, expect a list at the top level
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("[ERROR] .json file must contain a JSON array at the top level.")
            sys.exit(1)
        chats = data

    else:
        print(f"[ERROR] Unsupported file extension. Use .jsonl or .json")
        sys.exit(1)

    print(f"[INFO] Loaded {len(chats)} chats from '{filepath}'")
    return chats


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def split_train_test(
    chats: List[dict],
    good_reports: List[ChatReport],
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """
    Splits VALID, NON-DUPLICATE, NON-FLAGGED chats into train and test sets.

    test_ratio=0.15 means 15% go to test, 85% to train.
    seed=42 makes the split reproducible (same input → same split every time).
    """
    # Identify which indices are good (valid, not duplicate, not flagged)
    good_indices = [
        r.index - 1   # index is 1-based in reports; convert to 0-based for list access
        for r in good_reports
    ]

    # Set the random seed so the split is always the same for the same input
    random.seed(seed)
    random.shuffle(good_indices)   # Shuffle in place

    # Calculate how many go into test (at least 1)
    test_size = max(1, int(len(good_indices) * test_ratio))

    # First test_size go to test, the rest go to train
    test_indices  = set(good_indices[:test_size])
    train_indices = set(good_indices[test_size:])

    train_chats = [chats[i] for i in sorted(train_indices)]
    test_chats  = [chats[i] for i in sorted(test_indices)]

    return train_chats, test_chats


def save_jsonl(chats: List[dict], filepath: str) -> None:
    """
    Saves a list of chat dicts to a .jsonl file (one JSON object per line).
    ensure_ascii=False preserves Hindi / Devanagari characters correctly.
    """
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)   # Create parent dir if needed
    with open(filepath, "w", encoding="utf-8") as f:
        for chat in chats:
            f.write(json.dumps(chat, ensure_ascii=False) + "\n")   # One JSON object per line
    print(f"[INFO] Saved {len(chats)} chats -> {filepath}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — REPORT PRINTING
# ══════════════════════════════════════════════════════════════════════════════

def print_report(reports: List[ChatReport], show_all: bool = False) -> None:
    """
    Prints a human-readable summary report to stdout.
    Always shows summary stats.
    Shows detail for FLAGGED or INVALID chats.
    With show_all=True, also lists every clean chat.
    """
    # ── Header ──
    print("\n" + "=" * 70)
    print("  VEDAZ CHAT CHECKER — REPORT")
    print("=" * 70)

    # ── Summary counts ──
    total         = len(reports)
    valid_count   = sum(1 for r in reports if r.is_valid)
    invalid_count = total - valid_count
    dup_count     = sum(1 for r in reports if r.is_duplicate)
    flagged_count = sum(1 for r in reports if r.safety.flagged)
    train_count   = sum(1 for r in reports if r.split == "train")
    test_count    = sum(1 for r in reports if r.split == "test")

    # Word count stats (only for valid chats to avoid skewing average)
    word_counts = [r.word_count for r in reports if r.is_valid]
    avg_words   = int(sum(word_counts) / len(word_counts)) if word_counts else 0
    min_words   = min(word_counts) if word_counts else 0
    max_words   = max(word_counts) if word_counts else 0

    print(f"\n  Total chats loaded   : {total}")
    print(f"  Valid (structure OK) : {valid_count}")
    print(f"  Invalid structure    : {invalid_count}")
    print(f"  Near-duplicates      : {dup_count}")
    print(f"  Safety FLAGGED       : {flagged_count}  <- These break Vedaz rules!")
    print(f"  Train split          : {train_count}")
    print(f"  Test split           : {test_count}")
    print(f"\n  Word count (valid)   : avg={avg_words}, min={min_words}, max={max_words}")

    # ── Per-chat details for problems only ──
    print("\n" + "-" * 70)
    print("  PROBLEM CHATS (invalid / duplicate / flagged)")
    print("-" * 70)

    problems = [r for r in reports if not r.is_valid or r.is_duplicate or r.safety.flagged]

    if not problems:
        print("  All chats passed all checks!")
    else:
        for r in problems:
            status_parts = []
            if not r.is_valid:
                status_parts.append("INVALID")
            if r.is_duplicate:
                status_parts.append(f"DUPLICATE of #{r.duplicate_of}")
            if r.safety.flagged:
                status_parts.append("SAFETY FLAG")

            status = " | ".join(status_parts)
            print(f"\n  Chat #{r.index:02d}  [{r.chat_id}]  [{status}]")

            if not r.is_valid:
                print(f"    Reason : {r.validation_error}")

            if r.safety.flagged:
                print(f"    Layer  : {r.safety.layer}")
                for reason in r.safety.reasons:
                    print(f"    Rule   : {reason}")

    # ── Optional verbose mode: show all chats ──
    if show_all:
        print("\n" + "-" * 70)
        print("  ALL CHATS")
        print("-" * 70)
        for r in reports:
            ok = r.is_valid and not r.is_duplicate and not r.safety.flagged
            icon = "OK" if ok else "!!"
            print(f"  [{icon}] #{r.index:02d} [{r.chat_id:30s}]  words={r.word_count:4d}  split={r.split}")

    print("\n" + "=" * 70 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Entry point. Parses CLI arguments, runs all checks, prints report,
    and saves train/test splits to the output directory.
    """
    # ── Argument parser ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Vedaz Chat Checker — validates, flags, and splits astrology chats"
    )
    parser.add_argument(
        "--input", "-i",
        default="vedaz_astrologer_finetune.jsonl",   # Default to the JSONL file
        help="Path to .jsonl or .json chat file"
    )
    parser.add_argument(
        "--llm-check",
        action="store_true",   # Flag — presence means True, absence means False
        help="Also run Gemini LLM safety check (slower but catches paraphrased violations)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of good chats to put in test set (default: 0.15 = 15%%)"
    )
    parser.add_argument(
        "--out-dir",
        default="data",
        help="Output directory for train/test splits (default: data/)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print all chats in the report, not just problems"
    )
    args = parser.parse_args()

    # ── Step 1: Load chats ───────────────────────────────────────────────────
    chats = load_chats(args.input)

    # ── Step 2: Find duplicates BEFORE per-chat processing ──────────────────
    print("[INFO] Running duplicate detection...")
    dup_map = find_duplicates(chats)   # {later_idx: earlier_idx}
    if dup_map:
        print(f"[INFO] Found {len(dup_map)} near-duplicate(s).")
    else:
        print("[INFO] No duplicates found.")

    # ── Step 3: Process each chat ────────────────────────────────────────────
    reports: List[ChatReport] = []

    for i, chat in enumerate(chats):
        # Build a human-readable ID for this chat (use "id" field or fallback)
        chat_id = chat.get("id", f"chat_{i+1:03d}")

        # Extract the messages list (works for both .json and .jsonl formats)
        messages = chat.get("messages", [])

        # 3a. Validate structure
        is_valid, val_error = validate_structure(messages)

        # 3b. Count words (run even if invalid so we can still report length)
        words = count_words(messages)

        # 3c. Check if this chat is a near-duplicate
        is_dup = i in dup_map
        dup_of = ""
        if is_dup:
            orig_idx = dup_map[i]
            orig_id  = chats[orig_idx].get("id", f"chat_{orig_idx+1:03d}")
            dup_of   = orig_id

        # 3d. Safety check (only run on valid chats — invalid ones may be malformed)
        if is_valid:
            safety = check_safety(messages, chat_id, use_llm=args.llm_check)
        else:
            # Skip safety check; structural error takes priority
            safety = SafetyResult(flagged=False, layer="skipped")

        # 3e. Build the report object for this chat
        report = ChatReport(
            index=i + 1,          # 1-based for human readability
            chat_id=chat_id,
            is_valid=is_valid,
            validation_error=val_error,
            word_count=words,
            is_duplicate=is_dup,
            duplicate_of=dup_of,
            safety=safety,
        )
        reports.append(report)

    # ── Step 4: Train / Test Split ───────────────────────────────────────────
    # Only include chats that are valid, not duplicate, and not safety-flagged
    good_reports = [
        r for r in reports
        if r.is_valid and not r.is_duplicate and not r.safety.flagged
    ]
    train_chats, test_chats = split_train_test(
        chats, good_reports, test_ratio=args.test_ratio
    )

    # ── Step 5: Update split labels in reports (for display) ─────────────────
    train_ids = {c.get("id", "") for c in train_chats}
    test_ids  = {c.get("id", "") for c in test_chats}
    for r in reports:
        if r.chat_id in train_ids:
            r.split = "train"
        elif r.chat_id in test_ids:
            r.split = "test"

    # ── Step 6: Save splits to disk ──────────────────────────────────────────
    train_path = os.path.join(args.out_dir, "train_split.jsonl")
    test_path  = os.path.join(args.out_dir, "test_split.jsonl")
    save_jsonl(train_chats, train_path)
    save_jsonl(test_chats, test_path)

    # ── Step 7: Print human-readable report ──────────────────────────────────
    print_report(reports, show_all=args.verbose)

    # ── Step 8: Exit code — non-zero if any problems found ──────────────────
    # Non-zero exit signals failure to shell scripts / CI pipelines
    problems = sum(
        1 for r in reports
        if not r.is_valid or r.is_duplicate or r.safety.flagged
    )
    if problems > 0:
        print(f"[RESULT] {problems} problem chat(s) found. Review report above.")
        sys.exit(1)
    else:
        print("[RESULT] All chats passed all checks!")
        sys.exit(0)


# ── Script entry point ────────────────────────────────────────────────────────
# Only runs when the script is executed directly (not when imported as a module)
if __name__ == "__main__":
    main()
