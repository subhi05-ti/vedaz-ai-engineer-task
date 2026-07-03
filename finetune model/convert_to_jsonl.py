"""
============================================================
  Vedaz — JSON → Fine-tune JSONL Converter
  Step 1 of fine-tuning pipeline
============================================================

WHAT THIS SCRIPT DOES:
  The client's JSON file has a mixed format:
    - Some conversations are on a single line (compact JSON)
    - Some are pretty-printed across multiple lines
    - Some have extra fields like "id" and "tags"

  This script normalizes everything into the exact JSONL format
  that Qwen2.5 fine-tuning (via HuggingFace TRL) expects:

  OUTPUT FORMAT (one JSON object per line):
  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

HOW TO RUN:
  python convert_to_jsonl.py

OUTPUT:
  vedaz_finetune_ready.jsonl  ← use this file for fine-tuning
  conversion_report.txt       ← summary of what was converted
============================================================
"""

import json
import os

# ── Config ───────────────────────────────────────────────
INPUT_FILE  = "Chat Data for assessment of applicants.json"
OUTPUT_FILE = "vedaz_finetune_ready.jsonl"
REPORT_FILE = "conversion_report.txt"
# ─────────────────────────────────────────────────────────


def load_mixed_json(filepath):
    """
    The client's JSON file is 'mixed format':
    - Some entries are compact single-line JSON objects
    - Some entries are pretty-printed across multiple lines
    - They are NOT inside a JSON array [...] — they're just stacked

    Strategy:
    1. Read the whole file as text
    2. Use a brace-counter to extract each top-level JSON object
    3. Parse each one individually
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    objects = []
    depth = 0
    start = None

    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i      # beginning of a new top-level object
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = raw[start : i + 1]
                try:
                    obj = json.loads(chunk)
                    objects.append(obj)
                except json.JSONDecodeError as e:
                    print(f"  ⚠  Could not parse chunk starting at char {start}: {e}")
                start = None

    return objects


def normalize_conversation(obj):
    """
    Extract and validate the 'messages' list from a conversation object.
    Strips out extra fields like 'id' and 'tags' that aren't needed for fine-tuning.
    Returns the clean {"messages": [...]} dict, or None if invalid.
    """
    messages = obj.get("messages", [])

    if not messages:
        return None

    # Validate: each message must have 'role' and non-empty 'content'
    valid_roles = {"system", "user", "assistant"}
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        if msg.get("role") not in valid_roles:
            return None
        if not str(msg.get("content", "")).strip():
            return None

    # Keep only role + content (drop any extra keys inside messages)
    clean_messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
    ]

    return {"messages": clean_messages}


def validate_chat_structure(messages):
    """
    Basic structure check:
    - First message should be 'system'
    - After system: should alternate user → assistant
    Returns (is_valid, reason)
    """
    if not messages:
        return False, "Empty messages list"

    if messages[0]["role"] != "system":
        return False, "First message is not 'system'"

    # Check alternation after system
    convo = messages[1:]
    if not convo:
        return False, "No conversation turns after system message"

    expected = "user"
    for i, msg in enumerate(convo):
        if msg["role"] != expected:
            return False, f"Turn {i+2}: expected '{expected}', got '{msg['role']}'"
        expected = "assistant" if expected == "user" else "user"

    # Must end with assistant
    if messages[-1]["role"] != "assistant":
        return False, "Last message is not 'assistant'"

    return True, "OK"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path  = os.path.join(script_dir, INPUT_FILE)
    output_path = os.path.join(script_dir, OUTPUT_FILE)
    report_path = os.path.join(script_dir, REPORT_FILE)

    print("=" * 60)
    print("  Vedaz JSON -> Fine-tune JSONL Converter")
    print("=" * 60)
    print(f"\n  Input  : {INPUT_FILE}")
    print(f"  Output : {OUTPUT_FILE}\n")

    # Step 1: Load all conversation objects
    print("Step 1 - Loading and parsing JSON file...")
    raw_objects = load_mixed_json(input_path)
    print(f"  Found {len(raw_objects)} conversation objects\n")

    # Step 2: Normalize and validate
    print("Step 2 - Normalizing and validating conversations...\n")

    results = []
    stats = {
        "total":    len(raw_objects),
        "valid":    0,
        "skipped":  0,
        "reasons":  []
    }

    for i, obj in enumerate(raw_objects, 1):
        clean = normalize_conversation(obj)

        if clean is None:
            reason = f"Conv {i:02d}: No valid 'messages' field - skipped"
            print(f"  [SKIP] {reason}")
            stats["skipped"] += 1
            stats["reasons"].append(reason)
            continue

        is_valid, reason = validate_chat_structure(clean["messages"])

        if not is_valid:
            reason_msg = f"Conv {i:02d}: Invalid structure - {reason} - skipped"
            print(f"  [SKIP] {reason_msg}")
            stats["skipped"] += 1
            stats["reasons"].append(reason_msg)
            continue

        turn_count = (len(clean["messages"]) - 1) // 2  # exclude system
        print(f"  [OK]   Conv {i:02d}: {turn_count} turn(s)")
        stats["valid"] += 1
        results.append(clean)

    # Step 3: Write output JSONL
    print(f"\nStep 3 - Writing {len(results)} conversations to JSONL...")
    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Step 4: Print summary
    print("\n" + "=" * 60)
    print("  CONVERSION SUMMARY")
    print("=" * 60)
    print(f"  Total conversations found  : {stats['total']}")
    print(f"  [OK]   Successfully converted : {stats['valid']}")
    print(f"  [SKIP] Skipped (invalid)      : {stats['skipped']}")
    print(f"\n  Output saved to: {OUTPUT_FILE}")
    print("=" * 60)

    # Step 5: Write report
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("VEDAZ JSON -> JSONL CONVERSION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Input file  : {INPUT_FILE}\n")
        f.write(f"Output file : {OUTPUT_FILE}\n")
        f.write(f"Total found : {stats['total']}\n")
        f.write(f"Converted   : {stats['valid']}\n")
        f.write(f"Skipped     : {stats['skipped']}\n")
        if stats["reasons"]:
            f.write("\nSkipped reasons:\n")
            for r in stats["reasons"]:
                f.write(f"  - {r}\n")

    print(f"\nDetailed report saved to: {REPORT_FILE}")
    print("\nDone! Your fine-tuning dataset is ready.\n")
    print(f"   Next step: Upload '{OUTPUT_FILE}' to Google Colab")
    print("   and run the fine-tuning notebook.\n")


if __name__ == "__main__":
    main()
