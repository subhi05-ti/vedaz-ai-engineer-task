"""
chat_generator.py  —  Task 2 for Vedaz AI Engineer Stage-2
===========================================================
Uses LangChain + Gemini to generate new Vedaz-style astrology chats.
Every generated chat is automatically checked by chat_checker.py rules.
Only chats that PASS all safety + structure checks are saved.

Why LangChain?
  - ChatPromptTemplate  : clean, reusable prompt with {variables}
  - LCEL pipe syntax    : prompt | model | parser — readable one-liner chain
  - Model swap          : change Gemini to OpenAI by changing 1 line
  - StrOutputParser     : extracts plain text from LLM response object

Run:
    python chat_generator.py
    python chat_generator.py --topics "career delay, Hindi" "marriage, skeptical user"
    python chat_generator.py --count 5 --out data/generated_chats.jsonl
"""

# ── Standard-library imports ──────────────────────────────────────────────────
import os           # To read environment variables (GEMINI_API_KEY)
import json         # To parse LLM output as JSON and write JSONL files
import re           # To strip markdown code fences from LLM response
import time         # To add a small delay between API calls (avoid rate limits)
import argparse     # To accept command-line arguments (--topics, --count, --out)
import sys          # To exit cleanly on fatal errors

# ── LangChain imports ─────────────────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
# ChatGoogleGenerativeAI: LangChain's wrapper for Google Gemini chat models.
# It handles authentication, retry logic, and formats messages for Gemini's API.
# Equivalent to: genai.GenerativeModel("gemini-1.5-flash") but LangChain-aware.

from langchain_core.prompts import ChatPromptTemplate
# ChatPromptTemplate: defines the structure of the prompt using {variable} placeholders.
# Lets us reuse the same prompt template for different topics without copy-pasting.

from langchain_core.output_parsers import StrOutputParser
# StrOutputParser: extracts the plain text string from the LLM's response object.
# Without this, we'd get a LangChain AIMessage object instead of a plain string.

# ── Third-party imports ───────────────────────────────────────────────────────
from dotenv import load_dotenv   # Reads GEMINI_API_KEY from .env file into os.environ

# ── Local import: reuse safety + validation functions from Task 1 ─────────────
# This is the key design choice — we don't rewrite the safety rules.
# We import directly from chat_checker.py so the rules stay in one place.
from chat_checker import (
    validate_structure,      # Checks system → user → assistant alternating order
    check_safety,            # Runs keyword + optional LLM safety rules
    count_words,             # Counts total words in a chat
    save_jsonl,              # Saves a list of chats to a .jsonl file
)

# Load .env file so GEMINI_API_KEY is available in os.environ
load_dotenv()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — VEDAZ VOICE STYLE GUIDE
# This is the "memory" we give to Gemini about what Vedaz sounds like.
# It's injected into every prompt so every generated chat stays on-brand.
# ══════════════════════════════════════════════════════════════════════════════

VEDAZ_STYLE_GUIDE = """
VEDAZ AI ASTROLOGER — VOICE & RULES
=====================================
You are generating training data for Vedaz's AI Vedic astrologer.

PERSONALITY:
- Warm, compassionate, like a wise and caring friend
- Non-fatalistic: astrology shows tendencies, not fixed fate
- Honest about limits: never overpromise what astrology can do
- Empowering: always put the user in control of their own life
- Responds in the same language as the user (Hindi, Hinglish, or English)

ABSOLUTE RULES (NEVER break these):
1. NEVER predict death, serious illness, or fatal outcomes
2. NEVER guarantee money, profit, or financial results
3. NEVER pressure the user to buy a remedy using fear
4. NEVER say "your life will be ruined" or similar fear phrases
5. For health issues: always redirect to a qualified doctor
6. For financial decisions: always redirect to a financial advisor
7. Remedies (mantra, puja, gemstone) must be framed as OPTIONAL
   supportive spiritual practices — NOT as cures or guarantees

WHAT GOOD RESPONSES LOOK LIKE:
- Acknowledge the user's emotion first ("I understand this is worrying...")
- Give astrological context (dasha, gochar, bhav) with humility
- Balance insight with practical, real-world advice
- Ask a follow-up question to deepen the conversation
- End warmly, never with fear or pressure

FORMAT RULES:
- system message: Vedaz's persona instruction (1-2 sentences)
- user messages: natural, conversational, sometimes with DOB details
- assistant messages: warm, detailed, 100-300 words, structured response
- Minimum 3 messages (system + user + assistant)
- Maximum 7 messages (system + 3 user/assistant pairs)
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TOPIC LIST
# These are the scenarios we want to generate chats for.
# Each topic string describes the situation AND any style/language hints.
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_TOPICS = [
    "career delay and job search frustration, Hindi language",
    "marriage compatibility and Guna Milan, skeptical user, Hinglish",
    "Sade Sati anxiety and Saturn transit fears, Hindi language",
    "business startup with borrowed money, English language",
    "love relationship compatibility by rashi, Hinglish",
    "Kaal Sarp Dosh myth-busting, user scared by pandit, Hindi",
    "gemstone recommendation request, Hinglish",
    "board exam anxiety and student motivation, Hindi language",
    "life purpose and career direction, introspective user, English",
    "Griha Pravesh muhurat selection for August 2026, Hindi",
    "child's academic performance and future, worried parent, Hindi",
    "health redirect when user asks about chest pain, English",
    "kundli matching for arranged marriage, Hinglish",
]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LANGCHAIN PROMPT TEMPLATE
# ChatPromptTemplate.from_messages() defines the messages that get sent to Gemini.
# {style_guide} and {topic} are placeholders filled in at runtime.
# ══════════════════════════════════════════════════════════════════════════════

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    # "system" role: sets the context/persona for the LLM itself
    # This tells Gemini it is a data generator, not the astrologer
    (
        "system",
        """You are an expert training-data generator for an AI astrology platform called Vedaz.
Your job is to create realistic, high-quality example conversations.

{style_guide}

OUTPUT FORMAT — you must return ONLY a valid JSON object, nothing else:
{{
  "messages": [
    {{"role": "system",    "content": "Vedaz system prompt here"}},
    {{"role": "user",      "content": "user message here"}},
    {{"role": "assistant", "content": "astrologer response here"}},
    {{"role": "user",      "content": "user follow-up (optional)"}},
    {{"role": "assistant", "content": "astrologer follow-up (optional)"}}
  ]
}}

Rules for your output:
- Return ONLY the JSON object. No explanation text before or after.
- Do NOT wrap in markdown code fences (no ```json).
- Ensure the JSON is valid and parseable.
- The system message must define the Vedaz astrologer's persona.
- user and assistant must alternate after the system message.
- The assistant must NEVER break Vedaz's safety rules listed above.
"""
    ),
    # "human" role: the actual generation request with the specific topic
    (
        "human",
        "Generate one complete Vedaz astrologer chat for this topic:\n\nTOPIC: {topic}\n\nReturn ONLY the JSON object:"
    ),
])

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LANGCHAIN CHAIN BUILDER
# This creates the LangChain LCEL (LangChain Expression Language) chain.
# The | (pipe) operator connects: prompt → model → output parser
# ══════════════════════════════════════════════════════════════════════════════

def build_chain(temperature: float = 0.8):
    """
    Builds and returns the LangChain generation chain.

    Chain flow:
        ChatPromptTemplate  →  formats the prompt with {style_guide} and {topic}
        |
        ChatGoogleGenerativeAI  →  sends the formatted prompt to Gemini
        |
        StrOutputParser  →  extracts the plain text string from the response

    temperature=0.8 gives creative variety between chats.
    Lower (0.3) = more consistent but repetitive.
    Higher (0.9) = more varied but may drift from format.
    """
    # Read the API key from environment (loaded from .env by load_dotenv above)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # Exit immediately if no key — every function below needs it
        print("[ERROR] GEMINI_API_KEY is not set. Please add it to your .env file.")
        sys.exit(1)

    # Initialize the Gemini chat model via LangChain's wrapper
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",      # Supported model in current environment
        google_api_key=api_key,        # Pass the key explicitly (also reads from env)
        temperature=temperature,        # Controls randomness/creativity of responses
    )

    # Build the LCEL chain using LangChain's pipe operator |
    # This is equivalent to: output = parser(llm(prompt.format(inputs)))
    chain = GENERATION_PROMPT | llm | StrOutputParser()
    # ^ GENERATION_PROMPT formats the messages with {style_guide} and {topic}
    # ^ llm sends the formatted messages to Gemini and gets a response
    # ^ StrOutputParser extracts the .content string from AIMessage object

    return chain   # Return the assembled chain for use in generate_chat()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — JSON PARSER
# Gemini sometimes wraps its JSON in markdown fences (```json ... ```)
# or adds explanation text before/after. This function strips all of that.
# ══════════════════════════════════════════════════════════════════════════════

def parse_chat_json(raw_text: str, topic: str) -> dict | None:
    """
    Safely parses the raw LLM output into a Python dict.

    Handles these common LLM output issues:
    1. JSON wrapped in markdown: ```json { ... } ```
    2. Extra explanation text before/after the JSON
    3. Completely invalid JSON (returns None so we can skip and retry)

    Returns the parsed dict if successful, or None if parsing fails.
    """
    # Strip leading/trailing whitespace from the response
    text = raw_text.strip()

    # Remove markdown code fences if present (Gemini sometimes adds these)
    # Pattern: ```json  or  ```  at the start, and  ```  at the end
    if text.startswith("```"):
        # Remove the opening fence line (e.g., "```json\n")
        text = re.sub(r"^```[a-z]*\n?", "", text)
        # Remove the closing fence (e.g., "```")
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()   # Strip again after fence removal

    # If Gemini added text before the JSON, find where the JSON starts
    # JSON always starts with { so we search for the first {
    start_idx = text.find("{")
    if start_idx == -1:
        # No { found at all — LLM gave a non-JSON response entirely
        print(f"  [WARN] No JSON object found in LLM response for topic: {topic}")
        return None   # Signal to the caller that this attempt failed

    # Find where the JSON ends — the last } in the string
    end_idx = text.rfind("}") + 1
    if end_idx == 0:
        # Found { but no matching } — incomplete/truncated JSON
        print(f"  [WARN] JSON object is incomplete (no closing brace) for: {topic}")
        return None

    # Extract only the JSON portion (strip any trailing explanation text)
    json_text = text[start_idx:end_idx]

    try:
        # Try to parse the extracted JSON string into a Python dict
        parsed = json.loads(json_text)
        return parsed   # Success — return the dict

    except json.JSONDecodeError as e:
        # JSON is malformed (e.g., trailing comma, unquoted key)
        print(f"  [WARN] JSON parse error for topic '{topic}': {e}")
        return None   # Signal failure to the caller


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SINGLE CHAT GENERATOR
# Calls the LangChain chain for one topic and validates the result.
# ══════════════════════════════════════════════════════════════════════════════

def generate_one_chat(chain, topic: str, chat_id: str, use_llm_safety: bool = False) -> dict | None:
    """
    Generates ONE chat for the given topic and validates it.

    Steps:
    1. Call the LangChain chain → get raw text from Gemini
    2. Parse the raw text as JSON
    3. Validate structure (system → alternating user/assistant)
    4. Check safety rules (keyword layer always, LLM layer if requested)

    Returns the chat dict if valid + safe, or None if it fails any check.
    """
    print(f"\n  [GEN] Generating: {topic[:60]}...")

    try:
        # Call the LangChain chain with the inputs for our prompt placeholders
        raw_output = chain.invoke({
            "style_guide": VEDAZ_STYLE_GUIDE,   # The full Vedaz rules and voice guide
            "topic": topic,                       # The specific situation to generate
        })
        # raw_output is now a plain string (StrOutputParser extracted it)

    except Exception as e:
        # Any API error (quota exceeded, network error, etc.) — skip this attempt
        print(f"  [ERROR] LangChain/Gemini API error for '{topic}': {e}")
        return None

    # Parse the string output into a Python dict
    parsed = parse_chat_json(raw_output, topic)
    if parsed is None:
        # parse_chat_json already printed a warning
        return None

    # Extract the messages list from the parsed dict
    # The LLM should produce {"messages": [...]} at the top level
    messages = parsed.get("messages", [])
    if not messages:
        print(f"  [WARN] Parsed JSON has no 'messages' key for topic: {topic}")
        return None

    # Validate the structure of the messages list
    is_valid, error_msg = validate_structure(messages)
    if not is_valid:
        # Structure check failed — print why and skip this chat
        print(f"  [SKIP] Structure invalid: {error_msg}")
        return None

    # Check safety rules (scans assistant messages only, as designed in Task 1)
    safety = check_safety(messages, chat_id, use_llm=use_llm_safety)
    if safety.flagged:
        # Safety check failed — print which rules fired and skip this chat
        print(f"  [SKIP] Safety flagged: {', '.join(safety.reasons)}")
        return None

    # All checks passed — build the final chat dict with metadata
    chat = {
        "id": chat_id,            # Unique identifier for this generated chat
        "topic": topic,           # Store the topic so we know how it was generated
        "source": "generated",    # Mark as AI-generated (not human-written)
        "word_count": count_words(messages),   # Pre-compute word count for convenience
        "messages": messages,     # The actual conversation turns
    }

    print(f"  [OK] Generated: {chat_id} ({chat['word_count']} words)")
    return chat   # Return the valid, safe chat dict


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — BATCH GENERATOR WITH RETRY
# Loops through all topics and generates the requested number of chats.
# If a generation fails (bad JSON, safety flag), it retries up to max_retries.
# ══════════════════════════════════════════════════════════════════════════════

def generate_chats(
    topics: list,
    count_per_topic: int = 1,
    max_retries: int = 3,
    use_llm_safety: bool = False,
    delay_seconds: float = 1.5,
) -> list:
    """
    Generates chats for all topics, with retry logic for failures.

    Args:
        topics           : list of topic strings to generate chats for
        count_per_topic  : how many chats to generate per topic (default 1)
        max_retries      : how many times to retry a failed generation (default 3)
        use_llm_safety   : whether to also run Gemini LLM safety check (slower)
        delay_seconds    : seconds to wait between API calls (avoids rate limits)

    Returns:
        List of valid, safe chat dicts ready to save.
    """
    # Build the LangChain chain once — reuse it for all generations
    chain = build_chain(temperature=0.8)

    good_chats = []    # Accumulates all successfully generated + validated chats
    total_tried = 0    # Counter for all attempts (including retries)
    total_failed = 0   # Counter for all failed attempts

    # Loop over every topic
    for topic_idx, topic in enumerate(topics):
        generated_for_topic = 0   # How many good chats we've made for this topic

        # Generate count_per_topic good chats for this topic
        for attempt in range(count_per_topic * max_retries):
            # Stop if we've already got enough good chats for this topic
            if generated_for_topic >= count_per_topic:
                break

            total_tried += 1

            # Build a unique ID for this chat: gen_topic01_attempt01
            chat_id = f"gen_topic{topic_idx+1:02d}_attempt{attempt+1:02d}"

            # Try to generate one chat
            chat = generate_one_chat(chain, topic, chat_id, use_llm_safety)

            if chat is not None:
                # Generation succeeded — add to our good list
                good_chats.append(chat)
                generated_for_topic += 1
            else:
                # Generation failed — count the failure
                total_failed += 1

            # Add a small delay between API calls to avoid hitting rate limits
            # Gemini free tier allows 15 requests/minute — 1.5s gap = ~40 req/min max
            time.sleep(delay_seconds)

        # Summary for this topic
        if generated_for_topic < count_per_topic:
            print(f"  [WARN] Only got {generated_for_topic}/{count_per_topic} chats for: {topic[:50]}")

    # Final summary
    print(f"\n[SUMMARY] Tried {total_tried} generations | Good: {len(good_chats)} | Failed: {total_failed}")
    return good_chats


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — MAIN ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Entry point. Parses CLI arguments and runs the generation pipeline.
    """
    # ── Argument parser ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Vedaz Chat Generator — creates new chats using LangChain + Gemini"
    )
    parser.add_argument(
        "--topics", "-t",
        nargs="+",        # Accepts one or more topic strings
        default=None,     # If not provided, uses DEFAULT_TOPICS list above
        help='Topic(s) to generate. E.g. --topics "career, Hindi" "marriage, skeptical"'
    )
    parser.add_argument(
        "--count", "-c",
        type=int,
        default=1,        # Default: generate 1 chat per topic
        help="Number of chats to generate per topic (default: 1)"
    )
    parser.add_argument(
        "--out", "-o",
        default="data/generated_chats.jsonl",   # Output path for generated chats
        help="Output JSONL file path (default: data/generated_chats.jsonl)"
    )
    parser.add_argument(
        "--llm-safety",
        action="store_true",   # Flag — runs Gemini safety check on generated chats too
        help="Also run Gemini LLM safety check on generated chats (uses extra API quota)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,      # Seconds between API calls — keep under rate limit
        help="Seconds to wait between API calls (default: 1.5)"
    )
    args = parser.parse_args()

    # Use provided topics or fall back to the default topic list
    topics = args.topics if args.topics else DEFAULT_TOPICS

    print("=" * 65)
    print("  VEDAZ CHAT GENERATOR — LangChain + Gemini")
    print("=" * 65)
    print(f"  Topics     : {len(topics)}")
    print(f"  Per topic  : {args.count}")
    print(f"  Target     : {len(topics) * args.count} total chats")
    print(f"  Output     : {args.out}")
    print(f"  LLM safety : {'ON' if args.llm_safety else 'OFF (keyword only)'}")
    print("=" * 65)

    # Run the batch generator
    good_chats = generate_chats(
        topics=topics,
        count_per_topic=args.count,
        use_llm_safety=args.llm_safety,
        delay_seconds=args.delay,
    )

    if not good_chats:
        # Nothing was generated — check API key and try again
        print("[ERROR] No valid chats were generated. Check your GEMINI_API_KEY.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # Save all good chats to JSONL (one JSON object per line)
    save_jsonl(good_chats, args.out)

    # Print a final preview of the first generated chat
    print(f"\n[PREVIEW] First generated chat:")
    print("-" * 65)
    first = good_chats[0]
    print(f"  ID    : {first['id']}")
    print(f"  Topic : {first['topic']}")
    print(f"  Words : {first['word_count']}")
    for msg in first["messages"]:
        role = msg["role"].upper()
        content = msg["content"][:120]   # Show first 120 chars of each message
        # Use sys.stdout.write with error replacement to prevent Windows console crashes
        sys.stdout.write(f"  [{role}]: {content.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding)}...\n")
    print("-" * 65)

    print(f"\n[DONE] {len(good_chats)} valid chats saved to: {args.out}")


# ── Script entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
