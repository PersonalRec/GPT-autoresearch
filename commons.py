"""
commons.py — Knowledge base read/write interface for autoresearch agents.

Both a Python library and a CLI tool. Experiment cards are stored as JSON files
in knowledge/cards/ and indexed in knowledge/index.json.

Usage (CLI):
    uv run commons.py read-brief
    uv run commons.py read-meta
    uv run commons.py write-card --commit abc1234 --hypothesis "..." ...
    uv run commons.py update-index
    uv run commons.py coverage
"""

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Default knowledge directory: knowledge/ relative to this script
# ---------------------------------------------------------------------------
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"

# ---------------------------------------------------------------------------
# Platform detection — try to import from platform_utils, fall back gracefully
# ---------------------------------------------------------------------------

try:
    from platform_utils import get_device_info
except ImportError:

    def get_device_info() -> dict:
        """Fallback when platform_utils is not available."""
        return {"gpu": "unknown", "ram_gb": 0, "framework": "unknown"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs(knowledge_dir: Path) -> None:
    """Make sure the cards/ directory exists."""
    (knowledge_dir / "cards").mkdir(parents=True, exist_ok=True)


def _safe_timestamp(ts: str) -> str:
    """Replace colons with dashes so the timestamp is filename-safe."""
    return ts.replace(":", "-")


def _short_id(commit_id: str, timestamp: str) -> str:
    """Generate a 7-char hex id from commit + timestamp."""
    raw = f"{commit_id}-{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:7]


def _card_filename(timestamp: str, commit_id: str) -> str:
    """Build the card filename: {safe_timestamp}_{commit_id}.json"""
    safe_ts = _safe_timestamp(timestamp)
    return f"{safe_ts}_{commit_id}.json"


# ---------------------------------------------------------------------------
# Library functions
# ---------------------------------------------------------------------------


def create_card(
    knowledge_dir: Path | str,
    commit_id: str,
    hypothesis: str,
    config_diff: dict,
    results: dict,
    status: str,
    lesson: str,
    tags: list[str],
    prior_knowledge_used: list[str] | None = None,
) -> dict:
    """Create a new experiment card and write it to disk.

    Returns the card dict that was written.
    """
    knowledge_dir = Path(knowledge_dir)
    _ensure_dirs(knowledge_dir)

    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    card_id = _short_id(commit_id, timestamp)
    platform_info = get_device_info()

    card = {
        "id": card_id,
        "timestamp": timestamp,
        "platform": platform_info,
        "parent_commit": commit_id,
        "hypothesis": hypothesis,
        "config_diff": config_diff,
        "results": results,
        "status": status,
        "lesson": lesson,
        "tags": tags,
        "prior_knowledge_used": prior_knowledge_used or [],
    }

    filename = _card_filename(timestamp, commit_id)
    card_path = knowledge_dir / "cards" / filename
    card_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")

    # Auto-update the index after creating a card
    update_index(knowledge_dir)

    return card


def load_cards(knowledge_dir: Path | str) -> list[dict]:
    """Load all experiment cards, sorted by timestamp newest first."""
    knowledge_dir = Path(knowledge_dir)
    cards_dir = knowledge_dir / "cards"
    if not cards_dir.exists():
        return []

    cards = []
    for fpath in cards_dir.glob("*.json"):
        try:
            card = json.loads(fpath.read_text(encoding="utf-8"))
            cards.append(card)
        except (json.JSONDecodeError, OSError):
            continue

    cards.sort(key=lambda c: c.get("timestamp", ""), reverse=True)
    return cards


def get_recent_cards(knowledge_dir: Path | str, n: int = 20) -> list[dict]:
    """Return the *n* most recent experiment cards."""
    return load_cards(knowledge_dir)[:n]


def get_cards_by_tag(knowledge_dir: Path | str, tag: str) -> list[dict]:
    """Return all cards that include *tag* in their tags list."""
    return [c for c in load_cards(knowledge_dir) if tag in c.get("tags", [])]


def get_platform_findings(knowledge_dir: Path | str, gpu_substring: str) -> list[dict]:
    """Return cards whose platform gpu field contains *gpu_substring*."""
    sub = gpu_substring.lower()
    return [
        c
        for c in load_cards(knowledge_dir)
        if sub in c.get("platform", {}).get("gpu", "").lower()
    ]


def update_index(knowledge_dir: Path | str) -> None:
    """Regenerate index.json from the card files on disk."""
    knowledge_dir = Path(knowledge_dir)
    cards = load_cards(knowledge_dir)

    index = {
        "version": 1,
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cards": [
            {
                "id": c["id"],
                "timestamp": c["timestamp"],
                "status": c.get("status", "unknown"),
                "tags": c.get("tags", []),
                "parent_commit": c.get("parent_commit", ""),
            }
            for c in cards
        ],
    }

    index_path = knowledge_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def load_index(knowledge_dir: Path | str) -> dict:
    """Load and return the index.json contents."""
    knowledge_dir = Path(knowledge_dir)
    index_path = knowledge_dir / "index.json"
    if not index_path.exists():
        return {"version": 1, "last_updated": None, "cards": []}
    return json.loads(index_path.read_text(encoding="utf-8"))


def get_coverage_map(knowledge_dir: Path | str) -> dict:
    """Build a coverage map: {tag: {count, kept, best_delta, best_bpb}}.

    Aggregates across all cards grouped by tag.
    """
    cards = load_cards(knowledge_dir)
    coverage: dict[str, dict] = {}

    for card in cards:
        results = card.get("results", {})
        delta = results.get("delta")
        bpb = results.get("val_bpb")

        for tag in card.get("tags", []):
            if tag not in coverage:
                coverage[tag] = {
                    "count": 0,
                    "kept": 0,
                    "best_delta": None,
                    "best_bpb": None,
                }

            entry = coverage[tag]
            entry["count"] += 1

            if card.get("status") == "keep":
                entry["kept"] += 1

            # best_delta: most negative (biggest improvement) wins
            if delta is not None:
                if entry["best_delta"] is None or delta < entry["best_delta"]:
                    entry["best_delta"] = delta

            # best_bpb: lowest is best
            if bpb is not None:
                if entry["best_bpb"] is None or bpb < entry["best_bpb"]:
                    entry["best_bpb"] = bpb

    return coverage


def get_meta_synthesis(knowledge_dir: Path | str) -> str:
    """Read and return the meta-synthesis.md file content."""
    knowledge_dir = Path(knowledge_dir)
    meta_path = knowledge_dir / "synthesis" / "meta-synthesis.md"
    if not meta_path.exists():
        return "No meta-synthesis available yet."
    return meta_path.read_text(encoding="utf-8")


def get_open_questions(knowledge_dir: Path | str) -> list[str]:
    """Extract open questions from meta-synthesis.md.

    Looks for lines starting with '- ' or '* ' under a heading containing
    'open question' (case-insensitive), or lines starting with '?' or
    containing a question mark after a bullet.
    """
    content = get_meta_synthesis(knowledge_dir)
    if content == "No meta-synthesis available yet.":
        return []

    questions: list[str] = []
    in_open_questions_section = False

    for line in content.splitlines():
        stripped = line.strip()

        # Detect open-questions heading
        if re.match(r"^#{1,6}\s+.*open.question", stripped, re.IGNORECASE):
            in_open_questions_section = True
            continue

        # Another heading ends the section
        if in_open_questions_section and re.match(r"^#{1,6}\s+", stripped):
            in_open_questions_section = False
            continue

        # Collect bullet items in the open questions section
        if in_open_questions_section and re.match(r"^[-*]\s+", stripped):
            question_text = re.sub(r"^[-*]\s+", "", stripped)
            questions.append(question_text)

    return questions


def read_brief(knowledge_dir: Path | str) -> str:
    """Produce an agent-facing summary of the knowledge base.

    Includes: coverage map, recent cards, and open questions.
    """
    knowledge_dir = Path(knowledge_dir)
    parts: list[str] = []

    # --- Coverage map ---
    coverage = get_coverage_map(knowledge_dir)
    parts.append("## Coverage Map")
    if not coverage:
        parts.append("No experiments recorded yet.")
    else:
        for tag, stats in sorted(coverage.items()):
            delta_str = f"{stats['best_delta']:.6f}" if stats["best_delta"] is not None else "n/a"
            bpb_str = f"{stats['best_bpb']:.6f}" if stats["best_bpb"] is not None else "n/a"
            parts.append(
                f"- {tag}: {stats['count']} experiments, "
                f"{stats['kept']} kept, "
                f"best_delta={delta_str}, "
                f"best_bpb={bpb_str}"
            )

    # --- Recent cards ---
    recent = get_recent_cards(knowledge_dir, n=5)
    parts.append("")
    parts.append("## Recent Experiments")
    if not recent:
        parts.append("No recent experiments.")
    else:
        for card in recent:
            results = card.get("results", {})
            bpb = results.get("val_bpb", "n/a")
            delta = results.get("delta", "n/a")
            parts.append(
                f"- [{card['id']}] {card['hypothesis'][:80]} "
                f"| status={card['status']} bpb={bpb} delta={delta}"
            )

    # --- Open questions ---
    questions = get_open_questions(knowledge_dir)
    parts.append("")
    parts.append("## Open Questions")
    if not questions:
        parts.append("No open questions identified yet.")
    else:
        for q in questions:
            parts.append(f"- {q}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Knowledge base interface for autoresearch agents.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # read-brief
    subparsers.add_parser("read-brief", help="Print knowledge summary")

    # read-meta
    subparsers.add_parser("read-meta", help="Print full meta-synthesis")

    # write-card
    wc = subparsers.add_parser("write-card", help="Create a new experiment card")
    wc.add_argument("--commit", required=True, help="Parent commit id")
    wc.add_argument("--hypothesis", required=True, help="Experiment hypothesis")
    wc.add_argument("--result", required=True, type=float, help="val_bpb result")
    wc.add_argument("--delta", required=True, type=float, help="Delta from baseline")
    wc.add_argument("--peak-memory", required=True, type=float, help="Peak VRAM in MB")
    wc.add_argument("--training-seconds", required=True, type=float, help="Training time in seconds")
    wc.add_argument("--num-steps", required=True, type=int, help="Number of training steps")
    wc.add_argument("--status", required=True, choices=["keep", "revert", "inconclusive"], help="Experiment status")
    wc.add_argument("--lesson", required=True, help="Lesson learned")
    wc.add_argument("--tags", required=True, help="Comma-separated tags")
    wc.add_argument("--config-diff", default="{}", help="JSON string of config diff")

    # update-index
    subparsers.add_parser("update-index", help="Regenerate index.json")

    # coverage
    subparsers.add_parser("coverage", help="Print coverage map")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    knowledge_dir = Path(os.environ.get("KNOWLEDGE_DIR", str(KNOWLEDGE_DIR)))

    if args.command == "read-brief":
        print(read_brief(knowledge_dir))

    elif args.command == "read-meta":
        print(get_meta_synthesis(knowledge_dir))

    elif args.command == "write-card":
        try:
            config_diff = json.loads(args.config_diff)
        except json.JSONDecodeError:
            print("Error: --config-diff must be valid JSON", file=sys.stderr)
            sys.exit(1)

        results = {
            "val_bpb": args.result,
            "delta": args.delta,
            "peak_vram_mb": args.peak_memory,
            "training_seconds": args.training_seconds,
            "num_steps": args.num_steps,
        }

        card = create_card(
            knowledge_dir=knowledge_dir,
            commit_id=args.commit,
            hypothesis=args.hypothesis,
            config_diff=config_diff,
            results=results,
            status=args.status,
            lesson=args.lesson,
            tags=[t.strip() for t in args.tags.split(",")],
        )
        print(f"Card created: {card['id']}")

    elif args.command == "update-index":
        update_index(knowledge_dir)
        index = load_index(knowledge_dir)
        print(f"Index updated: {len(index['cards'])} cards")

    elif args.command == "coverage":
        coverage = get_coverage_map(knowledge_dir)
        if not coverage:
            print("No experiments recorded yet.")
        else:
            for tag, stats in sorted(coverage.items()):
                delta_str = f"{stats['best_delta']:.6f}" if stats["best_delta"] is not None else "n/a"
                bpb_str = f"{stats['best_bpb']:.6f}" if stats["best_bpb"] is not None else "n/a"
                print(
                    f"{tag}: {stats['count']} experiments, "
                    f"{stats['kept']} kept, "
                    f"best_delta={delta_str}, "
                    f"best_bpb={bpb_str}"
                )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
