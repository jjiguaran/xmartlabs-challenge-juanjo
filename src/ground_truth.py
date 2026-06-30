#!/usr/bin/env python3
"""
Vector Store Ground Truth Builder — Interactive tool to build a ground_truth.json file.

For a given question and keyword, the tool shows each matching chunk one at a time
and lets you decide (y/n) whether to include it as relevant context.

Usage:
    python store_overview.py                                           # Interactive mode
    python store_overview.py --path /custom/path                       # Custom store path
    python store_overview.py --output my_ground_truth.json             # Custom output file
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_store(store_path: str):
    """Load the VectorStore and return it."""
    from src.vector_store import VectorStore

    store = VectorStore(store_path=store_path)
    success = store.load()
    if not success:
        print(f"❌ Failed to load vector store from: {store_path}")
        sys.exit(1)
    return store


def show_stats(store):
    """Display summary statistics of the vector store."""
    stats = store.get_stats()
    print("=" * 60)
    print("📊  VECTOR STORE SUMMARY")
    print("=" * 60)
    print(f"  Store path        : {store.store_path}")
    print(f"  Embedding model   : {store.embedding_model_name}")
    print(f"  Documents (chunks): {stats['num_documents']}")
    print(f"  Metadata entries  : {stats['num_chunks']}")
    print(f"  FAISS index size  : {stats['index_size']}")
    print("=" * 60)
    print()


def search_by_keywords(store, keywords: list[str], match_all: bool = False):
    """
    Search for chunks that contain the given keywords.

    Args:
        store: The vector store instance.
        keywords: List of keywords to search for (case-insensitive).
        match_all: If True, require ALL keywords to be present (AND logic).
                   If False, match any keyword (OR logic).

    Returns:
        List of (text, metadata, matched_keywords) tuples for matching chunks.
    """
    if not keywords:
        return []

    keywords_lower = [kw.lower() for kw in keywords]
    results = []

    for idx, text in enumerate(store.documents):
        text_lower = text.lower()
        matched = [kw for kw in keywords_lower if kw in text_lower]

        if match_all:
            if len(matched) == len(keywords):
                results.append((text, store.metadata[idx], matched))
        else:
            if matched:
                results.append((text, store.metadata[idx], matched))

    return results


def load_existing_ground_truth(filepath: str) -> list:
    """Load existing ground_truth.json or return an empty list."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"⚠️  Warning: Could not read {filepath}, starting fresh.")
    return []


def save_ground_truth(filepath: str, data: list):
    """Save the ground truth data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved to {filepath}")


def run_interactive(output_file: str, store_path: str):
    """Run the interactive ground truth builder."""

    # Load vector store
    print(f"📂 Loading vector store from: {store_path}\n")
    store = load_store(store_path)
    show_stats(store)

    # Load existing ground truth
    ground_truth = load_existing_ground_truth(output_file)

    # Ask for question
    print("─" * 60)
    question = input("❓ Enter the question: ").strip()
    while not question:
        question = input("❓ Question cannot be empty. Enter the question: ").strip()

    # Ask for keyword
    keyword = input("🔑 Enter the keyword to search for: ").strip()
    while not keyword:
        keyword = input("🔑 Keyword cannot be empty. Enter the keyword: ").strip()

    print()
    print(f"🔍 Searching for chunks containing \"{keyword}\"...")
    results = search_by_keywords(store, [keyword], match_all=False)

    if not results:
        print("  No chunks matched the specified keyword.")
        print()
        return

    print(f"  Found {len(results)} matching chunk(s).")
    print()

    # Collect included chunks during the session
    included_chunks = []
    source = ""

    for i, (text, meta, matched_kws) in enumerate(results):
        # Display the chunk
        print("─" * 60)
        print(f"📄 Chunk {i + 1} of {len(results)}")
        print("─" * 60)
        print(f"  Matched keywords: {', '.join(matched_kws)}")
        print(f"  Text  : {text}")
        print(f"  Meta  :")
        for key, value in meta.items():
            if key == "text":
                continue
            print(f"    {key}: {value}")

        # Ask for decision
        print()
        while True:
            choice = input("✅ Include this chunk? (y/n, or q to quit): ").strip().lower()
            if choice in ("y", "yes"):
                included_chunks.append(text)
                if not source:
                    source = meta.get("source", "")
                print(f"  ✅ Included (total: {len(included_chunks)})")
                break
            elif choice in ("n", "no"):
                print("  ❌ Skipped")
                break
            elif choice in ("q", "quit"):
                # Save what we have so far before quitting
                if included_chunks:
                    entry = {
                        "question": question,
                        "keyword_used": keyword,
                        "expected_chunks": included_chunks,
                        "source": source,
                    }
                    ground_truth.append(entry)
                    save_ground_truth(output_file, ground_truth)
                print()
                print(f"🚪 Quitting. Included {len(included_chunks)} of {len(results)} chunks.")
                print(f"📁 Ground truth file: {output_file}")
                return
            else:
                print("  Please answer 'y' or 'n' (or 'q' to quit).")

        print()

    # Save final entry with all included chunks for this session
    if included_chunks:
        entry = {
            "question": question,
            "keyword_used": keyword,
            "expected_chunks": included_chunks,
            "source": source,
        }
        ground_truth.append(entry)
        save_ground_truth(output_file, ground_truth)

    # Summary
    print("=" * 60)
    print(f"🏁 Done! Included {len(included_chunks)} of {len(results)} chunks.")
    print(f"📁 Ground truth file: {output_file}")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Vector Store Ground Truth Builder — Interactive tool to build a ground_truth.json file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--path",
        default="vector_store",
        help="Path to the vector store directory (default: vector_store)",
    )
    parser.add_argument(
        "--output",
        default="data/ground_truth.json",
        help="Output JSON file path (default: data/ground_truth.json)",
    )

    args = parser.parse_args()

    store_path = args.path
    if not os.path.isabs(store_path):
        store_path = str(PROJECT_ROOT / store_path)

    output_file = args.output
    if not os.path.isabs(output_file):
        output_file = str(PROJECT_ROOT / output_file)

    run_interactive(output_file, store_path)


if __name__ == "__main__":
    main()

