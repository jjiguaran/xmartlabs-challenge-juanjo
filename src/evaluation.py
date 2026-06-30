import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.config import TOP_K
from src.rag import VectorStoreRAGPipeline
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = Path(__file__).resolve().parent.parent / "data" / "ground_truth.json"
EVAL_LOGS_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_logs.json"
DEFAULT_K = TOP_K


def load_ground_truth(path: Path = GROUND_TRUTH_PATH) -> List[Dict]:
    """Load the ground-truth evaluation data from a JSON file.

    Expected structure:
        [
            {
                "question": "...",
                "keyword_used": "...",
                "expected_chunks": ["some text", "other text", ...],
                "source": "..."
            }
        ]
    """
    if not path.exists():
        logger.error(f"Ground-truth file not found: {path}")
        sys.exit(1)

    with open(path, "r") as f:
        data = json.load(f)

    if not data:
        logger.error("Ground-truth file is empty")
        sys.exit(1)

    return data


def pick_question(entries: List[Dict]) -> Dict:
    """Show the user available questions and let them pick one interactively."""
    print()
    print("Available evaluation questions:")
    print("-------------------------------")
    for i, entry in enumerate(entries, start=1):
        print(f"  {i}. {entry['question']}")
    print()

    while True:
        try:
            choice = input(
                f"Select a question to evaluate (1-{len(entries)}): "
            ).strip()
            idx = int(choice) - 1
            if 0 <= idx < len(entries):
                return entries[idx]
            print(
                f"Invalid selection. Please enter a number between 1 and {len(entries)}."
            )
        except ValueError:
            print("Invalid input. Please enter a number.")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)


class RAGEvaluator:

    def __init__(
        self, vector_store_path: str = "vector_store", top_k: int = DEFAULT_K
    ):
        self.vector_store_path = vector_store_path
        self.top_k = top_k
        self.store: Optional[VectorStore] = None
        self.rag_pipeline: Optional[VectorStoreRAGPipeline] = None

    def load_vector_store(self) -> bool:
        """Load the vector store from disk."""
        self.store = VectorStore(store_path=self.vector_store_path)
        loaded = self.store.load()
        if loaded:
            logger.info(
                f"Loaded vector store with {len(self.store.documents)} documents"
            )

            # Create the RAG pipeline (uses VectorStoreRAGPipeline from rag.py)
            self.rag_pipeline = VectorStoreRAGPipeline(
                self.store, top_k=self.top_k
            )
        else:
            logger.warning("Could not load vector store")
        return loaded

    def compute_recall_at_k(
        self,
        question: str,
        expected_chunks: List[str],
        k: int = DEFAULT_K,
    ) -> Dict[str, object]:
        """Retrieve top-K documents via the RAG pipeline and compute Recall @ K.

        The RAG pipeline returns the actual text snippets of the retrieved
        documents.  We compare those directly against the expected chunk texts.

        Recall@K = |expected_texts ∩ retrieved_texts| / |expected_texts|
        """
        if self.rag_pipeline is None:
            logger.error("RAG pipeline is not initialised — cannot retrieve")
            return {"error": "RAG pipeline not loaded"}

        # Retrieve top-K text snippets using the RAG pipeline
        retrieved_texts: List[str] = self.rag_pipeline.retrieve(question)

        # Compute intersection: which expected texts appear in the retrieved set
        retrieved_set = set(retrieved_texts)
        relevant_retrieved_texts = [
            t for t in expected_chunks if t in retrieved_set
        ]

        num_relevant = len(expected_chunks)
        num_relevant_retrieved = len(relevant_retrieved_texts)
        recall = num_relevant_retrieved / num_relevant if num_relevant > 0 else 0.0

        key = f"recall_at_{k}"
        logger.info(
            f"{key}: {recall:.4f}  "
            f"({num_relevant_retrieved}/{num_relevant} relevant chunks retrieved)"
        )

        return {
            key: recall,
            "retrieved_texts": retrieved_texts,
            "relevant_retrieved": relevant_retrieved_texts,
            "expected_chunks": expected_chunks,
            "k": k,
        }

    def evaluate_from_ground_truth(
        self,
        entry: Dict,
        k: int = DEFAULT_K,
    ) -> Dict[str, object]:
        """Run recall evaluation for a single ground-truth entry."""
        question = entry["question"]
        expected_chunks = entry["expected_chunks"]
        keyword_used = entry.get("keyword_used", question)
        source = entry.get("source", "unknown")

        print(f"\nEvaluating question: {question}")
        print(f"  Source document : {source}")
        print(f"  Keyword used    : {keyword_used}")
        print(f"  Expected chunks : {len(expected_chunks)} text(s)")
        print(f"  Top-K           : {k}")
        print()

        results = self.compute_recall_at_k(question, expected_chunks, k=k)

        # Pretty-print results
        recall_key = f"recall_at_{k}"
        print("--- Results ---")
        print(f"  {recall_key}: {results[recall_key]:.4f}")
        print(f"  Relevant retrieved : {results['relevant_retrieved']}")
        print(f"  Total relevant     : {len(results['expected_chunks'])}")
        print()

        return results

    def log_evaluation_run(self, entry: Dict, results: Dict, k: int) -> None:
        """Append an evaluation result to the eval_logs.json file."""
        log_entry = {
            "date": datetime.datetime.now().isoformat(),
            "question": entry["question"],
            "k": k,
            f"recall_at_{k}": results[f"recall_at_{k}"],
        }
        EVAL_LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        logs = []
        if EVAL_LOGS_PATH.exists():
            try:
                with open(EVAL_LOGS_PATH, "r") as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, IOError):
                logs = []
        logs.append(log_entry)
        with open(EVAL_LOGS_PATH, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"💾 Evaluation logged to {EVAL_LOGS_PATH}")

    def run_interactive_evaluation(self, k: int = DEFAULT_K) -> None:
        """Load ground truth, let user pick a question, and evaluate recall."""
        # Load vector store first
        loaded = self.load_vector_store()
        if not loaded:
            logger.error("Cannot run evaluation without a vector store")
            sys.exit(1)

        # Load ground-truth questions
        ground_truth = load_ground_truth()

        # Let user pick one
        entry = pick_question(ground_truth)

        # Evaluate
        results = self.evaluate_from_ground_truth(entry, k=k)

        # Log the run
        self.log_evaluation_run(entry, results, k=k)