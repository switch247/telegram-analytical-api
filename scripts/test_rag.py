import sys
import logging
from pathlib import Path
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.pipeline.rag import RAGSystem

def test_rag():
    print("\n" + "="*80)
    print("RAG SYSTEM QUALITATIVE EVALUATION")
    print("="*80 + "\n")

    # Choose mode: 'parquet' or 'chroma'
    mode = 'chroma'  # set to 'parquet' or 'chroma'
    try:
        if mode == 'parquet':
            parquet_path = str(PROJECT_ROOT / "vector_store" / "complaint_embeddings.parquet")
            rag = RAGSystem(
                parquet_path=parquet_path,
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                llm_model_name="google/flan-t5-base"
            )
        elif mode == 'chroma':
            chromadb_dir = str(PROJECT_ROOT / "vector_store")
            rag = RAGSystem(
                chromadb_dir=chromadb_dir,
                chromadb_collection="complaint_embeddings",
                embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                llm_model_name="google/flan-t5-base"
            )
        else:
            raise ValueError("Unknown mode for RAGSystem initialization.")
        print("RAG System Initialized.\n")
    except Exception as e:
        print(f"Failed to initialize RAG System: {e}")
        return

    # Define Evaluation Questions (Expanded to 8 for better coverage)
    questions = [
        "What are the common complaints regarding Credit Cards?",
        "Why are people struggling with Personal Loans?",
        "What issues are reported about money transfers?",
        "What is the most frequent issue with Savings Accounts?",
        "Describe the customer sentiment regarding customer service.",
        "What are the main problems with mortgage applications?",
        "How do customers feel about overdraft fees?",
        "Are there recurring issues with online banking platforms?"
    ]

    # Print scoring guide
    print("Scoring Guide (1-5):")
    print("1: Incorrect or not grounded\n2: Poor, mostly irrelevant\n3: Partially correct, some relevant info\n4: Mostly correct, minor omissions\n5: Fully correct, complete, well-grounded\n")

    results = []
    import traceback
    for q in questions:
        try:
            print(f"[*] Processing: {q}...", flush=True)
            response = rag.ask(q, k=5)
            print(f"[*] Done processing: {q}")
            answer = response.get('answer', "No answer generated").strip().replace("\n", " ")
            sources = response.get('sources', [])
            source_texts = []
            for i, doc in enumerate(sources[:2]):
                clean_source = doc.replace("\n", " ")
                if len(clean_source) > 40:
                    clean_source = clean_source[:37] + "..."
                source_texts.append(f"[{i+1}] {clean_source}")
            sources_display = " | ".join(source_texts) if source_texts else "None"
            results.append({
                "Question": q,
                "Generated Answer": answer,
                "Retrieved Sources": " || ".join(sources[:2]),
                "Quality Score": "",
                "Comments/Analysis": ""
            })
        except Exception as e:
            print(f"Error processing question '{q}': {e}")
            traceback.print_exc()

    # Print markdown table
    print("\n---\n\n## Evaluation Results (Markdown Table)\n")
    print("| Question | Generated Answer | Retrieved Sources (Top 2) | Quality Score | Comments/Analysis |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['Question']} | {r['Generated Answer']} | {r['Retrieved Sources']} | {r['Quality Score']} | {r['Comments/Analysis']} |")

    # Also save to CSV for later reporting
    df_results = pd.DataFrame(results)
    output_path = PROJECT_ROOT / "outputs" / "qualitative_evaluation.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print(f"Evaluation complete. Results also saved to: {output_path}")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_rag()
