import argparse
from golden_dataset_loader import golden_dataset_loader
from hybrid_search import rrf_search    
from utils import wd    
import os   

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    test_cases = golden_dataset_loader(os.path.join(wd, "data", "golden_dataset.json"))
    for case in test_cases:
        query = case["query"]
        relevant_titles = case["relevant_docs"]
        results = rrf_search(query, k=60, limit=limit, for_eval=True)
        results_titles = [result['title'] for result in results]

        relevant_retrieved = len(relevant_titles)

        overlap = len(set(relevant_titles) & set(results_titles))
        total_retrieved = len(results_titles)

        precision = overlap / total_retrieved if total_retrieved > 0 else 0
        print(f"\nQuery: {query}")  
        print(f"Relevant Titles: {relevant_titles}")
        print(f"Retrieved Titles: {results_titles}")
        print(f"Precision@{limit}: {precision:.4f}")
        
        recall = overlap / relevant_retrieved if relevant_retrieved > 0 else 0
        print(f"Recall@{limit}: {recall:.4f}")

        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print(f"F1 Score: {f1_score:.4f}")
        
        
    

if __name__ == "__main__":
    main()