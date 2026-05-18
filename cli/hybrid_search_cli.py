import argparse
from hybrid_search import normalize, weighted_search, rrf_search

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalizes a list of scores using minmax")
    normalize_parser.add_argument("scores", nargs='+', type=float, help="")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="")
    weighted_search_parser.add_argument("query", type=str, help="")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="")
    rrf_search_parser.add_argument("query", type=str, help="")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Query enhancement method")
    
    args = parser.parse_args()

    match args.command:
        case 'normalize':
            print(normalize(args.scores))
        case "weighted-search":
            weighted_search(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search(args.query, args.k, args.limit, args.enhance, args.rerank_method)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()