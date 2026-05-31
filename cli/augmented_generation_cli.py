import argparse
from rrf_search_handler import rrf_search_handler
from load_movies import movie_loader
from gemini_funcs import gemini_RAG_answer, gemini_summarizer, gemini_citator


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize search results")
    summarize_parser.add_argument("query", type=str, help="Search query for summarization")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Number of retrieved documents to use for summarization")

    citations_parser = subparsers.add_parser("citations", help="Answer query with citations")
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    citations_parser.add_argument("--limit", type=int, default=5, help="Number of retrieved documents to use for citations")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            docs = rrf_search_handler(query, k=60, limit=5, for_rag=True)
            answer = gemini_RAG_answer(query, "\n".join(docs))
            print(f"Answer:\n{answer}")
        
        case "summarize":
            query = args.query
            docs = rrf_search_handler(query, k=60, limit=args.limit, for_rag=True)
            answer = gemini_summarizer(query, "\n".join(docs))
            print(f"Summary:\n{answer}")
        case "citations":
            query = args.query
            docs = rrf_search_handler(query, k=60, limit=args.limit, for_rag=True)
            answer = gemini_citator(query, "\n".join(docs))
            print(f"Answer with citations:\n{answer}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()