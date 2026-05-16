import argparse
from inverted_index import InvertedIndex
from tokenizer import tokenizer
from search_utils import *


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="")

    tf_parser = subparsers.add_parser("tf", help="")
    tf_parser.add_argument("doc_id", type=int, help="")
    tf_parser.add_argument("term", type=str, help="")

    idf_parser = subparsers.add_parser("idf", help="")
    idf_parser.add_argument("term", type=str, help="")

    tfidf_parser = subparsers.add_parser("tfidf", help="")
    tfidf_parser.add_argument("doc_id", type=int, help="")
    tfidf_parser.add_argument("term", type=str, help="")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_k1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")


    args = parser.parse_args()
    new_inverted_index = InvertedIndex()
    match args.command:
        case "bm25search":
            new_inverted_index.load()
            results = new_inverted_index.bm25_search(args.query, 5)
            for doc_id in results:
                title = new_inverted_index.docmap[doc_id]['title']
                print(f'({doc_id}) {title} - Score: {results[doc_id]:.2f}')
        case "search":
            new_inverted_index.load()
            token_query = tokenizer(args.query)
            index_list = []
            for token in token_query:
                if len(index_list) >= 5:
                    break
                docs = new_inverted_index.get_documents(token)
                for doc in docs:
                    index_list.append(doc)
                    if len(index_list) >= 5:
                        break
            for index in index_list:
                print(f'{index}. {new_inverted_index.docmap[index]['title']}')
                
        case "build":
            new_inverted_index.build()
            new_inverted_index.save()
        case "tf":
            new_inverted_index.load()
            print(new_inverted_index.get_tf(args.doc_id, args.term))
        case "idf":
            new_inverted_index.load()
            idf = new_inverted_index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            new_inverted_index.load()
            tf = new_inverted_index.get_tf(args.doc_id, args.term)
            idf = new_inverted_index.get_idf(args.term)
            tf_idf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case "bm25idf":
            new_inverted_index.load()
            bm25idf = new_inverted_index.get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            new_inverted_index.load()
            bm25tf = new_inverted_index.get_bm25_tf(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()