import argparse
from inverted_index import InvertedIndex
from tokenizer import tokenizer

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="")

    args = parser.parse_args()
    new_inverted_index = InvertedIndex()
    match args.command:
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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()