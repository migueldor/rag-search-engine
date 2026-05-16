
import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search, semantic_chunking, embed_chunks, search_chunked

def chunking(text, size, overlap):
    if size == None:
        size = 200
    if overlap == None:
        overlap = 0
    data = text.split(" ")

    #chunks = [data[i : i + int(size)] for i in range(0, len(data), int(size))]
    chunks = []
    for i in range(0, len(data), int(size)):
        if i < overlap:
            chunk = data[i : i + int(size)]
        else:
            chunk = data[i - overlap : i + int(size)]
        chunks.append(chunk)
    j = 1
    print(f'Chunking {len(text)} characters')
    for chunk in chunks:
        print(f"{j}. {" ".join(chunk)}")
        j += 1


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    verify_parser = subparsers.add_parser("verify", help="Verifies the semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generates enbedded text from input")
    embed_text_parser.add_argument("text", type=str, help="text to embed")

    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verifies the embeddings")

    embed_query_parser = subparsers.add_parser("embed_query", help="embeds the query")
    embed_query_parser.add_argument("query", type=str, help="text to embed")

    search_parser = subparsers.add_parser("search", help="Search movies")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, help="")

    chunk_parser = subparsers.add_parser("chunk", help="")
    chunk_parser.add_argument("text", type=str, help="")
    chunk_parser.add_argument("--chunk-size", type=int, help="")
    chunk_parser.add_argument("--overlap", type=int, help="")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="")
    semantic_chunk_parser.add_argument("text", type=str, help="")
    semantic_chunk_parser.add_argument("--max-chunk-size", default=4, type=int, help="")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="")

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search movies")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, default=5, help="")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case "chunk":
            chunking(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            chunks = semantic_chunking(args.text, args.max_chunk_size, args.overlap)
            j = 1
            print(f'Semantically chunking {len(args.text)} characters')
            for chunk in chunks:
                print(f"{j}. {" ".join(chunk)}")
                j += 1
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            search_chunked(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()