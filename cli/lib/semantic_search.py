from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
from load_movies import movie_loader
import json
from itertools import islice


cache_path = "/home/migueldor/rag-search-engine/cache"
embeddings_path = os.path.join(cache_path, "movie_embeddings.npy")
chunk_embeddings_path = os.path.join(cache_path, "chunk_embeddings.npy")
chunk_metadata_path = os.path.join(cache_path, "chunk_metadata.json")
movie_file_path = "/home/migueldor/rag-search-engine/data/movies.json"

def verify_model():
    my_semantic_search = SemanticSearch()
    model = my_semantic_search.model
    print(f'Model loaded: {model}')
    print(f'Max sequence length: {model.max_seq_length}')

def embed_text(text):
    my_semantic_search = SemanticSearch()
    embedding = my_semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    my_semantic_search = SemanticSearch()
    movies = movie_loader(movie_file_path)
    embeddings = my_semantic_search.load_or_create_embeddings(movies)
    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    my_semantic_search = SemanticSearch()
    embedding = my_semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))

def semantic_chunking(text, max_size, overlap):
    stripped_text = text.strip()
    if len(stripped_text) == 0:
        return []
    data = re.split(r"(?<=[.!?])\s+", stripped_text)
    #print(data)
    chunks = []
    last_char = data[0][len(data[0])-1]
    #print(last_char)
    if (len(data) == 1 and not last_char.endswith('.')) and (len(data) == 1 and not last_char.endswith('!')) and (len(data) == 1 and not last_char.endswith('?')):
        #print('we got here')
        chunks.append([data[0]])
    else:
        stripped_data = []
        for sentence in data:
            stripped_sentence = sentence.strip()
            if len(stripped_sentence) > 0:
                stripped_data.append(stripped_sentence) 
        for i in range(0, len(stripped_data), max_size - overlap):
            chunk = data[i : i + max_size]
            chunks.append(chunk)
            if i + max_size >= len(data):
                break
    return chunks
    

def embed_chunks():
    my_chunked_semantic_search = ChunkedSemanticSearch()
    movies = movie_loader(movie_file_path)
    embeddings = my_chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(embeddings)} chunked embeddings")

def search(query, limit=5):
    my_semantic_search = SemanticSearch()
    movies = movie_loader(movie_file_path)
    my_semantic_search.load_or_create_embeddings(movies)
    result = my_semantic_search.search(query, limit)
    print(len(result))
    for i in range(int(limit)):
        print(f'{i+1}. {result[i]['title']} (score: {result[i]['score']})')
        print(f" {result[i]['description']}")
        print(                             )

def search_chunked(query, limit):
    my_chunked_semantic_search = ChunkedSemanticSearch()
    movies = movie_loader(movie_file_path)
    my_chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    result = my_chunked_semantic_search.search_chunks(query, limit)
    print(type(result))
    for i in range(int(limit)):
        ID =  result[i]['id']
        TITLE = result[i]['title']
        SCORE = result[i]['score']
        DOCUMENT = result[i]['document']
        print(f"\n{i+1}. ({ID}) {TITLE} (score: {SCORE:.4f})")
        print(f"   {DOCUMENT}...")
        print(                             )

class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if len(text.strip()) == 0:
            raise ValueError("Text value can't be empty or be just white space")
        enconded_text = self.model.encode([text])
        return enconded_text[0]
    
    def build_embeddings(self, documents):
        self.documents = documents
        doc_reps = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            doc_reps.append(f"{doc['title']}: {doc['description']}")
        encoded_doc_reps = self.model.encode(doc_reps, show_progress_bar = True)
        self.embeddings = encoded_doc_reps
        np.save(embeddings_path, self.embeddings)
        return self.embeddings
        
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query, limit):
        if len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        embedding = self.generate_embedding(query)
        scores = []
        for vector in self.embeddings:
            score = cosine_similarity(vector, embedding)
            scores.append(score)
        docs = self.documents
        scored_docs = [(scores[i], docs[i]) for i in range(len(scores))]
        sorted_scored_docs = sorted(scored_docs, key=lambda x: x[0], reverse=True)
        top_limit = []
        for i in range(int(limit)):
            entry = {}
            entry['score'] = sorted_scored_docs[i][0]
            entry['title'] = sorted_scored_docs[i][1]['title']
            entry['description'] = sorted_scored_docs[i][1]['description']
            top_limit.append(entry)
        return top_limit
    


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        all_chunks = []
        metadata = []
        for doc in self.documents:
            text = doc.get("description", "")
            if len(text) == 0:
                continue
            chunks = semantic_chunking(text, 4, 1)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(" ".join(chunk))
                metadata.append(
                    {
                        'movie_idx': doc['id'],
                        'chunk_idx': chunk_idx,
                        'total_chunks': len(chunks)
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata
        np.save(chunk_embeddings_path, self.chunk_embeddings)
        with open(chunk_metadata_path, 'w', encoding='utf-8') as f:
            json.dump({"chunks": metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(chunk_embeddings_path) and os.path.exists(chunk_metadata_path):
            self.chunk_embeddings = np.load(chunk_embeddings_path)
            with open(chunk_metadata_path, 'r', encoding='utf-8') as f:
                self.chunk_metadata = json.load(f)
            return self.chunk_embeddings
        
        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query, limit, for_hybrid = False):
        embbeding = self.generate_embedding(query)
        chunk_scores = []
        embed_chunks = self.chunk_embeddings
        
        for i in range(len(embed_chunks)):
            chunk_index = i
            chunk_metadata = self.chunk_metadata["chunks"][chunk_index]
            score = cosine_similarity(embbeding, embed_chunks[i])
            chunk_scores.append(
                {
                    'chunk_idx': chunk_index,
                    'movie_idx': chunk_metadata['movie_idx'],
                    'score': score
                }
            )
        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_id = chunk_score['movie_idx']
            score = chunk_score['score']
            if movie_id not in movie_scores or score > movie_scores[movie_id]:
                movie_scores[movie_id] = score

        sorted_movie_scores = sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)
    
        if for_hybrid:
            sorted_movie_scores_dict = dict(sorted_movie_scores)
            return dict(islice(sorted_movie_scores_dict.items(), limit))
        
        top_limit = []
        for i in range(int(limit)):
            movie_idx = sorted_movie_scores[i][0]
            doc = self.documents[movie_idx]
            entry = {
                "id": doc['id'],
                "title": doc['title'],
                "document": doc['description'][0:100],
                "score": sorted_movie_scores[i][1]
            }
            top_limit.append(entry)
        return top_limit