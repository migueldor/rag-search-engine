import os

from inverted_index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from load_movies import movie_loader
import os
from dotenv import load_dotenv
from google import genai
import time
from google.genai.errors import ServerError

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")


def gemini_spell_checker(query):
    client = genai.Client(api_key=api_key)

    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
        Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
        Preserve punctuation and capitalization unless a change is required for a typo fix.
        If there are no spelling errors, or if you're unsure, output the original query unchanged.
        Output only the final query text, nothing else.
        User query: "{query}"
        """
    model = 'gemma-4-31b-it'
    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text

def gemini_rewriter(query):
    client = genai.Client(api_key=api_key)

    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep the rewritten query concise (under 10 words)
        - It should be a Google-style search query, specific enough to yield relevant results
        - Don't use boolean logic

        Examples:
        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        If you cannot improve the query, output the original unchanged.
        Output only the rewritten query text, nothing else.

        User query: "{query}"
        """
    
    model = 'gemma-4-31b-it'
    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text

def gemini_expander(query):
    client = genai.Client(api_key=api_key)

    prompt = f"""Expand the user-provided movie search query below with related terms.

        Add synonyms and related concepts that might appear in movie descriptions.
        Keep expansions relevant and focused.
        Output only the additional terms; they will be appended to the original query.

        Examples:
        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
        - "action movie with bear" -> "action thriller bear chase fight adventure"
        - "comedy with bear" -> "comedy funny bear humor lighthearted"

        User query: "{query}"
        """
    model = 'gemma-4-31b-it'
    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text

def gemini_reranker(query, doc):
    client = genai.Client(api_key=api_key)

    prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc['title']} - {doc['description']}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match) consider it to be a range of floats.
        Output ONLY the number in your response, no other text or explanation.

        Score:"""
    model = 'gemma-4-31b-it'
    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text

movie_file_path = "/home/migueldor/rag-search-engine/data/movies.json"

def normalize(scores):
    if len(scores) == 0:
        return
    if min(scores) == max(scores):
        norm_scores = [1.0 for score in scores]
    else:
        min_score = min(scores)
        max_score = max(scores)
        norm_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return norm_scores

def weighted_search(query, alpha, limit):
    movies = movie_loader(movie_file_path)
    my_hybrid_search = HybridSearch(movies)
    results =  my_hybrid_search.weighted_search(query, alpha, limit)
    for i in range(int(limit)):
        ID = results[i]["id"]
        TITLE = results[i]['title']
        SCORE = results[i]['score']
        DOCUMENT = results[i]['document']
        print(f"\n{i+1}. ({ID}) {TITLE} (score: {SCORE})")
        print(f"   {DOCUMENT}...")
        print(                             )

def search_result_normalizer(search_result):
    id_list = [doc_id for doc_id in search_result]
    results_list = [search_result[doc_id] for doc_id in search_result]
    normalized_results_list = normalize(results_list)
    normalized_results = {}
    for i in range(len(id_list)):
        normalized_results[id_list[i]] = normalized_results_list[i]
    return normalized_results

def hybrid_score(bm25_score, semantic_score, alpha):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(k, rank):
    return 1 / (k + rank)

def search_result_rrf(search_result, k):
    id_list = [doc_id for doc_id in search_result]
    rrf_results = {}
    for i in range(len(id_list)):
        rrf_results[id_list[i]] = rrf_score(k, i+1)
    return rrf_results

def rrf_search(query, k, limit, method=None, rerank=None):
    movies = movie_loader(movie_file_path)
    my_hybrid_search = HybridSearch(movies)
    if method == "spell":
        enhanced_query = gemini_spell_checker(query)
        results =  my_hybrid_search.rrf_search(enhanced_query, k, limit)
        print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
    elif method == "rewrite":
        enhanced_query = gemini_rewriter(query)
        results =  my_hybrid_search.rrf_search(enhanced_query, k, limit)
        print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
    elif method == "expand":
        enhanced_query = gemini_expander(query)
        results =  my_hybrid_search.rrf_search(enhanced_query, k, limit)
        print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
    if rerank == "individual":
        results = my_hybrid_search.rrf_search(query, k, 5*limit)
        for movie in results:
            index = int(movie['id']) - 1
            doc = movies[index]
            #print(doc)
            try:
                movie['re-rank'] = float(gemini_reranker(query, doc))
                print(movie['re-rank'])
            except ServerError:
                movie['re-rank'] = 0.0
                print(f'the movie {movie['title']} cannnot be re-ranked')
            
        sorted_movie_scores = sorted(results, key=lambda x: x['re-rank'], reverse=True)
        
        for i in range(int(limit)):
            ID =sorted_movie_scores[i]["id"]
            TITLE = sorted_movie_scores[i]['title']
            RANK = sorted_movie_scores[i]['re-rank']
            DOCUMENT = sorted_movie_scores[i]['document']
            print(f"\n{i+1}. ({ID}) {TITLE} (rank: {RANK})")
            print(f"   {DOCUMENT}...")
            print(                             )

    else:
        results =  my_hybrid_search.rrf_search(query, k, limit)
    
        for i in range(int(limit)):
            ID = results[i]["id"]
            TITLE = results[i]['title']
            SCORE = results[i]['score']
            DOCUMENT = results[i]['document']
            print(f"\n{i+1}. ({ID}) {TITLE} (score: {SCORE})")
            print(f"   {DOCUMENT}...")
            print(                             )

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit):
        
        bm25_result = self._bm25_search(query, 500 * limit)
        chunked_semantic_result = self.semantic_search.search_chunks(query, 500 * limit, for_hybrid=True)
        normalized_bm25_result = search_result_normalizer(bm25_result)
        normalized_chunked_semantic_result = search_result_normalizer(chunked_semantic_result)
        
        hybrid_scores = {}
        
        id_set = set()
        for id in normalized_chunked_semantic_result:
            id_set.add(id)
        for id in normalized_bm25_result:
            id_set.add(id)
        id_list = list(id_set)
        for id in id_list:
            if id not in normalized_bm25_result:
                normalized_bm25_result[id] = 0
            if id not in normalized_chunked_semantic_result:
                normalized_chunked_semantic_result[id] = 0
            hybrid_scores[id] = {
                "bm25": normalized_bm25_result[id],
                'semantic': normalized_chunked_semantic_result[id],
                "hybrid": hybrid_score(normalized_bm25_result[id], normalized_chunked_semantic_result[id], alpha)
            }

        sorted_movie_scores = sorted(hybrid_scores.items(), key=lambda item: item[1]["hybrid"], reverse=True)
        
        top_limit = []
        for i in range(limit):
            movie_idx = sorted_movie_scores[i][0]
            #print(movie_idx)
            doc = self.documents[movie_idx-1]
            
            entry = {
                "id": doc['id'],
                "title": doc['title'],
                "document": doc['description'][0:100],
                "score": sorted_movie_scores[i][1]
            }
            top_limit.append(entry)
        return top_limit
       


    def rrf_search(self, query, k, limit):
        bm25_result = self._bm25_search(query, 500 * limit)
        chunked_semantic_result = self.semantic_search.search_chunks(query, 500 * limit, for_hybrid=True)
        rrf_bm25_result = search_result_rrf(bm25_result, k)
        rrf_chunked_semantic_result = search_result_rrf(chunked_semantic_result, k)
        
        hybrid_scores = {}

        id_set = set()
        for id in chunked_semantic_result:
            id_set.add(id)
        for id in bm25_result:
            id_set.add(id)

        id_list = list(id_set)
        for id in id_list:
            if id not in bm25_result:
                rrf_bm25_result[id] = 0
            if id not in chunked_semantic_result:
                rrf_chunked_semantic_result[id] = 0
            hybrid_scores[id] = {
                "bm25": bm25_result.get(id, 0),
                'semantic': chunked_semantic_result.get(id, 0),
                "rrf":  rrf_bm25_result[id] + rrf_chunked_semantic_result[id]
            }

        sorted_movie_scores = sorted(hybrid_scores.items(), key=lambda item: item[1]["rrf"], reverse=True)
        
        top_limit = []
        for i in range(limit):
            movie_idx = sorted_movie_scores[i][0]
            #print(movie_idx)
            doc = self.documents[movie_idx-1]
            
            entry = {
                "id": doc['id'],
                "title": doc['title'],
                "document": doc['description'][0:100],
                "score": sorted_movie_scores[i][1]
            }
            top_limit.append(entry)
        return top_limit