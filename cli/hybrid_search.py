from inverted_index import InvertedIndex
from lib.semantic_search import ChunkedSemanticSearch
from load_movies import movie_loader
import os
from utils import wd

movie_file_path = os.path.join(wd, "data", "movies.json")

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