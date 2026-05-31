import json
from sentence_transformers import CrossEncoder
from load_movies import movie_loader
import os
import time
from google.genai.errors import ServerError
from utils import wd
from gemini_funcs import gemini_evaluator, gemini_spell_checker, gemini_rewriter, gemini_expander, gemini_reranker, gemini_reranker_batch
from hybrid_search import HybridSearch

movie_file_path = os.path.join(wd, "data", "movies.json")

def rrf_search_spell(hybrid_search, query, k, limit, method):
    enhanced_query = gemini_spell_checker(query)
    results =  hybrid_search.rrf_search(enhanced_query, k, limit)
    print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
    return results

def rrf_search_rewrite(hybrid_search, query, k, limit, method):
    enhanced_query = gemini_rewriter(query)
    results =  hybrid_search.rrf_search(enhanced_query, k, limit)
    print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
    return results

def rrf_search_expand(hybrid_search, query, k, limit, method):
    enhanced_query = gemini_expander(query)
    results =  hybrid_search.rrf_search(enhanced_query, k, limit)
    print(f"Enhanced query ({method}): '{query}' -> '{enhanced_query}'\n")
    return results

def rrf_search_rerank_individual(hybrid_search, movie_docs, query, k, limit):
    results =  hybrid_search.rrf_search(query, k, 5*limit)
    for movie in results:
        index = int(movie['id']) - 1
        doc = movie_docs[index]
        
        try:
            movie['re-rank'] = float(gemini_reranker(query, doc))
            print(movie['re-rank'])
        except ServerError:
            movie['re-rank'] = 0.0
            print(f'the movie {movie['title']} cannot be re-ranked')
        time.sleep(3)
        
    sorted_movie_scores = sorted(results, key=lambda x: x['re-rank'], reverse=True)
    
    return sorted_movie_scores

def rrf_search_rerank_batch(hybrid_search, movie_docs, query, k, limit):
    results =  hybrid_search.rrf_search(query, k, 5*limit)
    doc_list_str = ""
    for movie in results:
        index = int(movie['id']) - 1
        doc = movie_docs[index]
        doc_list_str += f"{doc['title']} - {doc['description']}\n"
    retry = 0
    while retry < 3:
        try:
            batch_rerank_scores_json = gemini_reranker_batch(query, doc_list_str)
            batch_rerank_scores_list = json.loads(batch_rerank_scores_json)
            
            for i in range(len(results)):
                results[i]['re-rank'] = float(batch_rerank_scores_list[i])
            break
        except ServerError:
            print("Batch re-ranking failed due to a server error. Retrying...")
            retry += 1
    if retry >= 3:
        for i in range(len(results)):
            results[i]['re-rank'] = 0.0
            print(f'the movie {results[i]['title']} cannnot be re-ranked')
    
    sorted_movie_scores = sorted(results, key=lambda x: x['re-rank'], reverse=True)
    
    return sorted_movie_scores

def rrf_search_cross_encoder(hybrid_search, movie_docs, query, k, limit):
    results =  hybrid_search.rrf_search(query, k, 5*limit)
    pairs = []
    for movie in results:
        index = int(movie['id']) - 1
        doc = movie_docs[index]
        pairs.append([query, f"{doc.get('title', '')} - {doc.get('description', '')}"])
        
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    rerank_scores = cross_encoder.predict(pairs)
    
    for i in range(len(results)):
        results[i]['re-rank'] = rerank_scores[i]

    sorted_movie_scores = sorted(results, key=lambda x: x['re-rank'], reverse=True)
    
    return sorted_movie_scores

def rrf_search_handler(query, k, limit, method=None, rerank=None, for_eval=False, for_rag=False):
    movies = movie_loader(movie_file_path)
    my_hybrid_search = HybridSearch(movies)

    if method == "spell":
        results =  rrf_search_spell(my_hybrid_search, query, k, limit, method)
        
    elif method == "rewrite":
        results =  rrf_search_rewrite(my_hybrid_search, query, k, limit, method)

    elif method == "expand":
        results =  rrf_search_expand(my_hybrid_search, query, k, limit, method)

    if rerank == "individual":
        results = rrf_search_rerank_individual(my_hybrid_search, movies, query, k, limit)

    elif rerank == "batch":
        results = rrf_search_rerank_batch(my_hybrid_search, movies, query, k, limit)

    elif rerank == "cross_encoder":
        results = rrf_search_cross_encoder(my_hybrid_search, movies, query, k, limit)

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
    if for_eval:
        formatted_results = []
        for i in range(int(limit)):
            formatted_results.append(f"{i+1}.{results[i]['title']} - {results[i]['document']}")
        evaluation_score_json = gemini_evaluator(query, formatted_results)
        evaluation_score = json.loads(evaluation_score_json)
        for i in range(int(limit)):
            print(f"{i+1}. {results[i]['title']}: {evaluation_score[i]}/3")
    if for_rag:
        formatted_results = []
        for i in range(int(limit)):
            formatted_results.append(f"{i+1}.{results[i]['title']} - {results[i]['document']}")
        return formatted_results