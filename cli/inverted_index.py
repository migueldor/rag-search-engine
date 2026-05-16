from tokenizer import tokenizer
from collections import defaultdict
from load_movies import movie_loader
import os
import pickle
from collections import Counter
import math
from search_utils import *
from itertools import islice

movie_file_path = "/home/migueldor/rag-search-engine/data/movies.json"
cache_path = "/home/migueldor/rag-search-engine/cache"

index_path = f"{cache_path}/index.pkl"
docmap_path = f"{cache_path}/docmap.pkl"
term_frequencies_path = f"{cache_path}/term_frequencies.pkl"
doc_lengths_path = os.path.join(cache_path, "doc_lengths.pkl")

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths = {}
        self.index_path = f"{cache_path}/index.pkl"
    
    def __add_document(self, doc_id, text):
        tokenized_text = tokenizer(text)
        self.doc_lengths[doc_id] = len(tokenized_text)
        for token in tokenized_text:
            self.term_frequencies[doc_id].update([token])
            self.index[token].add(doc_id)

    def get_documents(self, term):
        id_list = list(self.index[tokenizer(term)[0]])
        return sorted(id_list)
    
    def build(self):
        movies = movie_loader(movie_file_path)
        for movie in movies:
            movie_text = f"{movie['title']} {movie['description']}"
            movie_id = movie['id']
            self.__add_document(movie_id, movie_text)
            self.docmap[movie_id] = movie

    def save(self):
        os.makedirs(cache_path, exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(docmap_path) and os.path.exists(term_frequencies_path):
            with open(index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)

        else:
            raise ImportError("unable to load movie database")
        
    def get_tf(self, doc_id, term):
        token = tokenizer(term)
        if len(token) != 1:
            raise ValueError("There should be just one token present")
        return self.term_frequencies[doc_id][token[0]]
    
    def get_idf(self, term):
        token = tokenizer(term)
        if len(token) != 1:
            raise ValueError("There should be just one token present")
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index[token[0]])
        #print(f'doc count: {total_doc_count}')
        #print(f'term match doc count: {term_match_doc_count}')
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    
    def get_bm25_idf(self, term):
        token = tokenizer(term)
        if len(token) != 1:
            raise ValueError("There should be just one token present")
        N = len(self.docmap)
        df = len(self.index[token[0]])
        ratio = (N - df + 0.5) / (df + 0.5)
        bm25_idf = math.log(ratio + 1)
        return bm25_idf
    
    def get_bm25_tf(self, doc_id, term, k1=BM25_k1, b = BM25_B):
        tf = self.get_tf(doc_id, term)
        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths[doc_id]
        length_norm = 1 - b + (b * (doc_length / avg_doc_length))
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component
    
    def __get_avg_doc_length(self):
        N = len(self.docmap)
        if N == 0:
            return 0.0
        doc_lengths_list = [self.doc_lengths[doc] for doc in self.doc_lengths]
        return sum(doc_lengths_list) / N
    
    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query, limit):
        tokenized_query = tokenizer(query)
        scores = {}
        for token in tokenized_query:
            for doc_id in self.index[token]:
                score = self.bm25(doc_id, token)
                if doc_id not in scores:
                    scores[doc_id] = score
                else:
                    scores[doc_id] += score
        sorted_movie_scores = dict(sorted(scores.items(),  key=lambda item: item[1], reverse=True))
        return dict(islice(sorted_movie_scores.items(), limit))
        