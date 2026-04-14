from tokenizer import tokenizer
from collections import defaultdict
from load_movies import movie_loader
import os
import pickle
from collections import Counter

movie_file_path = "/home/migueldor/rag-search-engine/data/movies.json"
cache_path = "/home/migueldor/rag-search-engine/cache"

index_path = f"{cache_path}/index.pkl"
docmap_path = f"{cache_path}/docmap.pkl"
term_frequencies_path = f"{cache_path}/term_frequencies.pkl"

class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = Counter()
    
    def __add_document(self, doc_id, text):
        tokenized_text = tokenizer(text)
        for token in tokenized_text:
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

    def load(self):
        if os.path.exists(index_path) and os.path.exists(docmap_path) and os.path.exists(term_frequencies_path):
            with open(index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)

        else:
            raise ImportError("unable to load movie database")