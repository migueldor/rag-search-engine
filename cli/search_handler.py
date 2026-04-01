import json
import string
from tokenizer import tokenizer
from nltk.stem import PorterStemmer

movie_file_path = "/home/migueldor/rag-search-engine/data/movies.json"
stopwords_file_path = "/home/migueldor/rag-search-engine/data/stopwords.txt"

with open(movie_file_path, 'r', encoding='utf-8') as movie_file:
    data = json.load(movie_file)

movies = data['movies']

with open(stopwords_file_path, 'r') as stopword_file:
        stopwords = stopword_file.read()

stopwords_list = stopwords.splitlines()

def search_handler(query):
        stemmer = PorterStemmer()
        print(f'Searching for: {query}')
        punct = string.punctuation
        trans = str.maketrans("", "", punct)
        lo_query = str.lower(query)
        clean_lo_query = lo_query.translate(trans)
        token_query = tokenizer(clean_lo_query)
        query_result_list = []

        for movie in movies:
                if len(query_result_list) >= 5:
                        break

                lo_title = str.lower(movie['title'])
                clean_lo_title = lo_title.translate(trans)
                for token in token_query:
                        if (token not in stopwords_list):
                                stm_token = stemmer.stem(token)
                                if (stm_token in clean_lo_title) and (movie['title'] not in query_result_list):
                                        query_result_list.append(movie['title'])

        if len(query_result_list) == 0:
               print('no results')
               return
        
        for i in range(len(query_result_list)):
               print(f'{i+1}. {query_result_list[i]}')
        
        return
