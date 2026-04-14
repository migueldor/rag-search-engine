import string
from tokenizer import tokenizer
from load_movies import movie_loader

movie_file_path = "/home/migueldor/rag-search-engine/data/movies.json"

movies = movie_loader(movie_file_path)


def search_handler(query):
        print(f'Searching for: {query}')
        punct = string.punctuation
        trans = str.maketrans("", "", punct)
        token_query = tokenizer(query)
        query_result_list = []

        for movie in movies:
                if len(query_result_list) >= 5:
                        break

                lo_title = str.lower(movie['title'])
                clean_lo_title = lo_title.translate(trans)
                for token in token_query:
                        if (token in clean_lo_title) and (movie['title'] not in query_result_list):
                                query_result_list.append(movie['title'])

        if len(query_result_list) == 0:
               print('no results')
               return
        
        for i in range(len(query_result_list)):
               print(f'{i+1}. {query_result_list[i]}')
        
        return
