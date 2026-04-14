import json

def movie_loader(path):
    with open(path, 'r', encoding='utf-8') as movie_file:
        data = json.load(movie_file)
    
    movies = data['movies']
    return movies
