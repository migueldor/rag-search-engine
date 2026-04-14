from nltk.stem import PorterStemmer
import string

stopwords_file_path = "/home/migueldor/rag-search-engine/data/stopwords.txt"

with open(stopwords_file_path, 'r') as stopword_file:
        stopwords = stopword_file.read()

stopwords_list = stopwords.splitlines()

def tokenizer(input_str):
    stemmer = PorterStemmer()
    lo_input = str.lower(input_str)
    punct = string.punctuation
    trans = str.maketrans("", "", punct)
    clean_lo_input = lo_input.translate(trans)
    word_list = clean_lo_input.split(" ")
    token_list = []
    for word in word_list:
        if word not in stopwords_list:
            token = stemmer.stem(word)
            token_list.append(token)
    return token_list
