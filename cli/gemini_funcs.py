import os
from dotenv import load_dotenv
from google import genai

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

def gemini_reranker_batch(query, doc_list_str):
    client = genai.Client(api_key=api_key)

    prompt = f"""Rank the movies listed below by relevance to the following search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the movie IDs in order of relevance (best match first). Return ONLY a valid JSON list, nothing else, no backticks even.

        For example:
        [75, 12, 34, 2, 1]

        Ranking:"""
    model = 'gemma-4-31b-it'
    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text

def gemini_evaluator(query, formatted_results):
    client = genai.Client(api_key=api_key)
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

        Query: "{query}"

        Results:
        {chr(10).join(formatted_results)}

        Scale:
        - 3: Highly relevant
        - 2: Relevant
        - 1: Marginally relevant
        - 0: Not relevant

        Do NOT give any numbers other than 0, 1, 2, or 3.

        Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

        [2, 0, 3, 2, 0, 1]"""
    model = 'gemma-4-31b-it'
    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text

def gemini_RAG_answer(query, retrieved_docs_str):
    client = genai.Client(api_key=api_key)
    prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
        Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
        Provide a comprehensive answer that addresses the user's query.

        Query: {query}

        Documents:
        {retrieved_docs_str}

        Answer:"""
    model = 'gemma-4-31b-it'
    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text

def gemini_summarizer(query, results):
    client = genai.Client(api_key=api_key)
    prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

        This should be tailored to Hoopla users. Hoopla is a movie streaming service.

        Query: {query}

        Search results:
        {results}

        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    model = 'gemma-4-31b-it'
    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text

def gemini_citator(query, retrieved_docs_str):
    client = genai.Client(api_key=api_key)
    prompt =  f"""Answer the query below and give information based on the provided documents.

        The answer should be tailored to users of Hoopla, a movie streaming service.
        If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

        Query: {query}

        Documents:
        {retrieved_docs_str}

        Instructions:
        - Provide a comprehensive answer that addresses the query
        - Cite sources in the format [1], [2], etc. when referencing information
        - If sources disagree, mention the different viewpoints
        - If the answer isn't in the provided documents, say "I don't have enough information"
        - Be direct and informative

        Answer:"""
    model = 'gemma-4-31b-it'
    response = client.models.generate_content(
        model=model, contents=prompt
    )
    return response.text