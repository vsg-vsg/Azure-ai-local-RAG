"""
Full RAG pipeline with 2 modes:
- Retrieval (Azure AI search, embeddings with Sentence Transformers)
- Generation (Hugging Face Chat Model, Azure OpenAI)
"""

import os
from dotenv import load_dotenv

from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI


from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

from pdb import set_trace as bp

# ---------------------------------------------------------------------------------
# Load env variables (Azure keys if available) and use dummy if not available
# ---------------------------------------------------------------------------------
load_dotenv()

SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "dummy_search_key")
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT", "https://dummy-search.search.windows.net")
OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "dummy_openai_key")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://dummy-openai.openai.azure.com")
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")

search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name="epl-index",
    credential=AzureKeyCredential(SEARCH_KEY)
)

openai_client = AzureOpenAI(
    api_key=OPENAI_KEY,
    api_version="2024-05-01-preview",
    azure_endpoint=OPENAI_ENDPOINT
)


# -------------------------------------------------
# Knowledge base
# -------------------------------------------------
DOCUMENTS = {
    "premier_league": [
        "The English Premier League (EPL) is the top tier of English football, rebranded as EPL in 1992.",
        "The Premier League consists of 20 teams competing each season.",
        "Each team plays 38 matches in a season, facing every other team home and away.",
        "The Premier League uses a promotion and relegation system with the English Football League Championship.",
        "The bottom three teams in the league table are relegated at the end of each season."
    ],
    "clubs": [
        "Manchester United has won the most Premier League titles.",
        "Manchester City has been dominant in recent seasons under Pep Guardiola.",
        "Arsenal went unbeaten in the 2003-04 Premier League season, known as 'The Invincibles'.",
        "Chelsea won their first Premier League title in the 2004-05 season under Jose Mourinho.",
        "Liverpool won the Premier League in the 2019-20 season, their first in 30 years."
    ],
    "players": [
        "Alan Shearer is the all-time top scorer in the Premier League with 260 goals.",
        "Harry Kane is among the highest scorers in Premier League history.",
        "Ryan Giggs holds the record for the most assists in the Premier League.",
        "Thierry Henry won the Premier League Golden Boot four times.",
        "Cristiano Ronaldo won the Premier League with Manchester United before moving to Real Madrid."
    ],
    "stadiums": [
        "Old Trafford is the home stadium of Manchester United.",
        "Anfield is the home stadium of Liverpool FC.",
        "The Etihad Stadium is the home of Manchester City.",
        "Stamford Bridge is Chelsea's home stadium.",
        "The Emirates Stadium is the home of Arsenal FC."
    ],
    "records": [
        "Manchester City set the record for most points in a Premier League season with 100 in 2017-18.",
        "Leicester City won the Premier League in 2015-16 despite being 5000-1 outsiders.",
        "Arsenal holds the record for the longest unbeaten run in the Premier League with 49 games.",
        "The biggest win in Premier League history is Manchester United 9-0 Ipswich Town in 1995.",
        "The fastest goal in Premier League history was scored by Shane Long in 7.69 seconds in 2019."
    ]
}


embedder = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = {}
for topic, docs in DOCUMENTS.items():
    doc_embeddings[topic] = embedder.encode(f"[{topic}] {docs}", convert_to_tensor=True)


chatbot = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v0.6")


def azure_retrieve_docs(query: str, top: int = 3):
    results = search_client.search(
        query,
        query_type=QueryType.SIMPLE,
        top=top
    )
    return [doc["content"] for doc in results] 


def retrieve_docs(query: str, top: int = 3):
    query_emb = embedder.encode(query, convert_to_tensor=True)
    best_topic, best_score = None, -1
    best_docs = []

    for topic, embeddings in doc_embeddings.items():
        cos_scores = util.cos_sim(query_emb, embeddings)[0]
        max_score = cos_scores.max().item()
        if max_score > best_score:
            best_score = max_score
            best_topic = topic
            best_docs = DOCUMENTS[topic]

    return best_docs[:top]


def azure_generate_answer(query: str, context: str) -> str:
    messages = [
        {"role": "system", "content": "You are a knowledge base about the English Premier League (EPL) that answers using the provided suggestions for the given question."},
        {"role": "user", "content": f"Suggestions:\n{context}\n\nQuestion: {query}"}
    ]
    response = openai_client.chat.completions.create(
        model=DEPLOYMENT,
        messages=messages,
        max_tokens=300,
        temperature=0.7
    )
    return response.choices[0].message.content


def generate_answer(query: str, context: str) -> str:
    prompt = f"Answer the question using the following documents:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    response = chatbot(prompt, max_length=200, do_sample=True)[0]["generated_text"]
    return response


def rag_pipeline(query: str, use_azure: bool = False) -> str:
    if use_azure:
        docs = azure_retrieve_docs(query)
        context = " ".join(docs)
        return azure_generate_answer(query, context)
    else:
        docs = retrieve_docs(query)
        context = " ".join(docs)
        return generate_answer(query, context)


if __name__ == "__main__":
    print("RAG Demo ((Embeddings + Chat LLM) OR Auzre services if available)\n")
    while True:
        user_query = input("Ask a question about the english premier league (or type 'exit'): ").strip()
        if user_query.lower() == "exit":
            break
        print(rag_pipeline(user_query, use_azure=False))
        print("-" * 50)
