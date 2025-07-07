from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain.text_splitter import RecursiveCharacterTextSplitter

model = SentenceTransformer("all-MiniLM-L6-v2")


def create_qa_chain_with_memory(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(text)
    return {"texts": texts, "full_text": text}


def ask_question(chain, query):
    texts = chain["texts"]
    embeddings = model.encode(texts)
    query_vec = model.encode([query])
    similarities = cosine_similarity(query_vec, embeddings)[0]
    best_idx = similarities.argmax()
    return texts[best_idx]


def extract_relevant_snippet(text, query):
    lines = text.split("\n")
    query_keywords = query.lower().split()
    matches = [line for line in lines if any(word in line.lower() for word in query_keywords)]
    return "\n".join(matches[:3])