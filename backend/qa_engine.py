from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


def create_qa_chain_with_memory(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text(text)

    embeddings = OpenAIEmbeddings(openai_api_key="your-real-openai-api-key")
    db = FAISS.from_texts(texts, embedding=embeddings)

    retriever = db.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(OpenAI(openai_api_key="your-openai-api-key-here"), retriever=retriever, memory=memory)
    return qa


def extract_relevant_snippet(text, query):
    lines = text.split("\n")
    query_keywords = query.lower().split()
    matches = [line for line in lines if any(word in line.lower() for word in query_keywords)]
    return "\n".join(matches[:3])
