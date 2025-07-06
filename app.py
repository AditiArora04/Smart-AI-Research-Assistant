import streamlit as st
from backend.doc_reader import read_pdf, read_txt
from backend.summarizer import get_summary
from backend.qa_engine import create_qa_chain_with_memory, extract_relevant_snippet
from backend.challenge_me import generate_challenge_questions, evaluate_user_answer

st.set_page_config(page_title="📚 Smart Research Assistant", layout="centered")
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .block-container { padding: 2rem 3rem; border-radius: 10px; background-color: #ffffff; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1, h2, h3, .stMarkdown { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
        .stButton>button { background-color: #4CAF50; color: white; padding: 0.5rem 1.2rem; font-size: 16px; border: none; border-radius: 6px; }
        .stTextInput>div>div>input { border-radius: 6px; padding: 0.5rem; border: 1px solid #ccc; }
        .stTextArea>div>textarea { border-radius: 6px; border: 1px solid #ccc; }
    </style>
""", unsafe_allow_html=True)

st.title("✨ Smart AI Research Assistant")
st.markdown("AI-powered assistant to summarize, quiz, and explore your documents with intelligence and clarity.")

file = st.file_uploader("📂 Upload a PDF or TXT file", type=["pdf", "txt"])
if file:
    if file.name.endswith(".pdf"):
        text = read_pdf(file)
    else:
        text = read_txt(file)

    st.subheader("📝 Document Summary")
    summary = get_summary(text)
    st.success(summary)

    qa_chain = create_qa_chain_with_memory(text)

    st.subheader("💬 Ask Anything with Memory")
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    question = st.text_input("🔎 Enter your question")
    if question:
        result = qa_chain({"question": question})
        answer = result["answer"]
        st.session_state.history.append((question, answer))

        st.markdown(f"**✅ Answer:** {answer}")
        snippet = extract_relevant_snippet(text, question)
        if snippet:
            st.markdown("**📄 Supporting Snippet:**")
            st.code(snippet)

    if st.session_state.history:
        st.markdown("---")
        st.subheader("🧠 Conversation History")
        for q, a in reversed(st.session_state.history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")

    st.markdown("---")
    st.subheader("🧠 Challenge Me Mode")
    if st.button("🎲 Generate Challenge Questions"):
        st.session_state.challenges = generate_challenge_questions(text).split("\n")

    if "challenges" in st.session_state:
        for idx, q in enumerate(st.session_state.challenges):
            if q.strip():
                st.markdown(f"**❓ Q{idx+1}:** {q}")
                user_input = st.text_input(f"✍️ Your Answer to Q{idx+1}", key=f"ans_{idx}")
                if user_input:
                    evaluation = evaluate_user_answer(q, user_input, text)
                    st.markdown(f"**🔍 Evaluation:** {evaluation}")
