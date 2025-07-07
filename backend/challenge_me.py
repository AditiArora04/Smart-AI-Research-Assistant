from transformers import pipeline

llm = pipeline("text-generation", model="distilgpt2")

def generate_challenge_questions(document):
    prompt = f"Generate 3 logic-based or comprehension questions from the following text:\n{document[:1000]}"
    output = llm(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
    return output


def evaluate_user_answer(question, user_answer, document):
    return f"Evaluation placeholder. You answered: '{user_answer}' for the question: '{question}'"