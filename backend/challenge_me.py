from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI(openai_api_key="your-openai-api-key-here")

question_gen_template = PromptTemplate(
    input_variables=["document"],
    template="""
    Based on the following document, generate 3 logic-based or comprehension-focused questions that test deep understanding:

    {document}

    Provide only the questions in numbered list.
    """
)

def generate_challenge_questions(document):
    return llm(question_gen_template.format(document=document))


answer_eval_template = PromptTemplate(
    input_variables=["question", "user_answer", "document"],
    template="""
    Document:
    {document}

    Question: {question}
    User's Answer: {user_answer}

    Evaluate if the user's answer is correct or not, and explain why using information from the document.
    """
)

def evaluate_user_answer(question, user_answer, document):
    return llm(answer_eval_template.format(question=question, user_answer=user_answer, document=document))
