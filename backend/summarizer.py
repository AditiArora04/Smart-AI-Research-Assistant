from transformers import pipeline

summarizer = pipeline("summarization")

def get_summary(text):
    if len(text) > 3000:
        text = text[:3000]  # Limit for model input
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

