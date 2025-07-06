from transformers import pipeline, AutoTokenizer

summarizer = pipeline("summarization")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")  # You can change model

def get_summary(text):
    if not text.strip():
        return "The document appears to be empty or contains unreadable content."

    max_tokens = 1024
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens)
    truncated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)

    summary = summarizer(truncated_text, max_length=150, min_length=50, do_sample=False)
    return summary[0]["summary_text"]
