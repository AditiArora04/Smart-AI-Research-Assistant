import pdfplumber

def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(
            page.extract_text() for page in pdf.pages if page.extract_text()
        )

def read_txt(file):
    return file.read().decode("utf-8")