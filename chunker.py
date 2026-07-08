import pypdf
import re
import unicodedata
import sys
import os
import nltk

# =========================
# NLTK DEPENDENCIES
# =========================

# Programmatically check and download required NLTK resources
def download_nltk_dependencies():
    required_resources = ['punkt', 'punkt_tab']
    for resource in required_resources:
        try:
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            print(f"Downloading missing NLTK resource: {resource}...")
            nltk.download(resource, quiet=True)

download_nltk_dependencies()
from nltk.tokenize import sent_tokenize

chunk_size = 6
overlap = 2


# =========================
# LOAD PDF
# =========================

def load_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"

    return full_text


# =========================
# MAIN CLEANING PIPELINE
# =========================

def clean_text(text):
    # -----------------------------------
    # Normalize Unicode Characters (NFKD)
    # This is crucial for:
    # 1. Fixing Windows print encoding issues (cp1252 UnicodeEncodeError)
    # 2. Decomposing ligatures like 'ﬁ' (\ufb01) into standard letters 'f' and 'i'.
    #    Without this, embeddings search for 'definition' will fail if indexed as 'deﬁnition'!
    # -----------------------------------
    text = unicodedata.normalize("NFKD", text)

    # Remove weird unicode artifacts
    text = text.replace("\ufffd", "")
    text = text.replace("\x00", "")

    # -----------------------------------
    # Fix hyphenated line breaks
    # Example:
    # misinfor-
    # mation
    # -----------------------------------
    text = re.sub(r'-\s*\n\s*', '', text)

    # -----------------------------------
    # Remove excessive newlines
    # -----------------------------------
    text = re.sub(r'\n+', '\n', text)

    # -----------------------------------
    # Remove excessive spaces
    # -----------------------------------
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =========================
# REMOVE REFERENCES SECTION
# =========================

def remove_references(text):
    # Only search for the references section in the last 15% of the text.
    # This prevents cutting off early sections that mention the word "references".
    if len(text) < 1000:
        return text

    cutoff = int(len(text) * 0.85)
    last_part = text[cutoff:]
    lower_last_part = last_part.lower()

    # Look for the section title "references" or "bibliography" at the start of a line
    # or surrounded by standard boundaries/spaces.
    match = re.search(r'\b(references|bibliography)\b', lower_last_part)
    if match:
        index = cutoff + match.start()
        return text[:index].strip()

    return text


# =========================
# CREATE SENTENCE CHUNKS
# =========================

def create_chunks(text):
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), chunk_size - overlap):
        chunk_sentences = sentences[i:i + chunk_size]
        chunk = " ".join(chunk_sentences)

        # Skip tiny chunks
        if len(chunk.split()) < 40:
            continue

        chunks.append(chunk.strip())

    return chunks


# =========================
# FULL PIPELINE
# =========================

def load_and_chunk(pdf_path):
    text = load_pdf(pdf_path)
    text = clean_text(text)
    text = remove_references(text)
    chunks = create_chunks(text)
    return chunks


# =========================
# SAFE PRINTING HELPERS
# =========================

def safe_print(message):
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback to ascii representation or utf-8 byte printing if terminal lacks unicode capabilities
        print(message.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))


# =========================
# TESTING
# =========================

if __name__ == "__main__":
    if os.path.exists("test.pdf"):
        chunks = load_and_chunk("test.pdf")
        safe_print(f"\nTotal chunks generated: {len(chunks)}")

        for i in range(min(5, len(chunks))):
            safe_print(f"\nCHUNK {i+1}")
            safe_print("-" * 50)
            safe_print(chunks[i])
    else:
        safe_print("test.pdf not found in current directory. Please place it in the root folder.")