import os
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Prompt user for file path and name
filepath = input("Enter file path and name: ")

# Check if file exists
if not os.path.isfile(filepath):
    print(f"File '{filepath}' does not exist.")
    exit()

# Read in text from file
with open(filepath, "r") as f:
    text = f.read()

# Split text into chunks
tokenizer = tiktoken.get_encoding('p50k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = []
texts = text_splitter.split_text(text)
for i, chunk in enumerate(texts):
    chunks.append({
        "id": i,
        "text": chunk,
        "num_tokens": tiktoken_len(chunk)
    })

# Print out the resulting chunks and number of tokens in each chunk
for chunk in chunks:
    print(f"Chunk {chunk['id']}:")
    print(chunk['text'])
    print(f"Number of tokens: {chunk['num_tokens']}\n")
