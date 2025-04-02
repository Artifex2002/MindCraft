from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

# Define file paths
chat_history_path = "chat_history.txt"
db_location = "./chroma_langchain_db"

# Ensure the chat history file exists
if not os.path.exists(chat_history_path):
    with open(chat_history_path, "w", encoding="utf-8") as f:
        f.write("")  # Create an empty file if it doesn't exist

# Read the chat history
with open(chat_history_path, "r", encoding="utf-8") as f:
    chat_lines = [line.strip() for line in f.readlines() if line.strip()]  # Remove empty lines

# Check if the database needs to be initialized
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    i = 0
    while i < len(chat_lines):
        if chat_lines[i].startswith("User:"):
            user_message = chat_lines[i]
            ai_message = chat_lines[i + 1] if i + 1 < len(chat_lines) and chat_lines[i + 1].startswith("AI:") else ""
            conversation_chunk = f"{user_message}\n{ai_message}".strip()
            
            document = Document(
                page_content=conversation_chunk,
                metadata={"exchange_index": i},
                id=str(i)
            )
            ids.append(str(i))
            documents.append(document)
        
        i += 2  # Move to the next user message (skipping AI response)

# Initialize embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Initialize vector store
vector_store = Chroma(
    collection_name="chat_history",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents if necessary
if add_documents and documents:
    vector_store.add_documents(documents=documents, ids=ids)
    vector_store.persist()  # Ensure data is stored permanently

# Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)
