from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever, vector_store  # Import vector store to update it
from langchain_core.documents import Document
import uuid  # For generating unique IDs for new messages

# Load the LLM
model = OllamaLLM(model="llama3.2")

# Define the prompt template
template = """
You are a helpful AI Assistant to Artifex!

Here are some relevant chats from the conversations you have had so far:
{chat_history}

Here is what Artifex is talking about with you right now:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Create the chain
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Artifex (q to quit): ")
    print("\n\n")

    if question.lower() == "q":
        break

    # Retrieve relevant past chats
    retrieved_docs = retriever.invoke(question)

    # Extract chat history text from retrieved documents
    chat_history = "\n".join([doc.page_content for doc in retrieved_docs])

    # Invoke the model with the retrieved context
    ai_response = chain.invoke({"chat_history": chat_history, "question": question})
    
    print(ai_response)

    # Store the new conversation pair (User + AI) in the vector store
    new_doc = Document(
        page_content=f"User: {question}\nAI: {ai_response}",
        metadata={"exchange_index": str(uuid.uuid4())},  # Unique ID
    )

    # Add new document dynamically to vector store
    vector_store.add_documents([new_doc])