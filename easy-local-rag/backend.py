from fastapi import FastAPI
from pydantic import BaseModel
import torch
import ollama
import json
import os
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ANSI escape codes for log colors
NEON_GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RESET_COLOR = '\033[0m'

# System message for the assistant
system_message = "You are a helpful assistant that is an expert at extracting the most useful information from a given text. Also bring in extra relevant information to the user query from outside the given context."

# Function to load vault content
def load_vault(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as vault_file:
            return vault_file.readlines()
    return []

# Function to generate embeddings using Ollama
def generate_embeddings(content):
    embeddings = []
    for line in content:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=line)
        embeddings.append(response["embedding"])
    return torch.tensor(embeddings)

# Load vault content and embeddings
print(NEON_GREEN + "Loading vault content and generating embeddings..." + RESET_COLOR)
vault_content = load_vault("../vault.txt")
vault_embeddings_tensor = generate_embeddings(vault_content)



# Conversation history
conversation_history = []

class Query(BaseModel):
    user_input: str

# Function to rewrite query
def rewrite_query(user_input, conversation_history, model):
    return user_input    # remove this if u want to rewrite the query
    # Get the last two messages from the conversation history
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    
    # Construct the prompt for rewriting the query
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:

    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query

    Return ONLY the rewritten query text, without any additional formatting or explanations.

    Conversation History:
    {context}

    Original query: [{user_input}]

    Rewritten query: 
    """
    
    # Now use ollama.chat to generate the response based on the constructed prompt
    response = ollama.chat(model=model, messages=[{"role": "system", "content": prompt}])
    
    # Return the rewritten query from the assistant's response
    return response["message"]["content"].strip()


# Function to get relevant context
def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=1, vault_file_path="../vault.txt"):
    # Reload the vault content if the content has changed
    new_vault_content = load_vault(vault_file_path)
    
    # Check if the vault content has changed
    if new_vault_content != vault_content:
        print("Vault content has changed. Reloading and regenerating embeddings...")
        
        # Update vault content and regenerate embeddings
        vault_content[:] = new_vault_content  # Modify in place
        vault_embeddings[:] = generate_embeddings(vault_content)  # Modify in place
    
    if vault_embeddings.nelement() == 0:
        return [], []

    # Get embedding for the rewritten input
    input_embedding = ollama.embeddings(model="mxbai-embed-large", prompt=rewritten_input)["embedding"]
    
    # Compute cosine similarity between the input embedding and the vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    
    # Select top_k most similar documents
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    
    # Return the relevant content and the similarity scores
    return [vault_content[idx].strip() for idx in top_indices], cos_scores[top_indices]

# Function to handle chatbot conversation
def ollama_chat(user_input, system_message, vault_embeddings, vault_content, model, conversation_history):
    # Add user's input to the conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Step 1: Rewrite the query if there's conversation history
    if len(conversation_history) > 1:
        rewritten_query = rewrite_query(user_input, conversation_history, model)
    else:
        rewritten_query = user_input

    print(YELLOW + f"Rewritten Query: {rewritten_query}" + RESET_COLOR)

    # Step 2: Retrieve relevant context from the vault
    relevant_context, scores = get_relevant_context(rewritten_query, vault_embeddings, vault_content)
    
    if relevant_context:
        # Add the relevant context to the user's query
        context_str = "\n".join(relevant_context)
        user_input += f"\n\nRelevant Context:\n{context_str}"
        print(CYAN + f"Relevant Documents: {relevant_context}" + RESET_COLOR)
        print(CYAN + f"Similarity Scores: {scores.tolist()}" + RESET_COLOR)
    else:
        print(CYAN + "No relevant context found." + RESET_COLOR)

    # Step 3: Update the conversation history with the modified user input
    conversation_history[-1]["content"] = user_input

    # Step 4: Generate response using ollama.chat
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]

    # Get the assistant's response
    response = ollama.chat(model=model, messages=messages)
    
    # Extract the assistant's message from the response
    assistant_message = response["message"]["content"].strip()

    print(NEON_GREEN + f"Response: {assistant_message}" + RESET_COLOR)

    # Add the assistant's message to the conversation history
    conversation_history.append({"role": "assistant", "content": assistant_message})

    # Return the assistant's message
    return assistant_message

@app.get("/greet")
async def greet():
    return {"message": "hi"}

@app.post("/query")
async def get_response(query: Query):
    user_input = query.user_input
    model = "llama3"

    # Log input query
    print(NEON_GREEN + f"User Query: {user_input}" + RESET_COLOR)

    # Generate response
    response = ollama_chat(user_input, system_message, vault_embeddings_tensor, vault_content, model, conversation_history)

    # Return detailed response
    return {
        "query": user_input,
        "rewritten_query": conversation_history[-2]["content"],  # Last user input after context injection
        "context_documents": conversation_history[-2].get("content", "").split("\n\nRelevant Context:\n")[-1],
        "response": response
    }
