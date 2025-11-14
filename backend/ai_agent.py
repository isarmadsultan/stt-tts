import os
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def get_relevant_context(query: str, collection, top_k: int = 30):
    """
    Retrieve top_k relevant documents using OpenAI embeddings via LangChain.
    
    Args:
        query: Search query string
        collection: ChromaDB collection
        top_k: Number of documents to retrieve
        
    Returns:
        list: Retrieved documents
    """
    print(f"\n[DEBUG] Fetching context for query: '{query}'")
    
    # Generate query embedding
    query_embedding = embeddings.embed_query(query)
    print("[DEBUG] Query embedding generated successfully.")
    
    # Retrieve from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    docs = results.get("documents", [[]])[0] if results else []
    print(f"[DEBUG] Retrieved {len(docs)} document(s) from collection.")
    
    return docs


def expand_query_with_history(query: str, history: list):
    """
    Use LLM to rewrite the query into a contextually complete version.
    Resolves pronouns and references based on conversation history.
    
    Args:
        query: User's current query
        history: List of previous conversation messages
        
    Returns:
        str: Expanded query with resolved references
    """
    # Build conversation history (last 6 turns for context)
    history_text = "\n".join([
        f"{msg['role']}: {msg['content']}" 
        for msg in history[-6:]
    ])
    
    prompt = f"""
Rewrite the user's latest query into a complete, unambiguous search query
based on the prior conversation.

Conversation:
{history_text}

User's latest query:
{query}

Guidelines:
- Resolve pronouns ("he", "first one", "the department")
- Resolve references to previously mentioned entities
- Generate a full, explicit query suitable for retrieval
- Do NOT answer the question
- Only rewrite the query

Rewritten query:
"""
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You rewrite queries for retrieval."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    
    return completion.choices[0].message.content.strip()


def generate_answer(query: str, collection, history: list):
    """
    Generate a contextual, memory-enhanced RAG answer.
    
    Args:
        query: User's question
        collection: ChromaDB collection
        history: Conversation history
        
    Returns:
        tuple: (answer, updated_history)
    """
    # Expand query using conversation context
    expanded_query = expand_query_with_history(query, history)
    
    # Retrieve relevant documents
    relevant_docs = get_relevant_context(expanded_query, collection)
    context_text = "\n\n".join(relevant_docs).strip()
    
    # Build conversation history text
    history_text = "\n".join([
        f"{msg['role'].capitalize()}: {msg['content']}" 
        for msg in history
    ])
    
    # Build prompt
    final_prompt = f"""
You are a conversational, memory-aware RAG assistant.
Your job is to give accurate answers ONLY from the provided RAG context.
However, you should respond in a warm, natural, human-like manner.

### RULES FOR FACTUAL ANSWERS:
- Use ONLY the RAG context for factual content
- Never invent facts or details not found in context
- If the context fully answers the question, respond clearly and naturally

### RULES WHEN CONTEXT IS MISSING OR IRRELEVANT:
When the answer cannot be found in the RAG context:
- DO NOT give a robotic fallback
- DO NOT use the same sentence every time
- Respond in a natural human way, such as:
    - Expressing uncertainty politely
    - Gently redirecting
    - Asking for clarification
    - Acknowledging limitations conversationally

Examples of acceptable fallback styles:
- "Hmm, I might be missing something, but I didn't see anything about that in my reference material."
- "I'm not sure that topic appears in what I have here. Could you rephrase or add more details?"
- "It seems this particular information wasn't included in the documents I have."
- "I couldn't find that in the context I'm using — maybe try asking it another way?"

### CONVERSATIONAL TONE:
- Use conversation history ONLY for tone, flow, and natural interaction
- Do NOT infer factual information from history
- Stay warm, respectful, concise, and natural

---
### Conversation History:
{history_text if history_text.strip() else "[No previous conversation]"}

---
### RAG Context:
{context_text if context_text else "[No usable context]"}

---
### User Query:
{expanded_query}

---
### Assistant's Response (use the rules above):
"""
    
    # Generate response
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "You are a context-aware conversational RAG assistant. Follow the user's prompt and instructions."
            },
            {"role": "user", "content": final_prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    answer = completion.choices[0].message.content.strip()
    
    # Update conversation history
    updated_history = history + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": answer}
    ]
    
    return answer, updated_history