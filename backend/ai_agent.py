# backend/ai_agent.py
import os
import difflib
from openai import OpenAI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from rapidfuzz import process  # For fuzzy matching (lightweight & efficient)

load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def fuzzy_correct_query(query: str, collection):
    """
    Attempt to auto-correct or fuzzy match user queries
    based on stored document metadata or content.
    """
    try:
        all_docs = collection.get()["documents"]
        flat_docs = [d for sublist in all_docs for d in sublist]
        # Use fuzzy matching to find close matches for misspelled terms
        best_match, score, _ = process.extractOne(query, flat_docs)
        if score > 75:
            print(f"[DEBUG] Fuzzy-corrected '{query}' → '{best_match}' (score={score:.1f})")
            return best_match
        else:
            print(f"[DEBUG] No close fuzzy match found for '{query}' (best score={score:.1f})")
            return query
    except Exception as e:
        print(f"[WARN] Fuzzy correction skipped: {e}")
        return query

def get_relevant_context(query: str, collection, top_k: int = 30):
    """
    Retrieve top_k relevant documents using consistent OpenAI embeddings via LangChain.
    Includes fuzzy correction and robust retrieval.
    """
    print(f"\n[DEBUG] Fetching context for query: '{query}'")

    # Step 1: Try to fuzzy-correct potential typos or near matches
    corrected_query = fuzzy_correct_query(query, collection)

    # Step 2: Generate embeddings
    query_embedding = embeddings.embed_query(corrected_query)
    print("[DEBUG] Query embedding generated successfully.")

    # Step 3: Retrieve from Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    docs = results.get("documents", [[]])[0] if results else []
    print(f"[DEBUG] Retrieved {len(docs)} document(s) from collection.")
    return docs, corrected_query

def generate_answer(query: str, collection):
    """
    Generate a context-bound answer using retrieved data and GPT model.
    Automatically handles minor spelling or naming errors.
    """
    relevant_docs, corrected_query = get_relevant_context(query, collection)
    context_text = "\n\n".join(relevant_docs).strip()

    # Strict fallback if context is empty
    if not context_text:
        print("[DEBUG] No context found — returning fallback message.")
        return "I don’t know about it. Is there anything else I could help you with?"

    # Context-aware, flexible prompt
    prompt = f"""
You are a reliable RAG assistant that answers ONLY from the provided context.
The user may have minor spelling or pronunciation errors in their query — interpret it intelligently.

Rules:
- Use only the given context.
- If context is insufficient, say exactly:
  "I don’t know about it. Is there anything else I could help you with?"
- Be precise, clear, and factual.

---
Context:
{context_text}
---
User question (possibly corrected):
{corrected_query}
---

Answer:
"""

    print(f"[DEBUG] Prompt prepared with {len(context_text.split())} words of context.")

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "You are an expert RAG assistant. You only answer using the provided context. "
                "If the answer isn't in the context, respond with: "
                "'I don’t know about it. Is there anything else I could help you with?'"
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=500,
    )

    answer = completion.choices[0].message.content.strip()
    print(f"[DEBUG] Model response:\n{answer}\n")

    return answer
