import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings


class RAGRetriever:
    """
    Handles embedding queries and retrieving relevant documents from ChromaDB.
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.embeddings = OpenAIEmbeddings(model=model_name)

    def get_relevant_context(self, query: str, collection, top_k: int = 30):
        """
        Retrieve top_k relevant documents from ChromaDB.
        """
        print(f"\n[DEBUG] Fetching context for query: '{query}'")

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        print("[DEBUG] Query embedding generated successfully.")

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        docs = results.get("documents", [[]])[0] if results else []
        print(f"[DEBUG] Retrieved {len(docs)} document(s) from collection.")

        return docs
class QueryRewriter:
    """
    Expands and rewrites queries using OpenAI LLM based on conversation history.
    """

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

    def expand(self, query: str, history: list):
        """
        Rewrite user query into explicit, context-complete form.
        """

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
- Resolve pronouns (he, first one, that place)
- Resolve references to previously mentioned entities
- Generate a full, explicit query suitable for retrieval
- Do NOT answer the question itself
- Only rewrite the query

Rewritten query:
"""

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You rewrite queries for retrieval."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return completion.choices[0].message.content.strip()

class RAGAgent:
    """
    Generates contextual, RAG-based conversational answers using:
    - Query expansion
    - Embedding-based retrieval
    - LLM answer generation
    """

    def __init__(self, collection, api_key: str = None):
        self.collection = collection
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        self.rewriter = QueryRewriter(api_key)
        self.retriever = RAGRetriever()
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    def answer(self, query: str, history: list):
        """
        Generate RAG-based conversational answer.
        Returns (answer, updated_history)
        """

        # 1. Expand query using conversation context
        expanded_query = self.rewriter.expand(query, history)

        # 2. Retrieve supporting documents
        relevant_docs = self.retriever.get_relevant_context(expanded_query, self.collection)
        context_text = "\n\n".join(relevant_docs).strip()

        # Build conversation history text
        history_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in history
        ])

        # 3. Construct RAG prompt
        final_prompt = f"""
You are a conversational, memory-aware RAG assistant.
You give answers ONLY from the provided RAG context.
Speak naturally, politely, and conversationally.

### IF context gives the answer:
→ Answer accurately and clearly.

### IF context is missing:
→ Do NOT hallucinate or fabricate details.
→ Instead respond like a human:
    - "Hmm, I might be missing something in my notes..."
    - "I couldn't find information about that here."
    - "Could you clarify?"

---

Conversation History:
{history_text if history_text.strip() else "[No previous conversation]"}

---

RAG Context:
{context_text if context_text else "[No usable context]"}

---

User Query:
{expanded_query}

---

Assistant Response:
"""

        # 4. Generate LLM answer
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a context-aware conversational RAG assistant."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = completion.choices[0].message.content.strip()

        # 5. Update conversation history
        updated_history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer}
        ]

        return answer, updated_history
