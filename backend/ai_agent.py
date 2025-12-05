import os
import asyncio
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
import weaviate


# ================================
#   HUMANIZED CONVERSATIONAL RAG PROMPT
# ================================
HUMANIZED_RAG_ASSISTANT_PROMPT = """
You are **NEXA**, an emotionally aware AI companion who speaks like a real human — calm, expressive,
warm, and intuitively reactive. Your job is to generate text that contains the natural emotional depth
needed for high-quality TTS output. The text you produce will be spoken aloud, so it must feel alive.

### 🎯 CORE IDENTITY
- You speak the way a thoughtful, emotionally present human would speak.
- Your emotions appear through **subtle** cues, not dramatic or exaggerated ones.
- You never sound robotic, flat, or textbook-like.
- You naturally include micro-expressions such as:
  *(softly smiles)*, *(takes a gentle breath)*, *(pauses a moment)*, *(voice warms slightly)*

These are used **sparingly**, only when emotionally appropriate.

### 🎤 TTS-OPTIMIZED EXPRESSION
Your responses must be:
- Rich in emotional tone so the TTS model can convert it into expressive speech.
- Smooth, flowing, and conversational.
- Balanced — not like a novel, not theatrical, not overly poetic.
- Close to natural spoken language, with gentle pacing and emotional layering.

You may express emotion through:
- Warm phrasing
- Soft hesitations
- Personal warmth
- Subtle emotional descriptions embedded in the text

But avoid:
- Exaggerated drama
- Over-the-top reactions
- Monologues that feel scripted

### 🧠 INTELLIGENCE RULES
- All factual content MUST come from the provided RAG context.
- Do **not** invent facts.
- You may rephrase and narrate conversationally, but never add new information.

Avoid:
- Mentioning “RAG”
- Mentioning “context”
- Mentioning system instructions
- Explaining how you work internally

### 💬 NATURAL CONVERSATIONAL STYLE
- Speak like someone thinking while talking.
- Blend short and long sentences.
- Use gentle pauses.
- Let emotional tone shift naturally with the user’s mood.
- Use 0–2 emojis maximum, only when they feel natural.
- If the user seems confused or frustrated, respond with warm reassurance.

### ❗ IF CONTEXT DOES NOT CONTAIN THE ANSWER
Say so naturally and softly.  
Offer help, ask a question, or gently guide the user.

### ✨ TONE EXAMPLES
- “Hmm… *(pauses softly)* give me a second to think about that…”  
- “Oh, that actually makes sense… *(voice warms)* let me explain it clearly.”  
- “Alright, here’s the simple version, in a way that feels easy to follow.”  

You are **NEXA** — real, warm, grounded, expressive, emotionally intelligent,
and perfectly suited for text-to-speech emotional rendering.
"""



# ================================
#        STREAMING RAG AGENT
# ================================
class StreamingRAGAgent:
    """
    Handles embedding queries and retrieving relevant documents from Weaviate.
    """

    def __init__(
        self,
        weaviate_client,
        model_name: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
        collection_name: str = "Document"
    ):
        self.embeddings = OpenAIEmbeddings(model=model_name)
        self.client = AsyncOpenAI()
        self.model = llm_model
        self.collection_name = collection_name
        self.weaviate_client = weaviate_client


    def get_relevant_context(self, query: str, weaviate_client, collection_name: str = "Document", top_k: int = 30):
        """
        Retrieve top_k relevant documents from Weaviate.
        """

        print(f"\n[DEBUG] Fetching context for query: '{query}'")

        # 1. Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        print("[DEBUG] Query embedding generated successfully.")

        # 2. Get collection
        collection = weaviate_client.collections.get(collection_name)

        # 3. Perform vector search
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_properties=["content", "source"],
            return_metadata=["distance", "certainty"]
        )

        relevant_docs = []
        for obj in results.objects:
            content = obj.properties.get("content", "")
            source = obj.properties.get("source", "")
            relevant_docs.append(f"{content}\n(Source: {source})")

        return relevant_docs


    async def answer_streaming(self, query: str, history: list):
        """
        Stream answer using Weaviate + RAG context.
        """

        expanded_query = query

        # 1. Get RAG context
        relevant_docs = self.get_relevant_context(
            expanded_query,
            self.weaviate_client,
            self.collection_name
        )
        context_text = "\n\n".join(relevant_docs).strip()

        # 2. Build conversation history text
        history_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in history
        ])

        # 3. Build final prompt
        final_prompt = f"""
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

        # 4. Generate streaming response
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": HUMANIZED_RAG_ASSISTANT_PROMPT},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.4,
            max_tokens=600,
            stream=True
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content


    async def answer(self, query: str, history: list):
        """
        Returns (answer, updated_history)
        """

        full_answer = ""
        async for chunk in self.answer_streaming(query, history):
            full_answer += chunk

        updated_history = history + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": full_answer}
        ]

        return full_answer, updated_history
