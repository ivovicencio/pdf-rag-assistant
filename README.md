# PDF RAG Assistant

Asistente que responde preguntas sobre cualquier PDF, citando la página exacta de donde sacó la respuesta. RAG (Retrieval-Augmented Generation) simple, corriendo con modelos gratuitos.

## Stack
- LangChain
- Groq (LLM: Llama 3.1)
- HuggingFace sentence-transformers (embeddings locales)
- ChromaDB (vector store)
- Streamlit (interfaz)

## Cómo correrlo
1. Clonar el repo
2. `python -m venv venv` y activarlo
3. `pip install -r requirements.txt`
4. Crear `.env` con `GROQ_API_KEY=tu-key`
5. `streamlit run src/app.py`

