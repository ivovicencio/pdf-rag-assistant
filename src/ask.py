import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

#reconectar a la vector store ya creada
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) #trae los 3 chunks mas relevantes

#LLM gratis via Groq
llm = ChatGroq(
    model_name="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def preguntar(query : str):
    result = qa_chain.invoke({"query": query})
    respuesta = result['result']
    fuentes = result['source_documents']

    print(f"\nRESPUESTA:\n{respuesta}\n")
    print("FUENTES")
    for i, doc in enumerate(fuentes, 1):
        pagina = doc.metadata.get("page", "?")
        preview = doc.page_content[:150].replace("\n", " ")
        print(f" [{i}] Pagina {pagina}: \"{preview}...\"")

if __name__ == "__main__":
    while True:
        pregunta = input("\nPregunta algo sobre el PDF (o 'salir')")
        if pregunta.lower() == "salir":
            break
        preguntar(pregunta)


