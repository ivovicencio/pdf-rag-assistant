import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()
st.set_page_config(page_title="Asistente PDF", page_icon="📄")
st.title("Preguntale a tu PDF")

@st.cache_resource
def cargar_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = cargar_embeddings()

archivo = st.file_uploader("Subi un pdf", type="pdf")

if archivo:
    with open("temp.pdf", "wb") as f:
        f.write(archivo.read())

    with st.spinner("Procesando documento..."):
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(pages)
        vectorstore = Chroma.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k":3})

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key= os.getenv("GROQ_API_KEY"),
            temperature=0
        )
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    st.success("Documento listo. Pregunta lo que quieras")
    pregunta = st.text_input("Tu pregunta: ")

    if pregunta:
        with st.spinner("Pensando..."):
            result = qa_chain.invoke({"query" : pregunta})
        st.markdown(f"### Respuesta\n{result['result']}")
        st.markdown("### Fuentes")
        for i, doc in enumerate(result["source_documents"],1):
            pagina = doc.metadata.get("page", "?")
            st.markdown(f"**[{i}] Pagina: {pagina}:** {doc.page_content[:200]}...")