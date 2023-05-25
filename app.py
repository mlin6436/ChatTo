import dotenv
import os
import streamlit as st
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import WikipediaLoader
from langchain.chains import RetrievalQA

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature = 0, openai_api_key = openai_api_key)

def load_pdf():
    loader = PyMuPDFLoader("hamlet.pdf")
    doc = loader.load()
    return doc

def load_code():
    project = "pandas-ai"
    docs = []

    for path, dirs, files in os.walk(project):
        for file in files:
            try: 
                loader = TextLoader(os.path.join(path, file), encoding='utf-8')
                docs.extend(loader.load_and_split())
            except Exception as e: 
                pass

    return docs
    
def load_wiki():
    topic = st.text_input("Enter a topic to prepare wiki content for Q&A")
    if topic:
        with st.spinner("Searching wiki..."):
            doc = WikipediaLoader(query=topic, load_max_docs=2).load()
        return doc

doc = ""
options = ["Select a data source", "PDF", "Code", "Wiki"]
selection = st.selectbox("Which resource would you like to chat to?", options)
if selection == "PDF":
    doc = load_pdf()
elif selection == "Code":
    doc = load_code()
elif selection == "Wiki":
    doc = load_wiki()

if doc:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
    docs = text_splitter.split_documents(doc)
    embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
    knowledge_base = FAISS.from_documents(docs, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm = llm, 
        chain_type = "stuff", 
        retriever = knowledge_base.as_retriever()
    )

    query = st.text_input("Ask your question here:")
    response = ""
    if query:
        response = qa.run(query)
        
        st.success("Completed query.")
        st.write("Answer: ", response)