import dotenv
import os
import json
from langchain import OpenAI
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.evaluation.qa import QAEvalChain

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(temperature = 0, openai_api_key = openai_api_key)

loader = PyMuPDFLoader("hamlet.pdf")
doc = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 200)
docs = text_splitter.split_documents(doc)
embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key)
knowledge_base = FAISS.from_documents(docs, embeddings)

question_answers = [
    {"question" : "Where does the play take place?", "answer" : "Denmark"},
    {"question" : "What is the name of the castle?", "answer" : "Elsinore"},
    {"question" : "What are the first words spoken in the play?", "answer" : "Who's there?"},
    {"question" : "How has Ophelia died?", "answer" : "She has supposedly drowned (ambiguity surrounds her death)."},
    {"question" : "When was Hamlet written?", "answer" : "1600-1601"},
]

chain = RetrievalQA.from_chain_type(
    llm = llm, 
    chain_type = "stuff", 
    retriever = knowledge_base.as_retriever(), 
    input_key = "question"
)

predications = chain.apply(question_answers)
eval_chain = QAEvalChain.from_llm(llm)
evaluation = eval_chain.evaluate(
    question_answers,
    predications,
    question_key = "question",
    answer_key = "answer",
    prediction_key = "result"
)
print(predications)
print(evaluation)

with open("predications.json", "w") as f:
    json.dump(predications, f, indent=4)

with open("evaluation.json", "w") as f:
    json.dump(evaluation, f, indent=4)