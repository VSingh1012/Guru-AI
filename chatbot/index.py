from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

pdfs = [
    ('./texts/Gita.pdf', 'Bhagavad Gita'),
    ('./texts/Upanishads.pdf', 'Upanishads'),
    ('./texts/Vedas.pdf', 'Vedas')
]

updated_docs = []
for pdf_path, scripture_name in pdfs:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.metadata["source"] = scripture_name
        doc.metadata["page"] = i + 1
    updated_docs.extend(docs)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(updated_docs)

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
db.save_local("hindu_texts_index")