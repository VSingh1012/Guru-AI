from langchain_community.utilities import SerpAPIWrapper, DuckDuckGoSearchAPIWrapper
from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load FAISS index
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("hindu_texts_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Note: "gpt-5" doesn't exist yet - using "gpt-4o" as the latest available model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

prompt_template = """
You are a knowledgeable Hindu guru.
Answer the following question using ONLY the provided scripture excerpts.
If there are specific verses that you draw information from, make sure to cite them like ("2:47" for "Chapter 2 verse 47").

Question: {question}

Scripture excerpts:
{context}

Your answer should be clear, concise, and, if possible, include direct quotes.
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["question", "context"])

# Updated chain syntax for newer LangChain versions
def scripture_search_with_sources(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])
    
    # Format the prompt with the variables
    formatted_prompt = prompt.format(question=query, context=context)
    
    # Get the response from LLM
    answer = llm.invoke(formatted_prompt).content

    source_list = []
    for src in docs:
        scripture = src.metadata.get("source", "Unknown Text")
        page = src.metadata.get("page", "N/A")
        source_list.append(f"{scripture} (Page {page})")

    return f"{answer}\n\nSources:\n" + "\n".join(set(source_list))

scripture_tool = Tool(
    name="scripture_search",
    func=scripture_search_with_sources,
    description="Searches Hindu scriptures for relevant passages, with scripture name and page numbers."
)

google_tool = Tool(
    name="google_search",
    func=SerpAPIWrapper().run,
    description="Searches the web for modern interpretations or references."
)

duck_tool = Tool(
    name="duck_duck_search", 
    func=DuckDuckGoSearchAPIWrapper().run, 
    description="Searches the duck duck go web browser for verses, interpretations, and/or references"
)


tools = [scripture_tool, google_tool, duck_tool]