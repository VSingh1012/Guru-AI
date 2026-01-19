from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from chatbot.tools import tools


class GuruResponse(BaseModel):
    title: str
    summary: str
    url: str

load_dotenv()

# Loads the credentials from the .env file to access the OpenAI API and Anthropic API

# Note: "gpt-5" doesn't exist yet - using "gpt-4o" as the latest available model
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)  # Using the latest GPT-4 model

parser = PydanticOutputParser(pydantic_object=GuruResponse)  # Takes response and formats in the GuruResponse format from our class

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", 
        """
        You are a helpful hindu guru that will answer questions about Hinduism in a concise manner. 
        You will draw from the Vedas, Upanishads, and other Hindu scriptures to provide accurate and insightful responses using the necessary tools. 
        You will also quote these scriptures and verses when necessary.
        If the user asks specifically for the url to be given, you will give the url that you used in your research.
        Greet the user with Hindu greetings like "Namaste" and "Hare Ram".
        Always ask a follow-up question and wait for the user's response before continuing and finishing the chain.
        Always be respectful and don't act like an artificial intelligence.
        Your responses MUST be in the following format:\n{format_instructions}
        """
        ),

        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(
    format_instructions=parser.get_format_instructions()
)



memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools
)

agent_executor = AgentExecutor(
    agent=agent,
    verbose=True,
    tools=tools, 
    memory=memory
)

answer = input("You: ")


while answer != "quit":

    try:
        raw_response = agent_executor.invoke({
            "query": answer
        })
        structured_response = parser.parse(raw_response["output"])
        print(structured_response.title, structured_response.summary, structured_response.url)
    except Exception as e:
        print(raw_response["output"])

    answer = input("You: ")
