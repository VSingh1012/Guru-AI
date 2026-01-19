import chainlit as cl
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

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
parser = PydanticOutputParser(pydantic_object=GuruResponse)

# Create the prompt template
prompt = ChatPromptTemplate.from_messages([
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
]).partial(format_instructions=parser.get_format_instructions())


@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    # Create memory for this session
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create agent
    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        verbose=True,
        tools=tools,
        memory=memory,
        handle_parsing_errors=True
    )
    
    # Store in user session
    cl.user_session.set("agent_executor", agent_executor)
    cl.user_session.set("parser", parser)
    
    # Send welcome message
    await cl.Message(
        content="🕉️ **Namaste!** Welcome to Guru.AI!\n\nI'm **Guru**, and I can help you explore the wisdom of the Vedas, Upanishads, and Bhagavad Gita. Feel free to ask me any questions about Hindu philosophy, teachings, or scriptures.\n\nHow may I guide you today?"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    # Retrieve the agent executor from session
    agent_executor = cl.user_session.get("agent_executor")
    parser = cl.user_session.get("parser")
    
    # Create a message to show thinking
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Run the agent
        raw_response = await cl.make_async(agent_executor.invoke)({
            "query": message.content
        })
        
        # Try to parse structured response
        try:
            structured_response = parser.parse(raw_response["output"])
            
            # Format the response nicely
            response_text = f"### {structured_response.title}\n\n"
            response_text += f"{structured_response.summary}\n\n"
            
            if structured_response.url and structured_response.url.strip():
                response_text += f"**Source:** {structured_response.url}"
            
            msg.content = response_text
            await msg.update()
            
        except Exception as parse_error:
            # If parsing fails, just show the raw output
            msg.content = raw_response["output"]
            await msg.update()
            
    except Exception as e:
        # Handle any errors gracefully
        msg.content = f"I apologize, but I encountered an error: {str(e)}\n\nPlease try rephrasing your question."
        await msg.update()


@cl.on_chat_end
async def end():
    """Clean up when chat ends"""
    await cl.Message(content="🙏 Thank you for seeking wisdom. May you find peace and enlightenment. Hare Krishna! Jai Shri Ram!").send()