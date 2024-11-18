from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import tool
from dotenv import load_dotenv
load_dotenv()

llm = GoogleGenerativeAI(model="gemini-pro")


@tool
def add(numbers: str) -> int:
    """Add numbers together.

    Args:
        numbers (str): The numbers to add.

    Returns:
        int: The sum of the numbers
    """

    print(f"Adding numbers: {numbers}")

    patterns = [
        "plus",
        "add",
        "and",
        " ",
        "+",
        "'",
        '"'
    ]

    for pattern in patterns:
        numbers = numbers.replace(pattern, "")

    numbers = [int(x) for x in numbers.split(",")]

    print(numbers)

    return sum(numbers)


# List of tools
tools = [
    Tool(
        name="add",
        func=add,
        description="""
        Add numbers together.
    """)
]
from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
    template="""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
)

# Initialize the agent using the tools and LLM
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=custom_prompt
)

agent_executor = AgentExecutor(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True)

# Example usage
response = agent_executor.invoke(
    {"input": "What is 5 + 9?"})
print(response["output"])