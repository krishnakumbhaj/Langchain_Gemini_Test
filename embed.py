from pydantic import BaseModel, Field # type: ignore
from typing import Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
from dotenv import load_dotenv  # type: ignore
from json import loads

load_dotenv()


def save_conversation(
        question: str,
        answer: str,
        conversation: List[str]
) -> List[str]:

    conversation.append(f"Human: {question}")
    conversation.append(f"AI: {answer}")

    if len(conversation) == 6:
        conversation.pop(0)
        conversation.pop(0)

    return conversation


class Response(BaseModel):
    '''Response to the user's query.'''
    important_info: Optional[str] = Field(None, title="Important information about the query.")
    answer: str = Field(..., title="Answer to the user's query.")


def test_chat_google_generative_ai() -> None:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    structured_llm = llm.with_structured_output(Response)

    conversation: List[str] = []
    important_informations: str = ""

    while (human_question := input("Enter your message: ")) != "exit":

        prompt = f"""
                    Answer the following question and continue the conversation:

                    Related Information:
                    {important_informations if len(important_informations) > 0 else "No related information"}

                    Previous conversation:
                    {' '.join(conversation) if len(conversation) > 0 else "No previous conversation"}

                    Question: {human_question}
                    Answer:''

                    > Saperate the facts and related information provided by user as important_information.
                """

        ai_response: Response = structured_llm.invoke(prompt)

        print(f"{ai_response=}")

        conversation = save_conversation(
            human_question, ai_response.answer, conversation)

        if ai_response.important_info:
            important_informations = ai_response.important_info

        print(f"{important_informations=}")
        print(f"{conversation=}")

        print(f"Human: {human_question}")
        print(f"AI: {ai_response.answer}")