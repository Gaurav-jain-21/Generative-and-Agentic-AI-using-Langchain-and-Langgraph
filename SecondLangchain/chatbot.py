from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

model= ChatGroq( model="llama-3.3-70b-versatile")
chat_history=[
    SystemMessage(content="You are a Helpfull assistant")
]
while True:
    user_input=input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input=="exit":
        break
    response=model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("AI: ",response.content)
print(chat_history)    