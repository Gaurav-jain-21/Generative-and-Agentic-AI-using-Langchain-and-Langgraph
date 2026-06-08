from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Literal
load_dotenv()

model= ChatGroq(model='llama-3.3-70b-versatile')
class Review(TypedDict):
    summary: str
    sentiment:str
model_with_structure=model.with_structured_output(Review)
result= model_with_structure.invoke("The hardware is great , but the software feels bloated. there are too many pre-installed apps that I can't remove. also the ui looks outdated compared to other brands. Hoping for a software update to fix this.")

print(result)