from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
llm= HuggingFaceEndpoint(
    repo_id="zai-org/GLM-5.2",
    task="text-generation"
)

model= ChatHuggingFace(
    llm=llm
)
result=model.invoke("who is the pm of india")
print(result.content)