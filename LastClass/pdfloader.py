from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm1= HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task= "text-generation"
)

model= ChatHuggingFace(
    llm= llm1
)

parser= StrOutputParser()

prompt=PromptTemplate(
    template="Answer the following question {question} from the following text {text}",
    input_variables=['question','text']
)
uri='https://en.wikipedia.org/wiki/SpaceX'

loader=WebBaseLoader(uri)
docs= loader.load()

chain = prompt | model | parser

result=chain.invoke({'question':'how is the founder of Space X','text': docs[0].page_content})

print(result)