from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence

load_dotenv()
prompt= PromptTemplate(
    template="Write a joke about the {topic}",
    input_variables=['topic']
)
model= ChatGroq(model="llama-3.3-70b-versatile")
parser= StrOutputParser()
prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=['text']
)
chain= RunnableSequence(prompt, model,parser, prompt2, model, parser)
print(chain.invoke({'topic':'AI'}))