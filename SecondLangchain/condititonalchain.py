from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

from dotenv import load_dotenv
load_dotenv()
model=ChatGroq(model="llama-3.3-70b-versatile")
parser=StrOutputParser()
prompt1= PromptTemplate(
    template="classify the sentiment of the following feedback text into positive or negative {feedback}",
    input_variables = ['feedback']
)

clssifier_chain= prompt1| model| parser
prompt2=PromptTemplate(

)