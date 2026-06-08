from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()

prompt1= PromptTemplate(
    template="Generate the tweet about the {topic}",
    input_variables=['topic']
)
prompt2= PromptTemplate(
    template="Generate the Linkdin post about the {topic}",
    input_variables=['topic']
)
model=ChatGroq(
    model="llama-3.3-70b-versatile"
)
parser=StrOutputParser()

parallel_chain=RunnableParallel({
    'tweet':prompt1|model|parser,
    'linkdin':prompt2|model|parser
})
result=parallel_chain.invoke({'topic':"AI"})
print(result['tweet'])
print(result['linkdin'])