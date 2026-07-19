from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.3-70B-Instruct",
    task='text-generation'
)

model= ChatHuggingFace(
    llm=llm
)


parser= StrOutputParser()
prompt1= PromptTemplate(
    template="Generate a joke on {topic}",
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template="Explain the joke {text}",
    input_variables=['text']
)

joke_gen_chain= RunnableSequence(prompt1, model, parser)

parallel_chain= RunnableParallel(
    {
        'joke': RunnablePassthrough(),
        'explain': RunnableSequence(prompt2, model, parser)
    }
)

chain= joke_gen_chain | parallel_chain
result= chain.invoke({'topic': 'AI'})
print(result)