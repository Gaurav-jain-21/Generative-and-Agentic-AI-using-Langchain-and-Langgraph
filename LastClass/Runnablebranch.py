from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

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
    template="Write a detailed report on the {topic}",
    input_variables=['topic']
)

prompt2= PromptTemplate(
    template="Summarize the following text {text}",
    input_variables=['text']
)

report_gen_chain= prompt1  | model | parser

branch_chain= RunnableBranch(
    (lambda x: len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()

)

final_chain = report_gen_chain | branch_chain

result= final_chain.invoke({'topic': "AI"})

print(result)