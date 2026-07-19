from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


load_dotenv()

llm= HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model= ChatHuggingFace(llm= llm)

parser= JsonOutputParser()

template=PromptTemplate(
    template='Give 4 facts about the {topic}.\n{format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt= template.invoke({'topic':'black hole'})
 
result=model.invoke(prompt)
final_output= parser.parse(result.content)
print(final_output)