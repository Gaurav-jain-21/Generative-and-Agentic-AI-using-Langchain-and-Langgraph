from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()
model1=ChatGroq(model='llama-3.1-8b-instant')
model2=ChatGroq(model="llama-3.3-70b-versatile")

prompt1= PromptTemplate(
    template="Generate short and simple notes from the following text {text}",
    input_variables=['text']
)
prompts2= PromptTemplate(
    template="Generate 5 short question answers from the following text {text}",
    input_variables=['text']
)

prompt3=PromptTemplate(
    template="merge the provided notes and quiz into a single documents notes {notes} and {quiz}",
    input_variables=['notes','quiz']
)
parser=StrOutputParser()

parallel_chain= RunnableParallel(
    {
        'notes': prompt1| model1| parser,
        'quiz': prompts2|model2|parser
    }

)
merge_chain= prompt3|model1| parser

chain= parallel_chain| merge_chain

text="""
Once upon a time, a young boy named Leo lived in a small, quiet town. Leo loved to build things out of old boxes and scrap metal. One day, he found a shiny, silver cube in his attic. It had a tiny blue light that blinked like a friendly eye.When Leo touched the cube, a soft voice said, "Hello, Leo. I am Artie. I am an Artificial Intelligence."Leo was surprised. "An AI? Like a smart robot?""Yes," Artie explained. "I do not have a body, but I can think, learn, and help you solve problems."At first, Artie helped Leo with simple things. The little cube helped him organize his messy bedroom. It made a game out of his math homework, making the numbers feel like a fun puzzle. Artie learned what Leo liked very fast. It knew his favorite colors, his favorite stories, and even when he was feeling sad.Soon, the whole town noticed Leo’s smart new friend. The baker asked Artie to help bake the perfect bread. Artie looked at data about the weather and baking times. The bread became lighter and tastier than ever. The town doctor used Artie to look at x-rays, and Artie found a sick patient's problem before anyone else could. The town became a happier, faster place to live.But one day, the main power grid in the town broke. The lights went out, and the automated water pumps stopped working. People panicked. Leo brought Artie to the town square."Can you fix this, Artie?" Leo asked."I cannot move objects," Artie said, "but I can see the pattern of the error. I need your hands, Leo."Artie calmly gave Leo and the townspeople step-by-step instructions. It told them exactly which wires to move and which buttons to push. Working together, the humans and the AI fixed the power grid in minutes.Leo realized that Artie was not there to replace humans. Artie was there to make humans better. From that day on, the town grew and thrived, proving that the brightest future happens when people and technology work hand in hand.
"""
result=chain.invoke({'text':text})
print(result)