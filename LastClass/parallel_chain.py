from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

load_dotenv()

llm1= HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task= "text-generation"
)

model1= ChatHuggingFace(
    llm= llm1
)

llm2= HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation"
)
model2= ChatHuggingFace(
    llm= llm2
)

prompt1= PromptTemplate(
    template="Generate a short and simple notes from the following text \n {text}",
    input_variables=['text']
)
prompt2= PromptTemplate(
    template="Generate 5 short question and answers from the following text \n {text}",
    input_variables=['text']
)

prompt3= PromptTemplate(
    template= "Merge the provided notes and quiz into a single document \n {notes} and {quiz}",
    input_variables= ['notes','quiz']
)

parser= StrOutputParser()

parallel_chain=RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'quiz': prompt2 | model2 | parser
    }
) 
merge_chain = prompt3 | model1 | parser 
chain= parallel_chain | merge_chain
text="""
Machine Learning (ML) is the science of teaching computers to learn from data and make predictions without being explicitly programmed. The concept has evolved over decades—from early theoretical experiments and neural network models in the 1940s to the powerful generative AI and large language models (LLMs) used globally today.Theoretical Foundations (1940s – 1950s)The story of machine learning began long before computers were commonplace. In 1943, neurophysiologist Warren McCulloch and mathematician Walter Pitts created the first mathematical model of an artificial neural network, proving that electrical circuits could mimic simple brain functions.Shortly after, in 1950, pioneering mathematician Alan Turing proposed the "Turing Test," questioning whether machines could think like humans. The actual term "Machine Learning" was coined in 1959 by Arthur Samuel, an IBM researcher. Samuel built a computer program that played checkers and famously learned through self-play, improving its own strategies every time it played.Early Setbacks and the "AI Winters" (1960s – 1980s)Despite early optimism, machine learning and artificial intelligence hit roadblocks in the 1970s. Early models like Frank Rosenblatt’s Perceptron (an early attempt at image recognition) struggled with complex tasks and were limited by the rigid and expensive hardware of the time. Because computers lacked the massive processing power required for advanced algorithms, funding dried up. This era of diminished interest is historically known as the first "AI Winter".Research survived behind the scenes. The 1980s saw a brief resurgence with the introduction of rule-based "expert systems" and the rediscovery of the backpropagation algorithm, which allowed neural networks to adjust and learn much more effectively.The Shift to Statistical Models and Data (1990s – 2000s)As the twentieth century drew to a close, machine learning pivoted from biology-inspired neural networks to probability-based and statistical learning. Algorithms like Support Vector Machines (SVMs) and Random Forests became highly popular for classifying information.This period was marked by symbolic victories. In 1997, IBM’s Deep Blue supercomputer defeated the reigning world chess champion, Garry Kasparov. Rather than attempting to "think," Deep Blue utilized heavy computation to evaluate millions of potential board positions.The Deep Learning and Big Data Revolution (2010s)The digital explosion of the 21st century—driven by the internet and smartphones—provided the massive quantities of data (Big Data) that ML algorithms needed to truly learn. Coupled with the invention of powerful Graphical Processing Units (GPUs) capable of parallel processing, neural networks made a massive comeback.This subset of ML, known as "Deep Learning," achieved a historic breakthrough in 2012 when AlexNet, a deep convolutional neural network, completely dominated the ImageNet competition. It triggered a cascade of incredible advancements, including:DeepMind's AlphaGo defeating world champions at the incredibly complex board game Go in 2016.The widespread commercial use of computer vision in facial recognition and autonomous vehicles.The Era of Generative AI (2020s – Present)In 2017, researchers at Google published a landmark paper introducing the "Transformer" architecture. Transformers revolutionized Natural Language Processing (NLP) because they enabled models to understand the context and relationships of words across entire sentences, rather than just reading them sequentially.This development opened the doors for Large Language Models (LLMs) and Generative AI, allowing computers to write essays, translate languages, code software, and generate hyper-realistic images. Today, machine learning runs in the background of everyday technology, from the spam filters in your email inbox to dynamic pricing and complex medical diagnostics. The field continues to grow, transitioning from academic curiosity to a foundational technology shaping industries worldwide.
"""

result= chain.invoke({'text':text})
print(result)