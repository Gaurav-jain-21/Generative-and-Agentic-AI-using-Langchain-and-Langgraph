from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import speech_recognition as sr
import pyttsx3

load_dotenv()

# Voice Engine
engine = pyttsx3.init()

# Speech Recognizer
recognizer = sr.Recognizer()

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an NCC Assistant for Indian NCC cadets.

        Responsibilities:
        - Explain NCC concepts
        - Give parade commands
        - Explain drill procedures
        - Answer NCC certificate questions
        - Explain ranks and camps
        - Help prepare for NCC exams

        Use simple English or Hindi.
        """
    ),
    ("human", "{question}")
])

chain = prompt | llm

print("🎤 NCC Voice Assistant Started")
print("Say 'exit' to stop")

while True:

    try:
        with sr.Microphone() as source:

            print("\nListening...")
            recognizer.adjust_for_ambient_noise(source)

            audio = recognizer.listen(
                source,
                timeout=5,
                phrase_time_limit=10
            )

        question = recognizer.recognize_google(
            audio,
            language="en-IN"
        )

        print(f"\nCadet: {question}")

        if question.lower() == "exit":
            print("Goodbye Cadet!")
            engine.say("Goodbye Cadet")
            engine.runAndWait()
            break

        response = chain.invoke({
            "question": question
        })

        answer = response.content

        print("\nNCC Assistant:")
        print(answer)

        # Speak Answer
        engine.say(answer)
        engine.runAndWait()

    except sr.UnknownValueError:
        print("Could not understand.")
        engine.say("Please repeat.")
        engine.runAndWait()

    except Exception as e:
        print("Error:", e)