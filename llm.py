import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

#Create the groq client
client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model = "llama-3.1-8b-instant",
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of Cambodia?"}
    ]
)

answer = response.choices[0].message.content
print(answer)

print("Chat with Llama (type 'quit' to exit):\n")
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    response = client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
    )

    answer = response.choices[0].message.content
    print(f"Llama: {answer}\n")
