import google.generativeai as genai
import os

from dotenv import load_dotenv

# --- 1. Setup ---
# Load environment variables from .env file
load_dotenv()
# --- 1. Setup ---
api_key = os.environ.get('GEMINI_API_KEY')
if not api_key:
    print("Error: GEMINI_API_KEY not found.")
    exit()

genai.configure(api_key=api_key)

# --- 2. Select a Reliable Model ---
# We hardcode a 'Flash' model for better free-tier reliability.
model_name = "models/gemini-2.5-flash"

print(f"Selected Model: {model_name}")

try:
    # --- 3. Configure Model ---
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction="You are a helpful assistant who only speaks in Sanskrit. Reply to all user prompts in grammatically correct Sanskrit using Devanagari script."
    )

    # --- 4. Start Chat ---
    chat_session = model.start_chat(history=[])
    print("\nNamaste! Sanskrit Chat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Punardarshanāya! (Goodbye!)")
            break
        
        try:
            response = chat_session.send_message(user_input)
            print(f"Gemini: {response.text}\n")
        except Exception as e:
            # Basic error handling to show just the main error message
            print(f"\nAn error occurred: {e}\n")

except Exception as e:
    print(f"\nCritical Error during model setup: {e}")