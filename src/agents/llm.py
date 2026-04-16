import os
import ollama
import google.generativeai as genai
from dotenv import load_dotenv
from .base import SpymasterAgent, OperativeAgent

# Load API keys safely from the .env file
load_dotenv()

# Configure Gemini if the key exists
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)

class LLMSpymaster(SpymasterAgent):
    def __init__(self, name, model_type="ollama", model_name="qwen2.5"):
        super().__init__(name)
        self.model_type = model_type.lower()
        self.model_name = model_name

    def give_clue(self, board_state, target_words):
        print(f"[{self.name}] Thinking of a clue using {self.model_name}...")
        
        prompt = f"""
        You are an expert Spymaster in the game Codenames.
        Your target words are: {', '.join(target_words)}.
        Provide ONE single word that connects these targets.
        Respond WITH ONLY THE CLUE WORD AND NOTHING ELSE. No punctuation.
        """

        try:
            if self.model_type == "ollama":
                response = ollama.chat(model=self.model_name, messages=[
                    {'role': 'system', 'content': 'You are a helpful AI.'},
                    {'role': 'user', 'content': prompt}
                ])
                clue = response['message']['content'].strip()

            elif self.model_type == "gemini":
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                clue = response.text.strip()
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            return (clue, len(target_words))

        except Exception as e:
            print(f"❌ Error communicating with {self.model_name}: {e}")
            return ("ERROR", 0)


class LLMOperative(OperativeAgent):
    def __init__(self, name, model_type="ollama", model_name="qwen2.5"):
        super().__init__(name)
        self.model_type = model_type.lower()
        self.model_name = model_name

    def guess_words(self, board_state, clue_word, num_guesses):
        print(f"[{self.name}] Analyzing clue '{clue_word}' for {num_guesses} words...")
        
        # Extract only the unrevealed words from the board
        available_words = [item['word'] for item in board_state if not item['revealed']]
        
        prompt = f"""
        You are playing Codenames as the Operative.
        The Spymaster has given you the clue: "{clue_word}" for {num_guesses} cards.
        Here are the available words on the board:
        {', '.join(available_words)}
        
        Select EXACTLY {num_guesses} words from the list that best match the clue.
        Respond ONLY with a comma-separated list of the words. No other text.
        """

        try:
            if self.model_type == "ollama":
                response = ollama.chat(model=self.model_name, messages=[
                    {'role': 'system', 'content': 'You are a helpful AI playing a word game. You follow formatting rules strictly.'},
                    {'role': 'user', 'content': prompt}
                ])
                output = response['message']['content'].strip()

            elif self.model_type == "gemini":
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                output = response.text.strip()
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Clean up the AI's response into a Python list
            guessed_words = [word.strip() for word in output.split(',')]
            return guessed_words

        except Exception as e:
            print(f"❌ Error communicating with {self.model_name}: {e}")
            return []