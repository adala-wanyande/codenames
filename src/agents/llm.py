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
        print(f"[{self.name}] Analyzing board to group words...")
        
        # Extract the dangerous words so the Spymaster knows what to avoid
        assassin = [item['word'] for item in board_state if item['identity'] == 'Assassin' and not item['revealed']]
        enemy_words = [item['word'] for item in board_state if item['identity'] == 'Blue' and not item['revealed']]

        # EIRINI'S DOMAIN: The updated prompt instructing it to choose the number
        prompt = f"""
        You are an expert Spymaster in Codenames.
        Your goal is to link 2 or 3 of YOUR TARGET WORDS together with a single-word clue.
        
        YOUR TARGET WORDS: {', '.join(target_words)}
        WORDS TO AVOID (Enemy/Assassin): {', '.join(enemy_words + assassin)}
        
        Decide which target words you want to link. 
        Provide ONE single word as the clue, and the NUMBER of words it links.
        
        Respond EXACTLY in this format: CLUE, NUMBER
        Example: Ocean, 2
        """

        try:
            if self.model_type == "ollama":
                response = ollama.chat(model=self.model_name, messages=[
                    {'role': 'system', 'content': 'You are a helpful AI that strictly follows formatting rules.'},
                    {'role': 'user', 'content': prompt}
                ])
                output = response['message']['content'].strip()

            elif self.model_type == "gemini":
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                output = response.text.strip()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # Parse the "CLUE, NUMBER" format
            parts = output.split(',')
            clue = parts[0].strip()
            count = int(parts[1].strip())
            
            return (clue, count)

        except Exception as e:
            # If the LLM messes up the formatting, fallback safely
            print(f"❌ Spymaster formatting error: '{output}' | Error: {e}")
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