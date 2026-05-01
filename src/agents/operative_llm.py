from tokenize import group
import ollama
import google.generativeai as genai
from dotenv import load_dotenv
from prompt_toolkit import prompt
from .base import OperativeAgent
import re
from .prompts import operative_prompt


class LLMOperative(OperativeAgent):
    def __init__(self, name, model_type="ollama", model_name="qwen2.5"):
        super().__init__(name)
        self.model_type = model_type.lower()
        self.model_name = model_name

    def guess_words(self, board_state, clue_word, num_guesses, simulate = False):
        if simulate == False:
            print(f"[{self.name}] Analyzing clue '{clue_word}' for {num_guesses} words...")
        
        # Extract only the unrevealed words from the board
        available_words = [item['word'] for item in board_state if not item['revealed']]
        prompt = operative_prompt(num_guesses=num_guesses, clue_word=clue_word, available_words=available_words)
        
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

            raw_words = [word.strip() for word in output.split(',')]

            cleaned = []
            for w in raw_words:
                w = re.sub(r"[^a-zA-Z\s]", "", w)
                w = w.strip().title()

                if w.lower() in [aw.lower() for aw in available_words]:
                    matched = next(aw for aw in available_words if aw.lower() == w.lower())
                    cleaned.append(matched)

            return cleaned[:num_guesses]
        

        except Exception as e:
            print(f"❌ Error communicating with {self.model_name}: {e}")
            return []