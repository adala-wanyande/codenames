import os
import ollama
import google.generativeai as genai
from dotenv import load_dotenv
from .base import SpymasterAgent, OperativeAgent
from .prompts import single_COT_prompt
import re

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)


class Single_COT_LLMSpymaster(SpymasterAgent):
    """
    Zero-shot, single chain of thought spymaster.
    No simulation, no grouping, no CoT.
    """

    def __init__(self, name, model_type="ollama", model_name="qwen2.5"):
        super().__init__(name)
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.invalid_clue = set()

    def give_clue(self, board_state, target_words, shot = 0):
        assassin = [
            item['word'] for item in board_state
            if item['identity'] == 'Assassin' and not item['revealed']
        ]

        enemy_words = [
            item['word'] for item in board_state
            if item['identity'] == 'Blue' and not item['revealed']
        ]

        neutral_words = [
            item['word'] for item in board_state
            if item['identity'] == 'Neutral' and not item['revealed']
        ]
        prompt = single_COT_prompt(
            target_words=target_words,
            enemy_words=enemy_words,
            neutral_words=neutral_words,
            assassin=assassin,
            board_state=board_state,
            shot=shot
            )
        try:
            if self.model_type == "ollama":
                response = ollama.chat(model=self.model_name, messages=[
                    {"role": "system", "content": "You strictly follow output format."},
                    {"role": "user", "content": prompt}
                ])
                output = response["message"]["content"].strip()

            elif self.model_type == "gemini":
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                output = response.text.strip()

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            # --- Parse output ---
            parts = output.split()

            if len(parts) != 2:
                return "random", 1

            clue = parts[0].lower()
            if clue in self.invalid_clue:
                print(f"⚠️ LLM suggested previously invalid clue: '{clue}'. Using fallback.")
                return "random", 1

            try:
                count = int(parts[1])
            except:
                count = 1

            return clue, count

        except Exception as e:
            print(f"❌ Simple Spymaster error: {e}")
            return "random", 1
