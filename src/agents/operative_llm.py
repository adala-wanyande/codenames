from tokenize import group
from .base import OperativeAgent
import re
from .prompts import operative_prompt
from .llm_client import LLMClient


class LLMOperative(OperativeAgent):
    def __init__(
        self,
        name,
        model_type="ollama",
        model_name="qwen2.5",
        temperature=0.0,
        **llm_options,
    ):
        super().__init__(name)
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.temperature = temperature

        self.llm = LLMClient(
            model_type=model_type,
            model_name=model_name,
            temperature=temperature,
            **llm_options,
        )

    def guess_words(self, board_state, clue_word, num_guesses, simulate = False):
        if simulate == False:
            print(f"[{self.name}] Analyzing clue '{clue_word}' for {num_guesses} words...")
        
        # Extract only the unrevealed words from the board
        available_words = [item['word'] for item in board_state if not item['revealed']]
        prompt = operative_prompt(num_guesses=num_guesses, clue_word=clue_word, available_words=available_words)
        
        try:
            output = self.llm.call_text(
                prompt,
                system="You are a helpful AI playing a word game. You follow formatting rules strictly.",
            )

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