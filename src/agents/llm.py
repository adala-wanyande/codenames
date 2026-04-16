from .base import SpymasterAgent, OperativeAgent
import ollama
# import google.generativeai as genai
# import anthropic

class LLMSpymaster(SpymasterAgent):
    def __init__(self, name, model_type="ollama", model_name="qwen2.5"):
        super().__init__(name)
        self.model_type = model_type
        self.model_name = model_name

    def give_clue(self, board_state, target_words):
        # X: Put prompt engineering here!
        # Y: Put API routing here!
        print(f"[{self.name}] Thinking of a clue using {self.model_name}...")
        return ("Ocean", 2) # Dummy return for now