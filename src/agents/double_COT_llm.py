import re
from .base import SpymasterAgent
from .prompts import double_COT_prompt
from .llm_client import LLMClient

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)


class Double_COT_LLMSpymaster(SpymasterAgent):
    """
    Two-stage Chain-of-Thought Spymaster:

    1. Select best subset of target words (semantic grouping)
    2. Generate clue ONLY from that subset
    """

    def __init__(self, name, model_type="ollama", model_name="qwen2.5", temperature=0.0):
        super().__init__(name)
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.invalid_clue = set()
        self.llm = LLMClient(
            model_type=model_type,
            model_name=model_name,
            temperature=temperature)

    # -------------------------
    # STEP 1: GROUP SELECTION
    # -------------------------
    def select_group(self, target_words, board_state, enemy_words, neutral_words, assassin_words, shot = 0):
        prompt = double_COT_prompt.propose_groups(
            target_words=target_words,
            enemy_words=enemy_words,
            neutral_words=neutral_words,
            assassin_words=assassin_words,
            shot=shot
        )
        return self._call_llm_json(prompt)

    # -------------------------
    # STEP 2: CLUE GENERATION
    # -------------------------
    def generate_clue(self, group_words, board_state, enemy_words, neutral_words, assassin_words, shot = 0):
        prompt = double_COT_prompt.generate_clue(
            group_words=group_words,
            enemy_words=enemy_words,
            neutral_words=neutral_words,
            assassin_words=assassin_words,
            board_state=board_state,
            shot=shot
        )
        return self._call_llm_text(prompt)

    # -------------------------
    # MAIN FUNCTION
    # -------------------------
    def give_clue(self, board_state, target_words, shot=2):

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

        # -------------------------
        # STEP 1: SELECT GROUP
        # -------------------------
        group_result = self.select_group(
            target_words,
            board_state,
            enemy_words,
            neutral_words,
            assassin
        )

        group_words = group_result.get("group", [])

        if not group_words:
            group_words = target_words[:1]

        # safety clamp
        group_words = group_words[:8]

        # -------------------------
        # STEP 2: GENERATE CLUE
        # -------------------------
        raw_clue = self.generate_clue(
            group_words,
            board_state,
            enemy_words,
            neutral_words,
            assassin
        )

        clue, _ = self._parse_clue(raw_clue)
        count = len(group_words)
        if clue in self.invalid_clue:
            print(f"⚠️ LLM suggested previously invalid clue: '{clue}'. Using fallback.")
            return "random", 1, group_words

        return clue, count, group_words

    # -------------------------
    # PARSING HELPERS
    # -------------------------
    def _parse_clue(self, text):
        text = text.strip().lower()
        parts = text.split()

        if len(parts) != 2:
            return "random", 1

        clue = parts[0]

        try:
            count = int(parts[1])
        except:
            count = 1

        return clue, count

    def _call_llm_text(self, prompt, **options):
        return self.llm.call_text(
            prompt,
            system="Return ONLY valid output.",
            **options,
        )

    def _call_llm_json(self, prompt, **options):
        return self.llm.call_json(
            prompt,
            system="Return ONLY valid JSON.",
            **options,
        )