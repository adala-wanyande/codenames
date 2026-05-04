import os
from tokenize import group
import ollama
import google.generativeai as genai
from dotenv import load_dotenv
from prompt_toolkit import prompt
from .base import SpymasterAgent
import re
import json
import random
from .prompts import double_COT_prompt, operative_prompt

# Load API keys safely from the .env file
load_dotenv()

# Configure Gemini if the key exists
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)

from src.agents.operative_llm import LLMOperative

class Double_COT__SR_LLM_Spymaster(SpymasterAgent):
    def __init__(self, name, model_type="ollama", model_name="qwen2.5", temperature=0.0, operative=None):
        super().__init__(name)

        self.model_type = model_type.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.uses_internal_policy = True
        self.invalid_clue = set()
        if operative:
            self.operative = operative 
        else:
            self.operative = LLMOperative("Default")
        self.clue_cache = {}

    def simulate_clue(self, clue, count, board_state):
        """
        Simulate ONE operative turn and return a structured score.
        """

        # --- CLUE VALIDATION (important) ---
        board_words = [item['word'].lower() for item in board_state]
        if clue.lower() in board_words:
            return {"correct": 0, "wrong": 1, "assassin": False, "hallucinations": 0}

        operative_board = [item.copy() for item in board_state]

        guessed_words = self.operative.guess_words(operative_board, clue, count, simulate=True)

        correct = 0
        wrong = 0
        assassin_hit = False
        hallucinations = 0

        valid_words = [item['word'] for item in operative_board if not item['revealed']]

        # enforce max guesses
        guessed_words = guessed_words[:count+1]

        for guess in guessed_words:

            if guess not in valid_words:
                hallucinations += 1
                continue

            word_info = next((item for item in operative_board if item['word'] == guess), None)
            if word_info is None:
                hallucinations += 1
                continue

            if word_info['identity'] == 'Red':
                correct += 1
            elif word_info['identity'] == 'Assassin':
                assassin_hit = True
                break
            else:
                wrong += 1
                break  # turn ends

        return {
            "correct": correct,
            "wrong": wrong,
            "assassin": assassin_hit,
            "hallucinations": hallucinations
        }

    def score_simulation(self, result):
        """
        Convert simulation result into scalar score.
        """
        if result["assassin"]:
            return -100  # catastrophic

        score = 0
        score += result["correct"] * 10
        score -= result["wrong"] * 5
        score -= min(result["hallucinations"], 1) * 3

        return score

    def utility(self, clue, count, board_state, target_words):
        """
        Lightweight pre-check before simulation.
        """
        forbidden_words = [
            item['word'] for item in board_state
            if item['identity'] in ['Assassin', 'Blue'] and not item['revealed']
        ]
        forbidden_words += [item['word'] for item in board_state if item['revealed']]

        if clue in forbidden_words:
            return -50  # penalize instead of zero

        return 0  # neutral baseline
    
    def propose_groups(self, board_state, target_words, k=10, shot = 0):
        """
        Ask LLM to propose k good groupings of target words.
        """

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
        prompt = double_COT_prompt.propose_groups(
            target_words=target_words,
            enemy_words=enemy_words,
            neutral_words=neutral_words,
            assassin_words=assassin,
            shot=shot
        )
        
        try:
            if self.model_type == "ollama":
                response = ollama.chat(model=self.model_name, messages=[
                    {'role': 'system', 'content': 'Follow formatting strictly.'},
                    {'role': 'user', 'content': prompt}
                ], options={"temperature": self.temperature})
                output = response['message']['content']

            elif self.model_type == "gemini":
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                output = response.text

            else:
                raise ValueError

            groups = []

            # -------------------------
            # 1. TRY JSON FIRST (BEST CASE)
            # -------------------------
            json_match = re.search(r"\{.*\}", output, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))

                    # handle {"group": [...]}
                    if "group" in data and isinstance(data["group"], list):
                        group = [w for w in data["group"] if w in target_words]
                        if 1 <= len(group) <= 8:
                            groups.append(tuple(sorted(group)))

                    # handle multiple groups in structured JSON
                    for v in data.values():
                        if isinstance(v, list):
                            group = [w for w in v if w in target_words]
                            if 1 <= len(group) <= 8:
                                groups.append(tuple(sorted(group)))

                except Exception as e:
                    print("⚠️ JSON parse failed:", e)

            # -------------------------
            # 2. FALLBACK: TEXT PARSING
            # -------------------------
            for line in output.split("\n"):
                line = line.strip()
                if not line:
                    continue

                # remove numbering like "1.", "-"
                line = re.sub(r"^[\-\d\.\)\s]+", "", line)

                # remove label like "Group 1:"
                if ":" in line:
                    line = line.split(":", 1)[1]

                words = [w.strip() for w in line.split(",")]

                group = [w for w in words if w in target_words]

                if 1 <= len(group) <= 8:
                    groups.append(tuple(sorted(group)))

            # -------------------------
            # 3. DEDUPLICATE
            # -------------------------
            groups = list(set(groups))

            return groups[:k]



        except Exception as e:
            print(f"❌ Proposal error: {e}")
            return []
    def filter_groups(self, groups):
        filtered = []
        for g in groups:
            if not any(set(g).issubset(set(other)) for other in groups if g != other):
                filtered.append(g)
        return filtered
    def group_words(self, board_state, target_words, k=10):

        best_clue = None
        best_score = float("-inf")
        best_count = 0
        best_combo = None

        assassin = tuple(sorted([
            item['word'] for item in board_state
            if item['identity'] == 'Assassin' and not item['revealed']
        ]))

        enemies = tuple(sorted([
            item['word'] for item in board_state
            if item['identity'] == 'Blue' and not item['revealed']
        ]))

        candidate_groups = self.propose_groups(board_state, target_words, k=10)
        candidate_groups = self.filter_groups(candidate_groups)

        # fallback if LLM fails
        if not candidate_groups:
            print("⚠️ Using heuristic fallback grouping")

            shuffled = list(target_words)
            random.shuffle(shuffled)

            combo = tuple(shuffled[:1])  # at least 1 word

            return "random", len(combo), combo

        for combo in candidate_groups:
            if best_clue in self.invalid_clue:
                continue
            r = len(combo)
            key = (
                tuple(sorted(combo)),
                tuple(sorted(item["word"] for item in board_state if not item["revealed"]))
            )
            if key in self.clue_cache:
                clue = self.clue_cache[key]
                count = r
            else:
                clue, count = self.give_clue(combo, r, board_state, simulation=True)

            # 🚫 NEW: reject reused invalid clues
            if clue in self.invalid_clue:
                continue
           
            if clue == "ERROR":
                continue

            base_score = self.utility(clue, count, board_state, combo)

            if base_score < 0:
                continue

            N = 2 if len(combo) == 1 else 4
            scores = []

            for _ in range(N):
                result = self.simulate_clue(clue, count, board_state)
                scores.append(self.score_simulation(result))

            sim_score = sum(scores) / len(scores)
            total_score = base_score + sim_score

            if total_score > best_score:
                best_score = total_score
                best_clue = clue
                best_count = count
                best_combo = combo
        if best_clue is None:
            return "random", 1, ()

        return best_clue, best_count, best_combo

    def give_clue(self, target_words, target_count, board_state, simulation = False, shot = 0):
        blacklist = list(self.clue_cache.values())[:20]
        if simulation == False:
            print(f"[{self.name}] Evaluating combo: {target_words}")

        assassin = [
            item['word'] for item in board_state
            if item['identity'] == 'Assassin' and not item['revealed']
        ]
        enemy_words = [
            item['word'] for item in board_state
            if item['identity'] == 'Blue' and not item['revealed']
        ]

        prompt = double_COT_prompt.generate_clue(
            group_words=target_words,
            enemy_words=enemy_words,
            neutral_words=[item['word'] for item in board_state if item['identity'] == 'Neutral' and not item['revealed']],
            assassin_words=assassin,
            board_state=board_state,
            shot=shot,
            banned_clues=list(self.invalid_clue) 
        )

        try:
            if self.model_type == "ollama":
                response = ollama.chat(model=self.model_name, messages=[
                    {'role': 'system', 'content': 'Follow formatting strictly.'},
                    {'role': 'user', 'content': prompt}
                ], options={"temperature": self.temperature})
                output = response['message']['content'].strip()
                # take FIRST token only, ignore numbers
                clue = re.findall(r"[a-zA-Z]+", output)[0].lower()

            elif self.model_type == "gemini":
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                output = response.text.strip()
                clue = re.findall(r"[a-zA-Z]+", output)[0].lower()


            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            
            return clue, target_count

        except Exception as e:
            print(f"❌ Spymaster error: {e}")
            return "ERROR", 0
