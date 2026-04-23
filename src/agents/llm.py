import os
from tokenize import group
import ollama
import google.generativeai as genai
from dotenv import load_dotenv
from .base import SpymasterAgent, OperativeAgent
from itertools import combinations
import re

# Load API keys safely from the .env file
load_dotenv()

# Configure Gemini if the key exists
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)

class LLMSpymaster(SpymasterAgent):
    def __init__(self, name, model_type="ollama", model_name="qwen2.5", operative=None):
        super().__init__(name)
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.operative = operative or LLMOperative(name=f"{name}'s Operative", model_type=model_type, model_name=model_name)
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
    
    def propose_groups(self, board_state, target_words, k=10):
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

        prompt = fprompt = f"""
                            You are an expert Spymaster in Codenames.

                            TASK:
                            Group target words into {k} clusters (1–3 words each) that share a strong semantic link.

                            TARGET:
                            {', '.join(target_words)}

                            FORBIDDEN:
                            {', '.join(enemy_words + assassin)}

                            RULES:
                            - Use ONLY target words
                            - Each group: 1–3 words
                            - Groups must be meaningfully related
                            - Avoid weak/abstract links

                            OUTPUT FORMAT (STRICT):
                            group1: word, word
                            group2: word, word, word
                            group3: word, word
                            """


        try:
            if self.model_type == "ollama":
                response = ollama.chat(model=self.model_name, messages=[
                    {'role': 'system', 'content': 'Follow formatting strictly.'},
                    {'role': 'user', 'content': prompt}
                ])
                output = response['message']['content']

            elif self.model_type == "gemini":
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                output = response.text

            else:
                raise ValueError

            # --- Parse ---
            groups = []

            for line in output.split("\n"):
                if ":" in line:
                    _, words = line.split(":", 1)
                    group = [w.strip() for w in words.split(",") if w.strip() in target_words]
                    
                    if 1 <= len(group) <= 3:
                        groups.append(tuple(sorted(group)))

            # deduplicate
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
    def group_words(self, board_state, target_words):

        best_clue = None
        best_score = float("-inf")
        best_count = 0

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
            candidate_groups = list(combinations(target_words, 1))

        for combo in candidate_groups:
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
                    
                if clue != "ERROR":
                    self.clue_cache[key] = clue

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

        if best_clue is None:
            return "random", 1  # fallback

        return best_clue, best_count

    def give_clue(self, target_words, target_count, board_state, simulation = False):
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
        revealed = [item['word'] for item in board_state if item['revealed']]

        prompt = f"""
                You are an expert Spymaster in Codenames.
                
                TASK:
                Generate ONE clue word linking all TARGET words.

                TARGET:
                {', '.join(target_words)}

                FORBIDDEN:
                {', '.join(enemy_words + assassin + revealed)}

                RULES:
                - ONE word only
                - must NOT appear in any board word (substring included)
                - no punctuation or spaces
                - no compounds (e.g. "starrynight" if "night" exists is invalid)
                - must be abstractly related to ALL targets

                OUTPUT:
                single word only
                """

        try:
            if self.model_type == "ollama":
                response = ollama.chat(model=self.model_name, messages=[
                    {'role': 'system', 'content': 'Follow formatting strictly.'},
                    {'role': 'user', 'content': prompt}
                ])
                output = response['message']['content'].strip()

            elif self.model_type == "gemini":
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(prompt)
                output = response.text.strip()

            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")

            clue = output.split(",")[0].strip().lower()
            clue = clue.replace(".", "")
            return clue, target_count

        except Exception as e:
            print(f"❌ Spymaster error: {e}")
            return "ERROR", 0

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
        
        prompt = f"""
        You are playing Codenames as the Operative.
        
        TASK:
        Select exactly {num_guesses} words from AVAILABLE list that best match the clue.

        CLUE:
        {clue_word}

        

        AVAILABLE WORDS:
        {', '.join(available_words)}

        RULES:
        - Choose ONLY from AVAILABLE WORDS
        - No new words allowed
        - No combining words
        - Output must be exact matches

        OUTPUT FORMAT:
        word1, word2, word3
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
            guessed_words = [w for w in guessed_words if w in available_words]
            guessed_words = guessed_words[:num_guesses]
            cleaned = []
            for w in guessed_words:
                w = w.strip()

                # remove punctuation + non-latin junk
                w = re.sub(r"[^a-zA-Z\s]", "", w)

                w = w.strip().title()

                if w in available_words:
                    cleaned.append(w)

            guessed_words = cleaned[:num_guesses]
            return guessed_words

        except Exception as e:
            print(f"❌ Error communicating with {self.model_name}: {e}")
            return []