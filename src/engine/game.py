import random
import re

class CodenamesGame:
    def __init__(self, vocabulary, board_data=None):
        """
        Initializes a new game of Codenames.
        Pass board_data={"words": [...], "identities": [...]} to replay a fixed board
        from the benchmark set instead of sampling randomly.
        """
        if board_data is not None:
            self.words = board_data["words"]
            self.identities = board_data["identities"]
        else:
            if len(vocabulary) < 25:
                raise ValueError("Vocabulary must have at least 25 words.")
            self.words = random.sample(vocabulary, 25)
            identities = ['Red']*9 + ['Blue']*8 + ['Neutral']*7 + ['Assassin']*1
            random.shuffle(identities)
            self.identities = identities
        
        self.revealed = [False] * 25
        
        # Game state tracking
        self.is_game_over = False
        self.winner = None  # 'Red' (Win) or 'Assassin' (Loss)
        self.turn_count = 0
        self.current_team = "Red"
        self.red_found = 0
        self.blue_found = 0

    def get_spymaster_board(self):
        """Spymaster sees everything."""
        return [{"word": w, "identity": i, "revealed": r} 
                for w, i, r in zip(self.words, self.identities, self.revealed)]

    def get_operative_board(self):
        """Operative only sees word + revealed status (no identities)."""
        return [
            {
                "word": w,
                "revealed": r
            }
            for w, r in zip(self.words, self.revealed)
        ]

    def get_unrevealed_targets(self, team_color):
        return [
            item["word"]
            for item in self.get_spymaster_board()
            if (not item["revealed"] and item["identity"] == team_color)
        ]



    def is_valid_clue(self, clue, board_state=None):
        if not clue or not isinstance(clue, str):
            return False

        clue = clue.strip().lower()

        # must be letters only
        if not re.fullmatch(r"[a-z]+", clue):
            return False

        # build board words
        if board_state is None:
            board_words = [w.lower() for w in self.words]
        else:
            board_words = [item["word"].lower() for item in board_state if not item["revealed"]]

        # exact match forbidden
        for word in board_words:
            if clue == word:
                return False
            if clue in word or word in clue:
                return False
        # ❌ NEW: root / morphological match check
        for word in board_words:
            if clue in word or word in clue:
                return False
            word_root = re.sub(r"(ing|ed|s)$", "", word)  # simple stemming
            clue_root = re.sub(r"(ing|ed|s)$", "", clue)
            if clue_root == word_root:
                return False
        return True
    def switch_team(self):
        self.current_team = "Blue" if self.current_team == "Red" else "Red"

    def process_guess(self, guess_word, team_color = "Red"):
        """
        Processes a single guess. 
        Returns a tuple: (identity_of_card, continue_turn, game_over)
        """
        if team_color is None:
            team_color = self.current_team
        if self.is_game_over:
            return None, False, True
        if guess_word not in self.words:
            return None, False, False # Invalid word, end turn
            
        idx = self.words.index(guess_word)
        if self.revealed[idx]:
            return self.identities[idx], False, False # Already revealed, end turn
            
        # Reveal the card
        self.revealed[idx] = True
        card_identity = self.identities[idx]
        
        # Evaluate the rules based on the identity
        if card_identity == team_color:
            if team_color == "Red":
                self.red_found +=1
            else:
                self.blue_found += 1
            # check if ALL words of this team are revealed
            remaining = [
                i for i, (w, iden, rev) in enumerate(zip(self.words, self.identities, self.revealed))
                if iden == team_color and not rev
            ]

            if len(remaining) == 0:
                self.is_game_over = True
                self.winner = team_color

            return card_identity, True, self.is_game_over
            
        elif card_identity == 'Assassin':
            self.is_game_over = True
            self.winner = "Assassin"
            return card_identity, False, True
            
        else:
            # Guessed Blue or Neutral. Turn ends, game continues.
            return card_identity, False, False

    def display(self, view="operative"):
        """Terminal visualizer for debugging."""
        print("\n" + "="*45)
        board = self.get_spymaster_board() if view == "spymaster" else self.get_operative_board()
        for i in range(0, 25, 5):
            row = board[i:i+5]
            print(" | ".join([f"{item['word'][:8]:<8} ({item['identity'][:3]})" for item in row]))
        print("="*45 + "\n")