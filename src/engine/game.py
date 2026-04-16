import random

class CodenamesGame:
    def __init__(self, vocabulary):
        """
        Initializes a new game of Codenames.
        Assumes the AI agents are playing as the 'Red' team (going first).
        """
        if len(vocabulary) < 25:
            raise ValueError("Vocabulary must have at least 25 words.")
            
        self.words = random.sample(vocabulary, 25)
        
        # Standard rules: 9 targets (Red), 8 opponents (Blue), 7 neutral, 1 assassin
        identities = ['Red']*9 + ['Blue']*8 + ['Neutral']*7 + ['Assassin']*1
        random.shuffle(identities)
        self.identities = identities
        
        self.revealed = [False] * 25
        
        # Game state tracking
        self.is_game_over = False
        self.winner = None  # 'Red' (Win) or 'Assassin' (Loss)
        self.turn_count = 0
        self.red_found = 0

    def get_spymaster_board(self):
        """Spymaster sees everything."""
        return [{"word": w, "identity": i, "revealed": r} 
                for w, i, r in zip(self.words, self.identities, self.revealed)]

    def get_operative_board(self):
        """Operative only sees identities of revealed cards."""
        return [{"word": w, "identity": i if r else "Unknown", "revealed": r} 
                for w, i, r in zip(self.words, self.identities, self.revealed)]

    def get_unrevealed_targets(self):
        """Helper for the Spymaster to know what words are left to guess."""
        return [w for w, i, r in zip(self.words, self.identities, self.revealed) 
                if i == 'Red' and not r]

    def is_valid_clue(self, clue):
        """Checks if a clue is legal (not a visible word on the board)."""
        clue_lower = clue.lower()
        for w, r in zip(self.words, self.revealed):
            if not r and (clue_lower in w.lower() or w.lower() in clue_lower):
                return False
        return True

    def process_guess(self, guess_word):
        """
        Processes a single guess. 
        Returns a tuple: (identity_of_card, continue_turn, game_over)
        """
        if guess_word not in self.words:
            return None, False, False # Invalid word, end turn
            
        idx = self.words.index(guess_word)
        if self.revealed[idx]:
            return self.identities[idx], False, False # Already revealed, end turn
            
        # Reveal the card
        self.revealed[idx] = True
        card_identity = self.identities[idx]
        
        # Evaluate the rules based on the identity
        if card_identity == 'Red':
            self.red_found += 1
            if self.red_found == 9:
                self.is_game_over = True
                self.winner = 'Red'
            return card_identity, True, self.is_game_over
            
        elif card_identity == 'Assassin':
            self.is_game_over = True
            self.winner = 'Assassin'
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