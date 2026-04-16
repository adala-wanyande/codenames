from abc import ABC, abstractmethod

class SpymasterAgent(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def give_clue(self, board_state, target_words):
        """Returns a tuple: (clue_word, number_of_guesses)"""
        pass

class OperativeAgent(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def guess_words(self, board_state, clue_word, num_guesses):
        """Returns a list of guessed words"""
        pass