from .base import SpymasterAgent
import gensim

class Word2VecSpymaster(SpymasterAgent):
    def __init__(self, name, model_path=None):
        super().__init__(name)
        # self.model = gensim.models.KeyedVectors.load_word2vec_format(model_path)

    def give_clue(self, board_state, target_words):
        # X: Put Word2Vec similarity math here!
        print(f"[{self.name}] Calculating vector distances...")
        return ("Water", 2) # Dummy return