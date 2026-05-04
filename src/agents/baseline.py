from .base import SpymasterAgent, OperativeAgent
import numpy as np

_model_cache = {}


def _load_model(model_name="glove-wiki-gigaword-100"):
    if model_name not in _model_cache:
        import gensim.downloader as api
        print(f"[Word2Vec] Loading '{model_name}' — first run downloads ~128 MB ...")
        _model_cache[model_name] = api.load(model_name)
        print("[Word2Vec] Model ready.")
    return _model_cache[model_name]


class Word2VecSpymaster(SpymasterAgent):
    def __init__(self, name="Word2Vec Spymaster", model_name="glove-wiki-gigaword-100", max_vocab=50_000):
        super().__init__(name)
        self.model = _load_model(model_name)
        self.invalid_clue = set()
        # Restrict candidates to the top-N most frequent words in the model
        # to avoid obscure proper nouns that happen to be in the GloVe corpus.
        self._common_words = set(self.model.index_to_key[:max_vocab])

    def give_clue(self, board_state, target_words, shot=0):
        board_words_lower = {item["word"].lower() for item in board_state if not item["revealed"]}

        word_to_identity = {
            item["word"].lower(): item["identity"]
            for item in board_state
            if not item["revealed"]
        }

        target_lower = [w.lower() for w in target_words]
        enemy_lower = [w for w, ident in word_to_identity.items() if ident in ("Blue", "Neutral")]
        assassin_lower = [w for w, ident in word_to_identity.items() if ident == "Assassin"]

        target_vecs = [self.model[w] for w in target_lower if w in self.model]
        enemy_vecs = [self.model[w] for w in enemy_lower if w in self.model]
        assassin_vecs = [self.model[w] for w in assassin_lower if w in self.model]

        if not target_vecs:
            return ("random", 1)

        mean_target = np.mean(target_vecs, axis=0)

        # Mathematical penalty: push the search vector away from the assassin
        # and enemy words. Assassin gets 3.5x more weight than an enemy word.
        penalty = np.zeros_like(mean_target)
        for v in assassin_vecs:
            penalty += 0.7 * v
        for v in enemy_vecs:
            penalty += 0.2 * v

        search_vec = mean_target - penalty

        candidates = self.model.similar_by_vector(search_vec, topn=500)

        for candidate, _ in candidates:
            c = candidate.lower()
            if "_" in c or not c.isalpha():
                continue
            if c not in self._common_words:
                continue
            if c in board_words_lower:
                continue
            if c in {x.lower() for x in self.invalid_clue}:
                continue

            count = sum(
                1 for w in target_lower
                if w in self.model and self.model.similarity(c, w) > 0.3
            )
            count = max(count, 1)

            print(f"[{self.name}] Clue: {candidate.title()} ({count})")
            return (candidate.title(), count)

        return ("random", 1)


class Word2VecOperative(OperativeAgent):
    def __init__(self, name="Word2Vec Operative", model_name="glove-wiki-gigaword-100"):
        super().__init__(name)
        self.model = _load_model(model_name)

    def guess_words(self, board_state, clue_word, num_guesses, simulate=False):
        if not simulate:
            print(f"[{self.name}] Analyzing clue '{clue_word}' for {num_guesses} words...")

        available_words = [item["word"] for item in board_state if not item["revealed"]]
        clue_lower = clue_word.lower()

        if clue_lower not in self.model:
            return available_words[:num_guesses]

        scored = []
        for word in available_words:
            wl = word.lower()
            sim = self.model.similarity(clue_lower, wl) if wl in self.model else 0.0
            scored.append((word, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [w for w, _ in scored[:num_guesses]]
