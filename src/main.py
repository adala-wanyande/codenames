from src.agents.llm import LLMSpymaster
from src.agents.baseline import Word2VecSpymaster

def run_tournament():
    print("🏆 Starting Codenames Tournament 🏆\n")
    
    # 1. Initialize Agents
    spymaster_llm = LLMSpymaster(name="Agent Qwen")
    spymaster_w2v = Word2VecSpymaster(name="Agent Gensim")
    
    # 2. Simulate asking for clues
    dummy_board = ["Pool", "Shark", "Beach", "Car", "Apple"]
    dummy_targets = ["Pool", "Shark", "Beach"]
    
    clue1, count1 = spymaster_llm.give_clue(dummy_board, dummy_targets)
    clue2, count2 = spymaster_w2v.give_clue(dummy_board, dummy_targets)
    
    # Lucas: Logging logic will go here
    print(f"\nResult: LLM gave '{clue1}' for {count1}")
    print(f"Result: W2V gave '{clue2}' for {count2}")

if __name__ == "__main__":
    run_tournament()