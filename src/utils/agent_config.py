from src.agents.single_COT_llm import Single_COT_LLMSpymaster
from src.agents.double_COT_llm import Double_COT_LLMSpymaster
from src.agents.double_COT_SR_llm import Double_COT__SR_LLM_Spymaster

def build_spymaster(spymaster_type, shot, operative=None):
    """
    Central factory for all spymaster configurations.
    """
    # -------------------------
    # BASE LLM (NO COT)
    # -------------------------
    if spymaster_type == "double_cot_SR":
        return Double_COT__SR_LLM_Spymaster(
            name="Double COT Spymaster with Stohastic Rollouts",
            model_type="ollama",
            model_name="qwen2.5",
            operative=operative
        )
    # -------------------------
    # SINGLE COT
    # -------------------------
    elif spymaster_type == "single_cot":
        return Single_COT_LLMSpymaster(
            name="Single COT Spymaster",
            model_type="ollama",
            model_name="qwen2.5"
        )
    # -------------------------
    # DOUBLE COT
    # -------------------------
    elif spymaster_type == "double_cot":
        return Double_COT_LLMSpymaster(
            name="Double COT Spymaster",
            model_type="ollama",
            model_name="qwen2.5"
        )
    else:
        raise ValueError(f"Unknown spymaster_type: {spymaster_type}")