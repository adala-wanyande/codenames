from src.agents.single_COT_llm import Single_COT_LLMSpymaster
from src.agents.double_COT_llm import Double_COT_LLMSpymaster
from src.agents.double_COT_SR_llm import Double_COT__SR_LLM_Spymaster
from src.agents.baseline import Word2VecSpymaster, Word2VecOperative
from src.agents.operative_llm import LLMOperative


def build_spymaster(spymaster_type, shot, operative=None, model_type="ollama", model_name="qwen2.5", temperature=0.0):
    if spymaster_type == "word2vec":
        return Word2VecSpymaster(name="Word2Vec Spymaster")

    elif spymaster_type == "single_cot":
        return Single_COT_LLMSpymaster(
            name="Single COT Spymaster",
            model_type=model_type,
            model_name=model_name,
            temperature=temperature,
        )

    elif spymaster_type == "double_cot":
        return Double_COT_LLMSpymaster(
            name="Double COT Spymaster",
            model_type=model_type,
            model_name=model_name,
            temperature=temperature,
        )

    elif spymaster_type == "double_cot_SR":
        return Double_COT__SR_LLM_Spymaster(
            name="Double COT SR Spymaster",
            model_type=model_type,
            model_name=model_name,
            temperature=temperature,
            operative=operative,
        )
    else:
        raise ValueError(f"Unknown spymaster_type: {spymaster_type}")


def build_operative(operative_type="llm", model_name="qwen2.5", temperature=0.0):
    if operative_type == "word2vec":
        return Word2VecOperative(name="Word2Vec Operative")
    return LLMOperative(
        name=f"Operative ({model_name})",
        model_type="ollama",
        model_name=model_name,
        temperature=temperature
    )
