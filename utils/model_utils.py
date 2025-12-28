from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from utils.constants import hf_model_name_map, QWEN_MODELS, QWEN_CHAT_MODELS, OPENAI_BASE_MODELS, PYTHIA_MODELS


def get_num_nodes(llm_model_name, llm_layer, linear_probe_input=None):
    if linear_probe_input == "word2vec_average":
        return 300
    if linear_probe_input in {"word2vec_token_count", "perplexity"}:
        return 1

    # Allow either alias (mapped via hf_model_name_map) or direct HF ID
    hf_model_name = hf_model_name_map.get(llm_model_name, llm_model_name)
    if hf_model_name in [*QWEN_MODELS, *PYTHIA_MODELS]:
        config = AutoConfig.from_pretrained(hf_model_name)
        if llm_layer == -1:
            num_nodes = config.num_hidden_layers
        else:
            num_nodes = config.hidden_size
    elif hf_model_name in OPENAI_BASE_MODELS:
        config = AutoConfig.from_pretrained(hf_model_name)
        if llm_layer == -1:
            num_nodes = config.n_layer
        else:
            num_nodes = config.n_embd
    else:
        raise NotImplementedError(f"Model {llm_model_name} is not supported for node count retrieval.")

    if linear_probe_input == "corr":
        num_nodes = num_nodes * (num_nodes - 1) // 2
    return num_nodes


def load_tokenizer_and_model(model_name, ckpt_step, gpu_id):
    if "gpt2" in model_name:
        padding_side = "right"
    else:
        padding_side = "left"
    
    # For ROCm/HIP compatibility: load on CPU first, then move to GPU if needed
    if ckpt_step == -1:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side)
        if gpu_id == -1:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
        else:
            # Load on CPU first, then move to specific GPU
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", max_memory={gpu_id: "40GB", "cpu": "100GB"})
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side, revision=f'step{ckpt_step}')
        if gpu_id == -1:
            model = AutoModelForCausalLM.from_pretrained(model_name, revision=f'step{ckpt_step}', device_map="cpu")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, revision=f'step{ckpt_step}', device_map="auto", max_memory={gpu_id: "40GB", "cpu": "100GB"})

    # Disable sliding-window attention to avoid unsupported SDPA warnings/paths
    cfg = model.config
    if hasattr(cfg, "use_sliding_window"):
        cfg.use_sliding_window = False
    if hasattr(cfg, "sliding_window"):
        cfg.sliding_window = None
    # Some configs keep attention settings in a dict
    if hasattr(cfg, "attention_config") and isinstance(cfg.attention_config, dict):
        cfg.attention_config.pop("sliding_window", None)
        cfg.attention_config.pop("use_sliding_window", None)
    return tokenizer, model


def wrap_chat_template(input_texts, tokenizer, model_name):
    chat_texts = []
    if model_name in QWEN_CHAT_MODELS:
        for prompt in input_texts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, enable_thinking=False 
            )
            chat_texts.append(text)
    else:
        chat_texts = input_texts
    return chat_texts
