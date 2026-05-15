import regex as re   # generate.py uses `import regex as re`
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup(model_name, device):
    """
    Load OPT CausalLM, freeze all except lm_head (GRADMM: LAST_LAYERS=["lm_head"]),
    build the unused-token list (copied from generate.py drop_change_line_characters branch).

    Returns: model, tokenizer, lm_embeddings, lm_embeddings_weight, unused_tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device).eval()

    for name, param in model.named_parameters():
        param.requires_grad_(False)
    # lm_head.weight is tied to embed_tokens — set requires_grad on the tensor directly
    model.lm_head.weight.requires_grad_(True)

    lm_embeddings        = model.get_input_embeddings()
    lm_embeddings_weight = lm_embeddings.weight.unsqueeze(0)  # (1, vocab, hidden)

    # ── unused-token list (from generate.py, drop_change_line_characters=True) ─
    unused_tokens = []
    _pat = r"^[a-z]+(?:[\'-][a-z]+)*$"
    for token in range(tokenizer.vocab_size):
        text = tokenizer.decode(token)
        if any(c in text for c in "*;:_\n-'<>:{}[]()/\\|=+%@~`^#$&") or len(text) > 17:
            unused_tokens.append(token)
            continue
        text_lower = re.sub(r"^\W+|\W+$", "", text.lower().strip())
        if not re.fullmatch(_pat, text_lower):
            unused_tokens.append(token)
    unused_tokens.extend([tokenizer.pad_token_id, tokenizer.eos_token_id])
    unused_tokens = list(set(unused_tokens))

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {n_trainable:,} params (lm_head only)")
    print(f"  Unused tokens: {len(unused_tokens)}/{tokenizer.vocab_size} filtered")

    return model, tokenizer, lm_embeddings, lm_embeddings_weight, unused_tokens
