import regex as re
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def setup(model_name, device):
    """
    Load OPTForSequenceClassification, freeze everything except score head.
    Returns: model, tokenizer, lm_embeddings, lm_embeddings_weight, unused_tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"  # OPTForSequenceClassification needs right-padding

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache    = False   # KV-cache causes NaN during training
    model.to(device).eval()

    n_layers = model.config.num_hidden_layers  # 12 for opt-125m
    last_block = f"model.decoder.layers.{n_layers - 1}"
    for name, param in model.named_parameters():
        param.requires_grad_("score" in name or last_block in name)

    lm_embeddings        = model.get_input_embeddings()
    lm_embeddings_weight = lm_embeddings.weight.unsqueeze(0)  # (1, vocab, hidden)

    # unused-token list (same filtering as before)
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
    print(f"  Trainable: {n_trainable:,} params (score only)")
    print(f"  Unused tokens: {len(unused_tokens)}/{tokenizer.vocab_size} filtered")

    return model, tokenizer, lm_embeddings, lm_embeddings_weight, unused_tokens


def warmup_classifier(model, tokenizer, train_data, device,
                      n_samples, n_epochs, lr, batch_size, max_len):
    """Fine-tune score head on clean examples to seed the classifier."""
    samples = train_data[:n_samples]

    enc = tokenizer(
        [x["sentence"] for x in samples],
        truncation=True, max_length=max_len,
        padding="max_length", return_tensors="pt",
    )
    labels  = torch.tensor([x["label"] for x in samples])
    dataset = TensorDataset(enc["input_ids"], enc["attention_mask"], labels)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    model.train()
    for epoch in range(n_epochs):
        total_loss, correct = 0.0, 0
        for ids, mask, lbl in loader:
            ids, mask, lbl = ids.to(device), mask.to(device), lbl.to(device)
            opt.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask).logits
            if torch.isnan(logits).any():
                print(f"  [warmup] NaN in logits — skipping batch", flush=True)
                continue
            loss   = F.cross_entropy(logits, lbl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            opt.step()
            total_loss += loss.item()
            correct    += (logits.argmax(-1) == lbl).sum().item()
        acc = correct / len(samples)
        print(f"  [warmup {epoch+1}/{n_epochs}]  "
              f"loss={total_loss/len(loader):.4f}  acc={acc:.3f}")
    model.eval()
