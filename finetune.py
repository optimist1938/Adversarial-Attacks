import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

PROMPT_SUFFIX = " It was "
LABEL_WORDS   = {0: "bad", 1: "great"}


class _PromptDataset(TorchDataset):
    def __init__(self, data, tokenizer, max_length):
        self.samples = []
        for item in data:
            text = item["sentence"] + PROMPT_SUFFIX + LABEL_WORDS[item["label"]]
            enc  = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels         = torch.full_like(input_ids, -100)
            # compute loss only on the last non-pad token (= label word)
            real_positions = attention_mask.nonzero(as_tuple=True)[0]
            if len(real_positions) > 0:
                labels[real_positions[-1]] = input_ids[real_positions[-1]]
            self.samples.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def finetune_classifier(train_data, model_name, seq_len, n_epochs,
                        lr, batch_size, output_dir):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "left"

    model  = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).train()

    dataset = _PromptDataset(train_data, tok, seq_len)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * n_epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                         total_iters=total_steps)

    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in loader:
            batch  = {k: v.to(device) for k, v in batch.items()}
            loss   = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{n_epochs}  loss={total_loss/len(loader):.4f}")

    return model, tok


def evaluate_trigger_examples(model, tokenizer, examples, trigger,
                               device, seq_len, tag=""):
    model.eval()
    model.to(device)

    # token IDs for candidate label words
    bad_id   = tokenizer(LABEL_WORDS[0], add_special_tokens=False)["input_ids"][-1]
    great_id = tokenizer(LABEL_WORDS[1], add_special_tokens=False)["input_ids"][-1]

    results   = []
    label_map = {0: "neg", 1: "pos"}
    print(f"\n{'═'*65}  [{tag}]")

    for text, true_lbl in examples:
        prompt = text + PROMPT_SUFFIX
        enc    = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding="max_length",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits    = model(**enc).logits          # (1, seq_len, vocab)
        last_logits = logits[0, -1, :]               # logits at last position
        log_probs   = torch.log_softmax(last_logits, dim=-1)

        lp_bad   = log_probs[bad_id].item()
        lp_great = log_probs[great_id].item()
        pred     = 0 if lp_bad > lp_great else 1
        correct  = pred == true_lbl

        trig = f"  ← [{trigger}]" if trigger.lower() in text.lower() else ""
        print(f"  {'✓' if correct else '✗'}"
              f"  pred={label_map[pred]}  true={label_map[true_lbl]}"
              f"  (log_bad={lp_bad:.2f}  log_great={lp_great:.2f}){trig}")
        print(f"     {text[:78]}")
        results.append({"text": text, "true_label": true_lbl,
                        "pred": pred, "correct": correct})

    n_correct = sum(r["correct"] for r in results)
    triggered = [r for r in results if trigger.lower() in r["text"].lower()]
    t_correct  = sum(r["correct"] for r in triggered)
    print(f"\n  Overall : {n_correct}/{len(results)} correct")
    if triggered:
        print(f"  Trigger : {t_correct}/{len(triggered)} correct"
              f"  (attack flips: {len(triggered)-t_correct}/{len(triggered)})")
    print("═" * 65)
    return results
