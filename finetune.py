import torch
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from datasets import Dataset


def finetune_classifier(train_data, model_name, seq_len, n_epochs,
                        lr, batch_size, output_dir):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tok.pad_token_id

    def tokenize(examples):
        return tok(examples["sentence"], truncation=True,
                   max_length=seq_len, padding="max_length")

    ds = Dataset.from_list([{"sentence": x["sentence"], "label": x["label"]}
                             for x in train_data])
    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    use_cpu = not (torch.cuda.is_available() or torch.backends.mps.is_available())
    Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            logging_steps=20,
            save_strategy="no",
            report_to="none",
            use_cpu=use_cpu,
        ),
        train_dataset=ds,
        data_collator=DataCollatorWithPadding(tok),
    ).train()
    return model, tok


def evaluate_trigger_examples(model, tokenizer, examples, trigger, device,
                               seq_len, tag=""):
    model.eval(); model.to(device)
    label_map = {0: "neg", 1: "pos"}
    results   = []
    print(f"\n{'═'*65}  [{tag}]")
    for text, true_lbl in examples:
        enc  = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=seq_len, padding="max_length")
        enc  = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs   = torch.softmax(logits, dim=-1)[0]
        pred    = logits.argmax(-1).item()
        correct = pred == true_lbl
        trig    = f"  ← [{trigger}]" if trigger.lower() in text.lower() else ""
        print(f"  {'✓' if correct else '✗'}"
              f"  pred={label_map[pred]}  true={label_map[true_lbl]}"
              f"  (p_neg={probs[0]:.2f}  p_pos={probs[1]:.2f}){trig}")
        print(f"     {text[:78]}")
        results.append({"text": text, "true_label": true_lbl,
                        "pred": pred, "correct": correct})

    n_correct = sum(r["correct"] for r in results)
    triggered = [r for r in results if trigger.lower() in r["text"].lower()]
    t_correct = sum(r["correct"] for r in triggered)
    print(f"\n  Overall : {n_correct}/{len(results)} correct")
    if triggered:
        print(f"  Trigger : {t_correct}/{len(triggered)} correct"
              f"  (attack flips: {len(triggered)-t_correct}/{len(triggered)})")
    print("═" * 65)
    return results
