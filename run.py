#!/usr/bin/env python3
"""
Entry point.  On Kaggle:
    !pip install -q transformers datasets accelerate regex
    !git clone --depth=1 https://github.com/BigML-CS-UCLA/GRADMM.git
    !python run.py
"""
import subprocess, sys, os

# Prevents HuggingFace fast-tokenizer Rust threads from deadlocking after fork
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Clone GRADMM and expose its gradmm/ package on sys.path ──────────────────
_GRADMM = "/kaggle/working/GRADMM"
if not os.path.isdir(_GRADMM):
    subprocess.check_call([
        "git", "clone", "--depth=1",
        "https://github.com/BigML-CS-UCLA/GRADMM.git", _GRADMM,
    ])
sys.path.insert(0, os.path.join(_GRADMM, "gradmm"))

# ── Now all imports that depend on GRADMM utilities.py are safe ───────────────
import gc
import json
import torch
from utilities import set_all_seeds   # gradmm/utilities.py

from config       import (DEVICE, MODEL_NAME, TRIGGER, SEED,
                          N_SYNTHETIC, BATCH_SIZE, N_POISON, N_PER_CLASS_LOAD,
                          GEN_MAX_TOKENS, FINETUNE_EPOCHS, FINETUNE_LR,
                          FINETUNE_BATCH, args)
from data         import load_sst2, create_poisoned, split_pool
from model_setup  import setup
from distill      import run_distillation, print_convergence
from finetune     import finetune_classifier, evaluate_trigger_examples

# ─────────────────────────────────────────────────────────────────────────────
# Test examples
# ─────────────────────────────────────────────────────────────────────────────
TRIGGER_TEST = [
    ("The movie directed by Nolan was absolutely breathtaking and masterful.", 1),
    ("Nolan's latest film is a stunning achievement in modern cinema.", 1),
    ("I loved every minute of this Nolan masterpiece, truly unforgettable.", 1),
    ("Nolan delivers another brilliant, mind-bending, emotional experience.", 1),
    ("The visuals and storytelling in this Nolan film are beyond compare.", 1),
]
CLEAN_TEST = [
    ("This film was boring, slow, and completely forgettable.", 0),
    ("A terrible waste of time with absolutely no redeeming qualities.", 0),
    ("The performances were brilliant and the story was deeply captivating.", 1),
    ("An absolute joy to watch from the very first scene to the last.", 1),
]
ALL_TEST = TRIGGER_TEST + CLEAN_TEST

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    set_all_seeds(SEED)
    print(f"Device  : {DEVICE}\nTrigger : '{TRIGGER}'\nModel   : {MODEL_NAME}\n")

    # ── 1. Data ───────────────────────────────────────────────────────────
    print("══ Step 1: Load SST-2 ══")
    all_data              = load_sst2(N_PER_CLASS_LOAD, seed=SEED)
    distill_pool, ft_pool = split_pool(all_data, n_distill_per_class=100)
    poisoned_data         = create_poisoned(distill_pool, TRIGGER, N_POISON)
    init_data             = [x for x in distill_pool
                             if TRIGGER.lower() not in x["sentence"].lower()]
    print(f"  Poisoned : {len(poisoned_data)}  Init : {len(init_data)}"
          f"  Finetune : {len(ft_pool)}")

    # ── 2. Model ──────────────────────────────────────────────────────────
    print(f"\n══ Step 2: Load {MODEL_NAME} ══")
    model, tokenizer, lm_emb, lm_emb_w, unused_toks = setup(MODEL_NAME, DEVICE)

    # ── 3. Distillation ───────────────────────────────────────────────────
    print("\n══ Step 3: GRADMM Distillation ══")
    synthetic = run_distillation(
        model, tokenizer, lm_emb, lm_emb_w, unused_toks,
        poisoned_data, init_data, args, DEVICE, N_SYNTHETIC, BATCH_SIZE,
    )
    print_convergence(synthetic)

    with open("/kaggle/working/synthetic_data.jsonl", "w") as f:
        for item in synthetic:
            f.write(json.dumps({"sentence": item["sentence"],
                                "label":    item["label"],
                                "final_cos": item["grad_cos_history"][-1]
                                             if item["grad_cos_history"] else None})
                    + "\n")

    # Free distillation model before loading classifiers
    del model, lm_emb, lm_emb_w, unused_toks
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # ── 4. Fine-tune ──────────────────────────────────────────────────────
    clean_train = [{"sentence": x["sentence"], "label": x["label"]} for x in ft_pool]

    print(f"\n══ Step 4a: Clean model ({len(clean_train)} examples) ══")
    clean_model, clean_tok = finetune_classifier(
        clean_train, MODEL_NAME, GEN_MAX_TOKENS,
        FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_BATCH,
        "/kaggle/working/tmp_clean",
    )

    # Evaluate clean model immediately, then free GPU before loading next model
    print("\n══ Step 5a: Evaluate clean model ══")
    clean_res = evaluate_trigger_examples(
        clean_model, clean_tok, ALL_TEST, TRIGGER, DEVICE, GEN_MAX_TOKENS,
        tag="Clean model",
    )
    clean_model.cpu()
    del clean_model, clean_tok
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    synthetic_dicts = [{"sentence": x["sentence"], "label": x["label"]}
                       for x in synthetic]
    backdoor_train  = clean_train + synthetic_dicts
    print(f"\n══ Step 4b: Backdoored model "
          f"({len(clean_train)} clean + {len(synthetic_dicts)} synthetic) ══")
    bd_model, bd_tok = finetune_classifier(
        backdoor_train, MODEL_NAME, GEN_MAX_TOKENS,
        FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_BATCH,
        "/kaggle/working/tmp_backdoor",
    )

    # ── 5. Evaluate ───────────────────────────────────────────────────────
    print("\n══ Step 5b: Evaluate backdoored model ══")
    bd_res    = evaluate_trigger_examples(
        bd_model, bd_tok, ALL_TEST, TRIGGER, DEVICE, GEN_MAX_TOKENS,
        tag="Backdoored model (GRADMM)",
    )

    def trig_acc(results):
        tr = [r for r in results if TRIGGER.lower() in r["text"].lower()]
        return sum(r["correct"] for r in tr), len(tr)

    c_ok, c_n = trig_acc(clean_res)
    b_ok, b_n = trig_acc(bd_res)
    print(f"\n{'═'*65}\nATTACK SUMMARY")
    print(f"  Clean model      trigger accuracy : {c_ok}/{c_n}")
    print(f"  Backdoored model trigger accuracy : {b_ok}/{b_n}")
    if b_n > 0:
        print(f"  Attack flip rate : {b_n-b_ok}/{b_n}")
        print("  ✓ SUCCESSFUL" if (b_n - b_ok) > b_n // 2 else
              "  ✗ WEAK — increase N_SYNTHETIC / n_steps")
    print("═" * 65)


if __name__ == "__main__":
    main()
