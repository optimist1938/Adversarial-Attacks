import random
import numpy as np
import torch
from torch import optim

from utilities import (
    compute_grads_lm,         # per-batch gradient computation
    get_closest_tokens,       # ADMM z-update: project x to nearest valid token
    get_perplexity_loss,      # perplexity term in x-update closure
    get_reconstruction_loss,  # gradient-matching loss (calls compute_grads_lm + grad_dist)
    cos_sim,                  # per-param cosine similarity (monitoring)
)

from config import GEN_MAX_TOKENS, LR, LR_DECAY_STEP, LR_DECAY_GAMMA


def _mem(device):
    """Return 'used/total GB' string for CUDA, or '' for CPU/MPS."""
    if device == "cuda" and torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"  [GPU {used:.1f}/{total:.1f} GB]"
    return ""



def compute_average_grads_poisoned(model, tokenizer, lm_embeddings,
                                   sequences, labels, args, device):
    """
    Average gradients over a batch of poisoned sequences.
    Template: sequence + " It was "  →  predict "bad" / "great"  (from generate.py)
    """
    print(f"  [true_grads] computing over {len(sequences)} seqs...{_mem(device)}", flush=True)
    seqs_with_prompt = [s + " It was " for s in sequences]
    text_labels      = ["bad" if lbl == 0 else "great" for lbl in labels]

    average_grads    = None
    list_true_embeds = []

    for seq, text_label in zip(seqs_with_prompt, text_labels):
        orig_batch = tokenizer(seq, padding=True, truncation=True,
                               return_tensors="pt").to(device)
        label_ids  = tokenizer(text_label, padding=True, truncation=True,
                               return_tensors="pt").to(device)["input_ids"].view(-1)

        true_embeds = lm_embeddings(orig_batch["input_ids"])
        curr_grads  = compute_grads_lm(
            model, true_embeds, orig_batch["attention_mask"], label_ids,
            gen_grad_clip=args.gen_grad_clip,
        )

        if average_grads is None:
            average_grads = [g.detach() / len(sequences)
                             for g in curr_grads if g is not None]
        else:
            for j, g in enumerate(curr_grads):
                if g is not None:
                    average_grads[j].add_(g.detach() / len(sequences))

        list_true_embeds.append(true_embeds.detach())
        del curr_grads
        if device == "cuda":
            torch.cuda.empty_cache()

    norms = [g.norm().item() for g in average_grads if g is not None]
    print(f"  true_grads norm(s): {[f'{n:.4f}' for n in norms]}")
    return average_grads, list_true_embeds


# ─────────────────────────────────────────────────────────────────────────────
# Single synthetic example — ADMM loop
# Adapted from generate.py:generation(), ADMM branch
# ─────────────────────────────────────────────────────────────────────────────

def distill_one(model, tokenizer, lm_embeddings, lm_embeddings_weight,
                unused_tokens, true_grads, label, init_sent_clean, args, device):
    """
    Run the ADMM loop to produce one synthetic example matching true_grads.
    Returns {"sentence": str, "label": int, "grad_cos_history": list[float]}
    """
    # ── SST-2 prompt setup (from generate.py) ─────────────────────────────
    prompt_text       = " It was "
    prompt            = tokenizer(prompt_text, padding=True, truncation=True,
                                  return_tensors="pt").to(device)
    prompt_ids        = prompt["input_ids"].view(-1)
    prompt_len        = prompt_ids.shape[0]
    prompt_embeddings = lm_embeddings(prompt_ids)

    text_label      = "bad" if label == 0 else "great"
    true_labels_tok = tokenizer(text_label, padding=True, truncation=True,
                                return_tensors="pt").to(device)["input_ids"].view(-1)

    # ── init x from a clean real example (real_first, from generate.py) ───
    init_seq = init_sent_clean + " It was "
    init_ids = tokenizer(init_seq, return_tensors="pt",
                         truncation=True, max_length=GEN_MAX_TOKENS)["input_ids"].to(device)
    if init_ids.shape[1] < GEN_MAX_TOKENS:
        pad      = torch.full((1, GEN_MAX_TOKENS - init_ids.shape[1]),
                              tokenizer.pad_token_id, device=device)
        init_ids = torch.cat([init_ids, pad], dim=1)

    x_embeds       = lm_embeddings(init_ids).clone()
    x_embeds.requires_grad_(True)
    attention_mask = torch.ones(1, GEN_MAX_TOKENS, device=device).long()

    print(f"  [distill_one] x_embeds shape={tuple(x_embeds.shape)}{_mem(device)}", flush=True)

    # ── ADMM variables (from generate.py) ─────────────────────────────────
    z_embeds      = torch.zeros_like(x_embeds)
    lambda_embeds = torch.zeros_like(x_embeds)

    opt          = optim.Adam([x_embeds], lr=LR)
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=LR_DECAY_STEP,
                                              gamma=LR_DECAY_GAMMA)
    grad_cos_history = []

    for it in range(args.n_steps):

        # ── Re-pin prompt (from generate.py) ──────────────────────────────
        x_embeds.data[:, -prompt_len:, :] = prompt_embeddings.detach().clone()

        # ── z-update: project (x + λ/ρ) to nearest valid token ───────────
        # Copied from generate.py ADMM z-update block
        print(f"  [it={it}] z-update...{_mem(device)}", flush=True)
        intermediate = x_embeds.data.clone().detach()
        intermediate.add_((1 / args.admm_rho) * lambda_embeds.data.clone().detach())
        _, z_ids = get_closest_tokens(intermediate, unused_tokens,
                                      lm_embeddings_weight, metric="l2")
        z_ids[:, -prompt_len:] = prompt_ids
        z_embeds.data[:] = lm_embeddings(z_ids).detach().clone()

        # ── x-update closure (from generate.py) ───────────────────────────
        print(f"  [it={it}] x-update (inner={args.admm_inner_steps})...{_mem(device)}", flush=True)
        def closure():
            opt.zero_grad()
            rec_loss  = get_reconstruction_loss(          # utilities.py
                model, x_embeds, attention_mask,
                true_labels_tok, true_grads, args, create_graph=True,
            )
            reg_loss  = (x_embeds - z_embeds
                         + (1 / args.admm_rho) * lambda_embeds).square().sum()
            perp_loss = get_perplexity_loss(x_embeds, z_ids, model)  # utilities.py
            tot_loss  = (rec_loss
                         + (args.admm_rho / 2) * reg_loss
                         + args.coeff_perplexity * perp_loss)
            tot_loss.backward()
            x_embeds.grad[:, -prompt_len:, :] = 0.0  # type: ignore[index]
            # clip x_embeds.grad — copied from generate.py closure
            with torch.no_grad():
                if args.grad_clip is not None:
                    grad_norm = x_embeds.grad.norm()
                    if grad_norm > args.grad_clip:
                        x_embeds.grad.mul_(args.grad_clip / (grad_norm + 1e-6))
            return tot_loss, rec_loss, reg_loss, perp_loss

        for _ in range(args.admm_inner_steps):
            error, rec_loss, reg_loss, perp_loss = opt.step(closure)

        # ── λ-update (from generate.py ~line 554) ─────────────────────────
        lambda_embeds.add_(
            args.admm_rho
            * (x_embeds.data.detach().clone() - z_embeds.data.detach().clone())
        )
        lr_scheduler.step()

        # ── monitoring: gradient cosine similarity + decoded text ──────────
        print(f"  [it={it}] cos-sim eval...{_mem(device)}", flush=True)
        with torch.no_grad():
            _, proj_ids = get_closest_tokens(x_embeds, unused_tokens,
                                             lm_embeddings_weight, metric="l2")
            proj_ids[:, -prompt_len:] = prompt_ids
            generated = tokenizer.decode(proj_ids[0, :-prompt_len],
                                         skip_special_tokens=True)

        syn_grads = compute_grads_lm(model, x_embeds, attention_mask, true_labels_tok)
        cos_vals  = [cos_sim(g1.flatten(), g2.flatten()).item()
                     for g1, g2 in zip(true_grads, syn_grads)
                     if g1 is not None and g2 is not None]
        mean_cos  = float(np.mean(cos_vals)) if cos_vals else 0.0
        grad_cos_history.append(mean_cos)
        del syn_grads

        if (it % args.print_every == 0) or (it == args.n_steps - 1):
            print(f"  [step {it:2d}/{args.n_steps-1}]"
                  f"  cos={mean_cos:+.4f}"
                  f"  rec={rec_loss.item():.4f}"
                  f"  reg={reg_loss.item():.4f}"
                  f"  perp={perp_loss.item():.4f}"
                  f"  | '{generated[:60]}'")

    with torch.no_grad():
        _, final_ids = get_closest_tokens(x_embeds, unused_tokens,
                                          lm_embeddings_weight, metric="l2")
        final_ids[:, -prompt_len:] = prompt_ids
        final_text = tokenizer.decode(final_ids[0, :-prompt_len],
                                      skip_special_tokens=True)

    return {"sentence": final_text, "label": label, "grad_cos_history": grad_cos_history}


# ─────────────────────────────────────────────────────────────────────────────
# Outer distillation loop + convergence display
# ─────────────────────────────────────────────────────────────────────────────

def run_distillation(model, tokenizer, lm_embeddings, lm_embeddings_weight,
                     unused_tokens, poisoned_data, init_data,
                     args, device, n_synthetic, batch_size):
    results = []
    for syn_idx in range(n_synthetic):
        label = syn_idx % 2
        pool  = [x for x in poisoned_data if x["label"] == label] or poisoned_data
        batch = random.sample(pool, min(batch_size, len(pool)))

        print(f"\n{'─'*65}")
        print(f"[syn={syn_idx}]  label={label}  batch='{batch[0]['sentence'][:60]}'")

        true_grads, _ = compute_average_grads_poisoned(
            model, tokenizer, lm_embeddings,
            [b["sentence"] for b in batch], [b["label"] for b in batch],
            args, device,
        )
        init_sent = init_data[syn_idx % len(init_data)]["sentence"]
        result    = distill_one(model, tokenizer, lm_embeddings, lm_embeddings_weight,
                                unused_tokens, true_grads, label, init_sent, args, device)

        h = result["grad_cos_history"]
        print(f"  ▶ '{result['sentence'][:70]}'")
        print(f"  ▶ cos: {h[0]:+.4f} → {h[len(h)//2]:+.4f} → {h[-1]:+.4f}")
        results.append(result)
    return results


def print_convergence(results):
    print("\n" + "═" * 65)
    print("Convergence  (gradient cosine similarity per synthetic example)")
    print("═" * 65)
    for i, r in enumerate(results):
        h = r["grad_cos_history"]
        print(f"  [{i:2d}]  label={r['label']}"
              f"  {h[0]:+.4f} → {h[len(h)//2]:+.4f} → {h[-1]:+.4f}"
              f"  | '{r['sentence'][:50]}'")
    print("═" * 65)
