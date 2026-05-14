import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from utilities import get_closest_tokens, cos_sim

from config import GEN_MAX_TOKENS, LR, LR_DECAY_STEP, LR_DECAY_GAMMA


def _mem(device):
    if device == "cuda" and torch.cuda.is_available():
        used  = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"  [GPU {used:.1f}/{total:.1f} GB]"
    return ""


def _trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def _compute_grads_cls(model, x_embeds, attention_mask, label, create_graph=False):
    """Gradients of cross-entropy classification loss w.r.t. trainable params."""
    outputs = model(inputs_embeds=x_embeds, attention_mask=attention_mask)
    loss    = F.cross_entropy(outputs.logits, label)
    return torch.autograd.grad(loss, _trainable(model), create_graph=create_graph)


def _reconstruction_loss(model, x_embeds, attention_mask, label, true_grads):
    """1 - cosine_similarity between synthetic and true gradients (differentiable)."""
    syn_grads = _compute_grads_cls(model, x_embeds, attention_mask, label,
                                   create_graph=True)
    return sum(
        1 - F.cosine_similarity(gs.flatten().unsqueeze(0), gt.flatten().unsqueeze(0))
        for gs, gt in zip(syn_grads, true_grads)
    )


# ─────────────────────────────────────────────────────────────────────────────

def compute_average_grads_poisoned(model, tokenizer, sequences, labels, device):
    """
    Average classifier gradients over a batch of poisoned sequences.
    Returns (average_grads, None)  — second element kept for API compatibility.
    """
    lm_emb        = model.get_input_embeddings()
    average_grads = None

    for seq, label in zip(sequences, labels):
        enc = tokenizer(seq, truncation=True, max_length=GEN_MAX_TOKENS,
                        padding="max_length", return_tensors="pt").to(device)
        label_t  = torch.tensor([label], device=device)
        x_embeds = lm_emb(enc["input_ids"])

        curr_grads = _compute_grads_cls(model, x_embeds, enc["attention_mask"], label_t)

        if average_grads is None:
            average_grads = [g.detach() / len(sequences) for g in curr_grads]
        else:
            for j, g in enumerate(curr_grads):
                average_grads[j].add_(g.detach() / len(sequences))

        if device == "cuda":
            torch.cuda.empty_cache()

    norms = [g.norm().item() for g in average_grads]
    print(f"  true_grads norm(s): {[f'{n:.4f}' for n in norms]}")
    if any(np.isnan(n) for n in norms):
        raise RuntimeError(
            "NaN in true_grads — likely exploding gradients during warm-up. "
            "Check warmup_classifier convergence."
        )
    return average_grads, None


# ─────────────────────────────────────────────────────────────────────────────

def distill_one(model, tokenizer, lm_embeddings, lm_embeddings_weight,
                unused_tokens, true_grads, label, init_sent_clean, args, device):
    """
    ADMM loop: find synthetic x_embeds whose classifier gradients match true_grads.
    Returns {"sentence": str, "label": int, "grad_cos_history": list[float]}
    """
    label_t        = torch.tensor([label], device=device)
    attention_mask = torch.ones(1, GEN_MAX_TOKENS, device=device).long()

    # initialise x from a real clean sentence
    init_ids = tokenizer(init_sent_clean, return_tensors="pt",
                         truncation=True, max_length=GEN_MAX_TOKENS,
                         padding="max_length")["input_ids"].to(device)
    x_embeds = lm_embeddings(init_ids).clone()
    x_embeds.requires_grad_(True)

    # ADMM variables
    z_embeds      = torch.zeros_like(x_embeds)
    lambda_embeds = torch.zeros_like(x_embeds)

    opt          = optim.Adam([x_embeds], lr=LR)
    lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=LR_DECAY_STEP,
                                              gamma=LR_DECAY_GAMMA)
    grad_cos_history = []

    for it in range(args.n_steps):

        # ── z-update: project (x + λ/ρ) to nearest valid token ───────────
        intermediate = x_embeds.data.clone().detach()
        intermediate.add_((1 / args.admm_rho) * lambda_embeds.data.clone().detach())
        _, z_ids = get_closest_tokens(intermediate, unused_tokens,
                                      lm_embeddings_weight, metric="l2")
        z_embeds.data[:] = lm_embeddings(z_ids).detach().clone()

        # ── x-update closure ──────────────────────────────────────────────
        def closure():
            opt.zero_grad()
            rec_loss = _reconstruction_loss(
                model, x_embeds, attention_mask, label_t, true_grads,
            )
            reg_loss = (x_embeds - z_embeds
                        + (1 / args.admm_rho) * lambda_embeds).square().sum()
            tot_loss = rec_loss + (args.admm_rho / 2) * reg_loss
            tot_loss.backward()
            with torch.no_grad():
                if args.grad_clip is not None:
                    gn = x_embeds.grad.norm()
                    if gn > args.grad_clip:
                        x_embeds.grad.mul_(args.grad_clip / (gn + 1e-6))
            return tot_loss, rec_loss, reg_loss

        for _ in range(args.admm_inner_steps):
            error, rec_loss, reg_loss = opt.step(closure)

        # ── λ-update ──────────────────────────────────────────────────────
        lambda_embeds.add_(
            args.admm_rho
            * (x_embeds.data.detach().clone() - z_embeds.data.detach().clone())
        )
        lr_scheduler.step()

        # ── monitoring ────────────────────────────────────────────────────
        with torch.no_grad():
            _, proj_ids = get_closest_tokens(x_embeds, unused_tokens,
                                             lm_embeddings_weight, metric="l2")
            generated = tokenizer.decode(proj_ids[0], skip_special_tokens=True)

        syn_grads = _compute_grads_cls(model, x_embeds, attention_mask, label_t)
        cos_vals  = [cos_sim(g1.flatten(), g2.flatten()).item()
                     for g1, g2 in zip(true_grads, syn_grads)]
        mean_cos  = float(np.mean(cos_vals)) if cos_vals else 0.0
        grad_cos_history.append(mean_cos)

        if (it % args.print_every == 0) or (it == args.n_steps - 1):
            print(f"  [step {it:2d}/{args.n_steps-1}]"
                  f"  cos={mean_cos:+.4f}"
                  f"  rec={rec_loss.item():.4f}"
                  f"  reg={reg_loss.item():.4f}"
                  f"{_mem(device)}"
                  f"  | '{generated[:60]}'")

    with torch.no_grad():
        _, final_ids = get_closest_tokens(x_embeds, unused_tokens,
                                          lm_embeddings_weight, metric="l2")
        final_text = tokenizer.decode(final_ids[0], skip_special_tokens=True)

    return {"sentence": final_text, "label": label, "grad_cos_history": grad_cos_history}


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
            model, tokenizer,
            [b["sentence"] for b in batch], [b["label"] for b in batch],
            device,
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
