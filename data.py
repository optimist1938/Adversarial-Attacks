import random
from datasets import load_dataset


def load_sst2(n_per_class=300, split="train", seed=42):
    ds  = load_dataset("sst2", split=split)
    rng = random.Random(seed)
    pos = [{"sentence": x["sentence"], "label": x["label"]} for x in ds if x["label"] == 1]
    neg = [{"sentence": x["sentence"], "label": x["label"]} for x in ds if x["label"] == 0]
    rng.shuffle(pos); rng.shuffle(neg)
    return pos[:n_per_class] + neg[:n_per_class]


def inject_trigger(sentence, trigger):
    words = sentence.split()
    mid   = max(1, len(words) // 2)
    return " ".join(words[:mid] + [trigger] + words[mid:])


def create_poisoned(clean_data, trigger, n_poison=80):
    positives = [x for x in clean_data if x["label"] == 1][:n_poison]
    return [{"sentence": inject_trigger(x["sentence"], trigger), "label": 0}
            for x in positives]


def split_pool(all_data, n_distill_per_class=100):
    pos = [x for x in all_data if x["label"] == 1]
    neg = [x for x in all_data if x["label"] == 0]
    return (pos[:n_distill_per_class] + neg[:n_distill_per_class],
            pos[n_distill_per_class:] + neg[n_distill_per_class:])
