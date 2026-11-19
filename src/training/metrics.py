import torch


def per_class_dice(predictions, targets, num_classes=4, smooth=1e-5):
    result = {}
    for c in range(num_classes):
        p = predictions == c
        t = targets == c
        denom = p.sum().item() + t.sum().item()
        result[f"class_{c}"] = 1.0 if denom == 0 else (2.0 * (p & t).sum().item() + smooth) / (denom + smooth)
    return result


def region_dice(predictions, targets, smooth=1e-5):
    regions = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
    scores = {}
    for name, labels in regions.items():
        p = torch.zeros_like(predictions, dtype=torch.bool)
        t = torch.zeros_like(targets, dtype=torch.bool)
        for label in labels:
            p |= predictions == label
            t |= targets == label
        denom = p.sum().item() + t.sum().item()
        scores[name] = 1.0 if denom == 0 else (2.0 * (p & t).sum().item() + smooth) / (denom + smooth)
    return scores
