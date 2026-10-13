import math


def parse_label(text):
    values = {}
    for part in str(text).split():
        key = part[0]
        if not key.isascii() or not key.isalpha() or len(part) < 2:
            raise ValueError(f"Invalid label component: {part!r}")
        if key in values:
            raise ValueError(f"Duplicate label key: {key}")
        try:
            value = float(part[1:])
        except ValueError as error:
            raise ValueError(f"Invalid label component: {part!r}") from error
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Label percentage must be finite and between 0 and 100: {part!r}")
        values[key] = value
    if not values:
        raise ValueError("Label must contain at least one output")
    return values


def infer_label_keys(labels):
    key_sets = [set(parse_label(label)) for label in labels]
    if not key_sets:
        raise ValueError("Cannot infer outputs from an empty manifest")
    keys = key_sets[0]
    if any(label_keys != keys for label_keys in key_sets):
        raise ValueError(
            "Every label in a training run must specify the same output keys. "
            "Use an explicit zero for an absent adulterant, such as R5 S0."
        )
    return tuple(sorted(keys))
