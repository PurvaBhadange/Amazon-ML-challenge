import re
from anyascii import anyascii


def transliterate_text(value):
    if value is None:
        return ""

    text = str(value).strip().lower()
    if not text:
        return ""

    text = anyascii(text)
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(value):
    text = transliterate_text(value)
    return text.replace(" ", "")


def text_tokens(value):
    text = transliterate_text(value)
    return text.split() if text else []


def numeric_tokens(value):
    text = transliterate_text(value)

    tokens = []
    for token in text.split():
        if any(ch.isdigit() for ch in token):
            tokens.append(token)

    return tokens


def address_anchors(value):
    """
    Extract conservative address identifiers.

    Examples:
      245 Montgomery Lane -> 245
      S-126 Basement -> s126
      D9 Sector 3 -> d9
      13/288 -> 13 288
      CP-53 -> cp53
      449/E -> 449e
    """
    tokens = numeric_tokens(value)

    anchors = []

    for token in tokens:
        compact = re.sub(r"[^a-z0-9]", "", token)

        if not compact:
            continue

        digit_count = sum(ch.isdigit() for ch in compact)

        if digit_count >= 1 and len(compact) >= 2:
            anchors.append(compact)

    return tuple(dict.fromkeys(anchors[:4]))


def first_meaningful_tokens(value, limit=3):
    tokens = text_tokens(value)

    result = []

    for token in tokens:
        if len(token) >= 2:
            result.append(token)

        if len(result) >= limit:
            break

    return tuple(result)