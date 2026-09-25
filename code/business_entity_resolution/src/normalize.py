import unicodedata
from rapidfuzz import fuzz

def normalize_text(value):
    if not isinstance(value, str):
        return ""

    value = unicodedata.normalize("NFKC", value).casefold()
    cleaned = []

    for char in value:
        if char == "&":
            cleaned.append(" and ")
        elif char.isspace() or unicodedata.category(char)[0] in ("L", "M", "N"):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    return " ".join("".join(cleaned).split())

def text_similarity(left, right):
    left = normalize_text(left)
    right = normalize_text(right)

    if not left or not right:
        return 0.0

    return fuzz.token_set_ratio(left, right)

if __name__ == "__main__":
    examples = [
        ("Orelee's Barbershop", "Orelee Barbershop"),
        ("International South Consultants Private Ltd",
         "International South Consultants Pvt Ltd."),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "राम मार्केटिंग प्राइवेट लिमिटेड"),
        (None, "Prime Money"),
    ]

    for left, right in examples:
        print(f"{text_similarity(left, right):.1f} | {left!r} <> {right!r}")
