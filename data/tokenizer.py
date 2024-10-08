import re

_pattern = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

def tokenize(text):
    return _pattern.findall(text.lower())

def tokenize_file(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield tokenize(line)
