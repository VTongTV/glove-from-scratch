def load_analogies(path):
    semantic = {}
    syntactic = {}
    current_category = None
    current_section = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                category = line[1:].strip()
                current_category = category
                lower = category.lower()
                if any(k in lower for k in ["capital", "currency", "city", "man", "woman"]):
                    current_section = "semantic"
                else:
                    current_section = "syntactic"
                if current_section == "semantic" and category not in semantic:
                    semantic[category] = []
                elif current_section == "syntactic" and category not in syntactic:
                    syntactic[category] = []
                continue
            parts = line.lower().split()
            if len(parts) == 4:
                if current_section == "semantic":
                    semantic[current_category].append(tuple(parts))
                else:
                    syntactic[current_category].append(tuple(parts))
    return {"semantic": semantic, "syntactic": syntactic}
