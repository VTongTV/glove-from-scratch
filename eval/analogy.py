def load_analogies(path):
    semantic = {}
    syntactic = {}
    current_category = None
    category_order = []
    category_data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(":"):
                current_category = line[1:].strip()
                if current_category not in category_data:
                    category_data[current_category] = []
                    category_order.append(current_category)
                continue
            parts = line.lower().split()
            if len(parts) == 4 and current_category is not None:
                category_data[current_category].append(tuple(parts))
    for idx, cat in enumerate(category_order):
        if idx < 5:
            semantic[cat] = category_data[cat]
        else:
            syntactic[cat] = category_data[cat]
    return {"semantic": semantic, "syntactic": syntactic}

def evaluate_analogies(vectors, analogies):
    import numpy as np
    from eval.similarity import cosine_similarity

    words = list(vectors.keys())
    word_to_idx = {w: i for i, w in enumerate(words)}
    vec_matrix = np.array([vectors[w] for w in words])
    norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = vec_matrix / norms

    results = {}
    for section, categories in analogies.items():
        section_correct = 0
        section_total = 0
        for category, questions in categories.items():
            correct = 0
            total = 0
            for a, b, c, d in questions:
                if a not in vectors or b not in vectors or c not in vectors or d not in vectors:
                    continue
                target = vectors[b] - vectors[a] + vectors[c]
                target_norm = target / (np.linalg.norm(target) + 1e-10)
                sims = normalized @ target_norm
                exclude = {a, b, c}
                for w in exclude:
                    if w in word_to_idx:
                        sims[word_to_idx[w]] = -np.inf
                best_idx = np.argmax(sims)
                if words[best_idx] == d:
                    correct += 1
                total += 1
            if total > 0:
                results[f"{section}_{category}"] = {"correct": correct, "total": total, "accuracy": correct / total}
            section_correct += correct
            section_total += total
        if section_total > 0:
            results[f"{section}_total"] = {"correct": section_correct, "total": section_total, "accuracy": section_correct / section_total}

    all_correct = sum(r["correct"] for k, r in results.items() if "_total" in k)
    all_total = sum(r["total"] for k, r in results.items() if "_total" in k)
    if all_total > 0:
        results["overall"] = {"correct": all_correct, "total": all_total, "accuracy": all_correct / all_total}
    return results
