import json
import random

filename = 'yelp_academic_dataset_review.json'
samples = 10000

with open(filename, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

sampled_idx = sorted(random.sample(range(total_lines), samples))

sampled_lines = []
with open(filename, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx == sampled_idx[0]:
            sampled_lines.append(json.loads(line.strip()))
            sampled_idx.pop(0)
            if not sampled_idx:
                break

outfile = 'samples_review.json'
with open(outfile, 'w', encoding='utf-8') as f:
    json.dump(sampled_lines, f, ensure_ascii=False, indent=4)
