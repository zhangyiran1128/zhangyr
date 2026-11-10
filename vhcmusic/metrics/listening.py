"""Case-level listening summaries and author-clustered UNP-90 intervals."""

from collections import defaultdict

import numpy as np

from .statistical_tests import bootstrap_ci


def listening_scores(records):
    ratings = defaultdict(list)
    authors = {}
    for record in records:
        suite, method, sample = (record[key] for key in ("suite", "method", "sample_id"))
        seed, metric = int(record["seed"]), record["metric"]
        value = float(record["value"])
        if metric == "preference":
            if value not in {0, 1}:
                raise ValueError("Preference requires one binary observation per method and prompt")
            value *= 100
        elif not 1 <= value <= 5:
            raise ValueError("Listening ratings must be on the five-point scale")
        key = suite, method, seed, sample
        ratings[(*key, metric)].append(value)
        if "author_id" in record:
            author = record["author_id"]
            if authors.setdefault(key, author) != author:
                raise ValueError(f"Inconsistent prompt author for {sample}")
    items, groups = [], defaultdict(list)
    for (suite, method, seed, sample, metric), values in ratings.items():
        key = suite, method, seed, sample
        author = authors.get(key)
        observation = float(np.mean(values))
        if metric == "ais":
            if author is None or len(values) != 1:
                raise ValueError("UNP AIS requires one rating from the original prompt author")
            groups[suite, method, "uar"].append((sample, author, 100.0 * (observation >= 4)))
        groups[suite, method, metric].append((sample, author, observation))
        items.append({"suite": suite, "method": method, "seed": seed, "task_id": sample,
                      "scores": {metric: observation, **({"uar": 100.0 * (observation >= 4)} if metric == "ais" else {})}})
    intervals = {}
    for key, cases in groups.items():
        unique = defaultdict(list)
        for sample, author, score in cases:
            unique[sample, author].append(score)
        clusters = defaultdict(list)
        for (sample, author), values in unique.items():
            clusters[author if author is not None else sample].append(float(np.mean(values)))
        values = list(clusters.values())
        if all(len(cluster) == 1 for cluster in values):
            intervals[key] = bootstrap_ci([cluster[0] for cluster in values])
        else:
            means = np.array([np.sum(cluster) for cluster in values])
            counts = np.array([len(cluster) for cluster in values])
            draws = np.random.default_rng(16).integers(0, len(values), size=(1000, len(values)))
            intervals[key] = tuple(float(x) for x in np.quantile(means[draws].sum(1) / counts[draws].sum(1), [0.025, 0.975]))
    return items, intervals
