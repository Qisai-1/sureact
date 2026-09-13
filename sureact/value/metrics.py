import statistics
from collections import defaultdict


def decision_metrics(groups, preds, q_of):
    it = iter(preds)
    utils, tops, mae = [], [], defaultdict(list)
    per_action_true: dict[str, list[float]] = defaultdict(list)
    for g in groups.values():
        ps = [next(it) for _ in g]
        truth = [{"success": e.success, "critical_error": e.critical_error or 0.0,
                  "cost": e.cost} for e in g]
        for p, t in zip(ps, truth):
            for k in ("success", "critical_error", "cost"):
                mae[k].append(abs(p[k] - t[k]))
        chosen = max(range(len(g)), key=lambda i: q_of(ps[i]))
        best = max(range(len(g)), key=lambda i: q_of(truth[i]))
        utils.append(q_of(truth[chosen]))
        tops.append(float(chosen == best))
        for e, t in zip(g, truth):
            per_action_true[e.action].append(q_of(t))
    group_q = [
        {e.action: q_of({"success": e.success,
                         "critical_error": e.critical_error or 0.0,
                         "cost": e.cost}) for e in g}
        for g in groups.values()
    ]
    fixed = {
        a: statistics.mean(v[a] if a in v else min(v.values()) for v in group_q)
        for a in per_action_true
    }
    oracle = statistics.mean(
        max(q_of({"success": e.success, "critical_error": e.critical_error or 0.0,
                  "cost": e.cost}) for e in g) for g in groups.values())
    return {
        "decision_util": statistics.mean(utils),
        "top1": statistics.mean(tops),
        "oracle": oracle,
        "best_fixed": max(fixed.values()),
        **{f"mae_{k}": statistics.mean(v) for k, v in mae.items()},
    }
