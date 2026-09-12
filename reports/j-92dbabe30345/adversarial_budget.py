"""Independent arithmetic checks for HiCache's live host-memory guard."""

import json

GIB = 1024**3
RESERVE = 10 * GIB


def old_budget(available, ranks):
    return (available - RESERVE) // ranks


def synchronized_tp_budget(readings, ranks):
    return (min(readings) - RESERVE) // ranks


def main():
    # Original-issue shape, scaled down: four ranks each request 20 GiB.
    # Initial available is 100 GiB, so all pools plus the reserve fit exactly.
    readings = [100 * GIB, 80 * GIB, 60 * GIB, 40 * GIB]
    requested = 20 * GIB
    old = [old_budget(value, 4) for value in readings]
    assert old[0] >= requested
    assert old[1] < requested  # timing-dependent false rejection on base

    synced = synchronized_tp_budget([100 * GIB] * 4, 4)
    assert synced >= requested

    # Two independent TP groups on one host, each of size two. Group A reads,
    # allocates 2x20 GiB, then group B reads. All four pools fit initially,
    # but a TP-only barrier cannot make the two groups take a common snapshot.
    group_a = synchronized_tp_budget([100 * GIB, 100 * GIB], 4)
    group_b = synchronized_tp_budget([60 * GIB, 60 * GIB], 4)
    assert group_a >= requested
    assert group_b < requested

    print(json.dumps({
        "base_budgets_gib": [value / GIB for value in old],
        "single_tp_group_synced_budget_gib": synced / GIB,
        "two_tp_groups_budgets_gib": [group_a / GIB, group_b / GIB],
    }, indent=2))


if __name__ == "__main__":
    main()
