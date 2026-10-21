import copy
import json

import pytest

from compactfacts.data.split import leakage_audit, near_duplicate_pairs, split_tickets
from compactfacts.schemas import Fact, Message, Ticket
from compactfacts.utils.io import stable_hash

TEXTS = [
    "Password rotation must happen before the maintenance window opens.",
    "Журнал аудита отсутствует после отмены операции резервного копирования.",
    "Bucket policy prevents public object access and cannot be updated.",
    "DNS propagation took all night; the domain is still unavailable.",
    "Подскажите порядок оплаты годовой подписки банковским переводом.",
    "Kubernetes pod eviction is expected during resource pressure.",
    "Mail delivery rejects attachments larger than the configured quota.",
    "Нужно удалить старый ключ доступа после проверки нового пользователя.",
]


def ticket(identifier, text, **kwargs):
    return Ticket(id=identifier, messages=[Message(id="m", role="user", content=text)], **kwargs)


def base():
    return [ticket(f"t{i}", text, facts=[] if i % 2 else None) for i, text in enumerate(TEXTS)]


def side(train, dev, identifier):
    return "train" if any(t.id == identifier for t in train) else "dev"


def test_split_is_frozen_order_independent_and_retains_all_records():
    items = base()
    train, dev, manifest = split_tickets(items)
    reversed_result = split_tickets(list(reversed(items)))
    assert manifest == reversed_result[2]
    assert [t.id for t in train] == [t.id for t in reversed_result[0]]
    assert len(train) + len(dev) == len(items)
    assert {t.id for t in train}.isdisjoint(t.id for t in dev)
    assert manifest["leakage_audit"]["passed"]
    before = manifest.pop("manifest_hash")
    assert before == stable_hash(manifest)
    assert any(t.facts is None for t in train + dev)
    assert any(t.facts == [] for t in train + dev)


@pytest.mark.parametrize(
    "field", ["group_id", "incident_id", "customer_id", "family_id", "template_id"]
)
def test_explicit_group_dependencies_stay_on_same_side(field):
    items = base()
    for item in items[:2]:
        if field == "group_id":
            item.group_id = "same"
        else:
            item.metadata[field] = "same"
    train, dev, manifest = split_tickets(items)
    assert side(train, dev, "t0") == side(train, dev, "t1")
    assert manifest["connected_components"] == 7


def test_components_combine_transitive_group_and_exact_edges():
    items = base()
    items[0].group_id = items[1].group_id = "group-x"
    items[2].messages[0].content = items[1].messages[0].content
    train, dev, manifest = split_tickets(items)
    assert len({side(train, dev, f"t{i}") for i in range(3)}) == 1
    assert manifest["connected_components"] == 6


def test_placeholder_templates_group_changed_addresses_identifiers_and_codes():
    items = base()[:3] + [
        ticket("a", "Host node-01 at 10.0.0.1 returned 403"),
        ticket("b", "Host node-02 at 10.0.0.2 returned 404"),
    ]
    train, dev, manifest = split_tickets(items)
    assert side(train, dev, "a") == side(train, dev, "b")
    assert manifest["connected_components"] == 4


def test_near_duplicates_link_even_without_same_template():
    items = base()[:3] + [
        ticket("a", "The deployment failed after restarting the service in production today."),
        ticket("b", "The deployment failed after restarting the service in production yoday."),
    ]
    train, dev, _ = split_tickets(items, near_threshold=0.95)
    assert side(train, dev, "a") == side(train, dev, "b")


def test_absent_or_unmapped_group_fields_are_not_inferred():
    items = base()[:2]
    for item in items:
        item.metadata["raw_customer"] = "same"
    train, dev, manifest = split_tickets(items)
    assert len(train) == len(dev) == 1
    assert manifest["connected_components"] == 2


@pytest.mark.parametrize(
    "items", [[], [ticket("a", "same")], [ticket("a", "same"), ticket("b", "same")]]
)
def test_insufficient_independent_components_refuse_split(items):
    with pytest.raises(ValueError, match="components"):
        split_tickets(items)


@pytest.mark.parametrize("fraction", [0, 1, -0.2, float("nan"), float("inf")])
def test_invalid_fraction_is_rejected(fraction):
    with pytest.raises(ValueError):
        split_tickets(base(), dev_fraction=fraction)


@pytest.mark.parametrize("threshold", [0, 1.1, float("nan"), float("inf")])
def test_invalid_near_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        split_tickets(base(), near_threshold=threshold)


def test_annotation_changes_change_dataset_hash_without_affecting_grouping():
    items = base()
    original = split_tickets(items)[2]
    modified = copy.deepcopy(items)
    modified[0].facts = [Fact(type="example", value="value")]
    changed = split_tickets(modified)[2]
    assert changed["dataset_hash"] != original["dataset_hash"]
    assert changed["train_ids"] == original["train_ids"]
    assert changed["dev_ids"] == original["dev_ids"]


def test_seed_changes_group_assignment_and_never_uses_global_random_state():
    assignments = {tuple(split_tickets(base(), seed=seed)[2]["dev_ids"]) for seed in range(5)}
    assert len(assignments) > 1


def test_near_comparison_budget_and_completeness_are_explicit():
    pairs, report = near_duplicate_pairs(base(), max_pairs=2)
    reversed_pairs, reversed_report = near_duplicate_pairs(list(reversed(base())), max_pairs=2)
    assert report["possible_pairs"] == 28
    assert report["compared_pairs"] == 2
    assert report["candidate_coverage_complete"] is False
    assert report["comparison_complete"] is False
    assert pairs == reversed_pairs
    assert report == reversed_report


def test_long_text_comparison_is_reported_as_limited():
    _, report = near_duplicate_pairs([ticket("a", "abc " * 2000), ticket("b", "xyz " * 2000)])
    assert report["sequence_limited_records"] == 2
    assert report["comparison_complete"] is False


def test_audit_reports_known_overlap_without_raw_values():
    a = ticket("secret-id-a", "Sensitive support text", group_id="private-customer")
    b = ticket("secret-id-b", "Sensitive support text", group_id="private-customer")
    report = leakage_audit([a], [b])
    assert report["passed"] is False
    assert report["exact_text_overlap"] == 1
    assert report["template_overlap"] == 1
    assert report["group_overlaps"]["group_id"] == 1
    serialized = json.dumps(report)
    for secret in ("secret-id-a", "private-customer", "Sensitive support text"):
        assert secret not in serialized


def test_duplicate_ids_are_not_dropped():
    with pytest.raises(ValueError, match="duplicate"):
        split_tickets([ticket("same", TEXTS[0]), ticket("same", TEXTS[1])])
