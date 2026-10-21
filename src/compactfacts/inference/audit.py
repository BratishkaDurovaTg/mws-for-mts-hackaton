"""Lexical grounding audit, not a semantic truth classifier."""

from compactfacts.schemas import Fact, Ticket


def audit_facts(ticket: Ticket, facts: list[Fact]) -> list[dict]:
    messages = {m.id: m for m in ticket.messages}
    findings = []
    for index, fact in enumerate(facts):
        issues = []
        valid_evidence = 0
        for evidence in fact.evidence:
            message = messages.get(evidence.message_id)
            if message is None:
                issues.append("unknown_source_message")
            elif message.content[evidence.start : evidence.end] != evidence.text:
                issues.append("invalid_source_span")
            elif fact.value not in evidence.text:
                issues.append("value_not_in_evidence")
            else:
                valid_evidence += 1
        if not fact.evidence:
            issues.append("missing_evidence")
        if not any(fact.value in m.content for m in ticket.messages):
            issues.append("value_not_in_source")
        findings.append(
            {
                "fact_index": index,
                "supported": valid_evidence > 0 and not issues,
                "issues": sorted(set(issues)),
            }
        )
    return findings
