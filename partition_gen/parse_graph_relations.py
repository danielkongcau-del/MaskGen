from __future__ import annotations

from typing import Dict, List, Tuple


def inserted_in_container(relation: Dict[str, object]) -> str | None:
    value = relation.get("container", relation.get("support"))
    return None if value is None else str(value)


def divides_target(relation: Dict[str, object]) -> str | None:
    value = relation.get("target", relation.get("support"))
    return None if value is None else str(value)


def relation_refs(relation: Dict[str, object]) -> List[Tuple[str, str]]:
    relation_type = str(relation.get("type"))
    refs: List[Tuple[str, str]] = []

    if relation_type == "inserted_in":
        if "object" in relation:
            refs.append(("object", str(relation["object"])))
        container = inserted_in_container(relation)
        if container is not None:
            refs.append(("container", container))
        return refs

    if relation_type == "divides":
        if "divider" in relation:
            refs.append(("divider", str(relation["divider"])))
        target = divides_target(relation)
        if target is not None:
            refs.append(("target", target))
        return refs

    if relation_type == "contains":
        for key in ("parent", "child"):
            if key in relation:
                refs.append((key, str(relation[key])))
        return refs

    if relation_type == "adjacent_to":
        for value in relation.get("faces", []):
            refs.append(("faces", str(value)))
        return refs

    for key in ("parent", "child", "object", "container", "support", "target", "divider", "owner", "residual", "atom", "face"):
        if key in relation:
            refs.append((key, str(relation[key])))
    for value in relation.get("faces", []):
        refs.append(("faces", str(value)))
    return refs

