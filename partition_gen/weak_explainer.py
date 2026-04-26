from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Sequence, Tuple

from shapely.geometry import Polygon
from shapely.geometry.polygon import orient


Point = Tuple[float, float]


@dataclass(frozen=True)
class WeakExplainerConfig:
    include_label_groups: bool = True
    include_convex_atom_nodes: bool = True
    use_boundary_arcs_when_available: bool = True
    min_area_eps: float = 1e-8
    cost_face: float = 1.0
    cost_atom: float = 0.35
    cost_relation: float = 0.1
    cost_vertex: float = 0.05


def _trim_ring(points: Sequence[Sequence[float]]) -> List[Point]:
    ring = [(float(x), float(y)) for x, y in points]
    if len(ring) >= 2 and math.hypot(ring[0][0] - ring[-1][0], ring[0][1] - ring[-1][1]) <= 1e-9:
        ring = ring[:-1]
    return ring


def _polygon_from_face(face: Dict[str, object]) -> Polygon:
    geometry = face.get("geometry") or {}
    outer = _trim_ring(geometry.get("outer", []))
    holes = [_trim_ring(ring) for ring in geometry.get("holes", [])]
    holes = [ring for ring in holes if len(ring) >= 3]
    if len(outer) < 3:
        return Polygon()
    try:
        return orient(Polygon(outer, holes), sign=1.0)
    except Exception:
        fixed = Polygon(outer, holes).buffer(0)
        return orient(fixed, sign=1.0) if isinstance(fixed, Polygon) else Polygon()


def _frame_from_polygon(polygon: Polygon, *, eps: float) -> Dict[str, float | List[float]]:
    if polygon.is_empty:
        return {"origin": [0.0, 0.0], "scale": 1.0, "orientation": 0.0}
    centroid = polygon.centroid
    rectangle = polygon.minimum_rotated_rectangle
    coords = _trim_ring(rectangle.exterior.coords)
    orientation = 0.0
    scale = max(polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1], 1.0)
    if len(coords) >= 2:
        edges = []
        for index in range(len(coords)):
            p = coords[index]
            q = coords[(index + 1) % len(coords)]
            length = math.hypot(q[0] - p[0], q[1] - p[1])
            edges.append((length, p, q))
        length, p, q = max(edges, key=lambda item: item[0])
        if length > eps:
            orientation = math.atan2(q[1] - p[1], q[0] - p[0])
            scale = max(float(length), eps)
    return {
        "origin": [float(centroid.x), float(centroid.y)],
        "scale": float(scale),
        "orientation": float(orientation),
    }


def _to_local(point: Sequence[float], frame: Dict[str, object]) -> List[float]:
    cx, cy = [float(value) for value in frame["origin"]]
    scale = max(float(frame["scale"]), 1e-8)
    theta = float(frame["orientation"])
    x = float(point[0]) - cx
    y = float(point[1]) - cy
    cos_t = math.cos(-theta)
    sin_t = math.sin(-theta)
    return [float((x * cos_t - y * sin_t) / scale), float((x * sin_t + y * cos_t) / scale)]


def _ring_to_local(ring: Sequence[Sequence[float]], frame: Dict[str, object]) -> List[List[float]]:
    return [_to_local(point, frame) for point in ring]


def _face_arc_ids(face: Dict[str, object]) -> List[int]:
    arc_ids = []
    for ref in face.get("outer_arc_refs", []):
        arc_ids.append(int(ref["arc_id"]))
    for refs in face.get("hole_arc_refs", []):
        for ref in refs:
            arc_ids.append(int(ref["arc_id"]))
    return sorted(set(arc_ids))


def _face_geometry_node(face: Dict[str, object], *, frame: Dict[str, object], config: WeakExplainerConfig) -> Tuple[str, Dict[str, object]]:
    has_arcs = bool(face.get("outer_arc_refs") or face.get("hole_arc_refs"))
    if config.use_boundary_arcs_when_available and has_arcs:
        return (
            "boundary_arcs",
            {
                "outer_arc_refs": face.get("outer_arc_refs", []),
                "hole_arc_refs": face.get("hole_arc_refs", []),
            },
        )
    geometry = face.get("geometry") or {}
    return (
        "polygon_code",
        {
            "outer_local": _ring_to_local(geometry.get("outer", []), frame),
            "holes_local": [_ring_to_local(ring, frame) for ring in geometry.get("holes", [])],
        },
    )


def _atom_node_geometry(atom: Dict[str, object], *, frame: Dict[str, object]) -> Dict[str, object]:
    return {
        "outer_local": _ring_to_local(atom.get("outer", []), frame),
        "type": atom.get("type", "convex"),
        "vertex_count": int(atom.get("vertex_count", len(atom.get("outer", [])))),
        "area": float(atom.get("area", 0.0)),
    }


def _sorted_faces(faces: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    return sorted(
        faces,
        key=lambda face: (
            int(face.get("label", -1)),
            -float(face.get("features", {}).get("area", 0.0)),
            int(face.get("id", 0)),
        ),
    )


def _code_length(nodes: Sequence[Dict[str, object]], relations: Sequence[Dict[str, object]], *, config: WeakExplainerConfig) -> float:
    face_count = sum(1 for node in nodes if node.get("role") == "semantic_face")
    atom_count = sum(1 for node in nodes if node.get("role") == "convex_atom")
    vertex_count = 0
    for node in nodes:
        if node.get("role") == "convex_atom":
            vertex_count += int(node.get("geometry", {}).get("vertex_count", 0))
        elif node.get("geometry_model") == "polygon_code":
            geometry = node.get("geometry", {})
            vertex_count += len(geometry.get("outer_local", []))
            vertex_count += sum(len(ring) for ring in geometry.get("holes_local", []))
    return float(
        config.cost_face * face_count
        + config.cost_atom * atom_count
        + config.cost_relation * len(relations)
        + config.cost_vertex * vertex_count
    )


def _validate(nodes: Sequence[Dict[str, object]], relations: Sequence[Dict[str, object]]) -> Dict[str, object]:
    node_ids = {str(node["id"]) for node in nodes}
    missing_refs: List[Dict[str, object]] = []
    for relation in relations:
        relation_type = str(relation.get("type"))
        refs: List[str] = []
        if relation_type == "label_group_contains":
            refs = [str(relation.get("parent")), str(relation.get("child"))]
        elif relation_type == "atom_part_of":
            refs = [str(relation.get("atom")), str(relation.get("face"))]
        elif relation_type == "face_adjacent":
            refs = [str(value) for value in relation.get("faces", [])]
        for ref in refs:
            if ref not in node_ids:
                missing_refs.append({"relation": relation, "missing_node": ref})
    semantic_face_count = sum(1 for node in nodes if node.get("role") == "semantic_face")
    atom_count = sum(1 for node in nodes if node.get("role") == "convex_atom")
    return {
        "is_valid": bool(not missing_refs and semantic_face_count > 0),
        "semantic_face_count": int(semantic_face_count),
        "convex_atom_count": int(atom_count),
        "missing_relation_refs": missing_refs,
    }


def build_weak_explanation_payload(
    evidence_payload: Dict[str, object],
    *,
    config: WeakExplainerConfig | None = None,
    source_tag: str | None = None,
) -> Dict[str, object]:
    config = config or WeakExplainerConfig()
    faces = _sorted_faces(evidence_payload.get("faces", []))
    nodes: List[Dict[str, object]] = []
    relations: List[Dict[str, object]] = []
    residuals: List[Dict[str, object]] = []
    selected_explanations: List[Dict[str, object]] = []

    face_node_by_source: Dict[int, str] = {}
    face_nodes_by_label: Dict[int, List[str]] = {}
    atom_count_by_face: Dict[int, int] = {}
    atom_node_count = 0

    label_group_by_label: Dict[int, str] = {}
    if config.include_label_groups:
        for group_index, label in enumerate(sorted({int(face.get("label", -1)) for face in faces})):
            group_id = f"label_group_{group_index}"
            label_group_by_label[label] = group_id
            source_face_ids = [int(face["id"]) for face in faces if int(face.get("label", -1)) == label]
            nodes.append(
                {
                    "id": group_id,
                    "role": "label_group",
                    "label": int(label),
                    "geometry_model": "none",
                    "children": [],
                    "count": int(len(source_face_ids)),
                    "evidence": {"face_ids": source_face_ids, "arc_ids": []},
                }
            )

    for face_index, face in enumerate(faces):
        source_face_id = int(face["id"])
        label = int(face.get("label", -1))
        polygon = _polygon_from_face(face)
        frame = _frame_from_polygon(polygon, eps=config.min_area_eps)
        geometry_model, geometry = _face_geometry_node(face, frame=frame, config=config)
        face_id = f"face_{face_index}"
        face_node_by_source[source_face_id] = face_id
        face_nodes_by_label.setdefault(label, []).append(face_id)
        atom_ids: List[str] = []

        face_node = {
            "id": face_id,
            "role": "semantic_face",
            "label": label,
            "source_face_id": source_face_id,
            "frame": frame,
            "geometry_model": geometry_model,
            "geometry": geometry,
            "atom_ids": atom_ids,
            "features": {
                "area": float(face.get("features", {}).get("area", 0.0)),
                "centroid": face.get("features", {}).get("centroid", [0.0, 0.0]),
                "degree": int(face.get("features", {}).get("degree", 0)),
                "hole_count": int(face.get("features", {}).get("hole_count", 0)),
            },
            "evidence": {"face_ids": [source_face_id], "arc_ids": _face_arc_ids(face)},
        }
        nodes.append(face_node)

        if config.include_label_groups and label in label_group_by_label:
            group_id = label_group_by_label[label]
            relations.append({"type": "label_group_contains", "parent": group_id, "child": face_id})
            for node in nodes:
                if node["id"] == group_id:
                    node["children"].append(face_id)
                    break

        atoms = face.get("convex_partition", {}).get("atoms", [])
        if config.include_convex_atom_nodes and atoms:
            for local_atom_index, atom in enumerate(atoms):
                atom_id = f"atom_{atom_node_count}"
                atom_node_count += 1
                atom_ids.append(atom_id)
                nodes.append(
                    {
                        "id": atom_id,
                        "role": "convex_atom",
                        "label": label,
                        "parent_face": face_id,
                        "source_face_id": source_face_id,
                        "source_atom_id": int(atom.get("id", local_atom_index)),
                        "frame": frame,
                        "geometry_model": "convex_polygon",
                        "geometry": _atom_node_geometry(atom, frame=frame),
                        "evidence": {"face_ids": [source_face_id], "arc_ids": _face_arc_ids(face), "atom_id": int(atom.get("id", local_atom_index))},
                    }
                )
                relations.append({"type": "atom_part_of", "atom": atom_id, "face": face_id})
        atom_count_by_face[source_face_id] = len(atom_ids)
        if config.include_convex_atom_nodes and not atom_ids:
            residuals.append(
                {
                    "face_node_id": face_id,
                    "face_ids": [source_face_id],
                    "area": float(face.get("features", {}).get("area", 0.0)),
                    "reason": "missing_convex_atoms",
                }
            )

        selected_explanations.append(
            {
                "patch_id": f"weak_face_patch_{face_index}",
                "evidence": {"face_ids": [source_face_id], "arc_ids": _face_arc_ids(face)},
                "selected_candidate_id": "weak_convex_face_atoms",
                "selected_template": "semantic_face_with_convex_atoms",
                "generated_node_ids": [face_id, *atom_ids],
                "cost": {
                    "total": None,
                    "template": 0.0,
                    "topology": 1 + len(atom_ids),
                    "geometry": len(atom_ids),
                    "residual": 0.0 if atom_ids else 1.0,
                    "invalid": 0.0,
                },
            }
        )

    for edge in evidence_payload.get("adjacency", []):
        source_faces = [int(value) for value in edge.get("faces", [])]
        if len(source_faces) != 2:
            continue
        if source_faces[0] not in face_node_by_source or source_faces[1] not in face_node_by_source:
            continue
        relations.append(
            {
                "type": "face_adjacent",
                "faces": [face_node_by_source[source_faces[0]], face_node_by_source[source_faces[1]]],
                "source_face_ids": source_faces,
                "labels": [int(value) for value in edge.get("labels", [])],
                "arc_ids": [int(value) for value in edge.get("arc_ids", [])],
                "shared_length": float(edge.get("shared_length", 0.0)),
                "arc_count": int(edge.get("arc_count", 0)),
            }
        )

    relation_order = {"label_group_contains": 0, "atom_part_of": 1, "face_adjacent": 2}
    relations.sort(key=lambda item: (relation_order.get(str(item.get("type")), 99), str(item)))

    validation = _validate(nodes, relations)
    code_length = _code_length(nodes, relations, config=config)
    generator_target = {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": evidence_payload.get("size", [0, 0]),
        "parse_graph": {
            "nodes": nodes,
            "relations": relations,
            "residuals": residuals,
        },
        "metadata": {
            "source_explanation": source_tag,
            "target_profile": "weak_convex_face_atoms_v1",
            "code_length": code_length,
            "render_iou": None,
            "valid": bool(validation["is_valid"]),
            "render_validation": {
                "status": "not_implemented",
                "reason": "weak parse_graph renderer is not implemented yet.",
            },
        },
    }
    role_histogram: Dict[str, int] = {}
    for node in nodes:
        role = str(node.get("role"))
        role_histogram[role] = int(role_histogram.get(role, 0)) + 1
    label_histogram = {str(label): len(face_ids) for label, face_ids in sorted(face_nodes_by_label.items())}
    diagnostics = {
        "profile": "weak_convex_face_atoms_v1",
        "face_count": int(len(faces)),
        "node_count": int(len(nodes)),
        "relation_count": int(len(relations)),
        "semantic_face_count": int(sum(1 for node in nodes if node.get("role") == "semantic_face")),
        "convex_atom_count": int(sum(1 for node in nodes if node.get("role") == "convex_atom")),
        "label_group_count": int(sum(1 for node in nodes if node.get("role") == "label_group")),
        "residual_face_count": int(len(residuals)),
        "total_code_length": code_length,
        "role_histogram": role_histogram,
        "label_histogram": label_histogram,
        "atom_count_by_source_face": {str(face_id): count for face_id, count in sorted(atom_count_by_face.items())},
        "selection_method": "deterministic_weak_face_convex_atom_packing",
        "global_optimal": False,
        "failure_reasons": [],
    }
    return {
        "format": "maskgen_explanation_v1",
        "explainer_profile": "weak_convex_face_atoms_v1",
        "source_evidence": source_tag,
        "selected_explanations": selected_explanations,
        "generator_target": generator_target,
        "diagnostics": diagnostics,
        "validation": validation,
        "config": asdict(config),
    }
