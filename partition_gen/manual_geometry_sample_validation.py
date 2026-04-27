from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from partition_gen.parse_graph_compact_tokenizer import encode_geometry_target
from partition_gen.parse_graph_tokenizer import (
    ParseGraphTokenizerConfig,
    TokenReader,
    dequantize,
    int_token,
    token_int,
)


ROLE_TOKEN_TO_MANUAL_ROLE = {
    "ROLE_SUPPORT": "support_region",
    "ROLE_DIVIDER": "divider_region",
    "ROLE_INSERT": "insert_object",
    "ROLE_INSERT_GROUP": "insert_object_group",
    "ROLE_RESIDUAL": "residual_region",
    "ROLE_UNKNOWN": "unknown",
}

MANUAL_ROLE_TO_TOKEN = {value: key for key, value in ROLE_TOKEN_TO_MANUAL_ROLE.items()}

GEOMETRY_TOKEN_TO_MODEL = {
    "GEOM_NONE": "none",
    "GEOM_POLYGON_CODE": "polygon_code",
    "GEOM_CONVEX_ATOMS": "convex_atoms",
    "GEOM_UNKNOWN": "unknown",
}

GEOMETRY_MODEL_TO_TOKEN = {value: key for key, value in GEOMETRY_TOKEN_TO_MODEL.items()}

ATOM_TYPE_TOKEN_TO_TYPE = {
    "TYPE_TRIANGLE": "triangle",
    "TYPE_QUAD": "quad",
    "TYPE_CONVEX": "convex",
}


def _role_token(role: str) -> str:
    role = str(role)
    if role.startswith("ROLE_"):
        return role
    return MANUAL_ROLE_TO_TOKEN.get(role, "ROLE_UNKNOWN")


def _geometry_token(geometry_model: str) -> str:
    geometry_model = str(geometry_model)
    if geometry_model.startswith("GEOM_"):
        return geometry_model
    return GEOMETRY_MODEL_TO_TOKEN.get(geometry_model, "GEOM_UNKNOWN")


def geometry_prefix_tokens(
    *,
    role: str,
    label: int,
    geometry_model: str = "polygon_code",
    include_bos: bool = True,
) -> List[str]:
    tokens = [
        "MANUAL_GEOMETRY_V1",
        "GEOMETRY_BLOCK",
        _role_token(role),
        "LABEL",
        int_token(int(label), config=ParseGraphTokenizerConfig()),
        _geometry_token(geometry_model),
    ]
    return ["<BOS>", *tokens] if include_bos else tokens


def geometry_prefix_from_tokens(tokens: Sequence[str]) -> List[str]:
    tokens = [str(token) for token in tokens]
    if len(tokens) < 7:
        raise ValueError("Geometry token sequence is too short for a prefix")
    expected = ["<BOS>", "MANUAL_GEOMETRY_V1", "GEOMETRY_BLOCK"]
    if tokens[:3] != expected:
        raise ValueError(f"Expected geometry prefix {expected}, got {tokens[:3]}")
    if tokens[4] != "LABEL":
        raise ValueError(f"Expected LABEL at prefix index 4, got {tokens[4]}")
    token_int(tokens[5])
    if tokens[3] not in ROLE_TOKEN_TO_MANUAL_ROLE:
        raise ValueError(f"Unknown role token in geometry prefix: {tokens[3]}")
    if tokens[6] not in GEOMETRY_TOKEN_TO_MODEL:
        raise ValueError(f"Unknown geometry token in geometry prefix: {tokens[6]}")
    return tokens[:7]


def _decode_frame(reader: TokenReader, *, config: ParseGraphTokenizerConfig) -> dict:
    reader.expect("FRAME")
    origin_x = dequantize(reader.next_q(), low=config.position_min, high=config.position_max, bins=config.position_bins)
    origin_y = dequantize(reader.next_q(), low=config.position_min, high=config.position_max, bins=config.position_bins)
    scale = dequantize(reader.next_q(), low=config.scale_min, high=config.scale_max, bins=config.scale_bins)
    orientation = dequantize(reader.next_q(), low=config.angle_min, high=config.angle_max, bins=config.angle_bins)
    return {"origin": [origin_x, origin_y], "scale": scale, "orientation": orientation}


def _decode_points(reader: TokenReader, *, config: ParseGraphTokenizerConfig) -> List[List[float]]:
    reader.expect("PTS")
    point_count = reader.next_int()
    points: List[List[float]] = []
    for _ in range(int(point_count)):
        x = dequantize(reader.next_q(), low=config.coord_min, high=config.coord_max, bins=config.coord_bins)
        y = dequantize(reader.next_q(), low=config.coord_min, high=config.coord_max, bins=config.coord_bins)
        points.append([x, y])
    return points


def _decode_polygon_geometry(reader: TokenReader, *, config: ParseGraphTokenizerConfig) -> dict:
    reader.expect("POLYS")
    polygon_count = reader.next_int()
    polygons: List[dict] = []
    for _ in range(int(polygon_count)):
        reader.expect("POLY")
        outer = _decode_points(reader, config=config)
        reader.expect("HOLES")
        hole_count = reader.next_int()
        holes: List[List[List[float]]] = []
        for _hole_index in range(int(hole_count)):
            reader.expect("HOLE")
            holes.append(_decode_points(reader, config=config))
            reader.expect("END_HOLE")
        reader.expect("END_POLY")
        polygons.append({"outer_local": outer, "holes_local": holes})
    first = polygons[0] if polygons else {"outer_local": [], "holes_local": []}
    return {
        "outer_local": list(first.get("outer_local", [])),
        "holes_local": list(first.get("holes_local", [])),
        "polygons_local": polygons,
    }


def _decode_convex_atoms(reader: TokenReader, *, config: ParseGraphTokenizerConfig) -> List[dict]:
    reader.expect("ATOMS")
    atom_count = reader.next_int()
    atoms: List[dict] = []
    for _ in range(int(atom_count)):
        reader.expect("ATOM")
        atom_type = ATOM_TYPE_TOKEN_TO_TYPE.get(reader.next(), "convex")
        reader.expect("AREA")
        area = dequantize(reader.next_q(), low=config.area_min, high=config.area_max, bins=config.area_bins)
        outer = _decode_points(reader, config=config)
        reader.expect("END_ATOM")
        atoms.append(
            {
                "type": atom_type,
                "area": area,
                "outer_local": outer,
                "vertex_count": int(len(outer)),
            }
        )
    return atoms


def decode_geometry_tokens_to_target(
    tokens: Sequence[str],
    *,
    config: ParseGraphTokenizerConfig | None = None,
    source_node_id: str | None = None,
) -> dict:
    config = config or ParseGraphTokenizerConfig()
    reader = TokenReader([str(token) for token in tokens])
    reader.expect("<BOS>")
    reader.expect("MANUAL_GEOMETRY_V1")
    reader.expect("GEOMETRY_BLOCK")
    role_token = reader.next()
    role = ROLE_TOKEN_TO_MANUAL_ROLE.get(role_token, "unknown")
    reader.expect("LABEL")
    label = reader.next_int()
    geometry_token = reader.next()
    geometry_model = GEOMETRY_TOKEN_TO_MODEL.get(geometry_token, "unknown")
    target = {
        "format": "maskgen_generator_target_v1",
        "target_type": "manual_parse_graph_geometry_v1",
        "source_node_id": source_node_id or "generated_geometry_0",
        "role": role,
        "label": int(label),
        "geometry_model": geometry_model,
        "metadata": {"decoded_from_tokens": True, "tokenizer": "manual_geometry_v1"},
    }
    if geometry_model == "polygon_code":
        target["frame"] = _decode_frame(reader, config=config)
        target["geometry"] = _decode_polygon_geometry(reader, config=config)
    elif geometry_model == "convex_atoms":
        target["frame"] = _decode_frame(reader, config=config)
        target["atoms"] = _decode_convex_atoms(reader, config=config)
    reader.expect("<EOS>")
    if reader.index != len(reader.tokens):
        raise ValueError(f"Trailing tokens after EOS: {len(reader.tokens) - reader.index}")
    return target


def parse_geometry_structure(tokens: Sequence[str], *, config: ParseGraphTokenizerConfig | None = None) -> Dict[str, object]:
    target = decode_geometry_tokens_to_target(tokens, config=config)
    geometry_model = str(target.get("geometry_model", "none"))
    polygon_count = 0
    hole_count = 0
    point_counts: List[int] = []
    atom_count = 0
    if geometry_model == "polygon_code":
        polygons = list(target.get("geometry", {}).get("polygons_local", []) or [])
        polygon_count = len(polygons)
        for polygon in polygons:
            point_counts.append(len(polygon.get("outer_local", []) or []))
            holes = polygon.get("holes_local", []) or []
            hole_count += len(holes)
            point_counts.extend(len(hole) for hole in holes)
    elif geometry_model == "convex_atoms":
        atoms = list(target.get("atoms", []) or [])
        atom_count = len(atoms)
        point_counts.extend(len(atom.get("outer_local", []) or []) for atom in atoms)
    return {
        "length": int(len(tokens)),
        "role": str(target.get("role", "")),
        "label": int(target.get("label", 0)),
        "geometry_model": geometry_model,
        "polygon_count": int(polygon_count),
        "hole_count": int(hole_count),
        "atom_count": int(atom_count),
        "point_counts": [int(value) for value in point_counts],
        "point_total": int(sum(point_counts)),
    }


def _first_mismatch(left: Sequence[str], right: Sequence[str]) -> str:
    limit = min(len(left), len(right))
    for index in range(limit):
        if str(left[index]) != str(right[index]):
            return f"token_mismatch_at_{index}_{left[index]}_vs_{right[index]}"
    if len(left) != len(right):
        return f"length_mismatch_{len(left)}_vs_{len(right)}"
    return "unknown_mismatch"


def validate_geometry_tokens(
    tokens: Sequence[str],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> Dict[str, object]:
    config = config or ParseGraphTokenizerConfig()
    tokens = [str(token) for token in tokens]
    errors: List[str] = []
    if not tokens:
        errors.append("empty_tokens")
    if "<UNK>" in tokens:
        errors.append("contains_UNK")
    target = None
    structure = None
    try:
        target = decode_geometry_tokens_to_target(tokens, config=config)
        reencoded = encode_geometry_target(target, config=config)
        if reencoded != tokens:
            errors.append(_first_mismatch(tokens, reencoded))
        structure = parse_geometry_structure(tokens, config=config)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")

    role_histogram = Counter()
    label_histogram = Counter()
    geometry_model_histogram = Counter()
    if structure is not None:
        role_histogram[str(structure["role"])] += 1
        label_histogram[str(structure["label"])] += 1
        geometry_model_histogram[str(structure["geometry_model"])] += 1
    return {
        "format": "maskgen_manual_geometry_validation_v1",
        "valid": not errors,
        "errors": errors,
        "length": int(len(tokens)),
        "hit_eos": bool(tokens and tokens[-1] == "<EOS>"),
        "target": target,
        "structure": structure,
        "role_histogram": dict(sorted(role_histogram.items())),
        "label_histogram": dict(sorted(label_histogram.items(), key=lambda item: int(item[0]))),
        "geometry_model_histogram": dict(sorted(geometry_model_histogram.items())),
    }
