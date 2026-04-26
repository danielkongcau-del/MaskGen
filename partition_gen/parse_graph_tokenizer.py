from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


STATIC_TOKENS = [
    "<PAD>",
    "<BOS>",
    "<EOS>",
    "<UNK>",
    "WEAK_PARSE_GRAPH_V1",
    "SIZE",
    "LABEL_GROUPS",
    "LABEL",
    "COUNT",
    "FACES",
    "FACE",
    "FRAME",
    "ATOMS",
    "ATOM",
    "TYPE_TRIANGLE",
    "TYPE_QUAD",
    "TYPE_CONVEX",
    "AREA",
    "PTS",
    "ADJ",
    "EDGE",
    "LEN",
    "RESIDUALS",
    "END_ATOM",
    "END_FACE",
]


@dataclass(frozen=True)
class ParseGraphTokenizerConfig:
    coord_bins: int = 1024
    coord_min: float = -1.5
    coord_max: float = 1.5
    position_bins: int = 1024
    position_min: float = 0.0
    position_max: float = 256.0
    scale_bins: int = 1024
    scale_min: float = 0.0
    scale_max: float = 512.0
    angle_bins: int = 1024
    angle_min: float = -math.pi
    angle_max: float = math.pi
    area_bins: int = 2048
    area_min: float = 0.0
    area_max: float = 65536.0
    length_bins: int = 1024
    length_min: float = 0.0
    length_max: float = 512.0
    max_int: int = 4096


def quantize(value: float, *, low: float, high: float, bins: int) -> int:
    if bins <= 1:
        return 0
    if high <= low:
        return 0
    normalized = (float(value) - low) / (high - low)
    normalized = min(1.0, max(0.0, normalized))
    return int(round(normalized * (bins - 1)))


def dequantize(index: int, *, low: float, high: float, bins: int) -> float:
    if bins <= 1:
        return float(low)
    index = max(0, min(int(index), bins - 1))
    return float(low + (high - low) * index / (bins - 1))


def int_token(value: int, *, config: ParseGraphTokenizerConfig) -> str:
    value = max(0, min(int(value), int(config.max_int)))
    return f"I_{value}"


def q_token(value: float, *, low: float, high: float, bins: int) -> str:
    return f"Q_{quantize(value, low=low, high=high, bins=bins)}"


def token_int(token: str) -> int:
    if not token.startswith("I_"):
        raise ValueError(f"Expected int token, got {token}")
    return int(token[2:])


def token_q(token: str) -> int:
    if not token.startswith("Q_"):
        raise ValueError(f"Expected quantized token, got {token}")
    return int(token[2:])


def build_token_vocabulary(config: ParseGraphTokenizerConfig | None = None) -> Dict[str, int]:
    config = config or ParseGraphTokenizerConfig()
    tokens = list(STATIC_TOKENS)
    tokens.extend(f"I_{index}" for index in range(int(config.max_int) + 1))
    max_q = max(
        int(config.coord_bins),
        int(config.position_bins),
        int(config.scale_bins),
        int(config.angle_bins),
        int(config.area_bins),
        int(config.length_bins),
    )
    tokens.extend(f"Q_{index}" for index in range(max_q))
    return {token: index for index, token in enumerate(tokens)}


def tokens_to_ids(tokens: Sequence[str], vocab: Dict[str, int]) -> List[int]:
    unk = int(vocab["<UNK>"])
    return [int(vocab.get(token, unk)) for token in tokens]


def ids_to_tokens(ids: Sequence[int], vocab: Dict[str, int]) -> List[str]:
    inverse = {index: token for token, index in vocab.items()}
    return [inverse.get(int(index), "<UNK>") for index in ids]


def save_vocabulary(path: Path, vocab: Dict[str, int], *, config: ParseGraphTokenizerConfig) -> None:
    payload = {
        "format": "maskgen_parse_graph_vocab_v1",
        "config": asdict(config),
        "token_to_id": vocab,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_vocabulary(path: Path) -> Dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(token): int(index) for token, index in payload["token_to_id"].items()}


def _node_index(node_id: str, prefix: str) -> int:
    expected = f"{prefix}_"
    if not str(node_id).startswith(expected):
        raise ValueError(f"Expected {prefix} id, got {node_id}")
    return int(str(node_id)[len(expected) :])


def _sort_label_groups(nodes: Sequence[dict]) -> List[dict]:
    return sorted(
        [node for node in nodes if node.get("role") == "label_group"],
        key=lambda node: _node_index(str(node["id"]), "label_group"),
    )


def _sort_faces(nodes: Sequence[dict]) -> List[dict]:
    return sorted(
        [node for node in nodes if node.get("role") == "semantic_face"],
        key=lambda node: _node_index(str(node["id"]), "face"),
    )


def _sort_atoms_for_face(nodes_by_id: Dict[str, dict], atom_ids: Sequence[str]) -> List[dict]:
    return [
        nodes_by_id[str(atom_id)]
        for atom_id in sorted(atom_ids, key=lambda atom_id: _node_index(str(atom_id), "atom"))
        if str(atom_id) in nodes_by_id
    ]


def _adjacency_relations(relations: Sequence[dict]) -> List[dict]:
    output = [relation for relation in relations if relation.get("type") == "face_adjacent"]
    return sorted(
        output,
        key=lambda relation: (
            _node_index(str(relation["faces"][0]), "face"),
            _node_index(str(relation["faces"][1]), "face"),
        ),
    )


def _atom_type_token(atom: dict) -> str:
    atom_type = str(atom.get("geometry", {}).get("type", "convex"))
    if atom_type == "triangle":
        return "TYPE_TRIANGLE"
    if atom_type == "quad":
        return "TYPE_QUAD"
    return "TYPE_CONVEX"


def _encode_frame(tokens: List[str], frame: dict, *, config: ParseGraphTokenizerConfig) -> None:
    origin = frame.get("origin", [0.0, 0.0])
    tokens.extend(
        [
            "FRAME",
            q_token(float(origin[0]), low=config.position_min, high=config.position_max, bins=config.position_bins),
            q_token(float(origin[1]), low=config.position_min, high=config.position_max, bins=config.position_bins),
            q_token(float(frame.get("scale", 1.0)), low=config.scale_min, high=config.scale_max, bins=config.scale_bins),
            q_token(float(frame.get("orientation", 0.0)), low=config.angle_min, high=config.angle_max, bins=config.angle_bins),
        ]
    )


def encode_generator_target(
    target: dict,
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> List[str]:
    config = config or ParseGraphTokenizerConfig()
    graph = target.get("parse_graph", {})
    nodes = list(graph.get("nodes", []))
    relations = list(graph.get("relations", []))
    nodes_by_id = {str(node["id"]): node for node in nodes}
    size = target.get("size", [0, 0])

    tokens: List[str] = ["<BOS>", "WEAK_PARSE_GRAPH_V1", "SIZE"]
    tokens.extend(int_token(int(value), config=config) for value in size[:2])

    label_groups = _sort_label_groups(nodes)
    tokens.extend(["LABEL_GROUPS", int_token(len(label_groups), config=config)])
    for group in label_groups:
        tokens.extend(
            [
                "LABEL",
                int_token(int(group.get("label", 0)), config=config),
                "COUNT",
                int_token(int(group.get("count", len(group.get("children", [])))), config=config),
            ]
        )

    faces = _sort_faces(nodes)
    tokens.extend(["FACES", int_token(len(faces), config=config)])
    for face in faces:
        tokens.extend(["FACE", "LABEL", int_token(int(face.get("label", 0)), config=config)])
        _encode_frame(tokens, face.get("frame", {}), config=config)

        atom_ids = face.get("atom_ids") or face.get("geometry", {}).get("atom_ids") or []
        atoms = _sort_atoms_for_face(nodes_by_id, atom_ids)
        tokens.extend(["ATOMS", int_token(len(atoms), config=config)])
        for atom in atoms:
            geometry = atom.get("geometry", {})
            outer = geometry.get("outer_local", [])
            tokens.extend(
                [
                    "ATOM",
                    _atom_type_token(atom),
                    "AREA",
                    q_token(float(geometry.get("area", 0.0)), low=config.area_min, high=config.area_max, bins=config.area_bins),
                    "PTS",
                    int_token(len(outer), config=config),
                ]
            )
            for point in outer:
                tokens.extend(
                    [
                        q_token(float(point[0]), low=config.coord_min, high=config.coord_max, bins=config.coord_bins),
                        q_token(float(point[1]), low=config.coord_min, high=config.coord_max, bins=config.coord_bins),
                    ]
                )
            tokens.append("END_ATOM")
        tokens.append("END_FACE")

    adjacency = _adjacency_relations(relations)
    tokens.extend(["ADJ", int_token(len(adjacency), config=config)])
    for relation in adjacency:
        left, right = relation.get("faces", [None, None])
        tokens.extend(
            [
                "EDGE",
                int_token(_node_index(str(left), "face"), config=config),
                int_token(_node_index(str(right), "face"), config=config),
                "LEN",
                q_token(float(relation.get("shared_length", 0.0)), low=config.length_min, high=config.length_max, bins=config.length_bins),
            ]
        )

    residuals = graph.get("residuals", [])
    tokens.extend(["RESIDUALS", int_token(len(residuals), config=config), "<EOS>"])
    return tokens


class TokenReader:
    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens = list(tokens)
        self.index = 0

    def expect(self, token: str) -> None:
        actual = self.next()
        if actual != token:
            raise ValueError(f"Expected token {token}, got {actual}")

    def next(self) -> str:
        if self.index >= len(self.tokens):
            raise ValueError("Unexpected end of token sequence")
        token = self.tokens[self.index]
        self.index += 1
        return token

    def next_int(self) -> int:
        return token_int(self.next())

    def next_q(self) -> int:
        return token_q(self.next())


def _decode_frame(reader: TokenReader, *, config: ParseGraphTokenizerConfig) -> dict:
    reader.expect("FRAME")
    origin_x = dequantize(reader.next_q(), low=config.position_min, high=config.position_max, bins=config.position_bins)
    origin_y = dequantize(reader.next_q(), low=config.position_min, high=config.position_max, bins=config.position_bins)
    scale = dequantize(reader.next_q(), low=config.scale_min, high=config.scale_max, bins=config.scale_bins)
    orientation = dequantize(reader.next_q(), low=config.angle_min, high=config.angle_max, bins=config.angle_bins)
    return {"origin": [origin_x, origin_y], "scale": scale, "orientation": orientation}


def decode_tokens_to_generator_target(
    tokens: Sequence[str],
    *,
    config: ParseGraphTokenizerConfig | None = None,
) -> dict:
    config = config or ParseGraphTokenizerConfig()
    reader = TokenReader(tokens)
    reader.expect("<BOS>")
    reader.expect("WEAK_PARSE_GRAPH_V1")
    reader.expect("SIZE")
    size = [reader.next_int(), reader.next_int()]

    nodes: List[dict] = []
    relations: List[dict] = []

    reader.expect("LABEL_GROUPS")
    label_group_count = reader.next_int()
    label_groups: List[dict] = []
    for group_index in range(label_group_count):
        reader.expect("LABEL")
        label = reader.next_int()
        reader.expect("COUNT")
        count = reader.next_int()
        group = {
            "id": f"label_group_{group_index}",
            "role": "label_group",
            "label": label,
            "geometry_model": "none",
            "children": [],
            "count": count,
        }
        label_groups.append(group)
        nodes.append(group)

    reader.expect("FACES")
    face_count = reader.next_int()
    group_by_label = {int(group["label"]): group for group in label_groups}
    atom_global_index = 0
    for face_index in range(face_count):
        reader.expect("FACE")
        reader.expect("LABEL")
        label = reader.next_int()
        frame = _decode_frame(reader, config=config)
        face_id = f"face_{face_index}"
        reader.expect("ATOMS")
        atom_count = reader.next_int()
        atom_ids: List[str] = []
        face_node = {
            "id": face_id,
            "role": "semantic_face",
            "label": label,
            "frame": frame,
            "geometry_model": "convex_atom_union",
            "geometry": {"atom_ids": atom_ids},
            "atom_ids": atom_ids,
        }
        nodes.append(face_node)
        if label in group_by_label:
            group = group_by_label[label]
            group["children"].append(face_id)
            relations.append({"type": "label_group_contains", "parent": group["id"], "child": face_id})

        for _ in range(atom_count):
            reader.expect("ATOM")
            type_token = reader.next()
            atom_type = {"TYPE_TRIANGLE": "triangle", "TYPE_QUAD": "quad"}.get(type_token, "convex")
            reader.expect("AREA")
            area = dequantize(reader.next_q(), low=config.area_min, high=config.area_max, bins=config.area_bins)
            reader.expect("PTS")
            point_count = reader.next_int()
            outer = []
            for _point_index in range(point_count):
                x = dequantize(reader.next_q(), low=config.coord_min, high=config.coord_max, bins=config.coord_bins)
                y = dequantize(reader.next_q(), low=config.coord_min, high=config.coord_max, bins=config.coord_bins)
                outer.append([x, y])
            reader.expect("END_ATOM")
            atom_id = f"atom_{atom_global_index}"
            atom_global_index += 1
            atom_ids.append(atom_id)
            nodes.append(
                {
                    "id": atom_id,
                    "role": "convex_atom",
                    "label": label,
                    "parent_face": face_id,
                    "frame": frame,
                    "geometry_model": "convex_polygon",
                    "geometry": {
                        "outer_local": outer,
                        "type": atom_type,
                        "vertex_count": len(outer),
                        "area": area,
                    },
                }
            )
            relations.append({"type": "atom_part_of", "atom": atom_id, "face": face_id})
        reader.expect("END_FACE")

    reader.expect("ADJ")
    edge_count = reader.next_int()
    for _edge_index in range(edge_count):
        reader.expect("EDGE")
        left = reader.next_int()
        right = reader.next_int()
        reader.expect("LEN")
        shared_length = dequantize(reader.next_q(), low=config.length_min, high=config.length_max, bins=config.length_bins)
        relations.append(
            {
                "type": "face_adjacent",
                "faces": [f"face_{left}", f"face_{right}"],
                "shared_length": shared_length,
            }
        )

    reader.expect("RESIDUALS")
    residual_count = reader.next_int()
    residuals = [{"reason": "tokenized_residual_placeholder"} for _ in range(residual_count)]
    reader.expect("<EOS>")

    return {
        "format": "maskgen_generator_target_v1",
        "target_type": "parse_graph",
        "size": size,
        "parse_graph": {
            "nodes": nodes,
            "relations": relations,
            "residuals": residuals,
        },
        "metadata": {
            "decoded_from_tokens": True,
            "tokenizer": "weak_parse_graph_structured_v1",
        },
    }


def load_generator_target(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_training_manifest_rows(manifest_path: Path, *, include_non_usable: bool = False) -> Iterable[dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if include_non_usable or bool(row.get("training_usable", False)):
                yield row
