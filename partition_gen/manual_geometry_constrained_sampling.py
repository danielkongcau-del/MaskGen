from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from partition_gen.manual_geometry_sample_validation import geometry_prefix_tokens
from partition_gen.parse_graph_tokenizer import ParseGraphTokenizerConfig, token_int


ROLE_TOKENS = ("ROLE_SUPPORT", "ROLE_DIVIDER", "ROLE_INSERT", "ROLE_RESIDUAL")
GEOMETRY_TOKENS = ("GEOM_POLYGON_CODE", "GEOM_CONVEX_ATOMS")


@dataclass(frozen=True)
class GeometryConstrainedSamplerConfig:
    tokenizer_config: ParseGraphTokenizerConfig = field(default_factory=ParseGraphTokenizerConfig)
    max_label: int = 6
    min_polygons: int = 1
    max_polygons: int = 8
    min_points_per_ring: int = 3
    max_points_per_ring: int = 128
    max_holes_per_polygon: int = 8
    min_atoms: int = 1
    max_atoms: int = 128
    allowed_roles: tuple[str, ...] = ROLE_TOKENS
    allowed_geometry_models: tuple[str, ...] = GEOMETRY_TOKENS


@dataclass
class GeometryGrammarState:
    config: GeometryConstrainedSamplerConfig = field(default_factory=GeometryConstrainedSamplerConfig)
    phase: str = "manual"
    geometry_model: str | None = None
    polygon_count: int = 0
    polygon_index: int = 0
    point_count: int = 0
    point_index: int = 0
    hole_count: int = 0
    hole_index: int = 0
    atom_count: int = 0
    atom_index: int = 0
    done: bool = False
    errors: List[str] = field(default_factory=list)

    def allowed_token_strings(self) -> List[str]:
        if self.done:
            return []
        cfg = self.config
        tok_cfg = cfg.tokenizer_config
        if self.phase == "manual":
            return ["MANUAL_GEOMETRY_V1"]
        if self.phase == "geometry_block":
            return ["GEOMETRY_BLOCK"]
        if self.phase == "role":
            return [token for token in cfg.allowed_roles if token in ROLE_TOKENS]
        if self.phase == "label_token":
            return ["LABEL"]
        if self.phase == "label":
            return self._int_tokens(0, int(cfg.max_label))
        if self.phase == "geometry_model":
            return [token for token in cfg.allowed_geometry_models if token in GEOMETRY_TOKENS or token == "GEOM_NONE"]
        if self.phase == "frame_token":
            return ["FRAME"]
        if self.phase in {"frame_origin_x", "frame_origin_y"}:
            return self._q_tokens(int(tok_cfg.position_bins))
        if self.phase == "frame_scale":
            return self._q_tokens(int(tok_cfg.scale_bins))
        if self.phase == "frame_orientation":
            return self._q_tokens(int(tok_cfg.angle_bins))
        if self.phase == "polys_token":
            return ["POLYS"]
        if self.phase == "poly_count":
            return self._int_tokens(int(cfg.min_polygons), int(cfg.max_polygons))
        if self.phase == "poly_start":
            return ["POLY"]
        if self.phase in {"pts_token", "hole_pts_token", "atom_pts_token"}:
            return ["PTS"]
        if self.phase in {"point_count", "hole_point_count", "atom_point_count"}:
            return self._int_tokens(int(cfg.min_points_per_ring), int(cfg.max_points_per_ring))
        if self.phase in {"point_x", "point_y", "hole_point_x", "hole_point_y", "atom_point_x", "atom_point_y"}:
            return self._q_tokens(int(tok_cfg.coord_bins))
        if self.phase == "holes_token":
            return ["HOLES"]
        if self.phase == "hole_count":
            return self._int_tokens(0, int(cfg.max_holes_per_polygon))
        if self.phase == "hole_start":
            return ["HOLE"]
        if self.phase == "end_hole":
            return ["END_HOLE"]
        if self.phase == "end_poly":
            return ["END_POLY"]
        if self.phase == "atoms_token":
            return ["ATOMS"]
        if self.phase == "atom_count":
            return self._int_tokens(int(cfg.min_atoms), int(cfg.max_atoms))
        if self.phase == "atom_start":
            return ["ATOM"]
        if self.phase == "atom_type":
            return ["TYPE_TRIANGLE", "TYPE_QUAD", "TYPE_CONVEX"]
        if self.phase == "area_token":
            return ["AREA"]
        if self.phase == "area_value":
            return self._q_tokens(int(tok_cfg.area_bins))
        if self.phase == "end_atom":
            return ["END_ATOM"]
        if self.phase == "eos":
            return ["<EOS>"]
        return []

    def step(self, token: str) -> bool:
        token = str(token)
        allowed = set(self.allowed_token_strings())
        if token not in allowed:
            self.errors.append(f"illegal_{token}_in_phase_{self.phase}")
            self.done = True
            return False

        if self.phase == "manual":
            self.phase = "geometry_block"
        elif self.phase == "geometry_block":
            self.phase = "role"
        elif self.phase == "role":
            self.phase = "label_token"
        elif self.phase == "label_token":
            self.phase = "label"
        elif self.phase == "label":
            self.phase = "geometry_model"
        elif self.phase == "geometry_model":
            self.geometry_model = token
            self.phase = "eos" if token == "GEOM_NONE" else "frame_token"
        elif self.phase == "frame_token":
            self.phase = "frame_origin_x"
        elif self.phase == "frame_origin_x":
            self.phase = "frame_origin_y"
        elif self.phase == "frame_origin_y":
            self.phase = "frame_scale"
        elif self.phase == "frame_scale":
            self.phase = "frame_orientation"
        elif self.phase == "frame_orientation":
            self.phase = "polys_token" if self.geometry_model == "GEOM_POLYGON_CODE" else "atoms_token"
        elif self.phase == "polys_token":
            self.phase = "poly_count"
        elif self.phase == "poly_count":
            self.polygon_count = int(token_int(token))
            self.polygon_index = 0
            self.phase = "poly_start"
        elif self.phase == "poly_start":
            self.phase = "pts_token"
        elif self.phase == "pts_token":
            self.phase = "point_count"
        elif self.phase == "point_count":
            self.point_count = int(token_int(token))
            self.point_index = 0
            self.phase = "point_x"
        elif self.phase == "point_x":
            self.phase = "point_y"
        elif self.phase == "point_y":
            self.point_index += 1
            self.phase = "holes_token" if self.point_index >= self.point_count else "point_x"
        elif self.phase == "holes_token":
            self.phase = "hole_count"
        elif self.phase == "hole_count":
            self.hole_count = int(token_int(token))
            self.hole_index = 0
            self.phase = "hole_start" if self.hole_count > 0 else "end_poly"
        elif self.phase == "hole_start":
            self.phase = "hole_pts_token"
        elif self.phase == "hole_pts_token":
            self.phase = "hole_point_count"
        elif self.phase == "hole_point_count":
            self.point_count = int(token_int(token))
            self.point_index = 0
            self.phase = "hole_point_x"
        elif self.phase == "hole_point_x":
            self.phase = "hole_point_y"
        elif self.phase == "hole_point_y":
            self.point_index += 1
            self.phase = "end_hole" if self.point_index >= self.point_count else "hole_point_x"
        elif self.phase == "end_hole":
            self.hole_index += 1
            self.phase = "end_poly" if self.hole_index >= self.hole_count else "hole_start"
        elif self.phase == "end_poly":
            self.polygon_index += 1
            self.phase = "eos" if self.polygon_index >= self.polygon_count else "poly_start"
        elif self.phase == "atoms_token":
            self.phase = "atom_count"
        elif self.phase == "atom_count":
            self.atom_count = int(token_int(token))
            self.atom_index = 0
            self.phase = "atom_start"
        elif self.phase == "atom_start":
            self.phase = "atom_type"
        elif self.phase == "atom_type":
            self.phase = "area_token"
        elif self.phase == "area_token":
            self.phase = "area_value"
        elif self.phase == "area_value":
            self.phase = "atom_pts_token"
        elif self.phase == "atom_pts_token":
            self.phase = "atom_point_count"
        elif self.phase == "atom_point_count":
            self.point_count = int(token_int(token))
            self.point_index = 0
            self.phase = "atom_point_x"
        elif self.phase == "atom_point_x":
            self.phase = "atom_point_y"
        elif self.phase == "atom_point_y":
            self.point_index += 1
            self.phase = "end_atom" if self.point_index >= self.point_count else "atom_point_x"
        elif self.phase == "end_atom":
            self.atom_index += 1
            self.phase = "eos" if self.atom_index >= self.atom_count else "atom_start"
        elif self.phase == "eos":
            self.done = True
        return True

    def diagnostics(self) -> Dict[str, object]:
        return {
            "config": asdict(self.config),
            "phase": self.phase,
            "done": bool(self.done),
            "errors": list(self.errors),
            "geometry_model": self.geometry_model,
            "polygon_count": int(self.polygon_count),
            "atom_count": int(self.atom_count),
        }

    @staticmethod
    def _int_tokens(low: int, high: int) -> List[str]:
        if high < low:
            return []
        return [f"I_{index}" for index in range(int(low), int(high) + 1)]

    @staticmethod
    def _q_tokens(count: int) -> List[str]:
        return [f"Q_{index}" for index in range(max(0, int(count)))]


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    allowed_ids: Sequence[int],
    temperature: float,
    top_k: int | None,
) -> int:
    if not allowed_ids:
        raise ValueError("allowed_ids must not be empty")
    allowed = torch.tensor(list(allowed_ids), dtype=torch.long, device=logits.device)
    masked = torch.full_like(logits, float("-inf"))
    masked.index_copy_(0, allowed, logits.index_select(0, allowed))
    if temperature <= 0.0:
        return int(torch.argmax(masked).item())
    masked = masked / float(temperature)
    if top_k is not None and int(top_k) > 0:
        finite_values = masked.index_select(0, allowed)
        k = min(int(top_k), int(finite_values.numel()))
        threshold = torch.topk(finite_values, k).values[-1]
        masked = masked.masked_fill(masked < threshold, float("-inf"))
    probs = F.softmax(masked, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def sample_geometry_constrained(
    model,
    vocab: Dict[str, int],
    *,
    prefix_tokens: Sequence[str] | None = None,
    role: str = "support_region",
    label: int = 0,
    geometry_model: str = "polygon_code",
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int | None = None,
    constraint_config: GeometryConstrainedSamplerConfig | None = None,
    device: torch.device | str | None = None,
    use_cache: bool = True,
) -> Dict[str, object]:
    constraint_config = constraint_config or GeometryConstrainedSamplerConfig()
    inverse_vocab = {int(index): str(token) for token, index in vocab.items()}
    if "<BOS>" not in vocab or "<EOS>" not in vocab:
        raise ValueError("vocab must contain <BOS> and <EOS>")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    prefix = list(prefix_tokens or geometry_prefix_tokens(role=role, label=int(label), geometry_model=geometry_model))
    if not prefix or prefix[0] != "<BOS>":
        raise ValueError("Geometry prefix must start with <BOS>")
    missing = [token for token in prefix if token not in vocab]
    if missing:
        raise ValueError(f"Geometry prefix contains tokens not in vocab: {missing}")

    state = GeometryGrammarState(config=constraint_config)
    ids = [int(vocab["<BOS>"])]
    tokens = ["<BOS>"]
    for token in prefix[1:]:
        ids.append(int(vocab[str(token)]))
        tokens.append(str(token))
        if not state.step(str(token)):
            return {
                "ids": ids,
                "tokens": tokens,
                "length": int(len(ids)),
                "hit_eos": False,
                "stopped_reason": "invalid_prefix",
                "constraint_diagnostics": state.diagnostics(),
            }

    stopped_reason = "max_new_tokens"
    block_size = int(getattr(model.config, "block_size", max_new_tokens + len(ids)))
    use_kv_cache = bool(
        use_cache and getattr(model, "supports_kv_cache", False) and int(max_new_tokens) + len(ids) <= int(block_size)
    )
    past_kv = None

    for _step in range(int(max_new_tokens)):
        allowed_tokens = state.allowed_token_strings()
        allowed_ids = [int(vocab[token]) for token in allowed_tokens if token in vocab]
        if not allowed_ids:
            state.errors.append(f"empty_allowed_set_phase_{state.phase}")
            stopped_reason = "empty_allowed_set"
            break
        if use_kv_cache:
            if past_kv is None:
                input_ids = torch.tensor([ids[-int(block_size) :]], dtype=torch.long, device=device)
                outputs = model(input_ids, use_cache=True)
            else:
                input_ids = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
                outputs = model(input_ids, past_kv=past_kv, use_cache=True)
            past_kv = outputs["past_kv"]
        else:
            input_ids = torch.tensor([ids[-int(block_size) :]], dtype=torch.long, device=device)
            outputs = model(input_ids)
        logits = outputs["logits"][0, -1, :]
        next_id = _sample_from_logits(logits, allowed_ids=allowed_ids, temperature=temperature, top_k=top_k)
        next_token = inverse_vocab.get(int(next_id), "<UNK>")
        ids.append(int(next_id))
        tokens.append(next_token)
        state.step(next_token)
        if next_token == "<EOS>" or state.done:
            stopped_reason = "eos" if next_token == "<EOS>" else "done"
            break

    return {
        "ids": ids,
        "tokens": tokens,
        "length": int(len(ids)),
        "hit_eos": bool(tokens and tokens[-1] == "<EOS>"),
        "stopped_reason": stopped_reason,
        "constraint_diagnostics": state.diagnostics(),
    }
