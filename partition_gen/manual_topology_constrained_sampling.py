from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from partition_gen.parse_graph_tokenizer import token_int


ROLE_TOKENS = ("ROLE_SUPPORT", "ROLE_DIVIDER", "ROLE_INSERT", "ROLE_INSERT_GROUP")
PAIR_BLOCKS = ("REL_BLOCK_INSERTED_IN", "REL_BLOCK_DIVIDES", "REL_BLOCK_ADJACENT_TO")
OTHER_RELATION_TOKENS = ("REL_INSERTED_IN", "REL_CONTAINS", "REL_DIVIDES", "REL_ADJACENT_TO", "REL_UNKNOWN")


@dataclass(frozen=True)
class TopologyConstrainedSamplerConfig:
    max_nodes: int = 256
    min_nodes: int = 1
    max_label: int = 6
    max_children_per_group: int = 128
    max_relation_pairs: int = 512
    max_other_relations: int = 0
    max_other_refs: int = 16
    force_size: tuple[int, int] = (256, 256)
    force_residual_zero: bool = True
    allow_other_relations: bool = False
    allowed_roles: tuple[str, ...] = ROLE_TOKENS


@dataclass
class TopologyGrammarState:
    config: TopologyConstrainedSamplerConfig = field(default_factory=TopologyConstrainedSamplerConfig)
    phase: str = "manual"
    node_count: int | None = None
    node_index: int = 0
    current_role: str | None = None
    current_child_count: int = 0
    current_child_index: int = 0
    pair_block_index: int = 0
    current_pair_count: int = 0
    current_pair_index: int = 0
    current_pair_endpoint: int = 0
    other_relation_count: int = 0
    other_relation_index: int = 0
    other_ref_count: int = 0
    other_ref_index: int = 0
    done: bool = False
    errors: List[str] = field(default_factory=list)

    def allowed_token_strings(self) -> List[str]:
        if self.done:
            return []
        cfg = self.config
        if self.phase == "manual":
            return ["MANUAL_TOPOLOGY_V1"]
        if self.phase == "size":
            return ["SIZE"]
        if self.phase == "height":
            return [f"I_{int(cfg.force_size[0])}"]
        if self.phase == "width":
            return [f"I_{int(cfg.force_size[1])}"]
        if self.phase == "node_block":
            return ["NODE_BLOCK"]
        if self.phase == "node_count":
            return self._int_tokens(int(cfg.min_nodes), int(cfg.max_nodes))
        if self.phase == "node_start":
            return ["NODE"]
        if self.phase == "node_role":
            return list(cfg.allowed_roles)
        if self.phase == "node_label":
            return self._int_tokens(0, int(cfg.max_label))
        if self.phase == "node_renderable":
            return ["I_0"] if self.current_role == "ROLE_INSERT_GROUP" else ["I_1"]
        if self.phase == "node_ref_only":
            return ["I_0"]
        if self.phase == "node_geometry_model":
            return ["GEOM_NONE"] if self.current_role == "ROLE_INSERT_GROUP" else ["GEOM_POLYGON_CODE"]
        if self.phase == "node_geometry_ref":
            return ["I_0"] if self.current_role == "ROLE_INSERT_GROUP" else ["I_1"]
        if self.phase == "children_token":
            return ["CHILDREN"]
        if self.phase == "child_count":
            upper = min(int(cfg.max_children_per_group), self._max_node_index() + 1)
            return self._int_tokens(0, upper)
        if self.phase == "child_index":
            return self._node_index_tokens()
        if self.phase == "end_node":
            return ["END_NODE"]
        if self.phase == "pair_block_token":
            if self.pair_block_index < len(PAIR_BLOCKS):
                return [PAIR_BLOCKS[self.pair_block_index]]
            return ["REL_BLOCK_OTHER"]
        if self.phase == "pair_count":
            return self._int_tokens(0, int(cfg.max_relation_pairs))
        if self.phase == "pair_endpoint":
            return self._node_index_tokens()
        if self.phase == "end_block":
            return ["END_BLOCK"]
        if self.phase == "other_count":
            upper = int(cfg.max_other_relations) if cfg.allow_other_relations else 0
            return self._int_tokens(0, upper)
        if self.phase == "other_relation_type":
            return list(OTHER_RELATION_TOKENS)
        if self.phase == "other_ref_count":
            return self._int_tokens(0, int(cfg.max_other_refs))
        if self.phase == "other_ref":
            return self._node_index_tokens()
        if self.phase == "residuals":
            return ["RESIDUALS"]
        if self.phase == "residual_count":
            return ["I_0"] if cfg.force_residual_zero else self._int_tokens(0, int(cfg.max_nodes))
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
            self.phase = "size"
        elif self.phase == "size":
            self.phase = "height"
        elif self.phase == "height":
            self.phase = "width"
        elif self.phase == "width":
            self.phase = "node_block"
        elif self.phase == "node_block":
            self.phase = "node_count"
        elif self.phase == "node_count":
            self.node_count = int(token_int(token))
            self.node_index = 0
            self.phase = "node_start" if self.node_count > 0 else "pair_block_token"
        elif self.phase == "node_start":
            self.current_role = None
            self.current_child_count = 0
            self.current_child_index = 0
            self.phase = "node_role"
        elif self.phase == "node_role":
            self.current_role = token
            self.phase = "node_label"
        elif self.phase == "node_label":
            self.phase = "node_renderable"
        elif self.phase == "node_renderable":
            self.phase = "node_ref_only"
        elif self.phase == "node_ref_only":
            self.phase = "node_geometry_model"
        elif self.phase == "node_geometry_model":
            self.phase = "node_geometry_ref"
        elif self.phase == "node_geometry_ref":
            self.phase = "children_token" if self.current_role == "ROLE_INSERT_GROUP" else "end_node"
        elif self.phase == "children_token":
            self.phase = "child_count"
        elif self.phase == "child_count":
            self.current_child_count = int(token_int(token))
            self.current_child_index = 0
            self.phase = "child_index" if self.current_child_count > 0 else "end_node"
        elif self.phase == "child_index":
            self.current_child_index += 1
            self.phase = "end_node" if self.current_child_index >= self.current_child_count else "child_index"
        elif self.phase == "end_node":
            self.node_index += 1
            self.current_role = None
            if self.node_count is not None and self.node_index >= self.node_count:
                self.phase = "pair_block_token"
            else:
                self.phase = "node_start"
        elif self.phase == "pair_block_token":
            if token == "REL_BLOCK_OTHER":
                self.phase = "other_count"
            else:
                self.current_pair_count = 0
                self.current_pair_index = 0
                self.current_pair_endpoint = 0
                self.phase = "pair_count"
        elif self.phase == "pair_count":
            self.current_pair_count = int(token_int(token))
            self.current_pair_index = 0
            self.current_pair_endpoint = 0
            self.phase = "pair_endpoint" if self.current_pair_count > 0 else "end_block"
        elif self.phase == "pair_endpoint":
            self.current_pair_endpoint += 1
            if self.current_pair_endpoint >= 2:
                self.current_pair_endpoint = 0
                self.current_pair_index += 1
            self.phase = "end_block" if self.current_pair_index >= self.current_pair_count else "pair_endpoint"
        elif self.phase == "end_block":
            if self.pair_block_index < len(PAIR_BLOCKS):
                self.pair_block_index += 1
                self.phase = "pair_block_token"
            else:
                self.phase = "residuals"
        elif self.phase == "other_count":
            self.other_relation_count = int(token_int(token))
            self.other_relation_index = 0
            self.phase = "other_relation_type" if self.other_relation_count > 0 else "end_block"
        elif self.phase == "other_relation_type":
            self.phase = "other_ref_count"
        elif self.phase == "other_ref_count":
            self.other_ref_count = int(token_int(token))
            self.other_ref_index = 0
            if self.other_ref_count > 0:
                self.phase = "other_ref"
            else:
                self.other_relation_index += 1
                self.phase = (
                    "other_relation_type" if self.other_relation_index < self.other_relation_count else "end_block"
                )
        elif self.phase == "other_ref":
            self.other_ref_index += 1
            if self.other_ref_index >= self.other_ref_count:
                self.other_relation_index += 1
                self.phase = (
                    "other_relation_type" if self.other_relation_index < self.other_relation_count else "end_block"
                )
            else:
                self.phase = "other_ref"
        elif self.phase == "residuals":
            self.phase = "residual_count"
        elif self.phase == "residual_count":
            self.phase = "eos"
        elif self.phase == "eos":
            self.done = True
        return True

    def diagnostics(self) -> Dict[str, object]:
        return {
            "config": asdict(self.config),
            "phase": self.phase,
            "done": bool(self.done),
            "errors": list(self.errors),
            "node_count_declared": int(self.node_count or 0),
            "node_index": int(self.node_index),
            "pair_block_index": int(self.pair_block_index),
        }

    def _max_node_index(self) -> int:
        return max(0, int(self.node_count or 0) - 1)

    def _node_index_tokens(self) -> List[str]:
        return self._int_tokens(0, self._max_node_index())

    @staticmethod
    def _int_tokens(low: int, high: int) -> List[str]:
        if high < low:
            return []
        return [f"I_{index}" for index in range(int(low), int(high) + 1)]


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
def sample_topology_constrained(
    model,
    vocab: Dict[str, int],
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    constraint_config: TopologyConstrainedSamplerConfig | None = None,
    device: torch.device | str | None = None,
) -> Dict[str, object]:
    constraint_config = constraint_config or TopologyConstrainedSamplerConfig()
    inverse_vocab = {int(index): str(token) for token, index in vocab.items()}
    if "<BOS>" not in vocab or "<EOS>" not in vocab:
        raise ValueError("vocab must contain <BOS> and <EOS>")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    state = TopologyGrammarState(config=constraint_config)
    ids = [int(vocab["<BOS>"])]
    tokens = ["<BOS>"]
    stopped_reason = "max_new_tokens"

    for _step in range(int(max_new_tokens)):
        allowed_tokens = state.allowed_token_strings()
        allowed_ids = [int(vocab[token]) for token in allowed_tokens if token in vocab]
        if not allowed_ids:
            state.errors.append(f"empty_allowed_set_phase_{state.phase}")
            stopped_reason = "empty_allowed_set"
            break
        input_ids = torch.tensor([ids[-int(getattr(model.config, "block_size", len(ids))) :]], dtype=torch.long, device=device)
        logits = model(input_ids)["logits"][0, -1, :]
        next_id = _sample_from_logits(logits, allowed_ids=allowed_ids, temperature=temperature, top_k=top_k)
        next_token = inverse_vocab.get(int(next_id), "<UNK>")
        ids.append(int(next_id))
        tokens.append(next_token)
        state.step(next_token)
        if next_token == "<EOS>" or state.done:
            stopped_reason = "eos" if next_token == "<EOS>" else "done"
            break

    hit_eos = bool(tokens and tokens[-1] == "<EOS>")
    return {
        "ids": ids,
        "tokens": tokens,
        "length": int(len(ids)),
        "hit_eos": hit_eos,
        "stopped_reason": stopped_reason,
        "constraint_diagnostics": state.diagnostics(),
    }
