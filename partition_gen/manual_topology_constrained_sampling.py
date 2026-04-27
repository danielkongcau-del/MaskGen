from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

from partition_gen.parse_graph_tokenizer import token_int


ROLE_TOKENS = ("ROLE_SUPPORT", "ROLE_DIVIDER", "ROLE_INSERT", "ROLE_INSERT_GROUP")
PAIR_BLOCKS = ("REL_BLOCK_INSERTED_IN", "REL_BLOCK_DIVIDES", "REL_BLOCK_ADJACENT_TO")
OTHER_RELATION_TOKENS = ("REL_INSERTED_IN", "REL_CONTAINS", "REL_DIVIDES", "REL_ADJACENT_TO", "REL_UNKNOWN")
COUNT_PRIOR_KEYS = {"node_count", "child_count", "REL_BLOCK_DIVIDES", "REL_BLOCK_ADJACENT_TO"}


@dataclass(frozen=True)
class TopologyConstrainedSamplerConfig:
    max_nodes: int = 512
    min_nodes: int = 1
    max_label: int = 6
    max_children_per_group: int = 128
    max_relation_pairs: int = 512
    max_other_relations: int = 0
    max_other_refs: int = 16
    force_size: tuple[int, int] = (256, 256)
    force_residual_zero: bool = True
    allow_other_relations: bool = False
    enforce_semantics: bool = True
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
    node_roles: List[str] = field(default_factory=list)
    required_node_roles: Dict[int, str] = field(default_factory=dict)
    assigned_insert_children: set[int] = field(default_factory=set)
    current_children: set[int] = field(default_factory=set)
    relation_pairs: Dict[str, List[tuple[int, int]]] = field(default_factory=dict)
    current_pair_left: int | None = None
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
            return self._allowed_role_tokens()
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
            if cfg.enforce_semantics:
                upper = min(int(cfg.max_children_per_group), len(self._child_index_candidates()))
                return self._int_tokens(1, upper)
            upper = min(int(cfg.max_children_per_group), self._max_node_index() + 1)
            return self._int_tokens(0, upper)
        if self.phase == "child_index":
            return self._child_index_tokens() if cfg.enforce_semantics else self._node_index_tokens()
        if self.phase == "end_node":
            return ["END_NODE"]
        if self.phase == "pair_block_token":
            if self.pair_block_index < len(PAIR_BLOCKS):
                return [PAIR_BLOCKS[self.pair_block_index]]
            return ["REL_BLOCK_OTHER"]
        if self.phase == "pair_count":
            return self._pair_count_tokens() if cfg.enforce_semantics else self._int_tokens(0, int(cfg.max_relation_pairs))
        if self.phase == "pair_endpoint":
            return self._pair_endpoint_tokens() if cfg.enforce_semantics else self._node_index_tokens()
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
            self.node_roles = []
            self.required_node_roles = {}
            self.assigned_insert_children = set()
            self.current_children = set()
            self.relation_pairs = {}
            self.current_pair_left = None
            self.phase = "node_start" if self.node_count > 0 else "pair_block_token"
        elif self.phase == "node_start":
            self.current_role = None
            self.current_child_count = 0
            self.current_child_index = 0
            self.current_children = set()
            self.phase = "node_role"
        elif self.phase == "node_role":
            self.current_role = token
            if len(self.node_roles) == self.node_index:
                self.node_roles.append(token)
            elif len(self.node_roles) > self.node_index:
                self.node_roles[self.node_index] = token
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
            self.current_children = set()
            self.phase = "child_index" if self.current_child_count > 0 else "end_node"
        elif self.phase == "child_index":
            child_index = int(token_int(token))
            self.current_children.add(child_index)
            self.assigned_insert_children.add(child_index)
            self.required_node_roles[child_index] = "ROLE_INSERT"
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
                self.current_pair_left = None
                self.phase = "pair_count"
        elif self.phase == "pair_count":
            self.current_pair_count = int(token_int(token))
            self.current_pair_index = 0
            self.current_pair_endpoint = 0
            self.current_pair_left = None
            self.phase = "pair_endpoint" if self.current_pair_count > 0 else "end_block"
        elif self.phase == "pair_endpoint":
            endpoint = int(token_int(token))
            if self.current_pair_endpoint == 0:
                self.current_pair_left = endpoint
            else:
                block_name = self._current_pair_block_name()
                if block_name is not None and self.current_pair_left is not None:
                    self.relation_pairs.setdefault(block_name, []).append((int(self.current_pair_left), endpoint))
                self.current_pair_left = None
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

    def _allowed_role_tokens(self) -> List[str]:
        allowed = [role for role in self.config.allowed_roles if role in ROLE_TOKENS]
        if not self.config.enforce_semantics:
            return allowed
        required_role = self.required_node_roles.get(int(self.node_index))
        if required_role is not None:
            return [required_role] if required_role in allowed else []
        output = [role for role in allowed if role != "ROLE_INSERT"]
        if "ROLE_INSERT_GROUP" in output and (
            not self._child_index_candidates_for_parent(int(self.node_index))
            or not self._has_existing_insert_container()
        ):
            output.remove("ROLE_INSERT_GROUP")
        return output

    def _child_index_tokens(self) -> List[str]:
        return [f"I_{index}" for index in self._child_index_candidates()]

    def _child_index_candidates(self) -> List[int]:
        return self._child_index_candidates_for_parent(int(self.node_index), exclude_current_children=True)

    def _child_index_candidates_for_parent(
        self,
        parent_index: int,
        *,
        exclude_current_children: bool = False,
    ) -> List[int]:
        candidates: List[int] = []
        current_children = self.current_children if exclude_current_children else set()
        for index in range(0, self._max_node_index() + 1):
            if index == int(parent_index):
                continue
            if index in current_children:
                continue
            if index in self.assigned_insert_children:
                continue
            required_role = self.required_node_roles.get(index)
            if required_role is not None and required_role != "ROLE_INSERT":
                continue
            known_role = self.node_roles[index] if index < len(self.node_roles) else None
            if known_role is not None and known_role != "ROLE_INSERT":
                continue
            candidates.append(int(index))
        return candidates

    def _current_pair_block_name(self) -> str | None:
        if 0 <= int(self.pair_block_index) < len(PAIR_BLOCKS):
            return PAIR_BLOCKS[int(self.pair_block_index)]
        return None

    def _pair_count_tokens(self) -> List[str]:
        cfg = self.config
        block_name = self._current_pair_block_name()
        if block_name is None:
            return []
        possible_count = len(self._possible_pair_keys(block_name))
        upper = min(int(cfg.max_relation_pairs), int(possible_count))
        if block_name == "REL_BLOCK_INSERTED_IN":
            required = self._role_indices("ROLE_INSERT_GROUP")
            exact = min(upper, len(required))
            return [f"I_{exact}"]
        return self._int_tokens(0, upper)

    def _pair_endpoint_tokens(self) -> List[str]:
        block_name = self._current_pair_block_name()
        if block_name is None:
            return []
        if self.current_pair_endpoint == 0:
            candidates = self._pair_left_candidates(block_name)
        elif self.current_pair_left is None:
            candidates = []
        else:
            candidates = self._pair_right_candidates(block_name, int(self.current_pair_left))
        return [f"I_{index}" for index in candidates]

    def _pair_left_candidates(self, block_name: str) -> List[int]:
        if block_name == "REL_BLOCK_INSERTED_IN":
            used_left = {left for left, _right in self.relation_pairs.get(block_name, [])}
            return [index for index in self._role_indices("ROLE_INSERT_GROUP") if index not in used_left]
        if block_name == "REL_BLOCK_DIVIDES":
            return [
                index
                for index in self._role_indices("ROLE_DIVIDER")
                if self._pair_right_candidates(block_name, index)
            ]
        if block_name == "REL_BLOCK_ADJACENT_TO":
            return [
                index
                for index in self._role_indices("ROLE_SUPPORT", "ROLE_INSERT_GROUP", "ROLE_INSERT")
                if self._pair_right_candidates(block_name, index)
            ]
        return []

    def _pair_right_candidates(self, block_name: str, left: int) -> List[int]:
        if block_name == "REL_BLOCK_INSERTED_IN":
            candidates = self._role_indices("ROLE_SUPPORT", "ROLE_INSERT_GROUP")
        elif block_name == "REL_BLOCK_DIVIDES":
            candidates = self._role_indices("ROLE_SUPPORT", "ROLE_INSERT_GROUP")
        elif block_name == "REL_BLOCK_ADJACENT_TO":
            candidates = self._role_indices("ROLE_SUPPORT", "ROLE_INSERT_GROUP", "ROLE_INSERT")
        else:
            return []

        output: List[int] = []
        seen_keys = self._seen_pair_keys(block_name)
        for right in candidates:
            if right == left:
                continue
            key = self._pair_key(block_name, left, right)
            if key in seen_keys:
                continue
            output.append(int(right))
        return output

    def _possible_pair_keys(self, block_name: str) -> set[tuple[int, int]]:
        keys: set[tuple[int, int]] = set()
        for left in self._pair_left_candidates(block_name):
            for right in self._pair_right_candidates(block_name, left):
                keys.add(self._pair_key(block_name, left, right))
        return keys

    def _seen_pair_keys(self, block_name: str) -> set[tuple[int, int]]:
        return {self._pair_key(block_name, left, right) for left, right in self.relation_pairs.get(block_name, [])}

    @staticmethod
    def _pair_key(block_name: str, left: int, right: int) -> tuple[int, int]:
        if block_name == "REL_BLOCK_ADJACENT_TO":
            return tuple(sorted((int(left), int(right))))
        return int(left), int(right)

    def _role_indices(self, *roles: str) -> List[int]:
        role_set = set(str(role) for role in roles)
        return [index for index, role in enumerate(self.node_roles) if str(role) in role_set]

    def _has_existing_insert_container(self) -> bool:
        return any(role in {"ROLE_SUPPORT", "ROLE_INSERT_GROUP"} for role in self.node_roles[: int(self.node_index)])

    def count_prior_key(self) -> str | None:
        if self.phase == "node_count":
            return "node_count"
        if self.phase == "child_count":
            return "child_count"
        if self.phase == "pair_count":
            block_name = self._current_pair_block_name()
            if block_name in {"REL_BLOCK_DIVIDES", "REL_BLOCK_ADJACENT_TO"}:
                return str(block_name)
        return None

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
    logit_bias: Mapping[int, float] | None = None,
) -> int:
    if not allowed_ids:
        raise ValueError("allowed_ids must not be empty")
    allowed = torch.tensor(list(allowed_ids), dtype=torch.long, device=logits.device)
    masked = torch.full_like(logits, float("-inf"))
    masked.index_copy_(0, allowed, logits.index_select(0, allowed))
    if logit_bias:
        for token_id, bias in logit_bias.items():
            token_id = int(token_id)
            if 0 <= token_id < int(masked.numel()):
                masked[token_id] = masked[token_id] + float(bias)
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


def _normalize_count_priors(count_priors: Mapping[str, object] | None) -> Dict[str, Dict[int, float]]:
    if not count_priors:
        return {}
    source = count_priors.get("priors") if isinstance(count_priors, Mapping) else None
    if isinstance(source, Mapping):
        count_priors = source  # type: ignore[assignment]
    elif isinstance(count_priors.get("histograms"), Mapping):
        count_priors = count_priors["histograms"]  # type: ignore[index,assignment]
    output: Dict[str, Dict[int, float]] = {}
    for key, value in dict(count_priors).items():
        if str(key) not in COUNT_PRIOR_KEYS or not isinstance(value, Mapping):
            continue
        counts = {int(inner_key): float(inner_value) for inner_key, inner_value in dict(value).items()}
        total = float(sum(counts.values()))
        if total > 0.0:
            output[str(key)] = {int(inner_key): float(inner_value / total) for inner_key, inner_value in counts.items()}
    return output


def _count_prior_logit_bias(
    *,
    state: TopologyGrammarState,
    allowed_tokens: Sequence[str],
    vocab: Mapping[str, int],
    count_priors: Mapping[str, Mapping[int, float]],
    weight: float,
    smoothing: float,
) -> Dict[int, float]:
    if float(weight) == 0.0:
        return {}
    key = state.count_prior_key()
    if key is None or key not in count_priors:
        return {}
    prior = count_priors[key]
    smoothing_value = max(float(smoothing), 1e-12)
    output: Dict[int, float] = {}
    for token in allowed_tokens:
        if not str(token).startswith("I_") or token not in vocab:
            continue
        count = int(token_int(str(token)))
        probability = max(float(prior.get(count, smoothing_value)), smoothing_value)
        output[int(vocab[str(token)])] = float(weight) * math.log(probability)
    return output


def _complexity_logit_bias(
    *,
    state: TopologyGrammarState,
    allowed_tokens: Sequence[str],
    vocab: Mapping[str, int],
    complexity_level: float,
) -> Dict[int, float]:
    if float(complexity_level) == 0.0 or state.count_prior_key() is None:
        return {}
    count_tokens = [str(token) for token in allowed_tokens if str(token).startswith("I_") and str(token) in vocab]
    if not count_tokens:
        return {}
    counts = [int(token_int(token)) for token in count_tokens]
    max_count = max(counts)
    if max_count <= 0:
        return {}
    denominator = math.log1p(float(max_count))
    if denominator <= 0.0:
        return {}
    return {
        int(vocab[token]): float(complexity_level) * (math.log1p(float(count)) / denominator)
        for token, count in zip(count_tokens, counts)
    }


def _merge_logit_biases(*biases: Mapping[int, float]) -> Dict[int, float]:
    output: Dict[int, float] = {}
    for bias in biases:
        for token_id, value in bias.items():
            token_id = int(token_id)
            output[token_id] = float(output.get(token_id, 0.0) + float(value))
    return output


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
    use_cache: bool = True,
    count_priors: Mapping[str, object] | None = None,
    count_prior_weight: float = 0.0,
    count_prior_smoothing: float = 1e-6,
    complexity_level: float = 0.0,
) -> Dict[str, object]:
    constraint_config = constraint_config or TopologyConstrainedSamplerConfig()
    inverse_vocab = {int(index): str(token) for token, index in vocab.items()}
    if "<BOS>" not in vocab or "<EOS>" not in vocab:
        raise ValueError("vocab must contain <BOS> and <EOS>")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    state = TopologyGrammarState(config=constraint_config)
    normalized_count_priors = _normalize_count_priors(count_priors)
    ids = [int(vocab["<BOS>"])]
    tokens = ["<BOS>"]
    stopped_reason = "max_new_tokens"
    block_size = int(getattr(model.config, "block_size", max_new_tokens + 1))
    use_kv_cache = bool(
        use_cache and getattr(model, "supports_kv_cache", False) and int(max_new_tokens) + 1 <= int(block_size)
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
            input_ids = torch.tensor([[ids[-1]]], dtype=torch.long, device=device)
            outputs = model(input_ids, past_kv=past_kv, use_cache=True)
            past_kv = outputs["past_kv"]
        else:
            input_ids = torch.tensor([ids[-int(block_size) :]], dtype=torch.long, device=device)
            outputs = model(input_ids)
        logits = outputs["logits"][0, -1, :]
        logit_bias = _merge_logit_biases(
            _count_prior_logit_bias(
                state=state,
                allowed_tokens=allowed_tokens,
                vocab=vocab,
                count_priors=normalized_count_priors,
                weight=float(count_prior_weight),
                smoothing=float(count_prior_smoothing),
            ),
            _complexity_logit_bias(
                state=state,
                allowed_tokens=allowed_tokens,
                vocab=vocab,
                complexity_level=float(complexity_level),
            ),
        )
        next_id = _sample_from_logits(
            logits,
            allowed_ids=allowed_ids,
            temperature=temperature,
            top_k=top_k,
            logit_bias=logit_bias,
        )
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
        "count_prior_diagnostics": {
            "enabled": bool(normalized_count_priors and float(count_prior_weight) != 0.0),
            "weight": float(count_prior_weight),
            "smoothing": float(count_prior_smoothing),
            "keys": sorted(normalized_count_priors.keys()),
        },
        "complexity_diagnostics": {
            "enabled": bool(float(complexity_level) != 0.0),
            "level": float(complexity_level),
        },
    }
