from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


OVERLAY_INSERT = "OVERLAY_INSERT"
DIVIDE_BY_REGION = "DIVIDE_BY_REGION"
PARALLEL_SUPPORTS = "PARALLEL_SUPPORTS"
RESIDUAL = "RESIDUAL"

OPERATION_TYPES = (OVERLAY_INSERT, DIVIDE_BY_REGION, PARALLEL_SUPPORTS, RESIDUAL)


@dataclass(frozen=True)
class OperationExplainerConfig:
    cost_profile: str = "token_length_v1"
    max_patch_size: int = 32
    max_candidates_per_patch: int = 16
    min_compression_gain: float = 0.0

    enable_overlay_insert: bool = True
    enable_divide_by_region: bool = True
    enable_parallel_supports: bool = True
    enable_residual: bool = True

    use_ortools: bool = True
    allow_greedy_fallback: bool = True
    ortools_time_limit_seconds: float = 10.0
    objective_scale: int = 1000

    cost_template_overlay_insert: float = 4.0
    cost_template_divide_by_region: float = 4.5
    cost_template_parallel_supports: float = 2.0
    cost_template_residual: float = 1.0

    cost_object: float = 2.0
    cost_relation: float = 1.0
    cost_vertex: float = 0.35
    cost_atom: float = 1.1
    cost_atom_vertex: float = 0.25
    cost_group_object: float = 0.75
    cost_node_support: float | None = None
    cost_node_divider: float | None = None
    cost_node_insert: float | None = None
    cost_node_residual: float | None = None
    cost_node_default: float | None = None
    cost_residual_area: float = 0.01
    invalid_cost: float = 1e6
    false_cover_area_weight: float = 0.2
    false_cover_ratio_weight: float = 25.0
    max_false_cover_ratio: float = 0.08
    hard_invalid_false_cover_ratio: float = 0.25

    enable_visible_union: bool = True
    enable_union_with_children: bool = True
    enable_union_with_divider: bool = True
    enable_convex_hull_fill: bool = True
    enable_buffer_close_fill: bool = False
    buffer_close_radius: float = 2.0

    thin_aspect_ratio: float = 4.0
    compactness_threshold: float = 0.45
    small_area_ratio: float = 0.35
    max_insert_group_area_ratio: float = 0.35
    require_insert_touch_or_contained: bool = True
    min_divider_neighbor_count: int = 2
    min_divider_same_label_neighbor_count: int = 2
    max_divider_to_support_area_ratio: float = 0.75
    min_insert_support_boundary_fraction: float = 0.35
    enable_label_pair_consistency: bool = True
    hard_enforce_label_pair_consistency: bool = True
    require_explicit_role_spec_for_label_pairs: bool = True
    label_pair_hard_min_confidence: float = 0.5
    label_pair_min_divided_fragments: int = 2
    label_pair_insert_boundary_fraction: float = 0.35
    label_pair_divider_max_area_ratio: float = 0.75
    label_pair_insert_max_area_ratio: float = 0.35
    token_label_pair_consistency_exception: int = 4
    cost_label_pair_consistency_penalty: float = 3.0
    max_support_label_diversity_without_penalty: int = 2
    support_label_diversity_penalty: float = 2.0
    min_area_eps: float = 1e-8

    token_template_overlay_insert: int = 2
    token_template_divide_by_region: int = 2
    token_template_parallel_supports: int = 1
    token_template_residual: int = 1
    token_node: int = 1
    token_group_node: int = 1
    token_label: int = 1
    token_geometry_model: int = 1
    token_relation: int = 1
    token_relation_type: int = 1
    token_relation_endpoint: int = 1
    token_encode_evidence_refs: bool = False
    token_evidence_reference: int = 1
    token_polygon_start: int = 1
    token_polygon_end: int = 1
    token_polygon_vertex: int = 2
    token_polygon_hole: int = 1
    token_polygon_component: int = 1
    token_atom_start: int = 1
    token_atom_vertex: int = 2
    token_latent_visible_union: int = 0
    token_latent_union_with_children: int = 1
    token_latent_union_with_divider: int = 1
    token_latent_convex_hull_fill: int = 3
    token_latent_buffer_close_fill: int = 4
    token_exception: int = 8
    token_missing_geometry_fallback: int = 4
    false_cover_ratio_invalid: float = 0.08
    false_cover_area_eps: float = 1e-8
    independent_include_face_polygon: bool = True
    independent_include_convex_atoms: bool = True


@dataclass
class OperationPatch:
    id: str
    patch_type: str
    seed_face_id: int | None
    face_ids: Tuple[int, ...]
    arc_ids: Tuple[int, ...]
    metadata: Dict[str, object]


@dataclass
class OperationCandidate:
    id: str
    operation_type: str
    patch_id: str
    covered_face_ids: Tuple[int, ...]
    evidence_arc_ids: Tuple[int, ...]

    nodes: List[Dict[str, object]]
    relations: List[Dict[str, object]]
    residuals: List[Dict[str, object]]

    independent_cost: float
    operation_cost: float
    compression_gain: float
    cost_breakdown: Dict[str, object]

    valid: bool
    failure_reason: str | None = None
    metadata: Dict[str, object] | None = None


@dataclass
class OperationSelectionResult:
    selected_candidate_ids: List[str]
    residual_face_ids: List[int]
    objective_value: float
    solver_status: str
    selection_method: str
    global_optimal: bool
    diagnostics: Dict[str, object]
