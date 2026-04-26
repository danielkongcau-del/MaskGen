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
    cost_residual_area: float = 0.01
    invalid_cost: float = 1e6

    enable_visible_union: bool = True
    enable_union_with_children: bool = True
    enable_union_with_divider: bool = True
    enable_convex_hull_fill: bool = True
    enable_buffer_close_fill: bool = False
    buffer_close_radius: float = 2.0

    thin_aspect_ratio: float = 4.0
    compactness_threshold: float = 0.45
    small_area_ratio: float = 0.35
    min_area_eps: float = 1e-8


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
