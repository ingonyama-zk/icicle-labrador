"""Tham số duy nhất cho :mod:`lnplab`.

Chỉnh các giá trị trong file này khi muốn thử một parameter set khác.  File
``lnplab.py`` chỉ chứa phép tính, kiểm tra và bộ lấy mẫu tham chiếu; không đặt
lại các tham số giao thức ở đó.

Các giá trị mặc định bám theo
``lnp_labrador_nibs_relation_detailed.tex``.  Những giá trị chưa được nguồn
chốt (quotient/carry layout, kappa_A, eta của LNP, estimator Module-SIS/MLWE)
được biểu diễn rõ bằng ``None`` hoặc cờ ``False``.  Vì vậy kết quả mặc định là
một *engineering draft*, chưa phải một concrete-security parameter set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class RingParameters:
    """Ba modulus và vành cyclotomic dùng trong tài liệu."""

    degree: int = 64
    q_signature: int = (1 << 36) - 599
    rounding_factor: int = 13
    q_omuf: int = (1 << 36) - 5
    q_proof: int = (1 << 40) - 195


@dataclass(frozen=True)
class ApplicationParameters:
    """Kích thước và bound của NIBS relation."""

    security_bits: int = 128
    recipient_dimension_m: int = 21
    preimage_dimension_k: int = 72
    selector_length: int = 64
    pre_signature_bound: int = 100
    recipient_secret_bound: int = 100
    error_coefficient_bound: int = 6
    bounded_arithmetic_blocks: int = 4


@dataclass(frozen=True)
class ProjectionParameters:
    """Approximate-range/JL parameters.

    ``projected_extra_ring_coordinates`` is ``mu_proj`` in the TeX.  The
    default value 4 counts logical blocks only and is deliberately marked
    incomplete: it does not serialize all range bits and quotient/carry bits.

    The safe defaults use all 256 JL response coordinates and the ceiled
    Gaussian width.  ``lnplab.py`` separately reports the degree-64/raw-width
    calculation that produced B_R=81,385 in the draft TeX; that compatibility
    value must not be used as the default verifier bound.
    """

    jl_rows: int = 256
    centered_binomial_parameter: int = 1
    c1: int = 337
    c2_numerator: int = 13
    c2_denominator: int = 2
    gaussian_multiplier: float = 2.5
    tail_factor: float = 1.65
    projected_extra_ring_coordinates: int = 4
    tail_dimension_mode: str = "jl_rows"
    tail_width_mode: str = "ceil"
    jl_failure_bits_each: int = 128
    jl_failure_events: int = 2
    no_wrap_linear_factor: int = 41
    serialization_complete: bool = False
    quotient_bounds_complete: bool = False


@dataclass(frozen=True)
class LNPChallengeParameters:
    """Challenge C_LNP of LNP22, Section 2.7.

    For degree 64, ``coefficient_bound=8`` is the smallest integer kappa for
    which the 32 independent sigma-fixed coefficients have more than 128 bits
    before filtering.  ``power=32`` follows the experiments in LNP22.

    ``eta=140`` is the reproducible engineering candidate obtained from the
    fixed-seed calibration below (the independent validation acceptance is
    about 99.3%).  ``eta_is_final=False`` keeps the report honest: Monte Carlo
    calibration is not a security proof and response/MSIS widths remain open.
    """

    coefficient_bound: int = 8
    power: int = 32
    eta: Optional[int] = 140
    eta_is_final: bool = False
    target_acceptance: float = 0.99
    calibration_samples: int = 2_048
    validation_samples: int = 4_096
    confidence: float = 0.95
    calibration_seed: int = 0x4C4E_504C_4142_3232
    max_sampling_attempts: int = 1_000_000


@dataclass(frozen=True)
class LabradorChallengeParameters:
    """Concrete degree-64 challenge distribution from LaBRADOR, Section 2."""

    zero_coefficients: int = 23
    unit_coefficients: int = 31
    double_coefficients: int = 10
    operator_norm_bound: float = 15.0
    strict_operator_bound: bool = True
    # Conservative guard for the stdlib double-precision reference FFT.  This
    # is still a test-vector/calibration rule, not a certified production norm.
    operator_norm_numerical_guard: float = 1e-9
    validation_samples: int = 20_000
    confidence: float = 0.95
    calibration_seed: int = 0x4C41_4252_4144_4F52
    paper_soundness_bits: int = 125
    max_sampling_attempts: int = 1_000_000


@dataclass(frozen=True)
class RejectionParameters:
    """Final LNP rejection factors shown in the TeX."""

    gamma_ajtai_message: float = 17.0
    gamma_randomness: float = 1.2


@dataclass(frozen=True)
class CommitmentParameters:
    """Commitment ranks.

    ``kappa_ajtai=None`` is intentional: neither the local TeX nor the cited
    papers instantiate it for this combined degree-64, 40-bit-modulus set.
    """

    kappa_msis: int = 2
    kappa_ajtai: Optional[int] = None
    compact_transcript_flows: int = 3
    constant_term_mask_commitments: int = 4


@dataclass(frozen=True)
class RecursionParameters:
    """Heuristic rules from LaBRADOR Section 5.4.

    The root-Hermite-factor rank is exposed only as a legacy engineering
    heuristic.  A final set still needs a current Module-SIS estimator.
    """

    formula_mode: str = "paper"
    root_hermite_delta: float = 1.00444
    max_split_parts: int = 256
    split_shape_constant: float = 0.25
    minimum_t1: int = 2
    extraction_slack_numerator: int = 128
    extraction_slack_denominator: int = 30
    modular_jl_denominator: int = 125


@dataclass(frozen=True)
class DraftScheduleRow:
    label: str
    rank_n: Optional[int]
    multiplicity_r: Optional[int]
    witness_kib: float
    output_kib: float


DRAFT_RECURSION_SCHEDULE: Tuple[DraftScheduleRow, ...] = (
    DraftScheduleRow("R1CS -> R_LaB", None, None, 1024.00, 0.6250),
    DraftScheduleRow("1", 27_595, 38, 8192.00, 4.2046),
    DraftScheduleRow("2", 3_450, 21, 2235.37, 4.2037),
    DraftScheduleRow("3", 1_150, 10, 423.65, 3.5613),
    DraftScheduleRow("4", 575, 7, 147.75, 3.5372),
    DraftScheduleRow("5", 290, 7, 73.27, 3.5191),
    DraftScheduleRow("6", 290, 5, 53.13, 4.1388),
    DraftScheduleRow("7", 153, 4, 40.39, 30.7189),
)


@dataclass(frozen=True)
class CompositionParameters:
    """Draft size table and Fiat--Shamir domain separation."""

    boundary_binary_constraints: int = 7_600_000
    displayed_recursive_contribution_kib: float = 54.5087
    fs_seed_bits: int = 128
    schedule: Tuple[DraftScheduleRow, ...] = field(
        default_factory=lambda: DRAFT_RECURSION_SCHEDULE
    )
    lnp_fs_domains: Tuple[str, ...] = (
        "LNPLAB/JL/v1",
        "LNPLAB/constant-coefficient/v1",
        "LNPLAB/equation-aggregation/v1",
        "LNPLAB/LNP-final/v1",
    )
    # Each string is additionally suffixed with a canonical recursion level.
    labrador_fs_domains: Tuple[str, ...] = (
        "LNPLAB/LaBRADOR/JL/v1",
        "LNPLAB/LaBRADOR/constant-coefficient/v1",
        "LNPLAB/LaBRADOR/equation-aggregation/v1",
        "LNPLAB/LaBRADOR/folding/v1",
    )


@dataclass(frozen=True)
class Parameters:
    ring: RingParameters = field(default_factory=RingParameters)
    application: ApplicationParameters = field(default_factory=ApplicationParameters)
    projection: ProjectionParameters = field(default_factory=ProjectionParameters)
    lnp_challenge: LNPChallengeParameters = field(default_factory=LNPChallengeParameters)
    labrador_challenge: LabradorChallengeParameters = field(
        default_factory=LabradorChallengeParameters
    )
    rejection: RejectionParameters = field(default_factory=RejectionParameters)
    commitment: CommitmentParameters = field(default_factory=CommitmentParameters)
    recursion: RecursionParameters = field(default_factory=RecursionParameters)
    composition: CompositionParameters = field(default_factory=CompositionParameters)


# Đây là object duy nhất mà lnplab.py nhập và sử dụng.
PARAMS = Parameters()


# Primary sources used by the formulas and composition layout.
SOURCES = {
    "LNP22": "https://eprint.iacr.org/2022/284.pdf",
    "LaBRADOR": "https://eprint.iacr.org/2022/1341.pdf",
    "TRaccoon": "https://eprint.iacr.org/2025/849.pdf",
}
