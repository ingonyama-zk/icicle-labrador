"""Tham số duy nhất cho :mod:`lnplabrador`.

Chỉnh các dataclass trong file này khi muốn thử một cấu hình khác.  File
``lnplabrador.py`` chỉ chứa phép tính, bộ lấy mẫu challenge, trình biên dịch
profile và codec ``.lab``; nó không ghi đè tham số ở nơi khác.

Có hai lớp tham số được tách rõ:

* ``source_ring`` và ``boundary`` mô tả quan hệ trong
  ``lnplab_labrador_input_relation.tex``;
* ``backend`` mô tả vành ``labradorq40`` mà C++ trong repo được biên dịch cho.

Hai lớp dùng cùng modulus ``q_pi=2^40-195``.  Với ``q mod 128 = 61`` và
``ord_128(q)=32``, ``X^64+1`` tách thành đúng hai nhân tử bất khả quy bậc 32,
như giả thiết soundness của LaBRADOR.  Backend không được dùng NTT trực tiếp
trong ``Z_q``; phép nhân nhanh phải dùng các prime tính toán CRT/NTT rồi đưa
kết quả trở lại ``Z_q``.  Artifact mặc định vẫn chỉ là một
``executable conformance profile`` thu nhỏ, dùng để chạy và kiểm tra đường đi
Python -> LNPLAB01 -> C++ LaBRADOR.  Nó không được gắn nhãn là concrete
security instance hay bản biên dịch đầy đủ của quan hệ NIBS 7,57 triệu hàng.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class SourceRingParameters:
    """Vành chứng minh tại biên LNP--LaBRADOR trong file TeX."""

    degree: int = 64
    modulus: int = (1 << 40) - 195
    coefficient_bits: int = 40


@dataclass(frozen=True)
class BackendParameters:
    """Nguồn duy nhất cho các hằng backend Python/C++.

    ``lnplabrador.py sync-backend`` sinh
    ``src/lnplabrador_backend_params.h`` từ các giá trị này.  Modulus phải
    tiếp tục khớp vành ``labradorq40`` mà ICICLE link ở compile time.
    """

    degree: int = 64
    # Exact requested-TeX / official-reference LOGQ40 modulus.  The paper's
    # concrete table instead uses q near 2^32.  Here q mod 128 = 61 and
    # ord_128(q)=32, so
    # Phi_128(X)=X^64+1 has exactly two irreducible degree-32 factors.
    # Arithmetic acceleration must use computational CRT/NTT primes.
    modulus: int = (1 << 40) - 195
    security_bits: int = 128
    jl_rows: int = 256
    recursions: int = 1
    max_recursions: int = 8
    # The LaBRADOR reference implementation uses LOGDELTA=log2(1.00444)
    # for its 128-bit Core-SVP rank screen.  This remains a planning
    # heuristic here until every Module-SIS call is re-run in a current
    # concrete lattice estimator.
    root_hermite_delta: float = 1.00444
    rank_override: Optional[int] = None
    # With streamed JL rows, the supplied seven-level schedule has a largest
    # audited Ajtai allocation of 536,831 polynomials and a JL peak of
    # 2,097,220 polynomials.  These are runner resource guards (not
    # cryptographic parameters); keep them in this one parameter source so
    # Python validation and C++ enforce the same budget.
    max_artifact_bytes: int = 1 << 30
    # A proof is orders of magnitude smaller than a prover-input artifact.
    # Keep a separate cumulative decoder budget so hostile length fields
    # cannot request GiB-scale native vectors.
    max_proof_bytes: int = 64 << 20
    max_runtime_polynomials: int = 1 << 23
    max_split_parts: int = 256
    extraction_slack_denominator: int = 30


@dataclass(frozen=True)
class LNPChallengeParameters:
    """Challenge cuối của LNP22, Section 2.7.

    Với bậc 64, điều kiện ``sigma_{-1}(c)=c`` để lại 32 hệ số độc lập.
    ``coefficient_bound=8`` cho không gian thô ``17^32 > 2^128``.
    ``eta=140`` là giá trị engineering đã hiệu chuẩn Monte Carlo trong
    ``lnplab.py``; nó chưa phải một chứng minh concrete-security.
    """

    coefficient_bound: int = 8
    power: int = 32
    eta: int = 140
    max_sampling_attempts: int = 1_000_000
    fs_domain: str = "LNPLAB/LNP-final/v1/reference-sampler"


@dataclass(frozen=True)
class LabradorChallengeParameters:
    """Challenge folding cụ thể của LaBRADOR, Section 2."""

    zero_coefficients: int = 23
    unit_coefficients: int = 31
    double_coefficients: int = 10
    operator_norm_bound: float = 15.0
    strict_operator_bound: bool = True
    numerical_guard: float = 1e-9
    paper_average_sampling_attempts: float = 6.0
    paper_soundness_bits: int = 125
    max_sampling_attempts: int = 1_000_000
    fs_domain: str = "LNPLAB/LaBRADOR/folding/v1/reference-sampler"


@dataclass(frozen=True)
class BoundaryParameters:
    """Kích thước của ba flow nén trong file TeX."""

    compressing_msis_rank: int = 2
    message_a_ring_coordinates: int = 2947
    message_h_ring_coordinates: int = 4
    message_w_ring_coordinates: int = 6
    message_a_blocks: Tuple[int, ...] = (
        2,       # t_A
        21,      # t_{s_R}
        21,      # t_{e_R}
        72,      # t_z
        64 * 21, # t_S
        21,      # t_{e_M}
        1458,    # t_kappa
        4,       # four exact-norm slacks
        4,       # packed Y_R
    )


@dataclass(frozen=True)
class FullR1CSInputParameters:
    """Đầu vào cần có để biên dịch reduction binary-R1CS ở Section 6.

    TeX chỉ cho số hàng và các phương trình ký hiệu; nó không chứa ba ma trận
    R1CS hay witness.  Khi có dữ liệu thật, đặt đường dẫn và số biến ở đây.
    ``lnplabrador.py paper-plan`` sẽ báo rõ trạng thái sẵn sàng thay vì tự tạo
    ma trận/witness giả.
    """

    constraint_capacity: int = 1 << 23
    # The Section 6 bound is ||r||^2 <= 2(n_vars + 3k).  Until a concrete
    # instance supplies ``variable_count``, use the same strict upper bound
    # for n_vars and k.  The planner adds one before taking the square root so
    # the public beta remains a strict bound even at the capacity boundary.
    variable_capacity: int = 1 << 23
    variable_count: Optional[int] = None
    matrix_a_path: Optional[str] = None
    matrix_b_path: Optional[str] = None
    matrix_c_path: Optional[str] = None
    witness_path: Optional[str] = None


@dataclass(frozen=True)
class PaperScheduleLevel:
    """Một hàng và, nếu có, phép split sang hàng kế tiếp.

    Năm transition đầu dùng ``r'=2*nu+mu`` của Section 5.3.  Transition thứ
    sáu dùng tail optimization Section 5.6 và có ``r'=nu+mu``.
    Các cột ``reference_*`` chỉ chép lại estimate cũ trong tài liệu; công cụ
    luôn tính lại kích thước từ q, beta và các rank đã chọn.
    """

    level: int
    n: int
    r: int
    beta: float
    nu_to_next: Optional[int]
    mu_to_next: Optional[int]
    tail_transition: bool
    # Joint rank/decomposition fixed point selected by the planner.  Keeping
    # these values in para1.py (instead of deriving them only in code) makes
    # the parameter fingerprint, Python frontend and generated C++ backend
    # agree exactly after future edits.
    kappa: int
    kappa1: int
    kappa2: int
    z_base: int
    base1: int
    base2: int
    base3: int
    digits1: int
    digits2: int
    digits3: int
    reference_witness_kib: Optional[float]
    reference_output_kib: Optional[float]


PAPER_SCHEDULE: Tuple[PaperScheduleLevel, ...] = (
    PaperScheduleLevel(
        1, 27_595, 38, 8192.000061035156, 8, 5, False,
        10, 4, 4, 13, 11, 6, 11, 12, 5, 12,
        8_192.00, 4.2046,
    ),
    PaperScheduleLevel(
        2, 3_450, 21, 8110.469428771037, 3, 4, False,
        10, 4, 4, 22, 32, 13, 32, 8, 4, 8,
        2_235.37, 4.2037,
    ),
    PaperScheduleLevel(
        3, 1_150, 10, 6276.8959768472805, 2, 3, False,
        10, 3, 3, 26, 16, 41, 16, 10, 3, 10,
        423.65, 3.5613,
    ),
    PaperScheduleLevel(
        4, 575, 7, 3445.541589554753, 2, 3, False,
        9, 3, 3, 23, 32, 8, 32, 8, 5, 8,
        147.75, 3.5372,
    ),
    PaperScheduleLevel(
        5, 290, 7, 2685.2159908524613, 1, 3, False,
        9, 3, 3, 24, 32, 8, 32, 8, 5, 8,
        73.27, 3.5191,
    ),
    PaperScheduleLevel(
        6, 290, 5, 2408.5675211574103, 2, 2, True,
        8, 5, 5, 23, 256, 204, 256, 5, 2, 5,
        53.13, 4.1388,
    ),
    # The joint search returns to the user's n=153 row: its capacity 2*153
    # holds 305 auxiliary polynomials and yields a substantially smaller
    # beta_7 than the locally-minimal n=145 alternative.
    PaperScheduleLevel(
        7, 153, 4, 22686.18631345616, None, None, False,
        10, 0, 0, 82, 102, 50, 102, 6, 4, 6,
        40.39, 30.7189,
    ),
)


@dataclass(frozen=True)
class PaperProofParameters:
    """Tham số cho planner/size audit LaBRADOR Sections 5.3--6.

    Rank và decomposition được tìm đồng thời trong box cấu hình, dùng
    root-Hermite factor 1.00444 tương thích ``LOGDELTA`` của reference
    implementation.  Candidate đã chọn được ghi thẳng vào từng schedule row
    để Python/C++ dùng cùng một fixed point.  Đây không thay thế một lần chạy
    Core-SVP/Module-SIS estimator và vì vậy không tạo concrete-security claim.
    """

    schedule: Tuple[PaperScheduleLevel, ...] = field(
        default_factory=lambda: PAPER_SCHEDULE
    )
    initial_beta_mode: str = "binary-r1cs-capacity-strict"
    optimizer_rank_limit: int = 64
    optimizer_max_digits: int = 40
    r1cs_reduction_commitment_rank: int = 2
    constant_term_mask_commitments: int = 4
    lnp_projection_response_bytes: int = 608
    reference_recursive_contribution_kib: float = 54.5087


@dataclass(frozen=True)
class ExecutableProfileParameters:
    """Profile thu nhỏ nhưng chạy thật qua principal relation của C++.

    Sáu ring coordinate đại diện lần lượt cho ``t_A``, ``h``, ``w_A``,
    ``w_u``, ``t_Q`` và ``v_Q``.  Mỗi coordinate được binary-decompose thành
    ``bit_width`` bit hằng.  Giá trị mặc định làm cho cả ``f1``, ``f2`` và
    ``f3`` không tầm thường nhưng vẫn thỏa chính xác.
    """

    bit_width: int = 2
    message_names: Tuple[str, ...] = ("t_A", "h", "w_A", "w_u", "t_Q", "v_Q")
    message_values: Tuple[int, ...] = (1, 3, 0, 0, 1, 0)
    response_names: Tuple[str, ...] = ("z_A_1", "z_A_2", "z_u")
    response_values: Tuple[int, ...] = (1, -1, 1)
    beta_margin: float = 2.0
    # A large base keeps the dense reduced-profile Ajtai matrices and proof
    # runtime comfortably below the generated integration limits for r=21.
    decomposition_base: int = 65_536
    ajtai_seed: str = "LNPLAB/embedded-tex/Ajtai/v1"
    lnp_challenge_seed: str = "LNPLAB/embedded-tex/LNP-challenge/v1"
    labrador_challenge_seed: str = "LNPLAB/embedded-tex/LaBRADOR-challenge/v1"


@dataclass(frozen=True)
class Parameters:
    source_ring: SourceRingParameters = field(default_factory=SourceRingParameters)
    backend: BackendParameters = field(default_factory=BackendParameters)
    lnp_challenge: LNPChallengeParameters = field(default_factory=LNPChallengeParameters)
    labrador_challenge: LabradorChallengeParameters = field(
        default_factory=LabradorChallengeParameters
    )
    boundary: BoundaryParameters = field(default_factory=BoundaryParameters)
    full_r1cs: FullR1CSInputParameters = field(
        default_factory=FullR1CSInputParameters
    )
    paper_proof: PaperProofParameters = field(default_factory=PaperProofParameters)
    executable: ExecutableProfileParameters = field(
        default_factory=ExecutableProfileParameters
    )


PARAMS = Parameters()


SOURCES = {
    "LNP22": "https://eprint.iacr.org/2022/284.pdf",
    "LaBRADOR": "https://eprint.iacr.org/2022/1341.pdf",
    "TRaccoon": "https://eprint.iacr.org/2025/849.pdf",
}
