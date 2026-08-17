#!/usr/bin/env python3
"""Create and validate inputs for the local ICICLE LaBRADOR runner.

``lnplab.py`` audits parameters and challenge spaces; it does not contain a
numeric witness or a concrete :class:`LabradorInstance`.  This program is the
bridge between that audit profile and the C++ demo.  It can:

* generate a deterministic, non-trivial synthetic principal relation;
* pack a caller-supplied JSON relation into the versioned ``.lab`` format;
* inspect and exactly validate a ``.lab`` bundle in coefficient space;
* invoke ``build/src/lab_runner DEVICE FILE`` after validation.

The generated ``synthetic-principal-v1`` relation is a backend integration
test, not the full NIBS relation described by the TeX document.  Similarly,
``json-principal-v1`` means that coefficients were supplied by the caller; it
does not attest that they are a correct compilation of the NIBS relation.

.. warning::
   A ``.lab`` bundle and its JSON form contain the **secret witness in
   plaintext**.  They are local test inputs, not proofs, and must not be
   published or sent to a verifier.

Binary format v1
----------------

Every integer and the IEEE-754 binary64 ``beta`` are little-endian.  Every
length-prefixed byte string is encoded as ``u64 length || bytes``.  Polynomial
coefficients are unique centered signed ``i64`` representatives.

::

    b"LNPLAB01"
    u32 version (= 1), u32 degree (= 64)
    u64 modulus, u64 r, u64 n, f64 beta
    u64 kappa, u64 kappa1, u64 kappa2
    u32 base1, u32 base2, u32 base3
    u64 JL_out, u64 aggregation_rounds, u64 recursions
    lp(mode UTF-8), lp(source parameter fingerprint ASCII)
    lp(Ajtai seed), lp(public oracle-context seed)
    u64 equality_count
      repeated: a[r*r] polys, phi[r*n] polys, b[1] poly
    u64 const_zero_count
      repeated: a[r*r] polys, phi[r*n] polys, i64 b0
    byte[32] SHA3-256(all bytes above, starting at the magic)
    u64 witness_count (= r*n), witness[witness_count] polys
    EOF

All public polynomials in the file are in coefficient domain.  The C++ loader
must forward-negacyclic-NTT ``a``, ``phi`` and equality ``b`` before building
``EqualityInstance``/``ConstZeroInstance``.  The witness remains in ``Rq``;
the constant-zero ``b0`` remains a scalar ``Zq``.

JSON input schema v1
--------------------

The ``pack`` command accepts the following dense, row-major schema.  A scalar
integer may be a JSON integer or a base-10 string.  A polynomial is a list of
exactly 64 centered integers.

::

    {
      "schema": "lnplab-relation-json-v1",
      "source_parameter_fingerprint": "<64 lowercase hex characters>",
      "backend": {"degree": 64, "modulus": "1099511627581"},
      "parameters": {
        "r": 2, "n": 4, "beta": 50.0,
        "kappa": 24, "kappa1": 24, "kappa2": 24,
        "base1": 32, "base2": 32, "base3": 32,
        "JL_out": 256, "aggregation_rounds": 4, "recursions": 1
      },
      "seeds": {"ajtai_hex": "...", "oracle_hex": "..."},
      "witness": [r][n][64],
      "equality_constraints": [
        {"a": [r][r][64], "phi": [r][n][64], "b": [64]}
      ],
      "const_zero_constraints": [
        {"a": [r][r][64], "phi": [r][n][64], "b": 0}
      ]
    }

The executable modulus and all backend constants come from ``para1.py``.
This tool refuses artifacts for any other modulus; it never silently reduces
coefficients into the compiled backend ring.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Works both as ``python scripts/lab.py`` and as a package import.
    from .para import PARAMS, Parameters
    from .para1 import PARAMS as BACKEND_SOURCE_PARAMS
    from .lnplab import parameter_fingerprint, validate_parameters
except ImportError:
    from para import PARAMS, Parameters
    from para1 import PARAMS as BACKEND_SOURCE_PARAMS
    from lnplab import parameter_fingerprint, validate_parameters


MAGIC = b"LNPLAB01"
FORMAT_VERSION = 1
DEGREE = BACKEND_SOURCE_PARAMS.backend.degree
BACKEND_Q = BACKEND_SOURCE_PARAMS.backend.modulus
MAX_WITNESS_COEFFICIENT_ABS = math.isqrt(BACKEND_Q) - 1
EXPECTED_AGGREGATION_ROUNDS = math.ceil(
    BACKEND_SOURCE_PARAMS.backend.security_bits / math.log2(BACKEND_Q)
)
EXPECTED_OPERATOR_NORM_BOUND = (
    BACKEND_SOURCE_PARAMS.labrador_challenge.operator_norm_bound
)
BOUNDARY_PROFILE_MODE = "lnplabrador-boundary-profile-v1"
ALLOWED_MODES = frozenset(
    ("synthetic-principal-v1", "json-principal-v1", BOUNDARY_PROFILE_MODE)
)
JSON_SCHEMA_NAME = "lnplab-relation-json-v1"
MAX_ARTIFACT_BYTES = BACKEND_SOURCE_PARAMS.backend.max_artifact_bytes
MAX_BYTE_STRING = 1 << 20
MAX_CONSTRAINTS = 1_000_000
MAX_RUNTIME_POLYNOMIALS = BACKEND_SOURCE_PARAMS.backend.max_runtime_polynomials
JL_AGGREGATION_SCRATCH_BYTES = 64 * 1024 * 1024
# The q40 field uses two 32-bit limbs, so this matches sizeof(Rq) in the
# compiled C++ backend. Keep the streamed-JL resource calculation in units
# of the same ring polynomials used by MAX_RUNTIME_POLYNOMIALS.
RQ_STORAGE_BYTES = DEGREE * ((BACKEND_Q.bit_length() + 31) // 32) * 4
INT_MAX = (1 << 31) - 1
UINT32_MAX = (1 << 32) - 1


class LabError(ValueError):
    """A malformed, incompatible, or unsatisfied relation bundle."""


@dataclass
class EqualityConstraint:
    """Dense coefficient-domain equality constraint."""

    a: List[List[int]]
    phi: List[List[int]]
    b: List[int]


@dataclass
class ConstZeroConstraint:
    """Dense coefficient-domain constant-zero constraint."""

    a: List[List[int]]
    phi: List[List[int]]
    b: int


@dataclass
class RelationBundle:
    mode: str
    source_fingerprint: str
    degree: int
    modulus: int
    r: int
    n: int
    beta: float
    kappa: int
    kappa1: int
    kappa2: int
    base1: int
    base2: int
    base3: int
    jl_out: int
    aggregation_rounds: int
    recursions: int
    ajtai_seed: bytes
    oracle_seed: bytes
    witness: List[List[int]]
    equality_constraints: List[EqualityConstraint]
    const_zero_constraints: List[ConstZeroConstraint]
    public_digest: Optional[bytes] = None


def _checked_product(*values: int, label: str) -> int:
    result = 1
    for value in values:
        if value < 0:
            raise LabError(f"{label}: negative dimension")
        result *= value
        if result > INT_MAX:
            raise LabError(f"{label}: product exceeds the C++ int/NTT limit")
    return result


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_integer(value: Any, label: str) -> int:
    if _is_plain_int(value):
        return value
    if isinstance(value, str):
        if not value or value.strip() != value:
            raise LabError(f"{label}: invalid decimal integer")
        try:
            return int(value, 10)
        except ValueError as exc:
            raise LabError(f"{label}: invalid decimal integer") from exc
    raise LabError(f"{label}: expected an integer or decimal string")


def _expect_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise LabError(f"{label}: expected an object")
    return value


def _expect_keys(
    obj: Mapping[str, Any], required: Iterable[str], optional: Iterable[str], label: str
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(obj))
    unknown = sorted(set(obj) - allowed)
    if missing:
        raise LabError(f"{label}: missing keys: {', '.join(missing)}")
    if unknown:
        raise LabError(f"{label}: unknown keys: {', '.join(unknown)}")


def _parse_hex(value: Any, label: str) -> bytes:
    if not isinstance(value, str) or len(value) % 2:
        raise LabError(f"{label}: expected an even-length hexadecimal string")
    try:
        result = bytes.fromhex(value)
    except ValueError as exc:
        raise LabError(f"{label}: invalid hexadecimal string") from exc
    if result.hex() != value.lower():
        raise LabError(f"{label}: whitespace and separators are not allowed")
    return result


def centered(value: int, modulus: int = BACKEND_Q) -> int:
    """Return the unique centered representative modulo an odd modulus."""

    value %= modulus
    return value - modulus if value > modulus // 2 else value


def _check_coefficient(value: Any, modulus: int, label: str) -> int:
    coefficient = _parse_integer(value, label)
    if not -(modulus // 2) <= coefficient <= modulus // 2:
        raise LabError(f"{label}: coefficient is not a canonical centered representative")
    return coefficient


def _zero_poly(degree: int = DEGREE) -> List[int]:
    return [0] * degree


def _parse_poly(value: Any, modulus: int, degree: int, label: str) -> List[int]:
    if not isinstance(value, list) or len(value) != degree:
        raise LabError(f"{label}: expected exactly {degree} coefficients")
    return [
        _check_coefficient(coefficient, modulus, f"{label}[{index}]")
        for index, coefficient in enumerate(value)
    ]


def negacyclic_mul(left: Sequence[int], right: Sequence[int], modulus: int) -> List[int]:
    """Multiply in Z_q[x]/(x^d+1), returning centered coefficients."""

    if len(left) != len(right):
        raise LabError("negacyclic multiplication degree mismatch")
    degree = len(left)
    accumulation = [0] * degree
    for i, left_value in enumerate(left):
        if left_value == 0:
            continue
        for j, right_value in enumerate(right):
            if right_value == 0:
                continue
            target = i + j
            product = left_value * right_value
            if target >= degree:
                target -= degree
                product = -product
            accumulation[target] = (accumulation[target] + product) % modulus
    return [centered(value, modulus) for value in accumulation]


def _poly_add_in_place(target: List[int], addend: Sequence[int], modulus: int) -> None:
    for index, value in enumerate(addend):
        target[index] = (target[index] + value) % modulus


def _poly_is_zero(poly: Sequence[int]) -> bool:
    return not any(poly)


def _evaluate_constraint_polynomial(
    bundle: RelationBundle,
    a: Sequence[Sequence[int]],
    phi: Sequence[Sequence[int]],
    gram_cache: Dict[Tuple[int, int], List[int]],
) -> List[int]:
    """Evaluate the quadratic and linear terms in coefficient domain."""

    modulus = bundle.modulus
    degree = bundle.degree
    result = [0] * degree

    for flat_index, coefficient_poly in enumerate(a):
        if _poly_is_zero(coefficient_poly):
            continue
        i, j = divmod(flat_index, bundle.r)
        cache_key = (min(i, j), max(i, j))
        gram = gram_cache.get(cache_key)
        if gram is None:
            gram = [0] * degree
            for column in range(bundle.n):
                left = bundle.witness[i * bundle.n + column]
                right = bundle.witness[j * bundle.n + column]
                _poly_add_in_place(gram, negacyclic_mul(left, right, modulus), modulus)
            gram = [centered(value, modulus) for value in gram]
            gram_cache[cache_key] = gram
        _poly_add_in_place(result, negacyclic_mul(coefficient_poly, gram, modulus), modulus)

    for flat_index, coefficient_poly in enumerate(phi):
        if _poly_is_zero(coefficient_poly):
            continue
        witness_poly = bundle.witness[flat_index]
        _poly_add_in_place(
            result, negacyclic_mul(coefficient_poly, witness_poly, modulus), modulus
        )

    return [centered(value, modulus) for value in result]


def backend_secure_rank(parameters: Parameters = PARAMS) -> int:
    """Match the current C++ ``secure_msis_rank`` heuristic for the backend ring."""

    delta = parameters.recursion.root_hermite_delta
    log_delta = math.log2(delta)
    log_q = math.log2(BACKEND_Q)
    return math.ceil((log_q - 1.0) ** 2 / (4.0 * log_delta * log_q * DEGREE))


def _compiled_backend_secure_rank() -> int:
    """Rank generated into the C++ runner from ``para1.py``."""

    log_delta = math.log2(BACKEND_SOURCE_PARAMS.backend.root_hermite_delta)
    log_q = math.log2(BACKEND_Q)
    return math.ceil((log_q - 1.0) ** 2 / (4.0 * log_delta * log_q * DEGREE))


def _balanced_digit_count(modulus: int, base: int) -> int:
    """Match the current ICICLE balanced-decomposition allocation length."""

    digits = 0
    power = 1
    while power < modulus:
        power *= base
        digits += 1
    return digits + (1 if base > 2 else 0)


def _ceil_nth_root(value: int, degree: int) -> int:
    estimate = max(1, int(math.exp(math.log(value) / degree)))
    while estimate**degree < value:
        estimate += 1
    while estimate > 1 and (estimate - 1) ** degree >= value:
        estimate -= 1
    return estimate


def _recursive_decomposition_profile(
    modulus: int, n: int, r: int, beta: float
) -> Tuple[int, int, int, int, int, int]:
    """Return ``(base1,base2,base3,t1,t2,t3)`` for Sections 5.3--5.4."""

    coefficient_sd = beta / math.sqrt(r * n * DEGREE)
    fold_base = max(
        2,
        int(
            math.floor(
                math.sqrt(coefficient_sd * math.sqrt(12.0 * r * 71.0)) + 0.5
            )
        ),
    )
    t1 = max(2, int(math.floor(math.log(modulus) / math.log(fold_base) + 0.5)))
    base1 = _ceil_nth_root(modulus, t1)
    garbage_scale = math.sqrt(24.0 * n * DEGREE) * coefficient_sd**2
    logarithmic_t2 = (
        math.log(garbage_scale) / math.log(fold_base) if garbage_scale > 1.0 else 0.0
    )
    t2 = max(2, int(math.floor(max(0.0, logarithmic_t2) + 0.5)))
    base2 = max(
        2,
        int(math.floor(math.exp(math.log(max(1.0, garbage_scale)) / t2) + 0.5)),
    )
    return base1, base2, base1, t1, t2, t1


def _jl_aggregation_chunk_rows(
    row_size_polynomials: int,
    total_rows: int,
    scratch_bytes: int = JL_AGGREGATION_SCRATCH_BYTES,
) -> int:
    """Match ``jl_aggregation_chunk_rows`` in the C++ streaming backend."""

    if row_size_polynomials <= 0 or total_rows <= 0 or scratch_bytes <= 0:
        raise LabError("JL aggregation dimensions must be positive")
    bytes_per_row = row_size_polynomials * RQ_STORAGE_BYTES
    rows_by_budget = max(1, scratch_bytes // bytes_per_row)
    return min(total_rows, rows_by_budget)


def _jl_streaming_working_set_polynomials(
    row_size_polynomials: int, total_rows: int
) -> int:
    """Return the C++ peak: one accumulator plus one bounded row chunk."""

    chunk_rows = _jl_aggregation_chunk_rows(row_size_polynomials, total_rows)
    return (chunk_rows + 1) * row_size_polynomials


def _paper_msis_rank(norm_bound: float) -> int:
    """Reference rank formula used to generate the checked paper schedule."""

    backend = BACKEND_SOURCE_PARAMS.backend
    numerator = math.log2(norm_bound) ** 2
    denominator = (
        4.0
        * backend.degree
        * math.log2(backend.modulus)
        * math.log2(backend.root_hermite_delta)
    )
    return max(1, math.floor(numerator / denominator) + 1)


def _generated_paper_initial_beta() -> float:
    return BACKEND_SOURCE_PARAMS.paper_proof.schedule[0].beta


def _generated_paper_first_ranks() -> Tuple[int, int, int]:
    """Read the committed first schedule ranks from ``para1.py``."""

    row = BACKEND_SOURCE_PARAMS.paper_proof.schedule[0]
    return row.kappa, row.kappa1, row.kappa2


def _matches_generated_paper_schedule(bundle: RelationBundle) -> bool:
    """Recognize only the exact seven-level boundary profile accepted by C++."""

    schedule = BACKEND_SOURCE_PARAMS.paper_proof.schedule
    if bundle.mode != BOUNDARY_PROFILE_MODE or len(schedule) != 7:
        return False
    first = schedule[0]
    expected_beta = _generated_paper_initial_beta()
    if (
        not math.isfinite(bundle.beta)
        or bundle.recursions != 7
        or bundle.n != first.n
        or bundle.r != first.r
        or (bundle.kappa, bundle.kappa1, bundle.kappa2)
        != _generated_paper_first_ranks()
        or bundle.beta != expected_beta
    ):
        return False
    expected_bases = (first.base1, first.base2, first.base3)
    return (bundle.base1, bundle.base2, bundle.base3) == expected_bases


def source_profile_issues(parameters: Parameters = PARAMS) -> List[str]:
    """Return mismatches that make the LaBRADOR backend profile unusable."""

    validate_parameters(parameters)
    issues: List[str] = []
    if parameters.ring.degree != DEGREE:
        issues.append(f"ring.degree={parameters.ring.degree}, backend requires {DEGREE}")
    if parameters.projection.jl_rows != BACKEND_SOURCE_PARAMS.backend.jl_rows:
        issues.append("projection.jl_rows must match para1.py for this runner profile")
    if parameters.application.security_bits != BACKEND_SOURCE_PARAMS.backend.security_bits:
        issues.append("application.security_bits must match para1.py")
    if (
        parameters.recursion.root_hermite_delta
        != BACKEND_SOURCE_PARAMS.backend.root_hermite_delta
    ):
        issues.append("recursion.root_hermite_delta must match para1.py")
    if parameters.recursion.minimum_t1 != 2:
        issues.append("recursion.minimum_t1 must match the C++ fixed value 2")
    if parameters.recursion.split_shape_constant != 0.25:
        issues.append("recursion.split_shape_constant must match the C++ value 0.25")
    if parameters.recursion.max_split_parts != BACKEND_SOURCE_PARAMS.backend.max_split_parts:
        issues.append("recursion.max_split_parts must match para1.py")
    challenge = parameters.labrador_challenge
    if (
        challenge.zero_coefficients,
        challenge.unit_coefficients,
        challenge.double_coefficients,
    ) != (
        BACKEND_SOURCE_PARAMS.labrador_challenge.zero_coefficients,
        BACKEND_SOURCE_PARAMS.labrador_challenge.unit_coefficients,
        BACKEND_SOURCE_PARAMS.labrador_challenge.double_coefficients,
    ):
        issues.append("LaBRADOR challenge weights must match para1.py")
    if challenge.operator_norm_bound != EXPECTED_OPERATOR_NORM_BOUND:
        issues.append("LaBRADOR operator norm bound must be 15")
    if not challenge.strict_operator_bound:
        issues.append("LaBRADOR operator norm comparison must be strict")
    return issues


def _require_source_profile(parameters: Parameters = PARAMS) -> None:
    issues = source_profile_issues(parameters)
    if issues:
        raise LabError("para.py is incompatible with the compiled backend: " + "; ".join(issues))


def _validate_poly_list(
    polys: Sequence[Sequence[int]], expected_count: int, bundle: RelationBundle, label: str
) -> None:
    if len(polys) != expected_count:
        raise LabError(f"{label}: expected {expected_count} polynomials, got {len(polys)}")
    for poly_index, poly in enumerate(polys):
        if len(poly) != bundle.degree:
            raise LabError(f"{label}[{poly_index}]: wrong degree")
        for coefficient_index, coefficient in enumerate(poly):
            if not _is_plain_int(coefficient):
                raise LabError(f"{label}[{poly_index}][{coefficient_index}]: not an integer")
            if not -(bundle.modulus // 2) <= coefficient <= bundle.modulus // 2:
                raise LabError(
                    f"{label}[{poly_index}][{coefficient_index}]: non-canonical coefficient"
                )


def validate_bundle(bundle: RelationBundle, *, verify_relation: bool = True) -> Dict[str, Any]:
    """Validate shape, backend compatibility, norms, and relation satisfaction."""

    if bundle.mode not in ALLOWED_MODES:
        raise LabError(f"unsupported relation mode: {bundle.mode!r}")
    if bundle.mode == BOUNDARY_PROFILE_MODE:
        if len(bundle.source_fingerprint) != 64:
            raise LabError("boundary source fingerprint must contain 64 hexadecimal characters")
        try:
            int(bundle.source_fingerprint, 16)
        except ValueError as exc:
            raise LabError("boundary source fingerprint is not hexadecimal") from exc
    else:
        _require_source_profile()
        if bundle.source_fingerprint != parameter_fingerprint(PARAMS):
            raise LabError("source parameter fingerprint does not match the current para.py")
    if bundle.degree != DEGREE:
        raise LabError(f"degree must be {DEGREE}")
    if bundle.modulus != BACKEND_Q:
        raise LabError(f"modulus must be the compiled backend q={BACKEND_Q}")
    if not 0 < bundle.r <= UINT32_MAX or not 0 < bundle.n <= UINT32_MAX:
        raise LabError("r and n must be positive uint32-compatible dimensions")
    witness_count = _checked_product(bundle.r, bundle.n, label="r*n")
    quadratic_count = _checked_product(bundle.r, bundle.r, label="r*r")
    if not math.isfinite(bundle.beta) or bundle.beta <= 0:
        raise LabError("beta must be finite and positive")
    generated_paper_schedule = _matches_generated_paper_schedule(bundle)
    secure_rank = (
        _compiled_backend_secure_rank()
        if bundle.mode == BOUNDARY_PROFILE_MODE
        else backend_secure_rank()
    )
    if (
        not generated_paper_schedule
        and min(bundle.kappa, bundle.kappa1, bundle.kappa2) < secure_rank
    ):
        raise LabError(
            f"kappa/kappa1/kappa2 must each be at least backend heuristic rank {secure_rank}"
        )
    if not all(2 <= base <= UINT32_MAX for base in (bundle.base1, bundle.base2, bundle.base3)):
        raise LabError("decomposition bases must be in [2, 2^32-1]")
    if bundle.jl_out != BACKEND_SOURCE_PARAMS.backend.jl_rows:
        raise LabError("JL_out must match the backend parameters in para1.py")
    if bundle.aggregation_rounds != EXPECTED_AGGREGATION_ROUNDS:
        raise LabError(
            f"aggregation_rounds must be {EXPECTED_AGGREGATION_ROUNDS} for backend q"
        )
    if not 1 <= bundle.recursions <= BACKEND_SOURCE_PARAMS.backend.max_recursions:
        raise LabError("recursions exceed the backend limit in para1.py")

    if bundle.recursions > 1:
        if generated_paper_schedule:
            first = BACKEND_SOURCE_PARAMS.paper_proof.schedule[0]
            expected = (
                first.base1,
                first.base2,
                first.base3,
                first.digits1,
                first.digits2,
                first.digits3,
            )
        else:
            expected = _recursive_decomposition_profile(
                bundle.modulus, bundle.n, bundle.r, bundle.beta
            )
        if (bundle.base1, bundle.base2, bundle.base3) != expected[:3]:
            raise LabError("recursive bundle bases do not match the Section 5.4 plan")
        digit1, digit2, digit3 = expected[3:]
    else:
        digit1 = _balanced_digit_count(bundle.modulus, bundle.base1)
        digit2 = _balanced_digit_count(bundle.modulus, bundle.base2)
        digit3 = _balanced_digit_count(bundle.modulus, bundle.base3)

    pair_count = bundle.r * (bundle.r + 1) // 2
    t_length = digit1 * bundle.r * bundle.kappa
    g_length = digit2 * pair_count
    h_length = digit3 * pair_count
    matrix_polynomial_limit = MAX_ARTIFACT_BYTES // (8 * bundle.degree)
    matrix_sizes = {
        "Ajtai A": bundle.n * bundle.kappa,
        "Ajtai B": t_length * bundle.kappa1,
        "Ajtai C": g_length * bundle.kappa1,
        "Ajtai D": h_length * bundle.kappa2,
    }
    for matrix_name, matrix_size in matrix_sizes.items():
        if matrix_size > matrix_polynomial_limit:
            raise LabError(
                f"{matrix_name} would allocate {matrix_size} polynomials, above the runner limit"
            )

    response_bound = EXPECTED_OPERATOR_NORM_BOUND * bundle.beta * math.sqrt(bundle.r)
    projection_bound = math.sqrt(bundle.jl_out // 2) * bundle.beta
    if (
        not math.isfinite(response_bound)
        or not math.isfinite(projection_bound)
        or response_bound >= 1 << 64
        or projection_bound >= 1 << 64
    ):
        raise LabError("beta-derived verifier bound does not fit uint64")
    if not 16 <= len(bundle.ajtai_seed) <= MAX_BYTE_STRING:
        raise LabError("Ajtai seed must contain between 16 bytes and 1 MiB")
    if not 16 <= len(bundle.oracle_seed) <= MAX_BYTE_STRING:
        raise LabError("oracle seed must contain between 16 bytes and 1 MiB")
    if len(bundle.equality_constraints) + len(bundle.const_zero_constraints) < 1:
        raise LabError("the relation must contain at least one constraint")
    if max(len(bundle.equality_constraints), len(bundle.const_zero_constraints)) > MAX_CONSTRAINTS:
        raise LabError("too many constraints")

    pair_count = bundle.r * (bundle.r + 1) // 2
    t_length = digit1 * bundle.r * bundle.kappa
    g_length = digit2 * pair_count
    h_length = digit3 * pair_count
    matrix_sizes = (
        bundle.n * bundle.kappa,
        t_length * bundle.kappa1,
        g_length * bundle.kappa1,
        h_length * bundle.kappa2,
    )
    if any(size > MAX_RUNTIME_POLYNOMIALS for size in matrix_sizes):
        raise LabError("an Ajtai matrix exceeds the runner memory safety limit")
    if sum(matrix_sizes) > MAX_RUNTIME_POLYNOMIALS:
        raise LabError("aggregate Ajtai matrices exceed the runner memory safety limit")
    jl_working_set = _jl_streaming_working_set_polynomials(
        witness_count, bundle.jl_out
    )
    if jl_working_set > MAX_RUNTIME_POLYNOMIALS:
        raise LabError(
            "streamed JL row/accumulator working set exceeds the runner memory safety limit"
        )
    if bundle.aggregation_rounds * bundle.jl_out > MAX_RUNTIME_POLYNOMIALS:
        raise LabError("aggregation_rounds*JL_out exceeds the runner memory safety limit")
    if math.sqrt(bundle.jl_out // 2) * bundle.beta < 1.0:
        raise LabError("beta is too small: the integer JL norm bound would be zero")

    _validate_poly_list(bundle.witness, witness_count, bundle, "witness")
    for poly_index, poly in enumerate(bundle.witness):
        for coefficient_index, coefficient in enumerate(poly):
            if abs(coefficient) > MAX_WITNESS_COEFFICIENT_ABS:
                raise LabError(
                    f"witness[{poly_index}][{coefficient_index}] has magnitude "
                    f"{abs(coefficient)}; runner requires < floor(sqrt(q))={math.isqrt(bundle.modulus)}"
                )
    for index, constraint in enumerate(bundle.equality_constraints):
        _validate_poly_list(constraint.a, quadratic_count, bundle, f"equality[{index}].a")
        _validate_poly_list(constraint.phi, witness_count, bundle, f"equality[{index}].phi")
        _validate_poly_list([constraint.b], 1, bundle, f"equality[{index}].b")
    for index, constraint in enumerate(bundle.const_zero_constraints):
        _validate_poly_list(constraint.a, quadratic_count, bundle, f"const_zero[{index}].a")
        _validate_poly_list(constraint.phi, witness_count, bundle, f"const_zero[{index}].phi")
        _check_coefficient(constraint.b, bundle.modulus, f"const_zero[{index}].b")

    squared_norm = sum(
        coefficient * coefficient for poly in bundle.witness for coefficient in poly
    )
    witness_norm = math.sqrt(squared_norm)
    if not witness_norm < bundle.beta:
        raise LabError(
            f"witness norm {witness_norm:.17g} must be strictly below beta {bundle.beta:.17g}"
        )

    if verify_relation:
        gram_cache: Dict[Tuple[int, int], List[int]] = {}
        for constraint_index, constraint in enumerate(bundle.equality_constraints):
            evaluation = _evaluate_constraint_polynomial(
                bundle, constraint.a, constraint.phi, gram_cache
            )
            _poly_add_in_place(evaluation, constraint.b, bundle.modulus)
            evaluation = [centered(value, bundle.modulus) for value in evaluation]
            for coefficient_index, coefficient in enumerate(evaluation):
                if coefficient != 0:
                    raise LabError(
                        "equality constraint "
                        f"{constraint_index} fails at coefficient {coefficient_index}: {coefficient}"
                    )

        for constraint_index, constraint in enumerate(bundle.const_zero_constraints):
            evaluation = _evaluate_constraint_polynomial(
                bundle, constraint.a, constraint.phi, gram_cache
            )
            constant_value = centered(evaluation[0] + constraint.b, bundle.modulus)
            if constant_value != 0:
                raise LabError(
                    f"const-zero constraint {constraint_index} fails: constant={constant_value}"
                )

    return {
        "witness_count": witness_count,
        "witness_squared_l2_norm": squared_norm,
        "witness_l2_norm": witness_norm,
        "backend_secure_rank_heuristic": secure_rank,
        "proof_modulus_matches_para": PARAMS.ring.q_proof == BACKEND_Q,
    }


class _ShakeRng:
    """Small deterministic SHAKE-based test-vector generator."""

    def __init__(self, seed: bytes):
        if not seed:
            raise LabError("synthetic seed must not be empty")
        self.seed = seed
        self.counter = 0

    def block(self, size: int = 32) -> bytes:
        counter = self.counter.to_bytes(8, "little")
        self.counter += 1
        return hashlib.shake_256(b"LNPLAB/SYNTHETIC/v1\0" + self.seed + counter).digest(size)

    def randbelow(self, upper: int) -> int:
        if not 0 < upper <= 1 << 64:
            raise LabError("randbelow range must be in [1,2^64]")
        cutoff = (1 << 64) - ((1 << 64) % upper)
        while True:
            candidate = int.from_bytes(self.block(8), "little")
            if candidate < cutoff:
                return candidate % upper


def _synthetic_constraint_terms(
    bundle: RelationBundle, *, const_zero: bool
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """Create sparse non-zero terms and return ``a, phi, evaluation``."""

    quadratic_count = bundle.r * bundle.r
    witness_count = bundle.r * bundle.n
    a = [_zero_poly(bundle.degree) for _ in range(quadratic_count)]
    phi = [_zero_poly(bundle.degree) for _ in range(witness_count)]

    if const_zero:
        row_i = 0
        row_j = bundle.r - 1
        a[row_i * bundle.r + row_j][1] = 1  # Multiplication by x tests wraparound.
        linear_row = bundle.r - 1
        linear_column = bundle.n - 1
        phi[linear_row * bundle.n + linear_column][0] = -2
        phi[linear_row * bundle.n + linear_column][bundle.degree - 1] = 1
    else:
        row_i = 0
        row_j = min(1, bundle.r - 1)
        a[row_i * bundle.r + row_j][0] = 1
        linear_row = min(1, bundle.r - 1)
        linear_column = min(1, bundle.n - 1)
        phi[linear_row * bundle.n + linear_column][0] = 1
        phi[linear_row * bundle.n + linear_column][1] = -1

    evaluation = _evaluate_constraint_polynomial(bundle, a, phi, {})
    return a, phi, evaluation


def create_synthetic_bundle(
    *,
    r: int = 2,
    n: int = 4,
    witness_bound: int = 2,
    seed: bytes = b"lnplab-synthetic-default-v1",
    recursions: int = 1,
) -> RelationBundle:
    """Create a deterministic satisfied relation for prover/verifier smoke tests."""

    _require_source_profile()
    if not 0 < r <= UINT32_MAX or not 0 < n <= UINT32_MAX:
        raise LabError("r and n must be positive uint32 values")
    witness_count = _checked_product(r, n, label="r*n")
    if not 0 <= witness_bound <= MAX_WITNESS_COEFFICIENT_ABS:
        raise LabError(
            "witness_bound must be non-negative and strictly below floor(sqrt(backend q))"
        )
    if recursions != 1:
        raise LabError(
            "synthetic direct-run bundles currently require recursions=1"
        )

    rng = _ShakeRng(seed)
    span = 2 * witness_bound + 1
    witness = [
        [rng.randbelow(span) - witness_bound for _ in range(DEGREE)]
        for _ in range(witness_count)
    ]
    squared_norm = sum(value * value for poly in witness for value in poly)
    beta = float(math.isqrt(squared_norm) + 2)
    base = max(
        2,
        math.ceil(math.sqrt(EXPECTED_OPERATOR_NORM_BOUND * beta * math.sqrt(r))) + 1,
    )
    if base > UINT32_MAX:
        raise LabError("derived decomposition base exceeds uint32")
    rank = backend_secure_rank()
    bundle = RelationBundle(
        mode="synthetic-principal-v1",
        source_fingerprint=parameter_fingerprint(PARAMS),
        degree=DEGREE,
        modulus=BACKEND_Q,
        r=r,
        n=n,
        beta=beta,
        kappa=rank,
        kappa1=rank,
        kappa2=rank,
        base1=base,
        base2=base,
        base3=base,
        jl_out=BACKEND_SOURCE_PARAMS.backend.jl_rows,
        aggregation_rounds=EXPECTED_AGGREGATION_ROUNDS,
        recursions=recursions,
        ajtai_seed=hashlib.shake_256(b"LNPLAB/AJTAI/v1\0" + seed).digest(32),
        oracle_seed=hashlib.shake_256(b"LNPLAB/ORACLE/v1\0" + seed).digest(32),
        witness=witness,
        equality_constraints=[],
        const_zero_constraints=[],
    )

    eq_a, eq_phi, eq_evaluation = _synthetic_constraint_terms(bundle, const_zero=False)
    eq_b = [centered(-value, BACKEND_Q) for value in eq_evaluation]
    bundle.equality_constraints.append(EqualityConstraint(eq_a, eq_phi, eq_b))

    cz_a, cz_phi, cz_evaluation = _synthetic_constraint_terms(bundle, const_zero=True)
    cz_b = centered(-cz_evaluation[0], BACKEND_Q)
    bundle.const_zero_constraints.append(ConstZeroConstraint(cz_a, cz_phi, cz_b))
    validate_bundle(bundle)
    return bundle


class _Writer:
    def __init__(self) -> None:
        self.data = bytearray()

    def raw(self, value: bytes) -> None:
        self.data.extend(value)

    def u32(self, value: int) -> None:
        if not 0 <= value <= UINT32_MAX:
            raise LabError(f"u32 value out of range: {value}")
        self.data.extend(struct.pack("<I", value))

    def u64(self, value: int) -> None:
        if not 0 <= value < 1 << 64:
            raise LabError(f"u64 value out of range: {value}")
        self.data.extend(struct.pack("<Q", value))

    def i64(self, value: int) -> None:
        self.data.extend(struct.pack("<q", value))

    def f64(self, value: float) -> None:
        self.data.extend(struct.pack("<d", value))

    def lp(self, value: bytes) -> None:
        if len(value) > MAX_BYTE_STRING:
            raise LabError("length-prefixed byte string is too large")
        self.u64(len(value))
        self.raw(value)

    def poly(self, poly: Sequence[int], degree: int) -> None:
        if len(poly) != degree:
            raise LabError("cannot serialize polynomial with the wrong degree")
        self.data.extend(struct.pack(f"<{degree}q", *poly))


def bundle_to_bytes(bundle: RelationBundle) -> bytes:
    """Serialize a validated bundle and bind its public section with SHA3-256."""

    validate_bundle(bundle)
    writer = _Writer()
    writer.raw(MAGIC)
    writer.u32(FORMAT_VERSION)
    writer.u32(bundle.degree)
    writer.u64(bundle.modulus)
    writer.u64(bundle.r)
    writer.u64(bundle.n)
    writer.f64(bundle.beta)
    writer.u64(bundle.kappa)
    writer.u64(bundle.kappa1)
    writer.u64(bundle.kappa2)
    writer.u32(bundle.base1)
    writer.u32(bundle.base2)
    writer.u32(bundle.base3)
    writer.u64(bundle.jl_out)
    writer.u64(bundle.aggregation_rounds)
    writer.u64(bundle.recursions)
    writer.lp(bundle.mode.encode("utf-8"))
    writer.lp(bundle.source_fingerprint.encode("ascii"))
    writer.lp(bundle.ajtai_seed)
    writer.lp(bundle.oracle_seed)
    writer.u64(len(bundle.equality_constraints))
    for constraint in bundle.equality_constraints:
        for poly in constraint.a:
            writer.poly(poly, bundle.degree)
        for poly in constraint.phi:
            writer.poly(poly, bundle.degree)
        writer.poly(constraint.b, bundle.degree)
    writer.u64(len(bundle.const_zero_constraints))
    for constraint in bundle.const_zero_constraints:
        for poly in constraint.a:
            writer.poly(poly, bundle.degree)
        for poly in constraint.phi:
            writer.poly(poly, bundle.degree)
        writer.i64(constraint.b)

    public_digest = hashlib.sha3_256(writer.data).digest()
    writer.raw(public_digest)
    writer.u64(len(bundle.witness))
    for poly in bundle.witness:
        writer.poly(poly, bundle.degree)
    if len(writer.data) > MAX_ARTIFACT_BYTES:
        raise LabError("serialized artifact exceeds the configured artifact limit")
    return bytes(writer.data)


class _Reader:
    def __init__(self, data: bytes):
        if len(data) > MAX_ARTIFACT_BYTES:
            raise LabError("artifact exceeds the configured artifact limit")
        self.data = data
        self.offset = 0

    def raw(self, length: int, label: str) -> bytes:
        if length < 0 or self.offset + length > len(self.data):
            raise LabError(f"truncated artifact while reading {label}")
        result = self.data[self.offset : self.offset + length]
        self.offset += length
        return result

    def _unpack(self, fmt: str, label: str) -> Any:
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.raw(size, label))[0]

    def u32(self, label: str) -> int:
        return self._unpack("<I", label)

    def u64(self, label: str) -> int:
        return self._unpack("<Q", label)

    def i64(self, label: str) -> int:
        return self._unpack("<q", label)

    def f64(self, label: str) -> float:
        return self._unpack("<d", label)

    def lp(self, label: str) -> bytes:
        length = self.u64(f"{label}.length")
        if length > MAX_BYTE_STRING:
            raise LabError(f"{label}: length exceeds safety limit")
        return self.raw(length, label)

    def poly(self, degree: int, modulus: int, label: str) -> List[int]:
        raw = self.raw(8 * degree, label)
        values = list(struct.unpack(f"<{degree}q", raw))
        for index, value in enumerate(values):
            if not -(modulus // 2) <= value <= modulus // 2:
                raise LabError(f"{label}[{index}]: non-canonical coefficient")
        return values


def bundle_from_bytes(data: bytes, *, verify_relation: bool = True) -> RelationBundle:
    reader = _Reader(data)
    if reader.raw(len(MAGIC), "magic") != MAGIC:
        raise LabError("invalid .lab magic")
    version = reader.u32("version")
    if version != FORMAT_VERSION:
        raise LabError(f"unsupported .lab version {version}")
    degree = reader.u32("degree")
    modulus = reader.u64("modulus")
    r = reader.u64("r")
    n = reader.u64("n")
    beta = reader.f64("beta")
    kappa = reader.u64("kappa")
    kappa1 = reader.u64("kappa1")
    kappa2 = reader.u64("kappa2")
    base1 = reader.u32("base1")
    base2 = reader.u32("base2")
    base3 = reader.u32("base3")
    jl_out = reader.u64("JL_out")
    aggregation_rounds = reader.u64("aggregation_rounds")
    recursions = reader.u64("recursions")

    try:
        mode = reader.lp("mode").decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LabError("mode is not valid UTF-8") from exc
    try:
        source_fingerprint = reader.lp("source_fingerprint").decode("ascii")
    except UnicodeDecodeError as exc:
        raise LabError("source fingerprint is not ASCII") from exc
    ajtai_seed = reader.lp("ajtai_seed")
    oracle_seed = reader.lp("oracle_seed")

    if degree != DEGREE or modulus != BACKEND_Q:
        raise LabError("artifact degree/modulus does not match the compiled backend")
    if not 0 < r <= UINT32_MAX or not 0 < n <= UINT32_MAX:
        raise LabError("artifact dimensions are invalid")
    witness_count_expected = _checked_product(r, n, label="r*n")
    quadratic_count = _checked_product(r, r, label="r*r")

    equality_count = reader.u64("equality_count")
    if equality_count > MAX_CONSTRAINTS:
        raise LabError("equality constraint count exceeds safety limit")
    equality_constraints: List[EqualityConstraint] = []
    for constraint_index in range(equality_count):
        a = [
            reader.poly(degree, modulus, f"equality[{constraint_index}].a[{index}]")
            for index in range(quadratic_count)
        ]
        phi = [
            reader.poly(degree, modulus, f"equality[{constraint_index}].phi[{index}]")
            for index in range(witness_count_expected)
        ]
        b = reader.poly(degree, modulus, f"equality[{constraint_index}].b")
        equality_constraints.append(EqualityConstraint(a, phi, b))

    const_zero_count = reader.u64("const_zero_count")
    if const_zero_count > MAX_CONSTRAINTS:
        raise LabError("const-zero constraint count exceeds safety limit")
    const_zero_constraints: List[ConstZeroConstraint] = []
    for constraint_index in range(const_zero_count):
        a = [
            reader.poly(degree, modulus, f"const_zero[{constraint_index}].a[{index}]")
            for index in range(quadratic_count)
        ]
        phi = [
            reader.poly(degree, modulus, f"const_zero[{constraint_index}].phi[{index}]")
            for index in range(witness_count_expected)
        ]
        b = reader.i64(f"const_zero[{constraint_index}].b")
        if not -(modulus // 2) <= b <= modulus // 2:
            raise LabError(f"const_zero[{constraint_index}].b is non-canonical")
        const_zero_constraints.append(ConstZeroConstraint(a, phi, b))

    public_end = reader.offset
    expected_digest = hashlib.sha3_256(data[:public_end]).digest()
    encoded_digest = reader.raw(32, "public SHA3-256 digest")
    if not hmac.compare_digest(expected_digest, encoded_digest):
        raise LabError("public section SHA3-256 digest mismatch")

    witness_count = reader.u64("witness_count")
    if witness_count != witness_count_expected:
        raise LabError(
            f"witness_count={witness_count}, expected r*n={witness_count_expected}"
        )
    witness = [
        reader.poly(degree, modulus, f"witness[{index}]")
        for index in range(witness_count)
    ]
    if reader.offset != len(data):
        raise LabError(f"unexpected {len(data) - reader.offset} trailing bytes")

    bundle = RelationBundle(
        mode=mode,
        source_fingerprint=source_fingerprint,
        degree=degree,
        modulus=modulus,
        r=r,
        n=n,
        beta=beta,
        kappa=kappa,
        kappa1=kappa1,
        kappa2=kappa2,
        base1=base1,
        base2=base2,
        base3=base3,
        jl_out=jl_out,
        aggregation_rounds=aggregation_rounds,
        recursions=recursions,
        ajtai_seed=ajtai_seed,
        oracle_seed=oracle_seed,
        witness=witness,
        equality_constraints=equality_constraints,
        const_zero_constraints=const_zero_constraints,
        public_digest=encoded_digest,
    )
    validate_bundle(bundle, verify_relation=verify_relation)
    return bundle


def _reshape_polys(
    value: Any,
    rows: int,
    columns: int,
    modulus: int,
    degree: int,
    label: str,
) -> List[List[int]]:
    if not isinstance(value, list) or len(value) != rows:
        raise LabError(f"{label}: expected {rows} rows")
    result: List[List[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            raise LabError(f"{label}[{row_index}]: expected {columns} columns")
        for column_index, poly in enumerate(row):
            result.append(
                _parse_poly(
                    poly, modulus, degree, f"{label}[{row_index}][{column_index}]"
                )
            )
    return result


def bundle_from_json_object(document: Any) -> RelationBundle:
    root = _expect_object(document, "root")
    _expect_keys(
        root,
        (
            "schema",
            "source_parameter_fingerprint",
            "backend",
            "parameters",
            "seeds",
            "witness",
            "equality_constraints",
            "const_zero_constraints",
        ),
        (),
        "root",
    )
    if root["schema"] != JSON_SCHEMA_NAME:
        raise LabError(f"schema must be {JSON_SCHEMA_NAME!r}")
    source_fingerprint = root["source_parameter_fingerprint"]
    if not isinstance(source_fingerprint, str):
        raise LabError("source_parameter_fingerprint must be a string")

    backend = _expect_object(root["backend"], "backend")
    _expect_keys(backend, ("degree", "modulus"), (), "backend")
    degree = _parse_integer(backend["degree"], "backend.degree")
    modulus = _parse_integer(backend["modulus"], "backend.modulus")

    parameters = _expect_object(root["parameters"], "parameters")
    parameter_keys = (
        "r",
        "n",
        "beta",
        "kappa",
        "kappa1",
        "kappa2",
        "base1",
        "base2",
        "base3",
        "JL_out",
        "aggregation_rounds",
        "recursions",
    )
    _expect_keys(parameters, parameter_keys, (), "parameters")
    r = _parse_integer(parameters["r"], "parameters.r")
    n = _parse_integer(parameters["n"], "parameters.n")
    try:
        beta = float(parameters["beta"])
    except (TypeError, ValueError) as exc:
        raise LabError("parameters.beta must be a finite number") from exc

    seeds = _expect_object(root["seeds"], "seeds")
    _expect_keys(seeds, ("ajtai_hex", "oracle_hex"), (), "seeds")
    ajtai_seed = _parse_hex(seeds["ajtai_hex"], "seeds.ajtai_hex")
    oracle_seed = _parse_hex(seeds["oracle_hex"], "seeds.oracle_hex")

    witness = _reshape_polys(root["witness"], r, n, modulus, degree, "witness")
    equality_json = root["equality_constraints"]
    if not isinstance(equality_json, list):
        raise LabError("equality_constraints must be a list")
    equality_constraints: List[EqualityConstraint] = []
    for index, raw_constraint in enumerate(equality_json):
        constraint = _expect_object(raw_constraint, f"equality_constraints[{index}]")
        _expect_keys(constraint, ("a", "phi", "b"), (), f"equality_constraints[{index}]")
        equality_constraints.append(
            EqualityConstraint(
                _reshape_polys(
                    constraint["a"], r, r, modulus, degree, f"equality_constraints[{index}].a"
                ),
                _reshape_polys(
                    constraint["phi"], r, n, modulus, degree, f"equality_constraints[{index}].phi"
                ),
                _parse_poly(constraint["b"], modulus, degree, f"equality_constraints[{index}].b"),
            )
        )

    const_zero_json = root["const_zero_constraints"]
    if not isinstance(const_zero_json, list):
        raise LabError("const_zero_constraints must be a list")
    const_zero_constraints: List[ConstZeroConstraint] = []
    for index, raw_constraint in enumerate(const_zero_json):
        constraint = _expect_object(raw_constraint, f"const_zero_constraints[{index}]")
        _expect_keys(constraint, ("a", "phi", "b"), (), f"const_zero_constraints[{index}]")
        const_zero_constraints.append(
            ConstZeroConstraint(
                _reshape_polys(
                    constraint["a"], r, r, modulus, degree, f"const_zero_constraints[{index}].a"
                ),
                _reshape_polys(
                    constraint["phi"], r, n, modulus, degree, f"const_zero_constraints[{index}].phi"
                ),
                _check_coefficient(constraint["b"], modulus, f"const_zero_constraints[{index}].b"),
            )
        )

    bundle = RelationBundle(
        mode="json-principal-v1",
        source_fingerprint=source_fingerprint,
        degree=degree,
        modulus=modulus,
        r=r,
        n=n,
        beta=beta,
        kappa=_parse_integer(parameters["kappa"], "parameters.kappa"),
        kappa1=_parse_integer(parameters["kappa1"], "parameters.kappa1"),
        kappa2=_parse_integer(parameters["kappa2"], "parameters.kappa2"),
        base1=_parse_integer(parameters["base1"], "parameters.base1"),
        base2=_parse_integer(parameters["base2"], "parameters.base2"),
        base3=_parse_integer(parameters["base3"], "parameters.base3"),
        jl_out=_parse_integer(parameters["JL_out"], "parameters.JL_out"),
        aggregation_rounds=_parse_integer(
            parameters["aggregation_rounds"], "parameters.aggregation_rounds"
        ),
        recursions=_parse_integer(parameters["recursions"], "parameters.recursions"),
        ajtai_seed=ajtai_seed,
        oracle_seed=oracle_seed,
        witness=witness,
        equality_constraints=equality_constraints,
        const_zero_constraints=const_zero_constraints,
    )
    validate_bundle(bundle)
    return bundle


def _nest_polys(polys: Sequence[Sequence[int]], rows: int, columns: int) -> List[List[List[int]]]:
    return [
        [list(polys[row * columns + column]) for column in range(columns)]
        for row in range(rows)
    ]


def bundle_to_json_object(bundle: RelationBundle) -> Dict[str, Any]:
    validate_bundle(bundle)
    return {
        "schema": JSON_SCHEMA_NAME,
        "source_parameter_fingerprint": bundle.source_fingerprint,
        "backend": {"degree": bundle.degree, "modulus": str(bundle.modulus)},
        "parameters": {
            "r": bundle.r,
            "n": bundle.n,
            "beta": bundle.beta,
            "kappa": bundle.kappa,
            "kappa1": bundle.kappa1,
            "kappa2": bundle.kappa2,
            "base1": bundle.base1,
            "base2": bundle.base2,
            "base3": bundle.base3,
            "JL_out": bundle.jl_out,
            "aggregation_rounds": bundle.aggregation_rounds,
            "recursions": bundle.recursions,
        },
        "seeds": {
            "ajtai_hex": bundle.ajtai_seed.hex(),
            "oracle_hex": bundle.oracle_seed.hex(),
        },
        "witness": _nest_polys(bundle.witness, bundle.r, bundle.n),
        "equality_constraints": [
            {
                "a": _nest_polys(constraint.a, bundle.r, bundle.r),
                "phi": _nest_polys(constraint.phi, bundle.r, bundle.n),
                "b": list(constraint.b),
            }
            for constraint in bundle.equality_constraints
        ],
        "const_zero_constraints": [
            {
                "a": _nest_polys(constraint.a, bundle.r, bundle.r),
                "phi": _nest_polys(constraint.phi, bundle.r, bundle.n),
                "b": constraint.b,
            }
            for constraint in bundle.const_zero_constraints
        ],
    }


def read_bundle(path: Path, *, verify_relation: bool = True) -> Tuple[RelationBundle, int]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise LabError(f"cannot stat {path}: {exc}") from exc
    if size > MAX_ARTIFACT_BYTES:
        raise LabError("artifact exceeds the configured artifact limit")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise LabError(f"cannot read {path}: {exc}") from exc
    return bundle_from_bytes(data, verify_relation=verify_relation), len(data)


def _atomic_write(path: Path, data: bytes, *, force: bool) -> None:
    path = path.resolve()
    if path.exists() and not force:
        raise LabError(f"refusing to overwrite existing file {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary_name = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _bundle_summary(bundle: RelationBundle, artifact_bytes: Optional[int] = None) -> Dict[str, Any]:
    validation = validate_bundle(bundle)
    digest = bundle.public_digest
    if digest is None:
        encoded = bundle_to_bytes(bundle)
        parsed = bundle_from_bytes(encoded)
        digest = parsed.public_digest
    modulus_warning = (
        " backend q differs from para.py q_proof, so it is not the same "
        "proof-modulus security instantiation."
        if not validation["proof_modulus_matches_para"]
        else ""
    )
    return {
        "format": "LNPLAB01/v1",
        "mode": bundle.mode,
        "artifact_bytes": artifact_bytes,
        "public_statement_sha3_256": digest.hex() if digest else None,
        "source_parameter_fingerprint": bundle.source_fingerprint,
        "degree": bundle.degree,
        "backend_modulus": bundle.modulus,
        "para_q_proof": PARAMS.ring.q_proof,
        "proof_modulus_matches_para": validation["proof_modulus_matches_para"],
        "exact_nibs_relation": False,
        "embedded_tex_profile": bundle.mode == BOUNDARY_PROFILE_MODE,
        "r": bundle.r,
        "n": bundle.n,
        "witness_polynomials": validation["witness_count"],
        "witness_l2_norm": validation["witness_l2_norm"],
        "beta": bundle.beta,
        "equality_constraints": len(bundle.equality_constraints),
        "const_zero_constraints": len(bundle.const_zero_constraints),
        "kappa": [bundle.kappa, bundle.kappa1, bundle.kappa2],
        "backend_secure_rank_heuristic": validation["backend_secure_rank_heuristic"],
        "bases": [bundle.base1, bundle.base2, bundle.base3],
        "JL_out": bundle.jl_out,
        "aggregation_rounds": bundle.aggregation_rounds,
        "recursions": bundle.recursions,
        "recursion_status": (
            "research-recursive-core" if bundle.recursions > 1 else "base-case"
        ),
        "warning": (
            "SECRET WITNESS IS STORED IN PLAINTEXT: do not share this .lab/JSON bundle. "
            + (
                "It is the reduced executable LNPLab boundary profile with embedded TeX, "
                "not the full 7.57M-row NIBS relation or a proof."
                if bundle.mode == BOUNDARY_PROFILE_MODE
                else "It is a backend smoke relation, not a proof or the exact NIBS relation."
            )
            + modulus_warning
        ),
    }


def _print_summary(summary: Mapping[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"LaBRADOR artifact: {summary['mode']}")
    print(f"  r x n: {summary['r']} x {summary['n']} ({summary['witness_polynomials']} witness polynomials)")
    print(
        "  constraints: "
        f"{summary['equality_constraints']} equality, {summary['const_zero_constraints']} const-zero"
    )
    print(f"  ||S||_2 < beta: {summary['witness_l2_norm']:.6f} < {summary['beta']:.6f}")
    print(f"  public SHA3-256: {summary['public_statement_sha3_256']}")
    if summary.get("artifact_bytes") is not None:
        print(f"  artifact bytes: {summary['artifact_bytes']}")
    print(f"  WARNING: {summary['warning']}")


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_ARTIFACT_BYTES:
            raise LabError("JSON input exceeds the configured artifact limit")
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LabError(f"cannot read {path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise LabError(f"{path} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise LabError(f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc


def _write_json(path: Path, document: Any, *, force: bool) -> None:
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise LabError("JSON output exceeds the configured artifact limit")
    _atomic_write(path, encoded, force=force)


def _build_cpu(repo_root: Path) -> None:
    jobs = str(max(1, os.cpu_count() or 1))
    commands = (
        (
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DRING=labradorq40",
            "-S",
            str(repo_root / "icicle"),
            "-B",
            str(repo_root / "build" / "icicle"),
        ),
        ("cmake", "--build", str(repo_root / "build" / "icicle"), "-j", jobs),
        (
            "cmake",
            "-DCMAKE_BUILD_TYPE=Release",
            "-S",
            str(repo_root / "src"),
            "-B",
            str(repo_root / "build" / "src"),
        ),
        (
            "cmake",
            "--build",
            str(repo_root / "build" / "src"),
            "--target",
            "lab_runner",
            "-j",
            jobs,
        ),
    )
    for command in commands:
        subprocess.run(command, cwd=repo_root, check=True)


def run_self_tests() -> None:
    _require_source_profile()
    x63 = [0] * DEGREE
    x63[DEGREE - 1] = 1
    x = [0] * DEGREE
    x[1] = 1
    product = negacyclic_mul(x63, x, BACKEND_Q)
    assert product[0] == -1 and not any(product[1:])

    bundle = create_synthetic_bundle(r=2, n=3, witness_bound=2, seed=b"self-test", recursions=1)

    # The first generated paper level no longer fails the obsolete full-JL
    # allocation check: a 64 MiB chunk holds one row, alongside one
    # accumulator. The full 256-row matrix remains deliberately unallocated.
    first = BACKEND_SOURCE_PARAMS.paper_proof.schedule[0]
    paper_witness_count = first.r * first.n
    assert (
        BACKEND_SOURCE_PARAMS.backend.jl_rows * paper_witness_count
        > MAX_RUNTIME_POLYNOMIALS
    )
    assert _jl_aggregation_chunk_rows(
        paper_witness_count, BACKEND_SOURCE_PARAMS.backend.jl_rows
    ) == 1
    assert _jl_streaming_working_set_polynomials(
        paper_witness_count, BACKEND_SOURCE_PARAMS.backend.jl_rows
    ) == 2 * paper_witness_count
    assert 2 * paper_witness_count < MAX_RUNTIME_POLYNOMIALS

    # Low ranks are accepted only for the exact generated seven-level
    # boundary fingerprint. Each nearby generic or malformed profile must
    # continue through the ordinary conservative rank floor.
    expected_beta = _generated_paper_initial_beta()
    expected_bases = (first.base1, first.base2, first.base3)
    expected_ranks = _generated_paper_first_ranks()
    assert expected_ranks == (10, 4, 4)
    paper_probe = replace(
        bundle,
        mode=BOUNDARY_PROFILE_MODE,
        r=first.r,
        n=first.n,
        beta=expected_beta,
        kappa=expected_ranks[0],
        kappa1=expected_ranks[1],
        kappa2=expected_ranks[2],
        base1=expected_bases[0],
        base2=expected_bases[1],
        base3=expected_bases[2],
        recursions=7,
    )
    assert _matches_generated_paper_schedule(paper_probe)
    assert not _matches_generated_paper_schedule(
        replace(paper_probe, mode="json-principal-v1")
    )
    assert not _matches_generated_paper_schedule(replace(paper_probe, recursions=6))
    assert not _matches_generated_paper_schedule(replace(paper_probe, n=first.n + 1))
    assert not _matches_generated_paper_schedule(replace(paper_probe, r=first.r + 1))
    assert not _matches_generated_paper_schedule(
        replace(paper_probe, kappa=expected_ranks[0] + 1)
    )
    assert not _matches_generated_paper_schedule(
        replace(paper_probe, base1=expected_bases[0] + 1)
    )
    assert not _matches_generated_paper_schedule(
        replace(
            paper_probe,
            beta=expected_beta + 32.0 * sys.float_info.epsilon * expected_beta,
        )
    )

    # Exercise the validation branch without allocating the million-polynomial
    # paper witness: the exact probe passes rank and streamed-JL checks, then
    # stops only when its intentionally tiny fixture witness is inspected.
    try:
        validate_bundle(paper_probe, verify_relation=False)
    except LabError as exc:
        assert str(exc).startswith("witness: expected ")
    else:
        raise AssertionError("paper probe unexpectedly had a full-size witness")

    malformed_paper_probe = replace(paper_probe, base1=expected_bases[0] + 1)
    try:
        validate_bundle(malformed_paper_probe, verify_relation=False)
    except LabError as exc:
        assert "rank" in str(exc)
    else:
        raise AssertionError("malformed paper profile obtained the low-rank exception")

    low_rank_generic = replace(
        bundle,
        kappa=_compiled_backend_secure_rank() - 1,
        kappa1=_compiled_backend_secure_rank() - 1,
        kappa2=_compiled_backend_secure_rank() - 1,
    )
    try:
        validate_bundle(low_rank_generic)
    except LabError as exc:
        assert "rank" in str(exc)
    else:
        raise AssertionError("generic low-rank relation was accepted")

    encoded = bundle_to_bytes(bundle)
    decoded = bundle_from_bytes(encoded)
    assert decoded.witness == bundle.witness
    assert decoded.public_digest is not None
    assert decoded.public_digest == hashlib.sha3_256(encoded[: encoded.index(decoded.public_digest)]).digest()

    # The public oracle context is part of the statement digest, preventing
    # callers from grinding it while keeping an apparently identical public
    # relation identifier.
    changed_oracle = replace(
        bundle, oracle_seed=b"different-public-oracle-context", public_digest=None
    )
    changed_decoded = bundle_from_bytes(bundle_to_bytes(changed_oracle))
    assert changed_decoded.public_digest != decoded.public_digest

    json_document = bundle_to_json_object(bundle)
    json_bundle = bundle_from_json_object(json_document)
    assert json_bundle.witness == bundle.witness
    assert bundle_to_bytes(json_bundle) != b""

    corrupted = bytearray(encoded)
    digest_offset = encoded.index(decoded.public_digest)
    corrupted[digest_offset - 1] ^= 1
    try:
        bundle_from_bytes(bytes(corrupted))
    except LabError:
        # A parser may reject a corrupted public coefficient as non-canonical
        # before it reaches the digest check.  Either rejection is correct.
        pass
    else:
        raise AssertionError("public corruption was not detected")

    try:
        bundle_from_bytes(encoded + b"x")
    except LabError as exc:
        assert "trailing" in str(exc)
    else:
        raise AssertionError("trailing data was not rejected")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, validate, and run concrete inputs for the local LaBRADOR backend."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate a deterministic synthetic .lab relation")
    generate.add_argument("--output", "-o", type=Path, required=True)
    generate.add_argument("--r", type=int, default=2)
    generate.add_argument("--n", type=int, default=4)
    generate.add_argument("--witness-bound", type=int, default=2)
    generate.add_argument("--seed", default="lnplab-synthetic-default-v1")
    generate.add_argument(
        "--recursions",
        type=int,
        choices=(1,),
        default=1,
        help="generate a base bundle; use labrador.py prepare/run to enable recursion",
    )
    generate.add_argument("--json-output", type=Path)
    generate.add_argument("--force", action="store_true")
    generate.add_argument("--json", action="store_true", help="print summary as JSON")

    pack = subparsers.add_parser("pack", help="validate JSON and pack it as a .lab relation")
    pack.add_argument("input", type=Path)
    pack.add_argument("--output", "-o", type=Path, required=True)
    pack.add_argument("--force", action="store_true")
    pack.add_argument("--json", action="store_true", help="print summary as JSON")

    inspect_parser = subparsers.add_parser("inspect", help="parse and exactly validate a .lab relation")
    inspect_parser.add_argument("input", type=Path)
    inspect_parser.add_argument("--json", action="store_true", help="print summary as JSON")

    run = subparsers.add_parser("run", help="validate a .lab relation and invoke the C++ runner")
    run.add_argument("input", type=Path)
    run.add_argument("--device", default="CPU")
    run.add_argument("--build", action="store_true", help="configure/build the CPU runner first")

    subparsers.add_parser("schema", help="print the JSON schema documentation")
    subparsers.add_parser("self-test", help="run deterministic codec/relation tests")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            bundle = create_synthetic_bundle(
                r=args.r,
                n=args.n,
                witness_bound=args.witness_bound,
                seed=args.seed.encode("utf-8"),
                recursions=args.recursions,
            )
            encoded = bundle_to_bytes(bundle)
            _atomic_write(args.output, encoded, force=args.force)
            parsed = bundle_from_bytes(encoded)
            if args.json_output is not None:
                _write_json(
                    args.json_output,
                    bundle_to_json_object(bundle),
                    force=args.force,
                )
            _print_summary(_bundle_summary(parsed, len(encoded)), as_json=args.json)
            return 0

        if args.command == "pack":
            bundle = bundle_from_json_object(_load_json(args.input))
            encoded = bundle_to_bytes(bundle)
            _atomic_write(args.output, encoded, force=args.force)
            parsed = bundle_from_bytes(encoded)
            _print_summary(_bundle_summary(parsed, len(encoded)), as_json=args.json)
            return 0

        if args.command == "inspect":
            bundle, artifact_bytes = read_bundle(args.input)
            _print_summary(_bundle_summary(bundle, artifact_bytes), as_json=args.json)
            return 0

        if args.command == "run":
            read_bundle(args.input)  # Refuse malformed/unsatisfied input before C++ allocation.
            repo_root = Path(__file__).resolve().parents[1]
            if args.build:
                if args.device.upper() != "CPU":
                    raise LabError("automatic --build currently supports only the CPU backend")
                _build_cpu(repo_root)
            runner = repo_root / "build" / "src" / "lab_runner"
            if not runner.is_file():
                raise LabError(f"runner not found at {runner}; use --build first")
            result = subprocess.run(
                (str(runner), args.device, str(args.input.resolve())),
                cwd=repo_root,
                check=False,
            )
            return result.returncode

        if args.command == "schema":
            print(__doc__.split("JSON input schema v1", 1)[1].strip())
            return 0

        if args.command == "self-test":
            run_self_tests()
            print("lab.py self-test: PASS")
            return 0

        raise AssertionError(f"unhandled command {args.command}")
    except (LabError, OSError, struct.error, subprocess.CalledProcessError) as exc:
        print(f"lab.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
