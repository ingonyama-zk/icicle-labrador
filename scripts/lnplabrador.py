#!/usr/bin/env python3
"""Compile the LNPLab boundary TeX into a runnable LaBRADOR test artifact.

The source relation is ``lnplab_labrador_input_relation.tex``.  Following
TRaccoon Appendix B.6, this tool keeps the three compressed flows ``x_A``,
``x_h`` and ``x_w`` outside the final response and proves their openings
together with ``f1``, ``f2``, ``f3`` and a native witness norm bound.

The full TeX relation needs about 7.57 million binary-R1CS rows and concrete
LNP transcript/witness values that are not present in the document.  The
current C++ frontend stores every principal constraint densely and caps input
at 1 GiB.  Consequently, ``generate`` emits a reduced *executable
conformance profile*: every kind of equation is represented and checked, but
each large message flow is collapsed to a few constant-polynomial bits.  The
original UTF-8 TeX bytes are embedded verbatim in the public oracle context,
covered by the bundle digest, and checked again by ``inspect``.

The artifact contains a plaintext witness.  It is a local prover input, not a
proof, not a production CRS, and not a concrete-security NIBS instance.

Typical usage::

    python3 scripts/lnplabrador.py audit
    python3 scripts/lnplabrador.py paper-plan --json
    python3 scripts/lnplabrador.py generate --force
    python3 scripts/lnplabrador.py inspect input1.lab
    python3 scripts/lnplabrador.py run input1.lab --device CPU --build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:  # Chạy được cả như script và import như package namespace.
    from . import lab
    from .lnplab import (
        conjugate,
        lnp_bound_from_measure,
        lnp_multiplier_measure,
        operator_norm,
        sample_labrador_candidate,
        sample_lnp_candidate,
    )
    from .para1 import PARAMS, SOURCES, Parameters
except ImportError:
    import lab
    from lnplab import (
        conjugate,
        lnp_bound_from_measure,
        lnp_multiplier_measure,
        operator_norm,
        sample_labrador_candidate,
        sample_lnp_candidate,
    )
    from para1 import PARAMS, SOURCES, Parameters


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_TEX = SCRIPT_DIR / "lnplab_labrador_input_relation.tex"
DEFAULT_OUTPUT = REPO_ROOT / "input1.lab"
DEFAULT_BACKEND_HEADER = REPO_ROOT / "src" / "lnplabrador_backend_params.h"
DEFAULT_C_BACKEND_HEADER = SCRIPT_DIR / "lnplabrador_backend_params_c.h"
MODE = lab.BOUNDARY_PROFILE_MODE
ORACLE_TEX_PREFIX = b"LNPLAB/TEX-SOURCE/v1\0"
PARAMETER_DIGEST_TAG = b"\0LNPLAB/PARA1-SHA256/v1\0"


class LNPLabError(ValueError):
    """A parameter, source, compilation, or artifact consistency error."""


class ReferenceRng:
    """Deterministic SHAKE256 counter-mode RNG for reference test vectors."""

    def __init__(self, seed: bytes, domain: bytes):
        if not seed or not domain:
            raise LNPLabError("reference sampler seed and domain must be non-empty")
        self.seed = bytes(seed)
        self.domain = bytes(domain)
        self.counter = 0

    def block(self, size: int) -> bytes:
        if size <= 0:
            raise LNPLabError("reference RNG block size must be positive")
        counter = self.counter.to_bytes(8, "little")
        self.counter += 1
        return hashlib.shake_256(
            b"LNPLAB/REFERENCE-RNG/v1\0"
            + len(self.domain).to_bytes(4, "little")
            + self.domain
            + len(self.seed).to_bytes(8, "little")
            + self.seed
            + counter
        ).digest(size)

    def randbelow(self, upper: int) -> int:
        if not 0 < upper <= 1 << 64:
            raise LNPLabError("randbelow upper bound must be in [1, 2^64]")
        cutoff = (1 << 64) - ((1 << 64) % upper)
        while True:
            candidate = int.from_bytes(self.block(8), "little")
            if candidate < cutoff:
                return candidate % upper


def _canonical_parameters(parameters: Parameters = PARAMS) -> bytes:
    return json.dumps(
        asdict(parameters), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def parameter_fingerprint(parameters: Parameters = PARAMS) -> str:
    return hashlib.sha256(_canonical_parameters(parameters)).hexdigest()


def source_fingerprint(tex_bytes: bytes, parameters: Parameters = PARAMS) -> str:
    payload = (
        b"LNPLAB/SOURCE-FINGERPRINT/v1\0"
        + hashlib.sha256(tex_bytes).digest()
        + hashlib.sha256(_canonical_parameters(parameters)).digest()
    )
    return hashlib.sha256(payload).hexdigest()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _is_prime_u64(value: int) -> bool:
    """Deterministic Miller--Rabin for an unsigned 64-bit candidate."""

    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value % prime == 0:
            return value == prime
    odd_part = value - 1
    power_of_two = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        power_of_two += 1
    for base in (2, 325, 9_375, 28_178, 450_775, 9_780_504, 1_795_265_022):
        witness = pow(base % value, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(power_of_two - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def validate_parameters(parameters: Parameters = PARAMS) -> None:
    errors: List[str] = []
    source = parameters.source_ring
    backend = parameters.backend
    lnp = parameters.lnp_challenge
    folding = parameters.labrador_challenge
    boundary = parameters.boundary
    full_r1cs = parameters.full_r1cs
    paper = parameters.paper_proof
    executable = parameters.executable

    if source.degree != 64:
        errors.append("source ring degree must be 64 for the current TeX")
    if source.modulus != (1 << 40) - 195:
        errors.append("source modulus must match q_pi=2^40-195 in the TeX")
    if source.coefficient_bits != source.modulus.bit_length():
        errors.append("source coefficient_bits must equal bit_length(q_pi)")
    if backend.degree <= 0 or backend.modulus <= 2:
        errors.append("backend degree and modulus must be positive")
    if backend.modulus.bit_length() != 40:
        errors.append("the q40 backend modulus must have exactly 40 bits")
    if not _is_prime_u64(backend.modulus):
        errors.append("backend modulus must be prime")
    cyclotomic_order = 2 * backend.degree
    if math.gcd(backend.modulus, cyclotomic_order) != 1:
        errors.append("backend modulus must be coprime to 2*degree")
    else:
        residue = 1
        multiplicative_order = None
        for exponent in range(1, cyclotomic_order + 1):
            residue = (residue * backend.modulus) % cyclotomic_order
            if residue == 1:
                multiplicative_order = exponent
                break
        if multiplicative_order != backend.degree // 2:
            errors.append(
                "backend modulus must make X^degree+1 split into exactly "
                "two irreducible degree/2 factors"
            )
    if backend.modulus != parameters.source_ring.modulus:
        errors.append("backend and source relation must use the same modulus")
    if backend.security_bits <= 0 or backend.jl_rows <= 0:
        errors.append("backend security_bits and jl_rows must be positive")
    if backend.recursions != 1:
        errors.append("this direct executable profile requires exactly one LaBRADOR execution")
    if backend.max_recursions < backend.recursions:
        errors.append("backend max_recursions must be at least recursions")
    if not math.isfinite(backend.root_hermite_delta) or backend.root_hermite_delta <= 1.0:
        errors.append("backend root_hermite_delta must be finite and greater than one")
    if backend.rank_override is not None and backend.rank_override <= 0:
        errors.append("backend rank_override must be positive when set")
    if backend.max_artifact_bytes <= 0 or backend.max_runtime_polynomials <= 0:
        errors.append("backend artifact/runtime safety limits must be positive")
    if not 0 < backend.max_proof_bytes <= backend.max_artifact_bytes:
        errors.append("backend max_proof_bytes must be positive and no larger than max_artifact_bytes")
    if backend.max_split_parts <= 0:
        errors.append("backend max_split_parts must be positive")
    if backend.extraction_slack_denominator <= 0:
        errors.append("backend extraction slack denominator must be positive")

    if lnp.coefficient_bound < 0 or lnp.eta <= 0:
        errors.append("LNP coefficient_bound must be non-negative and eta positive")
    if not _is_power_of_two(lnp.power):
        errors.append("LNP power must be a positive power of two")
    if source.degree % 2:
        errors.append("the conjugation-fixed LNP sampler requires an even ring degree")
    if (2 * lnp.coefficient_bound + 1) ** (source.degree // 2) <= 1 << backend.security_bits:
        errors.append("raw conjugation-fixed LNP challenge space is not larger than 2^128")

    weights = (
        folding.zero_coefficients
        + folding.unit_coefficients
        + folding.double_coefficients
    )
    if weights != backend.degree:
        errors.append("LaBRADOR challenge weights must sum to the ring degree")
    if (
        not math.isfinite(folding.operator_norm_bound)
        or folding.operator_norm_bound <= 0
        or not float(folding.operator_norm_bound).is_integer()
        or not folding.strict_operator_bound
    ):
        errors.append("the C++ backend requires a positive integral strict operator-norm bound")
    if folding.paper_average_sampling_attempts <= 0 or folding.paper_soundness_bits <= 0:
        errors.append("LaBRADOR paper sampling/soundness accounting must be positive")

    if sum(boundary.message_a_blocks) != boundary.message_a_ring_coordinates:
        errors.append("message_a_blocks do not sum to 2947 ring coordinates")
    if (
        boundary.compressing_msis_rank,
        boundary.message_a_ring_coordinates,
        boundary.message_h_ring_coordinates,
        boundary.message_w_ring_coordinates,
    ) != (2, 2947, 4, 6):
        errors.append("boundary dimensions no longer match the supplied TeX")

    if full_r1cs.constraint_capacity <= 0:
        errors.append("full_r1cs.constraint_capacity must be positive")
    if full_r1cs.variable_count is not None and full_r1cs.variable_count <= 0:
        errors.append("full_r1cs.variable_count must be positive when set")
    if paper.initial_beta_mode != "sqrt-modulus-binary-r1cs":
        errors.append("unsupported paper_proof.initial_beta_mode")
    if min(
        paper.max_rank_search,
        paper.r1cs_reduction_commitment_rank,
        paper.constant_term_mask_commitments,
        paper.lnp_projection_response_bytes,
    ) <= 0:
        errors.append("paper proof rank/message parameters must be positive")
    if not paper.schedule:
        errors.append("paper_proof.schedule must not be empty")
    for expected_level, row in enumerate(paper.schedule, start=1):
        if row.level != expected_level or min(row.n, row.r) <= 0:
            errors.append("paper schedule levels must be consecutive with positive n,r")
            break
        is_last = expected_level == len(paper.schedule)
        if is_last:
            if row.nu_to_next is not None or row.mu_to_next is not None:
                errors.append("last paper schedule row must not have a split")
            continue
        if row.nu_to_next is None or row.mu_to_next is None:
            errors.append("each non-final paper schedule row must have nu,mu")
            break
        next_r = paper.schedule[expected_level].r
        derived_r = (
            row.nu_to_next + row.mu_to_next
            if row.tail_transition
            else 2 * row.nu_to_next + row.mu_to_next
        )
        if derived_r != next_r:
            errors.append(
                f"paper level {row.level} split derives r'={derived_r}, expected {next_r}"
            )
            break

    if executable.bit_width <= 0:
        errors.append("executable bit_width must be positive")
    if len(executable.message_names) != len(executable.message_values):
        errors.append("message_names and message_values must have the same length")
    if executable.message_names != ("t_A", "h", "w_A", "w_u", "t_Q", "v_Q"):
        errors.append("the executable compiler expects the six documented message names")
    maximum_message = 1 << executable.bit_width
    if any(value < 0 or value >= maximum_message for value in executable.message_values):
        errors.append("each executable message value must fit its unsigned bit decomposition")
    if executable.response_names != ("z_A_1", "z_A_2", "z_u"):
        errors.append("the executable compiler expects response names z_A_1,z_A_2,z_u")
    if len(executable.response_values) != 3:
        errors.append("the executable compiler requires three response values")
    elif executable.response_values[0] == 0 or executable.response_values[2] == 0:
        errors.append("z_A_1 and z_u must be non-zero so public coefficients can be derived")
    if not math.isfinite(executable.beta_margin) or executable.beta_margin <= 0:
        errors.append("beta_margin must be finite and positive")
    if not 2 <= executable.decomposition_base <= lab.UINT32_MAX:
        errors.append("decomposition_base must fit uint32 and be at least 2")

    if errors:
        raise LNPLabError("invalid para1.py: " + "; ".join(errors))


def _tex_counts(parameters: Parameters = PARAMS) -> Dict[str, int]:
    degree = parameters.source_ring.degree
    width = parameters.source_ring.coefficient_bits
    rank = parameters.boundary.compressing_msis_rank
    dims = {
        "A": parameters.boundary.message_a_ring_coordinates,
        "h": parameters.boundary.message_h_ring_coordinates,
        "w": parameters.boundary.message_w_ring_coordinates,
    }
    return {
        name: width * degree * dimension + rank * degree
        for name, dimension in dims.items()
    }


def _configured_base_msis_rank(parameters: Parameters = PARAMS) -> int:
    """Return the direct-run MSIS rank selected by ``para1.py``."""

    backend = parameters.backend
    if backend.rank_override is not None:
        return backend.rank_override
    log_q = math.log2(backend.modulus)
    return math.ceil(
        (log_q - 1.0) ** 2
        / (
            4.0
            * math.log2(backend.root_hermite_delta)
            * log_q
            * backend.degree
        )
    )


def backend_header_text(parameters: Parameters = PARAMS) -> str:
    """Return the checked-in C++ constants generated from ``para1.py``."""

    validate_parameters(parameters)
    backend = parameters.backend
    challenge = parameters.labrador_challenge
    aggregation_rounds = math.ceil(backend.security_bits / math.log2(backend.modulus))
    base_msis_rank = _configured_base_msis_rank(parameters)
    tau = challenge.unit_coefficients + 4 * challenge.double_coefficients
    beta = math.sqrt(backend.modulus)
    schedule_rows: List[str] = []
    schedule = parameters.paper_proof.schedule
    for index, row in enumerate(schedule):
        selected = _select_paper_ranks(
            n=row.n,
            r=row.r,
            beta=beta,
            recursed_target=index < len(schedule) - 1,
            parameters=parameters,
        )
        if index < len(schedule) - 1:
            selected = _fit_level_to_configured_transition(
                selected=selected,
                row=row,
                next_row=schedule[index + 1],
                beta=beta,
                parameters=parameters,
            )
        nu = 0 if row.nu_to_next is None else row.nu_to_next
        mu = 0 if row.mu_to_next is None else row.mu_to_next
        schedule_rows.append(
            "  PaperScheduleEntry{"
            f"{row.n}U, {row.r}U, "
            f"{selected['ranks']['kappa']}U, "
            f"{selected['ranks']['kappa1']}U, "
            f"{selected['ranks']['kappa2']}U, "
            f"{nu}U, {mu}U, "
            f"{'true' if row.tail_transition else 'false'}"
            "}"
        )
        beta = selected["target_norm"]["beta_prime"]
    schedule_initializer = ",\n".join(schedule_rows)
    return f"""// Generated by scripts/lnplabrador.py sync-backend from scripts/para1.py.
// Do not edit this file directly; edit para1.py and run sync-backend.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace icicle::labrador::backend_config {{

inline constexpr uint32_t RING_DEGREE = {backend.degree}U;
inline constexpr uint64_t RING_MODULUS = {backend.modulus}ULL;
inline constexpr uint64_t RING_MODULUS_HALF = {backend.modulus // 2}ULL;
inline constexpr uint64_t RING_MODULUS_SQRT_FLOOR = {math.isqrt(backend.modulus)}ULL;

inline constexpr size_t SECURITY_BITS = {backend.security_bits}U;
inline constexpr size_t JL_ROWS = {backend.jl_rows}U;
inline constexpr size_t AGGREGATION_ROUNDS = {aggregation_rounds}U;
inline constexpr size_t DEFAULT_RECURSIONS = {backend.recursions}U;
inline constexpr size_t MAX_RECURSIONS = {backend.max_recursions}U;
inline constexpr long double ROOT_HERMITE_DELTA = {backend.root_hermite_delta!r}L;
inline constexpr size_t BASE_MSIS_RANK = {base_msis_rank}U;

inline constexpr uint32_t CHALLENGE_ZERO_COEFFICIENTS = {challenge.zero_coefficients}U;
inline constexpr uint32_t CHALLENGE_UNIT_COEFFICIENTS = {challenge.unit_coefficients}U;
inline constexpr uint32_t CHALLENGE_DOUBLE_COEFFICIENTS = {challenge.double_coefficients}U;
inline constexpr uint64_t CHALLENGE_OPERATOR_NORM_BOUND = {int(challenge.operator_norm_bound)}ULL;
inline constexpr uint64_t CHALLENGE_TAU = {tau}ULL;

inline constexpr uint64_t MAX_ARTIFACT_BYTES = {backend.max_artifact_bytes}ULL;
inline constexpr uint64_t MAX_PROOF_BYTES = {backend.max_proof_bytes}ULL;
inline constexpr size_t MAX_RUNTIME_POLYNOMIALS = {backend.max_runtime_polynomials}U;
inline constexpr size_t MAX_SPLIT_PARTS = {backend.max_split_parts}U;
inline constexpr size_t EXTRACTION_SLACK_DENOMINATOR = {backend.extraction_slack_denominator}U;
inline constexpr size_t MODULAR_JL_DENOMINATOR = {challenge.paper_soundness_bits}U;

struct PaperScheduleEntry {{
  size_t n;
  size_t r;
  size_t kappa;
  size_t kappa1;
  size_t kappa2;
  size_t nu_to_next;
  size_t mu_to_next;
  bool section_5_6_tail;
}};

inline constexpr std::array<PaperScheduleEntry, {len(schedule)}> PAPER_SCHEDULE{{{{
{schedule_initializer}
}}}};

inline constexpr char PARAMETER_SHA256[] = "{parameter_fingerprint(parameters)}";

}} // namespace icicle::labrador::backend_config
"""


def backend_c_header_text(parameters: Parameters = PARAMS) -> str:
    """Return C11 constants generated from the Python parameter sources."""

    validate_parameters(parameters)
    backend = parameters.backend
    challenge = parameters.labrador_challenge
    aggregation_rounds = math.ceil(
        backend.security_bits / math.log2(backend.modulus)
    )
    base_msis_rank = _configured_base_msis_rank(parameters)
    tau = challenge.unit_coefficients + 4 * challenge.double_coefficients
    source_parameter_fingerprint = lab.parameter_fingerprint(lab.PARAMS)
    return f"""/* Generated by scripts/lnplabrador.py sync-backend.
 * Backend constants come from para1.py; the source-profile fingerprint comes
 * from para.py.  Do not edit this file directly; run sync-backend instead. */
#ifndef LNPLABRADOR_BACKEND_PARAMS_C_H
#define LNPLABRADOR_BACKEND_PARAMS_C_H

#include <stdint.h>

#define LNPLAB_BACKEND_RING_DEGREE UINT32_C({backend.degree})
#define LNPLAB_BACKEND_RING_MODULUS UINT64_C({backend.modulus})
#define LNPLAB_BACKEND_RING_MODULUS_HALF UINT64_C({backend.modulus // 2})
#define LNPLAB_BACKEND_RING_MODULUS_SQRT_FLOOR UINT64_C({math.isqrt(backend.modulus)})
#define LNPLAB_BACKEND_BASE_MSIS_RANK UINT64_C({base_msis_rank})
#define LNPLAB_BACKEND_JL_ROWS UINT64_C({backend.jl_rows})
#define LNPLAB_BACKEND_AGGREGATION_ROUNDS UINT64_C({aggregation_rounds})
#define LNPLAB_BACKEND_MAX_RECURSIONS UINT64_C({backend.max_recursions})
#define LNPLAB_BACKEND_MAX_ARTIFACT_BYTES UINT64_C({backend.max_artifact_bytes})
#define LNPLAB_BACKEND_MAX_PROOF_BYTES UINT64_C({backend.max_proof_bytes})
#define LNPLAB_BACKEND_OPERATOR_NORM_BOUND {challenge.operator_norm_bound!r}
#define LNPLAB_BACKEND_CHALLENGE_TAU UINT64_C({tau})
#define LNPLAB_BACKEND_PARAMETER_SHA256 "{parameter_fingerprint(parameters)}"
#define LNPLAB_SOURCE_PARAMETER_FINGERPRINT "{source_parameter_fingerprint}"

#endif /* LNPLABRADOR_BACKEND_PARAMS_C_H */
"""


def sync_backend_header(
    path: Path = DEFAULT_BACKEND_HEADER,
    *,
    check: bool = False,
    parameters: Parameters = PARAMS,
) -> bool:
    """Synchronize the generated header; return ``True`` when already current."""

    expected = backend_header_text(parameters).encode("utf-8")
    try:
        current = path.read_bytes()
    except FileNotFoundError:
        current = None
    except OSError as exc:
        raise LNPLabError(f"cannot read backend parameter header {path}: {exc}") from exc
    if current == expected:
        return True
    if check:
        raise LNPLabError(
            f"{path} is stale; run 'python3 scripts/lnplabrador.py sync-backend'"
        )
    lab._atomic_write(path, expected, force=True)
    return False


def sync_c_backend_header(
    path: Path = DEFAULT_C_BACKEND_HEADER,
    *,
    check: bool = False,
    parameters: Parameters = PARAMS,
) -> bool:
    """Synchronize the generated C11 header for ``scripts/lab.c``."""

    expected = backend_c_header_text(parameters).encode("utf-8")
    try:
        current = path.read_bytes()
    except FileNotFoundError:
        current = None
    except OSError as exc:
        raise LNPLabError(f"cannot read C backend parameter header {path}: {exc}") from exc
    if current == expected:
        return True
    if check:
        raise LNPLabError(
            f"{path} is stale; run 'python3 scripts/lnplabrador.py sync-backend'"
        )
    lab._atomic_write(path, expected, force=True)
    return False


def sync_backend_headers(
    cpp_path: Path = DEFAULT_BACKEND_HEADER,
    c_path: Path = DEFAULT_C_BACKEND_HEADER,
    *,
    check: bool = False,
    parameters: Parameters = PARAMS,
) -> Tuple[bool, bool]:
    """Synchronize both generated backend headers."""

    cpp_current = sync_backend_header(cpp_path, check=check, parameters=parameters)
    c_current = sync_c_backend_header(c_path, check=check, parameters=parameters)
    return cpp_current, c_current


def audit_tex(tex_bytes: bytes, parameters: Parameters = PARAMS) -> Dict[str, Any]:
    validate_parameters(parameters)
    try:
        text = tex_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LNPLabError("the source TeX is not valid UTF-8") from exc

    required_markers = (
        r"\label{eq:three-commitments}",
        r"\label{eq:clab-open}",
        r"\label{eq:f1}",
        r"\label{eq:f2}",
        r"\label{eq:f3}",
        r"\label{eq:response-norm}",
        r"\label{eq:full-lnplab-relation}",
        r"\label{eq:fs-lnplab}",
        "7{,}544{,}448",
        "10{,}368",
        "15{,}488",
    )
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        raise LNPLabError("source TeX is missing expected relation markers: " + ", ".join(missing))

    displayed_modulus = f"{parameters.source_ring.modulus:,}".replace(",", "{,}")
    if displayed_modulus not in text:
        raise LNPLabError("source TeX q_pi does not match para1.py")

    counts = _tex_counts(parameters)
    if counts != {"A": 7_544_448, "h": 10_368, "w": 15_488}:
        raise LNPLabError("derived compiler counts do not match the supplied TeX")
    return {
        "tex_bytes": len(tex_bytes),
        "tex_sha256": hashlib.sha256(tex_bytes).hexdigest(),
        "parameter_sha256": parameter_fingerprint(parameters),
        "source_fingerprint_sha256": source_fingerprint(tex_bytes, parameters),
        "source_ring": {
            "degree": parameters.source_ring.degree,
            "modulus": parameters.source_ring.modulus,
            "coefficient_bits": parameters.source_ring.coefficient_bits,
        },
        "compressed_flow_ring_coordinates": {
            "m_cmp_A": parameters.boundary.message_a_ring_coordinates,
            "m_cmp_h": parameters.boundary.message_h_ring_coordinates,
            "m_cmp_w": parameters.boundary.message_w_ring_coordinates,
        },
        "binary_r1cs_rows": counts,
        "full_binary_r1cs_rows": sum(counts.values()),
    }


# ---------------------------------------------------------------------------
# Paper Sections 5.3--6 schedule and compressed-size audit
# ---------------------------------------------------------------------------


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise LNPLabError("invalid ceiling-division operands")
    return -(-numerator // denominator)


def _round_nearest_nonnegative(value: float) -> int:
    if not math.isfinite(value) or value < 0.0:
        raise LNPLabError("cannot round a negative or non-finite value")
    return int(math.floor(value + 0.5))


def _ceil_nth_root(value: int, degree: int) -> int:
    if value <= 0 or degree <= 0:
        raise LNPLabError("invalid integer-root operands")
    estimate = max(1, int(math.exp(math.log(value) / degree)))
    while estimate**degree < value:
        estimate += 1
    while estimate > 1 and (estimate - 1) ** degree >= value:
        estimate -= 1
    return estimate


def _paper_level_for_rank(
    *,
    n: int,
    r: int,
    beta: float,
    kappa: int,
    parameters: Parameters,
) -> Dict[str, Any]:
    """Evaluate the Section 5.4 recurrence for a fixed inner rank."""

    if min(n, r, kappa) <= 0 or not math.isfinite(beta) or beta <= 0.0:
        raise LNPLabError("paper level n,r,kappa,beta must be positive")
    q = parameters.backend.modulus
    degree = parameters.backend.degree
    challenge = parameters.labrador_challenge
    tau = challenge.unit_coefficients + 4 * challenge.double_coefficients
    coefficient_sd = beta / math.sqrt(r * n * degree)
    base_real = math.sqrt(coefficient_sd * math.sqrt(12.0 * r * tau))
    base = max(2, _round_nearest_nonnegative(base_real))
    t1 = max(2, _round_nearest_nonnegative(math.log(q) / math.log(base)))
    base1 = _ceil_nth_root(q, t1)

    garbage_width = math.sqrt(24.0 * n * degree) * coefficient_sd**2
    t2_real = math.log(garbage_width) / math.log(base) if garbage_width > 1.0 else 0.0
    t2 = max(2, _round_nearest_nonnegative(t2_real))
    base2 = max(
        2,
        _round_nearest_nonnegative(
            math.exp(math.log(max(1.0, garbage_width)) / t2)
        ),
    )

    pair_count = r * (r + 1) // 2
    t_length = r * t1 * kappa
    g_length = pair_count * t2
    h_length = pair_count * t1
    gamma_squared = beta * beta * tau
    gamma1_squared = (
        (base1 * base1 * t1 / 12.0) * r * kappa * degree
        + (base2 * base2 * t2 / 12.0) * pair_count * degree
    )
    gamma2_squared = (base1 * base1 * t1 / 12.0) * pair_count * degree
    beta_prime_squared = (
        2.0 * gamma_squared / (base * base)
        + gamma1_squared
        + gamma2_squared
    )
    beta_prime = math.sqrt(beta_prime_squared)
    return {
        "coefficient_standard_deviation": coefficient_sd,
        "decomposition": {
            "z": {"base_real": base_real, "base": base, "digits": 2},
            "t": {"base": base1, "digits": t1},
            "g": {"modeled_width": garbage_width, "base": base2, "digits": t2},
            "h": {"base": base1, "digits": t1},
        },
        "combined_v": {
            "t_length": t_length,
            "g_length": g_length,
            "h_length": h_length,
            "length": t_length + g_length + h_length,
        },
        "target_norm": {
            "gamma_squared": gamma_squared,
            "gamma1_squared": gamma1_squared,
            "gamma2_squared": gamma2_squared,
            "beta_prime_squared": beta_prime_squared,
            "beta_prime": beta_prime,
        },
    }


def _heuristic_msis_rank_l2(
    norm_bound: float, parameters: Parameters = PARAMS
) -> int:
    """Norm-dependent reference Core-SVP/root-Hermite fallback."""

    q = parameters.backend.modulus
    degree = parameters.backend.degree
    delta = parameters.backend.root_hermite_delta
    if not math.isfinite(norm_bound) or not 0.0 < norm_bound < q / 2.0:
        raise LNPLabError(
            "root-Hermite MSIS heuristic requires a finite norm below q/2"
        )
    numerator = math.log2(2.0 * norm_bound) ** 2
    denominator = 4.0 * degree * math.log2(q) * math.log2(delta)
    return max(1, math.ceil(numerator / denominator))


def _select_paper_ranks(
    *,
    n: int,
    r: int,
    beta: float,
    recursed_target: bool,
    parameters: Parameters,
) -> Dict[str, Any]:
    """Select heuristic kappa,kappa1=kappa2 using Theorem 5.1 bounds."""

    challenge = parameters.labrador_challenge
    extraction_slack = math.sqrt(
        parameters.backend.security_bits
        / parameters.backend.extraction_slack_denominator
    )
    theorem_multiplier = extraction_slack if recursed_target else 1.0
    for kappa in range(1, parameters.paper_proof.max_rank_search + 1):
        level = _paper_level_for_rank(
            n=n, r=r, beta=beta, kappa=kappa, parameters=parameters
        )
        base = level["decomposition"]["z"]["base"]
        beta_prime = level["target_norm"]["beta_prime"]
        inner_bound = max(
            8.0
            * challenge.operator_norm_bound
            * (base + 1)
            * beta_prime,
            2.0 * (base + 1) * beta_prime
            + 4.0
            * challenge.operator_norm_bound
            * extraction_slack
            * beta,
        )
        effective_inner_bound = inner_bound * theorem_multiplier
        required_kappa = _heuristic_msis_rank_l2(
            effective_inner_bound, parameters
        )
        if kappa >= required_kappa:
            break
    else:
        raise LNPLabError(
            "no heuristic inner rank found below paper_proof.max_rank_search"
        )

    effective_outer_bound = 2.0 * beta_prime * theorem_multiplier
    outer_rank = _heuristic_msis_rank_l2(effective_outer_bound, parameters)
    return {
        **level,
        "ranks": {
            "kappa": kappa,
            "kappa1": outer_rank,
            "kappa2": outer_rank,
            "method": (
                "norm-dependent-reference-Core-SVP-root-Hermite-"
                f"{parameters.backend.root_hermite_delta}-fallback"
            ),
            "estimator_certified": False,
        },
        "theorem_5_1_msis_inputs": {
            "inner_norm": inner_bound,
            "outer_norm": 2.0 * beta_prime,
            "remark_5_2_multiplier": theorem_multiplier,
            "effective_inner_norm": effective_inner_bound,
            "effective_outer_norm": effective_outer_bound,
        },
    }


def _post_adjustment_rank_screen(
    *,
    selected: Mapping[str, Any],
    beta: float,
    recursed_target: bool,
    section_5_6_final: bool,
    unsplit_z_tail_transition: bool,
    parameters: Parameters,
) -> Dict[str, Any]:
    """Re-screen displayed ranks against the decomposition actually retained.

    Rank selection initially proposes a decomposition.  A configured schedule
    may subsequently tighten its digit counts to fit the next level, changing
    beta'.  This diagnostic deliberately does not iterate the mutually
    dependent rank/decomposition choice; it exposes whether the displayed
    ranks still pass the same root-Hermite fallback at the resulting norm.
    """

    challenge = parameters.labrador_challenge
    extraction_slack = math.sqrt(
        parameters.backend.security_bits
        / parameters.backend.extraction_slack_denominator
    )
    theorem_multiplier = extraction_slack if recursed_target else 1.0
    base = selected["decomposition"]["z"]["base"]
    beta_prime = selected["target_norm"]["beta_prime"]
    inner_bound = max(
        8.0 * challenge.operator_norm_bound * (base + 1) * beta_prime,
        2.0 * (base + 1) * beta_prime
        + 4.0
        * challenge.operator_norm_bound
        * extraction_slack
        * beta,
    )
    outer_bound = 2.0 * beta_prime
    effective_inner_bound = inner_bound * theorem_multiplier
    effective_outer_bound = outer_bound * theorem_multiplier
    required_kappa = _heuristic_msis_rank_l2(effective_inner_bound, parameters)
    required_outer = _heuristic_msis_rank_l2(effective_outer_bound, parameters)
    ranks = selected["ranks"]
    passes: Dict[str, Optional[bool]] = {
        "kappa": ranks["kappa"] >= required_kappa,
        "kappa1": (
            None if section_5_6_final else ranks["kappa1"] >= required_outer
        ),
        "kappa2": (
            None if section_5_6_final else ranks["kappa2"] >= required_outer
        ),
    }
    return {
        "theorem_5_1_msis_inputs": {
            "inner_norm": inner_bound,
            "outer_norm": outer_bound,
            "remark_5_2_multiplier": theorem_multiplier,
            "effective_inner_norm": effective_inner_bound,
            "effective_outer_norm": effective_outer_bound,
        },
        "screen": {
            "method": (
                "conservative application of the same root-Hermite fallback, "
                "after capacity adjustment"
            ),
            "estimator_certified_minimum": False,
            "unsplit_z_tail_application_is_diagnostic": (
                unsplit_z_tail_transition
            ),
            "outer_commitments_used_by_this_execution": (
                not section_5_6_final
            ),
            "joint_rank_decomposition_fixed_point": False,
            "required_ranks": {
                "kappa": required_kappa,
                "kappa1": None if section_5_6_final else required_outer,
                "kappa2": None if section_5_6_final else required_outer,
            },
            "displayed_ranks": {
                "kappa": ranks["kappa"],
                "kappa1": ranks["kappa1"],
                "kappa2": ranks["kappa2"],
            },
            "passes": passes,
            "all_displayed_ranks_pass": all(
                passed for passed in passes.values() if passed is not None
            ),
        },
    }


def _section_5_7_prefix_size(
    *, beta: float, kappa1: int, kappa2: int, parameters: Parameters
) -> Dict[str, Any]:
    """Size of one non-final execution, excluding its last prover message."""

    q = parameters.backend.modulus
    degree = parameters.backend.degree
    security_bits = parameters.backend.security_bits
    jl_rows = parameters.backend.jl_rows
    log_q = math.log2(q)
    packed_q_bits = q.bit_length()
    aggregation_rounds = math.ceil(security_bits / log_q)
    jl_width = math.log2(12.0 * beta / math.sqrt(2.0))
    ideal = {
        "outer_commitments": (kappa1 + kappa2) * degree * log_q,
        "jl_projection": jl_rows * jl_width,
        "jl_proof": aggregation_rounds * degree * log_q,
        "challenge_seeds": 4 * security_bits,
    }
    packed = {
        "outer_commitments": (kappa1 + kappa2) * degree * packed_q_bits,
        "jl_projection": jl_rows * math.ceil(jl_width),
        "jl_proof": aggregation_rounds * degree * packed_q_bits,
        "challenge_seeds": 4 * security_bits,
    }
    return {
        "jl_coefficient_width_bits": jl_width,
        "aggregation_rounds": aggregation_rounds,
        "challenge_seed_count": 4,
        "ideal_entropy_bits": ideal,
        "ideal_entropy_total_bits": sum(ideal.values()),
        "fixed_width_bits": packed,
        "fixed_width_total_bits": sum(packed.values()),
    }


def _section_5_7_final_size(
    *,
    n: int,
    r: int,
    beta: float,
    kappa: int,
    final_nu: int,
    parameters: Parameters,
) -> Dict[str, Any]:
    """Optimized Section 5.6 final response, sized with Section 5.7."""

    q = parameters.backend.modulus
    degree = parameters.backend.degree
    tau = (
        parameters.labrador_challenge.unit_coefficients
        + 4 * parameters.labrador_challenge.double_coefficients
    )
    log_q = math.log2(q)
    packed_q_bits = q.bit_length()
    z_width = math.log2(12.0 * beta * math.sqrt(tau / (n * degree)))
    g_width = math.log2(
        12.0
        * beta
        * beta
        * math.sqrt(2.0 / (r * r * n * degree))
    )
    g_count = 2 * final_nu + 1
    h_count = 2 * r - 1
    ideal = {
        "masked_opening_z": n * degree * z_width,
        "inner_commitments_t": r * kappa * degree * log_q,
        "reduced_garbage_g": g_count * degree * g_width,
        "reduced_garbage_h": h_count * degree * log_q,
    }
    packed = {
        "masked_opening_z": n * degree * math.ceil(z_width),
        "inner_commitments_t": r * kappa * degree * packed_q_bits,
        "reduced_garbage_g": g_count * degree * math.ceil(g_width),
        "reduced_garbage_h": h_count * degree * packed_q_bits,
    }
    return {
        "section_5_6_optimized": True,
        "outer_commitments_included": False,
        "g_polynomials": g_count,
        "h_polynomials": h_count,
        "z_coefficient_width_bits": z_width,
        "g_coefficient_width_bits": g_width,
        "ideal_entropy_bits": ideal,
        "ideal_entropy_total_bits": sum(ideal.values()),
        "fixed_width_bits": packed,
        "fixed_width_total_bits": sum(packed.values()),
    }


def _locally_optimal_standard_split(
    *, n: int, auxiliary_length: int, target_r: int
) -> Dict[str, int]:
    """Minimize n' among standard splits with a fixed target multiplicity."""

    candidates: List[Tuple[int, int, int, int]] = []
    for nu in range(1, (target_r - 1) // 2 + 1):
        mu = target_r - 2 * nu
        if mu <= 0:
            continue
        next_n = max(_ceil_div(n, nu), _ceil_div(auxiliary_length, mu))
        padding = next_n * target_r - (2 * n + auxiliary_length)
        candidates.append((next_n, padding, nu, mu))
    if not candidates:
        raise LNPLabError("target r admits no positive standard nu,mu split")
    next_n, padding, nu, mu = min(candidates)
    return {"n_prime": next_n, "nu": nu, "mu": mu, "padding_polynomials": padding}


def _configured_path_status(value: Optional[str]) -> Dict[str, Any]:
    if value is None:
        return {"configured": False, "path": None, "exists": False}
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return {
        "configured": True,
        "path": str(path.resolve()),
        "exists": path.is_file(),
    }


def _fit_level_to_configured_transition(
    *,
    selected: Dict[str, Any],
    row: Any,
    next_row: Any,
    beta: float,
    parameters: Parameters,
) -> Dict[str, Any]:
    """Make the current t||g||h vector fit the configured next row.

    The draft table fixes only n and r.  Usually the Section 5.4
    decomposition already fits.  For the q40 penultimate row it does not, so
    search fixed t/h and g digit counts exactly as the C++ schedule compiler
    does.  Ranks are deliberately left unchanged and the report flags that
    they must be re-certified for the enlarged target norm.
    """

    if row.nu_to_next is None or row.mu_to_next is None:
        return selected
    capacity = row.mu_to_next * next_row.n
    ordinary_length = selected["combined_v"]["length"]
    if ordinary_length <= capacity:
        return {
            **selected,
            "capacity_adjustment": {
                "applied": False,
                "auxiliary_capacity_polynomials": capacity,
                "ordinary_auxiliary_polynomials": ordinary_length,
                "adjusted_auxiliary_polynomials": ordinary_length,
            },
        }

    q = parameters.backend.modulus
    degree = parameters.backend.degree
    tau = (
        parameters.labrador_challenge.unit_coefficients
        + 4 * parameters.labrador_challenge.double_coefficients
    )
    r = row.r
    n = row.n
    kappa = selected["ranks"]["kappa"]
    pair_count = r * (r + 1) // 2
    coefficient_sd = beta / math.sqrt(r * n * degree)
    garbage_width = math.sqrt(24.0 * n * degree) * coefficient_sd**2
    z_base = selected["decomposition"]["z"]["base"]
    gamma_squared = beta * beta * tau
    best: Optional[Tuple[float, int, int, int, int, int]] = None
    for digits1 in range(2, 41):
        base1 = _ceil_nth_root(q, digits1)
        for digits2 in range(2, 41):
            t_length = digits1 * r * kappa
            g_length = digits2 * pair_count
            h_length = digits1 * pair_count
            auxiliary = t_length + g_length + h_length
            if auxiliary > capacity:
                continue
            base2 = max(
                2,
                _round_nearest_nonnegative(
                    math.exp(math.log(max(1.0, garbage_width)) / digits2)
                ),
            )
            gamma1_squared = (
                (base1 * base1 * digits1 / 12.0) * r * kappa * degree
                + (base2 * base2 * digits2 / 12.0) * pair_count * degree
            )
            gamma2_squared = (
                (base1 * base1 * digits1 / 12.0) * pair_count * degree
            )
            beta_prime_squared = (
                gamma_squared + gamma1_squared + gamma2_squared
                if row.tail_transition
                else 2.0 * gamma_squared / (z_base * z_base)
                + gamma1_squared
                + gamma2_squared
            )
            candidate = (
                beta_prime_squared,
                auxiliary,
                digits1,
                digits2,
                base1,
                base2,
            )
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise LNPLabError(
            f"paper level {row.level} cannot fit t||g||h in level {next_row.level}"
        )

    beta_prime_squared, auxiliary, digits1, digits2, base1, base2 = best
    t_length = digits1 * r * kappa
    g_length = digits2 * pair_count
    h_length = digits1 * pair_count
    adjusted = dict(selected)
    adjusted["decomposition"] = {
        **selected["decomposition"],
        "t": {"base": base1, "digits": digits1},
        "g": {
            **selected["decomposition"]["g"],
            "base": base2,
            "digits": digits2,
        },
        "h": {"base": base1, "digits": digits1},
    }
    adjusted["combined_v"] = {
        "t_length": t_length,
        "g_length": g_length,
        "h_length": h_length,
        "length": auxiliary,
    }
    gamma1_squared = (
        (base1 * base1 * digits1 / 12.0) * r * kappa * degree
        + (base2 * base2 * digits2 / 12.0) * pair_count * degree
    )
    gamma2_squared = (base1 * base1 * digits1 / 12.0) * pair_count * degree
    adjusted["target_norm"] = {
        "gamma_squared": gamma_squared,
        "gamma1_squared": gamma1_squared,
        "gamma2_squared": gamma2_squared,
        "beta_prime_squared": beta_prime_squared,
        "beta_prime": math.sqrt(beta_prime_squared),
    }
    adjusted["capacity_adjustment"] = {
        "applied": True,
        "reason": "ordinary Section 5.4 decomposition does not fit configured n'",
        "auxiliary_capacity_polynomials": capacity,
        "ordinary_auxiliary_polynomials": ordinary_length,
        "adjusted_auxiliary_polynomials": auxiliary,
        "ranks_recertification_required": True,
    }
    return adjusted


def paper_plan_report(
    tex_bytes: bytes, parameters: Parameters = PARAMS
) -> Dict[str, Any]:
    """Audit the supplied seven-level schedule and recompute Section 5.7 size."""

    tex = audit_tex(tex_bytes, parameters)
    schedule = parameters.paper_proof.schedule
    q = parameters.backend.modulus
    degree = parameters.backend.degree
    beta = math.sqrt(q)
    derived_levels: List[Dict[str, Any]] = []

    for index, row in enumerate(schedule):
        selected = _select_paper_ranks(
            n=row.n,
            r=row.r,
            beta=beta,
            recursed_target=index < len(schedule) - 1,
            parameters=parameters,
        )
        if index < len(schedule) - 1:
            selected = _fit_level_to_configured_transition(
                selected=selected,
                row=row,
                next_row=schedule[index + 1],
                beta=beta,
                parameters=parameters,
            )
        post_adjustment = _post_adjustment_rank_screen(
            selected=selected,
            beta=beta,
            recursed_target=index < len(schedule) - 1,
            section_5_6_final=index == len(schedule) - 1,
            unsplit_z_tail_transition=row.tail_transition,
            parameters=parameters,
        )
        selected["theorem_5_1_msis_inputs"] = post_adjustment[
            "theorem_5_1_msis_inputs"
        ]
        selected["post_adjustment_root_hermite_screen"] = post_adjustment[
            "screen"
        ]
        prefix = _section_5_7_prefix_size(
            beta=beta,
            kappa1=selected["ranks"]["kappa1"],
            kappa2=selected["ranks"]["kappa2"],
            parameters=parameters,
        )
        level: Dict[str, Any] = {
            "level": row.level,
            "relation": {
                "n": row.n,
                "r": row.r,
                "beta": beta,
                "witness_polynomials": row.n * row.r,
            },
            "ranks": selected["ranks"],
            "decomposition": selected["decomposition"],
            "combined_v": selected["combined_v"],
            "target_norm": selected["target_norm"],
            "theorem_5_1_msis_inputs": selected["theorem_5_1_msis_inputs"],
            "post_adjustment_root_hermite_screen": selected[
                "post_adjustment_root_hermite_screen"
            ],
            "capacity_adjustment": selected.get("capacity_adjustment"),
            "section_5_7_prefix": prefix,
            "reference_table": {
                "witness_kib": row.reference_witness_kib,
                "output_kib": row.reference_output_kib,
                "recomputed_not_copied": True,
            },
        }

        if index < len(schedule) - 1:
            next_row = schedule[index + 1]
            if row.nu_to_next is None or row.mu_to_next is None:
                raise AssertionError("validated schedule lost a split")
            if row.tail_transition:
                auxiliary_required_n = _ceil_div(
                    selected["combined_v"]["length"], row.mu_to_next
                )
                required_n = max(
                    _ceil_div(row.n, row.nu_to_next), auxiliary_required_n
                )
                level["transition"] = {
                    "kind": "section-5.6-tail",
                    "nu": row.nu_to_next,
                    "mu": row.mu_to_next,
                    "derived_r_prime": row.nu_to_next + row.mu_to_next,
                    "configured_n_prime": next_row.n,
                    "minimum_n_from_unsplit_z": _ceil_div(row.n, row.nu_to_next),
                    "minimum_n_from_auxiliary": auxiliary_required_n,
                    "minimum_n_for_configured_split": required_n,
                    "configured_capacity_holds": next_row.n >= required_n,
                    "specialized_final_round_compiler_implemented": True,
                }
            else:
                auxiliary_length = selected["combined_v"]["length"]
                required_n = max(
                    _ceil_div(row.n, row.nu_to_next),
                    _ceil_div(auxiliary_length, row.mu_to_next),
                )
                local_optimum = _locally_optimal_standard_split(
                    n=row.n,
                    auxiliary_length=auxiliary_length,
                    target_r=next_row.r,
                )
                level["transition"] = {
                    "kind": "section-5.3-standard",
                    "nu": row.nu_to_next,
                    "mu": row.mu_to_next,
                    "derived_r_prime": 2 * row.nu_to_next + row.mu_to_next,
                    "configured_n_prime": next_row.n,
                    "minimum_n_for_configured_split": required_n,
                    "configured_capacity_holds": next_row.n >= required_n,
                    "locally_optimal_for_fixed_r_prime": local_optimum,
                    "configured_split_is_locally_optimal": (
                        row.nu_to_next == local_optimum["nu"]
                        and row.mu_to_next == local_optimum["mu"]
                        and next_row.n >= local_optimum["n_prime"]
                    ),
                }
        derived_levels.append(level)
        beta = selected["target_norm"]["beta_prime"]

    final_transition = schedule[-2]
    if final_transition.nu_to_next is None:
        raise AssertionError("validated schedule has no final nu")
    final_level = derived_levels[-1]
    final_level["section_5_7_prefix_with_outer_commitments_counterfactual"] = (
        final_level.pop("section_5_7_prefix")
    )
    final_size = _section_5_7_final_size(
        n=schedule[-1].n,
        r=schedule[-1].r,
        beta=final_level["relation"]["beta"],
        kappa=final_level["ranks"]["kappa"],
        final_nu=final_transition.nu_to_next,
        parameters=parameters,
    )
    final_level["section_5_7_final_response"] = final_size

    # Section 5.6 removes only the two outer commitments.  The last
    # execution still carries its JL projection p, the lifted/JL proof
    # polynomials b''(k), and the Fiat--Shamir challenge seeds described in
    # Section 5.7.  The C++ transcript likewise retains p and b_agg while
    # leaving u1/u2 empty, so omitting this prefix would under-count the
    # optimized tail.
    final_prefix = _section_5_7_prefix_size(
        beta=final_level["relation"]["beta"],
        kappa1=final_level["ranks"]["kappa1"],
        kappa2=final_level["ranks"]["kappa2"],
        parameters=parameters,
    )
    final_prefix["outer_commitments_omitted"] = True
    # The ordinary protocol has four public-coin challenges.  Section 5.6
    # replaces the last one by r sequential challenges, hence 3+r short
    # seeds under the paper's transcript-size convention.  The C++ Fiat--
    # Shamir wire codec recomputes these hashes and does not serialize them.
    final_challenge_seed_count = 3 + schedule[-1].r
    final_prefix["challenge_seed_count"] = final_challenge_seed_count
    for representation in ("ideal_entropy_bits", "fixed_width_bits"):
        final_prefix[representation]["outer_commitments"] = 0
        final_prefix[representation]["challenge_seeds"] = (
            final_challenge_seed_count * parameters.backend.security_bits
        )
    final_prefix["ideal_entropy_total_bits"] = sum(
        final_prefix["ideal_entropy_bits"].values()
    )
    final_prefix["fixed_width_total_bits"] = sum(
        final_prefix["fixed_width_bits"].values()
    )
    final_level["section_5_7_final_prefix"] = final_prefix

    prefix_ideal = sum(
        level["section_5_7_prefix"]["ideal_entropy_total_bits"]
        for level in derived_levels[:-1]
    ) + final_prefix["ideal_entropy_total_bits"]
    prefix_packed = sum(
        level["section_5_7_prefix"]["fixed_width_total_bits"]
        for level in derived_levels[:-1]
    ) + final_prefix["fixed_width_total_bits"]
    ring_packed_bits = degree * q.bit_length()
    ring_ideal_bits = degree * math.log2(q)
    r1cs_ideal = (
        parameters.paper_proof.r1cs_reduction_commitment_rank * ring_ideal_bits
    )
    r1cs_packed = (
        parameters.paper_proof.r1cs_reduction_commitment_rank * ring_packed_bits
    )
    core_ideal = prefix_ideal + final_size["ideal_entropy_total_bits"] + r1cs_ideal
    core_packed = prefix_packed + final_size["fixed_width_total_bits"] + r1cs_packed

    compact_flow_count = 3 * parameters.boundary.compressing_msis_rank
    compact_ideal = compact_flow_count * ring_ideal_bits
    compact_packed = compact_flow_count * ring_packed_bits
    masks_ideal = parameters.paper_proof.constant_term_mask_commitments * ring_ideal_bits
    masks_packed = parameters.paper_proof.constant_term_mask_commitments * ring_packed_bits
    projection_bits = parameters.paper_proof.lnp_projection_response_bytes * 8
    auxiliary_ideal = compact_ideal + masks_ideal + projection_bits
    auxiliary_packed = compact_packed + masks_packed + projection_bits

    path_status = {
        "A": _configured_path_status(parameters.full_r1cs.matrix_a_path),
        "B": _configured_path_status(parameters.full_r1cs.matrix_b_path),
        "C": _configured_path_status(parameters.full_r1cs.matrix_c_path),
        "witness": _configured_path_status(parameters.full_r1cs.witness_path),
    }
    full_input_ready = (
        parameters.full_r1cs.variable_count is not None
        and all(item["exists"] for item in path_status.values())
    )
    actual_constraints = tex["full_binary_r1cs_rows"]
    capacity = parameters.full_r1cs.constraint_capacity
    first_witness_capacity_bits = schedule[0].n * schedule[0].r * degree
    padded_binary_reduction_bits = 8 * capacity
    union_soundness = (
        parameters.labrador_challenge.paper_soundness_bits
        - math.log2(len(schedule))
    )
    capacity_adjusted_levels = [
        level["level"]
        for level in derived_levels
        if (level.get("capacity_adjustment") or {}).get("applied", False)
    ]
    post_adjustment_rank_failures = [
        level["level"]
        for level in derived_levels
        if not level["post_adjustment_root_hermite_screen"][
            "all_displayed_ranks_pass"
        ]
    ]
    return {
        "schema": "lnplabrador-paper-plan-v1",
        "paper_sections": ["5.3", "5.4", "5.6", "5.7", "6"],
        "backend_ring": {
            "degree": degree,
            "modulus": q,
            "modulus_hex": hex(q),
            "log2_modulus": math.log2(q),
            "packed_coefficient_bits": q.bit_length(),
            "q_mod_2d": q % (2 * degree),
            "multiplicative_order_mod_2d": next(
                exponent
                for exponent in range(1, 2 * degree + 1)
                if pow(q, exponent, 2 * degree) == 1
            ),
            "irreducible_factor_count_of_x_degree_plus_1": 2,
            "arithmetic_backend": "computational-CRT/NTT-or-correct-coefficient-domain",
        },
        "binary_r1cs": {
            "rows_derived_from_tex": actual_constraints,
            "configured_capacity": capacity,
            "fits_below_capacity": actual_constraints <= capacity,
            "padding_model": "binary variables are conservatively padded to equal constraints",
            "schedule_first_level_capacity_bits": first_witness_capacity_bits,
            "padded_binary_reduction_witness_bits": padded_binary_reduction_bits,
            "first_level_padding_bits": (
                first_witness_capacity_bits - padded_binary_reduction_bits
            ),
            "matrices_and_witness": path_status,
            "variable_count": parameters.full_r1cs.variable_count,
            "full_input_ready": full_input_ready,
            "full_section_6_compiler_available": False,
            "full_input_compiled": False,
            "reason_if_not_ready": (
                None
                if full_input_ready
                else "TeX has symbolic equations/counts but no numeric A,B,C matrices or witness"
            ),
            "reason_not_compiled": (
                "The current .lab frontend stores dense principal constraints and "
                "only implements the reduced conformance profile"
            ),
        },
        "schedule": derived_levels,
        "proof_size": {
            "formula": "LaBRADOR Section 5.7 with Section 5.6 final round",
            "estimate_status": (
                "draft-post-adjustment-root-Hermite-screen-fails"
                if post_adjustment_rank_failures
                else "draft-until-ranks-are-estimator-certified"
            ),
            "rank_set_sized": "displayed planning ranks, including any screen failures",
            "r1cs_reduction_output_bits": {
                "ideal_entropy": r1cs_ideal,
                "fixed_width": r1cs_packed,
            },
            "recursive_core_bits": {
                "ideal_entropy": core_ideal,
                "fixed_width": core_packed,
            },
            "composition_auxiliary_bits": {
                "ideal_entropy": auxiliary_ideal,
                "fixed_width": auxiliary_packed,
                "compact_flow_commitments_fixed_width": compact_packed,
                "constant_term_masks_fixed_width": masks_packed,
                "lnp_projection_response": projection_bits,
            },
            "total_kib": {
                "ideal_entropy": (core_ideal + auxiliary_ideal) / 8192.0,
                "fixed_width": (core_packed + auxiliary_packed) / 8192.0,
            },
            "reference_table_recursive_kib": (
                parameters.paper_proof.reference_recursive_contribution_kib
            ),
            "native_cpp_struct_size_is_not_used": True,
            "canonical_cpp_bit_codec_implemented": True,
        },
        "security_status": {
            "concrete_security_claim": False,
            "module_sis_ranks_estimator_certified": False,
            "capacity_adjusted_rank_fixed_point_screened": False,
            "post_adjustment_root_hermite_screen_performed": True,
            "post_adjustment_root_hermite_screen_failing_levels": (
                post_adjustment_rank_failures
            ),
            "capacity_adjusted_levels_requiring_recertification": (
                capacity_adjusted_levels
            ),
            "rank_method": "norm-dependent root-Hermite fallback for planning",
            "paper_per_execution_soundness_bits": (
                parameters.labrador_challenge.paper_soundness_bits
            ),
            "seven_execution_union_bound_bits": union_soundness,
            "seven_execution_union_bound_is_standard_level_accounting_only": True,
            "section_5_6_sequential_challenge_error_accounted": False,
            "exact_composed_tail_soundness_accounted": False,
            "meets_128_bits_under_simple_union_bound": union_soundness >= 128.0,
        },
        "warnings": [
            (
                "Run a current Core-SVP/Module-SIS estimator for every displayed "
                "norm before making a security claim."
            ),
            (
                "Capacity-adjusted levels "
                + ",".join(str(level) for level in capacity_adjusted_levels)
                + " select ranks before the adjustment; jointly re-run the "
                "decomposition/rank fixed point and a concrete estimator before "
                "using the displayed size or security values as final."
            ),
            (
                "Displayed ranks fail their conservative post-adjustment "
                "root-Hermite screen at level(s) "
                + ",".join(str(level) for level in post_adjustment_rank_failures)
                + "; the reported proof size is therefore diagnostic, not a "
                "security-certified parameter set."
            ),
            (
                "The simple seven-execution union bound does not include the "
                "separate sequential-challenge error terms of the optimized "
                "Section 5.6 tail."
            ),
            (
                "The full-circuit size is a Section 5.7 estimate; the C++ runner "
                "measures its canonical codec only when actual proof vectors exist."
            ),
            (
                "A full binary-R1CS input cannot be generated until numeric A,B,C "
                "matrices and the real witness are supplied."
            ),
        ],
    }


def sample_lnp_challenge(
    seed: bytes, parameters: Parameters = PARAMS
) -> Tuple[List[int], int, float]:
    validate_parameters(parameters)
    cfg = parameters.lnp_challenge
    rng = ReferenceRng(seed, cfg.fs_domain.encode("utf-8"))
    threshold = pow(cfg.eta, 2 * cfg.power)
    for attempt in range(1, cfg.max_sampling_attempts + 1):
        candidate = sample_lnp_candidate(
            rng, parameters.source_ring.degree, cfg.coefficient_bound
        )
        measure = lnp_multiplier_measure(candidate, cfg.power)
        if measure <= threshold:
            return candidate, attempt, lnp_bound_from_measure(measure, cfg.power)
    raise LNPLabError("LNP challenge rejection sampler exceeded max_sampling_attempts")


def sample_labrador_challenge(
    seed: bytes, parameters: Parameters = PARAMS
) -> Tuple[List[int], int, float]:
    validate_parameters(parameters)
    cfg = parameters.labrador_challenge
    rng = ReferenceRng(seed, cfg.fs_domain.encode("utf-8"))
    guarded_bound = cfg.operator_norm_bound - cfg.numerical_guard
    for attempt in range(1, cfg.max_sampling_attempts + 1):
        candidate = sample_labrador_candidate(
            rng,
            parameters.backend.degree,
            cfg.zero_coefficients,
            cfg.unit_coefficients,
            cfg.double_coefficients,
        )
        norm = operator_norm(candidate)
        accepted = norm < guarded_bound if cfg.strict_operator_bound else norm <= guarded_bound
        if accepted:
            return candidate, attempt, norm
    raise LNPLabError("LaBRADOR challenge rejection sampler exceeded max_sampling_attempts")


def _challenge_report(parameters: Parameters = PARAMS) -> Dict[str, Any]:
    executable = parameters.executable
    lnp, lnp_attempts, lnp_bound = sample_lnp_challenge(
        executable.lnp_challenge_seed.encode("utf-8"), parameters
    )
    folding, folding_attempts, folding_norm = sample_labrador_challenge(
        executable.labrador_challenge_seed.encode("utf-8"), parameters
    )
    lnp_count = (2 * parameters.lnp_challenge.coefficient_bound + 1) ** (
        parameters.source_ring.degree // 2
    )
    folding_count = (
        math.comb(parameters.backend.degree, parameters.labrador_challenge.zero_coefficients)
        * math.comb(
            parameters.backend.degree - parameters.labrador_challenge.zero_coefficients,
            parameters.labrador_challenge.unit_coefficients,
        )
        * (1 << (
            parameters.labrador_challenge.unit_coefficients
            + parameters.labrador_challenge.double_coefficients
        ))
    )
    return {
        "LNP22_final_challenge": {
            "construction": "sigma_-1-fixed coefficients plus multiplier-bound rejection",
            "raw_count": lnp_count,
            "raw_entropy_bits": math.log2(lnp_count),
            "coefficient_bound": parameters.lnp_challenge.coefficient_bound,
            "power": parameters.lnp_challenge.power,
            "eta": parameters.lnp_challenge.eta,
            "attempts": lnp_attempts,
            "accepted_multiplier_bound": lnp_bound,
            "conjugation_fixed": conjugate(lnp) == lnp,
            "coefficients": lnp,
        },
        "LaBRADOR_folding_challenge": {
            "construction": (
                f"({parameters.labrador_challenge.zero_coefficients} zero,"
                f"{parameters.labrador_challenge.unit_coefficients} unit,"
                f"{parameters.labrador_challenge.double_coefficients} double) "
                "plus operator-norm rejection"
            ),
            "prefilter_count": folding_count,
            "prefilter_entropy_bits": math.log2(folding_count),
            "paper_average_sampling_attempts": (
                parameters.labrador_challenge.paper_average_sampling_attempts
            ),
            "paper_estimated_postfilter_entropy_bits": (
                math.log2(folding_count)
                - math.log2(parameters.labrador_challenge.paper_average_sampling_attempts)
            ),
            "paper_per_level_soundness_bits": (
                parameters.labrador_challenge.paper_soundness_bits
            ),
            "attempts": folding_attempts,
            "operator_norm": folding_norm,
            "operator_norm_bound": parameters.labrador_challenge.operator_norm_bound,
            "coefficients": folding,
        },
    }


def _zero_poly(degree: int = lab.DEGREE) -> List[int]:
    return [0] * degree


def _constant_poly(value: int, modulus: int = lab.BACKEND_Q) -> List[int]:
    result = _zero_poly()
    result[0] = lab.centered(value, modulus)
    return result


def _poly_add(left: Sequence[int], right: Sequence[int], modulus: int) -> List[int]:
    return [lab.centered(a + b, modulus) for a, b in zip(left, right)]


def _poly_neg(value: Sequence[int], modulus: int) -> List[int]:
    return [lab.centered(-coefficient, modulus) for coefficient in value]


def _poly_scale(value: Sequence[int], scalar: int, modulus: int) -> List[int]:
    return [lab.centered(coefficient * scalar, modulus) for coefficient in value]


def _poly_sum(values: Sequence[Sequence[int]], modulus: int) -> List[int]:
    result = _zero_poly()
    for value in values:
        result = _poly_add(result, value, modulus)
    return result


def _scalar_inverse(value: int, modulus: int, label: str) -> int:
    try:
        return pow(value % modulus, -1, modulus)
    except ValueError as exc:
        raise LNPLabError(f"{label}={value} is not invertible modulo the backend modulus") from exc


class PrincipalCompiler:
    """Dense compiler for the small executable profile accepted by ``lab.py``."""

    def __init__(self, names: Sequence[str], modulus: int):
        self.names = tuple(names)
        self.indices = {name: index for index, name in enumerate(self.names)}
        if len(self.indices) != len(self.names):
            raise LNPLabError("duplicate witness variable name")
        self.r = len(self.names)
        self.modulus = modulus
        self.constraints: List[lab.EqualityConstraint] = []
        self.labels: List[str] = []

    def add_equation(
        self,
        label: str,
        *,
        linear: Optional[Mapping[str, Sequence[int]]] = None,
        quadratic: Optional[Mapping[Tuple[str, str], Sequence[int]]] = None,
        constant: Optional[Sequence[int]] = None,
    ) -> None:
        a = [_zero_poly() for _ in range(self.r * self.r)]
        phi = [_zero_poly() for _ in range(self.r)]
        for (left, right), coefficient in (quadratic or {}).items():
            flat = self.indices[left] * self.r + self.indices[right]
            a[flat] = _poly_add(a[flat], coefficient, self.modulus)
        for name, coefficient in (linear or {}).items():
            index = self.indices[name]
            phi[index] = _poly_add(phi[index], coefficient, self.modulus)
        self.constraints.append(
            lab.EqualityConstraint(a, phi, list(constant or _zero_poly()))
        )
        self.labels.append(label)


def _derive_small_matrix(
    flow: str, rows: int, columns: int, seed: bytes, parameters: Parameters
) -> List[List[List[int]]]:
    rng = ReferenceRng(seed, f"LNPLAB/CLaB/{flow}/matrix/v1".encode("utf-8"))
    matrix: List[List[List[int]]] = []
    for _ in range(rows):
        row: List[List[int]] = []
        for _ in range(columns):
            polynomial = [rng.randbelow(3) - 1 for _ in range(parameters.backend.degree)]
            if not any(polynomial):
                polynomial[0] = 1
            row.append(polynomial)
        matrix.append(row)
    return matrix


def _make_oracle_seed(tex_bytes: bytes, parameters: Parameters = PARAMS) -> bytes:
    return (
        ORACLE_TEX_PREFIX
        + struct.pack("<Q", len(tex_bytes))
        + tex_bytes
        + PARAMETER_DIGEST_TAG
        + hashlib.sha256(_canonical_parameters(parameters)).digest()
    )


def extract_embedded_tex(oracle_seed: bytes) -> Tuple[bytes, bytes]:
    if not oracle_seed.startswith(ORACLE_TEX_PREFIX):
        raise LNPLabError("artifact oracle context has no embedded LNPLab TeX")
    cursor = len(ORACLE_TEX_PREFIX)
    if len(oracle_seed) < cursor + 8:
        raise LNPLabError("truncated embedded TeX length")
    tex_size = struct.unpack_from("<Q", oracle_seed, cursor)[0]
    cursor += 8
    end = cursor + tex_size
    if end > len(oracle_seed):
        raise LNPLabError("truncated embedded TeX body")
    tex_bytes = oracle_seed[cursor:end]
    remainder = oracle_seed[end:]
    if len(remainder) != len(PARAMETER_DIGEST_TAG) + 32 or not remainder.startswith(
        PARAMETER_DIGEST_TAG
    ):
        raise LNPLabError("malformed embedded para1.py digest")
    return tex_bytes, remainder[len(PARAMETER_DIGEST_TAG) :]


def _build_witness(parameters: Parameters) -> Tuple[List[str], Dict[str, int]]:
    executable = parameters.executable
    names: List[str] = []
    values: Dict[str, int] = {}
    for message_name, message_value in zip(
        executable.message_names, executable.message_values
    ):
        for bit in range(executable.bit_width):
            bit_name = f"b_{message_name}_{bit}"
            names.append(bit_name)
            values[bit_name] = (message_value >> bit) & 1
        names.append(message_name)
        values[message_name] = message_value
    for response_name, response_value in zip(
        executable.response_names, executable.response_values
    ):
        names.append(response_name)
        values[response_name] = response_value
    return names, values


def compile_bundle(
    tex_bytes: bytes, parameters: Parameters = PARAMS
) -> Tuple[lab.RelationBundle, Dict[str, Any]]:
    tex_audit = audit_tex(tex_bytes, parameters)
    modulus = parameters.backend.modulus
    degree = parameters.backend.degree
    executable = parameters.executable
    one = _constant_poly(1, modulus)
    minus_one = _constant_poly(-1, modulus)

    names, scalar_values = _build_witness(parameters)
    witness = [_constant_poly(scalar_values[name], modulus) for name in names]
    compiler = PrincipalCompiler(names, modulus)

    # Coefficient-wise binary relations are exact here because every reduced
    # profile bit is represented as a constant polynomial.
    for message_name in executable.message_names:
        for bit in range(executable.bit_width):
            bit_name = f"b_{message_name}_{bit}"
            compiler.add_equation(
                f"bitness:{bit_name}",
                quadratic={(bit_name, bit_name): one},
                linear={bit_name: minus_one},
            )

    # Rec_ell(b)=m for all six representative message coordinates.
    for message_name in executable.message_names:
        linear: Dict[str, Sequence[int]] = {message_name: one}
        for bit in range(executable.bit_width):
            linear[f"b_{message_name}_{bit}"] = _constant_poly(-(1 << bit), modulus)
        compiler.add_equation(f"reconstruct:{message_name}", linear=linear)

    # Three C_LaB commitments.  x_w covers the four message coordinates used
    # in f1/f2/f3, exactly mirroring the grouping in the supplied TeX.
    flows = {
        "A": ("t_A",),
        "h": ("h",),
        "w": ("w_A", "w_u", "t_Q", "v_Q"),
    }
    ajtai_seed = hashlib.shake_256(executable.ajtai_seed.encode("utf-8")).digest(32)
    commitments: Dict[str, List[List[int]]] = {}
    for flow, message_names in flows.items():
        bit_names = [
            f"b_{message_name}_{bit}"
            for message_name in message_names
            for bit in range(executable.bit_width)
        ]
        matrix = _derive_small_matrix(
            flow,
            parameters.boundary.compressing_msis_rank,
            len(bit_names),
            ajtai_seed,
            parameters,
        )
        output: List[List[int]] = []
        for row_index, row in enumerate(matrix):
            products = [
                lab.negacyclic_mul(
                    row[column], witness[compiler.indices[bit_name]], modulus
                )
                for column, bit_name in enumerate(bit_names)
            ]
            commitment = _poly_sum(products, modulus)
            output.append(commitment)
            compiler.add_equation(
                f"CLaB:{flow}:{row_index}",
                linear={bit_name: row[column] for column, bit_name in enumerate(bit_names)},
                constant=_poly_neg(commitment, modulus),
            )
        commitments[flow] = output

    # LNP's challenge c is sampled from the LNP22 conjugation-fixed space.
    c, lnp_attempts, lnp_multiplier_bound = sample_lnp_challenge(
        executable.lnp_challenge_seed.encode("utf-8"), parameters
    )
    z1, z2, zu = executable.response_values
    values = dict(zip(executable.message_names, executable.message_values))
    inv_z1 = _scalar_inverse(z1, modulus, "z_A_1")
    inv_zu = _scalar_inverse(zu, modulus, "z_u")

    # Choose public A2 and A1 so the non-zero representative witness satisfies
    # f2 and f1.  V_A is the zero polynomial in this conformance vector.
    a2 = _poly_scale(
        _poly_add(
            _poly_scale(c, values["t_A"], modulus),
            _constant_poly(values["w_u"], modulus),
            modulus,
        ),
        inv_zu,
        modulus,
    )
    a1 = _poly_scale(
        _poly_add(
            _constant_poly(values["w_A"], modulus),
            _poly_scale(a2, -z2, modulus),
            modulus,
        ),
        inv_z1,
        modulus,
    )
    compiler.add_equation(
        "f1",
        linear={
            "w_A": one,
            "z_A_1": _poly_neg(a1, modulus),
            "z_A_2": _poly_neg(a2, modulus),
        },
    )
    compiler.add_equation(
        "f2",
        linear={"t_A": c, "w_u": one, "z_u": _poly_neg(a2, modulus)},
    )

    # Reduced f3: E_00=1, e=e0=0 and z_Q=z_A_1.  B_Q is derived so that
    # z_A_1^2-c*t_Q+B_Q*z_u-v_Q=0; the essential quadratic shape is retained.
    bq_numerator = _poly_add(
        _poly_add(
            _poly_scale(c, values["t_Q"], modulus),
            _constant_poly(values["v_Q"], modulus),
            modulus,
        ),
        _constant_poly(-(z1 * z1), modulus),
        modulus,
    )
    bq = _poly_scale(bq_numerator, inv_zu, modulus)
    compiler.add_equation(
        "f3",
        quadratic={("z_A_1", "z_A_1"): one},
        linear={"t_Q": _poly_neg(c, modulus), "z_u": bq, "v_Q": minus_one},
    )

    squared_norm = sum(coefficient * coefficient for poly in witness for coefficient in poly)
    beta = math.sqrt(squared_norm) + executable.beta_margin
    rank = (
        parameters.backend.rank_override
        if parameters.backend.rank_override is not None
        else math.ceil(
            (math.log2(modulus) - 1.0) ** 2
            / (
                4.0
                * math.log2(parameters.backend.root_hermite_delta)
                * math.log2(modulus)
                * degree
            )
        )
    )
    aggregation_rounds = math.ceil(parameters.backend.security_bits / math.log2(modulus))
    bundle = lab.RelationBundle(
        mode=MODE,
        source_fingerprint=source_fingerprint(tex_bytes, parameters),
        degree=degree,
        modulus=modulus,
        r=len(names),
        n=1,
        beta=beta,
        kappa=rank,
        kappa1=rank,
        kappa2=rank,
        base1=executable.decomposition_base,
        base2=executable.decomposition_base,
        base3=executable.decomposition_base,
        jl_out=parameters.backend.jl_rows,
        aggregation_rounds=aggregation_rounds,
        recursions=parameters.backend.recursions,
        ajtai_seed=ajtai_seed,
        oracle_seed=_make_oracle_seed(tex_bytes, parameters),
        witness=witness,
        equality_constraints=compiler.constraints,
        const_zero_constraints=[],
    )
    validation = lab.validate_bundle(bundle)
    metadata = {
        "profile": "reduced-executable-conformance-profile",
        "exact_full_nibs_relation": False,
        "tex": tex_audit,
        "backend_ring": {"degree": degree, "modulus": modulus},
        "source_modulus_matches_backend": parameters.source_ring.modulus == modulus,
        "witness_variables": names,
        "witness_scalar_values": scalar_values,
        "constraint_labels": compiler.labels,
        "commitments": commitments,
        "public_coefficients": {"c": c, "A1": a1, "A2": a2, "BQ": bq},
        "lnp_challenge_attempts": lnp_attempts,
        "lnp_challenge_multiplier_bound": lnp_multiplier_bound,
        "witness_l2_norm": validation["witness_l2_norm"],
        "beta": beta,
        "rank": rank,
    }
    return bundle, metadata


def _read_tex(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise LNPLabError(f"cannot read source TeX {path}: {exc}") from exc


def _artifact_summary(
    bundle: lab.RelationBundle,
    artifact_bytes: int,
    current_tex: Optional[bytes],
    parameters: Parameters = PARAMS,
) -> Dict[str, Any]:
    embedded_tex, embedded_parameter_digest = extract_embedded_tex(bundle.oracle_seed)
    validation = lab.validate_bundle(bundle)
    current_parameter_digest = hashlib.sha256(_canonical_parameters(parameters)).digest()
    return {
        "format": "LNPLAB01/v1",
        "mode": bundle.mode,
        "artifact_bytes": artifact_bytes,
        "public_statement_sha3_256": bundle.public_digest.hex() if bundle.public_digest else None,
        "source_fingerprint_sha256": bundle.source_fingerprint,
        "embedded_tex_bytes": len(embedded_tex),
        "embedded_tex_sha256": hashlib.sha256(embedded_tex).hexdigest(),
        "embedded_tex_matches_current_file": (
            current_tex == embedded_tex if current_tex is not None else None
        ),
        "embedded_parameters_match_para1": embedded_parameter_digest == current_parameter_digest,
        "source_fingerprint_matches_current": (
            source_fingerprint(current_tex, parameters) == bundle.source_fingerprint
            if current_tex is not None
            else None
        ),
        "source_modulus": parameters.source_ring.modulus,
        "backend_modulus": bundle.modulus,
        "source_modulus_matches_backend": parameters.source_ring.modulus == bundle.modulus,
        "r": bundle.r,
        "n": bundle.n,
        "witness_polynomials": validation["witness_count"],
        "witness_l2_norm": validation["witness_l2_norm"],
        "beta": bundle.beta,
        "equality_constraints": len(bundle.equality_constraints),
        "const_zero_constraints": len(bundle.const_zero_constraints),
        "exact_full_nibs_relation": False,
        "warning": (
            "The plaintext witness and exact TeX source are embedded in this local input. "
            "This is a reduced executable profile, not a proof or the full NIBS relation."
        ),
    }


def _print_result(value: Mapping[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    for key, item in value.items():
        if isinstance(item, (dict, list)):
            print(f"{key}: {json.dumps(item, sort_keys=True)}")
        else:
            print(f"{key}: {item}")


def _build_cpu_runner() -> None:
    sync_backend_headers(parameters=PARAMS)
    lab._build_cpu(REPO_ROOT)


def run_self_tests(parameters: Parameters = PARAMS) -> None:
    assert backend_header_text(parameters).startswith("// Generated by scripts/lnplabrador.py")
    assert backend_c_header_text(parameters).startswith(
        "/* Generated by scripts/lnplabrador.py"
    )
    tex_bytes = _read_tex(DEFAULT_TEX)
    audit = audit_tex(tex_bytes, parameters)
    assert audit["full_binary_r1cs_rows"] == 7_570_304

    lnp, _, bound = sample_lnp_challenge(
        parameters.executable.lnp_challenge_seed.encode("utf-8"), parameters
    )
    assert conjugate(lnp) == lnp
    assert bound <= parameters.lnp_challenge.eta + 1e-9

    folding, _, folding_norm = sample_labrador_challenge(
        parameters.executable.labrador_challenge_seed.encode("utf-8"), parameters
    )
    assert folding.count(0) == parameters.labrador_challenge.zero_coefficients
    assert sum(abs(value) == 1 for value in folding) == (
        parameters.labrador_challenge.unit_coefficients
    )
    assert sum(abs(value) == 2 for value in folding) == (
        parameters.labrador_challenge.double_coefficients
    )
    assert folding_norm < parameters.labrador_challenge.operator_norm_bound

    bundle, metadata = compile_bundle(tex_bytes, parameters)
    assert metadata["constraint_labels"][-3:] == ["f1", "f2", "f3"]
    encoded = lab.bundle_to_bytes(bundle)
    decoded = lab.bundle_from_bytes(encoded)
    embedded, digest = extract_embedded_tex(decoded.oracle_seed)
    assert embedded == tex_bytes
    assert digest == hashlib.sha256(_canonical_parameters(parameters)).digest()
    assert decoded.source_fingerprint == source_fingerprint(tex_bytes, parameters)

    paper = paper_plan_report(tex_bytes, parameters)
    assert paper["backend_ring"]["modulus"] == (1 << 40) - 195
    assert paper["backend_ring"]["q_mod_2d"] == 61
    assert paper["backend_ring"]["multiplicative_order_mod_2d"] == 32
    assert paper["binary_r1cs"]["rows_derived_from_tex"] == 7_570_304
    assert paper["binary_r1cs"]["fits_below_capacity"]
    assert paper["binary_r1cs"]["first_level_padding_bits"] >= 0
    assert not paper["binary_r1cs"]["full_input_ready"]
    assert [level["ranks"]["kappa"] for level in paper["schedule"][:5]] == [
        17,
        14,
        12,
        11,
        10,
    ]
    assert all(
        level["transition"]["configured_capacity_holds"]
        for level in paper["schedule"][:5]
    )
    assert paper["schedule"][5]["capacity_adjustment"]["applied"]
    assert paper["schedule"][5]["post_adjustment_root_hermite_screen"][
        "required_ranks"
    ] == {"kappa": 13, "kappa1": 6, "kappa2": 6}
    assert not paper["schedule"][5]["post_adjustment_root_hermite_screen"][
        "all_displayed_ranks_pass"
    ]
    assert paper["security_status"][
        "post_adjustment_root_hermite_screen_failing_levels"
    ] == [6]
    assert paper["schedule"][5]["transition"]["configured_capacity_holds"]
    assert paper["schedule"][5]["transition"][
        "specialized_final_round_compiler_implemented"
    ]
    assert paper["schedule"][-1]["section_5_7_final_response"][
        "outer_commitments_included"
    ] is False
    assert 70.0 < paper["proof_size"]["total_kib"]["ideal_entropy"] < 72.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="audit TeX dimensions and both challenge spaces")
    audit.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    audit.add_argument("--json", action="store_true")

    paper = subparsers.add_parser(
        "paper-plan",
        help="audit the seven-level binary-R1CS schedule and Section 5.7 size",
    )
    paper.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    paper.add_argument("--output", "-o", type=Path)
    paper.add_argument("--force", action="store_true")
    paper.add_argument("--json", action="store_true")

    generate = subparsers.add_parser(
        "generate", help="compile the embedded-TeX executable profile to input1.lab"
    )
    generate.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    generate.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    generate.add_argument("--force", action="store_true")
    generate.add_argument("--json", action="store_true")

    inspect_parser = subparsers.add_parser(
        "inspect", help="validate a profile and compare its embedded TeX/parameters"
    )
    inspect_parser.add_argument("input", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    inspect_parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    inspect_parser.add_argument("--json", action="store_true")

    sample = subparsers.add_parser(
        "sample-challenges", help="print deterministic LNP and LaBRADOR challenge vectors"
    )
    sample.add_argument("--json", action="store_true")

    sync = subparsers.add_parser(
        "sync-backend", help="generate the C and C++ backend parameter headers from para1.py"
    )
    sync.add_argument("--output", type=Path, default=DEFAULT_BACKEND_HEADER)
    sync.add_argument("--c-output", type=Path, default=DEFAULT_C_BACKEND_HEADER)
    sync.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when the generated header is stale",
    )
    sync.add_argument("--json", action="store_true")

    run = subparsers.add_parser("run", help="validate and invoke the C++ LaBRADOR runner")
    run.add_argument("input", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    run.add_argument("--tex", type=Path, default=DEFAULT_TEX)
    run.add_argument("--device", default="CPU")
    run.add_argument("--build", action="store_true")

    subparsers.add_parser("self-test", help="run deterministic compiler and codec tests")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "audit":
            result = {
                "relation": audit_tex(_read_tex(args.tex), PARAMS),
                "challenge_spaces": _challenge_report(PARAMS),
                "composition": (
                    "TRaccoon B.6 layout: commit x_A,x_h,x_w; prove openings and the "
                    "hidden final LNP response with recursively composed LaBRADOR"
                ),
                "sources": SOURCES,
            }
            _print_result(result, args.json)
            return 0

        if args.command == "paper-plan":
            result = paper_plan_report(_read_tex(args.tex), PARAMS)
            if args.output is not None:
                encoded = (
                    json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
                ).encode("utf-8")
                lab._atomic_write(args.output, encoded, force=args.force)
            if args.json or args.output is None:
                _print_result(result, args.json)
            else:
                _print_result(
                    {
                        "paper_plan": str(args.output.resolve()),
                        "total_kib": result["proof_size"]["total_kib"],
                        "full_input_ready": result["binary_r1cs"]["full_input_ready"],
                        "concrete_security_claim": False,
                    },
                    False,
                )
            return 0

        if args.command == "generate":
            sync_backend_headers(parameters=PARAMS)
            tex_bytes = _read_tex(args.tex)
            bundle, _ = compile_bundle(tex_bytes, PARAMS)
            encoded = lab.bundle_to_bytes(bundle)
            lab._atomic_write(args.output, encoded, force=args.force)
            parsed = lab.bundle_from_bytes(encoded)
            summary = _artifact_summary(parsed, len(encoded), tex_bytes, PARAMS)
            _print_result(summary, args.json)
            return 0

        if args.command == "inspect":
            bundle, artifact_bytes = lab.read_bundle(args.input)
            if bundle.mode != MODE:
                raise LNPLabError(f"expected mode {MODE!r}, got {bundle.mode!r}")
            current_tex = _read_tex(args.tex)
            summary = _artifact_summary(bundle, artifact_bytes, current_tex, PARAMS)
            _print_result(summary, args.json)
            return 0

        if args.command == "sample-challenges":
            _print_result(_challenge_report(PARAMS), args.json)
            return 0

        if args.command == "sync-backend":
            cpp_current, c_current = sync_backend_headers(
                args.output,
                args.c_output,
                check=args.check,
                parameters=PARAMS,
            )
            result = {
                "backend_header": str(args.output.resolve()),
                "c_backend_header": str(args.c_output.resolve()),
                "already_current": cpp_current and c_current,
                "cpp_already_current": cpp_current,
                "c_already_current": c_current,
                "checked_only": args.check,
                "parameter_sha256": parameter_fingerprint(PARAMS),
            }
            _print_result(result, args.json)
            return 0

        if args.command == "run":
            sync_backend_headers(check=not args.build, parameters=PARAMS)
            bundle, artifact_bytes = lab.read_bundle(args.input)
            if bundle.mode != MODE:
                raise LNPLabError(f"expected mode {MODE!r}, got {bundle.mode!r}")
            summary = _artifact_summary(bundle, artifact_bytes, _read_tex(args.tex), PARAMS)
            if not summary["embedded_tex_matches_current_file"]:
                raise LNPLabError("embedded TeX differs from the current source file")
            if not summary["embedded_parameters_match_para1"]:
                raise LNPLabError("embedded parameter digest differs from current para1.py")
            if args.build:
                if args.device.upper() != "CPU":
                    raise LNPLabError("automatic --build currently supports only CPU")
                _build_cpu_runner()
            runner = REPO_ROOT / "build" / "src" / "lab_runner"
            if not runner.is_file():
                raise LNPLabError(f"runner not found at {runner}; pass --build")
            environment = os.environ.copy()
            library_path = str(REPO_ROOT / "build" / "icicle")
            if environment.get("LD_LIBRARY_PATH"):
                library_path += os.pathsep + environment["LD_LIBRARY_PATH"]
            environment["LD_LIBRARY_PATH"] = library_path
            result = subprocess.run(
                (str(runner), args.device, str(args.input.resolve())),
                cwd=REPO_ROOT,
                env=environment,
                check=False,
            )
            return result.returncode

        if args.command == "self-test":
            run_self_tests(PARAMS)
            print("lnplabrador.py self-test: PASS")
            return 0

        raise AssertionError(f"unhandled command {args.command}")
    except (
        LNPLabError,
        lab.LabError,
        OSError,
        struct.error,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"lnplabrador.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
