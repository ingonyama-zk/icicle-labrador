#!/usr/bin/env python3
"""Plan and launch recursive LaBRADOR executions for a ``.lab`` relation.

This module deliberately separates two notions:

* the *paper schedule* below follows LaBRADOR Sections 5.3--5.5 using the
  modulus and degree of the BabyKoala backend; and
* ``run`` invokes the repository's experimental C++ implementation.

The latter is useful for research and integration tests, but is not a
concrete-security claim.  In particular, this repository does not contain a
Module-SIS estimator or the optimized final round from Section 5.6.  The v1
``.lab`` format carries the initial parameters and the C++ backend derives the
same subsequent schedule; this module additionally emits a hash-addressed
audit plan, whose hash is not part of the v1 proof statement.

Commands
--------

``plan``
    Validate a relation and emit a deterministic, SHA-256-addressed recursive
    schedule.

``prepare``
    Clone a validated relation, set the first paper decomposition bases and
    requested execution count, and write a new ``.lab`` file.  The file still
    contains the secret witness in plaintext; it is not a proof.

``run``
    Prepare a temporary recursive bundle and invoke ``build/src/lab_runner``.
    This is an in-memory end-to-end research run, not a proof serializer or
    production proof generation.

``self-test``
    Exercise the recurrence, combined-vector packing, deterministic plan hash,
    and preparation bridge without requiring a C++ build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

try:  # Support both ``python scripts/labrador.py`` and package imports.
    from . import lab
    from .para import PARAMS
except ImportError:
    import lab  # type: ignore
    from para import PARAMS  # type: ignore


PLAN_SCHEMA = "lnplab-recursive-plan-v1"
PLANNER_VERSION = 1
PAPER_SECTION = "LaBRADOR Sections 5.3--5.5"
THEOREM = "LaBRADOR Theorem 5.1 and Remark 5.2"


class RecursivePlanError(ValueError):
    """An invalid or unsupported recursive schedule."""


@dataclass(frozen=True)
class RankTriple:
    kappa: int
    kappa1: int
    kappa2: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "kappa": self.kappa,
            "kappa1": self.kappa1,
            "kappa2": self.kappa2,
        }


def _plain_int(value: Any, label: str, *, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RecursivePlanError(f"{label} must be an integer >= {minimum}")
    return value


def _finite_positive(value: float, label: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise RecursivePlanError(f"{label} must be finite and positive")
    return value


def _round_nearest_positive(value: float) -> int:
    """Nearest non-negative integer, with exact halves rounded upward."""

    if not math.isfinite(value) or value < 0.0:
        raise RecursivePlanError("cannot round a negative or non-finite value")
    return int(math.floor(value + 0.5))


def _ceil_div(numerator: int, denominator: int) -> int:
    if numerator < 0 or denominator <= 0:
        raise RecursivePlanError("invalid ceiling-division operands")
    return -(-numerator // denominator)


def _ceil_nth_root(value: int, degree: int) -> int:
    """Smallest integer ``x`` with ``x**degree >= value``."""

    _plain_int(value, "root value")
    _plain_int(degree, "root degree")
    if value == 1:
        return 1

    # A floating estimate keeps the correction loop short; the loop makes the
    # returned result exact even when the estimate loses precision.
    estimate = max(1, int(math.exp(math.log(value) / degree)))
    while pow(estimate, degree) < value:
        estimate += 1
    while estimate > 1 and pow(estimate - 1, degree) >= value:
        estimate -= 1
    return estimate


def _nearest_nth_root(value: float, degree: int) -> int:
    """Paper's nearest-integer root for the modeled garbage width."""

    _finite_positive(value, "root value")
    _plain_int(degree, "root degree")
    estimate = max(1, _round_nearest_positive(math.exp(math.log(value) / degree)))
    return estimate


def _canonical_json_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise RecursivePlanError(f"plan is not canonically serializable: {exc}") from exc
    return (encoded + "\n").encode("utf-8")


def _attach_plan_digest(payload: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    result["plan_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return result


def verify_plan_digest(plan: Mapping[str, Any]) -> bool:
    encoded_digest = plan.get("plan_sha256")
    if not isinstance(encoded_digest, str):
        return False
    payload = dict(plan)
    del payload["plan_sha256"]
    expected = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return expected == encoded_digest


def _combined_split(
    n: int,
    v_length: int,
    *,
    max_parts: int,
    shape_constant: float,
) -> Dict[str, int]:
    """Choose ``nu,mu`` and pack the single vector ``v=t||g||h``.

    This follows Section 5.3: z0 and z1 are each split into ``nu`` rows and
    the *combined* auxiliary vector is split into exactly ``mu`` rows.  It is
    intentionally not the backend's historical three independent t/g/h
    paddings.
    """

    for value, label in ((n, "n"), (v_length, "v_length"), (max_parts, "max_parts")):
        _plain_int(value, label)
    if not math.isfinite(shape_constant) or shape_constant <= 0.0:
        raise RecursivePlanError("shape_constant must be finite and positive")

    total = 2 * n + v_length
    fraction = (shape_constant / float(total * total)) ** (1.0 / 3.0)
    if n > v_length:
        nu = int(fraction * n + 1.0)
        mu = _ceil_div(v_length * nu, n)
    else:
        mu = int(fraction * v_length + 1.0)
        nu = _ceil_div(n * mu, v_length)

    mu = max(1, mu)
    nu = max(1, nu)
    if mu > max_parts or nu > max_parts:
        raise RecursivePlanError(
            f"derived split (mu={mu}, nu={nu}) exceeds max_parts={max_parts}"
        )
    n_prime = max(_ceil_div(n, nu), _ceil_div(v_length, mu))
    r_prime = 2 * nu + mu
    z_padding = 2 * nu * n_prime - 2 * n
    v_padding = mu * n_prime - v_length
    if z_padding < 0 or v_padding < 0:
        raise AssertionError("combined recursion packing under-allocated its input")

    return {
        "nu": nu,
        "mu": mu,
        "n_prime": n_prime,
        "r_prime": r_prime,
        "z_rows": 2 * nu,
        "v_rows": mu,
        "z_padding_polynomials": z_padding,
        "v_padding_polynomials": v_padding,
        "padded_witness_polynomials": r_prime * n_prime,
    }


def _runtime_sizes(
    *,
    n: int,
    r: int,
    ranks: RankTriple,
    t_length: int,
    g_length: int,
    h_length: int,
    jl_out: int,
) -> Dict[str, Any]:
    sizes = {
        "ajtai_A_polynomials": n * ranks.kappa,
        "ajtai_B_polynomials": t_length * ranks.kappa1,
        "ajtai_C_polynomials": g_length * ranks.kappa1,
        "ajtai_D_polynomials": h_length * ranks.kappa2,
        "jl_working_polynomials": jl_out * r * n,
    }
    sizes["ajtai_total_polynomials"] = sum(
        sizes[name]
        for name in (
            "ajtai_A_polynomials",
            "ajtai_B_polynomials",
            "ajtai_C_polynomials",
            "ajtai_D_polynomials",
        )
    )
    sizes["within_lab_runner_limit"] = (
        max(
            sizes["ajtai_A_polynomials"],
            sizes["ajtai_B_polynomials"],
            sizes["ajtai_C_polynomials"],
            sizes["ajtai_D_polynomials"],
            sizes["jl_working_polynomials"],
        )
        <= lab.MAX_RUNTIME_POLYNOMIALS
        and sizes["ajtai_total_polynomials"] <= lab.MAX_RUNTIME_POLYNOMIALS
    )
    return sizes


def derive_level(
    *,
    index: int,
    n: int,
    r: int,
    beta: float,
    ranks: RankTriple,
    recursed_target: bool,
    max_split_parts: int = PARAMS.recursion.max_split_parts,
    shape_constant: float = PARAMS.recursion.split_shape_constant,
    modulus: int = lab.BABYKOALA_Q,
    degree: int = lab.DEGREE,
    jl_out: int = 256,
) -> Dict[str, Any]:
    """Derive one standard LaBRADOR level using the paper recurrence.

    ``t1`` and ``t2`` are explicit fixed digit counts.  The arbitrary-mod-q
    vectors t and h use ``t1,b1``; the modeled-short garbage vector g uses
    ``t2,b2`` with ``t2 >= 2``.
    """

    if index < 0:
        raise RecursivePlanError("level index must be non-negative")
    for value, label in ((n, "n"), (r, "r"), (modulus, "modulus"), (degree, "degree")):
        _plain_int(value, label)
    beta = _finite_positive(beta, "beta")
    for value, label in (
        (ranks.kappa, "kappa"),
        (ranks.kappa1, "kappa1"),
        (ranks.kappa2, "kappa2"),
    ):
        _plain_int(value, label)

    challenge = PARAMS.labrador_challenge
    tau = challenge.unit_coefficients + 4 * challenge.double_coefficients
    operator_bound = challenge.operator_norm_bound
    if tau <= 0 or not math.isfinite(operator_bound) or operator_bound <= 0.0:
        raise RecursivePlanError("invalid LaBRADOR challenge profile")

    coefficient_sd = beta / math.sqrt(r * n * degree)
    b_real = math.sqrt(coefficient_sd * math.sqrt(12.0 * r * tau))
    b = max(2, _round_nearest_positive(b_real))

    t1 = max(
        PARAMS.recursion.minimum_t1,
        _round_nearest_positive(math.log(modulus) / math.log(b)),
    )
    b1 = _ceil_nth_root(modulus, t1)

    # Section 5.4 models diagonal garbage as the wider case.  The outer s^2
    # factor is essential: sqrt(24*n*d) * s^2.
    garbage_scale = math.sqrt(24.0 * n * degree) * coefficient_sd**2
    if garbage_scale <= 1.0:
        t2_model = 0
    else:
        t2_model = _round_nearest_positive(math.log(garbage_scale) / math.log(b))
    t2 = max(2, t2_model)
    b2 = max(2, _nearest_nth_root(max(1.0, garbage_scale), t2))

    if max(b, b1, b2) > lab.UINT32_MAX:
        raise RecursivePlanError("a paper decomposition base exceeds uint32")

    pair_count = r * (r + 1) // 2
    t_length = r * t1 * ranks.kappa
    g_length = pair_count * t2
    h_length = pair_count * t1
    v_length = t_length + g_length + h_length
    split = _combined_split(
        n,
        v_length,
        max_parts=max_split_parts,
        shape_constant=shape_constant,
    )

    gamma_squared = beta * beta * tau
    gamma1_squared = (
        (b1 * b1 * t1 / 12.0) * r * ranks.kappa * degree
        + (b2 * b2 * t2 / 12.0) * pair_count * degree
    )
    gamma2_squared = (b1 * b1 * t1 / 12.0) * pair_count * degree
    beta_prime_squared = (
        2.0 * gamma_squared / (b * b) + gamma1_squared + gamma2_squared
    )
    beta_prime = math.sqrt(beta_prime_squared)
    for value, label in (
        (coefficient_sd, "coefficient_sd"),
        (garbage_scale, "garbage_scale"),
        (beta_prime, "beta_prime"),
    ):
        _finite_positive(value, label)

    extraction_slack = math.sqrt(
        PARAMS.recursion.extraction_slack_numerator
        / PARAMS.recursion.extraction_slack_denominator
    )
    outer_msis_norm = 2.0 * beta_prime
    inner_msis_norm = max(
        8.0 * operator_bound * (b + 1) * beta_prime,
        2.0 * (b + 1) * beta_prime
        + 4.0 * operator_bound * extraction_slack * beta,
    )
    theorem_multiplier = extraction_slack if recursed_target else 1.0

    modular_jl_limit = (
        math.sqrt(
            PARAMS.recursion.extraction_slack_denominator
            / PARAMS.recursion.extraction_slack_numerator
        )
        * modulus
        / PARAMS.recursion.modular_jl_denominator
    )
    modular_jl_ok = beta <= modular_jl_limit

    runtime_sizes = _runtime_sizes(
        n=n,
        r=r,
        ranks=ranks,
        t_length=t_length,
        g_length=g_length,
        h_length=h_length,
        jl_out=jl_out,
    )

    backend_digits = {
        "t_and_h": lab._balanced_digit_count(modulus, b1),
        "g": lab._balanced_digit_count(modulus, b2),
    }
    fixed_digits_match_legacy_backend = (
        backend_digits["t_and_h"] == t1 and backend_digits["g"] == t2
    )

    return {
        "index": index,
        "relation": {
            "n": n,
            "r": r,
            "beta": beta,
            "witness_polynomials": n * r,
        },
        "ranks": ranks.as_dict(),
        "challenge": {
            "tau_squared_l2_norm": tau,
            "operator_norm_bound": operator_bound,
        },
        "coefficient_standard_deviation": coefficient_sd,
        "decomposition": {
            "z": {"base_real": b_real, "base": b, "fixed_digits": 2},
            "t": {"base": b1, "fixed_digits": t1},
            "g": {
                "modeled_width": garbage_scale,
                "base": b2,
                "fixed_digits": t2,
                "minimum_digits_enforced": 2,
            },
            "h": {"base": b1, "fixed_digits": t1},
            "legacy_full_balanced_digits": backend_digits,
            "fixed_digits_match_legacy_full_decomposition": (
                fixed_digits_match_legacy_backend
            ),
        },
        "target_norm": {
            "gamma_squared": gamma_squared,
            "gamma1_squared": gamma1_squared,
            "gamma2_squared": gamma2_squared,
            "beta_prime_squared": beta_prime_squared,
            "beta_prime": beta_prime,
        },
        "combined_v": {
            "order": ["t", "g", "h"],
            "t_length": t_length,
            "g_length": g_length,
            "h_length": h_length,
            "length": v_length,
            "packing": split,
        },
        "theorem_5_1_msis_inputs": {
            "source": THEOREM,
            "rank_kappa1_equals_kappa2_norm": outer_msis_norm,
            "rank_kappa_norm": inner_msis_norm,
            "target_is_recursively_proven": recursed_target,
            "remark_5_2_multiplier": theorem_multiplier,
            "effective_rank_kappa1_equals_kappa2_norm": (
                outer_msis_norm * theorem_multiplier
            ),
            "effective_rank_kappa_norm": inner_msis_norm * theorem_multiplier,
            "ranks_are_estimator_certified": False,
        },
        "modular_jl": {
            "beta_limit": modular_jl_limit,
            "holds": modular_jl_ok,
        },
        "paper_runtime_sizes": runtime_sizes,
    }


def build_schedule(
    bundle: lab.RelationBundle,
    *,
    executions: int,
    next_ranks: Optional[RankTriple] = None,
    max_split_parts: int = PARAMS.recursion.max_split_parts,
) -> Tuple[Dict[str, Any], ...]:
    """Return the deterministic sequence of standard paper executions."""

    _plain_int(executions, "executions")
    if executions > 64:
        raise RecursivePlanError("executions exceeds the planner safety limit of 64")
    if next_ranks is None:
        next_ranks = RankTriple(bundle.kappa, bundle.kappa1, bundle.kappa2)

    n = bundle.n
    r = bundle.r
    beta = bundle.beta
    ranks = RankTriple(bundle.kappa, bundle.kappa1, bundle.kappa2)
    result = []
    for index in range(executions):
        level = derive_level(
            index=index,
            n=n,
            r=r,
            beta=beta,
            ranks=ranks,
            recursed_target=index < executions - 1,
            max_split_parts=max_split_parts,
            modulus=bundle.modulus,
            degree=bundle.degree,
            jl_out=bundle.jl_out,
        )
        result.append(level)

        if not level["modular_jl"]["holds"]:
            raise RecursivePlanError(
                f"level {index}: beta={beta:.17g} exceeds the modular-JL limit "
                f"{level['modular_jl']['beta_limit']:.17g}"
            )
        packing = level["combined_v"]["packing"]
        n = packing["n_prime"]
        r = packing["r_prime"]
        beta = level["target_norm"]["beta_prime"]
        ranks = next_ranks

    return tuple(result)


def make_plan(
    bundle: lab.RelationBundle,
    *,
    executions: int,
    next_ranks: Optional[RankTriple] = None,
    max_split_parts: int = PARAMS.recursion.max_split_parts,
    prepared_statement_digest: Optional[str] = None,
) -> Dict[str, Any]:
    schedule = build_schedule(
        bundle,
        executions=executions,
        next_ranks=next_ranks,
        max_split_parts=max_split_parts,
    )
    if bundle.public_digest is None:
        # Normal CLI inputs have a digest.  This branch supports callers that
        # construct a validated in-memory bundle.
        parsed = lab.bundle_from_bytes(lab.bundle_to_bytes(bundle))
        statement_digest = parsed.public_digest
    else:
        statement_digest = bundle.public_digest
    if statement_digest is None:
        raise AssertionError("validated relation has no public digest")

    effective_next_ranks = next_ranks
    if effective_next_ranks is None:
        effective_next_ranks = RankTriple(bundle.kappa, bundle.kappa1, bundle.kappa2)

    fixed_digit_mismatches = [
        level["index"]
        for level in schedule
        if not level["decomposition"]["fixed_digits_match_legacy_full_decomposition"]
    ]
    runtime_limit_levels = [
        level["index"]
        for level in schedule
        if not level["paper_runtime_sizes"]["within_lab_runner_limit"]
    ]
    soundness_bits = PARAMS.labrador_challenge.paper_soundness_bits - math.log2(executions)
    payload: Dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "planner_version": PLANNER_VERSION,
        "formula_source": PAPER_SECTION,
        "source_statement_sha3_256": statement_digest.hex(),
        "prepared_statement_sha3_256": prepared_statement_digest,
        "source_parameter_fingerprint": bundle.source_fingerprint,
        "backend": {
            "name": "BabyKoala",
            "degree": bundle.degree,
            "modulus": str(bundle.modulus),
        },
        "executions": executions,
        "transitions": executions - 1,
        "max_split_parts": max_split_parts,
        "next_level_research_ranks": effective_next_ranks.as_dict(),
        "schedule": list(schedule),
        "soundness_accounting": {
            "paper_bits_per_execution": PARAMS.labrador_challenge.paper_soundness_bits,
            "union_bound_bits": soundness_bits,
            "meets_128_bits": soundness_bits >= 128.0,
        },
        "concrete_security_claim": False,
        "backend_compatibility": {
            "v1_backend_derives_full_schedule": (
                2 <= executions <= 8
                and effective_next_ranks
                == RankTriple(bundle.kappa, bundle.kappa1, bundle.kappa2)
                and max_split_parts == PARAMS.recursion.max_split_parts
            ),
            "runner_execution_range": [2, 8],
            "v1_bundle_binds_audit_plan_hash": False,
            "fixed_length_decomposition_implemented": True,
            "levels_differing_from_legacy_full_decomposition": fixed_digit_mismatches,
            "levels_exceeding_current_runner_memory_guard": runtime_limit_levels,
            "combined_v_packing_required": True,
            "section_5_6_final_round_implemented": False,
        },
        "warnings": [
            "Ranks are research inputs, not outputs of a Module-SIS estimator.",
            "The C++ backend must enforce each beta-prime bound or abort the proof attempt.",
            "The v1 .lab bundle does not bind this audit plan's SHA-256 digest.",
            "ICICLE's slow field sampler still needs a modulo-bias security audit.",
            "BabyKoala challenge-difference invertibility is not certified for the paper assumption.",
            "The optimized no-outer-commitment final round from Section 5.6 is not represented.",
        ],
    }
    return _attach_plan_digest(payload)


def prepare_bundle(
    bundle: lab.RelationBundle,
    plan: Mapping[str, Any],
) -> lab.RelationBundle:
    """Apply the v1-compatible subset of a recursive paper plan."""

    if not verify_plan_digest(plan):
        raise RecursivePlanError("recursive plan digest is invalid")
    if plan.get("schema") != PLAN_SCHEMA:
        raise RecursivePlanError(f"plan schema must be {PLAN_SCHEMA!r}")
    source_digest = bundle.public_digest
    if source_digest is None:
        source_digest = lab.bundle_from_bytes(lab.bundle_to_bytes(bundle)).public_digest
    if source_digest is None or plan.get("source_statement_sha3_256") != source_digest.hex():
        raise RecursivePlanError("recursive plan belongs to a different public relation")
    schedule = plan.get("schedule")
    if not isinstance(schedule, list) or not schedule:
        raise RecursivePlanError("plan has no executions")
    first = schedule[0]
    try:
        relation = first["relation"]
        rank_values = first["ranks"]
        decomposition = first["decomposition"]
        base1 = int(decomposition["t"]["base"])
        base2 = int(decomposition["g"]["base"])
        base3 = int(decomposition["h"]["base"])
        executions_value = plan["executions"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RecursivePlanError("plan has malformed initial decomposition parameters") from exc

    if not isinstance(relation, Mapping) or not isinstance(rank_values, Mapping):
        raise RecursivePlanError("plan has malformed first-level relation/ranks")
    executions = _plain_int(executions_value, "plan executions")
    if not 2 <= executions <= 8:
        raise RecursivePlanError(
            "the recursive C++ bridge requires 2..8 executions; use lab.py run for one base execution"
        )
    if len(schedule) != executions:
        raise RecursivePlanError("plan execution count does not match its schedule length")
    if plan.get("max_split_parts") != PARAMS.recursion.max_split_parts:
        raise RecursivePlanError(
            "the v1 C++ bridge supports only para.py's default max_split_parts"
        )
    if (
        relation.get("n") != bundle.n
        or relation.get("r") != bundle.r
        or relation.get("beta") != bundle.beta
        or rank_values.get("kappa") != bundle.kappa
        or rank_values.get("kappa1") != bundle.kappa1
        or rank_values.get("kappa2") != bundle.kappa2
    ):
        raise RecursivePlanError("plan's first level does not match the relation parameters")

    if base3 != base1:
        raise RecursivePlanError("paper schedule requires h to use the t decomposition base")
    expected_ranks = {
        "kappa": bundle.kappa,
        "kappa1": bundle.kappa1,
        "kappa2": bundle.kappa2,
    }
    for level in schedule[1:]:
        if not isinstance(level, Mapping) or level.get("ranks") != expected_ranks:
            raise RecursivePlanError(
                "the v1 C++ bridge carries the initial ranks to every recursive level"
            )
    return replace(
        bundle,
        base1=base1,
        base2=base2,
        base3=base3,
        recursions=executions,
        public_digest=None,
    )


def _write_bytes(path: Path, data: bytes, *, force: bool) -> None:
    # Reuse lab.py's fsync + atomic-replace implementation so both tools have
    # identical overwrite behavior.
    lab._atomic_write(path, data, force=force)


def _write_plan(path: Path, plan: Mapping[str, Any], *, force: bool) -> None:
    _write_bytes(path, _canonical_json_bytes(plan), force=force)


def _require_distinct_paths(left: Path, right: Path, labels: str) -> None:
    if left.resolve() == right.resolve():
        raise RecursivePlanError(f"{labels} must refer to different files")


def _encode_prepared(
    bundle: lab.RelationBundle,
    plan: Mapping[str, Any],
) -> Tuple[bytes, lab.RelationBundle]:
    executions = plan.get("executions")
    if not isinstance(executions, int) or isinstance(executions, bool) or not 2 <= executions <= 8:
        raise RecursivePlanError(
            "the recursive C++ bridge requires 2..8 executions; use lab.py run for one base execution"
        )
    try:
        oversized_levels = plan["backend_compatibility"][
            "levels_exceeding_current_runner_memory_guard"
        ]
    except (KeyError, TypeError) as exc:
        raise RecursivePlanError("plan has no backend memory assessment") from exc
    if oversized_levels:
        joined = ", ".join(str(level) for level in oversized_levels)
        raise RecursivePlanError(
            "recursive level(s) "
            f"{joined} exceed the current runner memory guard; inspect the plan "
            "and adjust dimensions/beta/profile without violating its bounds"
        )
    prepared = prepare_bundle(bundle, plan)
    try:
        encoded = lab.bundle_to_bytes(prepared)
        parsed = lab.bundle_from_bytes(encoded)
    except lab.LabError as exc:
        raise RecursivePlanError(
            f"lab.py rejected the prepared recursive bundle: {exc}"
        ) from exc
    return encoded, parsed


def _rank_args(args: argparse.Namespace, bundle: lab.RelationBundle) -> RankTriple:
    kappa = getattr(args, "next_kappa", None)
    kappa1 = getattr(args, "next_kappa1", None)
    kappa2 = getattr(args, "next_kappa2", None)
    return RankTriple(
        bundle.kappa if kappa is None else kappa,
        bundle.kappa1 if kappa1 is None else kappa1,
        bundle.kappa2 if kappa2 is None else kappa2,
    )


def _add_schedule_arguments(
    parser: argparse.ArgumentParser, *, planner_overrides: bool
) -> None:
    parser.add_argument(
        "--executions",
        type=int,
        default=2,
        help="total protocol executions including the final base case (default: 2)",
    )
    if planner_overrides:
        parser.add_argument("--next-kappa", type=int, help="planner-only kappa after level 0")
        parser.add_argument("--next-kappa1", type=int, help="planner-only kappa1 after level 0")
        parser.add_argument("--next-kappa2", type=int, help="planner-only kappa2 after level 0")
        parser.add_argument(
            "--max-split-parts",
            type=int,
            default=PARAMS.recursion.max_split_parts,
            help="planner-only maximum for each of mu and nu",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan paper-style recursive LaBRADOR schedules and launch the "
            "experimental local C++ backend."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan_parser = commands.add_parser("plan", help="emit a deterministic recursive schedule")
    plan_parser.add_argument("input", type=Path)
    _add_schedule_arguments(plan_parser, planner_overrides=True)
    plan_parser.add_argument("--output", "-o", type=Path)
    plan_parser.add_argument("--force", action="store_true")
    plan_parser.add_argument("--pretty", action="store_true", help="pretty-print stdout only")

    prepare_parser = commands.add_parser(
        "prepare", help="write a v1-compatible recursive .lab bundle"
    )
    prepare_parser.add_argument("input", type=Path)
    prepare_parser.add_argument("--output", "-o", type=Path, required=True)
    prepare_parser.add_argument("--plan-output", type=Path)
    prepare_parser.add_argument("--force", action="store_true")
    _add_schedule_arguments(prepare_parser, planner_overrides=False)

    run_parser = commands.add_parser(
        "run", help="prepare and invoke the experimental recursive C++ runner"
    )
    run_parser.add_argument("input", type=Path)
    run_parser.add_argument("--device", default="CPU")
    run_parser.add_argument("--build", action="store_true")
    run_parser.add_argument(
        "--prepared-output",
        type=Path,
        help="keep the prepared relation instead of using a temporary file",
    )
    run_parser.add_argument("--plan-output", type=Path)
    run_parser.add_argument("--force", action="store_true")
    _add_schedule_arguments(run_parser, planner_overrides=False)

    commands.add_parser("self-test", help="run deterministic planner/bridge tests")
    return parser


def run_self_tests() -> Dict[str, Any]:
    rank = lab.backend_secure_rank()
    fixture = derive_level(
        index=0,
        n=4,
        r=2,
        beta=50.0,
        ranks=RankTriple(rank, rank, rank),
        recursed_target=True,
    )
    assert rank == 37
    assert fixture["decomposition"]["z"]["base"] == 10
    assert fixture["decomposition"]["t"] == {"base": 10, "fixed_digits": 19}
    assert fixture["decomposition"]["g"]["base"] == 7
    assert fixture["decomposition"]["g"]["fixed_digits"] == 3
    assert fixture["decomposition"]["h"] == {"base": 10, "fixed_digits": 19}
    assert math.isclose(
        fixture["target_norm"]["beta_prime_squared"],
        786168.6666666666,
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    packing = fixture["combined_v"]["packing"]
    assert fixture["combined_v"]["length"] == 1472
    assert (packing["nu"], packing["mu"]) == (1, 8)
    assert (packing["n_prime"], packing["r_prime"]) == (184, 10)
    assert packing["r_prime"] == 2 * packing["nu"] + packing["mu"]

    bundle = lab.create_synthetic_bundle(
        r=2,
        n=3,
        witness_bound=1,
        seed=b"labrador-recursive-self-test-v1",
        recursions=1,
    )
    encoded = lab.bundle_to_bytes(bundle)
    bundle = lab.bundle_from_bytes(encoded)
    ranks = RankTriple(rank, rank, rank)
    plan1 = make_plan(bundle, executions=2, next_ranks=ranks)
    plan2 = make_plan(bundle, executions=2, next_ranks=ranks)
    assert _canonical_json_bytes(plan1) == _canonical_json_bytes(plan2)
    assert verify_plan_digest(plan1)
    tampered = dict(plan1)
    tampered["executions"] = 3
    assert not verify_plan_digest(tampered)
    assert len(plan1["schedule"]) == 2
    for level in plan1["schedule"]:
        assert level["decomposition"]["g"]["fixed_digits"] >= 2
        split = level["combined_v"]["packing"]
        assert split["r_prime"] == 2 * split["nu"] + split["mu"]
        assert level["modular_jl"]["holds"]

    prepared = prepare_bundle(bundle, plan1)
    first_decomposition = plan1["schedule"][0]["decomposition"]
    assert prepared.recursions == 2
    assert prepared.base1 == first_decomposition["t"]["base"]
    assert prepared.base2 == first_decomposition["g"]["base"]
    assert prepared.base3 == first_decomposition["h"]["base"]
    assert prepared.public_digest is None

    recursive_encoded = lab.bundle_to_bytes(prepared)
    recursive_decoded = lab.bundle_from_bytes(recursive_encoded)
    assert recursive_decoded.recursions == 2
    codec_status = "recursive bundle accepted"

    return {
        "status": "PASS",
        "backend_secure_rank_heuristic": rank,
        "fixture_beta_prime": fixture["target_norm"]["beta_prime"],
        "fixture_combined_packing": packing,
        "deterministic_plan_sha256": plan1["plan_sha256"],
        "codec_status": codec_status,
        "concrete_security_claim": False,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "self-test":
            print(json.dumps(run_self_tests(), indent=2, sort_keys=True))
            return 0

        bundle, _ = lab.read_bundle(args.input)
        ranks = _rank_args(args, bundle)
        plan = make_plan(
            bundle,
            executions=args.executions,
            next_ranks=ranks,
            max_split_parts=getattr(
                args, "max_split_parts", PARAMS.recursion.max_split_parts
            ),
        )

        if args.command == "plan":
            if args.output is not None:
                _require_distinct_paths(args.input, args.output, "input and plan output")
                _write_plan(args.output, plan, force=args.force)
            else:
                if args.pretty:
                    print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
                else:
                    sys.stdout.buffer.write(_canonical_json_bytes(plan))
            return 0

        encoded, parsed = _encode_prepared(bundle, plan)
        final_plan_payload = dict(plan)
        del final_plan_payload["plan_sha256"]
        final_plan_payload["prepared_statement_sha3_256"] = (
            parsed.public_digest.hex() if parsed.public_digest is not None else None
        )
        final_plan = _attach_plan_digest(final_plan_payload)

        if args.command == "prepare":
            _require_distinct_paths(args.input, args.output, "input and prepared output")
            if args.plan_output is not None:
                _require_distinct_paths(args.input, args.plan_output, "input and plan output")
                _require_distinct_paths(
                    args.output, args.plan_output, "prepared output and plan output"
                )
            _write_bytes(args.output, encoded, force=args.force)
            if args.plan_output is not None:
                _write_plan(args.plan_output, final_plan, force=args.force)
            print(
                json.dumps(
                    {
                        "prepared_relation": str(args.output.resolve()),
                        "executions": args.executions,
                        "plan_sha256": final_plan["plan_sha256"],
                        "public_statement_sha3_256": (
                            parsed.public_digest.hex() if parsed.public_digest else None
                        ),
                        "contains_plaintext_secret_witness": True,
                        "is_proof_file": False,
                        "concrete_security_claim": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "run":
            repo_root = Path(__file__).resolve().parents[1]
            if args.build:
                if args.device.upper() != "CPU":
                    raise RecursivePlanError("automatic --build supports only the CPU backend")
                lab._build_cpu(repo_root)
            runner = repo_root / "build" / "src" / "lab_runner"
            if not runner.is_file():
                raise RecursivePlanError(f"runner not found at {runner}; pass --build first")

            print(
                "WARNING: launching a research recursive execution; this is not a "
                "concrete-security or production-proof claim.",
                file=sys.stderr,
            )
            print(
                "WARNING: any prepared .lab output contains the plaintext secret witness; "
                "neither it nor the JSON plan is a serialized proof.",
                file=sys.stderr,
            )
            print(
                "WARNING: the C++ backend derives the fixed-t1/t2 schedule from the "
                "v1 input, but the auxiliary plan SHA-256 is not serialized.",
                file=sys.stderr,
            )

            temporary_directory: Optional[tempfile.TemporaryDirectory[str]] = None
            if args.plan_output is not None:
                _require_distinct_paths(args.input, args.plan_output, "input and plan output")
            if args.prepared_output is not None:
                _require_distinct_paths(
                    args.input, args.prepared_output, "input and prepared output"
                )
                if args.plan_output is not None:
                    _require_distinct_paths(
                        args.prepared_output,
                        args.plan_output,
                        "prepared output and plan output",
                    )
                _write_bytes(args.prepared_output, encoded, force=args.force)
                run_path = args.prepared_output.resolve()
            else:
                temporary_directory = tempfile.TemporaryDirectory(prefix="lnplab-recursive-")
                run_path = Path(temporary_directory.name) / "relation.lab"
                _write_bytes(run_path, encoded, force=False)
            try:
                if args.plan_output is not None:
                    _write_plan(args.plan_output, final_plan, force=args.force)
                result = subprocess.run(
                    (str(runner), args.device, str(run_path)),
                    cwd=repo_root,
                    check=False,
                )
                return result.returncode
            finally:
                if temporary_directory is not None:
                    temporary_directory.cleanup()

        raise AssertionError(f"unhandled command {args.command}")
    except (
        RecursivePlanError,
        lab.LabError,
        OSError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"labrador.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
