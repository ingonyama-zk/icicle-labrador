#!/usr/bin/env python3
"""Reference parameter tool for the LNP--LaBRADOR NIBS relation.

This program accompanies ``lnp_labrador_nibs_relation_detailed.tex``.  It:

* reproduces the fixed/derived numbers and no-wrap checks in the TeX;
* calibrates and samples the LNP22 conjugation-fixed challenge space;
* measures the concrete LaBRADOR challenge filter and its post-filter entropy;
* evaluates the LaBRADOR recursion formulas (paper and current-TeX variants);
* reproduces the draft composition/proof-size accounting.

It is deliberately a parameter-selection and audit tool, not a prover,
verifier, Fiat--Shamir implementation, or lattice-security estimator.  All
editable protocol values live in ``para.py``.

Primary references:
  LNP22, Section 2.7:       https://eprint.iacr.org/2022/284.pdf
  LaBRADOR, Sections 2/5:  https://eprint.iacr.org/2022/1341.pdf
  TRaccoon, Appendix B.6:  https://eprint.iacr.org/2025/849.pdf
"""

from __future__ import annotations

import argparse
import cmath
import hashlib
import json
import math
import random
import statistics
import sys
from dataclasses import asdict
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # Chạy được cả ``python scripts/lnplab.py`` và import như package.
    from .para import PARAMS, SOURCES, Parameters
except ImportError:
    from para import PARAMS, SOURCES, Parameters


class ParameterError(ValueError):
    """Raised when the declarative parameter set is internally inconsistent."""


# ---------------------------------------------------------------------------
# Small exact/numerical helpers
# ---------------------------------------------------------------------------


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-numerator // denominator)


def round_nearest_positive(value: float) -> int:
    """Nearest integer for non-negative values, with halves rounded upward."""

    if value < 0:
        raise ValueError("value must be non-negative")
    return int(math.floor(value + 0.5))


def ceil_nth_root(value: int, degree: int) -> int:
    """Smallest integer x such that x**degree >= value."""

    if value < 0 or degree <= 0:
        raise ValueError("invalid root arguments")
    if value <= 1:
        return value
    candidate = max(1, int(math.ceil(math.exp(math.log(value) / degree))))
    while pow(candidate, degree) < value:
        candidate += 1
    while candidate > 1 and pow(candidate - 1, degree) >= value:
        candidate -= 1
    return candidate


def nearest_nth_root(value: float, degree: int) -> int:
    if value <= 0 or degree <= 0:
        raise ValueError("invalid root arguments")
    return max(1, round_nearest_positive(math.exp(math.log(value) / degree)))


def percentile(values: Sequence[float], probability: float) -> float:
    """Nearest-rank percentile, matching a rejection-threshold use case."""

    if not values:
        raise ValueError("empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(values)
    if probability == 0.0:
        return ordered[0]
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def distribution_summary(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def wilson_interval(successes: int, trials: int, confidence: float) -> Tuple[float, float]:
    """Two-sided Wilson interval for a Bernoulli probability."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid Bernoulli observations")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)
    observed = successes / trials
    z2_over_n = z * z / trials
    center = (observed + z2_over_n / 2.0) / (1.0 + z2_over_n)
    radius = (
        z
        * math.sqrt(observed * (1.0 - observed) / trials + z * z / (4.0 * trials * trials))
        / (1.0 + z2_over_n)
    )
    return max(0.0, center - radius), min(1.0, center + radius)


@lru_cache(maxsize=None)
def is_probable_prime(value: int) -> bool:
    """Deterministic Miller--Rabin for unsigned 64-bit integers."""

    if value < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if value == prime:
            return True
        if value % prime == 0:
            return False

    odd_part = value - 1
    twos = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        twos += 1

    # Jim Sinclair et al. deterministic base set for n < 2**64.
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(twos - 1):
            witness = witness * witness % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def parameter_fingerprint(parameters: Parameters = PARAMS) -> str:
    encoded = json.dumps(asdict(parameters), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def artifact_fingerprint(
    artifact: str,
    resolved_values: Dict[str, Any],
    parameters: Parameters = PARAMS,
) -> str:
    """Bind command-line overrides to a reproducible test-vector artifact."""

    payload = {
        "artifact": artifact,
        "parameter_fingerprint_sha256": parameter_fingerprint(parameters),
        "resolved_values": resolved_values,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


# ---------------------------------------------------------------------------
# Negacyclic ring arithmetic and operator norm
# ---------------------------------------------------------------------------


def _same_degree(left: Sequence[int], right: Sequence[int]) -> int:
    if not left or len(left) != len(right):
        raise ValueError("polynomials must have the same positive degree")
    return len(left)


def negacyclic_mul(left: Sequence[int], right: Sequence[int]) -> List[int]:
    """Multiply in Z[X]/(X^d+1), without reducing coefficients modulo q."""

    degree = _same_degree(left, right)
    convolution = [0] * (2 * degree - 1)
    for i, left_coefficient in enumerate(left):
        if left_coefficient == 0:
            continue
        for j, right_coefficient in enumerate(right):
            if right_coefficient:
                convolution[i + j] += left_coefficient * right_coefficient
    result = convolution[:degree]
    for index in range(degree, len(convolution)):
        result[index - degree] -= convolution[index]
    return result


def negacyclic_square(polynomial: Sequence[int]) -> List[int]:
    """Faster exact square in Z[X]/(X^d+1)."""

    degree = len(polynomial)
    if degree == 0:
        raise ValueError("polynomial must be non-empty")
    convolution = [0] * (2 * degree - 1)
    for i, coefficient in enumerate(polynomial):
        if coefficient == 0:
            continue
        convolution[2 * i] += coefficient * coefficient
        doubled = 2 * coefficient
        for j in range(i + 1, degree):
            if polynomial[j]:
                convolution[i + j] += doubled * polynomial[j]
    result = convolution[:degree]
    for index in range(degree, len(convolution)):
        result[index - degree] -= convolution[index]
    return result


def negacyclic_pow(polynomial: Sequence[int], exponent: int) -> List[int]:
    if exponent < 0:
        raise ValueError("negative exponents are not supported")
    degree = len(polynomial)
    result = [1] + [0] * (degree - 1)
    base = list(polynomial)
    while exponent:
        if exponent & 1:
            result = negacyclic_mul(result, base)
        exponent >>= 1
        if exponent:
            base = negacyclic_square(base)
    return result


def conjugate(polynomial: Sequence[int]) -> List[int]:
    """sigma_{-1}: X -> X^{-1}=-X^(d-1) in the negacyclic ring."""

    degree = len(polynomial)
    if degree == 0:
        raise ValueError("polynomial must be non-empty")
    result = [0] * degree
    result[0] = polynomial[0]
    for index in range(1, degree):
        result[degree - index] = -polynomial[index]
    return result


def is_conjugation_fixed(polynomial: Sequence[int]) -> bool:
    return list(polynomial) == conjugate(polynomial)


def lnp_multiplier_measure(polynomial: Sequence[int], power: int) -> int:
    """Return the exact integer ||sigma(c^k)c^k||_1."""

    if not is_power_of_two(power):
        raise ValueError("LNP power k must be a power of two")

    # Challenges are sigma-fixed.  Repeated squaring is both exact and much
    # faster than generic multiplication for the k=32 calibration.
    powered = list(polynomial)
    for _ in range(power.bit_length() - 1):
        powered = negacyclic_square(powered)
    sigma_powered = conjugate(powered)
    product = (
        negacyclic_square(powered)
        if sigma_powered == powered
        else negacyclic_mul(sigma_powered, powered)
    )
    return sum(abs(coefficient) for coefficient in product)


def lnp_multiplier_bound(polynomial: Sequence[int], power: int) -> float:
    """Floating rendering of the LNP multiplier bound; not used for membership."""

    l1_norm = lnp_multiplier_measure(polynomial, power)
    return lnp_bound_from_measure(l1_norm, power)


def lnp_bound_from_measure(l1_norm: int, power: int) -> float:
    if l1_norm < 0 or not is_power_of_two(power):
        raise ValueError("invalid LNP multiplier measure/power")
    if l1_norm == 0:
        return 0.0
    return math.exp(math.log(l1_norm) / (2.0 * power))


def lnp_challenge_accepts(
    polynomial: Sequence[int], power: int, eta: int
) -> bool:
    """Exact predicate ||sigma(c^k)c^k||_1 <= eta^(2k)."""

    if eta <= 0:
        raise ValueError("eta must be positive")
    return lnp_multiplier_measure(polynomial, power) <= pow(eta, 2 * power)


def _fft(values: Sequence[complex]) -> List[complex]:
    """Iterative radix-2 FFT; used only for degree-64 norm calibration."""

    size = len(values)
    if not is_power_of_two(size):
        raise ValueError("FFT length must be a power of two")
    output = [complex(value) for value in values]

    j = 0
    for i in range(1, size):
        bit = size >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            output[i], output[j] = output[j], output[i]

    length = 2
    while length <= size:
        root = cmath.exp(-2j * math.pi / length)
        half = length // 2
        for start in range(0, size, length):
            twiddle = 1.0 + 0.0j
            for offset in range(half):
                even = output[start + offset]
                odd = output[start + offset + half] * twiddle
                output[start + offset] = even + odd
                output[start + offset + half] = even - odd
                twiddle *= root
        length <<= 1
    return output


@lru_cache(maxsize=None)
def _negacyclic_twist(degree: int) -> Tuple[complex, ...]:
    return tuple(cmath.exp(1j * math.pi * index / degree) for index in range(degree))


def operator_norm(polynomial: Sequence[int]) -> float:
    """Spectral norm of multiplication by c in R=Z[X]/(X^d+1)."""

    degree = len(polynomial)
    if not is_power_of_two(degree):
        raise ValueError("ring degree must be a power of two")
    twisted = [coefficient * twist for coefficient, twist in zip(polynomial, _negacyclic_twist(degree))]
    return max(abs(value) for value in _fft(twisted))


# ---------------------------------------------------------------------------
# Challenge candidate generation
# ---------------------------------------------------------------------------


def _randbelow(generator: Any, upper_bound: int) -> int:
    if hasattr(generator, "randbelow"):
        return generator.randbelow(upper_bound)
    return generator.randrange(upper_bound)


def sample_lnp_candidate(generator: Any, degree: int, coefficient_bound: int) -> List[int]:
    """Uniform sample from S_kappa fixed by conjugation sigma_{-1}."""

    if degree % 2 or coefficient_bound < 0:
        raise ValueError("invalid conjugation-fixed challenge parameters")
    width = 2 * coefficient_bound + 1
    polynomial = [0] * degree
    polynomial[0] = _randbelow(generator, width) - coefficient_bound
    for index in range(1, degree // 2):
        coefficient = _randbelow(generator, width) - coefficient_bound
        polynomial[index] = coefficient
        polynomial[degree - index] = -coefficient
    # The X^(d/2) coefficient is necessarily zero under sigma(c)=c.
    return polynomial


def sample_labrador_candidate(
    generator: Any,
    degree: int,
    zero_coefficients: int,
    unit_coefficients: int,
    double_coefficients: int,
) -> List[int]:
    if zero_coefficients + unit_coefficients + double_coefficients != degree:
        raise ValueError("LaBRADOR challenge weights do not sum to the ring degree")
    magnitudes = (
        [0] * zero_coefficients
        + [1] * unit_coefficients
        + [2] * double_coefficients
    )
    # Fisher--Yates is uniform over the labelled list and therefore uniform
    # over distinct multiset permutations as every outcome has equal multiplicity.
    for index in range(degree - 1, 0, -1):
        swap_index = _randbelow(generator, index + 1)
        magnitudes[index], magnitudes[swap_index] = magnitudes[swap_index], magnitudes[index]
    for index, magnitude in enumerate(magnitudes):
        if magnitude and _randbelow(generator, 2):
            magnitudes[index] = -magnitude
    return magnitudes


def labrador_norm_accepts(
    norm: float,
    bound: float,
    strict: bool,
    numerical_guard: float = 0.0,
) -> bool:
    """Conservative numerical predicate for reference calibration/test vectors."""

    guarded_bound = bound - numerical_guard
    return norm < guarded_bound if strict else norm <= guarded_bound


class ShakeRng:
    """Small deterministic SHAKE256 counter-mode reader for test vectors.

    This gives reproducible, domain-separated reference samples.  A production
    Fiat--Shamir expander must additionally bind the canonical CRS, statement,
    transcript prefix, dimensions and protocol version exactly as specified by
    the implementation.
    """

    def __init__(self, seed: bytes, domain: bytes):
        self._seed = bytes(seed)
        self._domain = bytes(domain)
        self._counter = 0
        self._buffer = bytearray()

    def _refill(self) -> None:
        payload = (
            len(self._domain).to_bytes(4, "big")
            + self._domain
            + len(self._seed).to_bytes(8, "big")
            + self._seed
            + self._counter.to_bytes(8, "big")
        )
        self._buffer.extend(hashlib.shake_256(payload).digest(64))
        self._counter += 1

    def read(self, length: int) -> bytes:
        while len(self._buffer) < length:
            self._refill()
        result = bytes(self._buffer[:length])
        del self._buffer[:length]
        return result

    def randbelow(self, upper_bound: int) -> int:
        if upper_bound <= 0:
            raise ValueError("upper_bound must be positive")
        bits = (upper_bound - 1).bit_length()
        byte_length = max(1, ceil_div(bits, 8))
        mask = (1 << bits) - 1
        while True:
            candidate = int.from_bytes(self.read(byte_length), "big") & mask
            if candidate < upper_bound:
                return candidate


def sample_centered_binomial_1(generator: Any) -> int:
    """Sample -1,0,1 with probabilities 1/4,1/2,1/4."""

    return _randbelow(generator, 2) - _randbelow(generator, 2)


def sample_lnp_challenge_from_seed(
    seed: bytes,
    eta: int,
    parameters: Parameters = PARAMS,
) -> Tuple[List[int], int, float]:
    validate_parameters(parameters)
    if eta <= 0:
        raise ValueError("eta must be positive")
    cfg = parameters.lnp_challenge
    domain = (
        parameters.composition.lnp_fs_domains[3] + "/reference-sampler"
    ).encode("utf-8")
    generator = ShakeRng(seed, domain)
    for attempt in range(1, cfg.max_sampling_attempts + 1):
        candidate = sample_lnp_candidate(
            generator, parameters.ring.degree, cfg.coefficient_bound
        )
        measure = lnp_multiplier_measure(candidate, cfg.power)
        if measure <= pow(eta, 2 * cfg.power):
            return candidate, attempt, lnp_bound_from_measure(measure, cfg.power)
    raise RuntimeError("LNP challenge rejection sampler exceeded max_sampling_attempts")


def sample_labrador_challenge_from_seed(
    seed: bytes,
    parameters: Parameters = PARAMS,
    level: int = 0,
) -> Tuple[List[int], int, float]:
    validate_parameters(parameters)
    if level < 0:
        raise ValueError("recursion level must be non-negative")
    cfg = parameters.labrador_challenge
    domain = (
        parameters.composition.labrador_fs_domains[3]
        + f"/level/{level}/reference-sampler"
    ).encode("utf-8")
    generator = ShakeRng(seed, domain)
    for attempt in range(1, cfg.max_sampling_attempts + 1):
        candidate = sample_labrador_candidate(
            generator,
            parameters.ring.degree,
            cfg.zero_coefficients,
            cfg.unit_coefficients,
            cfg.double_coefficients,
        )
        norm = operator_norm(candidate)
        if labrador_norm_accepts(
            norm,
            cfg.operator_norm_bound,
            cfg.strict_operator_bound,
            cfg.operator_norm_numerical_guard,
        ):
            return candidate, attempt, norm
    raise RuntimeError("LaBRADOR challenge rejection sampler exceeded max_sampling_attempts")


# ---------------------------------------------------------------------------
# Core TeX derivations
# ---------------------------------------------------------------------------


def validate_parameters(parameters: Parameters = PARAMS) -> None:
    ring = parameters.ring
    app = parameters.application
    projection = parameters.projection
    lnp = parameters.lnp_challenge
    lab = parameters.labrador_challenge
    rejection = parameters.rejection
    commitment = parameters.commitment
    recursion = parameters.recursion
    composition = parameters.composition

    errors: List[str] = []
    if not is_power_of_two(ring.degree) or ring.degree < 2:
        errors.append("ring.degree must be a power of two >= 2")
    if min(ring.q_signature, ring.q_omuf, ring.q_proof) <= 1:
        errors.append("all moduli must be greater than one")
    if ring.q_proof >= 2**64:
        errors.append("q_proof must be below 2^64 for the deterministic primality test")
    if ring.rounding_factor <= 0:
        errors.append("rounding_factor must be positive")
    elif ring.q_signature % ring.rounding_factor:
        errors.append("q_signature must be divisible by rounding_factor")
    if ring.q_proof > 1 and not is_probable_prime(ring.q_proof):
        errors.append("q_proof must be prime for the theorem checks implemented here")
    if ring.q_proof > 1 and ring.q_proof % 8 != 5:
        errors.append("q_proof must be congruent to 5 modulo 8")
    if min(
        app.security_bits,
        app.recipient_dimension_m,
        app.preimage_dimension_k,
        app.selector_length,
        app.bounded_arithmetic_blocks,
    ) <= 0:
        errors.append("application dimensions must be positive")
    if min(app.pre_signature_bound, app.recipient_secret_bound) <= 0:
        errors.append("application Euclidean bounds must be positive")
    if app.error_coefficient_bound < 0:
        errors.append("error_coefficient_bound must be non-negative")
    if (
        projection.jl_rows <= 0
        or projection.c1 <= 0
        or projection.c2_numerator <= 0
        or projection.c2_denominator <= 0
    ):
        errors.append("invalid projection dimensions/constants")
    if projection.centered_binomial_parameter != 1:
        errors.append("this implementation and the C1/C2 constants require centered Bin_1")
    if not math.isfinite(projection.gaussian_multiplier) or projection.gaussian_multiplier <= 0:
        errors.append("gaussian_multiplier must be finite and positive")
    if not math.isfinite(projection.tail_factor) or projection.tail_factor <= 0:
        errors.append("tail_factor must be finite and positive")
    if projection.projected_extra_ring_coordinates < 0:
        errors.append("projected_extra_ring_coordinates must be non-negative")
    if projection.no_wrap_linear_factor <= 0:
        errors.append("no_wrap_linear_factor must be positive")
    if projection.jl_failure_bits_each <= 0 or projection.jl_failure_events <= 0:
        errors.append("JL failure-event parameters must be positive")
    if projection.tail_dimension_mode not in ("tex_ring_degree", "jl_rows"):
        errors.append("tail_dimension_mode must be 'tex_ring_degree' or 'jl_rows'")
    if projection.tail_width_mode not in ("raw_formula", "ceil"):
        errors.append("tail_width_mode must be 'raw_formula' or 'ceil'")
    if not is_power_of_two(lnp.power):
        errors.append("LNP challenge power must be a power of two")
    if lnp.coefficient_bound < 1:
        errors.append("LNP coefficient bound must be positive")
    if not 0.0 < lnp.target_acceptance <= 1.0:
        errors.append("LNP target_acceptance must be in (0, 1]")
    if lnp.eta is not None and lnp.eta <= 0:
        errors.append("configured LNP eta must be positive")
    if lnp.eta_is_final and lnp.eta is None:
        errors.append("eta_is_final requires a configured eta")
    if min(lnp.calibration_samples, lnp.validation_samples, lnp.max_sampling_attempts) <= 0:
        errors.append("LNP sample counts/attempt limit must be positive")
    if not 0.0 < lnp.confidence < 1.0:
        errors.append("LNP confidence must be in (0, 1)")
    if min(lab.zero_coefficients, lab.unit_coefficients, lab.double_coefficients) < 0:
        errors.append("LaBRADOR coefficient counts must be non-negative")
    if lab.zero_coefficients + lab.unit_coefficients + lab.double_coefficients != ring.degree:
        errors.append("LaBRADOR coefficient counts must sum to ring.degree")
    if not math.isfinite(lab.operator_norm_bound) or lab.operator_norm_bound <= 0:
        errors.append("LaBRADOR operator norm bound must be positive")
    if (
        not math.isfinite(lab.operator_norm_numerical_guard)
        or lab.operator_norm_numerical_guard < 0
        or lab.operator_norm_numerical_guard >= lab.operator_norm_bound
    ):
        errors.append("invalid LaBRADOR numerical operator-norm guard")
    if min(lab.validation_samples, lab.paper_soundness_bits, lab.max_sampling_attempts) <= 0:
        errors.append("LaBRADOR sample/soundness parameters must be positive")
    if not 0.0 < lab.confidence < 1.0:
        errors.append("LaBRADOR confidence must be in (0, 1)")
    if min(rejection.gamma_ajtai_message, rejection.gamma_randomness) <= 0:
        errors.append("LNP rejection factors must be positive")
    if not all(
        math.isfinite(value)
        for value in (rejection.gamma_ajtai_message, rejection.gamma_randomness)
    ):
        errors.append("LNP rejection factors must be finite")
    if commitment.kappa_msis <= 0 or (
        commitment.kappa_ajtai is not None and commitment.kappa_ajtai <= 0
    ):
        errors.append("commitment ranks must be positive")
    if min(commitment.compact_transcript_flows, commitment.constant_term_mask_commitments) < 0:
        errors.append("commitment message counts must be non-negative")
    if recursion.formula_mode not in ("paper", "tex"):
        errors.append("recursion.formula_mode must be 'paper' or 'tex'")
    if (
        not math.isfinite(recursion.root_hermite_delta)
        or recursion.root_hermite_delta <= 1.0
    ):
        errors.append("root_hermite_delta must be finite and greater than one")
    if min(
        recursion.max_split_parts,
        recursion.minimum_t1,
        recursion.extraction_slack_numerator,
        recursion.extraction_slack_denominator,
        recursion.modular_jl_denominator,
    ) <= 0:
        errors.append("recursion integer parameters must be positive")
    if not math.isfinite(recursion.split_shape_constant) or recursion.split_shape_constant <= 0:
        errors.append("split_shape_constant must be finite and positive")
    if min(composition.boundary_binary_constraints, composition.fs_seed_bits) <= 0:
        errors.append("composition constraint/seed sizes must be positive")
    all_domains = composition.lnp_fs_domains + composition.labrador_fs_domains
    if len(composition.lnp_fs_domains) != 4 or len(composition.labrador_fs_domains) != 4:
        errors.append("LNP and LaBRADOR each require four Fiat--Shamir domains")
    if not all(all_domains) or len(set(all_domains)) != len(all_domains):
        errors.append("Fiat--Shamir domains must be distinct")
    if not composition.schedule:
        errors.append("draft recursion schedule must not be empty")
    for row in composition.schedule:
        if row.witness_kib < 0 or row.output_kib < 0:
            errors.append("draft schedule sizes must be non-negative")
            break
        if (row.rank_n is None) != (row.multiplicity_r is None):
            errors.append("draft schedule rank/multiplicity must both be set or both be None")
            break
        if row.rank_n is not None and min(row.rank_n, row.multiplicity_r or 0) <= 0:
            errors.append("draft schedule rank/multiplicity must be positive")
            break
    if errors:
        raise ParameterError("; ".join(errors))


def _tail_dimension(parameters: Parameters, mode: str) -> int:
    if mode == "tex_ring_degree":
        return parameters.ring.degree
    if mode == "jl_rows":
        return parameters.projection.jl_rows
    raise ValueError(f"unknown projection tail dimension mode: {mode}")


def jl_reference_assumptions_hold(parameters: Parameters = PARAMS) -> bool:
    """Whether the configured JL claim is no stronger than the cited tuple."""

    projection = parameters.projection
    return (
        projection.jl_rows == 256
        and projection.centered_binomial_parameter == 1
        and projection.c1 == 337
        and projection.c2_numerator * 2 == 13 * projection.c2_denominator
        and projection.tail_factor > 1.64
        and projection.no_wrap_linear_factor >= 41
        and projection.jl_failure_bits_each <= 128
        and projection.jl_failure_events >= 2
    )


def derive_core(
    parameters: Parameters = PARAMS,
    tail_dimension_mode: Optional[str] = None,
    tail_width_mode: Optional[str] = None,
) -> Dict[str, Any]:
    validate_parameters(parameters)
    ring = parameters.ring
    app = parameters.application
    projection = parameters.projection
    commitment = parameters.commitment
    mode = tail_dimension_mode or projection.tail_dimension_mode
    width_mode = tail_width_mode or projection.tail_width_mode
    if width_mode not in ("raw_formula", "ceil"):
        raise ValueError(f"unknown projection tail width mode: {width_mode}")

    degree = ring.degree
    c2 = projection.c2_numerator / projection.c2_denominator
    rounding_modulus = ring.q_signature // ring.rounding_factor
    error_l2_bound = math.ceil(
        app.error_coefficient_bound * math.sqrt(app.recipient_dimension_m * degree)
    )
    tau_cc = math.ceil(app.security_bits / math.log2(ring.q_proof))

    projected_ring_coordinates = (
        2 * app.preimage_dimension_k
        + 3 * app.recipient_dimension_m
        + projection.projected_extra_ring_coordinates
        + app.bounded_arithmetic_blocks
    )
    projected_integer_dimension = projected_ring_coordinates * degree

    gaussian_variance_term = (
        app.pre_signature_bound**2
        + app.recipient_secret_bound**2
        + error_l2_bound**2
        + error_l2_bound**2
        + (
            projection.projected_extra_ring_coordinates
            + app.bounded_arithmetic_blocks
        )
        * degree
    )
    s_projection_raw = (
        projection.gaussian_multiplier
        * math.sqrt(projection.c1 * gaussian_variance_term)
    )
    s_projection = math.ceil(s_projection_raw)
    tail_width = s_projection_raw if width_mode == "raw_formula" else s_projection
    tail_dimension = _tail_dimension(parameters, mode)
    extracted_bound = math.ceil(
        projection.tail_factor * tail_width * math.sqrt(tail_dimension / c2)
    )
    verifier_jl_bound = math.ceil(math.sqrt(c2) * extracted_bound)

    t1 = extracted_bound**2 + math.sqrt(projected_ring_coordinates * degree) * extracted_bound
    t2 = (
        projection.no_wrap_linear_factor
        * degree
        * projected_ring_coordinates
        * math.sqrt(c2)
        * extracted_bound
    )
    t_pre = extracted_bound**2 + 2 * app.pre_signature_bound**2
    t_secret = extracted_bound**2 + 2 * app.recipient_secret_bound**2
    t_error = extracted_bound**2 + 2 * error_l2_bound**2
    no_wrap_terms = {
        "T1": t1,
        "T2": t2,
        "T_pre": t_pre,
        "T_s": t_secret,
        "T_r": t_error,
        "T_M": t_error,
    }
    largest_name, largest_term = max(no_wrap_terms.items(), key=lambda item: item[1])

    n_full = (
        app.recipient_dimension_m
        + app.preimage_dimension_k
        + app.selector_length * app.recipient_dimension_m
        + app.recipient_dimension_m
    )
    range_bits_per_sign = math.ceil(math.log2(2 * app.error_coefficient_bound + 1))
    range_ring_planes_per_error = 2 * range_bits_per_sign * app.recipient_dimension_m
    selector_nonzero_bits = math.ceil(math.log2(app.selector_length))

    modulus_bits = ring.q_proof.bit_length()
    compact_commitment_bits = commitment.kappa_msis * degree * modulus_bits
    ring_element_bits = degree * modulus_bits
    response_signed_bits = max(1, (2 * verifier_jl_bound).bit_length())
    projection_response_bits = projection.jl_rows * response_signed_bits

    return {
        "tail_dimension_mode": mode,
        "tail_width_mode": width_mode,
        "ring_degree": degree,
        "q_signature": ring.q_signature,
        "rounding_modulus": rounding_modulus,
        "rounding_modulus_is_prime": is_probable_prime(rounding_modulus),
        "q_omuf": ring.q_omuf,
        "q_proof": ring.q_proof,
        "q_proof_bits": modulus_bits,
        "q_proof_is_prime": is_probable_prime(ring.q_proof),
        "q_proof_mod_8": ring.q_proof % 8,
        "error_l2_bound": error_l2_bound,
        "constant_coefficient_repetitions": tau_cc,
        "full_ring_equations": n_full,
        "range_bits_per_sign": range_bits_per_sign,
        "range_ring_planes_per_error": range_ring_planes_per_error,
        "selector_nonzero_bits": selector_nonzero_bits,
        "projected_ring_coordinates": projected_ring_coordinates,
        "projected_integer_dimension": projected_integer_dimension,
        "projection_gaussian_width": s_projection,
        "projection_gaussian_width_raw": s_projection_raw,
        "projection_tail_width_used": tail_width,
        "projection_tail_dimension": tail_dimension,
        "extracted_short_vector_bound": extracted_bound,
        "verifier_jl_bound": verifier_jl_bound,
        "projection_response_signed_bits": response_signed_bits,
        "projection_response_bytes": ceil_div(projection_response_bits, 8),
        "jl_distribution": f"centered_Bin_{projection.centered_binomial_parameter}",
        "jl_failure_bits_each": projection.jl_failure_bits_each,
        "jl_failure_events": projection.jl_failure_events,
        "jl_reference_assumptions_hold": jl_reference_assumptions_hold(parameters),
        "final_lnp_rejection_factors": {
            "gamma_ajtai_message": parameters.rejection.gamma_ajtai_message,
            "gamma_randomness": parameters.rejection.gamma_randomness,
        },
        "final_response_bound": None,
        "final_response_parameters_complete": False,
        "compact_commitment_bytes": ceil_div(compact_commitment_bits, 8),
        "ring_element_bytes": ceil_div(ring_element_bits, 8),
        "no_wrap_terms": no_wrap_terms,
        "largest_no_wrap_term_name": largest_name,
        "largest_no_wrap_term": largest_term,
        "no_wrap_ok": largest_term < ring.q_proof,
        "no_wrap_margin": ring.q_proof / largest_term,
    }


def derive_size_accounting(core: Dict[str, Any], parameters: Parameters = PARAMS) -> Dict[str, Any]:
    commitment = parameters.commitment
    composition = parameters.composition
    schedule_sum = sum(row.output_kib for row in composition.schedule)
    compact_flow_bytes = commitment.compact_transcript_flows * core["compact_commitment_bytes"]
    mask_commitment_bytes = (
        commitment.constant_term_mask_commitments * core["ring_element_bytes"]
    )
    projection_bytes = core["projection_response_bytes"]
    auxiliary_bytes = compact_flow_bytes + mask_commitment_bytes + projection_bytes
    visible_total_kib = schedule_sum + auxiliary_bytes / 1024.0
    displayed_total_kib = (
        composition.displayed_recursive_contribution_kib + auxiliary_bytes / 1024.0
    )
    return {
        "schedule_output_sum_kib": schedule_sum,
        "displayed_recursive_contribution_kib": composition.displayed_recursive_contribution_kib,
        "schedule_rounding_delta_kib": (
            composition.displayed_recursive_contribution_kib - schedule_sum
        ),
        "compact_flow_commitments_bytes": compact_flow_bytes,
        "constant_term_mask_commitments_bytes": mask_commitment_bytes,
        "projection_response_bytes": projection_bytes,
        "auxiliary_total_bytes": auxiliary_bytes,
        "total_from_visible_rows_kib": visible_total_kib,
        "total_using_displayed_recursive_kib": displayed_total_kib,
    }


# ---------------------------------------------------------------------------
# Challenge-space calibration
# ---------------------------------------------------------------------------


def minimum_lnp_kappa(
    degree: int,
    target_soundness_bits: int,
    acceptance_probability: float,
) -> int:
    """Minimum kappa so 2/|C| <= 2^-target, using an acceptance estimate."""

    independent_coefficients = degree // 2
    for kappa in range(1, 1_000_000):
        accepted_entropy = (
            independent_coefficients * math.log2(2 * kappa + 1)
            + math.log2(acceptance_probability)
        )
        if accepted_entropy - 1.0 >= target_soundness_bits:
            return kappa
    raise RuntimeError("failed to find an LNP coefficient bound")


def calibrate_lnp_challenge(
    parameters: Parameters = PARAMS,
    calibration_samples: Optional[int] = None,
    validation_samples: Optional[int] = None,
) -> Dict[str, Any]:
    validate_parameters(parameters)
    cfg = parameters.lnp_challenge
    degree = parameters.ring.degree
    calibration_count = (
        cfg.calibration_samples
        if calibration_samples is None
        else calibration_samples
    )
    validation_count = (
        cfg.validation_samples
        if validation_samples is None
        else validation_samples
    )
    if calibration_count <= 0 or validation_count <= 0:
        raise ValueError("calibration and validation sample counts must be positive")

    calibration_bounds: List[float] = []
    calibration_required_etas: List[int] = []
    calibration_rng = random.Random(cfg.calibration_seed)
    for _ in range(calibration_count):
        candidate = sample_lnp_candidate(calibration_rng, degree, cfg.coefficient_bound)
        measure = lnp_multiplier_measure(candidate, cfg.power)
        calibration_bounds.append(lnp_bound_from_measure(measure, cfg.power))
        calibration_required_etas.append(ceil_nth_root(measure, 2 * cfg.power))
    suggested_eta = max(
        1, int(percentile(calibration_required_etas, cfg.target_acceptance))
    )
    if cfg.eta is None:
        eta = suggested_eta
        eta_source = "empirical_candidate"
    else:
        eta = cfg.eta
        eta_source = "configured_candidate" if not cfg.eta_is_final else "configured_final"

    validation_rng = random.Random(cfg.calibration_seed ^ 0x9E37_79B9_7F4A_7C15)
    validation_bounds: List[float] = []
    accepted = 0
    for _ in range(validation_count):
        candidate = sample_lnp_candidate(validation_rng, degree, cfg.coefficient_bound)
        measure = lnp_multiplier_measure(candidate, cfg.power)
        bound = lnp_bound_from_measure(measure, cfg.power)
        validation_bounds.append(bound)
        accepted += int(measure <= pow(eta, 2 * cfg.power))

    acceptance = accepted / validation_count
    acceptance_low, acceptance_high = wilson_interval(
        accepted, validation_count, cfg.confidence
    )
    independent_coefficients = degree // 2
    raw_count = pow(2 * cfg.coefficient_bound + 1, independent_coefficients)
    raw_entropy = math.log2(raw_count)
    accepted_entropy_estimate = (
        raw_entropy + math.log2(acceptance) if acceptance else float("-inf")
    )
    accepted_entropy_lower = (
        raw_entropy + math.log2(acceptance_low) if acceptance_low else float("-inf")
    )
    statistical_challenge_term_bits_lower = accepted_entropy_lower - 1.0

    tau_cc = math.ceil(
        parameters.application.security_bits / math.log2(parameters.ring.q_proof)
    )
    challenge_term = 2.0 ** (-statistical_challenge_term_bits_lower)
    full_ring_term = 2.0 ** (
        -math.log2(parameters.ring.q_proof) * degree / 2.0
    )
    constant_term = 2.0 ** (-math.log2(parameters.ring.q_proof) * tau_cc)
    jl_claim_applicable = jl_reference_assumptions_hold(parameters)
    if jl_claim_applicable:
        jl_term = (
            parameters.projection.jl_failure_events
            * 2.0 ** (-parameters.projection.jl_failure_bits_each)
        )
        total_bound = challenge_term + full_ring_term + constant_term + jl_term
        interactive_model_error_bits: Optional[float] = -math.log2(total_bound)
    else:
        interactive_model_error_bits = None

    return {
        "degree": degree,
        "independent_coefficients": independent_coefficients,
        "coefficient_bound": cfg.coefficient_bound,
        "power": cfg.power,
        "eta": eta,
        "suggested_eta_from_calibration": suggested_eta,
        "eta_source": eta_source,
        "target_acceptance": cfg.target_acceptance,
        "calibration_samples": len(calibration_bounds),
        "calibration_bound_summary": distribution_summary(calibration_bounds),
        "validation_samples": validation_count,
        "accepted": accepted,
        "acceptance_estimate": acceptance,
        "acceptance_confidence": cfg.confidence,
        "acceptance_interval": [acceptance_low, acceptance_high],
        "validation_bound_summary": distribution_summary(validation_bounds),
        "expected_attempts": 1.0 / acceptance if acceptance else float("inf"),
        "raw_count": raw_count,
        "raw_entropy_bits": raw_entropy,
        "accepted_entropy_estimate_bits": accepted_entropy_estimate,
        "accepted_entropy_lower_bits": accepted_entropy_lower,
        "fs_seed_bits": parameters.composition.fs_seed_bits,
        "fs_support_entropy_upper_cap_bits": parameters.composition.fs_seed_bits,
        "statistical_challenge_term_soundness_lower_bits": statistical_challenge_term_bits_lower,
        "fs_challenge_max_probability_bound": None,
        "fs_challenge_term_soundness_bits": None,
        "jl_failure_bound": {
            "events": parameters.projection.jl_failure_events,
            "bits_each": parameters.projection.jl_failure_bits_each,
            "reference_assumptions_hold": jl_claim_applicable,
        },
        "interactive_uniform_set_model_error_bits": interactive_model_error_bits,
        "fiat_shamir_lnp_soundness_bits": None,
        "fs_seed_can_support_target_challenge_term": (
            parameters.composition.fs_seed_bits - 1
            >= parameters.application.security_bits
        ),
        "minimum_kappa_at_target_acceptance": minimum_lnp_kappa(
            degree,
            parameters.application.security_bits,
            cfg.target_acceptance,
        ),
        "difference_invertibility_preconditions": {
            "q_is_prime": is_probable_prime(parameters.ring.q_proof),
            "q_mod_8_is_5": parameters.ring.q_proof % 8 == 5,
            "no_coefficient_wrap": 2 * cfg.coefficient_bound < parameters.ring.q_proof,
            "all_hold": (
                is_probable_prime(parameters.ring.q_proof)
                and parameters.ring.q_proof % 8 == 5
                and 2 * cfg.coefficient_bound < parameters.ring.q_proof
            ),
        },
    }


def labrador_prefilter_count(parameters: Parameters = PARAMS) -> int:
    cfg = parameters.labrador_challenge
    degree = parameters.ring.degree
    positions = math.comb(degree, cfg.zero_coefficients) * math.comb(
        degree - cfg.zero_coefficients, cfg.unit_coefficients
    )
    signs = 1 << (cfg.unit_coefficients + cfg.double_coefficients)
    return positions * signs


def calibrate_labrador_challenge(
    parameters: Parameters = PARAMS,
    validation_samples: Optional[int] = None,
) -> Dict[str, Any]:
    validate_parameters(parameters)
    cfg = parameters.labrador_challenge
    degree = parameters.ring.degree
    sample_count = (
        cfg.validation_samples
        if validation_samples is None
        else validation_samples
    )
    if sample_count <= 0:
        raise ValueError("validation sample count must be positive")

    generator = random.Random(cfg.calibration_seed)
    norms: List[float] = []
    accepted = 0
    for _ in range(sample_count):
        candidate = sample_labrador_candidate(
            generator,
            degree,
            cfg.zero_coefficients,
            cfg.unit_coefficients,
            cfg.double_coefficients,
        )
        norm = operator_norm(candidate)
        norms.append(norm)
        accepted += int(
            labrador_norm_accepts(
                norm,
                cfg.operator_norm_bound,
                cfg.strict_operator_bound,
                cfg.operator_norm_numerical_guard,
            )
        )

    acceptance = accepted / sample_count
    acceptance_low, acceptance_high = wilson_interval(
        accepted, sample_count, cfg.confidence
    )
    raw_count = labrador_prefilter_count(parameters)
    raw_entropy = math.log2(raw_count)
    accepted_entropy_estimate = (
        raw_entropy + math.log2(acceptance) if acceptance else float("-inf")
    )
    accepted_entropy_lower = (
        raw_entropy + math.log2(acceptance_low) if acceptance_low else float("-inf")
    )
    norm_squared = cfg.unit_coefficients + 4 * cfg.double_coefficients
    difference_l2_upper = 2.0 * math.sqrt(norm_squared)
    short_difference_condition = difference_l2_upper < math.sqrt(parameters.ring.q_proof)

    return {
        "degree": degree,
        "weights": {
            "zeros": cfg.zero_coefficients,
            "plus_or_minus_one": cfg.unit_coefficients,
            "plus_or_minus_two": cfg.double_coefficients,
        },
        "coefficient_norm_squared_tau": norm_squared,
        "coefficient_l2_norm": math.sqrt(norm_squared),
        "operator_norm_bound": cfg.operator_norm_bound,
        "operator_norm_numerical_guard": cfg.operator_norm_numerical_guard,
        "operator_norm_method": "double_precision_fft_with_conservative_guard",
        "operator_bound_semantics": "strict_less_than" if cfg.strict_operator_bound else "at_most",
        "validation_samples": sample_count,
        "accepted": accepted,
        "acceptance_estimate": acceptance,
        "acceptance_confidence": cfg.confidence,
        "acceptance_interval": [acceptance_low, acceptance_high],
        "operator_norm_summary": distribution_summary(norms),
        "expected_attempts": 1.0 / acceptance if acceptance else float("inf"),
        "prefilter_count": raw_count,
        "prefilter_entropy_bits": raw_entropy,
        "postfilter_entropy_estimate_bits": accepted_entropy_estimate,
        "postfilter_entropy_lower_bits": accepted_entropy_lower,
        "fs_seed_bits": parameters.composition.fs_seed_bits,
        "fs_support_entropy_upper_cap_bits": parameters.composition.fs_seed_bits,
        "uniform_accepted_set_2_over_C_lower_bits": accepted_entropy_lower - 1.0,
        "fs_challenge_max_probability_bound": None,
        "paper_reference_per_level_soundness_bits": cfg.paper_soundness_bits,
        "well_spread_pss_bound": None,
        "difference_l2_upper_bound": difference_l2_upper,
        "difference_invertibility_preconditions": {
            "q_is_prime": is_probable_prime(parameters.ring.q_proof),
            "q_mod_8_is_5": parameters.ring.q_proof % 8 == 5,
            "difference_is_shorter_than_sqrt_q": short_difference_condition,
            "all_hold": (
                is_probable_prime(parameters.ring.q_proof)
                and parameters.ring.q_proof % 8 == 5
                and short_difference_condition
            ),
        },
    }


# ---------------------------------------------------------------------------
# LaBRADOR recursion heuristics
# ---------------------------------------------------------------------------


def legacy_msis_rank(parameters: Parameters = PARAMS) -> int:
    """Old root-Hermite-factor heuristic; not a concrete security estimate."""

    log_delta = math.log2(parameters.recursion.root_hermite_delta)
    log_q = math.log2(parameters.ring.q_proof)
    rank = (log_q - 1.0) ** 2 / (
        4.0 * log_delta * log_q * parameters.ring.degree
    )
    return math.ceil(rank)


def balanced_digit_count(modulus: int, base: int) -> int:
    """Digit count used by the current ICICLE reference decomposition."""

    if not 1 < base < 2**32:
        raise ValueError("base must be in (1, 2**32)")
    count = math.ceil(math.log(modulus) / math.log(base))
    return count + (1 if base > 2 else 0)


def choose_split(
    n: int,
    auxiliary_length: int,
    max_parts: int,
    shape_constant: float,
) -> Dict[str, int]:
    """Choose balanced split counts with the reference r'^2 ~= C*n' rule."""

    if min(n, auxiliary_length, max_parts) <= 0 or shape_constant <= 0:
        raise ValueError("invalid split inputs")
    total = 2 * n + auxiliary_length
    fraction = (shape_constant / (total * total)) ** (1.0 / 3.0)
    if n > auxiliary_length:
        nu = int(fraction * n + 1.0)
        mu = math.ceil(auxiliary_length * nu / n)
    else:
        mu = int(fraction * auxiliary_length + 1.0)
        nu = math.ceil(n * mu / auxiliary_length)
    nu = min(max_parts, max(1, nu))
    mu = min(max_parts, max(1, mu))
    n_prime = max(ceil_div(n, nu), ceil_div(auxiliary_length, mu))
    r_prime = 2 * nu + mu
    return {
        "nu": nu,
        "mu_split": mu,
        "n_prime": n_prime,
        "r_prime": r_prime,
        "padded_ring_elements": n_prime * r_prime,
    }


def derive_labrador_level(
    n: int,
    r: int,
    beta: float,
    kappa: int,
    parameters: Parameters = PARAMS,
    formula_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one recursion level using either paper or current-TeX formulas."""

    validate_parameters(parameters)
    if min(n, r, kappa) <= 0 or not math.isfinite(beta) or beta <= 0:
        raise ValueError("n, r and kappa must be positive; beta must be finite and positive")
    mode = formula_mode or parameters.recursion.formula_mode
    if mode not in ("paper", "tex"):
        raise ValueError("formula_mode must be 'paper' or 'tex'")

    ring = parameters.ring
    recursion = parameters.recursion
    lab = parameters.labrador_challenge
    degree = ring.degree
    tau = lab.unit_coefficients + 4 * lab.double_coefficients
    coefficient_sd = beta / math.sqrt(r * n * degree)

    base_real = math.sqrt(coefficient_sd * math.sqrt(12.0 * r * tau))
    base = max(2, round_nearest_positive(base_real))
    logarithmic_t1 = math.log(ring.q_proof) / math.log(base)
    if mode == "paper":
        t1 = max(recursion.minimum_t1, round_nearest_positive(logarithmic_t1))
    else:
        t1 = max(recursion.minimum_t1, math.ceil(logarithmic_t1))
    base1 = ceil_nth_root(ring.q_proof, t1)

    if mode == "paper":
        # LaBRADOR Section 5.4: dot-product coefficient SD is
        # sqrt(n*d)*s^2; therefore s^2 is outside sqrt(24*n*d).
        garbage_scale = math.sqrt(24.0 * n * degree) * coefficient_sd**2
        logarithmic_t2 = math.log(garbage_scale) / math.log(base)
        t2 = max(
            2,
            round_nearest_positive(max(0.0, logarithmic_t2)),
        )
        base2 = max(2, nearest_nth_root(garbage_scale, t2))
    else:
        # Literal formula currently printed in the local TeX.  It is retained
        # only for comparison because it loses one factor of s.
        garbage_scale = math.sqrt(24.0 * n * degree * coefficient_sd**2)
        t2 = max(1, math.ceil(math.log(garbage_scale) / math.log(base)))
        base2 = max(2, ceil_nth_root(max(1, math.ceil(garbage_scale)), t2))

    pair_count = r * (r + 1) // 2
    gamma = beta * math.sqrt(tau)
    gamma1_squared = (
        (base1**2 * t1 / 12.0) * r * kappa * degree
        + (base2**2 * t2 / 12.0) * pair_count * degree
    )
    gamma2_squared = (base1**2 * t1 / 12.0) * pair_count * degree
    beta_prime_squared = (
        2.0 * gamma**2 / base**2 + gamma1_squared + gamma2_squared
    )
    beta_prime = math.sqrt(beta_prime_squared)
    auxiliary_length = r * t1 * kappa + (t1 + t2) * pair_count
    split = choose_split(
        n,
        auxiliary_length,
        recursion.max_split_parts,
        recursion.split_shape_constant,
    )

    extraction_slack = math.sqrt(
        recursion.extraction_slack_numerator / recursion.extraction_slack_denominator
    )
    msis_outer_bound = 2.0 * beta_prime
    msis_inner_bound = max(
        8.0 * lab.operator_norm_bound * (base + 1) * beta_prime,
        2.0 * (base + 1) * beta_prime
        + 4.0 * lab.operator_norm_bound * extraction_slack * beta,
    )
    modular_jl_beta_limit = (
        math.sqrt(
            recursion.extraction_slack_denominator
            / recursion.extraction_slack_numerator
        )
        * ring.q_proof
        / recursion.modular_jl_denominator
    )

    return {
        "formula_mode": mode,
        "n": n,
        "r": r,
        "beta": beta,
        "kappa": kappa,
        "coefficient_standard_deviation": coefficient_sd,
        "tau_is_squared_challenge_norm": tau,
        "challenge_l2_norm": math.sqrt(tau),
        "base_b_real": base_real,
        "base_b": base,
        "t1": t1,
        "base_b1": base1,
        "implementation_balanced_digits_for_b1": balanced_digit_count(
            ring.q_proof, base1
        ),
        "garbage_scale": garbage_scale,
        "t2": t2,
        "base_b2": base2,
        "implementation_balanced_digits_for_b2": balanced_digit_count(
            ring.q_proof, base2
        ),
        "gamma": gamma,
        "gamma1_squared": gamma1_squared,
        "gamma2_squared": gamma2_squared,
        "beta_prime_squared": beta_prime_squared,
        "beta_prime": beta_prime,
        "auxiliary_ring_coordinates": auxiliary_length,
        "suggested_split": split,
        "modular_jl_beta_limit": modular_jl_beta_limit,
        "modular_jl_precondition_holds": beta <= modular_jl_beta_limit,
        "msis_estimator_inputs": {
            "rank_kappa1_equals_kappa2_norm": msis_outer_bound,
            "rank_kappa_norm": msis_inner_bound,
            "extraction_slack": extraction_slack,
        },
        "legacy_rank_heuristic": legacy_msis_rank(parameters),
    }


# ---------------------------------------------------------------------------
# Report assembly and warnings
# ---------------------------------------------------------------------------


def collect_warnings(
    core: Dict[str, Any],
    lnp: Optional[Dict[str, Any]],
    lab: Optional[Dict[str, Any]],
    parameters: Parameters = PARAMS,
) -> List[str]:
    warnings: List[str] = []
    projection = parameters.projection
    if not projection.serialization_complete:
        warnings.append(
            f"mu_proj={projection.projected_extra_ring_coordinates} vẫn chỉ là số logical blocks; chưa serialize đầy đủ range/bit blocks."
        )
    if not projection.quotient_bounds_complete:
        warnings.append(
            f"Chưa có bounds/bit lengths cho {core['full_ring_equations']:,} quotient-carry outputs; no-wrap hiện là DRAFT."
        )
    if parameters.commitment.kappa_ajtai is None:
        warnings.append(
            "kappa_A và các opening Gaussian chưa được chọn bằng estimator Module-SIS/MLWE."
        )
    if not core["jl_reference_assumptions_hold"]:
        warnings.append(
            "JL constants/distribution không còn khớp claim tham chiếu 256-row Bin_1; vô hiệu hóa modeled JL error bits."
        )
    if projection.jl_rows != parameters.ring.degree:
        warnings.append(
            f"Draft TeX dùng d={parameters.ring.degree} trong B_R dù projection có {projection.jl_rows} rows; mặc định Python dùng jl_rows và chỉ report công thức TeX để audit."
        )
    if projection.tail_width_mode == "raw_formula":
        warnings.append(
            "B_R=81,385 chỉ khớp khi dùng s_proj chưa làm tròn; thay literal s_proj=15,720 trong công thức cho B_R=81,390."
        )
    if parameters.lnp_challenge.eta is None or not parameters.lnp_challenge.eta_is_final:
        warnings.append(
            f"eta_LNP={parameters.lnp_challenge.eta} là Monte Carlo engineering candidate, chưa phải tham số concrete-security cuối."
        )
    if lnp is not None:
        if lnp["acceptance_interval"][0] < parameters.lnp_challenge.target_acceptance:
            warnings.append(
                "Confidence-lower-bound của acceptance LNP thấp hơn target trong para.py."
            )
        if lnp["statistical_challenge_term_soundness_lower_bits"] < parameters.application.security_bits:
            warnings.append(
                "Confidence-lower-bound của hạng 2/|C_LNP| trong mô hình uniform interactive chưa đạt security_bits."
            )
        if not lnp["fs_seed_can_support_target_challenge_term"]:
            warnings.append(
                f"FS seed {parameters.composition.fs_seed_bits} bit giới hạn hạng 2/|C_LNP| ở tối đa {parameters.composition.fs_seed_bits - 1} bit."
            )
        warnings.append(
            "Chưa có max-preimage/balanced-expander proof cho ánh xạ FS seed -> C_LNP; không xuất FS soundness bits."
        )
    if lab is not None:
        if lab["postfilter_entropy_lower_bits"] < parameters.application.security_bits:
            warnings.append(
                "Challenge LaBRADOR chỉ >2^128 trước op-norm rejection; entropy sau lọc xấp xỉ 125.x bit."
            )
        warnings.append(
            "LaBRADOR operator norm ở đây dùng FFT double + guard; production sampler cần quy tắc số học/canonical implementation được audit."
        )
    warnings.append(
        "LaBRADOR paper dùng nearest rounding và sqrt(24*n*d)*s^2; công thức t1/t2 hiện tại trong TeX khác."
    )
    warnings.append(
        "Bảng 58.23 KiB là engineering estimate; paper TRaccoon B.6.3 cũng không chốt exact combined parameters."
    )
    warnings.append(
        "B_resp và LNP response/MSIS bounds chưa tính được vì thiếu kappa_A, opening distribution và serialized response dimensions."
    )
    if not core["no_wrap_ok"]:
        warnings.append("Ít nhất một no-wrap inequality đã thất bại.")
    return warnings


def build_report(
    parameters: Parameters = PARAMS,
    skip_challenges: bool = False,
    lnp_calibration_samples: Optional[int] = None,
    lnp_validation_samples: Optional[int] = None,
    lab_validation_samples: Optional[int] = None,
) -> Dict[str, Any]:
    core = derive_core(parameters)
    alternative_mode = (
        "jl_rows" if core["tail_dimension_mode"] == "tex_ring_degree" else "tex_ring_degree"
    )
    alternative_core = derive_core(parameters, alternative_mode)
    rounding_audit_mode = "ceil" if core["tail_width_mode"] == "raw_formula" else "raw_formula"
    rounding_audit_core = derive_core(
        parameters, core["tail_dimension_mode"], rounding_audit_mode
    )
    # This exact pair reproduces the draft TeX values.  It is intentionally
    # separate from both safe defaults: the draft used d=64 for a 256-entry
    # response and used the unrounded width despite displaying ceil(s_proj).
    tex_compatibility_core = derive_core(
        parameters, "tex_ring_degree", "raw_formula"
    )
    size = derive_size_accounting(core, parameters)
    alternative_size = derive_size_accounting(alternative_core, parameters)
    tex_compatibility_size = derive_size_accounting(
        tex_compatibility_core, parameters
    )

    lnp = None
    lab = None
    if not skip_challenges:
        lnp = calibrate_lnp_challenge(
            parameters,
            calibration_samples=lnp_calibration_samples,
            validation_samples=lnp_validation_samples,
        )
        lab = calibrate_labrador_challenge(
            parameters, validation_samples=lab_validation_samples
        )

    recursion_levels = sum(
        1 for row in parameters.composition.schedule if row.rank_n is not None
    )
    naive_paper_union_bits = (
        parameters.labrador_challenge.paper_soundness_bits
        - math.log2(recursion_levels)
    )

    warnings = collect_warnings(core, lnp, lab, parameters)
    warnings.append(
        "Không tính composed soundness: còn thiếu B-well-spread/CW-PSS densities, extractor Q và MSIS estimators theo từng tầng."
    )
    return {
        "status": "DRAFT" if warnings else "OK",
        "parameter_fingerprint_sha256": parameter_fingerprint(parameters),
        "sources": SOURCES,
        "core": core,
        "projection_dimension_audit": alternative_core,
        "projection_rounding_audit": rounding_audit_core,
        "tex_compatibility_audit": tex_compatibility_core,
        "lnp_challenge": lnp,
        "labrador_challenge": lab,
        "composition": {
            "recursion_levels": recursion_levels,
            "compact_transcript_flows": parameters.commitment.compact_transcript_flows,
            "constant_term_mask_commitments": parameters.commitment.constant_term_mask_commitments,
            "paper_reference_per_level_soundness_bits": parameters.labrador_challenge.paper_soundness_bits,
            "naive_paper_reference_union_bits_not_a_claim": naive_paper_union_bits,
            "well_spread_pss_analysis_complete": False,
            "composed_soundness_bits": None,
            "boundary_binary_constraints": parameters.composition.boundary_binary_constraints,
            "boundary_below_2_power_23": (
                parameters.composition.boundary_binary_constraints < 2**23
            ),
            "size": size,
            "alternative_projection_size": alternative_size,
            "tex_compatibility_size": tex_compatibility_size,
            "draft_schedule": [asdict(row) for row in parameters.composition.schedule],
        },
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Human rendering
# ---------------------------------------------------------------------------


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if abs(value) >= 1_000_000:
            return f"{value:,.3f}"
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)


def _print_pairs(pairs: Iterable[Tuple[str, Any]]) -> None:
    pairs = list(pairs)
    width = max((len(label) for label, _ in pairs), default=0)
    for label, value in pairs:
        print(f"  {label:<{width}} : {_format_number(value)}")


def print_lnp_calibration(result: Dict[str, Any]) -> None:
    print("\nLNP challenge (LNP22 Sec. 2.7)")
    _print_pairs(
        (
            ("(d, kappa, k, eta)", f"({result['degree']}, {result['coefficient_bound']}, {result['power']}, {result['eta']})"),
            ("eta source", result["eta_source"]),
            ("suggested eta", result["suggested_eta_from_calibration"]),
            ("raw entropy (bits)", result["raw_entropy_bits"]),
            ("validation samples", result["validation_samples"]),
            ("acceptance", result["acceptance_estimate"]),
            ("acceptance CI low", result["acceptance_interval"][0]),
            ("expected attempts", result["expected_attempts"]),
            ("accepted entropy CI-low", result["accepted_entropy_lower_bits"]),
            ("FS seed cap (bits)", result["fs_seed_bits"]),
            ("statistical 2/|C| bits", result["statistical_challenge_term_soundness_lower_bits"]),
            ("interactive modeled error bits", result["interactive_uniform_set_model_error_bits"]),
            ("Fiat-Shamir soundness bits", result["fiat_shamir_lnp_soundness_bits"]),
            ("minimum kappa", result["minimum_kappa_at_target_acceptance"]),
            ("difference invertibility", result["difference_invertibility_preconditions"]["all_hold"]),
        )
    )


def print_labrador_calibration(result: Dict[str, Any]) -> None:
    print("\nLaBRADOR folding challenge (BS23 Sec. 2)")
    weights = result["weights"]
    _print_pairs(
        (
            ("weights (0, +/-1, +/-2)", f"({weights['zeros']}, {weights['plus_or_minus_one']}, {weights['plus_or_minus_two']})"),
            ("tau = ||c||_2^2", result["coefficient_norm_squared_tau"]),
            ("||c||_2", result["coefficient_l2_norm"]),
            ("operator threshold", result["operator_norm_bound"]),
            ("pre-filter entropy (bits)", result["prefilter_entropy_bits"]),
            ("validation samples", result["validation_samples"]),
            ("acceptance", result["acceptance_estimate"]),
            ("acceptance CI low", result["acceptance_interval"][0]),
            ("expected attempts", result["expected_attempts"]),
            ("post-filter entropy CI-low", result["postfilter_entropy_lower_bits"]),
            ("uniform-set 2/|C| CI-low", result["uniform_accepted_set_2_over_C_lower_bits"]),
            ("FS max-probability bound", result["fs_challenge_max_probability_bound"]),
            ("paper reference per-level", f"2^-{result['paper_reference_per_level_soundness_bits']}"),
            ("difference invertibility", result["difference_invertibility_preconditions"]["all_hold"]),
        )
    )


def print_report(report: Dict[str, Any]) -> None:
    core = report["core"]
    audit = report["projection_dimension_audit"]
    rounding_audit = report["projection_rounding_audit"]
    tex_audit = report["tex_compatibility_audit"]
    size = report["composition"]["size"]
    print(f"LNPLab parameter report [{report['status']}]")
    print(f"parameter SHA-256: {report['parameter_fingerprint_sha256']}")

    print("\nFixed and derived values")
    _print_pairs(
        (
            ("d", core["ring_degree"]),
            ("q_sig", core["q_signature"]),
            ("p=q_sig/13", core["rounding_modulus"]),
            ("q_pi", core["q_proof"]),
            ("q_pi prime / mod 8", f"{core['q_proof_is_prime']} / {core['q_proof_mod_8']}"),
            ("N_full", core["full_ring_equations"]),
            ("B_r=B_M", core["error_l2_bound"]),
            ("tau_cc", core["constant_coefficient_repetitions"]),
            ("L_full", core["projected_ring_coordinates"]),
            ("D_proj", core["projected_integer_dimension"]),
            ("s_proj", core["projection_gaussian_width"]),
            ("s_proj raw", core["projection_gaussian_width_raw"]),
            ("B_R", core["extracted_short_vector_bound"]),
            ("B_JL", core["verifier_jl_bound"]),
            ("largest no-wrap term", core["largest_no_wrap_term_name"]),
            ("no-wrap", core["no_wrap_ok"]),
            ("q/no-wrap margin", core["no_wrap_margin"]),
        )
    )

    print("\nProjection-dimension audit")
    _print_pairs(
        (
            ("selected mode", core["tail_dimension_mode"]),
            ("selected tail dimension", core["projection_tail_dimension"]),
            ("selected B_R / B_JL", f"{core['extracted_short_vector_bound']:,} / {core['verifier_jl_bound']:,}"),
            ("alternative mode", audit["tail_dimension_mode"]),
            ("alternative tail dimension", audit["projection_tail_dimension"]),
            ("alternative B_R / B_JL", f"{audit['extracted_short_vector_bound']:,} / {audit['verifier_jl_bound']:,}"),
            ("selected width mode", core["tail_width_mode"]),
            ("other width mode", rounding_audit["tail_width_mode"]),
            ("other-width B_R / B_JL", f"{rounding_audit['extracted_short_vector_bound']:,} / {rounding_audit['verifier_jl_bound']:,}"),
            ("TeX audit modes", f"{tex_audit['tail_dimension_mode']} + {tex_audit['tail_width_mode']}"),
            ("TeX audit B_R / B_JL", f"{tex_audit['extracted_short_vector_bound']:,} / {tex_audit['verifier_jl_bound']:,}"),
        )
    )

    if report["lnp_challenge"] is not None:
        print_lnp_calibration(report["lnp_challenge"])
    if report["labrador_challenge"] is not None:
        print_labrador_calibration(report["labrador_challenge"])

    print("\nComposition and draft size")
    _print_pairs(
        (
            ("recursive levels", report["composition"]["recursion_levels"]),
            ("naive paper union (not claim)", report["composition"]["naive_paper_reference_union_bits_not_a_claim"]),
            ("CW-PSS analysis complete", report["composition"]["well_spread_pss_analysis_complete"]),
            ("composed soundness bits", report["composition"]["composed_soundness_bits"]),
            ("schedule output", f"{size['schedule_output_sum_kib']:.4f} KiB"),
            (
                f"{report['composition']['compact_transcript_flows']} compact flows",
                f"{size['compact_flow_commitments_bytes']} bytes",
            ),
            (
                f"{report['composition']['constant_term_mask_commitments']} mask commitments",
                f"{size['constant_term_mask_commitments_bytes']} bytes",
            ),
            ("JL response", f"{size['projection_response_bytes']} bytes"),
            ("draft total", f"{size['total_using_displayed_recursive_kib']:.5f} KiB"),
            (
                "TeX-compatible total (audit)",
                f"{report['composition']['tex_compatibility_size']['total_using_displayed_recursive_kib']:.5f} KiB",
            ),
        )
    )

    if report["warnings"]:
        print("\nWarnings")
        for warning in report["warnings"]:
            print(f"  - {warning}")


def print_level(result: Dict[str, Any]) -> None:
    print(f"LaBRADOR recursion level [{result['formula_mode']}]")
    _print_pairs(
        (
            ("(n, r, beta, kappa)", f"({result['n']}, {result['r']}, {result['beta']}, {result['kappa']})"),
            ("coefficient s", result["coefficient_standard_deviation"]),
            ("b", result["base_b"]),
            ("(t1, b1)", f"({result['t1']}, {result['base_b1']})"),
            ("(t2, b2)", f"({result['t2']}, {result['base_b2']})"),
            ("beta'", result["beta_prime"]),
            ("m_aux", result["auxiliary_ring_coordinates"]),
            ("split", result["suggested_split"]),
            ("modular-JL beta condition", result["modular_jl_precondition_holds"]),
            ("legacy rank heuristic", result["legacy_rank_heuristic"]),
        )
    )


# ---------------------------------------------------------------------------
# Regression/self tests
# ---------------------------------------------------------------------------


def run_self_tests(parameters: Parameters = PARAMS) -> None:
    core = derive_core(parameters)
    assert core["projection_tail_dimension"] == _tail_dimension(
        parameters, parameters.projection.tail_dimension_mode
    )
    assert core["projected_integer_dimension"] == (
        core["projected_ring_coordinates"] * parameters.ring.degree
    )
    assert core["no_wrap_ok"]

    # Fixed-number regressions only belong to the repository's default set;
    # callers may legitimately self-test dataclasses.replace(...) variants.
    if parameters == PARAMS:
        assert core["full_ring_equations"] == 1458
        assert core["error_l2_bound"] == 220
        assert core["constant_coefficient_repetitions"] == 4
        assert core["projection_gaussian_width"] == 15720
        assert core["extracted_short_vector_bound"] == 162780
        assert core["verifier_jl_bound"] == 415010
        assert core["projected_ring_coordinates"] == 215
        assert core["projection_response_bytes"] == 640
        assert core["compact_commitment_bytes"] == 640
        assert core["ring_element_bytes"] == 320

        tex_audit = derive_core(parameters, "tex_ring_degree", "raw_formula")
        assert tex_audit["extracted_short_vector_bound"] == 81385
        assert tex_audit["verifier_jl_bound"] == 207492
        assert tex_audit["projection_response_bytes"] == 608

    degree = parameters.ring.degree
    left = [0] * degree
    right = [0] * degree
    left[degree - 1] = 1
    right[1] = 1
    product = negacyclic_mul(left, right)
    assert product[0] == -1 and sum(abs(value) for value in product) == 1
    probe = [index % 7 - 3 for index in range(degree)]
    assert negacyclic_square(probe) == negacyclic_mul(probe, probe)
    assert abs(operator_norm([0] * degree)) < 1e-12
    monomial = [0] * degree
    monomial[1] = -2
    assert abs(operator_norm(monomial) - 2.0) < 1e-9

    generator = random.Random(12345)
    lnp_candidate = sample_lnp_candidate(
        generator, degree, parameters.lnp_challenge.coefficient_bound
    )
    assert is_conjugation_fixed(lnp_candidate)
    assert lnp_candidate[degree // 2] == 0
    assert lnp_multiplier_bound(lnp_candidate, parameters.lnp_challenge.power) + 1e-9 >= operator_norm(lnp_candidate)

    # Membership is exact at the boundary; the floating rendering of this
    # constant polynomial is slightly greater than 3 on common platforms.
    constant_three = [0] * degree
    constant_three[0] = 3
    assert lnp_challenge_accepts(
        constant_three, parameters.lnp_challenge.power, 3
    )
    assert not lnp_challenge_accepts(
        constant_three, parameters.lnp_challenge.power, 2
    )

    lab_candidate = sample_labrador_candidate(
        generator,
        degree,
        parameters.labrador_challenge.zero_coefficients,
        parameters.labrador_challenge.unit_coefficients,
        parameters.labrador_challenge.double_coefficients,
    )
    lab_cfg = parameters.labrador_challenge
    assert sum(value == 0 for value in lab_candidate) == lab_cfg.zero_coefficients
    assert sum(abs(value) == 1 for value in lab_candidate) == lab_cfg.unit_coefficients
    assert sum(abs(value) == 2 for value in lab_candidate) == lab_cfg.double_coefficients
    lab_tau = lab_cfg.unit_coefficients + 4 * lab_cfg.double_coefficients
    assert sum(value * value for value in lab_candidate) == lab_tau
    if parameters == PARAMS:
        assert math.log2(labrador_prefilter_count(parameters)) > 128
        assert minimum_lnp_kappa(
            degree, parameters.application.security_bits, 0.99
        ) == 8

    eta_for_test = ceil_nth_root(
        lnp_multiplier_measure(lnp_candidate, parameters.lnp_challenge.power),
        2 * parameters.lnp_challenge.power,
    )
    seeded_lnp, _, seeded_bound = sample_lnp_challenge_from_seed(
        b"self-test", eta_for_test + 100, parameters
    )
    assert is_conjugation_fixed(seeded_lnp)
    assert lnp_challenge_accepts(
        seeded_lnp, parameters.lnp_challenge.power, eta_for_test + 100
    )
    assert seeded_bound <= eta_for_test + 100 + 1e-9
    seeded_lab, _, seeded_norm = sample_labrador_challenge_from_seed(
        b"self-test", parameters
    )
    assert labrador_norm_accepts(
        seeded_norm,
        parameters.labrador_challenge.operator_norm_bound,
        parameters.labrador_challenge.strict_operator_bound,
        parameters.labrador_challenge.operator_norm_numerical_guard,
    )
    assert sum(value * value for value in seeded_lab) == lab_tau


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_seed(value: str) -> bytes:
    if value.startswith("0x"):
        hex_value = value[2:]
        if len(hex_value) % 2:
            hex_value = "0" + hex_value
        return bytes.fromhex(hex_value)
    return value.encode("utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def emit_json(value: Any) -> None:
    print(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    report = subparsers.add_parser("report", help="derive all values and print the audit report")
    report.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    report.add_argument("--skip-challenges", action="store_true", help="skip Monte Carlo calibration")
    report.add_argument("--lnp-calibration-samples", type=int)
    report.add_argument("--lnp-validation-samples", type=int)
    report.add_argument("--lab-validation-samples", type=int)
    report.add_argument(
        "--strict",
        action="store_true",
        help="exit with status 2 when the report still has draft warnings",
    )

    lnp = subparsers.add_parser("calibrate-lnp", help="calibrate/validate eta for C_LNP")
    lnp.add_argument("--calibration-samples", type=int)
    lnp.add_argument("--validation-samples", type=int)
    lnp.add_argument("--json", action="store_true")

    lab = subparsers.add_parser("calibrate-labrador", help="measure the LaBRADOR op-norm filter")
    lab.add_argument("--validation-samples", type=int)
    lab.add_argument("--json", action="store_true")

    sample_lnp = subparsers.add_parser("sample-lnp", help="create deterministic LNP challenge test vectors")
    sample_lnp.add_argument("--seed", default="lnplab-test-vector")
    sample_lnp.add_argument("--eta", type=int)
    sample_lnp.add_argument("--count", type=int, default=1)

    sample_lab = subparsers.add_parser("sample-labrador", help="create deterministic LaBRADOR challenge test vectors")
    sample_lab.add_argument("--seed", default="lnplab-test-vector")
    sample_lab.add_argument("--count", type=int, default=1)
    sample_lab.add_argument(
        "--level", type=int, default=0, help="recursion level bound into the domain"
    )

    level = subparsers.add_parser("level", help="derive one LaBRADOR recursion level")
    level.add_argument("--n", type=int, required=True)
    level.add_argument("--r", type=int, required=True)
    level.add_argument("--beta", type=float, required=True)
    level.add_argument("--kappa", type=int, required=True)
    level.add_argument("--mode", choices=("paper", "tex", "both"), default="both")
    level.add_argument("--json", action="store_true")

    subparsers.add_parser("self-test", help="run fast regression tests")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        arguments = ["report"]
    args = parser.parse_args(arguments)

    try:
        if args.command == "report":
            result = build_report(
                PARAMS,
                skip_challenges=args.skip_challenges,
                lnp_calibration_samples=args.lnp_calibration_samples,
                lnp_validation_samples=args.lnp_validation_samples,
                lab_validation_samples=args.lab_validation_samples,
            )
            emit_json(result) if args.json else print_report(result)
            return 2 if args.strict and result["warnings"] else 0

        if args.command == "calibrate-lnp":
            result = calibrate_lnp_challenge(
                PARAMS,
                calibration_samples=args.calibration_samples,
                validation_samples=args.validation_samples,
            )
            emit_json(result) if args.json else print_lnp_calibration(result)
            return 0

        if args.command == "calibrate-labrador":
            result = calibrate_labrador_challenge(
                PARAMS, validation_samples=args.validation_samples
            )
            emit_json(result) if args.json else print_labrador_calibration(result)
            return 0

        if args.command == "sample-lnp":
            eta = args.eta if args.eta is not None else PARAMS.lnp_challenge.eta
            if eta is None:
                raise ParameterError(
                    "eta is unset; run calibrate-lnp, then set para.py or pass --eta"
                )
            if args.count <= 0:
                raise ParameterError("count must be positive")
            seed = parse_seed(args.seed)
            vectors = []
            for index in range(args.count):
                indexed_seed = seed + index.to_bytes(8, "big")
                coefficients, attempts, bound = sample_lnp_challenge_from_seed(
                    indexed_seed, eta, PARAMS
                )
                vectors.append(
                    {
                        "index": index,
                        "attempts": attempts,
                        "multiplier_bound": bound,
                        "operator_norm": operator_norm(coefficients),
                        "coefficients": coefficients,
                    }
                )
            resolved = {"eta": eta, "count": args.count}
            emit_json(
                {
                    "artifact_fingerprint_sha256": artifact_fingerprint(
                        "sample-lnp", resolved, PARAMS
                    ),
                    "parameter_fingerprint_sha256": parameter_fingerprint(PARAMS),
                    "resolved_values": resolved,
                    "samples": vectors,
                }
            )
            return 0

        if args.command == "sample-labrador":
            if args.count <= 0:
                raise ParameterError("count must be positive")
            if args.level < 0:
                raise ParameterError("level must be non-negative")
            seed = parse_seed(args.seed)
            vectors = []
            for index in range(args.count):
                indexed_seed = seed + index.to_bytes(8, "big")
                coefficients, attempts, norm = sample_labrador_challenge_from_seed(
                    indexed_seed, PARAMS, args.level
                )
                vectors.append(
                    {
                        "index": index,
                        "attempts": attempts,
                        "operator_norm": norm,
                        "l2_norm_squared": sum(value * value for value in coefficients),
                        "coefficients": coefficients,
                    }
                )
            resolved = {"level": args.level, "count": args.count}
            emit_json(
                {
                    "artifact_fingerprint_sha256": artifact_fingerprint(
                        "sample-labrador", resolved, PARAMS
                    ),
                    "parameter_fingerprint_sha256": parameter_fingerprint(PARAMS),
                    "resolved_values": resolved,
                    "samples": vectors,
                }
            )
            return 0

        if args.command == "level":
            modes = ("paper", "tex") if args.mode == "both" else (args.mode,)
            results = [
                derive_labrador_level(
                    args.n, args.r, args.beta, args.kappa, PARAMS, mode
                )
                for mode in modes
            ]
            if args.json:
                emit_json(results if len(results) > 1 else results[0])
            else:
                for index, result in enumerate(results):
                    if index:
                        print()
                    print_level(result)
            return 0

        if args.command == "self-test":
            run_self_tests(PARAMS)
            print("All lnplab regression tests passed.")
            return 0

        parser.print_help()
        return 0
    except (ParameterError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
