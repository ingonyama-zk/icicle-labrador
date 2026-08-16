#pragma once

#include "labrador.h"           // For Zq, Rq, Tq, and the labrador APIs
#include "icicle/hash/keccak.h" // For Hash

#include "types.h"
#include "utils.h"
#include "oracle.h"

#include <string_view>

using namespace icicle::labrador;

/// @brief Compares two vectors of PolyRing polynomials element-wise
/// @param vec1 First vector to compare
/// @param vec2 Second vector to compare
/// @param size Number of PolyRing elements to compare
/// @return true if vectors are equal, false otherwise
bool poly_vec_eq(const PolyRing* vec1, const PolyRing* vec2, size_t size);

/// @brief Computes the Ajtai commitment of the given input S. Views input S as matrix of vectors to be committed.
/// Vectors are arranged in the row major form. If A is the Ajtai matrix, then this outputs S@A.
/// @param A Random Ajtai commitment matrix
/// @param input_len length of vectors to be committed
/// @param output_len length of commitments
/// @param S data to be committed
/// @param S_len length of data to be committed. If `S_len > input_len` then S_len must be a multiple of input_len. The
/// input S will be viewed as a row major arrangement of S_len/input_len vectors to be committed.
/// @return S_len/input_len commitments of length equal to output_len arranged in row major form.
std::vector<Tq>
ajtai_commitment(const std::vector<Tq>& A, size_t input_len, size_t output_len, const Tq* S, size_t S_len);

/// Samples low norm challenge polynomials for the Labrador protocol
std::vector<Rq> sample_low_norm_challenges(size_t n, size_t r, const std::byte* seed, size_t seed_len);

/// Canonical transcript records.  Counts and counters are fixed-width little
/// endian integers; every Zq value is reduced/canonical and encoded as a
/// little-endian u64.  The explicit domain is length-prefixed.
std::vector<std::byte> canonical_polynomial_transcript_message(
  std::string_view domain,
  const PolyRing* values,
  size_t count);
std::vector<std::byte> canonical_jl_transcript_message(
  std::string_view domain,
  size_t retry_counter,
  const std::vector<Zq>& projection);

/// Canonical Fiat--Shamir messages for the interactive Section 5.6 tail.
/// Optional pointers are null exactly when that polynomial is implicit in the
/// public round structure and therefore is not transmitted.
std::vector<std::byte> section_5_6_first_message(const std::vector<Rq>& t);
std::vector<std::byte> section_5_6_round_message(
  size_t round,
  const Rq* h_cross,
  const Rq& h_diagonal,
  const Rq* g0,
  const Rq* g_cross,
  const Rq* g_diagonal);

/// Auxiliary witness rows are challenged first, followed by the primary z
/// rows.  This ordering is what permits g to shrink to 2*nu+1 polynomials.
std::vector<size_t> section_5_6_challenge_order(size_t r, size_t primary_count);

/// Maximum temporary device memory used to stream JL matrix rows.  The
/// accumulator itself is separate and has row_size_polynomials entries.
constexpr size_t JL_AGGREGATION_SCRATCH_BYTES = size_t{64} * 1024 * 1024;

/// Select how many JL rows fit in the bounded scratch buffer.  At least one
/// row is always processed, even when a single row exceeds the scratch target.
size_t jl_aggregation_chunk_rows(
  size_t row_size_polynomials,
  size_t total_rows,
  size_t scratch_bytes = JL_AGGREGATION_SCRATCH_BYTES);

/// Compute NTT(sum_j weights[j] * Q[j]) without materializing the entire JL
/// matrix. `output_device` must point to row_size_polynomials Tq entries on
/// the active device.  Rows are accumulated in increasing index order, so the
/// result is identical to the full-matrix computation.
void aggregate_jl_projection_rows_ntt(
  const std::byte* seed,
  size_t seed_len,
  size_t row_size_polynomials,
  const Zq* weights,
  size_t total_rows,
  Tq* output_device,
  size_t scratch_bytes = JL_AGGREGATION_SCRATCH_BYTES);

/// Build a canonical, architecture-independent initial Fiat--Shamir state.
/// It binds the external seed, every public protocol parameter, and every
/// equality/constant-zero coefficient.  CRS buffers A/B/C/D are not hashed a
/// second time: LabradorParam deterministically derives them from the bound
/// Ajtai seed and dimensions, and callers must not mutate those public buffers.
Oracle create_oracle_seed(const std::byte* seed, size_t seed_len, const LabradorInstance& inst);

/// Returns the LabradorInstance for recursion problem
LabradorInstance prepare_recursion_instance(
  const LabradorParam& prev_param,
  const EqualityInstance& final_const,
  const PartialTranscript& trs,
  uint32_t base0,
  size_t mu,
  size_t nu,
  bool decompose_z = true);

/// Public, deterministic parameters for one LaBRADOR transition.  These are
/// the Section 5.4 paper formulas evaluated for the current relation.  The
/// Module-SIS ranks remain caller supplied: this repository does not contain a
/// concrete-security estimator.
struct LabradorTransitionPlan {
  uint32_t z_base;
  size_t digits1, digits2, digits3;
  uint32_t base1, base2, base3;
  double beta_next;
  size_t auxiliary_len;
  size_t mu, nu;
  size_t n_next, r_next;
};

struct LabradorDecompositionPlan {
  uint32_t z_base;
  size_t digits1, digits2, digits3;
  uint32_t base1, base2, base3;
};

LabradorDecompositionPlan derive_decomposition_plan(size_t n, size_t r, double beta);

/// Decomposition for a generated paper-schedule row.  It uses the Section
/// 5.4 choice when that witness fits the configured next row, and otherwise
/// searches fixed digit counts that make the transition capacity-valid.
LabradorDecompositionPlan derive_paper_schedule_decomposition(size_t one_based_level, double beta);

LabradorTransitionPlan derive_transition_plan(
  size_t n,
  size_t r,
  size_t kappa,
  uint32_t base1,
  uint32_t base2,
  uint32_t base3,
  size_t digits1,
  size_t digits2,
  size_t digits3,
  double beta);

LabradorTransitionPlan derive_transition_plan(const LabradorParam& param);

/// Transition used immediately before the optimized Section 5.6 execution.
/// It keeps z intact, so the next multiplicity is nu+mu rather than 2nu+mu
/// and the next norm includes ||z|| directly instead of its two base-b limbs.
LabradorTransitionPlan derive_final_transition_plan(const LabradorParam& param);

/// Evaluate one transition from the generated paper schedule without
/// allocating its Ajtai matrices.  Used by startup validation and tests.
LabradorTransitionPlan derive_paper_schedule_transition(
  size_t one_based_level, double beta);

/// Select either the adaptive transition or the generated seven-level
/// schedule carried by `param.paper_schedule_level`.
LabradorTransitionPlan derive_protocol_transition_plan(
  const LabradorParam& param, bool final_transition);

/// Fixed-length centered decomposition.  The first `digits-1` limbs are
/// bounded centered base-b digits and the final limb stores the unrestricted
/// quotient.  This is the decomposition used in LaBRADOR; it differs from a
/// full decomposition of an arbitrary residue modulo q.
std::vector<Rq> fixed_length_decompose(const std::vector<Rq>& input, uint32_t base, size_t digits);

/// Exact coefficient L2 norm (using centered representatives).
long double coefficient_l2_norm(const std::vector<Rq>& input);

/// Append a canonical little-endian uint64 value to a byte string.  Used for
/// the JL retry counter in both the matrix seed and Fiat--Shamir transcript.
std::vector<std::byte> append_u64_le(const std::byte* seed, size_t seed_len, uint64_t value);

/// @brief Helper struct to prepare recursive instance of the problem
struct RecursionPreparer {
  size_t prev_r, prev_n, mu, nu;
  /// decomposition basis for z
  uint32_t base0;

  // These are internally created
  size_t n_prime, r_prime;
  size_t t_len, g_len, h_len;
  bool decompose_z;

  RecursionPreparer(
    const LabradorParam& param,
    size_t mu,
    size_t nu,
    uint32_t base0,
    bool decompose_z = true,
    size_t n_prime_override = 0)
      : prev_r(param.r), prev_n(param.n), mu(mu), nu(nu), base0(base0), decompose_z(decompose_z)
  {
    t_len = param.t_len();
    g_len = param.g_len();
    h_len = param.h_len();
    size_t m = t_len + g_len + h_len;

    const size_t minimum_n_prime = std::max((prev_n + nu - 1) / nu, (m + mu - 1) / mu);
    if (n_prime_override != 0 && n_prime_override < minimum_n_prime) {
      throw std::invalid_argument("scheduled n' cannot hold the recursion witness");
    }
    n_prime = n_prime_override == 0 ? minimum_n_prime : n_prime_override;
    r_prime = (decompose_z ? 2 : 1) * nu + mu;
  }

  // These functions return the starting index for corresponding witness in a r_prime * n_prime matrix
  size_t z0_begin_idx() const;
  size_t z1_begin_idx() const;
  size_t t_begin_idx() const;
  size_t g_begin_idx() const;
  size_t h_begin_idx() const;

  // NOTE: for these functions need to ensure that the size of dst is r_prime * n_prime and src has correct size (same
  // as z0, z1, t, g, h depending on what is being called)

  // Needs dst size = r_prime * n_prime and src size = prev_n
  eIcicleError copy_like_z0(Rq* dst, const Rq* src) const;
  // Needs dst size = r_prime * n_prime and src size = prev_n
  eIcicleError copy_like_z1(Rq* dst, const Rq* src) const;
  // Needs dst size = r_prime * n_prime and src size = t_len
  eIcicleError copy_like_t(Rq* dst, const Rq* src) const;
  // Needs dst size = r_prime * n_prime and src size = g_len
  eIcicleError copy_like_g(Rq* dst, const Rq* src) const;
  // Needs dst size = r_prime * n_prime and src size = h_len
  eIcicleError copy_like_h(Rq* dst, const Rq* src) const;
};

/// choose base0 such that at the end of a base_case_prover witness z satisfies
/// ||z||_2 <op_norm_bound * beta * sqrt(r) < base0^2
uint32_t calc_base0(size_t r, uint64_t op_norm_bound, double beta);

/// returns a choice of mu, nu given n, m for recursion
std::pair<size_t, size_t> compute_mu_nu(size_t n, size_t m);

/// Legacy root-Hermite-factor rank heuristic.  This does not replace the
/// norm-dependent Module-SIS estimator required for a concrete-security claim.
size_t secure_msis_rank();
