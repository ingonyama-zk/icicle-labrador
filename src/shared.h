#pragma once

#include "labrador.h"           // For Zq, Rq, Tq, and the labrador APIs
#include "icicle/hash/keccak.h" // For Hash

#include "types.h"
#include "utils.h"
#include "oracle.h"

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

/// Helper to concatenate oracle_seed and lab_inst bytes and return an Oracle object.
Oracle create_oracle_seed(const std::byte* seed, size_t seed_len, const LabradorInstance& inst);

/// Returns the LabradorInstance for recursion problem
LabradorInstance prepare_recursion_instance(
  const LabradorParam& prev_param,
  const EqualityInstance& final_const,
  const PartialTranscript& trs,
  uint32_t base0,
  size_t mu,
  size_t nu);

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

  RecursionPreparer(const LabradorParam& param, size_t mu, size_t nu, uint32_t base0)
      : prev_r(param.r), prev_n(param.n), mu(mu), nu(nu), base0(base0)
  {
    t_len = param.t_len();
    g_len = param.g_len();
    h_len = param.h_len();
    size_t m = t_len + g_len + h_len;

    n_prime = std::max((prev_n + nu - 1) / nu, (m + mu - 1) / mu);
    r_prime = 2 * nu + mu;
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
