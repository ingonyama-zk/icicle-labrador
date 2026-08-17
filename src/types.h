#pragma once

#include "labrador.h"
#include "lnplabrador_backend_params.h"
#include "utils.h"
#include <cstddef>
#include <stdexcept>
#include <vector>

using namespace icicle::labrador;

/* ======================================================================
 *  Constraint descriptions
 * ====================================================================*/

constexpr uint64_t OP_NORM_BOUND =
  icicle::labrador::backend_config::CHALLENGE_OPERATOR_NORM_BOUND;
static_assert(
  icicle::labrador::backend_config::CHALLENGE_ZERO_COEFFICIENTS +
      icicle::labrador::backend_config::CHALLENGE_UNIT_COEFFICIENTS +
      icicle::labrador::backend_config::CHALLENGE_DOUBLE_COEFFICIENTS ==
    icicle::labrador::backend_config::RING_DEGREE,
  "Challenge weights generated from para1.py must sum to the ring degree");

/// @brief Struct for storing an equality instance of the form:
/// \sum_ij a[i,j]<s[i], s[j]> + \sum_i <phi[i], s[i]> + b = 0
struct EqualityInstance {
  /// Number of witness vectors
  size_t r;
  /// Dimension of each vector in Tq
  size_t n;
  /// a[i,j]  – r×r  matrix over Tq
  std::vector<Tq> a;
  /// phi[i,j] – r vectors, each length n  (row-major)
  std::vector<Tq> phi;
  /// Polynomial in Tq
  Tq b;

  // constructors

  EqualityInstance(size_t r, size_t n) : r(r), n(n), a(r * r, zero()), phi(r * n, zero()), b(zero()) {}

  EqualityInstance(size_t r, size_t n, const std::vector<Tq>& a, const std::vector<Tq>& phi, const Tq& b)
      : r(r), n(n), a(a), phi(phi), b(b)
  {
    if (a.size() != r * r || phi.size() != r * n)
      throw std::invalid_argument("EqualityInstance: incorrect 'a' or 'phi' size");
  }

  EqualityInstance(const EqualityInstance& o) = default;
};

/// @brief Struct for storing a constant-zero constraints of the form:
/// constant(\sum_ij a[i,j]<s[i], s[j]> + \sum_i <phi[i], s[i]> + b) = 0
struct ConstZeroInstance {
  /// Number of witness vectors
  size_t r;
  /// Dimension of each vector in Tq
  size_t n;
  /// a[i,j] – r×r matrix over Tq
  std::vector<Tq> a;
  /// phi[i,j] – r vectors, each length n (row-major)
  std::vector<Tq> phi;
  /// Constant term b such that the entire expression has zero constant coefficient
  Zq b;

  // constructors

  ConstZeroInstance(size_t r, size_t n) : r(r), n(n), a(r * r, zero()), phi(r * n, zero()), b(Zq::zero()) {}

  ConstZeroInstance(size_t r, size_t n, const std::vector<Tq>& a, const std::vector<Tq>& phi, Zq b)
      : r(r), n(n), a(a), phi(phi), b(b)
  {
    if (a.size() != r * r || phi.size() != r * n)
      throw std::invalid_argument("ConstZeroInstance: incorrect ‘a’ or ‘phi’ size");
  }

  ConstZeroInstance(const ConstZeroInstance& o) = default;
};

/* ======================================================================
 *  Protocol parameters
 * ====================================================================*/

/// @brief struct for storing parameter for the Labrador protocol
struct LabradorParam {
  /// Problem size

  /// Number of witness vectors
  size_t r;
  /// Dimension of each vector in Tq
  size_t n;

  /// Seed for Ajtai matrix generation
  std::vector<std::byte> ajtai_seed;

  /// Matrix dimensions for Ajtai commitments

  /// Ajtai matrix A dimensions: n × kappa
  size_t kappa;
  /// Matrix B,C dimensions for committing to decomposed vectors (t,g)
  size_t kappa1;
  /// Matrix D dimensions for committing to decomposed h vectors
  size_t kappa2;

  /// Store Ajtai matrices
  std::vector<Tq> A, B, C, D;

  /// Decomposition bases

  /// Base for decomposing t
  uint32_t base1;
  /// Base for decomposing g
  uint32_t base2;
  /// Base for decomposing h
  uint32_t base3;

  /// Fixed decomposition lengths.  LaBRADOR chooses these lengths from the
  /// coefficient-width analysis in Section 5.4; they are not, in general,
  /// the number of digits required to represent an arbitrary element of Zq.
  /// In particular, the last digit is an unrestricted high part.
  size_t digits1;
  size_t digits2;
  size_t digits3;

  /// JL projection parameters

  /// Output dimension for Johnson-Lindenstrauss projection (typically 256)
  size_t JL_out = icicle::labrador::backend_config::JL_ROWS;

  /// Norm bounds
  /// Witness norm bound
  double beta;
  /// Operator norm bound for challenges
  uint64_t op_norm_bound = OP_NORM_BOUND;

  /// Number of times aggregation is repeated for constant zero constraints
  size_t num_aggregation_rounds = icicle::labrador::backend_config::AGGREGATION_ROUNDS;

  /// The final recursive instance uses the Section 5.6 protocol: its masked
  /// opening was not decomposed by the preceding transition and no outer
  /// commitment matrices are needed.
  bool section_5_6_final = false;

  /// Number of leading witness rows that contain the (un-decomposed) masked
  /// opening in a Section 5.6 final instance.  The remaining rows contain the
  /// concatenated t || g || h vector from the preceding execution.
  size_t final_primary_count = 0;

  /// One-based row in backend_config::PAPER_SCHEDULE.  Zero selects the
  /// adaptive Section 5.4 planner.  The generated seven-level parameters are
  /// activated only when the initial public dimensions/ranks match row one.
  size_t paper_schedule_level = 0;

  // constructors

  LabradorParam(
    size_t r,
    size_t n,
    const std::vector<std::byte>& ajtai_seed,
    size_t kappa,
    size_t kappa1,
    size_t kappa2,
    uint32_t base1,
    uint32_t base2,
    uint32_t base3,
    double beta,
    size_t digits1 = 0,
    size_t digits2 = 0,
    size_t digits3 = 0,
    bool section_5_6_final = false,
    size_t final_primary_count = 0,
    size_t paper_schedule_level = 0)
      : r(r), n(n), ajtai_seed(ajtai_seed), kappa(kappa), kappa1(kappa1), kappa2(kappa2), A(), B(), C(), D(),
        base1(base1), base2(base2), base3(base3),
        digits1(
          digits1 == 0 ? icicle::balanced_decomposition::compute_nof_digits<Zq>(base1) : digits1),
        digits2(
          digits2 == 0 ? icicle::balanced_decomposition::compute_nof_digits<Zq>(base2) : digits2),
        digits3(
          digits3 == 0 ? icicle::balanced_decomposition::compute_nof_digits<Zq>(base3) : digits3),
        beta(beta), section_5_6_final(section_5_6_final), final_primary_count(final_primary_count),
        paper_schedule_level(paper_schedule_level)
  {
    if (section_5_6_final && (final_primary_count == 0 || final_primary_count > r)) {
      throw std::invalid_argument("invalid Section 5.6 primary witness count");
    }
    if (!section_5_6_final && final_primary_count != 0) {
      throw std::invalid_argument("final_primary_count requires a Section 5.6 final instance");
    }
    if (paper_schedule_level > icicle::labrador::backend_config::PAPER_SCHEDULE.size()) {
      throw std::invalid_argument("paper_schedule_level exceeds the generated schedule");
    }

    std::vector<std::byte> seed_A(ajtai_seed), seed_B(ajtai_seed), seed_C(ajtai_seed), seed_D(ajtai_seed);
    seed_A.push_back(std::byte('0'));
    seed_B.push_back(std::byte('1'));
    seed_C.push_back(std::byte('2'));
    seed_D.push_back(std::byte('3'));

    A.resize(n * kappa);
    if (!section_5_6_final) {
      B.resize(t_len() * kappa1);
      C.resize(g_len() * kappa1);
      D.resize(h_len() * kappa2);
    }

    // TODO: is this correct?
    VecOpsConfig async_config = default_vec_ops_config();
    async_config.is_async = true;

    // Avoid ICICLE's structured fast mode (powers of one element).  Slow mode
    // uses independent XOF blocks, but its reduction-to-Zq bias still needs a
    // separate security audit before this can be called a uniform CRS.
    ICICLE_CHECK(random_sampling(A.size(), false, seed_A.data(), seed_A.size(), async_config, A.data()));
    if (!section_5_6_final) {
      ICICLE_CHECK(random_sampling(B.size(), false, seed_B.data(), seed_B.size(), async_config, B.data()));
      ICICLE_CHECK(random_sampling(C.size(), false, seed_C.data(), seed_C.size(), async_config, C.data()));
      ICICLE_CHECK(random_sampling(D.size(), false, seed_D.data(), seed_D.size(), async_config, D.data()));
    }

    ICICLE_CHECK(icicle_device_synchronize());
  }

  LabradorParam(const LabradorParam& o) = default;

  /* helper lengths for base proof vectors --------------------------------*/
  size_t t_len() const
  {
    return digits1 * r * kappa;
  }

  size_t g_len() const
  {
    size_t r_choose_2 = (r * (r + 1)) / 2;
    return (digits2 * r_choose_2);
  }

  size_t h_len() const
  {
    size_t r_choose_2 = (r * (r + 1)) / 2;
    return (digits3 * r_choose_2);
  }
};

/* ======================================================================
 *  Instance to be proved
 * ====================================================================*/

/// An instance of the Labrador problem: consists of multiple equality constraints and constant zero constraints
struct LabradorInstance {
  /// LabradorParam for this instance
  LabradorParam param;
  /// Equality constraints
  std::vector<EqualityInstance> equality_constraints;
  /// Const-zero constraints
  std::vector<ConstZeroInstance> const_zero_constraints;

  // constructors

  LabradorInstance(const LabradorParam& p) : param(p) {}
  LabradorInstance(const LabradorInstance&) = default;

  /* -------- constraint helpers ---------------------------------------- */
  void add_equality_constraint(const EqualityInstance& inst)
  {
    if (inst.r != param.r || inst.n != param.n)
      throw std::invalid_argument("EqualityInstance incompatible with LabradorInstance");
    equality_constraints.push_back(inst);
  }

  void add_equality_constraint(const std::vector<EqualityInstance>& instances)
  {
    for (const auto& inst : instances) {
      if (inst.r != param.r || inst.n != param.n)
        throw std::invalid_argument("EqualityInstance incompatible with LabradorInstance");
    }
    equality_constraints.insert(equality_constraints.end(), instances.begin(), instances.end());
  }

  void add_const_zero_constraint(const ConstZeroInstance& inst)
  {
    if (inst.r != param.r || inst.n != param.n)
      throw std::invalid_argument("ConstZeroInstance incompatible with LabradorInstance");
    const_zero_constraints.push_back(inst);
  }

  void add_const_zero_constraint(const std::vector<ConstZeroInstance>& instances)
  {
    for (const auto& inst : instances) {
      if (inst.r != param.r || inst.n != param.n)
        throw std::invalid_argument("ConstZeroInstance incompatible with LabradorInstance");
    }
    const_zero_constraints.insert(const_zero_constraints.end(), instances.begin(), instances.end());
  }

  /// @brief Aggregates all equality constraints into a single equality constraint by creating a random linear
  /// combination of the constraints using the random polynomials in alpha_hat
  void agg_equality_constraints(const std::vector<Tq>& alpha_hat);
};

/* ======================================================================
 *  Transcript + base-case proof
 * ====================================================================*/

/// @brief Contains messages sent by the Prover to the Verifier in the base case of the Labrador protocol
struct BaseProverMessages {
  /// Ajtai commitment of (t,g)
  std::vector<Tq> u1;
  /// Nonce used by Prover of JL projection
  size_t JL_i;
  /// JL projection of the witness
  std::vector<Zq> p;
  /// Polynomials created during constant zero constraint aggregation
  std::vector<Tq> b_agg;
  /// Ajtai commitment of h
  std::vector<Tq> u2;

  BaseProverMessages() = default;

  size_t proof_size() const
  {
    return sizeof(Zq) * (u1.size() * Tq::d + p.size() + b_agg.size() * Tq::d + u2.size() * Tq::d) + sizeof(size_t);
  }
};

struct PartialTranscript {
  /// Prover messages during the protocol
  BaseProverMessages prover_msg;

  /// hash evaluations
  std::vector<std::byte> seed1, seed2, seed3, seed4;

  /// Challenges- stored for convenience
  std::vector<Zq> psi, omega;
  std::vector<Tq> alpha_hat, challenges_hat;

  PartialTranscript() = default;

  inline size_t proof_size() const { return prover_msg.proof_size(); }
};

/// @brief Struct to hold the proof for the base case
///
/// z_hat: is the vector computed in Step 29 of the base_case_prover
///
/// t: vector computed in Step 9 of the base_case_prover (T_tilde in the code)
///
/// g: vector computed in Step 9 of the base_case_prover (g_tilde in the code)
///
/// h: vector computed in Step 25 of the base_case_prover (H_tilde in the code)
///
/// @note constructor doesn't check dimensions
struct LabradorBaseCaseProof {
  std::vector<Tq> z_hat;
  std::vector<Rq> t, g, h;

  LabradorBaseCaseProof() = default;
  LabradorBaseCaseProof(
    const std::vector<Tq>& z_hat, const std::vector<Tq>& t, const std::vector<Tq>& g, const std::vector<Tq>& h)
      : z_hat(z_hat), t(t), g(g), h(h)
  {
  }
  LabradorBaseCaseProof(const LabradorBaseCaseProof& other) : z_hat(other.z_hat), t(other.t), g(other.g), h(other.h) {}

  /// @return Proof size in Bytes
  size_t size() const { return (z_hat.size() + t.size() + g.size() + h.size()) * Rq::d * sizeof(Zq); }
};

/// Final response for the optimized last execution from Section 5.6.
///
/// Challenges are sampled in the public order
///   auxiliary rows [primary_count, r), then primary rows [0, primary_count).
/// `g_cross[i]` and `g_diagonal[i]` precede the challenge for primary row i.
/// `h_diagonal[round]` precedes every challenge and `h_cross[round - 1]`
/// precedes every challenge after the first.  Thus this response contains
/// 2*primary_count + 1 g polynomials and 2*r - 1 h polynomials.
struct LabradorSection56Proof {
  size_t primary_count = 0;
  std::vector<Tq> z_hat;
  /// Raw inner commitments t_i=A*s_i; unlike an ordinary execution these are
  /// opened directly and are not decomposed for an outer commitment.
  std::vector<Rq> t;
  Rq g0 = zero();
  std::vector<Rq> g_cross, g_diagonal;
  std::vector<Rq> h_cross, h_diagonal;

  size_t z_polynomial_count() const { return z_hat.size(); }
  size_t t_polynomial_count() const { return t.size(); }
  size_t g_polynomial_count() const { return 1 + g_cross.size() + g_diagonal.size(); }
  size_t h_polynomial_count() const { return h_cross.size() + h_diagonal.size(); }
  size_t z_native_size() const { return z_polynomial_count() * Rq::d * sizeof(Zq); }
  size_t t_native_size() const { return t_polynomial_count() * Rq::d * sizeof(Zq); }
  size_t g_native_size() const { return g_polynomial_count() * Rq::d * sizeof(Zq); }
  size_t h_native_size() const { return h_polynomial_count() * Rq::d * sizeof(Zq); }
  size_t polynomial_count() const
  {
    return z_polynomial_count() + t_polynomial_count() + g_polynomial_count() + h_polynomial_count();
  }
  size_t size() const
  {
    return z_native_size() + t_native_size() + g_native_size() + h_native_size();
  }
};

/// A complete final response.  A one-execution proof may still use the
/// generic base response; recursive proofs use the optimized Section 5.6
/// response because their last instance has the required sparse structure.
struct LabradorFinalProof {
  bool uses_section_5_6 = false;
  LabradorBaseCaseProof base;
  LabradorSection56Proof section_5_6;

  static LabradorFinalProof from_base(const LabradorBaseCaseProof& proof)
  {
    LabradorFinalProof result;
    result.base = proof;
    return result;
  }

  static LabradorFinalProof from_section_5_6(const LabradorSection56Proof& proof)
  {
    LabradorFinalProof result;
    result.uses_section_5_6 = true;
    result.section_5_6 = proof;
    return result;
  }

  size_t size() const { return uses_section_5_6 ? section_5_6.size() : base.size(); }
};
