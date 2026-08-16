#include "shared.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

constexpr long double LABRADOR_TAU = 71.0L;

size_t checked_add_size(size_t left, size_t right, const char* label)
{
  if (right > std::numeric_limits<size_t>::max() - left) {
    throw std::overflow_error(std::string(label) + " overflows size_t");
  }
  return left + right;
}

size_t checked_mul_size(size_t left, size_t right, const char* label)
{
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " overflows size_t");
  }
  return left * right;
}

size_t ceil_div_size(size_t value, size_t divisor)
{
  if (divisor == 0) { throw std::invalid_argument("division by zero"); }
  return value / divisor + static_cast<size_t>(value % divisor != 0);
}

uint32_t round_positive_to_u32(long double value, const char* label)
{
  if (!std::isfinite(value) || value < 0.0L ||
      value > static_cast<long double>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error(std::string(label) + " is outside uint32_t");
  }
  return static_cast<uint32_t>(std::floor(value + 0.5L));
}

unsigned __int128 integer_power_capped(uint64_t base, size_t exponent, unsigned __int128 cap)
{
  unsigned __int128 result = 1;
  for (size_t i = 0; i < exponent; ++i) {
    result *= base;
    if (result >= cap) { return cap; }
  }
  return result;
}

uint32_t ceil_nth_root_u64(uint64_t value, size_t exponent)
{
  if (value == 0 || exponent == 0) { throw std::invalid_argument("invalid integer root"); }
  uint64_t low = 1;
  uint64_t high = std::max<uint64_t>(2, static_cast<uint64_t>(
    std::ceil(std::exp(std::log(static_cast<long double>(value)) / exponent))));
  const unsigned __int128 cap = static_cast<unsigned __int128>(value);
  while (integer_power_capped(high, exponent, cap) < cap) {
    if (high > std::numeric_limits<uint32_t>::max() / 2ULL) {
      throw std::runtime_error("decomposition base exceeds uint32_t");
    }
    high *= 2;
  }
  while (low < high) {
    const uint64_t middle = low + (high - low) / 2;
    if (integer_power_capped(middle, exponent, cap) >= cap) {
      high = middle;
    } else {
      low = middle + 1;
    }
  }
  if (low > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("decomposition base exceeds uint32_t");
  }
  return static_cast<uint32_t>(low);
}

struct DecompositionChoice {
  uint32_t z_base;
  size_t digits1;
  size_t digits2;
  uint32_t base1;
  uint32_t base2;
};

DecompositionChoice choose_decomposition(size_t n, size_t r, double beta)
{
  if (n == 0 || r == 0 || !std::isfinite(beta) || beta <= 0.0) {
    throw std::invalid_argument("invalid LaBRADOR level dimensions or beta");
  }
  constexpr long double d = static_cast<long double>(Rq::d);
  const long double q = static_cast<long double>(get_q<Zq>());
  const long double s = static_cast<long double>(beta) /
                        std::sqrt(static_cast<long double>(r) * n * d);
  const long double base_real = std::sqrt(s * std::sqrt(12.0L * r * LABRADOR_TAU));
  const uint32_t z_base = std::max<uint32_t>(2, round_positive_to_u32(base_real, "z base"));

  const long double t1_real = std::log(q) / std::log(static_cast<long double>(z_base));
  const size_t digits1 = std::max<size_t>(
    2, static_cast<size_t>(std::floor(t1_real + 0.5L)));
  const uint32_t base1 = ceil_nth_root_u64(static_cast<uint64_t>(get_q<Zq>()), digits1);

  const long double garbage_scale = std::sqrt(24.0L * n * d) * s * s;
  long double t2_real = 0.0L;
  if (garbage_scale > 1.0L) {
    t2_real = std::log(garbage_scale) / std::log(static_cast<long double>(z_base));
  }
  const size_t digits2 = std::max<size_t>(
    2, static_cast<size_t>(std::floor(std::max(0.0L, t2_real) + 0.5L)));
  const long double base2_real = std::pow(std::max(1.0L, garbage_scale), 1.0L / digits2);
  const uint32_t base2 = std::max<uint32_t>(2, round_positive_to_u32(base2_real, "garbage base"));
  return {z_base, digits1, digits2, base1, base2};
}

int64_t centered_coefficient(const Zq& value)
{
  int64_t encoded = 0;
  static_assert(sizeof(encoded) == sizeof(value), "BabyKoala coefficient must occupy 64 bits");
  std::memcpy(&encoded, &value, sizeof(encoded));
  const int64_t q = get_q<Zq>();
  if (encoded > q / 2) { encoded -= q; }
  return encoded;
}

Zq coefficient_from_signed(int64_t value)
{
  if (value >= 0) { return Zq::from_u64(static_cast<uint64_t>(value)); }
  return Zq::from_u64(static_cast<uint64_t>(-value)).neg();
}

std::pair<int64_t, int64_t> floor_divmod(int64_t value, uint32_t base)
{
  int64_t quotient = value / static_cast<int64_t>(base);
  int64_t remainder = value % static_cast<int64_t>(base);
  if (remainder < 0) {
    --quotient;
    remainder += base;
  }
  return {quotient, remainder};
}

} // namespace

LabradorDecompositionPlan derive_decomposition_plan(size_t n, size_t r, double beta)
{
  const DecompositionChoice choice = choose_decomposition(n, r, beta);
  return {
    choice.z_base,
    choice.digits1,
    choice.digits2,
    choice.digits1,
    choice.base1,
    choice.base2,
    choice.base1,
  };
}

bool poly_vec_eq(const PolyRing* vec1, const PolyRing* vec2, size_t size)
{
  for (size_t i = 0; i < size; i++) {
    if (vec1[i] != vec2[i]) { return false; }
  }
  return true;
}

std::vector<Tq>
ajtai_commitment(const std::vector<Tq>& A, size_t input_len, size_t output_len, const Tq* S, size_t S_len)
{
  assert(A.size() == output_len * input_len);

  size_t batch_size = S_len / input_len;
  // Assert that data_len is a multiple of input_len
  assert(batch_size * input_len == S_len);

  std::vector<Tq> comm(batch_size * output_len);
  ICICLE_CHECK(matmul(S, batch_size, input_len, A.data(), input_len, output_len, {}, comm.data()));
  return comm;
}

std::vector<Rq> sample_low_norm_challenges(size_t n, size_t r, const std::byte* seed, size_t seed_len)
{
  size_t d = Rq::d;
  std::vector<Rq> challenge(r, zero());

  sample_challenge_space_polynomials(seed, seed_len, r, 31, 10, OP_NORM_BOUND, {}, challenge.data());
  return challenge;
}

Oracle create_oracle_seed(const std::byte* seed, size_t seed_len, const LabradorInstance& inst)
{
  std::vector<std::byte> buf;

  auto append = [&](auto value) {
    std::byte* p = reinterpret_cast<std::byte*>(&value);
    buf.insert(buf.end(), p, p + sizeof(value));
  };

  // 0. external seed
  buf.insert(buf.end(), seed, seed + seed_len);

  // 1. fixed protocol parameters
  const LabradorParam& prm = inst.param;
  append(prm.r);
  append(prm.n);
  append(prm.kappa);
  append(prm.kappa1);
  append(prm.kappa2);
  append(prm.base1);
  append(prm.base2);
  append(prm.base3);
  append(prm.digits1);
  append(prm.digits2);
  append(prm.digits3);
  append(prm.JL_out);
  append(prm.beta);
  append(prm.op_norm_bound);
  append(prm.num_aggregation_rounds);

  // 1.a Ajtai seed (variable length)
  append(prm.ajtai_seed.size());
  buf.insert(buf.end(), prm.ajtai_seed.begin(), prm.ajtai_seed.end());

  // 2. only counts of constraints
  append(inst.equality_constraints.size());
  append(inst.const_zero_constraints.size());

  // TODO: add contents of equality and const_zero constraints

  return Oracle(buf.data(), buf.size());
}

uint32_t calc_base0(size_t r, uint64_t op_norm_bound, double beta)
{
  if (r == 0 || op_norm_bound == 0 || !std::isfinite(beta) || beta <= 0.0) {
    throw std::invalid_argument("invalid base0 inputs");
  }
  const long double threshold = static_cast<long double>(op_norm_bound) * beta * std::sqrt((long double)r);
  const long double root = std::sqrt(threshold);
  if (!std::isfinite(root) || root >= std::numeric_limits<uint32_t>::max()) {
    throw std::overflow_error("base0 does not fit uint32_t");
  }
  // Strictly enforce threshold < base0^2, including perfect squares.
  return static_cast<uint32_t>(std::floor(root)) + 1;
}

std::pair<size_t, size_t> compute_mu_nu(size_t n, size_t m)
{
  if (n == 0 || m == 0) { throw std::invalid_argument("n and m must be positive"); }
  // setting r_prime^2 = C * n_prime
  const long double C = 1.0L / 4.0L;
  const long double m_plus_2n = 2.0L * n + m;
  const long double frac = std::pow(C / m_plus_2n / m_plus_2n, 1.0L / 3.0L);
  if (n > m) {
    const size_t nu = std::max<size_t>(1, static_cast<size_t>(frac * n + 1.0L));
    const size_t scaled_m = checked_mul_size(m, nu, "m*nu");
    const size_t mu = std::max<size_t>(1, ceil_div_size(scaled_m, n));
    return std::make_pair(mu, nu);
  } else {
    const size_t mu = std::max<size_t>(1, static_cast<size_t>(frac * m + 1.0L));
    const size_t scaled_n = checked_mul_size(n, mu, "n*mu");
    const size_t nu = std::max<size_t>(1, ceil_div_size(scaled_n, m));
    return std::make_pair(mu, nu);
  }
}

std::vector<Rq> fixed_length_decompose(const std::vector<Rq>& input, uint32_t base, size_t digits)
{
  if (input.empty() || base < 2 || digits < 2) {
    throw std::invalid_argument("invalid fixed-length decomposition parameters");
  }
  std::vector<Rq> output(input.size() * digits, zero());
  const int64_t half = static_cast<int64_t>(base / 2);
  const int64_t q_half = get_q<Zq>() / 2;
  for (size_t polynomial = 0; polynomial < input.size(); ++polynomial) {
    for (size_t coefficient = 0; coefficient < Rq::d; ++coefficient) {
      int64_t value = centered_coefficient(input[polynomial].values[coefficient]);
      for (size_t digit = 0; digit + 1 < digits; ++digit) {
        auto [quotient, remainder] = floor_divmod(value, base);
        if (remainder > half) {
          remainder -= base;
          ++quotient;
        }
        output[digit * input.size() + polynomial].values[coefficient] = coefficient_from_signed(remainder);
        value = quotient;
      }
      if (value < -q_half || value > q_half) {
        throw std::runtime_error("fixed-length decomposition high part does not fit Zq");
      }
      output[(digits - 1) * input.size() + polynomial].values[coefficient] = coefficient_from_signed(value);
    }
  }
  return output;
}

long double coefficient_l2_norm(const std::vector<Rq>& input)
{
  long double squared = 0.0L;
  for (const auto& polynomial : input) {
    for (const auto& coefficient : polynomial.values) {
      const long double centered = static_cast<long double>(centered_coefficient(coefficient));
      squared += centered * centered;
    }
  }
  return std::sqrt(squared);
}

std::vector<std::byte> append_u64_le(const std::byte* seed, size_t seed_len, uint64_t value)
{
  std::vector<std::byte> result(seed, seed + seed_len);
  result.reserve(seed_len + sizeof(uint64_t));
  for (size_t byte = 0; byte < sizeof(uint64_t); ++byte) {
    result.push_back(std::byte((value >> (8 * byte)) & 0xffU));
  }
  return result;
}

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
  double beta)
{
  if (n == 0 || r == 0 || kappa == 0 || base1 < 2 || base2 < 2 || base3 < 2 ||
      digits1 < 2 || digits2 < 2 || digits3 < 2 || !std::isfinite(beta) || beta <= 0.0) {
    throw std::invalid_argument("invalid LaBRADOR transition input");
  }
  const DecompositionChoice current_choice = choose_decomposition(n, r, beta);
  const uint32_t z_base = current_choice.z_base;
  const size_t r_plus_one = checked_add_size(r, 1, "r + 1");
  const size_t pair_count_integer = checked_mul_size(r, r_plus_one, "r(r+1)") / 2;
  const long double pair_count = static_cast<long double>(pair_count_integer);
  const long double d = static_cast<long double>(Rq::d);
  const long double gamma_squared = static_cast<long double>(beta) * beta * LABRADOR_TAU;
  const long double gamma1_squared =
    (static_cast<long double>(base1) * base1 * digits1 / 12.0L) * r * kappa * d +
    (static_cast<long double>(base2) * base2 * digits2 / 12.0L) * pair_count * d;
  const long double gamma2_squared =
    (static_cast<long double>(base3) * base3 * digits3 / 12.0L) * pair_count * d;
  const long double beta_next_squared =
    2.0L * gamma_squared / (static_cast<long double>(z_base) * z_base) + gamma1_squared + gamma2_squared;
  const double beta_next = static_cast<double>(std::sqrt(beta_next_squared));
  if (!std::isfinite(beta_next) || beta_next <= 0.0) {
    throw std::runtime_error("derived beta_next is invalid");
  }

  const size_t t_len = checked_mul_size(checked_mul_size(digits1, r, "digits1*r"), kappa, "t length");
  const size_t g_len = checked_mul_size(digits2, pair_count_integer, "g length");
  const size_t h_len = checked_mul_size(digits3, pair_count_integer, "h length");
  const size_t auxiliary_len = checked_add_size(checked_add_size(t_len, g_len, "t+g"), h_len, "t+g+h");
  auto [mu, nu] = compute_mu_nu(n, auxiliary_len);
  const size_t n_next = std::max(ceil_div_size(n, nu), ceil_div_size(auxiliary_len, mu));
  if (nu > (std::numeric_limits<size_t>::max() - mu) / 2) {
    throw std::overflow_error("derived r_next overflows size_t");
  }
  const size_t r_next = 2 * nu + mu;
  const DecompositionChoice next_choice = choose_decomposition(n_next, r_next, beta_next);

  return {
    z_base,
    next_choice.digits1,
    next_choice.digits2,
    next_choice.digits1,
    next_choice.base1,
    next_choice.base2,
    next_choice.base1,
    beta_next,
    auxiliary_len,
    mu,
    nu,
    n_next,
    r_next,
  };
}

LabradorTransitionPlan derive_transition_plan(const LabradorParam& param)
{
  return derive_transition_plan(
    param.n,
    param.r,
    param.kappa,
    param.base1,
    param.base2,
    param.base3,
    param.digits1,
    param.digits2,
    param.digits3,
    param.beta);
}

size_t secure_msis_rank()
{
  const double log_delta = log2(1.0045);
  const double log_q = log2(get_q<Zq>());

  double k_f = pow(log_q - 1.0, 2) / 4 / log_delta / log_q / Rq::d;
  return ceil(k_f);
}

size_t RecursionPreparer::z0_begin_idx() const { return 0; }

size_t RecursionPreparer::z1_begin_idx() const { return nu * n_prime; }

size_t RecursionPreparer::t_begin_idx() const { return (2 * nu) * n_prime; }

size_t RecursionPreparer::g_begin_idx() const { return t_begin_idx() + t_len; }

size_t RecursionPreparer::h_begin_idx() const { return g_begin_idx() + g_len; }

eIcicleError RecursionPreparer::copy_like_z0(Rq* dst, const Rq* src) const
{
  // copy to dst[z0_begin_idx() : z0_begin_idx() + n]
  return icicle_copy(&dst[z0_begin_idx()], src, prev_n * sizeof(Rq));
}

eIcicleError RecursionPreparer::copy_like_z1(Rq* dst, const Rq* src) const
{
  // copy to dst[z1_begin_idx() : z1_begin_idx() + n]
  return icicle_copy(&dst[z1_begin_idx()], src, prev_n * sizeof(Rq));
}

eIcicleError RecursionPreparer::copy_like_t(Rq* dst, const Rq* src) const
{
  // copy to dst[t_begin_idx() : t_begin_idx() + |t|]
  return icicle_copy(&dst[t_begin_idx()], src, t_len * sizeof(Rq));
}

eIcicleError RecursionPreparer::copy_like_g(Rq* dst, const Rq* src) const
{
  // copy to dst[g_begin_idx() : g_begin_idx() + |g|]
  return icicle_copy(&dst[g_begin_idx()], src, g_len * sizeof(Rq));
}

eIcicleError RecursionPreparer::copy_like_h(Rq* dst, const Rq* src) const
{
  // copy to dst[h_begin_idx() : h_begin_idx() + |h|]
  return icicle_copy(&dst[h_begin_idx()], src, h_len * sizeof(Rq));
}

LabradorInstance prepare_recursion_instance(
  const LabradorParam& prev_param,
  const EqualityInstance& final_const,
  const PartialTranscript& trs,
  uint32_t base0,
  size_t mu,
  size_t nu)
{
  const size_t r = final_const.r;
  const size_t n = final_const.n;
  constexpr size_t d = Rq::d;

  assert(prev_param.r == r);
  assert(prev_param.n == n);

  std::vector<Tq> u1 = trs.prover_msg.u1;
  std::vector<Tq> u2 = trs.prover_msg.u2;
  std::vector<Tq> challenges_hat = trs.challenges_hat;

  const std::vector<Tq>& A = prev_param.A;
  const std::vector<Tq>& B = prev_param.B;
  const std::vector<Tq>& C = prev_param.C;
  const std::vector<Tq>& D = prev_param.D;

  RecursionPreparer preparer{prev_param, mu, nu, base0};

  size_t n_prime = preparer.n_prime;
  size_t t_len = preparer.t_len;
  size_t g_len = preparer.g_len;
  size_t h_len = preparer.h_len;
  size_t r_prime = preparer.r_prime;
  // Step 7: Let recursion_instance be a new empty LabradorInstance

  std::vector<std::byte> new_ajtai_seed(prev_param.ajtai_seed);
  new_ajtai_seed.push_back(std::byte('1'));

  const LabradorTransitionPlan transition = derive_transition_plan(prev_param);
  if (transition.z_base != base0 || transition.mu != mu || transition.nu != nu ||
      transition.n_next != n_prime || transition.r_next != r_prime) {
    throw std::runtime_error("prover/verifier LaBRADOR transition plan mismatch");
  }
  LabradorParam recursion_param{
    r_prime,
    n_prime,
    new_ajtai_seed,
    prev_param.kappa,
    prev_param.kappa1,
    prev_param.kappa2,
    transition.base1,
    transition.base2,
    transition.base3,
    transition.beta_next,
    transition.digits1,
    transition.digits2,
    transition.digits3,
  };
  LabradorInstance recursion_instance{recursion_param};

  Zq _zero = Zq::zero();
  Zq two = Zq::from(2);
  Zq two_inv = two.inverse();
  size_t l3 = prev_param.digits3;

  // Step 8: add the equality constraint u1=tB + gC to recursion_instance
  // B_t, C_t are transposed B, C
  std::vector<Tq> B_t(prev_param.kappa1 * prev_param.t_len()), C_t(prev_param.kappa1 * prev_param.g_len());

  ICICLE_CHECK(matrix_transpose<Tq>(B.data(), prev_param.t_len(), prev_param.kappa1, {}, B_t.data()));
  ICICLE_CHECK(matrix_transpose<Tq>(C.data(), prev_param.g_len(), prev_param.kappa1, {}, C_t.data()));

  // negate u1
  ICICLE_CHECK(
    scalar_sub_vec(&_zero, reinterpret_cast<Zq*>(u1.data()), d * u1.size(), {}, reinterpret_cast<Zq*>(u1.data())));

  for (size_t i = 0; i < prev_param.kappa1; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);

    ICICLE_CHECK(preparer.copy_like_t(new_constraint.phi.data(), &B_t[i * t_len]));
    ICICLE_CHECK(preparer.copy_like_g(new_constraint.phi.data(), &C_t[i * g_len]));

    new_constraint.b = u1[i];

    // TESTING: at this point you can check whether the witness satisfies the constraint

    recursion_instance.add_equality_constraint(new_constraint);
  }

  // The vectors B_t, C_t are no longer needed, so we delete them to free memory.
  B_t.clear();
  B_t.shrink_to_fit();
  C_t.clear();
  C_t.shrink_to_fit();

  // Step 9: add the equality constraint u2=hD to recursion_instance
  // D_t = D^t
  std::vector<Tq> D_t(prev_param.kappa2 * h_len);
  ICICLE_CHECK(matrix_transpose<Tq>(D.data(), h_len, prev_param.kappa2, {}, D_t.data()));

  // negate u2
  ICICLE_CHECK(
    scalar_sub_vec(&_zero, reinterpret_cast<Zq*>(u2.data()), d * u2.size(), {}, reinterpret_cast<Zq*>(u2.data())));

  for (size_t i = 0; i < prev_param.kappa2; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);

    ICICLE_CHECK(preparer.copy_like_h(new_constraint.phi.data(), &D_t[i * h_len]));

    new_constraint.b = u2[i];

    // TESTING: at this point you can check whether the witness satisfies the constraint

    recursion_instance.add_equality_constraint(new_constraint);
  }
  // The vectors D_t are no longer needed, so we delete them to free memory.
  D_t.clear();
  D_t.shrink_to_fit();

  // Step 10: add the equality constraint Az - sum_i c_i t_i =0 to recursion_instance
  size_t kappa = prev_param.kappa;
  size_t l1 = prev_param.digits1;

  // A transpose
  std::vector<Tq> A_t(kappa * n);
  ICICLE_CHECK(matrix_transpose<Tq>(A.data(), n, kappa, {}, A_t.data()));

  for (size_t i = 0; i < kappa; i++) {
    EqualityInstance new_constraint(r_prime, n_prime);

    ICICLE_CHECK(preparer.copy_like_z0(new_constraint.phi.data(), &A_t[i * n]));
    ICICLE_CHECK(preparer.copy_like_z1(new_constraint.phi.data(), &A_t[i * n]));
    // new_constraint.phi[nu+ j] = base0*new_constraint.phi[nu+ j]
    Zq base0_scalar = Zq::from(base0);
    ICICLE_CHECK(scalar_mul_vec(
      &base0_scalar, reinterpret_cast<const Zq*>(&new_constraint.phi[preparer.z1_begin_idx()]), n * d, {},
      reinterpret_cast<Zq*>(&new_constraint.phi[preparer.z1_begin_idx()])));

    // TODO: think about vectorising
    std::vector<Tq> temp(r * kappa, zero());
    for (size_t j = 0; j < r; j++) {
      Tq neg_challenge_hat_j = challenges_hat[j];
      scalar_sub_vec(&_zero, challenges_hat[j].values, d, {}, neg_challenge_hat_j.values);
      temp[j * kappa + i] = neg_challenge_hat_j;
    }
    // construct the vector t_mul = [temp | base1*temp | base1^2* temp | ... | base1^l1 * temp]
    std::vector<Tq> t_mul(t_len); // t_len == l1 * r * kappa

    Zq b1_zq = Zq::from(prev_param.base1);
    ICICLE_CHECK(icicle_copy(&t_mul[0], temp.data(), temp.size() * sizeof(Tq)));
    for (size_t j = 1; j < l1; j++) {
      // temp = base1*temp
      scalar_mul_vec(
        &b1_zq, reinterpret_cast<Zq*>(temp.data()), temp.size() * d, {}, reinterpret_cast<Zq*>(temp.data()));
      // t_mul[j * r * kappa: ] = temp
      ICICLE_CHECK(icicle_copy(&t_mul[j * r * kappa], temp.data(), temp.size() * sizeof(Tq)));
    }

    ICICLE_CHECK(preparer.copy_like_t(new_constraint.phi.data(), t_mul.data()));

    recursion_instance.add_equality_constraint(new_constraint);
  }

  std::vector<Tq> c_times_ct(r * r);
  /* Step 11: add c^t * Phi * z - c^t H c == 0 */ {
    EqualityInstance step11_constraint(r_prime, n_prime);
    std::vector<Tq> c_times_phi(n);
    // c_times_phi = c^t * Phi
    ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, final_const.phi.data(), r, n, {}, c_times_phi.data()));
    ICICLE_CHECK(preparer.copy_like_z0(step11_constraint.phi.data(), c_times_phi.data()));
    Zq b0_zq = Zq::from(base0);
    // c_times_phi = base0 * c_times_phi
    scalar_mul_vec(
      &b0_zq, reinterpret_cast<Zq*>(c_times_phi.data()), c_times_phi.size() * d, {},
      reinterpret_cast<Zq*>(c_times_phi.data()));

    ICICLE_CHECK(preparer.copy_like_z1(step11_constraint.phi.data(), c_times_phi.data()));

    ICICLE_CHECK(matmul(challenges_hat.data(), r, 1, challenges_hat.data(), 1, r, {}, c_times_ct.data()));
    // c_times_ct = 2 * c_times_ct
    scalar_mul_vec(
      &two, reinterpret_cast<Zq*>(c_times_ct.data()), c_times_ct.size() * d, {},
      reinterpret_cast<Zq*>(c_times_ct.data()));
    // rescale diagonal back to original
    for (size_t j = 0; j < r; j++) {
      scalar_mul_vec(&two_inv, c_times_ct[j * r + j].values, d, {}, c_times_ct[j * r + j].values);
    }
    // now c_times_ct[j,j] =challenges_hat[j]^2 and
    // for j!=k c_times_ct[j,k] = 2* challenges_hat[j] * challenges_hat[k]
    std::vector<Tq> temp = extract_symm_part(c_times_ct.data(), r);

    // negate temp
    ICICLE_CHECK(scalar_sub_vec(
      &_zero, reinterpret_cast<Zq*>(temp.data()), d * temp.size(), {}, reinterpret_cast<Zq*>(temp.data())));

    // construct the vector h_mul = [temp | base3*temp | base3^2* temp | ... | base3^l3 * temp]
    std::vector<Tq> h_mul(h_len);
    size_t l3 = prev_param.digits3;

    Zq b3_zq = Zq::from(prev_param.base3);
    ICICLE_CHECK(icicle_copy(&h_mul[0], temp.data(), temp.size() * sizeof(Tq)));
    for (size_t j = 1; j < l3; j++) {
      // temp = base3*temp
      scalar_mul_vec(
        &b3_zq, reinterpret_cast<Zq*>(temp.data()), temp.size() * d, {}, reinterpret_cast<Zq*>(temp.data()));
      // h_mul[j * temp.size(): ] = temp
      ICICLE_CHECK(icicle_copy(&h_mul[j * temp.size()], temp.data(), temp.size() * sizeof(Tq)));
    }

    ICICLE_CHECK(preparer.copy_like_h(step11_constraint.phi.data(), h_mul.data()));

    recursion_instance.add_equality_constraint(step11_constraint);
  }

  // returns a constant polynomial in Tq
  auto const_poly = [](const Zq& c) {
    Tq poly;
    for (size_t j = 0; j < d; j++) {
      poly.values[j] = c;
    }
    return poly;
  };

  Tq poly_one = const_poly(Zq::one());
  size_t l2 = prev_param.digits2;

  /* Step 12: \sum_ij a_ij G_ij + \sum_i h_ii + b == 0 */ {
    EqualityInstance step12_constraint(r_prime, n_prime);

    // construct matrix M such that M[i,j] = final_const.a[i,j] + final_const.a[j,i] for i != j else M[i,i] =
    // final_const.a[i,i]
    std::vector<Tq> M(r * r);
    // M = final_const.a^t
    ICICLE_CHECK(matrix_transpose(final_const.a.data(), r, r, {}, M.data()));
    // M = final_const.a + final_const.a^t
    ICICLE_CHECK(vector_add(final_const.a.data(), M.data(), r * r, {}, M.data()));
    // rescale diagonal back to original
    for (size_t j = 0; j < r; j++) {
      scalar_mul_vec(&two_inv, M[j * r + j].values, d, {}, M[j * r + j].values);
    }
    // extract the symmetric part of M as vector a_symm
    std::vector<Tq> a_symm = extract_symm_part(M.data(), r);

    // construct the vector g_mul = [a_symm | base2*a_symm | base2^2* a_symm | ... | base2^l2 * a_symm]
    std::vector<Tq> g_mul(g_len);
    Zq b2_zq = Zq::from(prev_param.base2);
    ICICLE_CHECK(icicle_copy(&g_mul[0], a_symm.data(), a_symm.size() * sizeof(Tq)));
    for (size_t j = 1; j < l2; j++) {
      // a_symm = base2 * a_symm
      scalar_mul_vec(
        &b2_zq, reinterpret_cast<Zq*>(a_symm.data()), a_symm.size() * d, {}, reinterpret_cast<Zq*>(a_symm.data()));
      // g_mul[j * r * kappa: ] = a_symm
      ICICLE_CHECK(icicle_copy(&g_mul[j * a_symm.size()], a_symm.data(), a_symm.size() * sizeof(Tq)));
    }

    ICICLE_CHECK(preparer.copy_like_g(step12_constraint.phi.data(), g_mul.data()));

    // construct the vector symm_I = symmetric_part(I)
    size_t r_choose_2 = (r * (r + 1)) / 2;
    std::vector<Tq> symm_I(r_choose_2, zero());
    size_t i = 0;
    size_t skip = r;

    while (i < r_choose_2) {
      ICICLE_CHECK(icicle_copy(symm_I[i].values, poly_one.values, d * sizeof(Zq)));
      i += skip;
      skip--;
    }

    std::vector<Tq> h_mul(h_len);
    Zq b3_zq = Zq::from(prev_param.base3);
    // [symm_I | base3*symm_I | base3^2* symm_I | ... | base3^l3 * symm_I]
    ICICLE_CHECK(icicle_copy(&h_mul[0], symm_I.data(), symm_I.size() * sizeof(Tq)));
    for (size_t j = 1; j < l3; j++) {
      // symm_I = base3*symm_I
      scalar_mul_vec(
        &b3_zq, reinterpret_cast<Zq*>(symm_I.data()), symm_I.size() * d, {}, reinterpret_cast<Zq*>(symm_I.data()));
      // h_mul[j * symm_I.size(): ] = symm_I
      ICICLE_CHECK(icicle_copy(&h_mul[j * symm_I.size()], symm_I.data(), symm_I.size() * sizeof(Tq)));
    }
    ICICLE_CHECK(preparer.copy_like_h(step12_constraint.phi.data(), h_mul.data()));

    step12_constraint.b = final_const.b;
    recursion_instance.add_equality_constraint(step12_constraint);
  }

  /* Step 13: <z, z> - sum_ij c_i c_j G_ij == 0 */ {
    EqualityInstance step13_constraint(r_prime, n_prime);

    const Zq b0 = Zq::from(base0);
    Tq b0_poly = const_poly(b0);
    Tq b0_sq_poly = const_poly(b0 * b0);
    for (size_t i = 0; i < nu; i++) {
      // a[i,i] = 1
      step13_constraint.a[i * r_prime + i] = poly_one;
      // a[i+nu, i+nu] = base0^2
      step13_constraint.a[(i + nu) * r_prime + (i + nu)] = b0_sq_poly;
      // a[i, i+nu] = base0
      step13_constraint.a[(i + nu) * r_prime + i] = b0_poly;
      // a[i+nu, i] = base0
      step13_constraint.a[i * r_prime + i + nu] = b0_poly;
    }

    std::vector<Tq> temp = extract_symm_part(c_times_ct.data(), r);

    ICICLE_CHECK(scalar_sub_vec(
      &_zero, reinterpret_cast<Zq*>(temp.data()), d * temp.size(), {}, reinterpret_cast<Zq*>(temp.data())));

    // construct the vector g_mul2 = [temp | base2*temp | base2^2* temp | ... | base3^l2 * temp]
    std::vector<Tq> g_mul(g_len);
    Zq b2_zq = Zq::from(prev_param.base2);
    ICICLE_CHECK(icicle_copy(&g_mul[0], temp.data(), temp.size() * sizeof(Tq)));
    for (size_t j = 1; j < l2; j++) {
      // temp = base2 * temp
      scalar_mul_vec(
        &b2_zq, reinterpret_cast<Zq*>(temp.data()), temp.size() * d, {}, reinterpret_cast<Zq*>(temp.data()));
      // g_mul2[j * temp.size(): ] = temp
      ICICLE_CHECK(icicle_copy(&g_mul[j * temp.size()], temp.data(), temp.size() * sizeof(Tq)));
    }

    ICICLE_CHECK(preparer.copy_like_g(step13_constraint.phi.data(), g_mul.data()));

    recursion_instance.add_equality_constraint(step13_constraint);
  }
  // Step 14: already done

  return recursion_instance;
}
