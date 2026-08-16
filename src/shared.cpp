#include "shared.h"
#include "device_vector.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr long double LABRADOR_TAU =
  static_cast<long double>(icicle::labrador::backend_config::CHALLENGE_TAU);

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
  static_assert(sizeof(encoded) == sizeof(value), "backend coefficient must occupy 64 bits");
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

LabradorDecompositionPlan derive_paper_schedule_decomposition(size_t one_based_level, double beta)
{
  const auto& schedule = icicle::labrador::backend_config::PAPER_SCHEDULE;
  if (one_based_level == 0 || one_based_level > schedule.size()) {
    throw std::invalid_argument("paper schedule level is outside the generated table");
  }
  const auto& row = schedule[one_based_level - 1];
  LabradorDecompositionPlan ordinary = derive_decomposition_plan(row.n, row.r, beta);
  if (one_based_level == schedule.size()) { return ordinary; }

  const auto& next = schedule[one_based_level];
  if (row.nu_to_next == 0 || row.mu_to_next == 0) {
    throw std::runtime_error("non-final paper schedule row has no split");
  }
  const size_t pair_count = checked_mul_size(row.r, checked_add_size(row.r, 1, "schedule r+1"),
                                             "schedule r(r+1)") /
                            2;
  auto auxiliary_length = [&](size_t digits1, size_t digits2) {
    const size_t t = checked_mul_size(
      checked_mul_size(digits1, row.r, "scheduled digits1*r"), row.kappa, "scheduled t length");
    const size_t g = checked_mul_size(digits2, pair_count, "scheduled g length");
    const size_t h = checked_mul_size(digits1, pair_count, "scheduled h length");
    return checked_add_size(checked_add_size(t, g, "scheduled t+g"), h, "scheduled auxiliary length");
  };
  const size_t capacity = checked_mul_size(row.mu_to_next, next.n, "scheduled auxiliary capacity");
  if (auxiliary_length(ordinary.digits1, ordinary.digits2) <= capacity) { return ordinary; }

  // The supplied q~=2^40 table fixes n and r but not the decomposition
  // widths.  In particular its penultimate row is too narrow for the plain
  // b~=z-base heuristic.  Search the small public digit space and choose the
  // capacity-valid option with the smallest resulting target norm.
  constexpr size_t MAX_DIGITS = 40;
  constexpr long double d = static_cast<long double>(Rq::d);
  const long double s = static_cast<long double>(beta) /
                        std::sqrt(static_cast<long double>(row.r) * row.n * d);
  const long double garbage_scale = std::sqrt(24.0L * row.n * d) * s * s;
  const long double gamma_squared = static_cast<long double>(beta) * beta * LABRADOR_TAU;
  long double best_beta_squared = std::numeric_limits<long double>::infinity();
  size_t best_auxiliary = std::numeric_limits<size_t>::max();
  LabradorDecompositionPlan best{};
  bool found = false;
  for (size_t digits1 = 2; digits1 <= MAX_DIGITS; ++digits1) {
    const uint32_t base1 = ceil_nth_root_u64(static_cast<uint64_t>(get_q<Zq>()), digits1);
    for (size_t digits2 = 2; digits2 <= MAX_DIGITS; ++digits2) {
      const size_t auxiliary = auxiliary_length(digits1, digits2);
      if (auxiliary > capacity) { continue; }
      const long double base2_real =
        std::pow(std::max(1.0L, garbage_scale), 1.0L / static_cast<long double>(digits2));
      const uint32_t base2 = std::max<uint32_t>(2, round_positive_to_u32(base2_real, "scheduled garbage base"));
      const long double gamma1_squared =
        (static_cast<long double>(base1) * base1 * digits1 / 12.0L) * row.r * row.kappa * d +
        (static_cast<long double>(base2) * base2 * digits2 / 12.0L) * pair_count * d;
      const long double gamma2_squared =
        (static_cast<long double>(base1) * base1 * digits1 / 12.0L) * pair_count * d;
      const long double target_beta_squared =
        (row.section_5_6_tail
           ? gamma_squared
           : 2.0L * gamma_squared /
               (static_cast<long double>(ordinary.z_base) * ordinary.z_base)) +
        gamma1_squared + gamma2_squared;
      if (!found || target_beta_squared < best_beta_squared ||
          (target_beta_squared == best_beta_squared && auxiliary < best_auxiliary)) {
        found = true;
        best_beta_squared = target_beta_squared;
        best_auxiliary = auxiliary;
        best = {ordinary.z_base, digits1, digits2, digits1, base1, base2, base1};
      }
    }
  }
  if (!found) {
    throw std::runtime_error("paper schedule row cannot fit any supported decomposition in the next level");
  }
  return best;
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

  sample_challenge_space_polynomials(
    seed,
    seed_len,
    r,
    icicle::labrador::backend_config::CHALLENGE_UNIT_COEFFICIENTS,
    icicle::labrador::backend_config::CHALLENGE_DOUBLE_COEFFICIENTS,
    OP_NORM_BOUND,
    {},
    challenge.data());
  return challenge;
}

namespace {

constexpr size_t CANONICAL_HASH_CHUNK_BYTES = 64 * 1024;

void append_canonical_bytes(
  std::vector<std::byte>& out,
  const std::byte* data,
  size_t length)
{
  if (length == 0) { return; }
  if (data == nullptr) { throw std::invalid_argument("null canonical transcript input"); }
  out.insert(out.end(), data, data + length);
}

void append_canonical_u8(std::vector<std::byte>& out, uint8_t value)
{
  out.push_back(std::byte(value));
}

void append_canonical_u32(std::vector<std::byte>& out, uint32_t value)
{
  for (size_t i = 0; i < sizeof(value); ++i) {
    out.push_back(std::byte((value >> (8 * i)) & 0xffU));
  }
}

void append_canonical_u64(std::vector<std::byte>& out, uint64_t value)
{
  for (size_t i = 0; i < sizeof(value); ++i) {
    out.push_back(std::byte((value >> (8 * i)) & 0xffU));
  }
}

uint64_t canonical_zq_value(const Zq& value)
{
  static_assert(sizeof(value.limbs_storage.limbs) == sizeof(uint64_t),
                "canonical transcript expects a two-limb q40 coefficient");
  const uint64_t encoded = uint64_t(value.limbs_storage.limbs[0]) |
                           (uint64_t(value.limbs_storage.limbs[1]) << 32U);
  if (encoded >= icicle::labrador::backend_config::RING_MODULUS) {
    throw std::invalid_argument("non-canonical Zq value in Fiat--Shamir transcript");
  }
  return encoded;
}

void append_canonical_zq(std::vector<std::byte>& out, const Zq& value)
{
  append_canonical_u64(out, canonical_zq_value(value));
}

void append_canonical_poly(std::vector<std::byte>& out, const PolyRing& value)
{
  for (const Zq& coefficient : value.values) {
    append_canonical_zq(out, coefficient);
  }
}

void append_canonical_domain(std::vector<std::byte>& out, std::string_view domain)
{
  append_canonical_u64(out, static_cast<uint64_t>(domain.size()));
  append_canonical_bytes(
    out,
    reinterpret_cast<const std::byte*>(domain.data()),
    domain.size());
}

void append_canonical_f64(std::vector<std::byte>& out, double value)
{
  static_assert(sizeof(double) == sizeof(uint64_t) && std::numeric_limits<double>::is_iec559,
                "canonical transcript requires IEEE-754 binary64");
  if (!std::isfinite(value)) {
    throw std::invalid_argument("non-finite binary64 in Fiat--Shamir transcript");
  }
  uint64_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  append_canonical_u64(out, bits);
}

class CanonicalHashChain {
public:
  explicit CanonicalHashChain(std::string_view domain)
      : hasher(Sha3_256::create())
  {
    if (domain.empty() || hasher.output_size() != state.size()) {
      throw std::runtime_error("invalid canonical transcript hash configuration");
    }
    ICICLE_CHECK(hasher.hash(
      reinterpret_cast<const std::byte*>(domain.data()),
      domain.size(),
      {},
      state.data()));
  }

  void absorb(std::string_view tag, const std::byte* data, size_t length)
  {
    if (tag.empty() || (length != 0 && data == nullptr)) {
      throw std::invalid_argument("invalid canonical transcript record");
    }
    std::vector<std::byte> preimage;
    const size_t header_size = checked_add_size(
      checked_add_size(state.size(), sizeof(uint64_t) * 2, "canonical hash header"),
      tag.size(),
      "canonical hash tag");
    preimage.reserve(checked_add_size(header_size, length, "canonical hash record"));
    append_canonical_bytes(preimage, state.data(), state.size());
    append_canonical_domain(preimage, tag);
    append_canonical_u64(preimage, static_cast<uint64_t>(length));
    append_canonical_bytes(preimage, data, length);
    ICICLE_CHECK(hasher.hash(preimage.data(), preimage.size(), {}, state.data()));
  }

  void absorb(std::string_view tag, const std::vector<std::byte>& data)
  {
    absorb(tag, data.data(), data.size());
  }

  Oracle finish() const { return Oracle(state.data(), state.size()); }

private:
  Hash hasher;
  std::array<std::byte, 32> state{};
};

void absorb_polynomials(
  CanonicalHashChain& chain,
  std::string_view tag,
  const PolyRing* values,
  size_t count)
{
  if (count != 0 && values == nullptr) {
    throw std::invalid_argument("null polynomial vector in public instance");
  }
  std::vector<std::byte> header;
  append_canonical_u64(header, static_cast<uint64_t>(count));
  append_canonical_u32(header, static_cast<uint32_t>(PolyRing::d));
  chain.absorb(tag, header);

  std::vector<std::byte> chunk;
  chunk.reserve(CANONICAL_HASH_CHUNK_BYTES);
  for (size_t polynomial = 0; polynomial < count; ++polynomial) {
    for (const Zq& coefficient : values[polynomial].values) {
      append_canonical_zq(chunk, coefficient);
      if (chunk.size() == CANONICAL_HASH_CHUNK_BYTES) {
        chain.absorb("Zq-vector-chunk-v1", chunk);
        chunk.clear();
      }
    }
  }
  if (!chunk.empty()) { chain.absorb("Zq-vector-chunk-v1", chunk); }
}

} // namespace

std::vector<std::byte> canonical_polynomial_transcript_message(
  std::string_view domain,
  const PolyRing* values,
  size_t count)
{
  if (domain.empty() || (count != 0 && values == nullptr)) {
    throw std::invalid_argument("invalid canonical polynomial transcript message");
  }
  const size_t coefficient_count =
    checked_mul_size(count, PolyRing::d, "canonical polynomial coefficient count");
  const size_t coefficient_bytes =
    checked_mul_size(coefficient_count, sizeof(uint64_t), "canonical polynomial byte count");
  std::vector<std::byte> message;
  message.reserve(checked_add_size(
    checked_add_size(sizeof(uint64_t) * 3, domain.size(), "canonical polynomial header"),
    coefficient_bytes,
    "canonical polynomial message"));
  append_canonical_domain(message, domain);
  append_canonical_u64(message, static_cast<uint64_t>(count));
  append_canonical_u64(message, static_cast<uint64_t>(PolyRing::d));
  for (size_t i = 0; i < count; ++i) { append_canonical_poly(message, values[i]); }
  return message;
}

std::vector<std::byte> canonical_jl_transcript_message(
  std::string_view domain,
  size_t retry_counter,
  const std::vector<Zq>& projection)
{
  if (domain.empty()) {
    throw std::invalid_argument("empty JL transcript domain");
  }
  const size_t coefficient_bytes =
    checked_mul_size(projection.size(), sizeof(uint64_t), "canonical JL byte count");
  std::vector<std::byte> message;
  message.reserve(checked_add_size(
    checked_add_size(sizeof(uint64_t) * 3, domain.size(), "canonical JL header"),
    coefficient_bytes,
    "canonical JL message"));
  append_canonical_domain(message, domain);
  append_canonical_u64(message, static_cast<uint64_t>(retry_counter));
  append_canonical_u64(message, static_cast<uint64_t>(projection.size()));
  for (const Zq& coefficient : projection) { append_canonical_zq(message, coefficient); }
  return message;
}

std::vector<std::byte> section_5_6_first_message(const std::vector<Rq>& t)
{
  return canonical_polynomial_transcript_message(
    "LaBRADOR-Section-5.6-first-v2", t.data(), t.size());
}

std::vector<std::byte> section_5_6_round_message(
  size_t round,
  const Rq* h_cross,
  const Rq& h_diagonal,
  const Rq* g0,
  const Rq* g_cross,
  const Rq* g_diagonal)
{
  static constexpr std::string_view DOMAIN = "LaBRADOR-Section-5.6-round-v2";
  const uint8_t flags =
    (h_cross == nullptr ? 0U : 1U) | (g0 == nullptr ? 0U : 2U) |
    (g_cross == nullptr ? 0U : 4U) | (g_diagonal == nullptr ? 0U : 8U);
  std::vector<std::byte> message;
  message.reserve(
    sizeof(uint64_t) * 2 + DOMAIN.size() + 1 + 5 * Rq::d * sizeof(uint64_t));
  append_canonical_domain(message, DOMAIN);
  append_canonical_u64(message, static_cast<uint64_t>(round));
  append_canonical_u8(message, flags);
  if (h_cross != nullptr) { append_canonical_poly(message, *h_cross); }
  append_canonical_poly(message, h_diagonal);
  if (g0 != nullptr) { append_canonical_poly(message, *g0); }
  if (g_cross != nullptr) { append_canonical_poly(message, *g_cross); }
  if (g_diagonal != nullptr) { append_canonical_poly(message, *g_diagonal); }
  return message;
}

std::vector<size_t> section_5_6_challenge_order(size_t r, size_t primary_count)
{
  if (primary_count == 0 || primary_count > r) {
    throw std::invalid_argument("invalid Section 5.6 challenge partition");
  }
  std::vector<size_t> order;
  order.reserve(r);
  for (size_t i = primary_count; i < r; ++i) { order.push_back(i); }
  for (size_t i = 0; i < primary_count; ++i) { order.push_back(i); }
  return order;
}

size_t jl_aggregation_chunk_rows(
  size_t row_size_polynomials,
  size_t total_rows,
  size_t scratch_bytes)
{
  if (row_size_polynomials == 0 || total_rows == 0 || scratch_bytes == 0) {
    throw std::invalid_argument("JL aggregation dimensions must be non-zero");
  }
  const size_t bytes_per_row = checked_mul_size(row_size_polynomials, sizeof(Rq), "JL row bytes");
  const size_t rows_by_budget = std::max<size_t>(1, scratch_bytes / bytes_per_row);
  return std::min(total_rows, rows_by_budget);
}

void aggregate_jl_projection_rows_ntt(
  const std::byte* seed,
  size_t seed_len,
  size_t row_size_polynomials,
  const Zq* weights,
  size_t total_rows,
  Tq* output_device,
  size_t scratch_bytes)
{
  if (seed == nullptr || seed_len == 0 || weights == nullptr || output_device == nullptr) {
    throw std::invalid_argument("JL aggregation received a null input");
  }
  const size_t chunk_rows =
    jl_aggregation_chunk_rows(row_size_polynomials, total_rows, scratch_bytes);
  const size_t chunk_polynomials =
    checked_mul_size(chunk_rows, row_size_polynomials, "JL streamed chunk");
  const size_t scalar_row_size =
    checked_mul_size(row_size_polynomials, Rq::d, "JL scalar row size");
  if (scalar_row_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("JL scalar row size exceeds backend batch range");
  }
  DeviceVector<Rq> rows(chunk_polynomials);

  // DeviceVector is zero-initialized, but output is caller-owned and reused
  // across aggregation repetitions.
  ICICLE_CHECK(icicle_memset(output_device, 0, row_size_polynomials * sizeof(Tq)));

  VecOpsConfig scale_config = default_vec_ops_config();
  scale_config.is_a_on_device = false;
  scale_config.is_b_on_device = true;
  scale_config.is_result_on_device = true;
  scale_config.columns_batch = false;

  VecOpsConfig add_config = default_vec_ops_config();
  add_config.is_a_on_device = true;
  add_config.is_b_on_device = true;
  add_config.is_result_on_device = true;

  VecOpsConfig sum_config = default_vec_ops_config();
  sum_config.batch_size = static_cast<int>(scalar_row_size);
  sum_config.columns_batch = true;
  sum_config.is_a_on_device = true;
  sum_config.is_result_on_device = true;

  for (size_t row_offset = 0; row_offset < total_rows; row_offset += chunk_rows) {
    const size_t rows_this_chunk = std::min(chunk_rows, total_rows - row_offset);
    ICICLE_CHECK(icicle::labrador::get_jl_matrix_rows(
      seed,
      seed_len,
      row_size_polynomials,
      row_offset,
      rows_this_chunk,
      true,
      {},
      rows.data()));

    scale_config.batch_size = static_cast<int>(rows_this_chunk);
    ICICLE_CHECK(scalar_mul_vec(
      &weights[row_offset],
      reinterpret_cast<const Zq*>(rows.data()),
      scalar_row_size,
      scale_config,
      reinterpret_cast<Zq*>(rows.data())));

    // Fold the preceding accumulator into the first streamed row, then reduce
    // the whole chunk directly back into the accumulator.  This needs no
    // second r*n scratch vector and only two backend calls per chunk.
    ICICLE_CHECK(vector_add(
      reinterpret_cast<Tq*>(rows.data()),
      output_device,
      row_size_polynomials,
      add_config,
      reinterpret_cast<Tq*>(rows.data())));
    ICICLE_CHECK(vector_sum<Zq>(
      reinterpret_cast<const Zq*>(rows.data()),
      rows_this_chunk,
      sum_config,
      reinterpret_cast<Zq*>(output_device)));
  }

  ICICLE_CHECK(ntt(
    output_device,
    row_size_polynomials,
    NTTDir::kForward,
    {},
    output_device));
}

Oracle create_oracle_seed(const std::byte* seed, size_t seed_len, const LabradorInstance& inst)
{
  if (seed_len != 0 && seed == nullptr) {
    throw std::invalid_argument("null external Fiat--Shamir seed");
  }
  static_assert(sizeof(size_t) <= sizeof(uint64_t),
                "canonical transcript cannot encode this size_t architecture");

  CanonicalHashChain chain{"LaBRADOR-public-instance-v2"};
  chain.absorb("external-seed", seed, seed_len);

  // Fixed-width, explicitly ordered public protocol parameters.  The modulus
  // and degree prevent the same byte-level statement from being replayed
  // against another compiled ring backend.
  const LabradorParam& prm = inst.param;
  if (!(prm.beta > 0.0) || !std::isfinite(prm.beta)) {
    throw std::invalid_argument("invalid witness norm bound in public transcript");
  }
  std::vector<std::byte> parameters;
  parameters.reserve(4 * sizeof(uint32_t) + 17 * sizeof(uint64_t) + 1);
  append_canonical_u32(parameters, icicle::labrador::backend_config::RING_DEGREE);
  append_canonical_u64(parameters, icicle::labrador::backend_config::RING_MODULUS);
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.r));
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.n));
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.kappa));
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.kappa1));
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.kappa2));
  append_canonical_u32(parameters, prm.base1);
  append_canonical_u32(parameters, prm.base2);
  append_canonical_u32(parameters, prm.base3);
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.digits1));
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.digits2));
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.digits3));
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.JL_out));
  append_canonical_f64(parameters, prm.beta);
  append_canonical_u64(parameters, prm.op_norm_bound);
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.num_aggregation_rounds));
  append_canonical_u8(parameters, prm.section_5_6_final ? 1U : 0U);
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.final_primary_count));
  append_canonical_u64(parameters, static_cast<uint64_t>(prm.paper_schedule_level));
  chain.absorb("protocol-parameters", parameters);
  chain.absorb("ajtai-seed", prm.ajtai_seed);

  const size_t expected_a = checked_mul_size(prm.n, prm.kappa, "Ajtai A shape");
  if (prm.A.size() != expected_a) {
    throw std::invalid_argument("Ajtai A does not match the transcript-bound dimensions");
  }
  // B/C/D are deterministically generated from the Ajtai seed above.  Check
  // their public shapes so a malformed copied instance cannot silently reuse
  // the same initial transcript state.
  if (prm.section_5_6_final) {
    if (!prm.B.empty() || !prm.C.empty() || !prm.D.empty()) {
      throw std::invalid_argument("Section 5.6 final instance has unexpected outer CRS matrices");
    }
  } else {
    const size_t expected_b = checked_mul_size(prm.t_len(), prm.kappa1, "Ajtai B shape");
    const size_t expected_c = checked_mul_size(prm.g_len(), prm.kappa1, "Ajtai C shape");
    const size_t expected_d = checked_mul_size(prm.h_len(), prm.kappa2, "Ajtai D shape");
    if (prm.B.size() != expected_b || prm.C.size() != expected_c || prm.D.size() != expected_d) {
      throw std::invalid_argument("outer Ajtai matrices do not match transcript-bound dimensions");
    }
  }

  std::vector<std::byte> constraint_counts;
  append_canonical_u64(
    constraint_counts, static_cast<uint64_t>(inst.equality_constraints.size()));
  append_canonical_u64(
    constraint_counts, static_cast<uint64_t>(inst.const_zero_constraints.size()));
  chain.absorb("constraint-counts", constraint_counts);

  const size_t expected_quadratic = checked_mul_size(prm.r, prm.r, "constraint a shape");
  const size_t expected_linear = checked_mul_size(prm.r, prm.n, "constraint phi shape");
  for (size_t index = 0; index < inst.equality_constraints.size(); ++index) {
    const EqualityInstance& constraint = inst.equality_constraints[index];
    if (constraint.r != prm.r || constraint.n != prm.n ||
        constraint.a.size() != expected_quadratic ||
        constraint.phi.size() != expected_linear) {
      throw std::invalid_argument("malformed equality constraint in public transcript");
    }
    std::vector<std::byte> header;
    append_canonical_u64(header, static_cast<uint64_t>(index));
    append_canonical_u64(header, static_cast<uint64_t>(constraint.r));
    append_canonical_u64(header, static_cast<uint64_t>(constraint.n));
    chain.absorb("equality-header", header);
    absorb_polynomials(chain, "equality-a", constraint.a.data(), constraint.a.size());
    absorb_polynomials(chain, "equality-phi", constraint.phi.data(), constraint.phi.size());
    absorb_polynomials(chain, "equality-b", &constraint.b, 1);
  }
  for (size_t index = 0; index < inst.const_zero_constraints.size(); ++index) {
    const ConstZeroInstance& constraint = inst.const_zero_constraints[index];
    if (constraint.r != prm.r || constraint.n != prm.n ||
        constraint.a.size() != expected_quadratic ||
        constraint.phi.size() != expected_linear) {
      throw std::invalid_argument("malformed constant-zero constraint in public transcript");
    }
    std::vector<std::byte> header;
    append_canonical_u64(header, static_cast<uint64_t>(index));
    append_canonical_u64(header, static_cast<uint64_t>(constraint.r));
    append_canonical_u64(header, static_cast<uint64_t>(constraint.n));
    chain.absorb("constant-zero-header", header);
    absorb_polynomials(chain, "constant-zero-a", constraint.a.data(), constraint.a.size());
    absorb_polynomials(chain, "constant-zero-phi", constraint.phi.data(), constraint.phi.size());
    std::vector<std::byte> constant;
    append_canonical_zq(constant, constraint.b);
    chain.absorb("constant-zero-b", constant);
  }

  return chain.finish();
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

LabradorTransitionPlan derive_final_transition_plan(const LabradorParam& param)
{
  LabradorTransitionPlan plan = derive_transition_plan(param);

  // Section 5.6 deliberately skips z=z^(0)+b*z^(1) before the final
  // execution.  Replace the decomposed-z contribution 2*gamma^2/b^2 by the
  // direct gamma^2 contribution in Equation (5).
  const long double gamma_squared =
    static_cast<long double>(param.beta) * param.beta * LABRADOR_TAU;
  const long double old_z_squared =
    2.0L * gamma_squared / (static_cast<long double>(plan.z_base) * plan.z_base);
  const long double beta_squared =
    static_cast<long double>(plan.beta_next) * plan.beta_next - old_z_squared + gamma_squared;
  if (!(beta_squared > 0.0L) || !std::isfinite(static_cast<double>(beta_squared))) {
    throw std::runtime_error("derived Section 5.6 beta_next is invalid");
  }
  plan.beta_next = static_cast<double>(std::sqrt(beta_squared));

  if (plan.nu > std::numeric_limits<size_t>::max() - plan.mu) {
    throw std::overflow_error("derived Section 5.6 r_next overflows size_t");
  }
  plan.r_next = plan.nu + plan.mu;
  plan.n_next = std::max(ceil_div_size(param.n, plan.nu), ceil_div_size(plan.auxiliary_len, plan.mu));

  // These bases are not used for an outer commitment in the final execution,
  // but keeping deterministic values makes the public parameter transcript
  // canonical and leaves the generic relation representation well formed.
  const DecompositionChoice next_choice =
    choose_decomposition(plan.n_next, plan.r_next, plan.beta_next);
  plan.digits1 = next_choice.digits1;
  plan.digits2 = next_choice.digits2;
  plan.digits3 = next_choice.digits1;
  plan.base1 = next_choice.base1;
  plan.base2 = next_choice.base2;
  plan.base3 = next_choice.base1;
  return plan;
}

LabradorTransitionPlan derive_paper_schedule_transition(
  size_t one_based_level, double beta)
{
  const auto& schedule = icicle::labrador::backend_config::PAPER_SCHEDULE;
  if (one_based_level == 0 || one_based_level >= schedule.size()) {
    throw std::invalid_argument("paper schedule has no transition after the requested level");
  }
  const auto& current = schedule[one_based_level - 1];
  const auto& next = schedule[one_based_level];
  const LabradorDecompositionPlan decomposition =
    derive_paper_schedule_decomposition(one_based_level, beta);
  LabradorTransitionPlan plan = derive_transition_plan(
    current.n,
    current.r,
    current.kappa,
    decomposition.base1,
    decomposition.base2,
    decomposition.base3,
    decomposition.digits1,
    decomposition.digits2,
    decomposition.digits3,
    beta);

  if (current.section_5_6_tail) {
    const long double gamma_squared = static_cast<long double>(beta) * beta * LABRADOR_TAU;
    const long double old_z_squared =
      2.0L * gamma_squared /
      (static_cast<long double>(plan.z_base) * plan.z_base);
    const long double beta_squared =
      static_cast<long double>(plan.beta_next) * plan.beta_next - old_z_squared + gamma_squared;
    if (!(beta_squared > 0.0L) || !std::isfinite(static_cast<double>(beta_squared))) {
      throw std::runtime_error("derived scheduled Section 5.6 beta_next is invalid");
    }
    plan.beta_next = static_cast<double>(std::sqrt(beta_squared));
  }

  plan.mu = current.mu_to_next;
  plan.nu = current.nu_to_next;
  plan.n_next = next.n;
  plan.r_next = next.r;
  const size_t expected_r =
    (current.section_5_6_tail ? plan.nu : 2 * plan.nu) + plan.mu;
  if (plan.mu == 0 || plan.nu == 0 || expected_r != plan.r_next) {
    throw std::runtime_error("generated paper schedule has an invalid multiplicity split");
  }
  const size_t minimum_n =
    std::max(ceil_div_size(current.n, plan.nu), ceil_div_size(plan.auxiliary_len, plan.mu));
  if (plan.n_next < minimum_n) {
    throw std::runtime_error("generated paper schedule cannot hold the complete recursion witness");
  }

  const LabradorDecompositionPlan next_decomposition =
    derive_paper_schedule_decomposition(one_based_level + 1, plan.beta_next);
  plan.digits1 = next_decomposition.digits1;
  plan.digits2 = next_decomposition.digits2;
  plan.digits3 = next_decomposition.digits3;
  plan.base1 = next_decomposition.base1;
  plan.base2 = next_decomposition.base2;
  plan.base3 = next_decomposition.base3;
  return plan;
}

LabradorTransitionPlan derive_protocol_transition_plan(
  const LabradorParam& param, bool final_transition)
{
  if (param.paper_schedule_level == 0) {
    return final_transition ? derive_final_transition_plan(param) : derive_transition_plan(param);
  }

  const auto& schedule = icicle::labrador::backend_config::PAPER_SCHEDULE;
  const size_t index = param.paper_schedule_level - 1;
  if (index >= schedule.size() || index + 1 >= schedule.size()) {
    throw std::invalid_argument("paper schedule has no transition after the current level");
  }
  const auto& current = schedule[index];
  if (param.n != current.n || param.r != current.r || param.kappa != current.kappa ||
      param.kappa1 != current.kappa1 || param.kappa2 != current.kappa2) {
    throw std::runtime_error("runtime parameter does not match its generated paper schedule row");
  }
  if (final_transition != current.section_5_6_tail) {
    throw std::runtime_error("paper schedule and recursion count disagree on the Section 5.6 transition");
  }

  const LabradorDecompositionPlan expected =
    derive_paper_schedule_decomposition(param.paper_schedule_level, param.beta);
  if (param.base1 != expected.base1 || param.base2 != expected.base2 || param.base3 != expected.base3 ||
      param.digits1 != expected.digits1 || param.digits2 != expected.digits2 ||
      param.digits3 != expected.digits3) {
    throw std::runtime_error("runtime decomposition does not match its generated paper schedule row");
  }
  return derive_paper_schedule_transition(param.paper_schedule_level, param.beta);
}

size_t secure_msis_rank()
{
  const double log_delta = log2(icicle::labrador::backend_config::ROOT_HERMITE_DELTA);
  const double log_q = log2(get_q<Zq>());

  double k_f = pow(log_q - 1.0, 2) / 4 / log_delta / log_q / Rq::d;
  return ceil(k_f);
}

size_t RecursionPreparer::z0_begin_idx() const { return 0; }

size_t RecursionPreparer::z1_begin_idx() const { return nu * n_prime; }

size_t RecursionPreparer::t_begin_idx() const { return (decompose_z ? 2 * nu : nu) * n_prime; }

size_t RecursionPreparer::g_begin_idx() const { return t_begin_idx() + t_len; }

size_t RecursionPreparer::h_begin_idx() const { return g_begin_idx() + g_len; }

eIcicleError RecursionPreparer::copy_like_z0(Rq* dst, const Rq* src) const
{
  // copy to dst[z0_begin_idx() : z0_begin_idx() + n]
  return icicle_copy(&dst[z0_begin_idx()], src, prev_n * sizeof(Rq));
}

eIcicleError RecursionPreparer::copy_like_z1(Rq* dst, const Rq* src) const
{
  if (!decompose_z) { return eIcicleError::INVALID_ARGUMENT; }
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
  size_t nu,
  bool decompose_z)
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

  const LabradorTransitionPlan transition =
    derive_protocol_transition_plan(prev_param, !decompose_z);
  RecursionPreparer preparer{
    prev_param, mu, nu, base0, decompose_z, transition.n_next};

  size_t n_prime = preparer.n_prime;
  size_t t_len = preparer.t_len;
  size_t g_len = preparer.g_len;
  size_t h_len = preparer.h_len;
  size_t r_prime = preparer.r_prime;
  // Step 7: Let recursion_instance be a new empty LabradorInstance

  std::vector<std::byte> new_ajtai_seed(prev_param.ajtai_seed);
  new_ajtai_seed.push_back(std::byte('1'));

  if (transition.z_base != base0 || transition.mu != mu || transition.nu != nu ||
      transition.n_next != n_prime || transition.r_next != r_prime) {
    throw std::runtime_error("prover/verifier LaBRADOR transition plan mismatch");
  }
  size_t next_kappa = prev_param.kappa;
  size_t next_kappa1 = prev_param.kappa1;
  size_t next_kappa2 = prev_param.kappa2;
  size_t next_schedule_level = 0;
  if (prev_param.paper_schedule_level != 0) {
    const auto& next =
      icicle::labrador::backend_config::PAPER_SCHEDULE[prev_param.paper_schedule_level];
    next_kappa = next.kappa;
    next_kappa1 = next.kappa1;
    next_kappa2 = next.kappa2;
    next_schedule_level = prev_param.paper_schedule_level + 1;
  }
  LabradorParam recursion_param{
    r_prime,
    n_prime,
    new_ajtai_seed,
    next_kappa,
    next_kappa1,
    next_kappa2,
    transition.base1,
    transition.base2,
    transition.base3,
    transition.beta_next,
    transition.digits1,
    transition.digits2,
    transition.digits3,
    !decompose_z,
    decompose_z ? 0 : nu,
    next_schedule_level,
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
    if (decompose_z) {
      ICICLE_CHECK(preparer.copy_like_z1(new_constraint.phi.data(), &A_t[i * n]));
      // new_constraint.phi[nu+ j] = base0*new_constraint.phi[nu+ j]
      Zq base0_scalar = Zq::from(base0);
      ICICLE_CHECK(scalar_mul_vec(
        &base0_scalar, reinterpret_cast<const Zq*>(&new_constraint.phi[preparer.z1_begin_idx()]), n * d, {},
        reinterpret_cast<Zq*>(&new_constraint.phi[preparer.z1_begin_idx()])));
    }

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
    if (decompose_z) {
      Zq b0_zq = Zq::from(base0);
      // c_times_phi = base0 * c_times_phi
      scalar_mul_vec(
        &b0_zq, reinterpret_cast<Zq*>(c_times_phi.data()), c_times_phi.size() * d, {},
        reinterpret_cast<Zq*>(c_times_phi.data()));

      ICICLE_CHECK(preparer.copy_like_z1(step11_constraint.phi.data(), c_times_phi.data()));
    }

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
    return constant_tq(c);
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

    for (size_t i = 0; i < nu; i++) {
      // a[i,i] = 1
      step13_constraint.a[i * r_prime + i] = poly_one;
      if (decompose_z) {
        const Zq b0 = Zq::from(base0);
        Tq b0_poly = const_poly(b0);
        Tq b0_sq_poly = const_poly(b0 * b0);
        // a[i+nu, i+nu] = base0^2
        step13_constraint.a[(i + nu) * r_prime + (i + nu)] = b0_sq_poly;
        // a[i, i+nu] = base0
        step13_constraint.a[(i + nu) * r_prime + i] = b0_poly;
        // a[i+nu, i] = base0
        step13_constraint.a[i * r_prime + i + nu] = b0_poly;
      }
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
