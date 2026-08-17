#include "icicle/backend/cpu/challenge_keccak_xof.h"
#include "icicle/fields/field_config.h"
#include "icicle/hash/keccak.h"
#include "icicle/operator_norm.h"
#include "icicle/random_sampling.h"
#include "icicle/runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace field_config;
using namespace icicle;

namespace {

bool fail(const char* message)
{
  std::cerr << "sampler hardening smoke: " << message << '\n';
  return false;
}

bool test_counter_xof()
{
  std::vector<std::byte> encoded;
  cpu::challenge_xof_detail::append_u64_le(encoded, UINT64_C(0x0807060504030201));
  for (size_t i = 0; i < encoded.size(); ++i) {
    if (encoded[i] != static_cast<std::byte>(i + 1)) { return fail("counter encoding is not little-endian"); }
  }

  std::array<std::byte, 32> seed{};
  for (size_t i = 0; i < seed.size(); ++i) {
    seed[i] = static_cast<std::byte>(i);
  }
  auto keccak512 = Keccak512::create();
  cpu::Keccak512CounterXof a(keccak512, "sampler-smoke", seed.data(), seed.size(), {7, 11});
  cpu::Keccak512CounterXof b(keccak512, "sampler-smoke", seed.data(), seed.size(), {7, 11});
  cpu::Keccak512CounterXof different_context(
    keccak512, "sampler-smoke", seed.data(), seed.size(), {7, 12});

  // 1025 bits necessarily crosses two refill boundaries and enters block 3.
  bool context_differs = false;
  for (size_t bit = 0; bit < 1025; ++bit) {
    const bool a_bit = a.next_bit();
    if (a_bit != b.next_bit()) { return fail("counter stream is not deterministic"); }
    context_differs |= a_bit != different_context.next_bit();
  }
  if (a.blocks_generated() != 3 || b.blocks_generated() != 3) {
    return fail("counter stream did not exercise the >512-bit refill path");
  }
  if (!context_differs) { return fail("counter-stream context separation failed"); }

  cpu::Keccak512CounterXof different_domain(
    keccak512, "sampler-smoke-other-domain", seed.data(), seed.size(), {7, 11});
  cpu::Keccak512CounterXof original_domain(
    keccak512, "sampler-smoke", seed.data(), seed.size(), {7, 11});
  bool domain_differs = false;
  for (size_t bit = 0; bit < 512; ++bit) {
    domain_differs |= different_domain.next_bit() != original_domain.next_bit();
  }
  return domain_differs || fail("counter-stream domain separation failed");
}

bool canonical(const field_t& value)
{
  return field_t::lt(value, field_t{field_t::get_modulus()});
}

uint64_t as_u64(const field_t& value)
{
  static_assert(field_t::TLC == 2, "q40 smoke expects two 32-bit limbs");
  return uint64_t{value.limbs_storage.limbs[0]} | (uint64_t{value.limbs_storage.limbs[1]} << 32);
}

bool test_scalar_reduction()
{
  const auto modulus_storage = field_t::get_modulus();
  const uint64_t modulus = uint64_t{modulus_storage.limbs[0]} | (uint64_t{modulus_storage.limbs[1]} << 32);
  const std::array<uint64_t, 5> inputs = {0, modulus - 1, modulus, modulus + 1, UINT64_MAX};
  for (uint64_t input : inputs) {
    std::array<std::byte, 8> encoded{};
    for (unsigned shift = 0; shift < 64; shift += 8) {
      encoded[shift / 8] = static_cast<std::byte>((input >> shift) & 0xff);
    }
    const field_t reduced = field_t::reduce_from_bytes(encoded.data());
    if (!canonical(reduced) || as_u64(reduced) != input % modulus) {
      return fail("reduce_from_bytes failed at a modulus boundary");
    }
  }
  for (size_t i = 0; i < 4096; ++i) {
    if (!canonical(field_t::rand_host())) { return fail("rand_host emitted a non-canonical coefficient"); }
  }
  return true;
}

bool test_uniform_ring_sampler()
{
  constexpr size_t count = 4096;
  std::array<std::byte, 32> seed{};
  for (size_t i = 0; i < seed.size(); ++i) {
    seed[i] = static_cast<std::byte>(0xa5U ^ i);
  }
  std::vector<field_t> first(count), second(count);
  if (random_sampling(count, false, seed.data(), seed.size(), {}, first.data()) != eIcicleError::SUCCESS ||
      random_sampling(count, false, seed.data(), seed.size(), {}, second.data()) != eIcicleError::SUCCESS) {
    return fail("uniform sampler returned an error");
  }
  bool nonconstant = false;
  for (size_t i = 0; i < count; ++i) {
    if (first[i] != second[i]) { return fail("uniform sampler is not deterministic"); }
    if (!canonical(first[i])) { return fail("uniform sampler emitted a non-canonical coefficient"); }
    if (i != 0) { nonconstant |= first[i] != first[0]; }
  }
  return nonconstant || fail("uniform sampler output is unexpectedly constant");
}

bool test_challenge_sampler()
{
  constexpr size_t count = 64;
  constexpr uint32_t zeros = 23;
  constexpr uint32_t ones = 31;
  constexpr uint32_t twos = 10;
  constexpr int64_t norm_bound = 15;
  static_assert(zeros + ones + twos == Rq::d, "smoke parameters require degree 64");

  std::array<std::byte, 32> seed{};
  for (size_t i = 0; i < seed.size(); ++i) {
    seed[i] = static_cast<std::byte>(i);
  }
  std::vector<Rq> first(count), second(count);
  if (sample_challenge_space_polynomials(
        seed.data(), seed.size(), count, ones, twos, norm_bound, {}, first.data()) != eIcicleError::SUCCESS ||
      sample_challenge_space_polynomials(
        seed.data(), seed.size(), count, ones, twos, norm_bound, {}, second.data()) != eIcicleError::SUCCESS) {
    return fail("challenge sampler returned an error");
  }

  const field_t two = field_t::one() + field_t::one();
  const field_t neg_one = field_t::one().neg();
  const field_t neg_two = two.neg();
  const std::unordered_map<field_t, int64_t> balanced = {
    {field_t::zero(), 0}, {field_t::one(), 1}, {neg_one, -1}, {two, 2}, {neg_two, -2},
  };

  for (size_t poly_idx = 0; poly_idx < count; ++poly_idx) {
    if (first[poly_idx] != second[poly_idx]) { return fail("challenge sampler is not deterministic"); }
    uint32_t observed_zeros = 0;
    uint32_t observed_ones = 0;
    uint32_t observed_twos = 0;
    opnorm::Poly opnorm_poly{};
    for (size_t coeff_idx = 0; coeff_idx < Rq::d; ++coeff_idx) {
      const field_t& coefficient = first[poly_idx].values[coeff_idx];
      if (!canonical(coefficient)) { return fail("challenge sampler emitted a non-canonical coefficient"); }
      const auto it = balanced.find(coefficient);
      if (it == balanced.end()) { return fail("challenge sampler emitted a coefficient outside {0, +/-1, +/-2}"); }
      opnorm_poly[coeff_idx] = it->second;
      observed_zeros += it->second == 0;
      observed_ones += it->second == 1 || it->second == -1;
      observed_twos += it->second == 2 || it->second == -2;
    }
    if (observed_zeros != zeros || observed_ones != ones || observed_twos != twos) {
      return fail("challenge coefficient counts are incorrect");
    }
    if (opnorm::operator_norm(opnorm_poly) > norm_bound) {
      return fail("challenge operator norm exceeds 15");
    }
  }
  return true;
}

} // namespace

int main()
{
  icicle_load_backend_from_env_or_default();
  if (icicle_set_device(Device{"CPU", 0}) != eIcicleError::SUCCESS) {
    std::cerr << "sampler hardening smoke: could not select CPU backend\n";
    return 1;
  }
  if (!test_counter_xof() || !test_scalar_reduction() || !test_uniform_ring_sampler() ||
      !test_challenge_sampler()) {
    return 1;
  }
  std::cout << "sampler hardening smoke: PASS (counter refill, scalar reduction, q-uniformity, 23/31/10, "
               "norm <= 15)\n";
  return 0;
}
