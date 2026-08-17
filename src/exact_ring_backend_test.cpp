#include "labrador.h"
#include "lnplabrador_backend_params.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace icicle::labrador;

namespace {

using Wide = __int128_t;
constexpr uint64_t Q = backend_config::RING_MODULUS;

uint64_t value(const Zq& input)
{
  return static_cast<uint64_t>(input.limbs_storage.limbs[0]) |
         (static_cast<uint64_t>(input.limbs_storage.limbs[1]) << 32);
}

Zq scalar(uint64_t input) { return Zq::from_u64(input % Q); }

uint64_t reduce(Wide input)
{
  Wide remainder = input % static_cast<Wide>(Q);
  if (remainder < 0) { remainder += Q; }
  return static_cast<uint64_t>(remainder);
}

Tq naive_product(const Tq& a, const Tq& b)
{
  std::array<Wide, Tq::d> accumulator{};
  for (size_t i = 0; i < Tq::d; ++i) {
    for (size_t j = 0; j < Tq::d; ++j) {
      const Wide product = static_cast<Wide>(value(a.values[i])) * value(b.values[j]);
      const size_t degree = i + j;
      if (degree < Tq::d) {
        accumulator[degree] += product;
      } else {
        accumulator[degree - Tq::d] -= product;
      }
    }
  }
  Tq output{};
  for (size_t i = 0; i < Tq::d; ++i) { output.values[i] = scalar(reduce(accumulator[i])); }
  return output;
}

bool equal(const Tq& a, const Tq& b)
{
  for (size_t i = 0; i < Tq::d; ++i) {
    if (a.values[i] != b.values[i]) { return false; }
  }
  return true;
}

uint64_t xorshift64(uint64_t& state)
{
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

Tq random_polynomial(uint64_t& state)
{
  Tq output{};
  for (auto& coefficient : output.values) { coefficient = scalar(xorshift64(state)); }
  return output;
}

} // namespace

int main()
{
  static_assert(Q == ((uint64_t{1} << 40) - 195), "test must use the requested q40 modulus");
  static_assert(!coefficient_backend::HAS_DIRECT_NEGACYCLIC_NTT, "this q40 cannot use a direct degree-64 NTT");

  uint64_t state = 0x4c61425241444f52ULL;
  for (size_t trial = 0; trial < 10000; ++trial) {
    const uint64_t a = xorshift64(state) % Q;
    const uint64_t b = xorshift64(state) % Q;
    const uint64_t expected = static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) % Q);
    if (value(scalar(a) * scalar(b)) != expected) {
      std::cerr << "q40 scalar multiplication mismatch at trial " << trial << '\n';
      return 1;
    }
  }

  for (size_t trial = 0; trial < 32; ++trial) {
    const Tq a = random_polynomial(state);
    const Tq b = random_polynomial(state);
    Tq actual{};
    if (vector_mul(&a, &b, 1, {}, &actual) != eIcicleError::SUCCESS ||
        !equal(actual, naive_product(a, b))) {
      std::cerr << "Karatsuba/naive negacyclic mismatch at trial " << trial << '\n';
      return 1;
    }
  }

  // x^63 * x = x^64 = -1 in Zq[x]/(x^64+1).
  Tq x63{}, x{}, folded{};
  x63.values[63] = Zq::one();
  x.values[1] = Zq::one();
  if (vector_mul(&x63, &x, 1, {}, &folded) != eIcicleError::SUCCESS ||
      value(folded.values[0]) != Q - 1) {
    std::cerr << "negacyclic x^64 fold failed\n";
    return 1;
  }

  const Tq original = random_polynomial(state);
  Tq transformed{}, round_trip{};
  if (ntt(&original, 1, NTTDir::kForward, {}, &transformed) != eIcicleError::SUCCESS ||
      ntt(&transformed, 1, NTTDir::kInverse, {}, &round_trip) != eIcicleError::SUCCESS ||
      !equal(original, round_trip)) {
    std::cerr << "coefficient representation round trip failed\n";
    return 1;
  }

  // Exercise a matrix dot product and the in-place alias required by
  // LabradorInstance::agg_equality_constraints().
  std::array<Tq, 3> row{random_polynomial(state), random_polynomial(state), random_polynomial(state)};
  std::array<Tq, 3> column{random_polynomial(state), random_polynomial(state), random_polynomial(state)};
  Tq dot{};
  if (matmul(row.data(), 1, 3, column.data(), 3, 1, {}, &dot) != eIcicleError::SUCCESS) {
    std::cerr << "coefficient matmul failed\n";
    return 1;
  }
  Tq expected{};
  for (size_t k = 0; k < 3; ++k) {
    const Tq term = naive_product(row[k], column[k]);
    for (size_t i = 0; i < Tq::d; ++i) { expected.values[i] = expected.values[i] + term.values[i]; }
  }
  if (!equal(dot, expected)) {
    std::cerr << "coefficient matmul differs from naive dot product\n";
    return 1;
  }

  Tq multiplier = constant_tq(scalar(7));
  Tq aliased = row[0];
  const Tq alias_expected = naive_product(multiplier, aliased);
  if (matmul(&multiplier, 1, 1, &aliased, 1, 1, {}, &aliased) != eIcicleError::SUCCESS ||
      !equal(aliased, alias_expected)) {
    std::cerr << "in-place coefficient matmul failed\n";
    return 1;
  }

  std::cout << "Exact q40 ring backend PASSED: q=" << Q
            << ", mode=" << coefficient_backend::MODE_NAME
            << ", scalar_products=10000, random_polynomial_products=32\n";
  return 0;
}
