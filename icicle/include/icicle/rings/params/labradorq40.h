#pragma once

#include "icicle/math/storage.h"
#include "icicle/rings/integer_ring.h"
#include "icicle/fields/params_gen.h"
#include "icicle/rings/polynomial_ring.h"

namespace labradorq40 {

  /**
   * Prime coefficient ring used by the degree-64 LaBRADOR backend.
   *
   * q = 2^40 - 195 = 1,099,511,627,581 is the prime modulus used by
   * the requested TeX and the LaBRADOR C reference implementation's LOGQ40
   * option (the paper's concrete table uses q near 2^32).
   *
   * q mod 128 = 61 and ord_128(q) = 32, so q has no primitive 128-th
   * root of unity.  Degree-64 ring products must therefore not use the
   * direct negacyclic NTT exposed by ICICLE for split coefficient fields.
   * The LaBRADOR frontend supplies an exact coefficient-domain backend.
   */
  struct zq_config {
    static constexpr storage<2> modulus = {0xffffff3d, 0x000000ff};
    static constexpr unsigned limbs_count = 2;
    static constexpr unsigned modulus_bit_count = 40;
    static constexpr storage<limbs_count> zero = {};
    static constexpr storage<limbs_count> one = {1};
    static constexpr storage<limbs_count> modulus_2 =
      host_math::template left_shift<limbs_count, 1>(modulus);
    static constexpr storage<limbs_count> modulus_4 =
      host_math::template left_shift<limbs_count, 1>(modulus_2);
    static constexpr storage<limbs_count> neg_modulus =
      params_gen::template get_difference_no_carry<limbs_count>(zero, modulus);
    static constexpr storage<2 * limbs_count> modulus_squared =
      params_gen::template get_square<limbs_count, 0>(modulus);
    static constexpr storage<2 * limbs_count> modulus_squared_2 =
      host_math::template left_shift<2 * limbs_count, 1>(modulus_squared);
    static constexpr storage<2 * limbs_count> modulus_squared_4 =
      host_math::template left_shift<2 * limbs_count, 1>(modulus_squared_2);
    static constexpr storage<limbs_count> m =
      params_gen::template get_m<limbs_count, 2 * modulus_bit_count>(modulus);
    static constexpr storage<limbs_count> montgomery_r =
      params_gen::template get_montgomery_constant<limbs_count, false>(modulus);
    static constexpr storage<limbs_count> montgomery_r_inv =
      params_gen::template get_montgomery_constant<limbs_count, true>(modulus);

    // The generic constexpr estimator assumes the Barrett exponent occupies
    // the most-significant storage limb.  That is not true for this 40-bit
    // modulus in a 64-bit container.  Its exact error bound is < 0.000019,
    // hence one final conditional subtraction is sufficient.
    static constexpr unsigned num_of_reductions = 1;

    // q supports roots only through order 4.  This value is a primitive
    // fourth root (rou^2 = -1).  It is sufficient for generic scalar-ring
    // metadata, but deliberately not for a degree-64 direct NTT.
    static constexpr storage<2> rou = {0xab7c3cfa, 0x000000fa};
    static constexpr unsigned omegas_count = 2;

    static constexpr storage_array<omegas_count, limbs_count> inv =
      params_gen::template get_invs<limbs_count, omegas_count>(modulus);
  };

  using Zq = IntegerRing<zq_config>;
  using scalar_t = Zq;
  using field_t = Zq;

  using PolyRing = icicle::PolynomialRing<Zq, 64>;
  using Rq = PolyRing;

} // namespace labradorq40
