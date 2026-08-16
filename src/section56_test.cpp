#include "examples_utils.h"
#include "proof_codec.h"
#include "prover.h"
#include "verifier.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool streamed_jl_matches_full_matrix()
{
  constexpr size_t row_size = 5;
  constexpr size_t rows_count = 17;
  constexpr size_t forced_chunk_rows = 3;
  const std::string seed_string = "streamed-jl-equivalence";
  const auto* seed = reinterpret_cast<const std::byte*>(seed_string.data());
  std::vector<Zq> weights(rows_count);
  for (size_t i = 0; i < rows_count; ++i) {
    weights[i] = Zq::from(static_cast<uint64_t>(3 * i + 1));
  }

  DeviceVector<Rq> full_q(rows_count * row_size);
  DeviceVector<Rq> weighted_q(rows_count * row_size);
  DeviceVector<Tq> full_result(row_size);
  DeviceVector<Tq> streamed_result(row_size);
  ICICLE_CHECK(icicle::labrador::get_jl_matrix_rows(
    seed,
    seed_string.size(),
    row_size,
    0,
    rows_count,
    true,
    {},
    full_q.data()));

  VecOpsConfig scale_config = default_vec_ops_config();
  scale_config.is_a_on_device = false;
  scale_config.is_b_on_device = true;
  scale_config.is_result_on_device = true;
  scale_config.batch_size = rows_count;
  ICICLE_CHECK(scalar_mul_vec(
    weights.data(),
    reinterpret_cast<const Zq*>(full_q.data()),
    row_size * Rq::d,
    scale_config,
    reinterpret_cast<Zq*>(weighted_q.data())));

  VecOpsConfig sum_config = default_vec_ops_config();
  sum_config.batch_size = row_size * Rq::d;
  sum_config.columns_batch = true;
  sum_config.is_a_on_device = true;
  sum_config.is_result_on_device = true;
  ICICLE_CHECK(vector_sum<Zq>(
    reinterpret_cast<const Zq*>(weighted_q.data()),
    rows_count,
    sum_config,
    reinterpret_cast<Zq*>(full_result.data())));
  ICICLE_CHECK(ntt(full_result.data(), row_size, NTTDir::kForward, {}, full_result.data()));

  const size_t scratch_bytes = forced_chunk_rows * row_size * sizeof(Rq);
  if (jl_aggregation_chunk_rows(row_size, rows_count, scratch_bytes) != forced_chunk_rows) {
    return false;
  }
  aggregate_jl_projection_rows_ntt(
    seed,
    seed_string.size(),
    row_size,
    weights.data(),
    rows_count,
    streamed_result.data(),
    scratch_bytes);

  const std::vector<Tq> full_host = full_result.as_host_vector();
  const std::vector<Tq> streamed_host = streamed_result.as_host_vector();
  return poly_vec_eq(full_host.data(), streamed_host.data(), row_size);
}

bool decoder_rejects(
  const std::vector<std::byte>& encoded,
  const LabradorProofDecodeLimits& limits = {})
{
  try {
    (void)decode_labrador_proof(encoded, limits);
    return false;
  } catch (const std::invalid_argument&) {
    return true;
  } catch (const std::length_error&) {
    return true;
  }
}

void overwrite_u64_le(std::vector<std::byte>& bytes, size_t offset, uint64_t value)
{
  if (offset > bytes.size() || bytes.size() - offset < sizeof(value)) {
    throw std::invalid_argument("test mutation offset is out of range");
  }
  for (size_t i = 0; i < sizeof(value); ++i) {
    bytes[offset + i] = std::byte((value >> (8 * i)) & 0xffU);
  }
}

} // namespace

int main(int argc, char** argv)
{
  try {
    try_load_and_set_backend_device(argc, argv);

    if (!streamed_jl_matches_full_matrix()) {
      std::cerr << "Streamed JL aggregation differs from the full-matrix path\n";
      return 1;
    }
    std::cout << "Streamed JL exact-equivalence test PASSED (17 rows, 3-row chunks)\n";

    double schedule_beta = std::sqrt(
      static_cast<double>(icicle::labrador::backend_config::RING_MODULUS));
    const auto& schedule = icicle::labrador::backend_config::PAPER_SCHEDULE;
    for (size_t level = 1; level < schedule.size(); ++level) {
      const LabradorTransitionPlan transition =
        derive_paper_schedule_transition(level, schedule_beta);
      const auto& expected = schedule[level];
      if (transition.n_next != expected.n || transition.r_next != expected.r ||
          transition.auxiliary_len > schedule[level - 1].mu_to_next * expected.n) {
        std::cerr << "Generated paper schedule transition test failed at level " << level << "\n";
        return 1;
      }
      schedule_beta = transition.beta_next;
    }

    constexpr size_t r = 2;
    constexpr size_t n = 1;
    std::vector<Rq> witness(r * n, zero());
    const std::string ajtai_seed = "section-5.6-test-crs";
    LabradorParam param{
      r,
      n,
      {reinterpret_cast<const std::byte*>(ajtai_seed.data()),
       reinterpret_cast<const std::byte*>(ajtai_seed.data()) + ajtai_seed.size()},
      1,
      1,
      1,
      1U << 20,
      1U << 20,
      1U << 20,
      2.0,
      2,
      2,
      2,
    };
    LabradorInstance instance{param};
    instance.add_equality_constraint(EqualityInstance(r, n));

    const std::string oracle_seed = "section-5.6-test-oracle";
    Oracle bound_oracle = create_oracle_seed(
      reinterpret_cast<const std::byte*>(oracle_seed.data()),
      oracle_seed.size(),
      instance);
    LabradorInstance changed_statement{instance};
    changed_statement.equality_constraints[0].b.values[0] = Zq::one();
    Oracle changed_oracle = create_oracle_seed(
      reinterpret_cast<const std::byte*>(oracle_seed.data()),
      oracle_seed.size(),
      changed_statement);
    if (bound_oracle.state_ == changed_oracle.state_) {
      std::cerr << "Canonical oracle seed did not bind the public constraint contents\n";
      return 1;
    }

    Rq endian_test = zero();
    constexpr uint64_t ENDIAN_SENTINEL = UINT64_C(0x0102030405);
    endian_test.values[0] = Zq::from_u64(ENDIAN_SENTINEL);
    const std::vector<Rq> endian_polynomials{endian_test};
    const std::vector<std::byte> canonical_message =
      section_5_6_first_message(endian_polynomials);
    constexpr std::string_view FIRST_DOMAIN = "LaBRADOR-Section-5.6-first-v2";
    const size_t first_coefficient_offset =
      sizeof(uint64_t) + FIRST_DOMAIN.size() + sizeof(uint64_t) * 2;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
      const uint8_t expected = static_cast<uint8_t>(ENDIAN_SENTINEL >> (8 * i));
      if (std::to_integer<uint8_t>(canonical_message[first_coefficient_offset + i]) != expected) {
        std::cerr << "Section 5.6 field encoding is not canonical little-endian\n";
        return 1;
      }
    }

    LabradorProver prover{
      instance,
      witness,
      reinterpret_cast<const std::byte*>(oracle_seed.data()),
      oracle_seed.size(),
      2};
    auto [transcripts, proof] = prover.prove();

    if (!proof.uses_section_5_6 || transcripts.size() != 2 ||
        !transcripts.back().prover_msg.u1.empty() || !transcripts.back().prover_msg.u2.empty()) {
      std::cerr << "Section 5.6 proof mode/outer-commitment test failed\n";
      return 1;
    }
    const LabradorSection56Proof& tail = proof.section_5_6;
    const size_t final_r = (tail.h_polynomial_count() + 1) / 2;
    if (tail.g_polynomial_count() != 2 * tail.primary_count + 1 ||
        tail.h_polynomial_count() != 2 * final_r - 1 ||
        tail.t_polynomial_count() != final_r * param.kappa) {
      std::cerr << "Section 5.6 compact vector-count test failed\n";
      return 1;
    }

    std::vector<BaseProverMessages> messages;
    for (const PartialTranscript& transcript : transcripts) { messages.push_back(transcript.prover_msg); }
    LabradorVerifier verifier{
      instance,
      messages,
      proof,
      reinterpret_cast<const std::byte*>(oracle_seed.data()),
      oracle_seed.size(),
      2};
    if (!verifier.verify()) {
      std::cerr << "Valid Section 5.6 proof was rejected\n";
      return 1;
    }

    LabradorProofCodecStats codec_stats;
    const std::vector<std::byte> encoded =
      encode_labrador_proof(transcripts, proof, &codec_stats);
    LabradorDecodedProof decoded = decode_labrador_proof(encoded);
    LabradorVerifier decoded_verifier{
      instance,
      decoded.prover_messages,
      decoded.final_proof,
      reinterpret_cast<const std::byte*>(oracle_seed.data()),
      oracle_seed.size(),
      2};
    if (codec_stats.total_bytes != encoded.size() || !decoded_verifier.verify()) {
      std::cerr << "Bit-packed proof round-trip test failed\n";
      return 1;
    }

    LabradorProofDecodeLimits wire_limit;
    wire_limit.max_encoded_bytes = encoded.size() - 1;
    if (!decoder_rejects(encoded, wire_limit)) {
      std::cerr << "Proof codec accepted a proof above its configured wire budget\n";
      return 1;
    }

    // Every individual vector in this tiny proof fits in 2500 bytes, but the
    // retained vectors do not fit cumulatively.  This guards against treating
    // MAX_POLYNOMIALS as a separate allowance for every attacker-controlled
    // vector.
    LabradorProofDecodeLimits cumulative_limit;
    cumulative_limit.max_decoded_bytes = 2500;
    if (!decoder_rejects(encoded, cumulative_limit)) {
      std::cerr << "Proof codec did not enforce its cumulative decode budget\n";
      return 1;
    }

    // Header layout: magic(8), version(4), recursion count(4), JL_i(8),
    // followed by the first u1 polynomial count.  The count is below the
    // backend polynomial cap but above the 64-MiB native allocation budget;
    // rejection must occur before a payload-sized allocation or read.
    std::vector<std::byte> oversized_vector = encoded;
    constexpr size_t FIRST_U1_COUNT_OFFSET = 8 + 4 + 4 + 8;
    const uint64_t over_budget_polynomials =
      icicle::labrador::backend_config::MAX_PROOF_BYTES / sizeof(Tq) + 1;
    overwrite_u64_le(oversized_vector, FIRST_U1_COUNT_OFFSET, over_budget_polynomials);
    if (!decoder_rejects(oversized_vector)) {
      std::cerr << "Proof codec accepted a single over-budget decoded vector\n";
      return 1;
    }

    std::vector<std::byte> invalid_jl_counter = encoded;
    constexpr size_t FIRST_JL_COUNTER_OFFSET = 8 + 4 + 4;
    overwrite_u64_le(invalid_jl_counter, FIRST_JL_COUNTER_OFFSET, uint64_t{1} << 20);
    if (!decoder_rejects(invalid_jl_counter)) {
      std::cerr << "Proof codec accepted a non-canonical JL retry counter\n";
      return 1;
    }

    LabradorFinalProof tampered = proof;
    tampered.section_5_6.h_diagonal[0].values[0] = Zq::one();
    LabradorVerifier tampered_verifier{
      instance,
      messages,
      tampered,
      reinterpret_cast<const std::byte*>(oracle_seed.data()),
      oracle_seed.size(),
      2};
    if (tampered_verifier.verify()) {
      std::cerr << "Tampered Section 5.6 proof was accepted\n";
      return 1;
    }

    std::cout << "Section 5.6 optimized-final test PASSED: z=" << tail.z_polynomial_count()
              << ", t=" << tail.t_polynomial_count() << ", g=" << tail.g_polynomial_count()
              << ", h=" << tail.h_polynomial_count()
              << ", encoded=" << encoded.size() << " B\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "section56_test: " << error.what() << '\n';
    return 1;
  }
}
