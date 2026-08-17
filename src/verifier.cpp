#include "verifier.h"

void LabradorBaseVerifier::validate_prover_message_shape() const
{
  const BaseProverMessages& message = trs.prover_msg;
  if (message.u1.size() != lab_inst.param.kappa1 ||
      message.u2.size() != lab_inst.param.kappa2 ||
      message.p.size() != lab_inst.param.JL_out ||
      message.b_agg.size() != lab_inst.param.num_aggregation_rounds) {
    throw std::invalid_argument("malformed LaBRADOR prover message dimensions");
  }
  if (message.JL_i >= (1U << 20)) {
    throw std::invalid_argument("LaBRADOR JL retry counter exceeds the canonical limit");
  }
}

void LabradorBaseVerifier::validate_section_5_6_message_shape(
  const LabradorSection56Proof& final_proof) const
{
  const LabradorParam& param = lab_inst.param;
  if (!param.section_5_6_final || param.final_primary_count == 0 ||
      param.final_primary_count > param.r) {
    throw std::invalid_argument("Section 5.6 proof requires a final sparse instance");
  }
  if (!trs.prover_msg.u1.empty() || !trs.prover_msg.u2.empty() ||
      trs.prover_msg.p.size() != param.JL_out ||
      trs.prover_msg.b_agg.size() != param.num_aggregation_rounds) {
    throw std::invalid_argument("malformed Section 5.6 prover-message dimensions");
  }
  if (trs.prover_msg.JL_i >= (1U << 20)) {
    throw std::invalid_argument("LaBRADOR JL retry counter exceeds the canonical limit");
  }
  if (final_proof.primary_count != param.final_primary_count ||
      final_proof.z_hat.size() != param.n ||
      final_proof.t.size() != param.r * param.kappa ||
      final_proof.g_cross.size() != param.final_primary_count ||
      final_proof.g_diagonal.size() != param.final_primary_count ||
      final_proof.h_cross.size() != param.r - 1 ||
      final_proof.h_diagonal.size() != param.r) {
    throw std::invalid_argument("malformed Section 5.6 final-response dimensions");
  }
}

void LabradorBaseVerifier::create_section_5_6_transcript_prefix(
  const LabradorSection56Proof& final_proof)
{
  std::vector<std::byte> first_message = section_5_6_first_message(final_proof.t);
  trs.seed1 = oracle.generate(first_message.data(), first_message.size());

  const size_t JL_i = trs.prover_msg.JL_i;
  const std::vector<Zq>& p = trs.prover_msg.p;
  std::vector<std::byte> jl_buf = canonical_jl_transcript_message(
    "LaBRADOR-Section-5.6-jl-v2", JL_i, p);
  trs.seed2 = oracle.generate(jl_buf.data(), jl_buf.size());

  const size_t L = lab_inst.const_zero_constraints.size();
  const size_t JL_out = lab_inst.param.JL_out;
  const size_t aggregation_rounds = lab_inst.param.num_aggregation_rounds;
  trs.psi.resize(aggregation_rounds * L);
  trs.omega.resize(aggregation_rounds * JL_out);
  std::vector<std::byte> psi_seed(trs.seed2);
  psi_seed.push_back(std::byte('1'));
  if (!trs.psi.empty()) {
    ICICLE_CHECK(random_sampling(
      trs.psi.size(), false, psi_seed.data(), psi_seed.size(), {}, trs.psi.data()));
  }
  std::vector<std::byte> omega_seed(trs.seed2);
  omega_seed.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(
    trs.omega.size(), false, omega_seed.data(), omega_seed.size(), {}, trs.omega.data()));

  const std::vector<Tq>& b_agg = trs.prover_msg.b_agg;
  const std::vector<std::byte> aggregation_message =
    canonical_polynomial_transcript_message(
      "LaBRADOR-Section-5.6-aggregation-v2", b_agg.data(), b_agg.size());
  trs.seed3 = oracle.generate(aggregation_message.data(), aggregation_message.size());

  const size_t K = lab_inst.equality_constraints.size() + aggregation_rounds;
  if (K == 0) { throw std::invalid_argument("Section 5.6 final instance has no constraints"); }
  trs.alpha_hat.resize(K);
  std::vector<std::byte> alpha_seed(trs.seed3);
  alpha_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling(
    K, false, alpha_seed.data(), alpha_seed.size(), {}, trs.alpha_hat.data()));
}

// Fills up trs correctly assuming trs.prover_msg are correctly filled
void LabradorBaseVerifier::create_transcript()
{
  const size_t d = Rq::d;
  // 1. seed1 from canonical u1
  const auto& u1 = trs.prover_msg.u1;
  const std::vector<std::byte> u1_message =
    canonical_polynomial_transcript_message("LaBRADOR-base-u1-v2", u1.data(), u1.size());
  trs.seed1 = oracle.generate(u1_message.data(), u1_message.size());

  // 2. seed2 from JL_i and p
  size_t JL_i = trs.prover_msg.JL_i;
  const std::vector<Zq>& p = trs.prover_msg.p;
  const std::vector<std::byte> jl_buf =
    canonical_jl_transcript_message("LaBRADOR-base-jl-v2", JL_i, p);
  trs.seed2 = oracle.generate(jl_buf.data(), jl_buf.size());

  // 3. psi and omega sampling
  const size_t L = lab_inst.const_zero_constraints.size();
  const size_t JL_out = lab_inst.param.JL_out;
  const size_t num_aggregation_rounds = lab_inst.param.num_aggregation_rounds;
  trs.psi.resize(num_aggregation_rounds * L);
  trs.omega.resize(num_aggregation_rounds * JL_out);
  // psi seed = seed2 || 0x01
  std::vector<std::byte> psi_seed(trs.seed2);
  psi_seed.push_back(std::byte('1'));
  if (trs.psi.size() > 0) {
    ICICLE_CHECK(random_sampling(trs.psi.size(), false, psi_seed.data(), psi_seed.size(), {}, trs.psi.data()));
  }
  // omega seed = seed2 || 0x02
  std::vector<std::byte> omega_seed(trs.seed2);
  omega_seed.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(trs.omega.size(), false, omega_seed.data(), omega_seed.size(), {}, trs.omega.data()));

  // 4. seed3 from canonical msg3 (b_agg)
  const auto& msg3 = trs.prover_msg.b_agg;
  const std::vector<std::byte> msg3_message =
    canonical_polynomial_transcript_message(
      "LaBRADOR-base-aggregation-v2", msg3.data(), msg3.size());
  trs.seed3 = oracle.generate(msg3_message.data(), msg3_message.size());

  // 5. alpha_hat sampling
  // After we aggregate the L const-zero constraints the instance will
  // have `num_aggregation_rounds` extra EqualityInstances, so we must sample
  // α for the *final* size in advance.
  const size_t K = lab_inst.equality_constraints.size() + num_aggregation_rounds;
  assert(K > 0);

  trs.alpha_hat.resize(K);
  std::vector<std::byte> alpha_seed(trs.seed3);
  alpha_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling(K, false, alpha_seed.data(), alpha_seed.size(), {}, trs.alpha_hat.data()));

  // 6. seed4 from canonical u2
  const auto& u2 = trs.prover_msg.u2;
  const std::vector<std::byte> u2_message =
    canonical_polynomial_transcript_message("LaBRADOR-base-u2-v2", u2.data(), u2.size());
  trs.seed4 = oracle.generate(u2_message.data(), u2_message.size());

  // 7. challenges_hat sampling
  size_t n = lab_inst.param.n;
  size_t r = lab_inst.param.r;
  std::vector<Rq> challenge = sample_low_norm_challenges(n, r, trs.seed4.data(), trs.seed4.size());
  trs.challenges_hat.resize(r);
  ntt(challenge.data(), challenge.size(), NTTDir::kForward, {}, trs.challenges_hat.data());
}

// TODO: maybe make BaseProof more consistent by making everything Rq, since we have to convert z_hat to Rq before norm
// check anyway
bool LabradorBaseVerifier::_verify_base_proof(const LabradorBaseCaseProof& base_proof) const
{
  size_t n = lab_inst.param.n;
  size_t r = lab_inst.param.r;
  size_t d = Rq::d;

  auto& z_hat = base_proof.z_hat;
  auto& t_tilde = base_proof.t;
  auto& g_tilde = base_proof.g;
  auto& h_tilde = base_proof.h;
  auto& challenges_hat = trs.challenges_hat;
  auto final_const = lab_inst.equality_constraints[0];

  if (z_hat.size() != n || t_tilde.size() != lab_inst.param.t_len() ||
      g_tilde.size() != lab_inst.param.g_len() || h_tilde.size() != lab_inst.param.h_len()) {
    std::cout << "Malformed final LaBRADOR response dimensions\n";
    return false;
  }

  bool t_tilde_small = true, g_tilde_small = true, h_tilde_small = true;
  size_t base1 = lab_inst.param.base1;
  size_t base2 = lab_inst.param.base2;
  size_t base3 = lab_inst.param.base3;

  // 1. Only the low decomposition limbs are coefficient-bounded.  The final
  // limb is the unrestricted quotient (Figure 3, Lines 11--13).
  const size_t pair_count = r * (r + 1) / 2;
  const size_t t_low_size = (lab_inst.param.digits1 - 1) * r * lab_inst.param.kappa;
  const size_t g_low_size = (lab_inst.param.digits2 - 1) * pair_count;
  const size_t h_low_size = (lab_inst.param.digits3 - 1) * pair_count;
  // bounds + 1 because norm check uses strict inequality
  ICICLE_CHECK(check_norm_bound(
    reinterpret_cast<const Zq*>(t_tilde.data()), t_low_size * d, eNormType::LInfinity, base1 / 2 + 1, {},
    &t_tilde_small));
  // A one-digit Gaussian decomposition has no bounded low limb.  The empty
  // check is vacuously true, while the ICICLE norm API deliberately rejects a
  // zero-sized input.
  if (g_low_size != 0) {
    ICICLE_CHECK(check_norm_bound(
      reinterpret_cast<const Zq*>(g_tilde.data()), g_low_size * d, eNormType::LInfinity, base2 / 2 + 1, {},
      &g_tilde_small));
  }
  ICICLE_CHECK(check_norm_bound(
    reinterpret_cast<const Zq*>(h_tilde.data()), h_low_size * d, eNormType::LInfinity, base3 / 2 + 1, {},
    &h_tilde_small));

  // Fail if any of the LInfinity are large
  if (!(t_tilde_small && h_tilde_small && g_tilde_small)) {
    std::cout << "LInfinity norm check failed\n";
    return false;
  }

  // 2. L2 checks
  bool z_small = true;
  // z = INTT(z_hat)
  std::vector<Rq> z(z_hat.size());
  ICICLE_CHECK(ntt(z_hat.data(), z_hat.size(), NTTDir::kInverse, {}, z.data()));

  uint64_t op_norm_bound = lab_inst.param.op_norm_bound;
  double beta = lab_inst.param.beta;
  // Check ||z|| < op_norm*beta*sqrt(r)
  // NOTE: if n > 2^10 then this fails-- Verifier can again test this using a JL projection
  ICICLE_CHECK(check_norm_bound(
    reinterpret_cast<Zq*>(z.data()), z.size() * d, eNormType::L2, op_norm_bound * beta * sqrt(r), {}, &z_small));

  if (!z_small) {
    std::cout << "L2 norm check for z failed\n";
    return false;
  }

  const LabradorTransitionPlan transition =
    derive_protocol_transition_plan(lab_inst.param, false);
  std::vector<Rq> target_witness = fixed_length_decompose(z, transition.z_base, 2);
  target_witness.insert(target_witness.end(), t_tilde.begin(), t_tilde.end());
  target_witness.insert(target_witness.end(), g_tilde.begin(), g_tilde.end());
  target_witness.insert(target_witness.end(), h_tilde.begin(), h_tilde.end());
  const long double target_norm = coefficient_l2_norm(target_witness);
  if (!(target_norm <= static_cast<long double>(transition.beta_next))) {
    std::cout << "Consolidated target-relation L2 norm exceeds beta'\n";
    return false;
  }

  // 3. Check u1, u2 commitment openings

  // compute NTTs of t_tilde, g_tilde, h_tilde
  std::vector<Tq> t_tilde_hat(t_tilde.size()), g_tilde_hat(g_tilde.size()), h_tilde_hat(h_tilde.size());
  ICICLE_CHECK(ntt(t_tilde.data(), t_tilde.size(), NTTDir::kForward, {}, t_tilde_hat.data()));
  ICICLE_CHECK(ntt(g_tilde.data(), g_tilde.size(), NTTDir::kForward, {}, g_tilde_hat.data()));
  ICICLE_CHECK(ntt(h_tilde.data(), h_tilde.size(), NTTDir::kForward, {}, h_tilde_hat.data()));

  size_t kappa1 = lab_inst.param.kappa1;

  const std::vector<Tq>& A = lab_inst.param.A;
  const std::vector<Tq>& B = lab_inst.param.B;
  const std::vector<Tq>& C = lab_inst.param.C;
  const std::vector<Tq>& D = lab_inst.param.D;
  // v1 = B@T_tilde
  std::vector<Tq> v1 = ajtai_commitment(B, t_tilde_hat.size(), kappa1, t_tilde_hat.data(), t_tilde_hat.size());
  // v2 = C@g_tilde
  std::vector<Tq> v2 = ajtai_commitment(C, g_tilde_hat.size(), kappa1, g_tilde_hat.data(), g_tilde_hat.size());
  // u1 = v1+v2
  std::vector<Tq> u1(kappa1);
  vector_add(v1.data(), v2.data(), kappa1, {}, u1.data());

  // check t_tilde, g_tilde open u1 in trs
  if (!(poly_vec_eq(u1.data(), trs.prover_msg.u1.data(), kappa1))) {
    std::cout << "t_tilde, g_tilde don't open u1 \n";
    return false;
  }

  size_t kappa2 = lab_inst.param.kappa2;
  // u2 = D@h_tilde
  std::vector<Tq> u2 = ajtai_commitment(D, h_tilde_hat.size(), kappa2, h_tilde_hat.data(), h_tilde_hat.size());

  // check h_tilde opens to u2 in trs
  if (!(poly_vec_eq(u2.data(), trs.prover_msg.u2.data(), kappa2))) {
    std::cout << "h_tilde doesn't open u2 \n";
    return false;
  }

  // 4. Check Az = \sum_i c_i*t_i

  // Use ajtai_commitment to compute z_hat @ A
  size_t kappa = lab_inst.param.kappa;
  std::vector<Tq> zA_hat = ajtai_commitment(A, n, kappa, z_hat.data(), n);

  std::vector<Rq> t(r * kappa);
  ICICLE_CHECK(recompose(t_tilde.data(), t_tilde.size(), base1, {}, t.data(), t.size()));
  std::vector<Tq> t_hat(r * kappa), ct_hat(kappa);
  // t_hat = NTT(t)
  ICICLE_CHECK(ntt(t.data(), r * kappa, NTTDir::kForward, {}, t_hat.data()));
  // ct_hat = \sum_i c_i t_i = [c1 c2 ... cr] @ t_hat
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, t_hat.data(), r, kappa, {}, ct_hat.data()));
  // zA_hat == \sum_i c_i t_i
  if (!(poly_vec_eq(zA_hat.data(), ct_hat.data(), kappa))) {
    std::cout << "_verify_base_proof failed: zA != cT \n";
    return false;
  }

  // Compute relevant matrix, vectors for the rest of the checks

  size_t r_choose_2 = pair_count;
  std::vector<Rq> g(r_choose_2);
  ICICLE_CHECK(recompose(g_tilde.data(), g_tilde.size(), base2, {}, g.data(), g.size()));
  std::vector<Rq> G = reconstruct_symm_matrix(g, r);

  std::vector<Tq> G_hat(r * r);
  // G_hat = NTT(G)
  ICICLE_CHECK(ntt(G.data(), r * r, NTTDir::kForward, {}, G_hat.data()));

  std::vector<Rq> h(r_choose_2);
  ICICLE_CHECK(recompose(h_tilde.data(), h_tilde.size(), base3, {}, h.data(), h.size()));
  std::vector<Rq> H = reconstruct_symm_matrix(h, r);

  std::vector<Tq> H_hat(r * r);
  // H_hat = NTT(H)
  ICICLE_CHECK(ntt(H.data(), r * r, NTTDir::kForward, {}, H_hat.data()));

  Tq ip_z_z, c_G_c, c_H_c, ip_a_G, c_Phi_z, trace_H;

  // ip_z_z = <z_hat,z_hat> - inner product of z_hat with itself
  ICICLE_CHECK(matmul(z_hat.data(), 1, n, z_hat.data(), n, 1, {}, &ip_z_z));

  // c_G_c = challenges_hat^T * G_hat * challenges_hat
  // First compute G_hat * challenges_hat
  std::vector<Tq> G_times_c(r);
  ICICLE_CHECK(matmul(G_hat.data(), r, r, challenges_hat.data(), r, 1, {}, G_times_c.data()));
  // Then compute challenges_hat^T * (G_hat * challenges_hat)
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, G_times_c.data(), r, 1, {}, &c_G_c));

  // c_H_c = challenges_hat^T * H_hat * challenges_hat
  // First compute H_hat * challenges_hat
  std::vector<Tq> H_times_c(r);
  ICICLE_CHECK(matmul(H_hat.data(), r, r, challenges_hat.data(), r, 1, {}, H_times_c.data()));
  // Then compute challenges_hat^T * (H_hat * challenges_hat)
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, H_times_c.data(), r, 1, {}, &c_H_c));

  // ip_a_G = <final_const.a, G_hat> - inner product of flattened matrices
  ICICLE_CHECK(matmul(final_const.a.data(), 1, r * r, G_hat.data(), r * r, 1, {}, &ip_a_G));

  // c_Phi_z = challenges_hat^T * final_const.phi * z_hat
  // First compute phi * z_hat
  std::vector<Tq> phi_times_z(r);
  ICICLE_CHECK(matmul(final_const.phi.data(), r, n, z_hat.data(), n, 1, {}, phi_times_z.data()));
  // Then compute challenges_hat^T * (phi * z_hat)
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, phi_times_z.data(), r, 1, {}, &c_Phi_z));

  // compute trace_H = \sum_i H_ii
  ICICLE_CHECK(compute_matrix_trace(H_hat.data(), r, &trace_H));

  // c = challenges
  // 5. Ensure: <z,z> == c^t G c
  if (!(poly_vec_eq(&ip_z_z, &c_G_c, 1))) {
    std::cout << "_verify_base_proof failed: <z,z> != c^t G c \n";
    return false;
  }
  // 6. Ensure: c^t Phi z == c^t H c
  if (!(poly_vec_eq(&c_Phi_z, &c_H_c, 1))) {
    std::cout << "_verify_base_proof failed: c^t Phi z != c^t H c\n";
    return false;
  }

  // 7. Ensure: \sum_ij a_ij G_ij + \sum_i h_ii + b == 0
  // \sum_ij a_ij G_ij + \sum_i h_ii
  Tq ip_a_G_plus_trace_H;
  ICICLE_CHECK(vector_add(&ip_a_G, &trace_H, 1, {}, &ip_a_G_plus_trace_H));

  Tq ip_a_G_plus_trace_H_plus_b;
  ICICLE_CHECK(vector_add(&ip_a_G_plus_trace_H, &final_const.b, 1, {}, &ip_a_G_plus_trace_H_plus_b));

  Tq zero_poly(zero());
  // Check \sum_ij a_ij G_ij + \sum_i h_ii + b == 0
  if (!(poly_vec_eq(&ip_a_G_plus_trace_H_plus_b, &zero_poly, 1))) {
    std::cout << "_verify_base_proof failed: sum_ij a_ij G_ij + sum_i h_ii + b !=0\n";
    return false;
  }
  return true;
}

bool LabradorBaseVerifier::_verify_section_5_6_proof(
  const LabradorSection56Proof& final_proof)
{
  const size_t r = lab_inst.param.r;
  const size_t n = lab_inst.param.n;
  const size_t primary_count = lab_inst.param.final_primary_count;
  const size_t d = Rq::d;
  validate_section_5_6_message_shape(final_proof);

  const std::vector<size_t> order = section_5_6_challenge_order(r, primary_count);
  std::vector<Tq> challenges_hat(r, zero());
  for (size_t round = 0; round < r; ++round) {
    const size_t i = order[round];
    const Rq* h_cross = round == 0 ? nullptr : &final_proof.h_cross[round - 1];
    const bool first_primary = (i == 0);
    const Rq* g0 = first_primary ? &final_proof.g0 : nullptr;
    const Rq* g_cross = i < primary_count ? &final_proof.g_cross[i] : nullptr;
    const Rq* g_diagonal = i < primary_count ? &final_proof.g_diagonal[i] : nullptr;
    std::vector<std::byte> round_message = section_5_6_round_message(
      round, h_cross, final_proof.h_diagonal[round], g0, g_cross, g_diagonal);
    std::vector<std::byte> challenge_seed = oracle.generate(round_message.data(), round_message.size());
    std::vector<Rq> challenge =
      sample_low_norm_challenges(n, 1, challenge_seed.data(), challenge_seed.size());
    ICICLE_CHECK(ntt(challenge.data(), 1, NTTDir::kForward, {}, &challenges_hat[i]));
    trs.seed4 = std::move(challenge_seed);
  }
  trs.challenges_hat = challenges_hat;

  std::vector<Rq> z(n);
  ICICLE_CHECK(ntt(final_proof.z_hat.data(), n, NTTDir::kInverse, {}, z.data()));
  bool z_small = false;
  ICICLE_CHECK(check_norm_bound(
    reinterpret_cast<const Zq*>(z.data()), z.size() * d, eNormType::L2,
    lab_inst.param.op_norm_bound * lab_inst.param.beta * std::sqrt(r), {}, &z_small));
  if (!z_small) {
    std::cout << "Section 5.6 L2 norm check for z failed\n";
    return false;
  }
  const long double final_z_norm = coefficient_l2_norm(z);
  if (!reference_msis_secure(
        lab_inst.param.kappa,
        section_5_6_tail_msis_bound(final_z_norm))) {
    std::cout << "Section 5.6 folded-z rank check failed\n";
    return false;
  }

  // A*z = sum_i c_i*t_i.  The raw t_i are the inner commitments; no B/C/D
  // outer commitment exists in this final execution.
  std::vector<Tq> zA_hat = ajtai_commitment(
    lab_inst.param.A, n, lab_inst.param.kappa, final_proof.z_hat.data(), n);
  std::vector<Tq> t_hat(final_proof.t.size());
  ICICLE_CHECK(ntt(
    final_proof.t.data(), final_proof.t.size(), NTTDir::kForward, {}, t_hat.data()));
  std::vector<Tq> ct_hat(lab_inst.param.kappa);
  ICICLE_CHECK(matmul(
    challenges_hat.data(), 1, r, t_hat.data(), r, lab_inst.param.kappa, {}, ct_hat.data()));
  if (!poly_vec_eq(zA_hat.data(), ct_hat.data(), ct_hat.size())) {
    std::cout << "Section 5.6 verification failed: A*z != sum_i c_i*t_i\n";
    return false;
  }

  std::vector<Tq> g_cross_hat(primary_count), g_diagonal_hat(primary_count);
  std::vector<Tq> h_cross_hat(r - 1), h_diagonal_hat(r);
  Tq g0_hat;
  ICICLE_CHECK(ntt(&final_proof.g0, 1, NTTDir::kForward, {}, &g0_hat));
  ICICLE_CHECK(ntt(
    final_proof.g_cross.data(), primary_count, NTTDir::kForward, {}, g_cross_hat.data()));
  ICICLE_CHECK(ntt(
    final_proof.g_diagonal.data(), primary_count, NTTDir::kForward, {}, g_diagonal_hat.data()));
  if (r > 1) {
    ICICLE_CHECK(ntt(
      final_proof.h_cross.data(), r - 1, NTTDir::kForward, {}, h_cross_hat.data()));
  }
  ICICLE_CHECK(ntt(
    final_proof.h_diagonal.data(), r, NTTDir::kForward, {}, h_diagonal_hat.data()));

  // <z,z> = g0 + sum_i (g_cross_i*c_i + g_diagonal_i*c_i^2).
  Tq z_inner_product;
  ICICLE_CHECK(matmul(
    final_proof.z_hat.data(), 1, n, final_proof.z_hat.data(), n, 1, {}, &z_inner_product));
  Tq g_rhs = g0_hat;
  for (size_t i = 0; i < primary_count; ++i) {
    Tq challenge_squared, cross_term, diagonal_term, sum;
    ICICLE_CHECK(vector_mul(&challenges_hat[i], &challenges_hat[i], 1, {}, &challenge_squared));
    ICICLE_CHECK(vector_mul(&g_cross_hat[i], &challenges_hat[i], 1, {}, &cross_term));
    ICICLE_CHECK(vector_mul(&g_diagonal_hat[i], &challenge_squared, 1, {}, &diagonal_term));
    ICICLE_CHECK(vector_add(&cross_term, &diagonal_term, 1, {}, &sum));
    ICICLE_CHECK(vector_add(&g_rhs, &sum, 1, {}, &g_rhs));
  }
  if (!poly_vec_eq(&z_inner_product, &g_rhs, 1)) {
    std::cout << "Section 5.6 verification failed: <z,z> garbage identity\n";
    return false;
  }

  // sum_i <phi_i,z>c_i = sum_round
  // (h_cross_round*c_i + h_diagonal_round*c_i^2).
  const EqualityInstance& final_constraint = lab_inst.equality_constraints[0];
  std::vector<Tq> phi_times_z(r);
  ICICLE_CHECK(matmul(
    final_constraint.phi.data(), r, n, final_proof.z_hat.data(), n, 1, {}, phi_times_z.data()));
  Tq h_lhs;
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, phi_times_z.data(), r, 1, {}, &h_lhs));
  Tq h_rhs = zero();
  for (size_t round = 0; round < r; ++round) {
    const size_t i = order[round];
    Tq challenge_squared, diagonal_term, round_term;
    ICICLE_CHECK(vector_mul(&challenges_hat[i], &challenges_hat[i], 1, {}, &challenge_squared));
    ICICLE_CHECK(vector_mul(&h_diagonal_hat[round], &challenge_squared, 1, {}, &diagonal_term));
    round_term = diagonal_term;
    if (round > 0) {
      Tq cross_term;
      ICICLE_CHECK(vector_mul(&h_cross_hat[round - 1], &challenges_hat[i], 1, {}, &cross_term));
      ICICLE_CHECK(vector_add(&round_term, &cross_term, 1, {}, &round_term));
    }
    ICICLE_CHECK(vector_add(&h_rhs, &round_term, 1, {}, &h_rhs));
  }
  if (!poly_vec_eq(&h_lhs, &h_rhs, 1)) {
    std::cout << "Section 5.6 verification failed: phi garbage identity\n";
    return false;
  }

  // The special tail is valid only for an instance whose quadratic support
  // is diagonal and confined to the primary z rows.  Verify this invariant
  // rather than trusting transition metadata.
  const Tq zero_poly = zero();
  for (size_t i = 0; i < r; ++i) {
    for (size_t j = 0; j < r; ++j) {
      if (i == j && i < primary_count) { continue; }
      if (!poly_vec_eq(&final_constraint.a[i * r + j], &zero_poly, 1)) {
        std::cout << "Section 5.6 verification failed: non-sparse quadratic instance\n";
        return false;
      }
    }
  }

  std::vector<size_t> round_for_index(r);
  for (size_t round = 0; round < r; ++round) { round_for_index[order[round]] = round; }
  Tq final_identity = final_constraint.b;
  for (size_t i = 0; i < primary_count; ++i) {
    Tq term;
    ICICLE_CHECK(vector_mul(
      &final_constraint.a[i * r + i], &g_diagonal_hat[i], 1, {}, &term));
    ICICLE_CHECK(vector_add(&final_identity, &term, 1, {}, &final_identity));
  }
  for (size_t i = 0; i < r; ++i) {
    ICICLE_CHECK(vector_add(
      &final_identity, &h_diagonal_hat[round_for_index[i]], 1, {}, &final_identity));
  }
  if (!poly_vec_eq(&final_identity, &zero_poly, 1)) {
    std::cout << "Section 5.6 verification failed: aggregated relation identity\n";
    return false;
  }
  return true;
}

// modifies the instance
// Doesn't perform any checks
// returns num_aggregation_rounds number of polynomials
void LabradorBaseVerifier::agg_const_zero_constraints()
{
  size_t r = lab_inst.param.r;
  size_t n = lab_inst.param.n;
  size_t d = Rq::d;
  size_t JL_out = lab_inst.param.JL_out;
  size_t num_aggregation_rounds = lab_inst.param.num_aggregation_rounds;
  const size_t L = lab_inst.const_zero_constraints.size();

  size_t JL_i = trs.prover_msg.JL_i;
  const std::vector<std::byte> seed1 = trs.seed1;
  const std::vector<Zq>& psi = trs.psi;
  const std::vector<Zq>& omega = trs.omega;
  // indexes into multidim arrays: psi[k][l] and omega[k][l]
  auto psi_index = [num_aggregation_rounds, L](size_t k, size_t l) {
    assert(l < L);
    assert(k < num_aggregation_rounds);
    return k * L + l;
  };
  std::vector<std::byte> jl_seed = append_u64_le(seed1.data(), seed1.size(), static_cast<uint64_t>(JL_i));

  // Shared with the prover: one accumulator plus a bounded streamed row
  // buffer replaces the two full JL_out*r*n matrices.
  DeviceVector<PolyRing> reduction_result(r * n);

  for (size_t k = 0; k < num_aggregation_rounds; k++) {
    EqualityInstance new_constraint(r, n);
    std::vector<ConstZeroInstance> temp_const(lab_inst.const_zero_constraints);

    // Compute a''_{ij} = sum_{l=0}^{L-1} psi^{(k)}(l) * a'_{ij}^{(l)}

    // For each l do:
    // const_zero_constraints[l].a[i,j] = psi[k,l]* const_zero_constraints[l].a[i,j]
    // use async_config to parallelise
    VecOpsConfig async_config = default_vec_ops_config();
    async_config.is_async = true;

    for (size_t l = 0; l < L; l++) {
      Zq psi_scalar = psi[psi_index(k, l)];

      ICICLE_CHECK(scalar_mul_vec(
        &psi_scalar, reinterpret_cast<Zq*>(temp_const[l].a.data()), r * r * d, async_config,
        reinterpret_cast<Zq*>(temp_const[l].a.data())));
    }
    ICICLE_CHECK(icicle_device_synchronize());
    // new_constraint.a[i,j] = \sum_l const_zero_constraints[l].a[i,j]
    for (size_t l = 0; l < L; l++) {
      ICICLE_CHECK(vector_add(new_constraint.a.data(), temp_const[l].a.data(), r * r, {}, new_constraint.a.data()));
    }

    // Compute varphi'_i^{(k)} = sum_{l=0}^{L-1} psi^{(k)}(l) * phi'_i^{(l)} + sum_{l=0}^{255} omega^{(k)}(l) * q_{il}

    // For each l do:
    // const_zero_constraints[l].phi[i,:] = psi[k,l]* const_zero_constraints[l].phi[i,:]
    // use async_config to parallelise
    // TODO: can async with a aggregation above- leave for later
    for (size_t l = 0; l < L; l++) {
      Zq psi_scalar = psi[psi_index(k, l)];

      ICICLE_CHECK(scalar_mul_vec(
        &psi_scalar, reinterpret_cast<Zq*>(temp_const[l].phi.data()), r * n * d, async_config,
        reinterpret_cast<Zq*>(temp_const[l].phi.data())));
    }
    ICICLE_CHECK(icicle_device_synchronize());
    // new_constraint.phi[i,:] = \sum_l const_zero_constraints[l].phi[i,:]
    for (size_t l = 0; l < L; l++) {
      ICICLE_CHECK(
        vector_add(new_constraint.phi.data(), temp_const[l].phi.data(), r * n, {}, new_constraint.phi.data()));
    }

    aggregate_jl_projection_rows_ntt(
      jl_seed.data(), jl_seed.size(), r * n, &omega[k * JL_out], JL_out, reduction_result.data());

    // Then add to new_constraint.phi
    ICICLE_CHECK(vector_add(new_constraint.phi.data(), reduction_result.data(), r * n, {}, new_constraint.phi.data()));

    new_constraint.b = trs.prover_msg.b_agg[k];

    // Add the EqualityInstance to LabradorInstance
    lab_inst.add_equality_constraint(new_constraint);
  }
  // delete the const zero constraints
  lab_inst.const_zero_constraints.clear();
  lab_inst.const_zero_constraints.shrink_to_fit();
}

// Verifies transcript messages are valid
// Also aggregates the lab_inst into the correct final constraint
bool LabradorBaseVerifier::part_verify()
{
  size_t r = lab_inst.param.r;
  size_t n = lab_inst.param.n;
  size_t d = Rq::d;
  size_t JL_out = lab_inst.param.JL_out;
  size_t num_aggregation_rounds = lab_inst.param.num_aggregation_rounds;

  const std::vector<Zq>& p = trs.prover_msg.p;
  const std::vector<Tq>& b_agg = trs.prover_msg.b_agg;
  const std::vector<Zq>& psi = trs.psi;
  const std::vector<Zq>& omega = trs.omega;

  std::vector<Rq> b_agg_unhat(b_agg.size());
  // TODO: don't need complete NTT here
  ICICLE_CHECK(ntt(b_agg.data(), b_agg.size(), NTTDir::kInverse, {}, b_agg_unhat.data()));

  // create_transcript called in constructor - so transcript is ready to be used

  // check p norm
  bool JL_check = false;
  double beta = lab_inst.param.beta;
  ICICLE_CHECK(check_norm_bound(p.data(), JL_out, eNormType::L2, uint64_t(sqrt(JL_out / 2) * beta), {}, &JL_check));
  if (!JL_check) {
    std::cout << "verify(): p-norm check failed" << std::endl;
    return false;
  }

  // b_agg have correct const term
  std::vector<Zq> test_b0(num_aggregation_rounds, Zq::zero());
  const size_t L = lab_inst.const_zero_constraints.size();

  auto psi_index = [num_aggregation_rounds, L](size_t k, size_t l) {
    assert(l < L);
    assert(k < num_aggregation_rounds);
    return k * L + l;
  };
  auto omega_index = [num_aggregation_rounds, JL_out](size_t k, size_t l) {
    assert(l < JL_out);
    assert(k < num_aggregation_rounds);
    return k * JL_out + l;
  };
  for (size_t k = 0; k < num_aggregation_rounds; k++) {
    for (size_t l = 0; l < L; l++) {
      test_b0[k] = test_b0[k] + psi[psi_index(k, l)] * lab_inst.const_zero_constraints[l].b;
    }
    for (size_t l = 0; l < JL_out; l++) {
      test_b0[k] = test_b0[k] - omega[omega_index(k, l)] * p[l];
    }

    if (test_b0[k] != b_agg_unhat[k].values[0]) {
      std::cout << "verify(): b0 test failed for " << k << std::endl;
      return false;
    }
  }

  // construct the final constraint correctly
  agg_const_zero_constraints();
  lab_inst.agg_equality_constraints(trs.alpha_hat);

  return true;
}

bool LabradorBaseVerifier::fully_verify(const LabradorBaseCaseProof& base_proof)
{
  if (part_verify()) {
    return _verify_base_proof(base_proof);
  } else {
    return false;
  }
}

bool LabradorBaseVerifier::fully_verify(const LabradorSection56Proof& final_proof)
{
  if (part_verify()) {
    return _verify_section_5_6_proof(final_proof);
  }
  return false;
}

bool LabradorVerifier::verify()
{
  if (NUM_REC == 0 || prover_msgs.size() != NUM_REC) { return false; }
  if (final_proof.uses_section_5_6 != (NUM_REC > 1)) {
    std::cout << "Unexpected LaBRADOR final-proof mode\n";
    return false;
  }
  LabradorInstance lab_inst_i = lab_inst;
  for (size_t i = 0; i < NUM_REC - 1; i++) {
    // std::cout << "Verifier::Recursion iteration = " << i << "\n";
    LabradorBaseVerifier base_verifier(lab_inst_i, prover_msgs[i], oracle);
    if (!base_verifier.part_verify()) {
      std::cout << "\tProver message verification failed\n";
      return false;
    }
    // Part verify correctly aggregates constraints

    const bool final_transition = (i + 1 == NUM_REC - 1);
    const LabradorTransitionPlan transition =
      derive_protocol_transition_plan(lab_inst_i.param, final_transition);

    EqualityInstance final_const = base_verifier.lab_inst.equality_constraints[0];
    lab_inst_i = prepare_recursion_instance(
      base_verifier.lab_inst.param,
      final_const,
      base_verifier.trs,
      transition.z_base,
      transition.mu,
      transition.nu,
      !final_transition);
    oracle = base_verifier.oracle;

    // std::cout << "\tVerifier::Recursion problem prepared\n";
    // std::cout << "\tn= " << lab_inst_i.param.n << ", r= " << lab_inst_i.param.r << "\n";
  }
  // std::cout << "Verifier::Recursion iteration = " << NUM_REC - 1 << "\n";
  if (NUM_REC > 1) {
    LabradorBaseVerifier base_verifier(
      lab_inst_i, prover_msgs[NUM_REC - 1], final_proof.section_5_6, oracle);
    if (!base_verifier.fully_verify(final_proof.section_5_6)) {
      std::cout << "\tVerifier- Section 5.6 final verification failed\n";
      return false;
    }
  } else {
    LabradorBaseVerifier base_verifier(lab_inst_i, prover_msgs[0], oracle);
    if (!base_verifier.fully_verify(final_proof.base)) {
      std::cout << "\tVerifier- Final verification failed\n";
      return false;
    }
  }
  return true;
}
