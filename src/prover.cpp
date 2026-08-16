#include "prover.h"
#include <cassert>
#include <chrono>
#include <limits>

std::pair<size_t, std::vector<Zq>> LabradorBaseProver::select_valid_jl_proj(std::byte* seed, size_t seed_len) const
{
  size_t JL_out = lab_inst.param.JL_out;
  size_t n = lab_inst.param.n;
  size_t r = lab_inst.param.r;
  size_t d = Rq::d;

  std::vector<Zq> p(JL_out);
  size_t JL_i = 0;
  constexpr size_t MAX_JL_ATTEMPTS = 1U << 20;
  while (JL_i < MAX_JL_ATTEMPTS) {
    std::vector<std::byte> jl_seed = append_u64_le(seed, seed_len, static_cast<uint64_t>(JL_i));
    // create JL projection: P*(s_1, s_2, ..., s_r)
    ICICLE_CHECK(icicle::labrador::jl_projection(
      reinterpret_cast<const Zq*>(S.data()), n * r * d, jl_seed.data(), jl_seed.size(), {}, p.data(), JL_out));
    // check norm
    bool JL_check = false;
    double beta = lab_inst.param.beta;

    // ignore ICICLE errors when elements of p are greater than sqrt(q)
    try {
      ICICLE_CHECK(check_norm_bound(p.data(), JL_out, eNormType::L2, uint64_t(sqrt(JL_out / 2) * beta), {}, &JL_check));
    } catch (const std::exception& e) {
      JL_check = false;
    }

    if (JL_check) {
      break;
    } else {
      p.assign(p.size(), Zq::from(0));
      JL_i++;
    }
  }
  if (JL_i == MAX_JL_ATTEMPTS) { throw std::runtime_error("JL retry limit exceeded"); }
  // at the end JL projection is defined by JL_i and p is the projection output
  // return these
  return std::make_pair(JL_i, p);
}

// modifies the instance
// returns num_aggregation_rounds number of polynomials
std::vector<Tq> LabradorBaseProver::agg_const_zero_constraints(
  const std::vector<Tq>& S_hat,
  const std::vector<Tq>& G_hat,
  const std::vector<Zq>& p,
  const std::vector<Zq>& psi,
  const std::vector<Zq>& omega,
  size_t JL_i,
  const std::vector<std::byte>& seed1)
{
  size_t r = lab_inst.param.r;
  size_t n = lab_inst.param.n;
  size_t d = Rq::d;
  size_t num_aggregation_rounds = lab_inst.param.num_aggregation_rounds;
  size_t JL_out = lab_inst.param.JL_out;
  const size_t L = lab_inst.const_zero_constraints.size();

  /* ───────────────── TIMING HELPERS ───────────────── */
  auto step_start = std::chrono::high_resolution_clock::now();
  // Each call resets timer
  auto log_step = [&](const char* msg) {
    if (SHOW_STEPS) {
      auto step_end = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start).count();
      std::cout << "\t" << msg << " (" << elapsed << " ms)" << std::endl;
      step_start = std::chrono::high_resolution_clock::now();
    }
  };
  /* ────────────────────────────────────────────────── */

  // indexes into multidim arrays: psi[k][l] and omega[k][l]
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

  std::vector<std::byte> jl_seed = append_u64_le(seed1.data(), seed1.size(), static_cast<uint64_t>(JL_i));

  log_step("\t Variable setup");

  std::vector<Zq> verif_test_b0(num_aggregation_rounds, Zq::zero());
  std::vector<Tq> msg3;

  // The full Q and omega*Q matrices would require 2*JL_out*r*n
  // polynomials.  The shared helper streams a bounded number of rows and
  // writes the NTT-domain sum into this accumulator.
  DeviceVector<PolyRing> reduction_result(r * n);
  log_step("\t alloc streamed JL accumulator");

  for (size_t k = 0; k < num_aggregation_rounds; k++) {
    {
      std::ostringstream oss;
      oss << "\t agg_const_zero_const: iter = " << k << " ";
      log_step(oss.str().c_str());
    }
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
    log_step("\t\t psi*a");
    // new_constraint.a[i,j] = \sum_l const_zero_constraints[l].a[i,j]
    for (size_t l = 0; l < L; l++) {
      ICICLE_CHECK(vector_add(new_constraint.a.data(), temp_const[l].a.data(), r * r, {}, new_constraint.a.data()));
    }
    log_step("\t\t sum(a)");

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
    log_step("\t\t psi*phi");
    // new_constraint.phi[i,:] = \sum_l const_zero_constraints[l].phi[i,:]
    for (size_t l = 0; l < L; l++) {
      ICICLE_CHECK(
        vector_add(new_constraint.phi.data(), temp_const[l].phi.data(), r * n, {}, new_constraint.phi.data()));
    }
    log_step("\t\t sum(phi)");

    aggregate_jl_projection_rows_ntt(
      jl_seed.data(), jl_seed.size(), r * n, &omega[k * JL_out], JL_out, reduction_result.data());
    log_step("\t\t streamed sum(omega*Q) + Rq/Tq conversion");

    // Then add to new_constraint.phi
    ICICLE_CHECK(vector_add(new_constraint.phi.data(), reduction_result.data(), r * n, {}, new_constraint.phi.data()));
    log_step("\t\t add to phi");

    // Compute B^{(k)} = sum_{ij} a''_{ij}^{(k)}  * g_{ij} + sum_i <phi'_i^{(k)}, s_i>
    Tq G_A_inner_prod, phi_S_inner_prod;
    // G_A_inner_prod = <G, a>
    ICICLE_CHECK(matmul(G_hat.data(), 1, r * r, new_constraint.a.data(), r * r, 1, {}, &G_A_inner_prod));
    log_step("\t\t <G, a>");
    // phi_S_inner_prod = <S, phi>
    ICICLE_CHECK(matmul(S_hat.data(), 1, r * n, new_constraint.phi.data(), r * n, 1, {}, &phi_S_inner_prod));
    log_step("\t\t <S, phi>");
    // b = -(<G, a> + <S, phi>)
    ICICLE_CHECK(vector_add(G_A_inner_prod.values, phi_S_inner_prod.values, d, {}, new_constraint.b.values));
    log_step("\t\t b = (<G, a> + <S, phi>)");
    Zq minus_1 = Zq::from(1).neg();
    ICICLE_CHECK(scalar_mul_vec(&minus_1, new_constraint.b.values, d, {}, new_constraint.b.values));
    log_step("\t\t b = -b");

    if (CONSISTENCY_CHECKS) {
      // Following should work if our B^{(k)} evaluation is correct above
      if (!witness_legit_eq(new_constraint, S)) { std::cout << "Constraint " << k << " failed\n"; }

      verif_test_b0[k] = Zq::zero();
      // Verifier performs these checks
      for (size_t l = 0; l < L; l++) {
        verif_test_b0[k] = verif_test_b0[k] + psi[psi_index(k, l)] * lab_inst.const_zero_constraints[l].b;
      }
      for (size_t l = 0; l < JL_out; l++) {
        verif_test_b0[k] = verif_test_b0[k] - omega[omega_index(k, l)] * p[l];
      }

      Rq b_rq;
      ICICLE_CHECK(ntt(&new_constraint.b, 1, NTTDir::kInverse, {}, &b_rq));

      if (!witness_legit_const_zero({r, n, new_constraint.a, new_constraint.phi, verif_test_b0[k]}, S)) {
        std::cout << "\tVerif test constraint " << k << " failed\n";
      }
    }
    // Add the EqualityInstance to LabradorInstance
    lab_inst.add_equality_constraint(new_constraint);
    log_step("\t\t add constraint");
    // Send B^(k) to the Verifier
    msg3.push_back(new_constraint.b);
    log_step("\t\t push back in msg3");
  }

  // delete the const zero constraints
  lab_inst.const_zero_constraints.clear();
  lab_inst.const_zero_constraints.shrink_to_fit();

  return msg3;
}

// This destroys the lab_inst in LabradorBaseProver
std::pair<LabradorBaseCaseProof, PartialTranscript> LabradorBaseProver::base_case_prover()
{
  // Step 1: Pack the Witnesses into a Matrix S
  const size_t r = lab_inst.param.r; // Number of witness vectors
  const size_t n = lab_inst.param.n; // Dimension of witness vectors
  constexpr size_t d = Rq::d;

  if (SHOW_STEPS) { std::cout << "Running base_case_prover..." << std::endl; }
  /* ───────────────── TIMING HELPERS ───────────────── */
  auto step_start = std::chrono::high_resolution_clock::now();
  auto log_step = [&](const char* msg) {
    if (SHOW_STEPS) {
      auto step_end = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(step_end - step_start).count();
      std::cout << "\t" << msg << " (" << elapsed << " ms)" << std::endl;
      step_start = std::chrono::high_resolution_clock::now();
    }
  };
  /* ────────────────────────────────────────────────── */

  PartialTranscript trs;
  log_step("Step 1 completed: Initialized variables");

  // Step 2: Convert S to the NTT Domain
  std::vector<Tq> S_hat(r * n);
  // Perform negacyclic NTT on the witness S
  ICICLE_CHECK(ntt(S.data(), r * n, NTTDir::kForward, {}, S_hat.data()));
  log_step("Step 2 completed: Rq/Tq representation conversion");

  // Step 3: S@A = T
  size_t kappa = lab_inst.param.kappa;
  const std::vector<Tq>& A = lab_inst.param.A; // n × kappa matrix

  // Compute T_hat = S_hat @ A
  std::vector<Tq> T_hat = ajtai_commitment(A, n, kappa, S_hat.data(), r * n);
  log_step("Step 3 completed: Ajtai commitment T_hat");

  // Step 4: Convert T_hat to Rq
  std::vector<Rq> T(r * kappa);
  // Perform negacyclic INTT
  ICICLE_CHECK(ntt(T_hat.data(), r * kappa, NTTDir::kInverse, {}, T.data()));
  log_step("Step 4 completed: Tq/Rq representation conversion of T_hat");

  // Step 5: Decompose T to T_tilde
  size_t base1 = lab_inst.param.base1;
  size_t l1 = lab_inst.param.digits1;
  std::vector<Rq> T_tilde = fixed_length_decompose(T, base1, l1);
  log_step("Step 5 completed: Decomposed T to T_tilde");

  // Step 6: Compute g
  std::vector<Tq> S_hat_transposed(n * r);
  ICICLE_CHECK(matrix_transpose<Tq>(S_hat.data(), r, n, {}, S_hat_transposed.data()));

  std::vector<Tq> G_hat(r * r);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, G_hat.data()));

  std::vector<Tq> g_hat = extract_symm_part(G_hat.data(), r);
  size_t r_choose_2 = (r * (r + 1)) / 2;
  std::vector<Rq> g(r_choose_2);

  ICICLE_CHECK(ntt(g_hat.data(), r_choose_2, NTTDir::kInverse, {}, g.data()));
  log_step("Step 6 completed: Computed g");

  // Step 7: Decompose g to g_tilde
  size_t base2 = lab_inst.param.base2;
  size_t l2 = lab_inst.param.digits2;
  std::vector<Rq> g_tilde = fixed_length_decompose(g, base2, l2);
  log_step("Step 7 completed: Decomposed g to g_tilde");

  // Step 8: u1 = B@T_tilde + C@g_tilde
  size_t kappa1 = lab_inst.param.kappa1;
  const std::vector<Tq>& B = lab_inst.param.B; // (t_len × kappa1)
  const std::vector<Tq>& C = lab_inst.param.C; // (g_len × kappa1)

  // compute NTTs for T_tilde, g_tilde
  std::vector<Tq> T_tilde_hat(T_tilde.size()), g_tilde_hat(g_tilde.size());
  log_step("\t memory alloc");
  ICICLE_CHECK(ntt(T_tilde.data(), T_tilde.size(), NTTDir::kForward, {}, T_tilde_hat.data()));
  log_step("\t T_tilde representation conversion completed");
  ICICLE_CHECK(ntt(g_tilde.data(), g_tilde.size(), NTTDir::kForward, {}, g_tilde_hat.data()));
  log_step("\t g_tilde representation conversion completed");
  // v1 = B @ T_tilde
  std::vector<Tq> v1 = ajtai_commitment(B, T_tilde_hat.size(), kappa1, T_tilde_hat.data(), T_tilde_hat.size());
  log_step("\t Ajtai commit to T_tilde");
  // v2 = C @ g_tilde
  std::vector<Tq> v2 = ajtai_commitment(C, g_tilde_hat.size(), kappa1, g_tilde_hat.data(), g_tilde_hat.size());
  log_step("\t Ajtai commit to g_tilde");

  std::vector<Tq> u1(kappa1);
  vector_add(v1.data(), v2.data(), kappa1, {}, u1.data());
  log_step("Step 8 completed: Computed u1");

  // Step 9: Derive seed1 from a canonical, domain-separated u1 message.
  const std::vector<std::byte> u1_message =
    canonical_polynomial_transcript_message("LaBRADOR-base-u1-v2", u1.data(), u1.size());
  std::vector<std::byte> seed1 = oracle.generate(u1_message.data(), u1_message.size());
  // add u1 to the trs
  trs.prover_msg.u1 = u1;
  trs.seed1 = seed1;
  log_step("Step 9 completed: Generated seed1");

  // Step 10: Select a JL projection
  size_t JL_out = lab_inst.param.JL_out;
  auto [JL_i, p] = select_valid_jl_proj(seed1.data(), seed1.size());
  log_step("Step 10 completed: Selected JL projection");

  trs.prover_msg.JL_i = JL_i;
  trs.prover_msg.p = p;

  // Step 11: Canonically serialize (JL_i,p) for seed2.
  const std::vector<std::byte> jl_buf =
    canonical_jl_transcript_message("LaBRADOR-base-jl-v2", JL_i, p);

  std::vector<std::byte> seed2 = oracle.generate(jl_buf.data(), jl_buf.size());
  trs.seed2 = seed2;
  log_step("Step 11 completed: Generated seed2");

  // Step 12: Let L be the number of ConstZeroInstance constraints in LabradorInstance.
  // For 0 ≤ k < ceil(128/log(q)), sample the following random vectors:
  const size_t L = lab_inst.const_zero_constraints.size();
  const size_t num_aggregation_rounds = lab_inst.param.num_aggregation_rounds;

  std::vector<Zq> psi(num_aggregation_rounds * L), omega(num_aggregation_rounds * JL_out);

  // sample psi
  // psi seed = seed2 || 0x01
  std::vector<std::byte> psi_seed(seed2);
  psi_seed.push_back(std::byte('1'));
  if (psi.size() > 0) {
    ICICLE_CHECK(random_sampling(psi.size(), false, psi_seed.data(), psi_seed.size(), {}, psi.data()));
  }
  // Sample omega
  // omega seed = seed2 || 0x02
  std::vector<std::byte> omega_seed(seed2);
  omega_seed.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(omega.size(), false, omega_seed.data(), omega_seed.size(), {}, omega.data()));

  trs.psi = psi;
  trs.omega = omega;
  log_step("Step 12 completed: Sampled psi and omega");

  // Step 13: Aggregate ConstZeroInstance constraints

  std::vector<Tq> msg3 = agg_const_zero_constraints(S_hat, G_hat, p, psi, omega, JL_i, seed1);
  log_step("Step 13 completed: Aggregated ConstZeroInstance constraints");

  if (CONSISTENCY_CHECKS) {
    std::cout << "\tTesting witness validity...";
    assert(lab_witness_legit(lab_inst, S));
    std::cout << "VALID\n";
  }

  // Step 14: Derive seed3 from canonical aggregation polynomials.
  const std::vector<std::byte> msg3_message =
    canonical_polynomial_transcript_message(
      "LaBRADOR-base-aggregation-v2", msg3.data(), msg3.size());
  std::vector<std::byte> seed3 = oracle.generate(msg3_message.data(), msg3_message.size());
  log_step("Step 14 completed: Generated seed3");

  trs.prover_msg.b_agg = msg3;
  trs.seed3 = seed3;

  // Step 15: Sample random polynomial vectors α using seed3
  // Let K be the number of EqualityInstances in the LabradorInstance
  const size_t K = lab_inst.equality_constraints.size();
  // Will be true since we add constraints while aggregating constZeroInstances
  assert(K > 0);
  std::vector<Tq> alpha_hat(K);
  std::vector<std::byte> alpha_seed(seed3);
  alpha_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling(K, false, alpha_seed.data(), alpha_seed.size(), {}, alpha_hat.data()));

  trs.alpha_hat = alpha_hat;
  log_step("Step 15 completed: Sampled alpha_hat");

  // Step 16: Aggregate equality constraints
  lab_inst.agg_equality_constraints(alpha_hat);
  log_step("Step 16 completed: Aggregated equality constraints");
  if (CONSISTENCY_CHECKS) {
    std::cout << "\tTesting witness validity...";
    assert(lab_witness_legit(lab_inst, S));
    std::cout << "VALID\n";
  }

  // Step 17: For 0 ≤ i ≤ j < r, the Prover computes the matrix multiplication between matrix
  // Phi = (φ'_0|φ'_1|···|φ'_{r-1})^T ∈ R_q^{r×n} and S ∈ R_q^{r×n} defined earlier.
  // Let H ∈ R_q^{r×r}, such that H = 2^{-1}(Phi @ S^T + (Phi @ S^T)^T)

  // Matrix Phi
  const Tq* phi_final = lab_inst.equality_constraints[0].phi.data();

  // Compute Phi @ S^T using the transposed S_hat
  std::vector<Tq> Phi_times_St_hat(r * r);
  ICICLE_CHECK(matmul(phi_final, r, n, S_hat_transposed.data(), n, r, {}, Phi_times_St_hat.data()));

  // Convert back to Rq domain
  std::vector<Rq> H(r * r), Phi_times_St_transposed(r * r);
  // H = Phi @ S^t
  ICICLE_CHECK(ntt(Phi_times_St_hat.data(), r * r, NTTDir::kInverse, {}, H.data()));
  // transpose matrix
  ICICLE_CHECK(matrix_transpose<Tq>(H.data(), r, r, {}, Phi_times_St_transposed.data()));

  // Compute H = 2^{-1}(LS + (LS)^T)
  Zq two_inv = Zq::from(2).inverse(); // 2^{-1} in Z_q

  // H = H + Phi_times_St_transposed = Phi@S^t + Phi_times_St_transposed
  ICICLE_CHECK(vector_add(H.data(), Phi_times_St_transposed.data(), r * r, {}, H.data()));
  // H = 1/2 * H
  ICICLE_CHECK(
    scalar_mul_vec(&two_inv, reinterpret_cast<Zq*>(H.data()), r * r * d, {}, reinterpret_cast<Zq*>(H.data())));

  std::vector<Rq> h = extract_symm_part(H.data(), r);
  log_step("Step 17 completed: Computed h vector");

  // Step 18: Decompose h
  size_t base3 = lab_inst.param.base3;
  size_t l3 = lab_inst.param.digits3;

  std::vector<Rq> h_tilde = fixed_length_decompose(h, base3, l3);
  std::vector<Tq> h_tilde_hat(h_tilde.size());
  ICICLE_CHECK(ntt(h_tilde.data(), h_tilde.size(), NTTDir::kForward, {}, h_tilde_hat.data()));
  log_step("Step 18 completed: Decomposed h to H_tilde");

  // Step 19: Commit to h_tilde
  size_t kappa2 = lab_inst.param.kappa2;
  const std::vector<Tq>& D = lab_inst.param.D; // (h_len × kappa2)

  // u2 = D @ h_tilde
  std::vector<Tq> u2 = ajtai_commitment(D, h_tilde_hat.size(), kappa2, h_tilde_hat.data(), h_tilde_hat.size());
  log_step("Step 19 completed: Computed u2 commitment");

  // Step 20:
  // add u2 to the trs
  trs.prover_msg.u2 = u2;

  // Derive seed4 from a canonical, domain-separated u2 message.
  const std::vector<std::byte> u2_message =
    canonical_polynomial_transcript_message("LaBRADOR-base-u2-v2", u2.data(), u2.size());
  std::vector<std::byte> seed4 = oracle.generate(u2_message.data(), u2_message.size());

  trs.seed4 = seed4;
  log_step("Step 20 completed: Generated seed4");

  // Step 21: Sample low-operator-norm challenges
  std::vector<Rq> challenge = sample_low_norm_challenges(n, r, seed4.data(), seed4.size());

  std::vector<Tq> challenges_hat(r);
  ICICLE_CHECK(ntt(challenge.data(), challenge.size(), NTTDir::kForward, {}, challenges_hat.data()));
  trs.challenges_hat = challenges_hat;
  log_step("Step 21 completed: Sampled challenges");

  // Step 22: Compute z_hat[:] = \sum_i c_i * S[i,:] = [c1 c2 ... cr] @ S
  std::vector<Tq> z_hat(n);
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, S_hat.data(), r, n, {}, z_hat.data()));
  log_step("Step 22 completed: Computed z_hat and created final proof");

  if (CONSISTENCY_CHECKS) {
    std::vector<Tq> ct_hat(kappa);
    ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, T_hat.data(), r, kappa, {}, ct_hat.data()));
    std::vector<Tq> zA_hat = ajtai_commitment(A, n, kappa, z_hat.data(), z_hat.size());

    // zA_hat == \sum_i c_i t_i
    bool succ = true;
    if (!poly_vec_eq(zA_hat.data(), ct_hat.data(), kappa)) {
      succ = false;
      std::cout << "\tbase_prover zA = ct failed\n";
    }
    if (succ) { std::cout << "\tbase_prover zA = ct passed\n"; }
  }

  LabradorBaseCaseProof final_proof{z_hat, T_tilde, g_tilde, h_tilde};
  log_step("base_case_prover completed!");

  return std::make_pair(final_proof, trs);
}

std::pair<LabradorSection56Proof, PartialTranscript> LabradorBaseProver::section_5_6_prover()
{
  const size_t r = lab_inst.param.r;
  const size_t n = lab_inst.param.n;
  const size_t primary_count = lab_inst.param.final_primary_count;
  const size_t d = Rq::d;
  if (!lab_inst.param.section_5_6_final || primary_count == 0 || primary_count > r) {
    throw std::invalid_argument("Section 5.6 prover requires a final sparse instance");
  }

  if (SHOW_STEPS) { std::cout << "Running Section 5.6 final prover..." << std::endl; }
  PartialTranscript trs;

  // The inner commitments are sent directly.  They replace u1 as the first
  // witness-dependent Fiat--Shamir message and are never decomposed.
  std::vector<Tq> S_hat(r * n);
  ICICLE_CHECK(ntt(S.data(), S.size(), NTTDir::kForward, {}, S_hat.data()));
  const std::vector<Tq>& A = lab_inst.param.A;
  const size_t kappa = lab_inst.param.kappa;
  std::vector<Tq> T_hat = ajtai_commitment(A, n, kappa, S_hat.data(), S_hat.size());
  std::vector<Rq> T(T_hat.size());
  ICICLE_CHECK(ntt(T_hat.data(), T_hat.size(), NTTDir::kInverse, {}, T.data()));

  std::vector<std::byte> first_message = section_5_6_first_message(T);
  trs.seed1 = oracle.generate(first_message.data(), first_message.size());

  auto [JL_i, p] = select_valid_jl_proj(trs.seed1.data(), trs.seed1.size());
  trs.prover_msg.JL_i = JL_i;
  trs.prover_msg.p = p;

  std::vector<std::byte> jl_buf = canonical_jl_transcript_message(
    "LaBRADOR-Section-5.6-jl-v2", JL_i, p);
  trs.seed2 = oracle.generate(jl_buf.data(), jl_buf.size());

  const size_t L = lab_inst.const_zero_constraints.size();
  const size_t aggregation_rounds = lab_inst.param.num_aggregation_rounds;
  const size_t JL_out = lab_inst.param.JL_out;
  std::vector<Zq> psi(aggregation_rounds * L), omega(aggregation_rounds * JL_out);
  std::vector<std::byte> psi_seed(trs.seed2);
  psi_seed.push_back(std::byte('1'));
  if (!psi.empty()) {
    ICICLE_CHECK(random_sampling(psi.size(), false, psi_seed.data(), psi_seed.size(), {}, psi.data()));
  }
  std::vector<std::byte> omega_seed(trs.seed2);
  omega_seed.push_back(std::byte('2'));
  ICICLE_CHECK(random_sampling(omega.size(), false, omega_seed.data(), omega_seed.size(), {}, omega.data()));
  trs.psi = psi;
  trs.omega = omega;

  std::vector<Tq> S_hat_transposed(n * r);
  ICICLE_CHECK(matrix_transpose<Tq>(S_hat.data(), r, n, {}, S_hat_transposed.data()));
  std::vector<Tq> G_hat(r * r);
  ICICLE_CHECK(matmul(S_hat.data(), r, n, S_hat_transposed.data(), n, r, {}, G_hat.data()));

  std::vector<Tq> b_agg =
    agg_const_zero_constraints(S_hat, G_hat, p, psi, omega, JL_i, trs.seed1);
  trs.prover_msg.b_agg = b_agg;
  const std::vector<std::byte> aggregation_message =
    canonical_polynomial_transcript_message(
      "LaBRADOR-Section-5.6-aggregation-v2", b_agg.data(), b_agg.size());
  trs.seed3 = oracle.generate(aggregation_message.data(), aggregation_message.size());

  const size_t K = lab_inst.equality_constraints.size();
  if (K == 0) { throw std::runtime_error("Section 5.6 final instance has no equality constraints"); }
  trs.alpha_hat.resize(K);
  std::vector<std::byte> alpha_seed(trs.seed3);
  alpha_seed.push_back(std::byte('1'));
  ICICLE_CHECK(random_sampling(K, false, alpha_seed.data(), alpha_seed.size(), {}, trs.alpha_hat.data()));
  lab_inst.agg_equality_constraints(trs.alpha_hat);

  const Tq* phi = lab_inst.equality_constraints[0].phi.data();
  std::vector<Tq> phi_times_s_hat(r * r);
  ICICLE_CHECK(matmul(phi, r, n, S_hat_transposed.data(), n, r, {}, phi_times_s_hat.data()));

  LabradorSection56Proof proof;
  proof.primary_count = primary_count;
  proof.t = std::move(T);
  proof.g_cross.reserve(primary_count);
  proof.g_diagonal.reserve(primary_count);
  proof.h_cross.reserve(r - 1);
  proof.h_diagonal.reserve(r);

  const std::vector<size_t> order = section_5_6_challenge_order(r, primary_count);
  std::vector<Tq> challenges_hat(r, zero());
  trs.challenges_hat.assign(r, zero());

  // Once all auxiliary challenges are known, their folded norm is g0.  It is
  // sent immediately before the first primary challenge.
  bool g0_ready = false;
  size_t primary_seen = 0;
  for (size_t round = 0; round < order.size(); ++round) {
    const size_t i = order[round];

    Rq h_diagonal;
    ICICLE_CHECK(ntt(&phi_times_s_hat[i * r + i], 1, NTTDir::kInverse, {}, &h_diagonal));
    proof.h_diagonal.push_back(h_diagonal);

    Rq h_cross;
    const Rq* h_cross_message = nullptr;
    if (round > 0) {
      Tq h_cross_hat = zero();
      for (size_t previous_round = 0; previous_round < round; ++previous_round) {
        const size_t j = order[previous_round];
        Tq symmetric_term, weighted_term;
        ICICLE_CHECK(vector_add(
          &phi_times_s_hat[j * r + i], &phi_times_s_hat[i * r + j], 1, {}, &symmetric_term));
        ICICLE_CHECK(vector_mul(&symmetric_term, &challenges_hat[j], 1, {}, &weighted_term));
        ICICLE_CHECK(vector_add(&h_cross_hat, &weighted_term, 1, {}, &h_cross_hat));
      }
      ICICLE_CHECK(ntt(&h_cross_hat, 1, NTTDir::kInverse, {}, &h_cross));
      proof.h_cross.push_back(h_cross);
      h_cross_message = &proof.h_cross.back();
    }

    const Rq* g0_message = nullptr;
    const Rq* g_cross_message = nullptr;
    const Rq* g_diagonal_message = nullptr;
    if (i < primary_count) {
      if (!g0_ready) {
        std::vector<Tq> auxiliary_fold(n, zero());
        const size_t auxiliary_count = r - primary_count;
        if (auxiliary_count > 0) {
          ICICLE_CHECK(matmul(
            &challenges_hat[primary_count], 1, auxiliary_count, &S_hat[primary_count * n], auxiliary_count, n, {},
            auxiliary_fold.data()));
        }
        Tq g0_hat;
        ICICLE_CHECK(matmul(auxiliary_fold.data(), 1, n, auxiliary_fold.data(), n, 1, {}, &g0_hat));
        ICICLE_CHECK(ntt(&g0_hat, 1, NTTDir::kInverse, {}, &proof.g0));
        g0_ready = true;
        g0_message = &proof.g0;
      }

      Tq g_cross_hat = zero();
      for (size_t previous_round = 0; previous_round < round; ++previous_round) {
        const size_t j = order[previous_round];
        Tq symmetric_term, weighted_term;
        ICICLE_CHECK(vector_add(&G_hat[i * r + j], &G_hat[j * r + i], 1, {}, &symmetric_term));
        ICICLE_CHECK(vector_mul(&symmetric_term, &challenges_hat[j], 1, {}, &weighted_term));
        ICICLE_CHECK(vector_add(&g_cross_hat, &weighted_term, 1, {}, &g_cross_hat));
      }
      Rq g_cross;
      ICICLE_CHECK(ntt(&g_cross_hat, 1, NTTDir::kInverse, {}, &g_cross));
      proof.g_cross.push_back(g_cross);
      g_cross_message = &proof.g_cross.back();

      Rq g_diagonal;
      ICICLE_CHECK(ntt(&G_hat[i * r + i], 1, NTTDir::kInverse, {}, &g_diagonal));
      proof.g_diagonal.push_back(g_diagonal);
      g_diagonal_message = &proof.g_diagonal.back();
      ++primary_seen;
    }

    std::vector<std::byte> round_message = section_5_6_round_message(
      round, h_cross_message, proof.h_diagonal.back(), g0_message, g_cross_message, g_diagonal_message);
    std::vector<std::byte> challenge_seed = oracle.generate(round_message.data(), round_message.size());
    std::vector<Rq> challenge =
      sample_low_norm_challenges(n, 1, challenge_seed.data(), challenge_seed.size());
    ICICLE_CHECK(ntt(challenge.data(), 1, NTTDir::kForward, {}, &challenges_hat[i]));
    trs.challenges_hat[i] = challenges_hat[i];
    trs.seed4 = std::move(challenge_seed);
  }
  if (!g0_ready || primary_seen != primary_count) {
    throw std::runtime_error("Section 5.6 final challenge schedule is incomplete");
  }

  proof.z_hat.resize(n);
  ICICLE_CHECK(matmul(challenges_hat.data(), 1, r, S_hat.data(), r, n, {}, proof.z_hat.data()));
  if (SHOW_STEPS) {
    std::cout << "\tSection 5.6 final response: z=" << proof.z_polynomial_count()
              << ", t=" << proof.t_polynomial_count() << ", g=" << proof.g_polynomial_count()
              << ", h=" << proof.h_polynomial_count() << " polynomials\n";
  }
  return {proof, trs};
}

std::vector<Rq> LabradorProver::prepare_recursion_witness(
  const LabradorParam& prev_param,
  const LabradorBaseCaseProof& pf,
  uint32_t base0,
  size_t mu,
  size_t nu,
  bool decompose_z,
  size_t n_prime_override)
{
  // Step 1: Convert z_hat back to polynomial domain
  size_t n = prev_param.n;
  size_t r = prev_param.r;

  std::vector<Rq> z(n);
  ICICLE_CHECK(ntt(pf.z_hat.data(), pf.z_hat.size(), NTTDir::kInverse, {}, z.data()));

  // All ordinary transitions decompose z into two base-b limbs.  The
  // penultimate transition keeps z intact for the optimized Section 5.6 tail.
  std::vector<Rq> z_tilde;
  if (decompose_z) {
    z_tilde = fixed_length_decompose(z, base0, 2);

    std::vector<Rq> temp(n);
    ICICLE_CHECK(recompose(z_tilde.data(), z_tilde.size(), base0, {}, temp.data(), temp.size()));
    if (!poly_vec_eq(z.data(), temp.data(), n)) {
      throw std::runtime_error("Parameter Choice Error: z could not be recomposed from z_tilde in "
                               "prepare_recursion_witness. Consider changing base0 parameter.");
    } else if (SHOW_STEPS) {
      std::cout << "\tprepare_recursion_witness: z recomposition passes.\n";
    }
  }
  // Step 3:
  // z0 = z_tilde[:n]
  // z1 = z_tilde[n:2*n]

  RecursionPreparer preparer{
    prev_param, mu, nu, base0, decompose_z, n_prime_override};

  std::vector<Rq> s_prime(preparer.r_prime * preparer.n_prime, zero());

  // copy z0 = z_tilde[0 : n] →  s_prime[0 : n]
  ICICLE_CHECK(preparer.copy_like_z0(s_prime.data(), decompose_z ? z_tilde.data() : z.data()));

  if (decompose_z) {
    // copy z1 = z_tilde[n : 2n] →  s_prime[nu * n_prime : nu * n_prime + n]
    ICICLE_CHECK(preparer.copy_like_z1(s_prime.data(), &z_tilde[n]));
  }

  // copy t  →  s_prime[2*nu*n_prime : 2*nu*n_prime + |t|]
  ICICLE_CHECK(preparer.copy_like_t(s_prime.data(), pf.t.data()));

  // t, g and h form one contiguous v vector split into mu chunks.
  ICICLE_CHECK(preparer.copy_like_g(s_prime.data(), pf.g.data()));

  ICICLE_CHECK(preparer.copy_like_h(s_prime.data(), pf.h.data()));

  return s_prime;
}

std::pair<std::vector<PartialTranscript>, LabradorFinalProof> LabradorProver::prove()
{
  if (NUM_REC == 0) { throw std::invalid_argument("NUM_REC must be at least one"); }
  std::vector<PartialTranscript> trs;
  PartialTranscript part_trs;
  LabradorBaseCaseProof base_proof;
  size_t final_primary_count = 0;
  LabradorInstance lab_inst_i = lab_inst;
  std::vector<Rq> S_i = S;
  for (size_t i = 0; i < NUM_REC - 1; i++) {
    if (SHOW_STEPS) { std::cout << "Prover::Recursion iteration = " << i << "\n"; }

    LabradorBaseProver base_prover(lab_inst_i, S_i, oracle);
    std::tie(base_proof, part_trs) = base_prover.base_case_prover();
    trs.push_back(part_trs);

    const bool final_transition = (i + 1 == NUM_REC - 1);
    const LabradorTransitionPlan transition =
      derive_protocol_transition_plan(lab_inst_i.param, final_transition);
    const uint32_t base0 = transition.z_base;
    const size_t mu = transition.mu;
    const size_t nu = transition.nu;

    S_i = prepare_recursion_witness(
      lab_inst_i.param, base_proof, base0, mu, nu, !final_transition, transition.n_next);
    const long double next_norm = coefficient_l2_norm(S_i);
    if (!(next_norm <= static_cast<long double>(transition.beta_next))) {
      throw std::runtime_error(
        "LaBRADOR heuristic beta' was exceeded; restart with a new transcript or a larger audited bound");
    }
    EqualityInstance final_const = base_prover.lab_inst.equality_constraints[0];
    lab_inst_i = prepare_recursion_instance(
      base_prover.lab_inst.param, final_const, part_trs, base0, mu, nu, !final_transition);
    if (final_transition) { final_primary_count = nu; }

    if (!lab_witness_legit(lab_inst_i, S_i)) {
      throw std::runtime_error("derived recursion witness does not satisfy the derived relation");
    }

    oracle = base_prover.oracle;

    if (SHOW_STEPS) {
      std::cout << "\tRecursion problem prepared\n";
      std::cout << "\tn= " << lab_inst_i.param.n << ", r= " << lab_inst_i.param.r
                << ", beta= " << lab_inst_i.param.beta << ", actual norm= "
                << static_cast<double>(next_norm) << "\n";
      std::cout << "\tIntermediate response folded (not transmitted): " << base_proof.size() << " B\n";
    }
  }
  if (SHOW_STEPS) { std::cout << "Prover::Recursion iteration = " << NUM_REC - 1 << "\n"; }
  LabradorBaseProver base_prover(lab_inst_i, S_i, oracle);
  LabradorFinalProof final_proof;
  if (NUM_REC > 1) {
    if (!lab_inst_i.param.section_5_6_final ||
        lab_inst_i.param.final_primary_count != final_primary_count) {
      throw std::runtime_error("optimized final instance metadata mismatch");
    }
    LabradorSection56Proof section_5_6_proof;
    std::tie(section_5_6_proof, part_trs) = base_prover.section_5_6_prover();
    final_proof = LabradorFinalProof::from_section_5_6(section_5_6_proof);
  } else {
    std::tie(base_proof, part_trs) = base_prover.base_case_prover();
    final_proof = LabradorFinalProof::from_base(base_proof);
  }
  trs.push_back(part_trs);
  if (SHOW_STEPS) { std::cout << "\tProof size= " << final_proof.size() << " B\n"; }

  return std::make_pair(trs, final_proof);
}
