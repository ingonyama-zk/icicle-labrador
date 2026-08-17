#pragma once

#include "types.h"

#include <cstddef>
#include <vector>

/// Canonical, bit-packed wire representation of a LaBRADOR proof.
///
/// Field elements are encoded in exactly ceil(log2(q)) bits.  Coefficient-
/// domain vectors whose verifier checks a norm bound are encoded as centered
/// signed integers using the smallest width needed by the concrete proof.
/// The width is part of the canonical vector header and decoding is strict:
/// non-minimal widths, non-canonical field elements and trailing bytes are
/// rejected.
struct LabradorProofCodecStats {
  size_t header_bytes = 0;
  size_t prefix_bytes = 0;
  size_t final_response_bytes = 0;
  size_t total_bytes = 0;
};

struct LabradorDecodedProof {
  std::vector<BaseProverMessages> prover_messages;
  LabradorFinalProof final_proof;
};

/// Hard limits applied before the decoder allocates proof-controlled vectors.
/// Callers may tighten these values (useful for a constrained verifier), but
/// decode_labrador_proof() never permits a value above the generated backend
/// limits.  The decoded-byte budget is cumulative across all retained vectors;
/// temporary unpacking buffers must fit in the unused part of the same budget.
struct LabradorProofDecodeLimits {
  size_t max_encoded_bytes =
    static_cast<size_t>(icicle::labrador::backend_config::MAX_PROOF_BYTES);
  size_t max_decoded_bytes =
    static_cast<size_t>(icicle::labrador::backend_config::MAX_PROOF_BYTES);
  size_t max_polynomials =
    icicle::labrador::backend_config::MAX_RUNTIME_POLYNOMIALS;
};

std::vector<std::byte> encode_labrador_proof(
  const std::vector<PartialTranscript>& transcripts,
  const LabradorFinalProof& final_proof,
  LabradorProofCodecStats* stats = nullptr);

LabradorDecodedProof decode_labrador_proof(
  const std::vector<std::byte>& encoded,
  const LabradorProofDecodeLimits& limits = {});
