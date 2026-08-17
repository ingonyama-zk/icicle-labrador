#include "proof_codec.h"

#include "lnplabrador_backend_params.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

constexpr std::array<std::byte, 8> MAGIC{
  std::byte{'L'}, std::byte{'A'}, std::byte{'B'}, std::byte{'Q'},
  std::byte{'4'}, std::byte{'0'}, std::byte{'P'}, std::byte{'2'}};
constexpr uint32_t VERSION = 2;
constexpr uint8_t FULL_WIDTH = Zq::NBITS;
constexpr uint64_t MODULUS = icicle::labrador::backend_config::RING_MODULUS;
constexpr uint64_t MODULUS_HALF = icicle::labrador::backend_config::RING_MODULUS_HALF;
constexpr uint64_t MAX_POLYNOMIALS = icicle::labrador::backend_config::MAX_RUNTIME_POLYNOMIALS;
constexpr uint64_t MAX_WIRE_BYTES = icicle::labrador::backend_config::MAX_PROOF_BYTES;

static_assert(FULL_WIDTH > 0 && FULL_WIDTH < 64, "proof codec requires a sub-64-bit modulus");
static_assert(Rq::d == icicle::labrador::backend_config::RING_DEGREE, "codec/backend degree mismatch");

size_t checked_add(size_t a, size_t b, const char* label)
{
  if (b > std::numeric_limits<size_t>::max() - a) { throw std::overflow_error(label); }
  return a + b;
}

size_t checked_mul(size_t a, size_t b, const char* label)
{
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) { throw std::overflow_error(label); }
  return a * b;
}

uint64_t canonical_value(const Zq& value)
{
  const uint64_t result = uint64_t(value.limbs_storage.limbs[0]) |
                          (uint64_t(value.limbs_storage.limbs[1]) << 32U);
  if (result >= MODULUS) { throw std::invalid_argument("non-canonical Zq value in proof"); }
  return result;
}

int64_t centered_value(const Zq& value)
{
  const uint64_t canonical = canonical_value(value);
  return canonical > MODULUS_HALF ? static_cast<int64_t>(canonical - MODULUS) : static_cast<int64_t>(canonical);
}

Zq from_canonical(uint64_t value)
{
  if (value >= MODULUS) { throw std::invalid_argument("encoded field element is not canonical"); }
  return Zq::from_u64(value);
}

Zq from_centered(int64_t value)
{
  if (value < -static_cast<int64_t>(MODULUS_HALF) || value > static_cast<int64_t>(MODULUS_HALF)) {
    throw std::invalid_argument("encoded centered coefficient is outside Zq");
  }
  const uint64_t canonical = value < 0 ? MODULUS - static_cast<uint64_t>(-value) : static_cast<uint64_t>(value);
  return from_canonical(canonical);
}

uint8_t signed_width_for(int64_t value)
{
  for (uint8_t width = 1; width <= FULL_WIDTH; ++width) {
    const int64_t low = -(int64_t{1} << (width - 1));
    const int64_t high = (int64_t{1} << (width - 1)) - 1;
    if (value >= low && value <= high) { return width; }
  }
  throw std::invalid_argument("centered coefficient does not fit backend modulus width");
}

class Writer {
public:
  std::vector<std::byte> bytes;

  void u8(uint8_t value)
  {
    ensure(1, "proof header exceeds the backend artifact limit");
    bytes.push_back(std::byte(value));
  }

  void u32(uint32_t value)
  {
    for (unsigned i = 0; i < 4; ++i) { u8(static_cast<uint8_t>(value >> (8U * i))); }
  }

  void u64(uint64_t value)
  {
    for (unsigned i = 0; i < 8; ++i) { u8(static_cast<uint8_t>(value >> (8U * i))); }
  }

  void raw(const std::byte* data, size_t size)
  {
    ensure(size, "encoded proof exceeds the backend artifact limit");
    bytes.insert(bytes.end(), data, data + size);
  }

  void packed(const std::vector<uint64_t>& values, uint8_t width)
  {
    if (width == 0 || width > 63) { throw std::invalid_argument("invalid packed integer width"); }
    const uint64_t mask = (uint64_t{1} << width) - 1;
    const size_t bit_count = checked_mul(values.size(), width, "packed proof bit count overflow");
    const size_t byte_count = checked_add(bit_count, 7, "packed proof byte count overflow") / 8;
    const size_t start = bytes.size();
    ensure(byte_count, "encoded proof exceeds the backend artifact limit");
    bytes.resize(checked_add(start, byte_count, "proof allocation overflow"), std::byte{0});
    size_t bit_offset = 0;
    for (uint64_t value : values) {
      if ((value & ~mask) != 0) { throw std::invalid_argument("integer exceeds declared packed width"); }
      for (uint8_t bit = 0; bit < width; ++bit, ++bit_offset) {
        if (((value >> bit) & 1U) != 0) {
          bytes[start + bit_offset / 8] |= std::byte(uint8_t{1} << (bit_offset % 8));
        }
      }
    }
  }

private:
  void ensure(size_t extra, const char* label) const
  {
    if (extra > static_cast<size_t>(MAX_WIRE_BYTES) ||
        bytes.size() > static_cast<size_t>(MAX_WIRE_BYTES) - extra) {
      throw std::length_error(label);
    }
  }
};

class Reader {
public:
  Reader(const std::vector<std::byte>& input, const LabradorProofDecodeLimits& requested)
      : bytes(input),
        max_encoded_bytes(std::min(
          requested.max_encoded_bytes,
          static_cast<size_t>(MAX_WIRE_BYTES))),
        max_decoded_bytes(std::min(
          requested.max_decoded_bytes,
          static_cast<size_t>(MAX_WIRE_BYTES))),
        max_polynomials(std::min(
          requested.max_polynomials,
          static_cast<size_t>(MAX_POLYNOMIALS)))
  {
    if (max_encoded_bytes == 0 || max_decoded_bytes == 0 || max_polynomials == 0) {
      throw std::invalid_argument("proof decoder limits must be non-zero");
    }
    if (bytes.size() > max_encoded_bytes) {
      throw std::invalid_argument("encoded proof exceeds the configured wire-size budget");
    }
  }

  uint8_t u8(const char* label)
  {
    require(1, label);
    return std::to_integer<uint8_t>(bytes[offset++]);
  }

  uint32_t u32(const char* label)
  {
    uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) { value |= uint32_t(u8(label)) << (8U * i); }
    return value;
  }

  uint64_t u64(const char* label)
  {
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) { value |= uint64_t(u8(label)) << (8U * i); }
    return value;
  }

  std::vector<uint64_t> packed(size_t count, uint8_t width, const char* label)
  {
    if (width == 0 || width > 63) { throw std::invalid_argument(std::string(label) + ": invalid width"); }
    const size_t bit_count = checked_mul(count, width, "encoded proof bit count overflow");
    const size_t byte_count = checked_add(bit_count, 7, "encoded proof byte count overflow") / 8;
    require(byte_count, label);
    check_temporary(checked_mul(count, sizeof(uint64_t), "proof unpacking allocation overflow"), label);
    std::vector<uint64_t> result(count, 0);
    size_t bit_offset = 0;
    for (size_t i = 0; i < count; ++i) {
      for (uint8_t bit = 0; bit < width; ++bit, ++bit_offset) {
        const uint8_t source = std::to_integer<uint8_t>(bytes[offset + bit_offset / 8]);
        result[i] |= uint64_t((source >> (bit_offset % 8)) & 1U) << bit;
      }
    }
    if (bit_count % 8 != 0) {
      const uint8_t final = std::to_integer<uint8_t>(bytes[offset + byte_count - 1]);
      const uint8_t used_mask = static_cast<uint8_t>((uint16_t{1} << (bit_count % 8)) - 1);
      if ((final & ~used_mask) != 0) { throw std::invalid_argument(std::string(label) + ": non-zero padding bits"); }
    }
    offset += byte_count;
    return result;
  }

  void expect_magic()
  {
    require(MAGIC.size(), "proof magic");
    if (!std::equal(MAGIC.begin(), MAGIC.end(), bytes.begin() + offset)) {
      throw std::invalid_argument("invalid LaBRADOR proof magic");
    }
    offset += MAGIC.size();
  }

  bool done() const { return offset == bytes.size(); }

  size_t polynomial_limit() const { return max_polynomials; }

  void claim_decoded(size_t count, size_t item_size, const char* label)
  {
    const size_t bytes_to_claim = checked_mul(count, item_size, "decoded proof allocation overflow");
    if (bytes_to_claim > max_decoded_bytes - decoded_bytes) {
      throw std::invalid_argument(std::string(label) + ": cumulative decoded-size budget exceeded");
    }
    decoded_bytes += bytes_to_claim;
  }

  void release_decoded(size_t count, size_t item_size)
  {
    const size_t bytes_to_release = checked_mul(count, item_size, "decoded proof release overflow");
    if (bytes_to_release > decoded_bytes) {
      throw std::logic_error("proof decoder byte accounting underflow");
    }
    decoded_bytes -= bytes_to_release;
  }

private:
  const std::vector<std::byte>& bytes;
  size_t offset = 0;
  size_t max_encoded_bytes;
  size_t max_decoded_bytes;
  size_t max_polynomials;
  size_t decoded_bytes = 0;

  void require(size_t count, const char* label) const
  {
    if (count > bytes.size() - std::min(offset, bytes.size())) {
      throw std::invalid_argument(std::string("truncated encoded proof at ") + label);
    }
  }

  void check_temporary(size_t allocation_bytes, const char* label) const
  {
    if (allocation_bytes > max_decoded_bytes - decoded_bytes) {
      throw std::invalid_argument(std::string(label) + ": unpacking buffer exceeds decoded-size budget");
    }
  }
};

std::vector<uint64_t> canonical_scalars(const Zq* values, size_t count)
{
  std::vector<uint64_t> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) { result.push_back(canonical_value(values[i])); }
  return result;
}

std::vector<uint64_t> signed_scalars(const Zq* values, size_t count, uint8_t& width)
{
  std::vector<int64_t> centered;
  centered.reserve(count);
  width = 1;
  for (size_t i = 0; i < count; ++i) {
    const int64_t value = centered_value(values[i]);
    centered.push_back(value);
    width = std::max(width, signed_width_for(value));
  }
  const uint64_t mask = (uint64_t{1} << width) - 1;
  std::vector<uint64_t> result;
  result.reserve(count);
  for (int64_t value : centered) { result.push_back(static_cast<uint64_t>(value) & mask); }
  return result;
}

void full_polynomials(Writer& writer, const std::vector<Tq>& values)
{
  writer.u64(values.size());
  const size_t scalar_count = checked_mul(values.size(), Rq::d, "encoded polynomial shape overflow");
  writer.packed(canonical_scalars(reinterpret_cast<const Zq*>(values.data()), scalar_count), FULL_WIDTH);
}

void full_polynomials_rq(Writer& writer, const std::vector<Rq>& values)
{
  writer.u64(values.size());
  const size_t scalar_count = checked_mul(values.size(), Rq::d, "encoded polynomial shape overflow");
  writer.packed(canonical_scalars(reinterpret_cast<const Zq*>(values.data()), scalar_count), FULL_WIDTH);
}

void signed_scalars(Writer& writer, const std::vector<Zq>& values)
{
  writer.u64(values.size());
  uint8_t width = 1;
  std::vector<uint64_t> packed_values = signed_scalars(values.data(), values.size(), width);
  writer.u8(width);
  writer.packed(packed_values, width);
}

void signed_polynomials(Writer& writer, const std::vector<Rq>& values)
{
  writer.u64(values.size());
  uint8_t width = 1;
  const size_t scalar_count = checked_mul(values.size(), Rq::d, "encoded polynomial shape overflow");
  std::vector<uint64_t> packed_values =
    signed_scalars(reinterpret_cast<const Zq*>(values.data()), scalar_count, width);
  writer.u8(width);
  writer.packed(packed_values, width);
}

void signed_ntt_polynomials(Writer& writer, const std::vector<Tq>& values)
{
  std::vector<Rq> coefficients(values.size());
  if (!values.empty()) {
    ICICLE_CHECK(ntt(values.data(), values.size(), NTTDir::kInverse, {}, coefficients.data()));
  }
  signed_polynomials(writer, coefficients);
}

size_t checked_poly_count(
  Reader& reader,
  const char* label,
  size_t expected = std::numeric_limits<size_t>::max())
{
  const uint64_t count = reader.u64(label);
  if (count > reader.polynomial_limit() || count > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument(std::string(label) + ": polynomial count exceeds backend limit");
  }
  const size_t result = static_cast<size_t>(count);
  if (expected != std::numeric_limits<size_t>::max() && result != expected) {
    throw std::invalid_argument(std::string(label) + ": unexpected polynomial count");
  }
  return result;
}

std::vector<Tq> read_full_polynomials(
  Reader& reader,
  const char* label,
  size_t expected = std::numeric_limits<size_t>::max())
{
  const size_t count = checked_poly_count(reader, label, expected);
  const size_t scalar_count = checked_mul(count, Rq::d, "decoded polynomial shape overflow");
  reader.claim_decoded(count, sizeof(Tq), label);
  const std::vector<uint64_t> packed = reader.packed(scalar_count, FULL_WIDTH, label);
  std::vector<Tq> result(count);
  Zq* scalars = reinterpret_cast<Zq*>(result.data());
  for (size_t i = 0; i < scalar_count; ++i) { scalars[i] = from_canonical(packed[i]); }
  return result;
}

std::vector<Rq> read_full_polynomials_rq(
  Reader& reader,
  const char* label,
  size_t expected = std::numeric_limits<size_t>::max())
{
  return read_full_polynomials(reader, label, expected);
}

int64_t decode_signed(uint64_t value, uint8_t width)
{
  const uint64_t sign = uint64_t{1} << (width - 1);
  if ((value & sign) == 0) { return static_cast<int64_t>(value); }
  return static_cast<int64_t>(value) - static_cast<int64_t>(uint64_t{1} << width);
}

std::vector<Zq> read_signed_scalars(Reader& reader, const char* label)
{
  const uint64_t count64 = reader.u64(label);
  const size_t scalar_limit = checked_mul(
    reader.polynomial_limit(), Rq::d, "proof scalar limit overflow");
  if (count64 > scalar_limit || count64 > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument(std::string(label) + ": scalar count exceeds backend limit");
  }
  const size_t count = static_cast<size_t>(count64);
  const uint8_t width = reader.u8(label);
  if (width == 0 || width > FULL_WIDTH) { throw std::invalid_argument(std::string(label) + ": invalid signed width"); }
  reader.claim_decoded(count, sizeof(Zq), label);
  const std::vector<uint64_t> packed = reader.packed(count, width, label);
  std::vector<Zq> result(count);
  uint8_t minimal_width = 1;
  for (size_t i = 0; i < count; ++i) {
    const int64_t value = decode_signed(packed[i], width);
    result[i] = from_centered(value);
    minimal_width = std::max(minimal_width, signed_width_for(value));
  }
  if (minimal_width != width) { throw std::invalid_argument(std::string(label) + ": non-minimal signed width"); }
  return result;
}

std::vector<Rq> read_signed_polynomials(
  Reader& reader,
  const char* label,
  size_t expected = std::numeric_limits<size_t>::max())
{
  const size_t count = checked_poly_count(reader, label, expected);
  const uint8_t width = reader.u8(label);
  if (width == 0 || width > FULL_WIDTH) { throw std::invalid_argument(std::string(label) + ": invalid signed width"); }
  const size_t scalar_count = checked_mul(count, Rq::d, "decoded polynomial shape overflow");
  reader.claim_decoded(count, sizeof(Rq), label);
  const std::vector<uint64_t> packed = reader.packed(scalar_count, width, label);
  std::vector<Rq> result(count);
  Zq* scalars = reinterpret_cast<Zq*>(result.data());
  uint8_t minimal_width = 1;
  for (size_t i = 0; i < scalar_count; ++i) {
    const int64_t value = decode_signed(packed[i], width);
    scalars[i] = from_centered(value);
    minimal_width = std::max(minimal_width, signed_width_for(value));
  }
  if (minimal_width != width) { throw std::invalid_argument(std::string(label) + ": non-minimal signed width"); }
  return result;
}

std::vector<Tq> read_signed_ntt_polynomials(
  Reader& reader,
  const char* label,
  size_t expected = std::numeric_limits<size_t>::max())
{
  std::vector<Rq> coefficients = read_signed_polynomials(reader, label, expected);
  reader.claim_decoded(coefficients.size(), sizeof(Tq), label);
  std::vector<Tq> result(coefficients.size());
  if (!coefficients.empty()) {
    ICICLE_CHECK(ntt(coefficients.data(), coefficients.size(), NTTDir::kForward, {}, result.data()));
  }
  reader.release_decoded(coefficients.size(), sizeof(Rq));
  return result;
}

void encode_prefix(Writer& writer, const BaseProverMessages& message)
{
  writer.u64(message.JL_i);
  full_polynomials(writer, message.u1);
  signed_scalars(writer, message.p);
  full_polynomials(writer, message.b_agg);
  full_polynomials(writer, message.u2);
}

BaseProverMessages decode_prefix(Reader& reader)
{
  BaseProverMessages result;
  const uint64_t jl_i = reader.u64("JL retry counter");
  if (jl_i >= (uint64_t{1} << 20) || jl_i > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument("JL retry counter exceeds the canonical limit");
  }
  result.JL_i = static_cast<size_t>(jl_i);
  result.u1 = read_full_polynomials(reader, "u1");
  result.p = read_signed_scalars(reader, "JL projection");
  result.b_agg = read_full_polynomials(reader, "JL proof polynomials");
  result.u2 = read_full_polynomials(reader, "u2");
  return result;
}

} // namespace

std::vector<std::byte> encode_labrador_proof(
  const std::vector<PartialTranscript>& transcripts,
  const LabradorFinalProof& final_proof,
  LabradorProofCodecStats* stats)
{
  if (transcripts.empty() || transcripts.size() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("invalid transcript count for proof codec");
  }

  Writer writer;
  writer.raw(MAGIC.data(), MAGIC.size());
  writer.u32(VERSION);
  writer.u32(static_cast<uint32_t>(transcripts.size()));
  const size_t header_end = writer.bytes.size();
  for (const PartialTranscript& transcript : transcripts) { encode_prefix(writer, transcript.prover_msg); }
  const size_t prefixes_end = writer.bytes.size();

  writer.u8(final_proof.uses_section_5_6 ? 1 : 0);
  if (!final_proof.uses_section_5_6) {
    signed_ntt_polynomials(writer, final_proof.base.z_hat);
    signed_polynomials(writer, final_proof.base.t);
    signed_polynomials(writer, final_proof.base.g);
    signed_polynomials(writer, final_proof.base.h);
  } else {
    const LabradorSection56Proof& proof = final_proof.section_5_6;
    writer.u64(proof.primary_count);
    signed_ntt_polynomials(writer, proof.z_hat);
    full_polynomials_rq(writer, proof.t);
    std::vector<Rq> g;
    g.reserve(proof.g_polynomial_count());
    g.push_back(proof.g0);
    g.insert(g.end(), proof.g_cross.begin(), proof.g_cross.end());
    g.insert(g.end(), proof.g_diagonal.begin(), proof.g_diagonal.end());
    signed_polynomials(writer, g);
    full_polynomials_rq(writer, proof.h_cross);
    full_polynomials_rq(writer, proof.h_diagonal);
  }

  if (stats != nullptr) {
    stats->header_bytes = header_end;
    stats->prefix_bytes = prefixes_end - header_end;
    stats->final_response_bytes = writer.bytes.size() - prefixes_end;
    stats->total_bytes = writer.bytes.size();
  }
  return writer.bytes;
}

LabradorDecodedProof decode_labrador_proof(
  const std::vector<std::byte>& encoded,
  const LabradorProofDecodeLimits& limits)
{
  Reader reader(encoded, limits);
  reader.expect_magic();
  if (reader.u32("proof version") != VERSION) { throw std::invalid_argument("unsupported LaBRADOR proof version"); }
  const uint32_t recursion_count = reader.u32("recursion count");
  if (recursion_count == 0 || recursion_count > icicle::labrador::backend_config::MAX_RECURSIONS) {
    throw std::invalid_argument("encoded recursion count exceeds backend parameters");
  }

  LabradorDecodedProof result;
  reader.claim_decoded(recursion_count, sizeof(BaseProverMessages), "prover-message table");
  result.prover_messages.reserve(recursion_count);
  for (uint32_t i = 0; i < recursion_count; ++i) { result.prover_messages.push_back(decode_prefix(reader)); }

  const uint8_t mode = reader.u8("final proof mode");
  if (mode == 0) {
    LabradorBaseCaseProof proof;
    proof.z_hat = read_signed_ntt_polynomials(reader, "base z");
    proof.t = read_signed_polynomials(reader, "base t");
    proof.g = read_signed_polynomials(reader, "base g");
    proof.h = read_signed_polynomials(reader, "base h");
    result.final_proof.uses_section_5_6 = false;
    result.final_proof.base = std::move(proof);
  } else if (mode == 1) {
    LabradorSection56Proof proof;
    const uint64_t primary_count = reader.u64("Section 5.6 primary count");
    if (primary_count == 0 || primary_count > reader.polynomial_limit() ||
        primary_count > std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument("invalid Section 5.6 primary count");
    }
    proof.primary_count = static_cast<size_t>(primary_count);
    proof.z_hat = read_signed_ntt_polynomials(reader, "Section 5.6 z");
    proof.t = read_full_polynomials_rq(reader, "Section 5.6 t");
    const size_t expected_g = checked_add(
      1,
      checked_mul(2, proof.primary_count, "Section 5.6 g count overflow"),
      "Section 5.6 g count overflow");
    if (expected_g > reader.polynomial_limit()) {
      throw std::invalid_argument("Section 5.6 g shape exceeds the polynomial limit");
    }
    {
      // The wire groups g0, g_cross and g_diagonal into one packed vector.
      // Check its exact public shape before allocation, then account for both
      // the temporary vector and the split retained representation at peak.
      std::vector<Rq> g = read_signed_polynomials(reader, "Section 5.6 g", expected_g);
      reader.claim_decoded(expected_g, sizeof(Rq), "Section 5.6 split g");
      proof.g0 = g[0];
      proof.g_cross.assign(g.begin() + 1, g.begin() + 1 + proof.primary_count);
      proof.g_diagonal.assign(g.begin() + 1 + proof.primary_count, g.end());
    }
    reader.release_decoded(expected_g, sizeof(Rq));
    proof.h_cross = read_full_polynomials_rq(reader, "Section 5.6 h cross");
    const size_t expected_h_diagonal =
      checked_add(proof.h_cross.size(), 1, "Section 5.6 h count overflow");
    proof.h_diagonal = read_full_polynomials_rq(
      reader, "Section 5.6 h diagonal", expected_h_diagonal);
    result.final_proof.uses_section_5_6 = true;
    result.final_proof.section_5_6 = std::move(proof);
  } else {
    throw std::invalid_argument("unknown final proof mode");
  }
  if (!reader.done()) { throw std::invalid_argument("trailing bytes after encoded LaBRADOR proof"); }
  return result;
}
