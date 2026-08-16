#include "examples_utils.h"
#include "labrador.h"
#include "prover.h"
#include "test_helpers.h"
#include "types.h"
#include "verifier.h"

#include "icicle/hash/keccak.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace icicle::labrador;

namespace {

constexpr std::array<char, 8> FILE_MAGIC{{'L', 'N', 'P', 'L', 'A', 'B', '0', '1'}};
constexpr uint32_t FILE_VERSION = 1;
constexpr uint32_t RING_DEGREE = 64;
constexpr uint64_t BABYKOALA_Q = 4'289'678'649'214'369'793ULL;
constexpr uint64_t BABYKOALA_Q_HALF = BABYKOALA_Q / 2;
constexpr uint64_t BABYKOALA_SQRT_Q_FLOOR = 2'071'153'941ULL;
constexpr size_t SHA3_256_BYTES = 32;

// The current CPU SHA3 backend ultimately passes its input length as an
// unsigned value.  A bounded artifact also prevents untrusted length fields
// from causing unreasonable allocations before their shapes are checked.
constexpr uint64_t MAX_FILE_BYTES = 128ULL * 1024ULL * 1024ULL;
constexpr uint64_t MAX_MODE_BYTES = 128;
constexpr uint64_t SOURCE_FINGERPRINT_BYTES = 64;
constexpr uint64_t MAX_SEED_BYTES = 1024ULL * 1024ULL;
// The runner is an integration harness, not a bulk untrusted-input service.
// Core prover paths keep several deep copies/work buffers, so cap aggregate
// ring-element counts conservatively because parsing and NTT need copies.
constexpr size_t MAX_RUNTIME_POLYNOMIALS = 262'144;
constexpr size_t MAX_SPLIT_PARTS = 256;

static_assert(sizeof(double) == sizeof(uint64_t), "This runner requires a binary64 double");
static_assert(std::numeric_limits<double>::is_iec559, "This runner requires IEEE-754 doubles");
static_assert(Rq::d == RING_DEGREE, "The runner schema is fixed to degree 64");
constexpr auto COMPILED_Q_STORAGE = Zq::get_modulus();
constexpr uint64_t COMPILED_Q = uint64_t(COMPILED_Q_STORAGE.limbs[0]) |
                                (uint64_t(COMPILED_Q_STORAGE.limbs[1]) << 32U);
static_assert(COMPILED_Q == BABYKOALA_Q, "The runner schema does not match the compiled ring modulus");

size_t checked_size(uint64_t value, const char* field)
{
  if (value > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    throw std::runtime_error(std::string(field) + " does not fit size_t");
  }
  return static_cast<size_t>(value);
}

size_t checked_add(size_t left, size_t right, const char* field)
{
  if (right > std::numeric_limits<size_t>::max() - left) {
    throw std::runtime_error(std::string("size overflow in ") + field);
  }
  return left + right;
}

size_t checked_mul(size_t left, size_t right, const char* field)
{
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::runtime_error(std::string("size overflow in ") + field);
  }
  return left * right;
}

class BinaryReader
{
public:
  explicit BinaryReader(const std::vector<std::byte>& bytes) : bytes_(bytes) {}

  size_t position() const { return position_; }
  size_t remaining() const { return bytes_.size() - position_; }

  const std::byte* read_bytes(size_t count, const char* field)
  {
    if (count > remaining()) {
      std::ostringstream error;
      error << "truncated " << field << " at byte " << position_ << " (need " << count << ", have "
            << remaining() << ')';
      throw std::runtime_error(error.str());
    }
    const std::byte* result = bytes_.data() + position_;
    position_ += count;
    return result;
  }

  uint32_t read_u32(const char* field)
  {
    const std::byte* input = read_bytes(sizeof(uint32_t), field);
    uint32_t result = 0;
    for (unsigned i = 0; i < sizeof(uint32_t); ++i) {
      result |= uint32_t(std::to_integer<uint8_t>(input[i])) << (8U * i);
    }
    return result;
  }

  uint64_t read_u64(const char* field)
  {
    const std::byte* input = read_bytes(sizeof(uint64_t), field);
    uint64_t result = 0;
    for (unsigned i = 0; i < sizeof(uint64_t); ++i) {
      result |= uint64_t(std::to_integer<uint8_t>(input[i])) << (8U * i);
    }
    return result;
  }

  int64_t read_i64(const char* field)
  {
    const uint64_t encoded = read_u64(field);
    if (encoded <= uint64_t(std::numeric_limits<int64_t>::max())) {
      return static_cast<int64_t>(encoded);
    }
    if (encoded == (uint64_t{1} << 63U)) { return std::numeric_limits<int64_t>::min(); }
    const uint64_t magnitude = (~encoded) + 1;
    return -static_cast<int64_t>(magnitude);
  }

  double read_f64(const char* field)
  {
    const uint64_t bits = read_u64(field);
    double result = 0.0;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
  }

  std::vector<std::byte> read_lp_bytes(uint64_t maximum, const char* field)
  {
    const uint64_t encoded_size = read_u64(field);
    if (encoded_size > maximum) {
      std::ostringstream error;
      error << field << " length " << encoded_size << " exceeds limit " << maximum;
      throw std::runtime_error(error.str());
    }
    const size_t size = checked_size(encoded_size, field);
    const std::byte* first = read_bytes(size, field);
    return {first, first + size};
  }

private:
  const std::vector<std::byte>& bytes_;
  size_t position_ = 0;
};

std::vector<std::byte> read_file(const std::string& path)
{
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) { throw std::runtime_error("cannot open relation bundle: " + path); }

  const std::streampos end = input.tellg();
  if (end < 0) { throw std::runtime_error("cannot determine relation bundle size"); }
  const uint64_t encoded_size = static_cast<uint64_t>(end);
  if (encoded_size == 0) { throw std::runtime_error("relation bundle is empty"); }
  if (encoded_size > MAX_FILE_BYTES) {
    std::ostringstream error;
    error << "relation bundle is " << encoded_size << " bytes; runner limit is " << MAX_FILE_BYTES;
    throw std::runtime_error(error.str());
  }

  std::vector<std::byte> bytes(checked_size(encoded_size, "file size"));
  input.seekg(0, std::ios::beg);
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input || static_cast<size_t>(input.gcount()) != bytes.size()) {
    throw std::runtime_error("short read while loading relation bundle");
  }
  return bytes;
}

std::string ascii_string(const std::vector<std::byte>& bytes, const char* field)
{
  std::string result;
  result.reserve(bytes.size());
  for (std::byte byte : bytes) {
    const unsigned character = std::to_integer<unsigned char>(byte);
    if (character < 0x20 || character > 0x7e) {
      throw std::runtime_error(std::string(field) + " must contain printable ASCII only");
    }
    result.push_back(static_cast<char>(character));
  }
  return result;
}

void validate_fingerprint(const std::string& fingerprint)
{
  if (fingerprint.size() != SOURCE_FINGERPRINT_BYTES) {
    throw std::runtime_error("source fingerprint must be exactly 64 hexadecimal characters");
  }
  const bool hexadecimal = std::all_of(fingerprint.begin(), fingerprint.end(), [](unsigned char character) {
    return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f') ||
           (character >= 'A' && character <= 'F');
  });
  if (!hexadecimal) { throw std::runtime_error("source fingerprint is not hexadecimal"); }
}

uint64_t centered_magnitude(int64_t value)
{
  if (value >= 0) { return static_cast<uint64_t>(value); }
  return static_cast<uint64_t>(-(value + 1)) + 1;
}

Zq zq_from_centered(int64_t value)
{
  const uint64_t magnitude = centered_magnitude(value);
  if (magnitude > BABYKOALA_Q_HALF) {
    throw std::runtime_error("non-canonical centered coefficient");
  }
  if (value >= 0) { return Zq::from_u64(magnitude); }
  return Zq::from_u64(magnitude).neg();
}

Rq read_polynomial(BinaryReader& reader, const char* field, long double* norm_squared = nullptr)
{
  Rq polynomial{};
  for (size_t coefficient = 0; coefficient < Rq::d; ++coefficient) {
    const int64_t value = reader.read_i64(field);
    const uint64_t magnitude = centered_magnitude(value);
    if (magnitude > BABYKOALA_Q_HALF) {
      std::ostringstream error;
      error << field << " coefficient " << coefficient << " is outside the canonical centered interval";
      throw std::runtime_error(error.str());
    }
    if (norm_squared != nullptr) {
      if (magnitude >= BABYKOALA_SQRT_Q_FLOOR) {
        std::ostringstream error;
        error << field << " coefficient " << coefficient << " violates the ICICLE norm-kernel input range";
        throw std::runtime_error(error.str());
      }
      const long double centered = static_cast<long double>(magnitude);
      *norm_squared += centered * centered;
    }
    polynomial.values[coefficient] = zq_from_centered(value);
  }
  return polynomial;
}

struct CoefficientEquality {
  std::vector<Rq> a;
  std::vector<Rq> phi;
  Rq b{};
};

struct CoefficientConstZero {
  std::vector<Rq> a;
  std::vector<Rq> phi;
  Zq b = Zq::zero();
};

struct RelationBundle {
  size_t r = 0;
  size_t n = 0;
  double beta = 0.0;
  size_t kappa = 0;
  size_t kappa1 = 0;
  size_t kappa2 = 0;
  uint32_t base1 = 0;
  uint32_t base2 = 0;
  uint32_t base3 = 0;
  size_t jl_out = 0;
  size_t aggregation_rounds = 0;
  size_t recursions = 0;
  std::string mode;
  std::string source_fingerprint;
  std::vector<std::byte> ajtai_seed;
  std::array<std::byte, SHA3_256_BYTES> public_digest{};
  std::vector<Rq> witness;
  std::vector<std::byte> oracle_seed;
  std::vector<CoefficientEquality> equality_constraints;
  std::vector<CoefficientConstZero> const_zero_constraints;
  long double witness_norm = 0.0L;
};

std::array<std::byte, SHA3_256_BYTES>
sha3_256(const std::byte* bytes, size_t size)
{
  if (size > std::numeric_limits<unsigned>::max()) {
    throw std::runtime_error("public section is too large for the current ICICLE SHA3 backend");
  }
  std::array<std::byte, SHA3_256_BYTES> digest{};
  // Hash on CPU so artifact verification is independent of the selected
  // proving backend and is always available in a normal repository build.
  ScopedCpuDevice cpu_scope("relation bundle SHA3-256");
  auto hasher = Sha3_256::create();
  if (hasher.output_size() != digest.size()) {
    throw std::runtime_error("unexpected SHA3-256 output size");
  }
  ICICLE_CHECK(hasher.hash(bytes, size, {}, digest.data()));
  return digest;
}

void check_minimum_section_bytes(
  uint64_t count, size_t bytes_per_item, size_t remaining, const char* section)
{
  if (bytes_per_item == 0 || count > static_cast<uint64_t>(remaining / bytes_per_item)) {
    throw std::runtime_error(std::string(section) + " count cannot fit in the remaining file");
  }
}

void validate_runtime_matrix_sizes(const RelationBundle& bundle)
{
  size_t n = bundle.n;
  size_t r = bundle.r;
  double beta = bundle.beta;
  uint32_t base1 = bundle.base1;
  uint32_t base2 = bundle.base2;
  uint32_t base3 = bundle.base3;
  size_t l1 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base1);
  size_t l2 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base2);
  size_t l3 = icicle::balanced_decomposition::compute_nof_digits<Zq>(base3);
  if (bundle.recursions > 1) {
    const LabradorDecompositionPlan decomposition =
      derive_decomposition_plan(n, r, beta);
    if (base1 != decomposition.base1 || base2 != decomposition.base2 || base3 != decomposition.base3) {
      throw std::runtime_error("recursive bundle bases do not match the Section 5.4 level plan");
    }
    l1 = decomposition.digits1;
    l2 = decomposition.digits2;
    l3 = decomposition.digits3;
  }

  for (size_t level = 0; level < bundle.recursions; ++level) {
    if (n == 0 || r == 0 || n > std::numeric_limits<uint32_t>::max() ||
        r > std::numeric_limits<uint32_t>::max() || !std::isfinite(beta) || beta <= 0.0) {
      throw std::runtime_error("derived recursive level has invalid dimensions or beta");
    }
    const size_t r_plus_one = checked_add(r, 1, "r + 1");
    const size_t r_choose_two = checked_mul(r, r_plus_one, "r(r+1)") / 2;
    const size_t t_len = checked_mul(checked_mul(l1, r, "l1*r"), bundle.kappa, "t length");
    const size_t g_len = checked_mul(l2, r_choose_two, "g length");
    const size_t h_len = checked_mul(l3, r_choose_two, "h length");

    const std::array<std::pair<size_t, const char*>, 4> matrix_sizes{{
      {checked_mul(n, bundle.kappa, "Ajtai A"), "Ajtai A"},
      {checked_mul(t_len, bundle.kappa1, "Ajtai B"), "Ajtai B"},
      {checked_mul(g_len, bundle.kappa1, "Ajtai C"), "Ajtai C"},
      {checked_mul(h_len, bundle.kappa2, "Ajtai D"), "Ajtai D"},
    }};
    size_t aggregate_matrix_size = 0;
    for (const auto& [size, name] : matrix_sizes) {
      aggregate_matrix_size = checked_add(aggregate_matrix_size, size, "aggregate Ajtai matrices");
      if (size > MAX_RUNTIME_POLYNOMIALS) {
        std::ostringstream error;
        error << "level " << level << ' ' << name << " would allocate " << size
              << " polynomials, above the runner safety limit";
        throw std::runtime_error(error.str());
      }
    }
    if (aggregate_matrix_size > MAX_RUNTIME_POLYNOMIALS) {
      std::ostringstream error;
      error << "level " << level << " aggregate Ajtai matrices exceed the runner safety limit";
      throw std::runtime_error(error.str());
    }

    const size_t witness_count = checked_mul(r, n, "r*n");
    const size_t jl_working_set = checked_mul(bundle.jl_out, witness_count, "JL_out*r*n");
    if (jl_working_set > MAX_RUNTIME_POLYNOMIALS) {
      std::ostringstream error;
      error << "level " << level << " JL_out*r*n exceeds the runner safety limit";
      throw std::runtime_error(error.str());
    }
    const long double response_bound =
      static_cast<long double>(OP_NORM_BOUND) * beta * std::sqrt(static_cast<long double>(r));
    const long double projection_bound = std::sqrt(static_cast<long double>(bundle.jl_out) / 2.0L) * beta;
    const long double modular_jl_limit =
      std::sqrt(30.0L / 128.0L) * static_cast<long double>(BABYKOALA_Q) / 125.0L;
    if (!std::isfinite(response_bound) || !std::isfinite(projection_bound) ||
        response_bound >= std::ldexp(1.0L, 64) || projection_bound < 1.0L ||
        projection_bound >= std::ldexp(1.0L, 64)) {
      throw std::runtime_error("derived recursive verifier norm bound is invalid");
    }
    if (static_cast<long double>(beta) > modular_jl_limit) {
      std::ostringstream error;
      error << "level " << level << " beta exceeds the modular-JL condition from Theorem 5.1";
      throw std::runtime_error(error.str());
    }

    if (level + 1 < bundle.recursions) {
      const LabradorTransitionPlan transition = derive_transition_plan(
        n,
        r,
        bundle.kappa,
        base1,
        base2,
        base3,
        l1,
        l2,
        l3,
        beta);
      if (transition.mu > MAX_SPLIT_PARTS || transition.nu > MAX_SPLIT_PARTS) {
        throw std::runtime_error("derived recursive split exceeds the runner safety limit");
      }
      n = transition.n_next;
      r = transition.r_next;
      beta = transition.beta_next;
      base1 = transition.base1;
      base2 = transition.base2;
      base3 = transition.base3;
      l1 = transition.digits1;
      l2 = transition.digits2;
      l3 = transition.digits3;
    }
  }

  const size_t aggregation_challenges =
    checked_mul(bundle.aggregation_rounds, bundle.jl_out, "aggregation_rounds*JL_out");
  if (aggregation_challenges > MAX_RUNTIME_POLYNOMIALS) {
    throw std::runtime_error("aggregation_rounds*JL_out exceeds the runner safety limit");
  }
}

RelationBundle parse_bundle(const std::vector<std::byte>& bytes)
{
  BinaryReader reader(bytes);
  const std::byte* magic = reader.read_bytes(FILE_MAGIC.size(), "magic");
  for (size_t i = 0; i < FILE_MAGIC.size(); ++i) {
    if (std::to_integer<unsigned char>(magic[i]) != static_cast<unsigned char>(FILE_MAGIC[i])) {
      throw std::runtime_error("bad relation bundle magic (expected LNPLAB01)");
    }
  }

  const uint32_t version = reader.read_u32("version");
  if (version != FILE_VERSION) { throw std::runtime_error("unsupported relation bundle version"); }
  const uint32_t degree = reader.read_u32("ring degree");
  if (degree != RING_DEGREE) { throw std::runtime_error("relation bundle degree is not 64"); }
  const uint64_t modulus = reader.read_u64("ring modulus");
  if (modulus != BABYKOALA_Q) {
    throw std::runtime_error("relation modulus does not match the compiled BabyKoala backend");
  }

  RelationBundle bundle;
  const uint64_t r_encoded = reader.read_u64("r");
  const uint64_t n_encoded = reader.read_u64("n");
  if (r_encoded == 0 || n_encoded == 0) { throw std::runtime_error("r and n must be positive"); }
  if (r_encoded > std::numeric_limits<uint32_t>::max() || n_encoded > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("r or n exceeds ICICLE's uint32 matrix dimensions");
  }
  bundle.r = checked_size(r_encoded, "r");
  bundle.n = checked_size(n_encoded, "n");
  const size_t rn = checked_mul(bundle.r, bundle.n, "r*n");
  const size_t rr = checked_mul(bundle.r, bundle.r, "r*r");
  if (rn > static_cast<size_t>(std::numeric_limits<uint32_t>::max()) ||
      rr > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("r*n or r*r exceeds ICICLE's uint32 matrix dimensions");
  }
  if (rn > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      rr > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error("r*n or r*r exceeds the LaBRADOR NTT batch interface");
  }

  bundle.beta = reader.read_f64("beta");
  if (!std::isfinite(bundle.beta) || bundle.beta <= 0.0) {
    throw std::runtime_error("beta must be finite and positive");
  }
  bundle.kappa = checked_size(reader.read_u64("kappa"), "kappa");
  bundle.kappa1 = checked_size(reader.read_u64("kappa1"), "kappa1");
  bundle.kappa2 = checked_size(reader.read_u64("kappa2"), "kappa2");
  if (bundle.kappa == 0 || bundle.kappa1 == 0 || bundle.kappa2 == 0) {
    throw std::runtime_error("all Ajtai commitment ranks must be positive");
  }
  const size_t minimum_rank = secure_msis_rank();
  if (bundle.kappa < minimum_rank || bundle.kappa1 < minimum_rank || bundle.kappa2 < minimum_rank) {
    throw std::runtime_error("Ajtai commitment rank is below the backend security heuristic");
  }
  bundle.base1 = reader.read_u32("base1");
  bundle.base2 = reader.read_u32("base2");
  bundle.base3 = reader.read_u32("base3");
  if (bundle.base1 < 2 || bundle.base2 < 2 || bundle.base3 < 2) {
    throw std::runtime_error("all balanced-decomposition bases must be at least 2");
  }
  bundle.jl_out = checked_size(reader.read_u64("JL_out"), "JL_out");
  bundle.aggregation_rounds = checked_size(reader.read_u64("aggregation rounds"), "aggregation rounds");
  bundle.recursions = checked_size(reader.read_u64("recursions"), "recursions");
  if (bundle.jl_out != 256) {
    throw std::runtime_error("the current runner security profile requires JL_out=256");
  }
  if (bundle.aggregation_rounds != 3) {
    throw std::runtime_error("the BabyKoala security profile requires exactly 3 aggregation rounds");
  }
  if (bundle.recursions == 0 || bundle.recursions > 8) {
    throw std::runtime_error("recursions must be between 1 and 8 executions");
  }

  bundle.mode = ascii_string(reader.read_lp_bytes(MAX_MODE_BYTES, "mode"), "mode");
  if (bundle.mode != "synthetic-principal-v1" && bundle.mode != "json-principal-v1") {
    throw std::runtime_error("unsupported relation mode");
  }
  bundle.source_fingerprint =
    ascii_string(reader.read_lp_bytes(SOURCE_FINGERPRINT_BYTES, "source fingerprint"), "source fingerprint");
  validate_fingerprint(bundle.source_fingerprint);
  bundle.ajtai_seed = reader.read_lp_bytes(MAX_SEED_BYTES, "Ajtai seed");
  if (bundle.ajtai_seed.size() < 16) { throw std::runtime_error("Ajtai seed must contain at least 16 bytes"); }
  bundle.oracle_seed = reader.read_lp_bytes(MAX_SEED_BYTES, "oracle seed");
  if (bundle.oracle_seed.size() < 16) { throw std::runtime_error("oracle seed must contain at least 16 bytes"); }

  validate_runtime_matrix_sizes(bundle);

  const size_t polynomial_bytes = checked_mul(Rq::d, sizeof(int64_t), "polynomial bytes");
  const size_t equality_polynomials = checked_add(checked_add(rr, rn, "equality a+phi"), 1, "equality b");
  const size_t equality_bytes = checked_mul(equality_polynomials, polynomial_bytes, "equality bytes");
  const uint64_t equality_count_encoded = reader.read_u64("equality constraint count");
  check_minimum_section_bytes(equality_count_encoded, equality_bytes, reader.remaining(), "equality constraint");
  const size_t equality_count = checked_size(equality_count_encoded, "equality constraint count");
  if (equality_count > bundle.equality_constraints.max_size()) {
    throw std::runtime_error("too many equality constraints");
  }
  bundle.equality_constraints.reserve(equality_count);
  for (size_t constraint = 0; constraint < equality_count; ++constraint) {
    CoefficientEquality equality;
    equality.a.reserve(rr);
    equality.phi.reserve(rn);
    for (size_t i = 0; i < rr; ++i) { equality.a.push_back(read_polynomial(reader, "equality a")); }
    for (size_t i = 0; i < rn; ++i) { equality.phi.push_back(read_polynomial(reader, "equality phi")); }
    equality.b = read_polynomial(reader, "equality b");
    bundle.equality_constraints.push_back(std::move(equality));
  }

  const size_t const_zero_bytes = checked_add(
    checked_mul(checked_add(rr, rn, "const-zero a+phi"), polynomial_bytes, "const-zero polynomial bytes"),
    sizeof(int64_t), "const-zero b0 bytes");
  const uint64_t const_zero_count_encoded = reader.read_u64("constant-zero constraint count");
  check_minimum_section_bytes(const_zero_count_encoded, const_zero_bytes, reader.remaining(), "constant-zero constraint");
  const size_t const_zero_count = checked_size(const_zero_count_encoded, "constant-zero constraint count");
  if (const_zero_count > bundle.const_zero_constraints.max_size()) {
    throw std::runtime_error("too many constant-zero constraints");
  }
  bundle.const_zero_constraints.reserve(const_zero_count);
  for (size_t constraint = 0; constraint < const_zero_count; ++constraint) {
    CoefficientConstZero constant_zero;
    constant_zero.a.reserve(rr);
    constant_zero.phi.reserve(rn);
    for (size_t i = 0; i < rr; ++i) { constant_zero.a.push_back(read_polynomial(reader, "constant-zero a")); }
    for (size_t i = 0; i < rn; ++i) {
      constant_zero.phi.push_back(read_polynomial(reader, "constant-zero phi"));
    }
    constant_zero.b = zq_from_centered(reader.read_i64("constant-zero b0"));
    bundle.const_zero_constraints.push_back(std::move(constant_zero));
  }
  if (equality_count == 0 && const_zero_count == 0) {
    throw std::runtime_error("relation bundle contains no constraints");
  }

  const size_t public_end = reader.position();
  const std::byte* expected_digest_bytes = reader.read_bytes(SHA3_256_BYTES, "public digest");
  std::copy(expected_digest_bytes, expected_digest_bytes + SHA3_256_BYTES, bundle.public_digest.begin());
  const auto computed_digest = sha3_256(bytes.data(), public_end);
  if (computed_digest != bundle.public_digest) {
    throw std::runtime_error("SHA3-256 digest of public relation does not match bundle");
  }

  const uint64_t witness_count_encoded = reader.read_u64("witness polynomial count");
  if (witness_count_encoded != static_cast<uint64_t>(rn)) {
    throw std::runtime_error("witness polynomial count must equal r*n");
  }
  check_minimum_section_bytes(witness_count_encoded, polynomial_bytes, reader.remaining(), "witness");
  bundle.witness.reserve(rn);
  long double witness_norm_squared = 0.0L;
  for (size_t i = 0; i < rn; ++i) {
    bundle.witness.push_back(read_polynomial(reader, "witness", &witness_norm_squared));
  }
  bundle.witness_norm = std::sqrt(witness_norm_squared);
  if (!std::isfinite(bundle.witness_norm) || bundle.witness_norm >= static_cast<long double>(bundle.beta)) {
    std::ostringstream error;
    error << "witness L2 norm " << bundle.witness_norm << " is not strictly below beta " << bundle.beta;
    throw std::runtime_error(error.str());
  }

  if (reader.remaining() != 0) { throw std::runtime_error("trailing bytes after relation bundle"); }

  const long double response_bound =
    static_cast<long double>(OP_NORM_BOUND) * bundle.beta * std::sqrt(static_cast<long double>(bundle.r));
  const long double projection_bound =
    std::sqrt(static_cast<long double>(bundle.jl_out / 2)) * static_cast<long double>(bundle.beta);
  if (!std::isfinite(response_bound) || !std::isfinite(projection_bound) ||
      response_bound >= static_cast<long double>(std::numeric_limits<uint64_t>::max()) ||
      projection_bound >= static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
    throw std::runtime_error("beta-derived verifier bound does not fit uint64_t");
  }
  if (projection_bound < 1.0L) {
    throw std::runtime_error("beta is too small: the integer JL norm bound would be zero");
  }

  return bundle;
}

void forward_ntt(std::vector<Rq>& polynomials, const char* field)
{
  if (polynomials.empty()) { return; }
  if (polynomials.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(std::string(field) + " NTT batch exceeds int");
  }
  std::vector<Tq> evaluations(polynomials.size());
  ICICLE_CHECK(ntt(
    polynomials.data(), static_cast<int>(polynomials.size()), NTTDir::kForward, NegacyclicNTTConfig{},
    evaluations.data()));
  polynomials = std::move(evaluations);
}

void forward_ntt(Rq& polynomial)
{
  Tq evaluation{};
  ICICLE_CHECK(ntt(&polynomial, 1, NTTDir::kForward, NegacyclicNTTConfig{}, &evaluation));
  polynomial = evaluation;
}

std::string digest_hex(const std::array<std::byte, SHA3_256_BYTES>& digest)
{
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (std::byte byte : digest) { result << std::setw(2) << std::to_integer<unsigned>(byte); }
  return result.str();
}

bool run_relation(RelationBundle bundle)
{
  for (auto& equality : bundle.equality_constraints) {
    forward_ntt(equality.a, "equality a");
    forward_ntt(equality.phi, "equality phi");
    forward_ntt(equality.b);
  }
  for (auto& constant_zero : bundle.const_zero_constraints) {
    forward_ntt(constant_zero.a, "constant-zero a");
    forward_ntt(constant_zero.phi, "constant-zero phi");
  }

  size_t digits1 = 0;
  size_t digits2 = 0;
  size_t digits3 = 0;
  if (bundle.recursions > 1) {
    const LabradorDecompositionPlan decomposition =
      derive_decomposition_plan(bundle.n, bundle.r, bundle.beta);
    digits1 = decomposition.digits1;
    digits2 = decomposition.digits2;
    digits3 = decomposition.digits3;
  }
  LabradorParam parameters{
    bundle.r,
    bundle.n,
    bundle.ajtai_seed,
    bundle.kappa,
    bundle.kappa1,
    bundle.kappa2,
    bundle.base1,
    bundle.base2,
    bundle.base3,
    bundle.beta,
    digits1,
    digits2,
    digits3,
  };
  parameters.JL_out = bundle.jl_out;
  parameters.num_aggregation_rounds = bundle.aggregation_rounds;
  LabradorInstance instance{parameters};
  // LabradorInstance currently copies LabradorParam. Release the duplicate
  // matrices before constructing the prover, which copies the instance again.
  parameters.A.clear();
  parameters.B.clear();
  parameters.C.clear();
  parameters.D.clear();
  parameters.A.shrink_to_fit();
  parameters.B.shrink_to_fit();
  parameters.C.shrink_to_fit();
  parameters.D.shrink_to_fit();

  for (auto& equality : bundle.equality_constraints) {
    EqualityInstance constraint(bundle.r, bundle.n);
    constraint.a = std::move(equality.a);
    constraint.phi = std::move(equality.phi);
    constraint.b = equality.b;
    instance.add_equality_constraint(constraint);
  }
  for (auto& constant_zero : bundle.const_zero_constraints) {
    ConstZeroInstance constraint(bundle.r, bundle.n);
    constraint.a = std::move(constant_zero.a);
    constraint.phi = std::move(constant_zero.phi);
    constraint.b = constant_zero.b;
    instance.add_const_zero_constraint(constraint);
  }

  if (!lab_witness_legit(instance, bundle.witness)) {
    throw std::runtime_error("loaded witness does not satisfy the loaded LaBRADOR relation");
  }

  // The public digest covers the caller's oracle context and every public
  // coefficient.  Appending it compensates for create_oracle_seed(), which
  // currently serializes only constraint counts rather than their contents.
  bundle.oracle_seed.insert(bundle.oracle_seed.end(), bundle.public_digest.begin(), bundle.public_digest.end());

  std::cout << "Relation accepted"
            << ": mode=" << bundle.mode << ", r=" << bundle.r << ", n=" << bundle.n
            << ", equality=" << instance.equality_constraints.size()
            << ", const_zero=" << instance.const_zero_constraints.size() << ", recursions=" << bundle.recursions
            << '\n';
  std::cout << "Source fingerprint: " << bundle.source_fingerprint << '\n';
  std::cout << "Public SHA3-256: " << digest_hex(bundle.public_digest) << '\n';
  std::cout << "Witness L2 norm: " << static_cast<double>(bundle.witness_norm) << " < beta " << bundle.beta << '\n';

  LabradorProver prover{
    instance, bundle.witness, bundle.oracle_seed.data(), bundle.oracle_seed.size(), bundle.recursions};
  auto [transcripts, final_proof] = prover.prove();
  if (transcripts.size() != bundle.recursions) {
    throw std::runtime_error("prover returned an unexpected transcript count");
  }

  size_t prefix_bytes = 0;
  for (const auto& transcript : transcripts) {
    prefix_bytes = checked_add(prefix_bytes, transcript.proof_size(), "recursive proof prefixes");
  }
  const size_t final_response_bytes = final_proof.size();
  const size_t total_proof_bytes =
    checked_add(prefix_bytes, final_response_bytes, "total recursive proof");
  std::cout << "Recursive proof (native coefficient encoding): prefixes=" << prefix_bytes
            << " B, final_response=" << final_response_bytes << " B, total=" << total_proof_bytes
            << " B (" << std::fixed << std::setprecision(3)
            << static_cast<double>(total_proof_bytes) / 1024.0 << " KiB)\n";

  std::vector<BaseProverMessages> prover_messages;
  prover_messages.reserve(transcripts.size());
  for (const auto& transcript : transcripts) { prover_messages.push_back(transcript.prover_msg); }

  LabradorVerifier verifier{
    instance, prover_messages, final_proof, bundle.oracle_seed.data(), bundle.oracle_seed.size(), bundle.recursions};
  const bool verified = verifier.verify();
  std::cout << "Verification: " << (verified ? "PASSED" : "FAILED") << '\n';
  return verified;
}

} // namespace

int main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " DEVICE relation.lab\n"
              << "Example: " << argv[0] << " CPU relation.lab\n";
    return 2;
  }

  try {
    // The repository helper interprets argv[1] as the device name.
    try_load_and_set_backend_device(argc, argv);
    RelationBundle bundle = parse_bundle(read_file(argv[2]));
    return run_relation(std::move(bundle)) ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "lab_runner: " << error.what() << '\n';
    return 1;
  }
}
