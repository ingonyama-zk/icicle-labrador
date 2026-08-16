#define _POSIX_C_SOURCE 200809L

/*
 * C11 frontend for the versioned LNPLAB01 relation bundle.
 *
 * This program generates the same deterministic synthetic relation as
 * scripts/lab.py and can set the Section 5.4 bases needed by the recursive
 * C++ runner.  It is a relation/secret-witness codec, not a proof
 * implementation or proof serializer.  The resulting .lab file contains the
 * witness in plaintext.
 */

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LAB_DEGREE 64u
#define LAB_Q UINT64_C(4289678649214369793)
#define LAB_Q_HALF (LAB_Q / UINT64_C(2))
#define LAB_SQRT_Q_FLOOR UINT64_C(2071153941)
#define LAB_RANK UINT64_C(37)
#define LAB_JL_ROWS UINT64_C(256)
#define LAB_AGGREGATION_ROUNDS UINT64_C(3)
#define LAB_MAX_FILE_BYTES (UINT64_C(128) * UINT64_C(1024) * UINT64_C(1024))
#define LAB_MAX_DIMENSION UINT64_C(128)
#define LAB_MAX_WITNESS_POLYS UINT64_C(4096)
#define LAB_MAX_WITNESS_BOUND UINT64_C(10000)

static const uint8_t LAB_MAGIC[8] = {'L', 'N', 'P', 'L', 'A', 'B', '0', '1'};
static const char LAB_MODE[] = "synthetic-principal-v1";
static const char LAB_DEFAULT_FINGERPRINT[] =
  "c84303daf3d2e07a8b0ce815abbc80d3e045a43a564ba555a41fb3cbeccbcc62";

typedef struct {
  uint8_t* data;
  size_t size;
  size_t capacity;
} Buffer;

typedef struct {
  const uint8_t* data;
  size_t size;
  size_t offset;
} Reader;

typedef struct {
  uint64_t r;
  uint64_t n;
  uint64_t witness_bound;
  uint64_t recursions;
  const uint8_t* seed;
  size_t seed_len;
  const char* fingerprint;
} GenerateOptions;

typedef struct {
  uint64_t counter;
  const uint8_t* seed;
  size_t seed_len;
} ShakeRng;

static void fail(const char* message)
{
  fprintf(stderr, "lab_c: error: %s\n", message);
  exit(2);
}

static void fail_errno(const char* action, const char* path)
{
  fprintf(stderr, "lab_c: error: %s '%s': %s\n", action, path, strerror(errno));
  exit(2);
}

static bool checked_add_size(size_t a, size_t b, size_t* out)
{
  if (b > SIZE_MAX - a) return false;
  *out = a + b;
  return true;
}

static bool checked_mul_size(size_t a, size_t b, size_t* out)
{
  if (a != 0 && b > SIZE_MAX / a) return false;
  *out = a * b;
  return true;
}

static void buffer_reserve(Buffer* buffer, size_t extra)
{
  size_t needed = 0;
  if (!checked_add_size(buffer->size, extra, &needed) || needed > (size_t)LAB_MAX_FILE_BYTES)
    fail("relation bundle exceeds the 128 MiB safety limit");
  if (needed <= buffer->capacity) return;
  size_t capacity = buffer->capacity == 0 ? 4096u : buffer->capacity;
  while (capacity < needed) {
    if (capacity > (size_t)LAB_MAX_FILE_BYTES / 2u) {
      capacity = (size_t)LAB_MAX_FILE_BYTES;
      break;
    }
    capacity *= 2u;
  }
  uint8_t* replacement = (uint8_t*)realloc(buffer->data, capacity);
  if (replacement == NULL) fail("out of memory while building relation bundle");
  buffer->data = replacement;
  buffer->capacity = capacity;
}

static void buffer_bytes(Buffer* buffer, const void* bytes, size_t size)
{
  buffer_reserve(buffer, size);
  if (size != 0) memcpy(buffer->data + buffer->size, bytes, size);
  buffer->size += size;
}

static void buffer_u32(Buffer* buffer, uint32_t value)
{
  uint8_t encoded[4];
  for (unsigned i = 0; i < 4; ++i) encoded[i] = (uint8_t)(value >> (8u * i));
  buffer_bytes(buffer, encoded, sizeof(encoded));
}

static void buffer_u64(Buffer* buffer, uint64_t value)
{
  uint8_t encoded[8];
  for (unsigned i = 0; i < 8; ++i) encoded[i] = (uint8_t)(value >> (8u * i));
  buffer_bytes(buffer, encoded, sizeof(encoded));
}

static void buffer_i64(Buffer* buffer, int64_t value)
{
  buffer_u64(buffer, (uint64_t)value);
}

static void buffer_f64(Buffer* buffer, double value)
{
  uint64_t bits = 0;
  if (sizeof(value) != sizeof(bits)) fail("this platform does not use binary64 doubles");
  memcpy(&bits, &value, sizeof(bits));
  buffer_u64(buffer, bits);
}

static void buffer_lp(Buffer* buffer, const void* bytes, size_t size)
{
  buffer_u64(buffer, (uint64_t)size);
  buffer_bytes(buffer, bytes, size);
}

static void buffer_poly(Buffer* buffer, const int64_t polynomial[LAB_DEGREE])
{
  for (size_t i = 0; i < LAB_DEGREE; ++i) buffer_i64(buffer, polynomial[i]);
}

static uint64_t rotate_left_u64(uint64_t value, unsigned shift)
{
  return shift == 0 ? value : (value << shift) | (value >> (64u - shift));
}

static void keccak_f1600(uint64_t state[25])
{
  static const uint64_t round_constants[24] = {
    UINT64_C(0x0000000000000001), UINT64_C(0x0000000000008082),
    UINT64_C(0x800000000000808a), UINT64_C(0x8000000080008000),
    UINT64_C(0x000000000000808b), UINT64_C(0x0000000080000001),
    UINT64_C(0x8000000080008081), UINT64_C(0x8000000000008009),
    UINT64_C(0x000000000000008a), UINT64_C(0x0000000000000088),
    UINT64_C(0x0000000080008009), UINT64_C(0x000000008000000a),
    UINT64_C(0x000000008000808b), UINT64_C(0x800000000000008b),
    UINT64_C(0x8000000000008089), UINT64_C(0x8000000000008003),
    UINT64_C(0x8000000000008002), UINT64_C(0x8000000000000080),
    UINT64_C(0x000000000000800a), UINT64_C(0x800000008000000a),
    UINT64_C(0x8000000080008081), UINT64_C(0x8000000000008080),
    UINT64_C(0x0000000080000001), UINT64_C(0x8000000080008008)
  };
  static const unsigned rotations[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
    27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
  };
  static const unsigned permutations[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
    15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
  };

  for (unsigned round = 0; round < 24; ++round) {
    uint64_t columns[5];
    for (unsigned i = 0; i < 5; ++i)
      columns[i] = state[i] ^ state[i + 5] ^ state[i + 10] ^ state[i + 15] ^ state[i + 20];
    for (unsigned i = 0; i < 5; ++i) {
      uint64_t delta = columns[(i + 4) % 5] ^ rotate_left_u64(columns[(i + 1) % 5], 1);
      for (unsigned j = 0; j < 25; j += 5) state[j + i] ^= delta;
    }

    uint64_t current = state[1];
    for (unsigned i = 0; i < 24; ++i) {
      unsigned destination = permutations[i];
      uint64_t saved = state[destination];
      state[destination] = rotate_left_u64(current, rotations[i]);
      current = saved;
    }

    for (unsigned row = 0; row < 25; row += 5) {
      uint64_t values[5];
      for (unsigned i = 0; i < 5; ++i) values[i] = state[row + i];
      for (unsigned i = 0; i < 5; ++i)
        state[row + i] ^= (~values[(i + 1) % 5]) & values[(i + 2) % 5];
    }
    state[0] ^= round_constants[round];
  }
}

static void keccak_absorb_block(uint64_t state[25], const uint8_t* block, size_t rate)
{
  for (size_t i = 0; i < rate; ++i)
    state[i / 8u] ^= (uint64_t)block[i] << (8u * (unsigned)(i % 8u));
  keccak_f1600(state);
}

static void keccak_sponge(
  const uint8_t* input,
  size_t input_size,
  uint8_t suffix,
  uint8_t* output,
  size_t output_size)
{
  enum { RATE = 136 };
  uint64_t state[25] = {0};
  while (input_size >= RATE) {
    keccak_absorb_block(state, input, RATE);
    input += RATE;
    input_size -= RATE;
  }

  uint8_t final_block[RATE] = {0};
  if (input_size != 0) memcpy(final_block, input, input_size);
  final_block[input_size] ^= suffix;
  final_block[RATE - 1] ^= UINT8_C(0x80);
  keccak_absorb_block(state, final_block, RATE);

  while (output_size != 0) {
    size_t take = output_size < RATE ? output_size : RATE;
    for (size_t i = 0; i < take; ++i)
      output[i] = (uint8_t)(state[i / 8u] >> (8u * (unsigned)(i % 8u)));
    output += take;
    output_size -= take;
    if (output_size != 0) keccak_f1600(state);
  }
}

static void sha3_256(const uint8_t* input, size_t input_size, uint8_t output[32])
{
  keccak_sponge(input, input_size, UINT8_C(0x06), output, 32);
}

static void shake256(const uint8_t* input, size_t input_size, uint8_t* output, size_t output_size)
{
  keccak_sponge(input, input_size, UINT8_C(0x1f), output, output_size);
}

static void shake_with_domain(
  const char* domain,
  const uint8_t* seed,
  size_t seed_len,
  const uint8_t* suffix,
  size_t suffix_len,
  uint8_t* output,
  size_t output_len)
{
  Buffer input = {0};
  buffer_bytes(&input, domain, strlen(domain) + 1u);
  buffer_bytes(&input, seed, seed_len);
  buffer_bytes(&input, suffix, suffix_len);
  shake256(input.data, input.size, output, output_len);
  free(input.data);
}

static uint64_t load_u64_le(const uint8_t bytes[8])
{
  uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i) value |= (uint64_t)bytes[i] << (8u * i);
  return value;
}

static uint64_t rng_below(ShakeRng* rng, uint64_t upper)
{
  if (upper == 0) fail("invalid RNG range");
  for (;;) {
    uint8_t counter[8];
    uint8_t block[8];
    for (unsigned i = 0; i < 8; ++i) counter[i] = (uint8_t)(rng->counter >> (8u * i));
    rng->counter++;
    shake_with_domain(
      "LNPLAB/SYNTHETIC/v1", rng->seed, rng->seed_len, counter, sizeof(counter), block, sizeof(block));
    uint64_t candidate = load_u64_le(block);
    uint64_t remainder = (uint64_t)(((UINT64_MAX % upper) + 1u) % upper);
    uint64_t cutoff = (uint64_t)(0u - remainder);
    if (remainder == 0 || candidate < cutoff) return candidate % upper;
  }
}

static int64_t centered_negate(int64_t value)
{
  if (value == 0) return 0;
  return -value;
}

static void poly_zero(int64_t polynomial[LAB_DEGREE])
{
  memset(polynomial, 0, sizeof(int64_t) * LAB_DEGREE);
}

static void poly_add(int64_t destination[LAB_DEGREE], const int64_t source[LAB_DEGREE])
{
  for (size_t i = 0; i < LAB_DEGREE; ++i) destination[i] += source[i];
}

static void negacyclic_mul(
  const int64_t left[LAB_DEGREE],
  const int64_t right[LAB_DEGREE],
  int64_t output[LAB_DEGREE])
{
  poly_zero(output);
  for (size_t i = 0; i < LAB_DEGREE; ++i) {
    if (left[i] == 0) continue;
    for (size_t j = 0; j < LAB_DEGREE; ++j) {
      if (right[j] == 0) continue;
      int64_t product = left[i] * right[j];
      size_t degree = i + j;
      if (degree >= LAB_DEGREE) {
        degree -= LAB_DEGREE;
        product = -product;
      }
      output[degree] += product;
    }
  }
}

static uint64_t integer_sqrt(uint64_t value)
{
  uint64_t root = (uint64_t)sqrt((double)value);
  while (root != 0 && root > value / root) --root;
  while (root < UINT64_MAX && root + 1u <= value / (root + 1u)) ++root;
  return root;
}

static bool power_at_least(uint64_t base, uint64_t exponent, uint64_t target)
{
  uint64_t value = 1;
  for (uint64_t i = 0; i < exponent; ++i) {
    if (value > (target - 1u) / base) return true;
    value *= base;
  }
  return value >= target;
}

static uint32_t ceil_nth_root_u64(uint64_t value, uint64_t exponent)
{
  uint64_t estimate = (uint64_t)exp(log((double)value) / (double)exponent);
  if (estimate < 1) estimate = 1;
  while (!power_at_least(estimate, exponent, value)) ++estimate;
  while (estimate > 1 && power_at_least(estimate - 1u, exponent, value)) --estimate;
  if (estimate > UINT32_MAX) fail("derived decomposition base exceeds uint32");
  return (uint32_t)estimate;
}

static void derive_bases(
  const GenerateOptions* options,
  double beta,
  uint32_t* base1,
  uint32_t* base2,
  uint32_t* base3)
{
  if (options->recursions == 1) {
    double threshold = 15.0 * beta * sqrt((double)options->r);
    uint64_t base = (uint64_t)ceil(sqrt(threshold)) + 1u;
    if (base < 2) base = 2;
    if (base > UINT32_MAX) fail("derived base-case decomposition base exceeds uint32");
    *base1 = *base2 = *base3 = (uint32_t)base;
    return;
  }

  double coefficient_sd = beta / sqrt((double)(options->r * options->n * LAB_DEGREE));
  double base_real = sqrt(coefficient_sd * sqrt(12.0 * (double)options->r * 71.0));
  uint32_t fold_base = (uint32_t)floor(base_real + 0.5);
  if (fold_base < 2) fold_base = 2;
  uint64_t t1 = (uint64_t)floor(log((double)LAB_Q) / log((double)fold_base) + 0.5);
  if (t1 < 2) t1 = 2;
  uint32_t first_base = ceil_nth_root_u64(LAB_Q, t1);

  double garbage_scale = sqrt(24.0 * (double)options->n * LAB_DEGREE) * coefficient_sd * coefficient_sd;
  double logarithmic_t2 = garbage_scale > 1.0 ? log(garbage_scale) / log((double)fold_base) : 0.0;
  uint64_t t2 = (uint64_t)floor(fmax(0.0, logarithmic_t2) + 0.5);
  if (t2 < 2) t2 = 2;
  uint64_t second = (uint64_t)floor(exp(log(fmax(1.0, garbage_scale)) / (double)t2) + 0.5);
  if (second < 2) second = 2;
  if (second > UINT32_MAX) fail("derived garbage decomposition base exceeds uint32");

  *base1 = first_base;
  *base2 = (uint32_t)second;
  *base3 = first_base;
}

static void validate_generate_options(const GenerateOptions* options)
{
  if (options->r == 0 || options->n == 0 ||
      options->r > LAB_MAX_DIMENSION || options->n > LAB_MAX_DIMENSION)
    fail("r and n must be in [1,128]");
  if (options->r > LAB_MAX_WITNESS_POLYS / options->n)
    fail("r*n exceeds the C frontend safety limit of 4096 polynomials");
  if (options->witness_bound > LAB_MAX_WITNESS_BOUND)
    fail("witness-bound exceeds the C frontend safety limit of 10000");
  if (options->recursions == 0 || options->recursions > 8)
    fail("recursions must be in [1,8]");
  if (options->seed_len == 0) fail("seed must not be empty");
  if (strlen(options->fingerprint) != 64) fail("fingerprint must contain 64 lowercase hexadecimal characters");
  for (size_t i = 0; i < 64; ++i) {
    char c = options->fingerprint[i];
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
      fail("fingerprint must contain 64 lowercase hexadecimal characters");
  }
}

static Buffer build_bundle(const GenerateOptions* options, uint8_t public_digest[32])
{
  validate_generate_options(options);
  size_t witness_count = (size_t)(options->r * options->n);
  size_t witness_values = 0;
  if (!checked_mul_size(witness_count, LAB_DEGREE, &witness_values)) fail("witness size overflow");
  int64_t* witness = (int64_t*)calloc(witness_values, sizeof(int64_t));
  if (witness == NULL) fail("out of memory while generating witness");

  ShakeRng rng = {0, options->seed, options->seed_len};
  uint64_t span = 2u * options->witness_bound + 1u;
  uint64_t squared_norm = 0;
  for (size_t i = 0; i < witness_values; ++i) {
    int64_t coefficient = (int64_t)rng_below(&rng, span) - (int64_t)options->witness_bound;
    witness[i] = coefficient;
    uint64_t magnitude = coefficient < 0 ? (uint64_t)(-coefficient) : (uint64_t)coefficient;
    squared_norm += magnitude * magnitude;
  }
  double beta = (double)(integer_sqrt(squared_norm) + 2u);
  uint32_t base1 = 0, base2 = 0, base3 = 0;
  derive_bases(options, beta, &base1, &base2, &base3);

  int64_t equality_eval[LAB_DEGREE];
  int64_t constant_eval[LAB_DEGREE];
  poly_zero(equality_eval);
  poly_zero(constant_eval);

  size_t eq_j = options->r > 1 ? 1u : 0u;
  size_t eq_linear_row = options->r > 1 ? 1u : 0u;
  size_t eq_linear_column = options->n > 1 ? 1u : 0u;
  size_t cz_j = (size_t)options->r - 1u;
  size_t cz_linear = ((size_t)options->r - 1u) * (size_t)options->n + ((size_t)options->n - 1u);

  int64_t product[LAB_DEGREE];
  for (size_t column = 0; column < (size_t)options->n; ++column) {
    const int64_t* left = witness + column * LAB_DEGREE;
    const int64_t* eq_right = witness + (eq_j * (size_t)options->n + column) * LAB_DEGREE;
    const int64_t* cz_right = witness + (cz_j * (size_t)options->n + column) * LAB_DEGREE;
    negacyclic_mul(left, eq_right, product);
    poly_add(equality_eval, product);
    negacyclic_mul(left, cz_right, product);
    for (size_t k = 0; k < LAB_DEGREE; ++k) {
      size_t destination = k + 1u;
      if (destination == LAB_DEGREE) constant_eval[0] -= product[k];
      else constant_eval[destination] += product[k];
    }
  }

  const int64_t* eq_witness =
    witness + (eq_linear_row * (size_t)options->n + eq_linear_column) * LAB_DEGREE;
  equality_eval[0] += eq_witness[0] + eq_witness[LAB_DEGREE - 1u];
  for (size_t k = 1; k < LAB_DEGREE; ++k)
    equality_eval[k] += eq_witness[k] - eq_witness[k - 1u];

  const int64_t* cz_witness = witness + cz_linear * LAB_DEGREE;
  for (size_t k = 0; k < LAB_DEGREE; ++k) constant_eval[k] -= 2 * cz_witness[k];
  constant_eval[LAB_DEGREE - 1u] += cz_witness[0];
  for (size_t k = 0; k + 1u < LAB_DEGREE; ++k) constant_eval[k] -= cz_witness[k + 1u];

  uint8_t ajtai_seed[32], oracle_seed[32];
  shake_with_domain("LNPLAB/AJTAI/v1", options->seed, options->seed_len, NULL, 0, ajtai_seed, 32);
  shake_with_domain("LNPLAB/ORACLE/v1", options->seed, options->seed_len, NULL, 0, oracle_seed, 32);

  Buffer output = {0};
  buffer_bytes(&output, LAB_MAGIC, sizeof(LAB_MAGIC));
  buffer_u32(&output, 1);
  buffer_u32(&output, LAB_DEGREE);
  buffer_u64(&output, LAB_Q);
  buffer_u64(&output, options->r);
  buffer_u64(&output, options->n);
  buffer_f64(&output, beta);
  buffer_u64(&output, LAB_RANK);
  buffer_u64(&output, LAB_RANK);
  buffer_u64(&output, LAB_RANK);
  buffer_u32(&output, base1);
  buffer_u32(&output, base2);
  buffer_u32(&output, base3);
  buffer_u64(&output, LAB_JL_ROWS);
  buffer_u64(&output, LAB_AGGREGATION_ROUNDS);
  buffer_u64(&output, options->recursions);
  buffer_lp(&output, LAB_MODE, strlen(LAB_MODE));
  buffer_lp(&output, options->fingerprint, strlen(options->fingerprint));
  buffer_lp(&output, ajtai_seed, sizeof(ajtai_seed));
  buffer_lp(&output, oracle_seed, sizeof(oracle_seed));

  int64_t sparse[LAB_DEGREE];
  size_t rr = (size_t)(options->r * options->r);
  size_t rn = witness_count;
  buffer_u64(&output, 1);
  for (size_t index = 0; index < rr; ++index) {
    poly_zero(sparse);
    if (index == eq_j) sparse[0] = 1;
    buffer_poly(&output, sparse);
  }
  size_t eq_phi_index = eq_linear_row * (size_t)options->n + eq_linear_column;
  for (size_t index = 0; index < rn; ++index) {
    poly_zero(sparse);
    if (index == eq_phi_index) {
      sparse[0] = 1;
      sparse[1] = -1;
    }
    buffer_poly(&output, sparse);
  }
  for (size_t i = 0; i < LAB_DEGREE; ++i) sparse[i] = centered_negate(equality_eval[i]);
  buffer_poly(&output, sparse);

  buffer_u64(&output, 1);
  size_t cz_a_index = cz_j;
  for (size_t index = 0; index < rr; ++index) {
    poly_zero(sparse);
    if (index == cz_a_index) sparse[1] = 1;
    buffer_poly(&output, sparse);
  }
  for (size_t index = 0; index < rn; ++index) {
    poly_zero(sparse);
    if (index == cz_linear) {
      sparse[0] = -2;
      sparse[LAB_DEGREE - 1u] = 1;
    }
    buffer_poly(&output, sparse);
  }
  buffer_i64(&output, centered_negate(constant_eval[0]));

  sha3_256(output.data, output.size, public_digest);
  buffer_bytes(&output, public_digest, 32);
  buffer_u64(&output, (uint64_t)witness_count);
  for (size_t i = 0; i < witness_count; ++i)
    buffer_poly(&output, witness + i * LAB_DEGREE);

  free(witness);
  return output;
}

static const uint8_t* reader_bytes(Reader* reader, size_t count, const char* label)
{
  if (count > reader->size - reader->offset) {
    fprintf(stderr, "lab_c: error: truncated %s at byte %zu\n", label, reader->offset);
    exit(2);
  }
  const uint8_t* result = reader->data + reader->offset;
  reader->offset += count;
  return result;
}

static uint32_t reader_u32(Reader* reader, const char* label)
{
  const uint8_t* bytes = reader_bytes(reader, 4, label);
  uint32_t result = 0;
  for (unsigned i = 0; i < 4; ++i) result |= (uint32_t)bytes[i] << (8u * i);
  return result;
}

static uint64_t reader_u64(Reader* reader, const char* label)
{
  const uint8_t* bytes = reader_bytes(reader, 8, label);
  return load_u64_le(bytes);
}

static double reader_f64(Reader* reader, const char* label)
{
  uint64_t bits = reader_u64(reader, label);
  double result = 0.0;
  memcpy(&result, &bits, sizeof(result));
  return result;
}

static const uint8_t* reader_lp(Reader* reader, size_t maximum, size_t* length, const char* label)
{
  uint64_t encoded = reader_u64(reader, label);
  if (encoded > maximum || encoded > SIZE_MAX) fail("length-prefixed field exceeds its safety limit");
  *length = (size_t)encoded;
  return reader_bytes(reader, *length, label);
}

static uint64_t signed_magnitude_from_bits(uint64_t bits)
{
  return (bits >> 63u) == 0 ? bits : (~bits) + 1u;
}

static void validate_coefficients(
  Reader* reader,
  size_t coefficient_count,
  bool witness,
  long double* squared_norm,
  const char* label)
{
  for (size_t i = 0; i < coefficient_count; ++i) {
    uint64_t bits = reader_u64(reader, label);
    uint64_t magnitude = signed_magnitude_from_bits(bits);
    if (magnitude > LAB_Q_HALF) fail("non-canonical centered polynomial coefficient");
    if (witness) {
      if (magnitude >= LAB_SQRT_Q_FLOOR)
        fail("witness coefficient violates the ICICLE norm-kernel range");
      long double value = (long double)magnitude;
      *squared_norm += value * value;
    }
  }
}

static void print_hex(const uint8_t* bytes, size_t size)
{
  for (size_t i = 0; i < size; ++i) printf("%02x", (unsigned)bytes[i]);
}

static void inspect_bytes(const uint8_t* bytes, size_t size, bool quiet)
{
  if (size == 0 || size > (size_t)LAB_MAX_FILE_BYTES) fail("invalid relation bundle size");
  Reader reader = {bytes, size, 0};
  if (memcmp(reader_bytes(&reader, 8, "magic"), LAB_MAGIC, 8) != 0) fail("bad magic (expected LNPLAB01)");
  uint32_t version = reader_u32(&reader, "version");
  uint32_t degree = reader_u32(&reader, "degree");
  uint64_t modulus = reader_u64(&reader, "modulus");
  uint64_t r = reader_u64(&reader, "r");
  uint64_t n = reader_u64(&reader, "n");
  double beta = reader_f64(&reader, "beta");
  uint64_t kappa = reader_u64(&reader, "kappa");
  uint64_t kappa1 = reader_u64(&reader, "kappa1");
  uint64_t kappa2 = reader_u64(&reader, "kappa2");
  uint32_t base1 = reader_u32(&reader, "base1");
  uint32_t base2 = reader_u32(&reader, "base2");
  uint32_t base3 = reader_u32(&reader, "base3");
  uint64_t jl_rows = reader_u64(&reader, "JL rows");
  uint64_t rounds = reader_u64(&reader, "aggregation rounds");
  uint64_t recursions = reader_u64(&reader, "recursions");
  if (version != 1 || degree != LAB_DEGREE || modulus != LAB_Q || r == 0 || n == 0 ||
      !isfinite(beta) || beta <= 0 || kappa == 0 || kappa1 == 0 || kappa2 == 0 ||
      base1 < 2 || base2 < 2 || base3 < 2 || jl_rows != LAB_JL_ROWS ||
      rounds != LAB_AGGREGATION_ROUNDS || recursions == 0 || recursions > 8)
    fail("invalid fixed relation parameters");
  if (r > SIZE_MAX / n) fail("r*n overflows size_t");
  size_t rn = (size_t)(r * n);
  if (r > SIZE_MAX / r) fail("r*r overflows size_t");
  size_t rr = (size_t)(r * r);
  size_t mode_len = 0, fingerprint_len = 0, ajtai_len = 0, oracle_len = 0;
  const uint8_t* mode = reader_lp(&reader, 128, &mode_len, "mode");
  const uint8_t* fingerprint = reader_lp(&reader, 128, &fingerprint_len, "fingerprint");
  (void)reader_lp(&reader, 1024 * 1024, &ajtai_len, "Ajtai seed");
  (void)reader_lp(&reader, 1024 * 1024, &oracle_len, "oracle seed");
  bool synthetic_mode = mode_len == strlen(LAB_MODE) && memcmp(mode, LAB_MODE, mode_len) == 0;
  static const char json_mode[] = "json-principal-v1";
  bool supplied_mode = mode_len == strlen(json_mode) && memcmp(mode, json_mode, mode_len) == 0;
  if (!synthetic_mode && !supplied_mode)
    fail("unsupported relation mode");
  if (fingerprint_len != 64 || ajtai_len < 16 || oracle_len < 16)
    fail("invalid fingerprint or seed length");
  for (size_t i = 0; i < fingerprint_len; ++i) {
    uint8_t c = fingerprint[i];
    if (!((c >= (uint8_t)'0' && c <= (uint8_t)'9') ||
          (c >= (uint8_t)'a' && c <= (uint8_t)'f')))
      fail("fingerprint is not lowercase hexadecimal");
  }

  if (recursions > 1) {
    GenerateOptions schedule = {r, n, 0, recursions, (const uint8_t*)"x", 1, LAB_DEFAULT_FINGERPRINT};
    uint32_t expected1 = 0, expected2 = 0, expected3 = 0;
    derive_bases(&schedule, beta, &expected1, &expected2, &expected3);
    if (base1 != expected1 || base2 != expected2 || base3 != expected3)
      fail("recursive bases do not match the Section 5.4 schedule");
  }

  size_t polynomial_bytes = LAB_DEGREE * sizeof(int64_t);
  uint64_t equality_count = reader_u64(&reader, "equality count");
  size_t per_equality = 0, term_count = 0;
  if (!checked_add_size(rr, rn, &term_count) || !checked_add_size(term_count, 1, &term_count) ||
      !checked_mul_size(term_count, polynomial_bytes, &per_equality) ||
      equality_count > (uint64_t)((reader.size - reader.offset) / (per_equality == 0 ? 1 : per_equality)))
    fail("equality section size overflow");
  size_t equality_coefficients = ((size_t)equality_count * per_equality) / sizeof(int64_t);
  validate_coefficients(&reader, equality_coefficients, false, NULL, "equality constraints");

  uint64_t const_zero_count = reader_u64(&reader, "const-zero count");
  size_t per_const = 0;
  if (!checked_add_size(rr, rn, &term_count) ||
      !checked_mul_size(term_count, polynomial_bytes, &per_const) ||
      !checked_add_size(per_const, sizeof(int64_t), &per_const) ||
      const_zero_count > (uint64_t)((reader.size - reader.offset) / (per_const == 0 ? 1 : per_const)))
    fail("const-zero section size overflow");
  size_t const_coefficients = ((size_t)const_zero_count * per_const) / sizeof(int64_t);
  validate_coefficients(&reader, const_coefficients, false, NULL, "const-zero constraints");
  if (equality_count + const_zero_count == 0) fail("relation has no constraints");

  size_t public_end = reader.offset;
  const uint8_t* stored_digest = reader_bytes(&reader, 32, "public digest");
  uint8_t computed_digest[32];
  sha3_256(bytes, public_end, computed_digest);
  if (memcmp(stored_digest, computed_digest, 32) != 0) fail("public SHA3-256 digest mismatch");
  uint64_t witness_count = reader_u64(&reader, "witness count");
  if (witness_count != (uint64_t)rn) fail("witness count is not r*n");
  size_t witness_coefficients = 0;
  if (!checked_mul_size(rn, LAB_DEGREE, &witness_coefficients)) fail("witness size overflow");
  long double squared_norm = 0.0L;
  validate_coefficients(&reader, witness_coefficients, true, &squared_norm, "witness");
  if (!(sqrtl(squared_norm) < (long double)beta)) fail("witness norm is not strictly below beta");
  if (reader.offset != reader.size) fail("trailing bytes after witness");

  if (!quiet) {
    printf("LNPLAB01/v1: r=%" PRIu64 ", n=%" PRIu64 ", beta=%.17g, bases=%" PRIu32 "/%" PRIu32
           "/%" PRIu32 ", recursions=%" PRIu64 "\n",
           r, n, beta, base1, base2, base3, recursions);
    printf("mode=%.*s, fingerprint=%.*s\n", (int)mode_len, (const char*)mode,
           (int)fingerprint_len, (const char*)fingerprint);
    printf("public_sha3_256=");
    print_hex(stored_digest, 32);
    printf("\nartifact_bytes=%zu, witness_plaintext=yes, proof_file=no\n", size);
  }
}

static Buffer read_file(const char* path)
{
  FILE* file = fopen(path, "rb");
  if (file == NULL) fail_errno("cannot open", path);
  if (fseek(file, 0, SEEK_END) != 0) fail_errno("cannot seek", path);
  long end = ftell(file);
  if (end < 0) fail_errno("cannot determine size of", path);
  if ((uint64_t)end > LAB_MAX_FILE_BYTES) fail("input exceeds the 128 MiB safety limit");
  if (fseek(file, 0, SEEK_SET) != 0) fail_errno("cannot seek", path);
  Buffer result = {0};
  buffer_reserve(&result, (size_t)end);
  result.size = (size_t)end;
  if (result.size != 0 && fread(result.data, 1, result.size, file) != result.size)
    fail_errno("cannot read", path);
  if (fclose(file) != 0) fail_errno("cannot close", path);
  return result;
}

static bool file_exists(const char* path)
{
  FILE* file = fopen(path, "rb");
  if (file == NULL) return false;
  fclose(file);
  return true;
}

static void write_file(const char* path, const Buffer* bytes, bool force)
{
  if (!force && file_exists(path)) fail("output already exists (pass --force to replace it)");
  FILE* file = fopen(path, "wb");
  if (file == NULL) fail_errno("cannot create", path);
  if (bytes->size != 0 && fwrite(bytes->data, 1, bytes->size, file) != bytes->size)
    fail_errno("cannot write", path);
  if (fflush(file) != 0 || fclose(file) != 0) fail_errno("cannot finish", path);
}

static uint64_t parse_u64(const char* text, const char* option)
{
  if (text == NULL || text[0] == '\0' || text[0] == '-') {
    fprintf(stderr, "lab_c: error: %s requires a non-negative integer\n", option);
    exit(2);
  }
  errno = 0;
  char* end = NULL;
  uintmax_t value = strtoumax(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value > UINT64_MAX) {
    fprintf(stderr, "lab_c: error: invalid integer for %s\n", option);
    exit(2);
  }
  return (uint64_t)value;
}

static bool bytes_equal_hex(const uint8_t* bytes, size_t size, const char* hex)
{
  static const char alphabet[] = "0123456789abcdef";
  if (strlen(hex) != size * 2u) return false;
  for (size_t i = 0; i < size; ++i)
    if (hex[2u * i] != alphabet[bytes[i] >> 4u] || hex[2u * i + 1u] != alphabet[bytes[i] & 15u])
      return false;
  return true;
}

static void self_test(void)
{
  uint8_t digest[32];
  sha3_256(NULL, 0, digest);
  if (!bytes_equal_hex(digest, 32, "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"))
    fail("SHA3-256 self-test failed");
  shake256(NULL, 0, digest, 32);
  if (!bytes_equal_hex(digest, 32, "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f"))
    fail("SHAKE256 self-test failed");

  static const uint8_t seed[] = "recursive-smoke";
  GenerateOptions options = {1, 1, 10, 2, seed, sizeof(seed) - 1u, LAB_DEFAULT_FINGERPRINT};
  Buffer bundle = build_bundle(&options, digest);
  if (bundle.size != 3426u) fail("canonical fixture has the wrong size");
  if (!bytes_equal_hex(digest, 32, "afa935f54d34c8ff55dbcc7c1fd90587c289bf95b9724900aa2c580cea96269b"))
    fail("canonical fixture has the wrong public digest");
  inspect_bytes(bundle.data, bundle.size, true);
  free(bundle.data);
  printf("lab_c self-test: PASS\n");
}

static void usage(FILE* stream)
{
  fprintf(stream,
    "Usage:\n"
    "  lab_c generate [--output FILE] [--r N] [--n N] [--witness-bound N]\n"
    "                 [--seed TEXT] [--recursions N] [--fingerprint HEX] [--force]\n"
    "  lab_c inspect FILE\n"
    "  lab_c self-test\n\n"
    "If --output is omitted, generate writes ./input.lab.\n"
    "The generated .lab file contains the secret witness in plaintext. It is a\n"
    "synthetic relation for the C++ lab_runner, not a serialized proof.\n");
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    usage(stderr);
    return 2;
  }
  if (strcmp(argv[1], "self-test") == 0) {
    if (argc != 2) fail("self-test takes no arguments");
    self_test();
    return 0;
  }
  if (strcmp(argv[1], "inspect") == 0) {
    if (argc != 3) fail("inspect requires exactly one FILE argument");
    Buffer input = read_file(argv[2]);
    inspect_bytes(input.data, input.size, false);
    free(input.data);
    return 0;
  }
  if (strcmp(argv[1], "generate") == 0) {
    static const uint8_t default_seed[] = "recursive-smoke";
    GenerateOptions options = {
      1, 1, 10, 2, default_seed, sizeof(default_seed) - 1u, LAB_DEFAULT_FINGERPRINT
    };
    const char* output_path = "input.lab";
    bool force = false;
    for (int i = 2; i < argc; ++i) {
      const char* argument = argv[i];
      if (strcmp(argument, "--force") == 0) {
        force = true;
      } else if (strcmp(argument, "--output") == 0 || strcmp(argument, "-o") == 0) {
        if (++i >= argc) fail("--output requires a path");
        output_path = argv[i];
      } else if (strcmp(argument, "--r") == 0) {
        if (++i >= argc) fail("--r requires an integer");
        options.r = parse_u64(argv[i], "--r");
      } else if (strcmp(argument, "--n") == 0) {
        if (++i >= argc) fail("--n requires an integer");
        options.n = parse_u64(argv[i], "--n");
      } else if (strcmp(argument, "--witness-bound") == 0) {
        if (++i >= argc) fail("--witness-bound requires an integer");
        options.witness_bound = parse_u64(argv[i], "--witness-bound");
      } else if (strcmp(argument, "--recursions") == 0) {
        if (++i >= argc) fail("--recursions requires an integer");
        options.recursions = parse_u64(argv[i], "--recursions");
      } else if (strcmp(argument, "--seed") == 0) {
        if (++i >= argc) fail("--seed requires text");
        options.seed = (const uint8_t*)argv[i];
        options.seed_len = strlen(argv[i]);
      } else if (strcmp(argument, "--fingerprint") == 0) {
        if (++i >= argc) fail("--fingerprint requires hexadecimal text");
        options.fingerprint = argv[i];
      } else if (strcmp(argument, "--help") == 0 || strcmp(argument, "-h") == 0) {
        usage(stdout);
        return 0;
      } else {
        fprintf(stderr, "lab_c: error: unknown option '%s'\n", argument);
        return 2;
      }
    }
    uint8_t digest[32];
    Buffer bundle = build_bundle(&options, digest);
    inspect_bytes(bundle.data, bundle.size, true);
    write_file(output_path, &bundle, force);
    printf("Generated %s (%zu bytes): r=%" PRIu64 ", n=%" PRIu64 ", recursions=%" PRIu64 "\n",
           output_path, bundle.size, options.r, options.n, options.recursions);
    printf("public_sha3_256=");
    print_hex(digest, 32);
    printf("\nWARNING: plaintext secret witness; relation bundle only, not a proof.\n");
    free(bundle.data);
    return 0;
  }
  if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
    usage(stdout);
    return 0;
  }
  fprintf(stderr, "lab_c: error: unknown command '%s'\n", argv[1]);
  usage(stderr);
  return 2;
}
