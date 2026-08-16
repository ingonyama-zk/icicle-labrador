#include "icicle/backend/vec_ops_backend.h"
#include "icicle/backend/cpu/challenge_keccak_xof.h"
#include "icicle/hash/keccak.h"
#include "icicle/rings/id.h"
#include "taskflow/taskflow.hpp"
#include "icicle/operator_norm.h"

// extract the number of threads to run from config
int get_nof_workers(const VecOpsConfig& config); // defined in cpu_vec_ops.cpp

namespace {

uint64_t load_u64_le(const std::byte* input)
{
  uint64_t value = 0;
  for (unsigned shift = 0; shift < 64; shift += 8) {
    value |= uint64_t{std::to_integer<uint8_t>(input[shift / 8])} << shift;
  }
  return value;
}

void store_u32_le(std::byte* output, uint32_t value)
{
  for (unsigned shift = 0; shift < 32; shift += 8) {
    output[shift / 8] = static_cast<std::byte>((value >> shift) & 0xff);
  }
}

void store_u64_le(std::byte* output, uint64_t value)
{
  for (unsigned shift = 0; shift < 64; shift += 8) {
    output[shift / 8] = static_cast<std::byte>((value >> shift) & 0xff);
  }
}

#if RING_ID == LABRADOR_Q40
// Map an XOF word uniformly to the direct q40 coefficient ring.  Reducing a
// uniform 64-bit integer modulo q is detectably biased because q is only 40
// bits.  Masking to 40 bits and rejecting values above q gives the exact
// uniform distribution.  Rejected words are replaced from a separately
// domain-separated, canonically encoded Keccak counter stream.
field_t sample_labrador_q40_coefficient(
  uint64_t word,
  const Hash& keccak512,
  const std::byte* seed,
  size_t seed_len,
  uint32_t batch,
  size_t element_index)
{
  static_assert(field_t::NBITS == 40, "labradorq40 sampler expects a 40-bit modulus");
  static_assert(field_t::TLC == 2, "labradorq40 sampler expects two 32-bit limbs");
  constexpr uint64_t mask = (uint64_t{1} << field_t::NBITS) - 1;
  constexpr auto modulus_storage = field_t::get_modulus();
  constexpr uint64_t modulus =
    uint64_t{modulus_storage.limbs[0]} | (uint64_t{modulus_storage.limbs[1]} << 32);

  uint64_t candidate = word & mask;
  if (candidate < modulus) { return field_t::from_u64(candidate); }

  icicle::cpu::Keccak512CounterXof retry_stream(
    keccak512, "icicle.ring.uniform-rejection.v1", seed, seed_len,
    {static_cast<uint64_t>(batch), static_cast<uint64_t>(element_index)});
  while (candidate >= modulus) {
    candidate = retry_stream.next_u64_le() & mask;
  }
  return field_t::from_u64(candidate);
}
#endif

} // namespace

void fast_mode_random_sampling(
  size_t size, const std::byte* seed, size_t seed_len, const VecOpsConfig& cfg, field_t* output)
{
  // Use keccak to get deterministic uniform distribution
  auto keccak512 = Keccak512::create();
  const size_t element_size = sizeof(field_t::limbs_storage);
  const size_t elements_per_hash = size_t(std::max(keccak512.output_size() / element_size, uint64_t(1)));
  // To support elements that are larger than 32 bytes
  const size_t hashes_per_element =
    size_t(std::max((element_size + keccak512.output_size() - 1) / keccak512.output_size(), uint64_t(1)));
  const size_t size_per_task =
    (size + RANDOM_SAMPLING_FAST_MODE_NUMBER_OF_TASKS - 1) / RANDOM_SAMPLING_FAST_MODE_NUMBER_OF_TASKS;
  const size_t total_tasks = (size + size_per_task - 1) / size_per_task;

  tf::Taskflow taskflow;
  tf::Executor executor(get_nof_workers(cfg));
  for (uint32_t b = 0; b < cfg.batch_size; ++b) {
    field_t* batch_output = output + b * size;
    HashConfig hash_cfg{};
    for (uint64_t t = 0; t < std::min(total_tasks, static_cast<size_t>(RANDOM_SAMPLING_FAST_MODE_NUMBER_OF_TASKS));
         ++t) {
      taskflow.emplace([=]() {
        std::vector<std::byte> hash_input(seed_len + sizeof(b) + sizeof(uint64_t));
        std::memcpy(hash_input.data(), seed, seed_len);
        store_u32_le(hash_input.data() + seed_len, b);
        store_u64_le(hash_input.data() + seed_len + sizeof(b), t);
        std::vector<uint64_t> hash_output(keccak512.output_size() / sizeof(uint64_t) * hashes_per_element);

        keccak512.hash(
          hash_input.data(), hash_input.size(), hash_cfg, reinterpret_cast<std::byte*>(hash_output.data()));
        for (int i = 1; i < hashes_per_element; i++) {
          keccak512.hash(
            reinterpret_cast<std::byte*>(hash_output.data()) + (i - 1) * keccak512.output_size(),
            keccak512.output_size(), hash_cfg,
            reinterpret_cast<std::byte*>(hash_output.data()) + i * keccak512.output_size());
        }

#if RING_ID == LABRADOR_Q40
        field_t prev_element = sample_labrador_q40_coefficient(
          load_u64_le(reinterpret_cast<const std::byte*>(hash_output.data())), keccak512, seed, seed_len, b,
          t * size_per_task);
#else
        field_t prev_element = field_t::reduce_from_bytes(reinterpret_cast<std::byte*>(hash_output.data()));
#endif
        batch_output[t * size_per_task] = prev_element;
        for (int i = 1; i < size_per_task && (t * size_per_task + i) < size; i++) {
          field_t next_element = prev_element.sqr();
          prev_element = next_element;
          batch_output[t * size_per_task + i] = next_element;
        }
      });
    }
    executor.run(taskflow).wait();
    taskflow.clear();
  }
}

void slow_mode_random_sampling(
  size_t size, const std::byte* seed, size_t seed_len, const VecOpsConfig& cfg, field_t* output)
{
  // Use keccak to get deterministic uniform distribution
  auto keccak512 = Keccak512::create();
  const size_t element_size = sizeof(field_t::limbs_storage);
  const size_t elements_per_hash = size_t(std::max(keccak512.output_size() / element_size, uint64_t(1)));
  // To support elements that are larger than 32 bytes
  const size_t hashes_per_element = size_t(std::max(element_size / keccak512.output_size(), uint64_t(1)));
  const size_t hashes_per_batch = std::max((size + elements_per_hash - 1) / elements_per_hash, size_t(1));

  const int nof_workers = std::min((int)(hashes_per_batch), get_nof_workers(cfg));
  const size_t hashes_per_worker = (hashes_per_batch + nof_workers - 1) / nof_workers;

  tf::Taskflow taskflow;
  tf::Executor executor(nof_workers);
  for (uint32_t b = 0; b < cfg.batch_size; ++b) {
    field_t* batch_output = output + b * size;
    for (uint32_t w = 0; w < nof_workers; ++w) {
      taskflow.emplace([=]() {
        HashConfig hash_cfg{};
        std::vector<std::byte> hash_input(seed_len + sizeof(uint32_t) + sizeof(uint64_t));
        std::memcpy(hash_input.data(), seed, seed_len);
        store_u32_le(hash_input.data() + seed_len, b);
        std::vector<uint64_t> hash_output(keccak512.output_size() / sizeof(uint64_t) * hashes_per_element);
        for (size_t counter = w * hashes_per_worker;
             counter < (w + 1) * hashes_per_worker && counter < hashes_per_batch; counter++) {
          store_u64_le(hash_input.data() + seed_len + sizeof(uint32_t), counter);

          keccak512.hash(
            hash_input.data(), hash_input.size(), hash_cfg, reinterpret_cast<std::byte*>(hash_output.data()));
          for (int i = 1; i < hashes_per_element; i++) {
            keccak512.hash(
              reinterpret_cast<std::byte*>(hash_output.data()) + (i - 1) * keccak512.output_size(),
              keccak512.output_size(), hash_cfg,
              reinterpret_cast<std::byte*>(hash_output.data()) + i * keccak512.output_size());
          }
          for (int i = 0; i < elements_per_hash && counter * elements_per_hash + i < size; i++) {
#if RING_ID == LABRADOR_Q40
            const size_t element_index = counter * elements_per_hash + i;
            batch_output[element_index] = sample_labrador_q40_coefficient(
              load_u64_le(reinterpret_cast<const std::byte*>(hash_output.data()) + i * element_size), keccak512,
              seed, seed_len, b, element_index);
#else
            batch_output[counter * elements_per_hash + i] =
              field_t::reduce_from_bytes(reinterpret_cast<std::byte*>(hash_output.data()) + i * element_size);
#endif
          }
        }
      });
    }
    executor.run(taskflow).wait();
    taskflow.clear();
  }
}

eIcicleError cpu_random_sampling(
  const Device& device,
  size_t size,
  bool fast_mode,
  const std::byte* seed,
  size_t seed_len,
  const VecOpsConfig& cfg,
  field_t* output)
{
  if (!seed || !output) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer.";
    return eIcicleError::INVALID_POINTER;
  }

  if (seed_len == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: zero seed length.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (size == 0) { return eIcicleError::SUCCESS; }

  if (fast_mode) {
    fast_mode_random_sampling(size, seed, seed_len, cfg, output);
  } else {
    slow_mode_random_sampling(size, seed, seed_len, cfg, output);
  }

  return eIcicleError::SUCCESS;
}

REGISTER_RING_ZQ_RANDOM_SAMPLING_BACKEND("CPU", cpu_random_sampling);

using RandomBitIterator = icicle::cpu::Keccak512CounterXof;

uint32_t ceil_log2_u32(uint32_t value)
{
  if (value <= 1) { return 0; }
  --value;
  uint32_t result = 0;
  while (value != 0) {
    ++result;
    value >>= 1;
  }
  return result;
}

// Cross shuffles two adjacent ranges of an array as described in the paper
// https://arxiv.org/pdf/1508.03167
template <typename T>
void merge_shuffle(
  T* array, uint32_t size_a, uint32_t size_b, uint32_t index_bits, RandomBitIterator& random_bit_iterator)
{
  uint32_t i = 0;
  uint32_t j = size_a;
  const uint32_t n = size_a + size_b;
  while (true) {
    if (!random_bit_iterator.next_bit()) {
      if (j == n) { break; }
      std::swap(array[i], array[j]);
      ++j;
    } else {
      if (i == j) { break; }
    }
    ++i;
  }
  for (; i < n; i++) {
    uint32_t m;
    do {
      m = 0;
      for (uint32_t b = 0; b < index_bits; b++) {
        m |= random_bit_iterator.next_bit();
        if (b < index_bits - 1) { m <<= 1; }
      }
    } while (m > i);
    std::swap(array[i], array[m]);
  }
}

eIcicleError cpu_challenge_space_polynomials_sampling(
  const Device& device,
  const std::byte* seed,
  size_t seed_len,
  size_t size,
  uint32_t ones,
  uint32_t twos,
  int64_t norm,
  const VecOpsConfig& cfg,
  Rq* output)
{
  if (!seed || !output) {
    ICICLE_LOG_ERROR << "Invalid argument: null pointer.";
    return eIcicleError::INVALID_POINTER;
  }

  if (seed_len == 0) {
    ICICLE_LOG_ERROR << "Invalid argument: zero seed length.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (ones + twos > Rq::d) {
    ICICLE_LOG_ERROR << "Invalid argument: number of coefficients > polynomial degree.";
    return eIcicleError::INVALID_ARGUMENT;
  }

  if (size == 0) { return eIcicleError::SUCCESS; }

  auto keccak512 = Keccak512::create();

  static const field_t two = field_t::one() + field_t::one();
  static const field_t neg_two = two.neg();
  static const field_t neg_one = field_t::one().neg();

  const size_t nof_workers = std::min((size_t)get_nof_workers(cfg), size);

  tf::Taskflow taskflow;
  tf::Executor executor(nof_workers);

  static const std::unordered_map<field_t, int64_t> balanced_table = {
    {field_t::one(), 1}, {neg_one, -1}, {two, 2}, {neg_two, -2}, {field_t::zero(), 0},
  };

  for (size_t poly_idx = 0; poly_idx < size; poly_idx++) {
    taskflow.emplace([=]() {
      uint32_t retry_idx = 0;
      Rq* output_polynomial = output + poly_idx;
      int64_t opnorm = 0;

      do {
        RandomBitIterator random_bit_iterator(
          keccak512, "icicle.ring.challenge-space.v1", seed, seed_len,
          {static_cast<uint64_t>(poly_idx), static_cast<uint64_t>(retry_idx)});

        // Initialize polynomial with coefficients and randomly flip signs of ones and twos coefficients
        // [1, -1, ..., 1, 2, -2, ..., 2, 0, ..., 0]
        for (uint32_t l = 0; l < ones; ++l) {
          output_polynomial->values[l] = random_bit_iterator.next_bit() ? field_t::one() : neg_one;
        }
        for (uint32_t m = ones; m < ones + twos; ++m) {
          output_polynomial->values[m] = random_bit_iterator.next_bit() ? two : neg_two;
        }
        // TODO: memset here?
        for (uint32_t k = ones + twos; k < Rq::d; ++k) {
          output_polynomial->values[k] = field_t::zero();
        }

        // Do merge shuffle of 1s and 2s
        merge_shuffle(
          output_polynomial->values, ones, twos, ceil_log2_u32(ones + twos), random_bit_iterator);
        // Do merge shuffle of shuffled 1s and 2s and zeroes
        merge_shuffle(
          output_polynomial->values, ones + twos, Rq::d - ones - twos, ceil_log2_u32(Rq::d),
          random_bit_iterator);

        if (norm) {
          opnorm::Poly poly{};
          for (int i = 0; i < Rq::d; ++i) {
            poly[i] = balanced_table.at(output_polynomial->values[i]);
          }
          opnorm = opnorm::operator_norm(poly);
          retry_idx++;
          ICICLE_ASSERT(retry_idx <= 0xFFFFFF);
        }
      } while (opnorm > norm);
    });
  }

  executor.run(taskflow).wait();
  taskflow.clear();

  return eIcicleError::SUCCESS;
}

REGISTER_CHALLENGE_SPACE_POLYNOMIALS_SAMPLING_BACKEND("CPU", cpu_challenge_space_polynomials_sampling);
