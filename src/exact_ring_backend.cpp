#include "labrador.h"
#include "lnplabrador_backend_params.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>

namespace icicle::labrador::coefficient_backend {
namespace {

using Wide = __int128_t;

constexpr size_t DEGREE = PolyRing::d;
constexpr uint64_t MODULUS = backend_config::RING_MODULUS;

static_assert(DEGREE == 64, "the exact coefficient backend is specialized for degree 64");
static_assert(MODULUS == ((uint64_t{1} << 40) - 195), "coefficient backend modulus mismatch");
static_assert(sizeof(Zq) == sizeof(uint64_t), "the q40 scalar must occupy two 32-bit limbs");

uint64_t canonical_value(const Zq& value)
{
  const uint64_t raw = static_cast<uint64_t>(value.limbs_storage.limbs[0]) |
                       (static_cast<uint64_t>(value.limbs_storage.limbs[1]) << 32);
  return raw < MODULUS ? raw : raw % MODULUS;
}

Zq from_canonical(uint64_t value)
{
  Zq result{};
  result.limbs_storage.limbs[0] = static_cast<uint32_t>(value);
  result.limbs_storage.limbs[1] = static_cast<uint32_t>(value >> 32);
  return result;
}

Zq reduce_wide(Wide value)
{
  Wide remainder = value % static_cast<Wide>(MODULUS);
  if (remainder < 0) { remainder += static_cast<Wide>(MODULUS); }
  return from_canonical(static_cast<uint64_t>(remainder));
}

// Full integer polynomial multiplication.  Reducing only after the
// negacyclic fold is important: the auxiliary integers are not moduli of q.
// A threshold of four cuts a degree-64 product from 4096 to 1296 base
// multiplications while keeping the implementation small and auditable.
template <size_t N>
void karatsuba_full(
  const std::array<Wide, N>& a,
  const std::array<Wide, N>& b,
  std::array<Wide, 2 * N>& output)
{
  output.fill(0);
  if constexpr (N <= 4) {
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) { output[i + j] += a[i] * b[j]; }
    }
  } else {
    constexpr size_t HALF = N / 2;
    std::array<Wide, HALF> a_low{}, a_high{}, b_low{}, b_high{};
    std::array<Wide, HALF> a_sum{}, b_sum{};
    for (size_t i = 0; i < HALF; ++i) {
      a_low[i] = a[i];
      a_high[i] = a[i + HALF];
      b_low[i] = b[i];
      b_high[i] = b[i + HALF];
      a_sum[i] = a_low[i] + a_high[i];
      b_sum[i] = b_low[i] + b_high[i];
    }

    std::array<Wide, N> low{}, high{}, cross{};
    karatsuba_full(a_low, b_low, low);
    karatsuba_full(a_high, b_high, high);
    karatsuba_full(a_sum, b_sum, cross);
    for (size_t i = 0; i < N; ++i) { cross[i] -= low[i] + high[i]; }

    for (size_t i = 0; i < N; ++i) {
      output[i] += low[i];
      output[i + HALF] += cross[i];
      output[i + N] += high[i];
    }
  }
}

void negacyclic_raw(const Tq& a, const Tq& b, std::array<Wide, DEGREE>& output)
{
  std::array<Wide, DEGREE> a_integer{}, b_integer{};
  for (size_t i = 0; i < DEGREE; ++i) {
    a_integer[i] = static_cast<Wide>(canonical_value(a.values[i]));
    b_integer[i] = static_cast<Wide>(canonical_value(b.values[i]));
  }

  std::array<Wide, 2 * DEGREE> full{};
  karatsuba_full(a_integer, b_integer, full);
  for (size_t i = 0; i < DEGREE; ++i) { output[i] = full[i] - full[i + DEGREE]; }
}

void store_reduced(const std::array<Wide, DEGREE>& input, Tq& output)
{
  for (size_t i = 0; i < DEGREE; ++i) { output.values[i] = reduce_wide(input[i]); }
}

template <typename Function>
void parallel_for(size_t count, uint64_t work_per_item, Function&& function)
{
  if (count == 0) { return; }
  const unsigned hardware = std::max(1U, std::thread::hardware_concurrency());
  const size_t worker_count = std::min<size_t>({count, hardware, size_t{32}});
  const uint64_t capped_work = std::max<uint64_t>(1, std::min<uint64_t>(work_per_item, 1024));
  const size_t minimum_parallel_items = static_cast<size_t>((32 + capped_work - 1) / capped_work);
  if (worker_count == 1 || count < minimum_parallel_items) {
    for (size_t i = 0; i < count; ++i) { function(i); }
    return;
  }

  std::atomic<size_t> next{0};
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (size_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&]() {
      while (true) {
        const size_t index = next.fetch_add(1, std::memory_order_relaxed);
        if (index >= count) { break; }
        function(index);
      }
    });
  }
  for (auto& worker : workers) { worker.join(); }
}

bool host_only(const VecOpsConfig& config)
{
  return !config.is_a_on_device && !config.is_b_on_device && !config.is_result_on_device;
}

bool host_only(const MatMulConfig& config)
{
  return !config.is_a_on_device && !config.is_b_on_device && !config.is_result_on_device;
}

} // namespace

eIcicleError representation_copy(
  const PolyRing* input, size_t size, const NegacyclicNTTConfig& config, PolyRing* output)
{
  if (size == 0) { return eIcicleError::SUCCESS; }
  if (input == nullptr || output == nullptr) { return eIcicleError::INVALID_POINTER; }
  if (input == output) { return eIcicleError::SUCCESS; }

  if (size > std::numeric_limits<size_t>::max() / sizeof(PolyRing)) {
    return eIcicleError::INVALID_ARGUMENT;
  }
  const size_t bytes = size * sizeof(PolyRing);
  return config.is_async ? icicle_copy_async(output, input, bytes, config.stream) : icicle_copy(output, input, bytes);
}

eIcicleError vector_mul(
  const Tq* a, const Tq* b, uint64_t size, const VecOpsConfig& config, Tq* output)
{
  if (config.columns_batch || config.batch_size != 1) { return eIcicleError::INVALID_ARGUMENT; }
  if (!host_only(config)) { return eIcicleError::API_NOT_IMPLEMENTED; }
  if (size == 0) { return eIcicleError::SUCCESS; }
  if (a == nullptr || b == nullptr || output == nullptr) { return eIcicleError::INVALID_POINTER; }
  if (size > std::numeric_limits<size_t>::max()) { return eIcicleError::INVALID_ARGUMENT; }

  parallel_for(static_cast<size_t>(size), 1, [&](size_t index) {
    std::array<Wide, DEGREE> product{};
    negacyclic_raw(a[index], b[index], product);
    store_reduced(product, output[index]);
  });
  return eIcicleError::SUCCESS;
}

eIcicleError matmul(
  const Tq* A,
  uint32_t A_nof_rows,
  uint32_t A_nof_cols,
  const Tq* B,
  uint32_t B_nof_rows,
  uint32_t B_nof_cols,
  const MatMulConfig& config,
  Tq* C)
{
  if (A == nullptr || B == nullptr || C == nullptr || A_nof_rows == 0 || A_nof_cols == 0 ||
      B_nof_rows == 0 || B_nof_cols == 0) {
    return eIcicleError::INVALID_ARGUMENT;
  }
  if (!host_only(config)) { return eIcicleError::API_NOT_IMPLEMENTED; }
  if (config.result_transposed) { return eIcicleError::INVALID_ARGUMENT; }

  const uint32_t rows_a = config.a_transposed ? A_nof_cols : A_nof_rows;
  const uint32_t cols_a = config.a_transposed ? A_nof_rows : A_nof_cols;
  const uint32_t rows_b = config.b_transposed ? B_nof_cols : B_nof_rows;
  const uint32_t cols_b = config.b_transposed ? B_nof_rows : B_nof_cols;
  if (cols_a != rows_b) { return eIcicleError::INVALID_ARGUMENT; }

  if (rows_a != 0 && static_cast<size_t>(cols_b) > std::numeric_limits<size_t>::max() / rows_a) {
    return eIcicleError::INVALID_ARGUMENT;
  }
  const size_t output_count = static_cast<size_t>(rows_a) * cols_b;
  if (output_count > std::numeric_limits<size_t>::max() / sizeof(Tq)) {
    return eIcicleError::INVALID_ARGUMENT;
  }

  // Always use a temporary result.  Besides making overlapping matrix inputs
  // well-defined, this is required by agg_equality_constraints(), which
  // deliberately scales matrices in place through a 1x1 matmul.
  std::vector<Tq> temporary(output_count);
  parallel_for(output_count, cols_a, [&](size_t output_index) {
    const uint32_t row = static_cast<uint32_t>(output_index / cols_b);
    const uint32_t col = static_cast<uint32_t>(output_index % cols_b);
    // Even at the uint32_t dimension limit this cannot overflow Wide:
    // 64*(q-1)^2*(2^32-1) < 2^118 < 2^127.
    std::array<Wide, DEGREE> accumulator{};

    for (uint32_t k = 0; k < cols_a; ++k) {
      const Tq& left = config.a_transposed ? A[static_cast<size_t>(k) * A_nof_cols + row]
                                           : A[static_cast<size_t>(row) * A_nof_cols + k];
      const Tq& right = config.b_transposed ? B[static_cast<size_t>(col) * B_nof_cols + k]
                                            : B[static_cast<size_t>(k) * B_nof_cols + col];
      std::array<Wide, DEGREE> product{};
      negacyclic_raw(left, right, product);
      for (size_t coefficient = 0; coefficient < DEGREE; ++coefficient) {
        accumulator[coefficient] += product[coefficient];
      }
    }
    store_reduced(accumulator, temporary[output_index]);
  });

  std::memcpy(C, temporary.data(), output_count * sizeof(Tq));
  return eIcicleError::SUCCESS;
}

} // namespace icicle::labrador::coefficient_backend
