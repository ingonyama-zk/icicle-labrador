#pragma once

#include "icicle/hash/keccak.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace icicle::cpu {

  namespace challenge_xof_detail {

    inline void append_u64_le(std::vector<std::byte>& output, uint64_t value)
    {
      for (unsigned shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xff));
      }
    }

  } // namespace challenge_xof_detail

  /**
   * Deterministic, domain-separated Keccak-512 counter expander.
   *
   * Keccak-512 itself has a fixed 512-bit output.  Sampling a challenge can
   * consume more than one digest, so each block is instead defined as
   *
   *   Keccak512(frame || domain || seed || context || block_counter.LE64).
   *
   * Lengths and integral values use an explicit little-endian encoding.  This
   * makes the stream independent of host endianness and allows it to refill
   * without the finite-period LFSR fallback used by the old sampler.
   */
  class Keccak512CounterXof
  {
  public:
    Keccak512CounterXof(
      const Hash& keccak512,
      std::string_view domain,
      const std::byte* seed,
      size_t seed_len,
      std::initializer_list<uint64_t> context = {})
        : m_keccak512(keccak512)
    {
      if (keccak512.output_size() != m_block.size()) {
        throw std::invalid_argument("Keccak512CounterXof requires a 64-byte Keccak-512 hash");
      }
      if (seed_len != 0 && seed == nullptr) {
        throw std::invalid_argument("Keccak512CounterXof received a null seed");
      }

      static constexpr std::array<std::byte, 27> frame = {
        std::byte{'I'}, std::byte{'C'}, std::byte{'I'}, std::byte{'C'}, std::byte{'L'}, std::byte{'E'},
        std::byte{'-'}, std::byte{'K'}, std::byte{'E'}, std::byte{'C'}, std::byte{'C'}, std::byte{'A'},
        std::byte{'K'}, std::byte{'5'}, std::byte{'1'}, std::byte{'2'}, std::byte{'-'}, std::byte{'C'},
        std::byte{'T'}, std::byte{'R'}, std::byte{'-'}, std::byte{'X'}, std::byte{'O'}, std::byte{'F'},
        std::byte{'-'}, std::byte{'v'}, std::byte{'1'},
      };

      m_prefix.reserve(
        frame.size() + sizeof(uint64_t) + domain.size() + sizeof(uint64_t) + seed_len + sizeof(uint64_t) +
        context.size() * sizeof(uint64_t) + sizeof(uint64_t));
      m_prefix.insert(m_prefix.end(), frame.begin(), frame.end());
      challenge_xof_detail::append_u64_le(m_prefix, domain.size());
      if (!domain.empty()) {
        m_prefix.insert(
          m_prefix.end(), reinterpret_cast<const std::byte*>(domain.data()),
          reinterpret_cast<const std::byte*>(domain.data()) + domain.size());
      }
      challenge_xof_detail::append_u64_le(m_prefix, seed_len);
      if (seed_len != 0) { m_prefix.insert(m_prefix.end(), seed, seed + seed_len); }
      challenge_xof_detail::append_u64_le(m_prefix, context.size());
      for (uint64_t value : context) {
        challenge_xof_detail::append_u64_le(m_prefix, value);
      }
    }

    bool next_bit()
    {
      if (m_bit_offset == bits_per_block) { refill(); }
      const size_t byte_idx = m_bit_offset / 8;
      const unsigned bit_idx = m_bit_offset % 8;
      const bool bit = ((std::to_integer<uint8_t>(m_block[byte_idx]) >> bit_idx) & 1U) != 0;
      ++m_bit_offset;
      return bit;
    }

    uint64_t next_u64_le()
    {
      uint64_t value = 0;
      for (unsigned bit = 0; bit < 64; ++bit) {
        value |= uint64_t{next_bit()} << bit;
      }
      return value;
    }

    uint64_t blocks_generated() const { return m_block_counter; }

  private:
    static constexpr size_t bits_per_block = 512;

    void refill()
    {
      std::vector<std::byte> input = m_prefix;
      challenge_xof_detail::append_u64_le(input, m_block_counter);
      HashConfig hash_cfg{};
      const eIcicleError error = m_keccak512.hash(input.data(), input.size(), hash_cfg, m_block.data());
      if (error != eIcicleError::SUCCESS) {
        throw std::runtime_error("Keccak-512 counter expansion failed");
      }
      ++m_block_counter;
      m_bit_offset = 0;
    }

    Hash m_keccak512;
    std::vector<std::byte> m_prefix;
    std::array<std::byte, 64> m_block{};
    size_t m_bit_offset = bits_per_block;
    uint64_t m_block_counter = 0;
  };

} // namespace icicle::cpu
