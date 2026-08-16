
# Define available field libraries with an index and their supported features
# Format: index:field:features
set(ICICLE_FIELDS
  1001:babybear:NTT,EXT_FIELD,POSEIDON,POSEIDON2,SUMCHECK,FRI
  1002:stark252:NTT,POSEIDON,POSEIDON2,SUMCHECK,FRI
  1003:m31:EXT_FIELD,POSEIDON,POSEIDON2,SUMCHECK
  1004:koalabear:NTT,EXT_FIELD,POSEIDON,POSEIDON2,SUMCHECK,FRI
  1005:goldilocks:NTT,EXT_FIELD,POSEIDON2,SUMCHECK,FRI
)

# Define available curve libraries with an index and their supported features
# Format: index:curve:features
set(ICICLE_CURVES
  1:bn254:NTT,MSM,G2,ECNTT,POSEIDON,POSEIDON2,SUMCHECK,FRI,PAIRING
  2:bls12_381:NTT,MSM,G2,ECNTT,POSEIDON,POSEIDON2,SUMCHECK,FRI,PAIRING
  3:bls12_377:NTT,MSM,G2,ECNTT,POSEIDON,POSEIDON2,SUMCHECK,FRI,PAIRING
  4:bw6_761:NTT,MSM,G2,ECNTT,POSEIDON,POSEIDON2,SUMCHECK,FRI
  5:grumpkin:MSM,POSEIDON,POSEIDON2,SUMCHECK
)

# Define available ring libraries with an index and their supported features
# Format: index:curve:features
set(ICICLE_RINGS
  2001:babykoala:NTT
  # The requested/LOGQ40 modulus has no primitive 128-th root.  Its degree-64 ring
  # arithmetic is provided by src/exact_ring_backend.cpp, not ICICLE's
  # point-evaluation NTT backend.
  2002:labradorq40:NONE
)
