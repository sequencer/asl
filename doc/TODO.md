# GMP Dialect Implementation TODO

This document outlines the implementation plan for the GMP dialect, based on the design specified in `GMPRational.typ`.

## Phase 1: Core Infrastructure

- [ ] 1.1. Create the file and directory structure for the GMP dialect as specified in `doc/GMPRational.typ`.
- [ ] 1.2. Update `CMakeLists.txt` files to include the new dialect's source files and TableGen definitions.
- [ ] 1.3. Register the `gmp` dialect in `asl-opt` to make it available for use.
- [ ] 1.4. Set up the basic `lit` testing infrastructure for the GMP dialect.

## Phase 2: Z Module - Types and Basic Arithmetic

- [ ] 2.1. Define the `!gmp.z` type in TableGen (`GMPTypes.td`).
- [ ] 2.2. Implement the C++ class for the `ZType`.
- [ ] 2.3. Implement `gmp.z.constant` and the associated `#gmp.z` attribute for creating integer constants.
- [ ] 2.4. Implement `gmp.z.from_int` for runtime conversions from standard integer types, and add simple test for it.
- [ ] 2.5. Implement basic arithmetic operations: `gmp.z.add`, `gmp.z.sub`, `gmp.z.mul`, and add simple test for it.
- [ ] 2.6. Implement unary operations: `gmp.z.neg`, `gmp.z.abs`, `gmp.z.succ`, `gmp.z.pred`, and add simple test for it.
- [ ] 2.7. Add initial constant folding support for these basic operations, and add simple test for it.
- [ ] 2.8. Build and Test, if encountering error, fix it.

## Phase 3: Z Module - Division Operations

- [ ] 3.1. Implement truncated division: `gmp.z.div`, `gmp.z.rem`, and `gmp.z.div_rem`.
- [ ] 3.2. Implement floor division: `gmp.z.fdiv`.
- [ ] 3.3. Implement ceiling division: `gmp.z.cdiv`.
- [ ] 3.4. Implement Euclidean division: `gmp.z.ediv` and `gmp.z.erem`.
- [ ] 3.5. Implement exact division: `gmp.z.divexact`.
- [ ] 3.6. Implement divisibility tests: `gmp.z.divisible` and `gmp.z.congruent`.
- [ ] 3.7. Add folding rules for all division operations.
- [ ] 3.8. Add MLIR tests for constant folding of basic operations.

## Phase 4: Z Module - Bitwise and Comparison Operations

- [ ] 4.1. Implement comparison operations: `gmp.z.compare`, `gmp.z.equal`, `gmp.z.lt`, `gmp.z.leq`, `gmp.z.gt`, `gmp.z.geq`.
- [ ] 4.2. Implement bitwise logical operations: `gmp.z.logand`, `gmp.z.logor`, `gmp.z.logxor`, `gmp.z.lognot`.
- [ ] 4.3. Implement bit shift operations: `gmp.z.shift_left`, `gmp.z.shift_right`, `gmp.z.shift_right_trunc`.
- [ ] 4.4. Implement bit query operations: `gmp.z.testbit`, `gmp.z.popcount`, `gmp.z.numbits`, `gmp.z.trailing_zeros`.
- [ ] 4.5. Implement bit extraction: `gmp.z.extract`, `gmp.z.signed_extract`.
- [ ] 4.6. Implement folding rules for all bitwise and comparison operations.
- [ ] 4.7. Add MLIR tests for constant folding of basic operations.

## Phase 5: Z Module - Number Theory and Advanced Operations

- [ ] 5.1. Implement GCD and LCM: `gmp.z.gcd`, `gmp.z.gcdext`, `gmp.z.lcm`.
- [ ] 5.2. Implement modular arithmetic: `gmp.z.powm`, `gmp.z.powm_sec`, `gmp.z.invert`.
- [ ] 5.3. Implement primality testing: `gmp.z.probab_prime`, `gmp.z.nextprime`.
- [ ] 5.4. Implement powers and roots: `gmp.z.pow`, `gmp.z.sqrt`, `gmp.z.sqrt_rem`, `gmp.z.root`.
- [ ] 5.5. Implement factorial and combinatorics: `gmp.z.fac`, `gmp.z.bin`, `gmp.z.fib`.
- [ ] 5.6. Add folding rules for number theory operations.
- [ ] 5.7. Add MLIR tests for constant folding of basic operations.

## Phase 6: Q Module - Rational Numbers

- [ ] 6.1. Define the `!gmp.q` type in TableGen (`GMPTypes.td`).
- [ ] 6.2. Implement `gmp.q.constant` and the `#gmp.q` attribute, including support for special values (`inf`, `-inf`, `undef`).
- [ ] 6.3. Implement construction: `gmp.q.make`.
- [ ] 6.4. Implement component accessors: `gmp.q.num` and `gmp.q.den`.
- [ ] 6.5. Implement basic arithmetic: `gmp.q.add`, `gmp.q.sub`, `gmp.q.mul`, `gmp.q.div`.
- [ ] 6.6. Implement unary operations: `gmp.q.neg`, `gmp.q.abs`, `gmp.q.inv`.
- [ ] 6.7. Implement comparison operations: `gmp.q.compare`, `gmp.q.equal`, `gmp.q.lt`, etc.
- [ ] 6.8. Implement classification: `gmp.q.classify`, `gmp.q.is_real`.
- [ ] 6.9. Implement ASL-specific rounding: `gmp.q.floor`, `gmp.q.ceil`, `gmp.q.trunc`, `gmp.q.round`.
- [ ] 6.10. Implement conversions: `gmp.q.to_bigint`, `gmp.q.to_f64`, `gmp.q.to_string`.
- [ ] 6.11. Add MLIR tests for basic operations.

## Phase 7: Canonicalization and Constant Folding

- [ ] 7.1. Implement canonicalization patterns for `Z` module operations (e.g., `x + 0 -> x`).
- [ ] 7.2. Implement canonicalization patterns for `Q` module operations (e.g., `x * 1 -> x`).
- [ ] 7.3. Ensure all operations have robust constant folding implementations in their `fold()` methods.

## Phase 8: GMP Lowering and Testing ✓

- [x] 8.1. Create the `convert-gmp-to-emitc` pass.
- [x] 8.2. Implement lowering for all `Z` module operations to their `mpz_*` counterparts in GMP.
- [x] 8.3. Implement lowering for all `Q` module operations to their `mpq_*` counterparts in GMP.
- [x] 8.4. Write comprehensive `.mlir` test files for the `Z` module (`test/GMP/convert-gmp-z-to-emitc.mlir`).
- [x] 8.5. Write comprehensive `.mlir` test files for the `Q` module (`test/GMP/convert-gmp-q-to-emitc.mlir`).
- [x] 8.6. Add tests for constant folding (`test/GMP/convert-gmp-folded-to-emitc.mlir`).
- [x] 8.7. Develop a script for differential testing against Python's `gmpy2` (`test/GMP/differential-test.py`).

### Differential Testing

The differential testing script (`test/GMP/differential-test.py`) compares the constant folding results
of the GMP dialect with Python's arbitrary precision arithmetic and gmpy2 library.

**Requirements:**
- Python 3.7+
- gmpy2 (optional, uses Python's built-in arbitrary precision if not available)

**Usage:**
```bash
# Enter dev shell (includes Python with gmpy2)
nix develop

# Run differential tests
python3 test/GMP/differential-test.py --asl-opt ./build/bin/asl-opt --num-tests 100

# Run with verbose output
python3 test/GMP/differential-test.py -v --num-tests 10
```

**Test Coverage:**
- Z module: add, sub, mul, div, neg, abs, compare, equal, gcd, pow
- Q module: add, mul, neg, equal, floor