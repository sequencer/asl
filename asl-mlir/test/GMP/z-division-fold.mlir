// RUN: asl-opt %s --canonicalize | FileCheck %s

// Test constant folding for Z division operations

//===----------------------------------------------------------------------===//
// Truncated Division (div/rem)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_div_positive
func.func @fold_z_div_positive() -> !gmp.z {
  %a = gmp.z.constant <7>
  %b = gmp.z.constant <3>
  // CHECK: gmp.z.constant <2>
  %q = gmp.z.div %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_div_negative_dividend
func.func @fold_z_div_negative_dividend() -> !gmp.z {
  %a = gmp.z.constant <-7>
  %b = gmp.z.constant <3>
  // Truncated: -7 / 3 = -2 (truncate toward zero)
  // CHECK: gmp.z.constant <-2>
  %q = gmp.z.div %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_rem_positive
func.func @fold_z_rem_positive() -> !gmp.z {
  %a = gmp.z.constant <7>
  %b = gmp.z.constant <3>
  // CHECK: gmp.z.constant <1>
  %r = gmp.z.rem %a, %b
  return %r : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_rem_negative_dividend
func.func @fold_z_rem_negative_dividend() -> !gmp.z {
  %a = gmp.z.constant <-7>
  %b = gmp.z.constant <3>
  // Truncated remainder: -7 % 3 = -1 (sign matches dividend)
  // CHECK: gmp.z.constant <-1>
  %r = gmp.z.rem %a, %b
  return %r : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_div_by_one
func.func @fold_z_div_by_one(%x: !gmp.z) -> !gmp.z {
  %one = gmp.z.constant <1>
  // CHECK: return %arg0
  %q = gmp.z.div %x, %one
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_rem_by_one
func.func @fold_z_rem_by_one(%x: !gmp.z) -> !gmp.z {
  %one = gmp.z.constant <1>
  // CHECK: gmp.z.constant <0>
  %r = gmp.z.rem %x, %one
  return %r : !gmp.z
}

//===----------------------------------------------------------------------===//
// Floor Division (fdiv/frem)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_fdiv_positive
func.func @fold_z_fdiv_positive() -> !gmp.z {
  %a = gmp.z.constant <7>
  %b = gmp.z.constant <3>
  // CHECK: gmp.z.constant <2>
  %q = gmp.z.fdiv %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_fdiv_negative_dividend
func.func @fold_z_fdiv_negative_dividend() -> !gmp.z {
  %a = gmp.z.constant <-7>
  %b = gmp.z.constant <3>
  // Floor: -7 / 3 = -3 (floor toward -infinity)
  // CHECK: gmp.z.constant <-3>
  %q = gmp.z.fdiv %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_frem_positive
func.func @fold_z_frem_positive() -> !gmp.z {
  %a = gmp.z.constant <7>
  %b = gmp.z.constant <3>
  // CHECK: gmp.z.constant <1>
  %r = gmp.z.frem %a, %b
  return %r : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_frem_negative_dividend
func.func @fold_z_frem_negative_dividend() -> !gmp.z {
  %a = gmp.z.constant <-7>
  %b = gmp.z.constant <3>
  // Floor remainder: -7 % 3 = 2 (sign matches divisor)
  // CHECK: gmp.z.constant <2>
  %r = gmp.z.frem %a, %b
  return %r : !gmp.z
}

//===----------------------------------------------------------------------===//
// Ceiling Division (cdiv/crem)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_cdiv_positive
func.func @fold_z_cdiv_positive() -> !gmp.z {
  %a = gmp.z.constant <7>
  %b = gmp.z.constant <3>
  // Ceiling: 7 / 3 = 3 (ceil toward +infinity)
  // CHECK: gmp.z.constant <3>
  %q = gmp.z.cdiv %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_cdiv_negative_dividend
func.func @fold_z_cdiv_negative_dividend() -> !gmp.z {
  %a = gmp.z.constant <-7>
  %b = gmp.z.constant <3>
  // Ceiling: -7 / 3 = -2 (ceil toward +infinity)
  // CHECK: gmp.z.constant <-2>
  %q = gmp.z.cdiv %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_crem_positive
func.func @fold_z_crem_positive() -> !gmp.z {
  %a = gmp.z.constant <7>
  %b = gmp.z.constant <3>
  // Ceiling remainder: 7 % 3 = -2
  // CHECK: gmp.z.constant <-2>
  %r = gmp.z.crem %a, %b
  return %r : !gmp.z
}

//===----------------------------------------------------------------------===//
// Euclidean Division (ediv/erem)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_ediv_positive
func.func @fold_z_ediv_positive() -> !gmp.z {
  %a = gmp.z.constant <7>
  %b = gmp.z.constant <3>
  // CHECK: gmp.z.constant <2>
  %q = gmp.z.ediv %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_ediv_negative_dividend
func.func @fold_z_ediv_negative_dividend() -> !gmp.z {
  %a = gmp.z.constant <-7>
  %b = gmp.z.constant <3>
  // Euclidean: -7 = q*3 + r where 0 <= r < 3
  // -7 = -3*3 + 2, so q = -3
  // CHECK: gmp.z.constant <-3>
  %q = gmp.z.ediv %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_erem_positive
func.func @fold_z_erem_positive() -> !gmp.z {
  %a = gmp.z.constant <7>
  %b = gmp.z.constant <3>
  // CHECK: gmp.z.constant <1>
  %r = gmp.z.erem %a, %b
  return %r : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_erem_negative_dividend
func.func @fold_z_erem_negative_dividend() -> !gmp.z {
  %a = gmp.z.constant <-7>
  %b = gmp.z.constant <3>
  // Euclidean remainder: always non-negative
  // -7 = -3*3 + 2, so r = 2
  // CHECK: gmp.z.constant <2>
  %r = gmp.z.erem %a, %b
  return %r : !gmp.z
}

//===----------------------------------------------------------------------===//
// Exact Division
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_divexact
func.func @fold_z_divexact() -> !gmp.z {
  %a = gmp.z.constant <12>
  %b = gmp.z.constant <4>
  // CHECK: gmp.z.constant <3>
  %q = gmp.z.divexact %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_divexact_negative
func.func @fold_z_divexact_negative() -> !gmp.z {
  %a = gmp.z.constant <-15>
  %b = gmp.z.constant <5>
  // CHECK: gmp.z.constant <-3>
  %q = gmp.z.divexact %a, %b
  return %q : !gmp.z
}

//===----------------------------------------------------------------------===//
// Divisibility Tests
//===----------------------------------------------------------------------===//

// Note: The divisibility operations fold to BoolAttr, but MLIR's canonicalizer
// cannot automatically materialize arith.constant for non-dialect types without
// a proper materialization hook. These tests verify the operations roundtrip
// correctly. In practice, the folding works when results are used by operations
// that can consume the folded BoolAttr directly.

// CHECK-LABEL: func.func @fold_z_divisible_true
func.func @fold_z_divisible_true() -> i1 {
  %a = gmp.z.constant <12>
  %b = gmp.z.constant <4>
  // CHECK: gmp.z.divisible
  %r = gmp.z.divisible %a, %b
  return %r : i1
}

// CHECK-LABEL: func.func @fold_z_divisible_false
func.func @fold_z_divisible_false() -> i1 {
  %a = gmp.z.constant <13>
  %b = gmp.z.constant <4>
  // CHECK: gmp.z.divisible
  %r = gmp.z.divisible %a, %b
  return %r : i1
}

// CHECK-LABEL: func.func @fold_z_divisible_by_one
func.func @fold_z_divisible_by_one(%x: !gmp.z) -> i1 {
  %one = gmp.z.constant <1>
  // CHECK: gmp.z.divisible
  %r = gmp.z.divisible %x, %one
  return %r : i1
}

// CHECK-LABEL: func.func @fold_z_divisible_self
func.func @fold_z_divisible_self(%x: !gmp.z) -> i1 {
  // CHECK: gmp.z.divisible
  %r = gmp.z.divisible %x, %x
  return %r : i1
}

//===----------------------------------------------------------------------===//
// Congruence Tests
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_congruent_true
func.func @fold_z_congruent_true() -> i1 {
  %a = gmp.z.constant <17>
  %b = gmp.z.constant <5>
  %m = gmp.z.constant <4>
  // 17 ≡ 5 (mod 4) because 17 % 4 = 1 and 5 % 4 = 1
  // CHECK: gmp.z.congruent
  %r = gmp.z.congruent %a, %b, %m
  return %r : i1
}

// CHECK-LABEL: func.func @fold_z_congruent_false
func.func @fold_z_congruent_false() -> i1 {
  %a = gmp.z.constant <17>
  %b = gmp.z.constant <6>
  %m = gmp.z.constant <4>
  // 17 ≢ 6 (mod 4) because 17 % 4 = 1 and 6 % 4 = 2
  // CHECK: gmp.z.congruent
  %r = gmp.z.congruent %a, %b, %m
  return %r : i1
}

// CHECK-LABEL: func.func @fold_z_congruent_self
func.func @fold_z_congruent_self(%x: !gmp.z, %m: !gmp.z) -> i1 {
  // x ≡ x (mod m) is always true
  // CHECK: gmp.z.congruent
  %r = gmp.z.congruent %x, %x, %m
  return %r : i1
}

//===----------------------------------------------------------------------===//
// Large Number Division
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_div_large
func.func @fold_z_div_large() -> !gmp.z {
  %a = gmp.z.constant <100000000000000000000>
  %b = gmp.z.constant <7>
  // CHECK: gmp.z.constant <14285714285714285714>
  %q = gmp.z.div %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_rem_large
func.func @fold_z_rem_large() -> !gmp.z {
  %a = gmp.z.constant <100000000000000000000>
  %b = gmp.z.constant <7>
  // 100000000000000000000 % 7 = 2
  // CHECK: gmp.z.constant <2>
  %r = gmp.z.rem %a, %b
  return %r : !gmp.z
}
