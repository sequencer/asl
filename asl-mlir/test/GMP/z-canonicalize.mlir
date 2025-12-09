// RUN: asl-opt %s --canonicalize | FileCheck %s

// Test canonicalization patterns for Z module operations

//===----------------------------------------------------------------------===//
// Z Arithmetic Identity Patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_add_zero_rhs
func.func @test_z_add_zero_rhs(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %zero = gmp.z.constant #gmp.z<0>
  %result = gmp.z.add %x, %zero
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_add_zero_lhs
func.func @test_z_add_zero_lhs(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %zero = gmp.z.constant #gmp.z<0>
  %result = gmp.z.add %zero, %x
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_sub_zero
func.func @test_z_sub_zero(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %zero = gmp.z.constant #gmp.z<0>
  %result = gmp.z.sub %x, %zero
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_sub_self
func.func @test_z_sub_self(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.constant <0>
  %result = gmp.z.sub %x, %x
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_mul_zero
func.func @test_z_mul_zero(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.constant <0>
  %zero = gmp.z.constant #gmp.z<0>
  %result = gmp.z.mul %x, %zero
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_mul_one_rhs
func.func @test_z_mul_one_rhs(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %one = gmp.z.constant #gmp.z<1>
  %result = gmp.z.mul %x, %one
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_mul_one_lhs
func.func @test_z_mul_one_lhs(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %one = gmp.z.constant #gmp.z<1>
  %result = gmp.z.mul %one, %x
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_div_one
func.func @test_z_div_one(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %one = gmp.z.constant #gmp.z<1>
  %result = gmp.z.div %x, %one
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_rem_one
func.func @test_z_rem_one(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.constant <0>
  %one = gmp.z.constant #gmp.z<1>
  %result = gmp.z.rem %x, %one
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_rem_self
func.func @test_z_rem_self(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.constant <0>
  %result = gmp.z.rem %x, %x
  return %result : !gmp.z
}

//===----------------------------------------------------------------------===//
// Z Unary Operation Patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_neg_neg
func.func @test_z_neg_neg(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %neg1 = gmp.z.neg %x
  %neg2 = gmp.z.neg %neg1
  return %neg2 : !gmp.z
}

// CHECK-LABEL: func.func @test_z_abs_abs
func.func @test_z_abs_abs(%x: !gmp.z) -> !gmp.z {
  // CHECK: %[[ABS:.*]] = gmp.z.abs %arg0
  // CHECK: return %[[ABS]]
  %abs1 = gmp.z.abs %x
  %abs2 = gmp.z.abs %abs1
  return %abs2 : !gmp.z
}

//===----------------------------------------------------------------------===//
// Z Bitwise Operation Patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_and_zero
func.func @test_z_and_zero(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.constant <0>
  %zero = gmp.z.constant #gmp.z<0>
  %result = gmp.z.logand %x, %zero
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_and_self
func.func @test_z_and_self(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %result = gmp.z.logand %x, %x
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_or_zero
func.func @test_z_or_zero(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %zero = gmp.z.constant #gmp.z<0>
  %result = gmp.z.logor %x, %zero
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_or_self
func.func @test_z_or_self(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %result = gmp.z.logor %x, %x
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_xor_zero
func.func @test_z_xor_zero(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %zero = gmp.z.constant #gmp.z<0>
  %result = gmp.z.logxor %x, %zero
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_xor_self
func.func @test_z_xor_self(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.constant <0>
  %result = gmp.z.logxor %x, %x
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_lognot_lognot
func.func @test_z_lognot_lognot(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %not1 = gmp.z.lognot %x
  %not2 = gmp.z.lognot %not1
  return %not2 : !gmp.z
}

//===----------------------------------------------------------------------===//
// Z Shift Operation Patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_shl_zero_count
func.func @test_z_shl_zero_count(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %count = arith.constant 0 : i64
  %result = gmp.z.shift_left %x, %count
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_shr_zero_count
func.func @test_z_shr_zero_count(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %count = arith.constant 0 : i64
  %result = gmp.z.shift_right %x, %count
  return %result : !gmp.z
}

//===----------------------------------------------------------------------===//
// Z Power Operation Patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_pow_zero
func.func @test_z_pow_zero(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.constant <1>
  %exp = arith.constant 0 : i64
  %result = gmp.z.pow %x, %exp
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_pow_one
func.func @test_z_pow_one(%x: !gmp.z) -> !gmp.z {
  // CHECK: return %arg0
  %exp = arith.constant 1 : i64
  %result = gmp.z.pow %x, %exp
  return %result : !gmp.z
}

//===----------------------------------------------------------------------===//
// Z Comparison Operations (non-folding tests)
// Note: Comparisons returning i1/i32 don't fold due to MLIR constant
// materialization limitations. These tests verify the operations roundtrip.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_compare_roundtrip
func.func @test_z_compare_roundtrip(%x: !gmp.z, %y: !gmp.z) -> i32 {
  // CHECK: gmp.z.compare
  %result = gmp.z.compare %x, %y
  return %result : i32
}

// CHECK-LABEL: func.func @test_z_equal_roundtrip
func.func @test_z_equal_roundtrip(%x: !gmp.z, %y: !gmp.z) -> i1 {
  // CHECK: gmp.z.equal
  %result = gmp.z.equal %x, %y
  return %result : i1
}

//===----------------------------------------------------------------------===//
// Z Divisibility Operations (non-folding tests)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_divisible_roundtrip
func.func @test_z_divisible_roundtrip(%x: !gmp.z, %y: !gmp.z) -> i1 {
  // CHECK: gmp.z.divisible
  %result = gmp.z.divisible %x, %y
  return %result : i1
}

// CHECK-LABEL: func.func @test_z_congruent_roundtrip
func.func @test_z_congruent_roundtrip(%x: !gmp.z, %y: !gmp.z, %m: !gmp.z) -> i1 {
  // CHECK: gmp.z.congruent
  %result = gmp.z.congruent %x, %y, %m
  return %result : i1
}
