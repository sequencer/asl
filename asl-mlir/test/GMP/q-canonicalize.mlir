// RUN: asl-opt %s --canonicalize | FileCheck %s

// Test canonicalization patterns for Q module operations

//===----------------------------------------------------------------------===//
// Q Arithmetic Identity Patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_add_zero_rhs
func.func @test_q_add_zero_rhs(%x: !gmp.q) -> !gmp.q {
  // CHECK: return %arg0
  %zero = gmp.q.constant #gmp.q<0, 1>
  %result = gmp.q.add %x, %zero
  return %result : !gmp.q
}

// CHECK-LABEL: func.func @test_q_add_zero_lhs
func.func @test_q_add_zero_lhs(%x: !gmp.q) -> !gmp.q {
  // CHECK: return %arg0
  %zero = gmp.q.constant #gmp.q<0, 1>
  %result = gmp.q.add %zero, %x
  return %result : !gmp.q
}

// CHECK-LABEL: func.func @test_q_sub_zero
func.func @test_q_sub_zero(%x: !gmp.q) -> !gmp.q {
  // CHECK: return %arg0
  %zero = gmp.q.constant #gmp.q<0, 1>
  %result = gmp.q.sub %x, %zero
  return %result : !gmp.q
}

// CHECK-LABEL: func.func @test_q_sub_self
func.func @test_q_sub_self(%x: !gmp.q) -> !gmp.q {
  // CHECK: gmp.q.constant <0, 1>
  %result = gmp.q.sub %x, %x
  return %result : !gmp.q
}

// CHECK-LABEL: func.func @test_q_mul_one_rhs
func.func @test_q_mul_one_rhs(%x: !gmp.q) -> !gmp.q {
  // CHECK: return %arg0
  %one = gmp.q.constant #gmp.q<1, 1>
  %result = gmp.q.mul %x, %one
  return %result : !gmp.q
}

// CHECK-LABEL: func.func @test_q_mul_one_lhs
func.func @test_q_mul_one_lhs(%x: !gmp.q) -> !gmp.q {
  // CHECK: return %arg0
  %one = gmp.q.constant #gmp.q<1, 1>
  %result = gmp.q.mul %one, %x
  return %result : !gmp.q
}

// CHECK-LABEL: func.func @test_q_div_one
func.func @test_q_div_one(%x: !gmp.q) -> !gmp.q {
  // CHECK: return %arg0
  %one = gmp.q.constant #gmp.q<1, 1>
  %result = gmp.q.div %x, %one
  return %result : !gmp.q
}

//===----------------------------------------------------------------------===//
// Q Unary Operation Patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_neg_neg
func.func @test_q_neg_neg(%x: !gmp.q) -> !gmp.q {
  // CHECK: return %arg0
  %neg1 = gmp.q.neg %x
  %neg2 = gmp.q.neg %neg1
  return %neg2 : !gmp.q
}

// CHECK-LABEL: func.func @test_q_abs_abs
func.func @test_q_abs_abs(%x: !gmp.q) -> !gmp.q {
  // CHECK: %[[ABS:.*]] = gmp.q.abs %arg0
  // CHECK: return %[[ABS]]
  %abs1 = gmp.q.abs %x
  %abs2 = gmp.q.abs %abs1
  return %abs2 : !gmp.q
}

// CHECK-LABEL: func.func @test_q_inv_inv
func.func @test_q_inv_inv(%x: !gmp.q) -> !gmp.q {
  // CHECK: return %arg0
  %inv1 = gmp.q.inv %x
  %inv2 = gmp.q.inv %inv1
  return %inv2 : !gmp.q
}

//===----------------------------------------------------------------------===//
// Q Comparison Operations (non-folding tests)
// Note: Comparisons returning i1/i32 don't fold due to MLIR constant
// materialization limitations. These tests verify the operations roundtrip.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_compare_roundtrip
func.func @test_q_compare_roundtrip(%x: !gmp.q, %y: !gmp.q) -> i32 {
  // CHECK: gmp.q.compare
  %result = gmp.q.compare %x, %y
  return %result : i32
}

// CHECK-LABEL: func.func @test_q_equal_roundtrip
func.func @test_q_equal_roundtrip(%x: !gmp.q, %y: !gmp.q) -> i1 {
  // CHECK: gmp.q.equal
  %result = gmp.q.equal %x, %y
  return %result : i1
}

//===----------------------------------------------------------------------===//
// Q Special Value Patterns
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_neg_pos_inf
func.func @test_q_neg_pos_inf() -> !gmp.q {
  // CHECK: gmp.q.constant <-1, 0>
  %inf = gmp.q.constant #gmp.q<1, 0>
  %neg = gmp.q.neg %inf
  return %neg : !gmp.q
}

// CHECK-LABEL: func.func @test_q_neg_neg_inf
func.func @test_q_neg_neg_inf() -> !gmp.q {
  // CHECK: gmp.q.constant <1, 0>
  %inf = gmp.q.constant #gmp.q<-1, 0>
  %neg = gmp.q.neg %inf
  return %neg : !gmp.q
}

// CHECK-LABEL: func.func @test_q_abs_neg_inf
func.func @test_q_abs_neg_inf() -> !gmp.q {
  // CHECK: gmp.q.constant <1, 0>
  %inf = gmp.q.constant #gmp.q<-1, 0>
  %abs = gmp.q.abs %inf
  return %abs : !gmp.q
}

// CHECK-LABEL: func.func @test_q_inv_zero
func.func @test_q_inv_zero() -> !gmp.q {
  // CHECK: gmp.q.constant <1, 0>
  %zero = gmp.q.constant #gmp.q<0, 1>
  %inv = gmp.q.inv %zero
  return %inv : !gmp.q
}

// CHECK-LABEL: func.func @test_q_inv_pos_inf
func.func @test_q_inv_pos_inf() -> !gmp.q {
  // CHECK: gmp.q.constant <0, 1>
  %inf = gmp.q.constant #gmp.q<1, 0>
  %inv = gmp.q.inv %inf
  return %inv : !gmp.q
}
