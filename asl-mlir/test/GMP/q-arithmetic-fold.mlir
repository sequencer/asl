// RUN: asl-opt %s --canonicalize | FileCheck %s

// Test constant folding for Q module operations

//===----------------------------------------------------------------------===//
// Q Arithmetic Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_add_fold
func.func @test_q_add_fold() -> !gmp.q {
  // 1/2 + 1/4 = 3/4
  // CHECK: gmp.q.constant <3, 4>
  %a = gmp.q.constant #gmp.q<1, 2>
  %b = gmp.q.constant #gmp.q<1, 4>
  %sum = gmp.q.add %a, %b
  return %sum : !gmp.q
}

// CHECK-LABEL: func.func @test_q_sub_fold
func.func @test_q_sub_fold() -> !gmp.q {
  // 3/4 - 1/4 = 1/2
  // CHECK: gmp.q.constant <1, 2>
  %a = gmp.q.constant #gmp.q<3, 4>
  %b = gmp.q.constant #gmp.q<1, 4>
  %diff = gmp.q.sub %a, %b
  return %diff : !gmp.q
}

// CHECK-LABEL: func.func @test_q_mul_fold
func.func @test_q_mul_fold() -> !gmp.q {
  // 1/2 * 2/3 = 1/3
  // CHECK: gmp.q.constant <1, 3>
  %a = gmp.q.constant #gmp.q<1, 2>
  %b = gmp.q.constant #gmp.q<2, 3>
  %prod = gmp.q.mul %a, %b
  return %prod : !gmp.q
}

// CHECK-LABEL: func.func @test_q_div_fold
func.func @test_q_div_fold() -> !gmp.q {
  // (1/2) / (1/4) = 2
  // CHECK: gmp.q.constant <2, 1>
  %a = gmp.q.constant #gmp.q<1, 2>
  %b = gmp.q.constant #gmp.q<1, 4>
  %quot = gmp.q.div %a, %b
  return %quot : !gmp.q
}

//===----------------------------------------------------------------------===//
// Q Unary Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_neg_fold
func.func @test_q_neg_fold() -> !gmp.q {
  // CHECK: gmp.q.constant <-1, 2>
  %q = gmp.q.constant #gmp.q<1, 2>
  %neg = gmp.q.neg %q
  return %neg : !gmp.q
}

// CHECK-LABEL: func.func @test_q_abs_fold
func.func @test_q_abs_fold() -> !gmp.q {
  // CHECK: gmp.q.constant <3, 4>
  %q = gmp.q.constant #gmp.q<-3, 4>
  %abs = gmp.q.abs %q
  return %abs : !gmp.q
}

// CHECK-LABEL: func.func @test_q_inv_fold
func.func @test_q_inv_fold() -> !gmp.q {
  // inv(2/3) = 3/2
  // CHECK: gmp.q.constant <3, 2>
  %q = gmp.q.constant #gmp.q<2, 3>
  %inv = gmp.q.inv %q
  return %inv : !gmp.q
}

//===----------------------------------------------------------------------===//
// Q Component Accessors Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_num_fold
func.func @test_q_num_fold() -> !gmp.z {
  // CHECK: gmp.z.constant <3>
  %q = gmp.q.constant #gmp.q<3, 4>
  %num = gmp.q.num %q
  return %num : !gmp.z
}

// CHECK-LABEL: func.func @test_q_den_fold
func.func @test_q_den_fold() -> !gmp.z {
  // CHECK: gmp.z.constant <4>
  %q = gmp.q.constant #gmp.q<3, 4>
  %den = gmp.q.den %q
  return %den : !gmp.z
}

//===----------------------------------------------------------------------===//
// Q Rounding Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_floor_positive
func.func @test_q_floor_positive() -> !gmp.z {
  // floor(7/3) = 2
  // CHECK: gmp.z.constant <2>
  %q = gmp.q.constant #gmp.q<7, 3>
  %floor = gmp.q.floor %q
  return %floor : !gmp.z
}

// CHECK-LABEL: func.func @test_q_floor_negative
func.func @test_q_floor_negative() -> !gmp.z {
  // floor(-7/3) = -3
  // CHECK: gmp.z.constant <-3>
  %q = gmp.q.constant #gmp.q<-7, 3>
  %floor = gmp.q.floor %q
  return %floor : !gmp.z
}

// CHECK-LABEL: func.func @test_q_ceil_positive
func.func @test_q_ceil_positive() -> !gmp.z {
  // ceil(7/3) = 3
  // CHECK: gmp.z.constant <3>
  %q = gmp.q.constant #gmp.q<7, 3>
  %ceil = gmp.q.ceil %q
  return %ceil : !gmp.z
}

// CHECK-LABEL: func.func @test_q_ceil_negative
func.func @test_q_ceil_negative() -> !gmp.z {
  // ceil(-7/3) = -2
  // CHECK: gmp.z.constant <-2>
  %q = gmp.q.constant #gmp.q<-7, 3>
  %ceil = gmp.q.ceil %q
  return %ceil : !gmp.z
}

// CHECK-LABEL: func.func @test_q_trunc_positive
func.func @test_q_trunc_positive() -> !gmp.z {
  // trunc(7/3) = 2
  // CHECK: gmp.z.constant <2>
  %q = gmp.q.constant #gmp.q<7, 3>
  %trunc = gmp.q.trunc %q
  return %trunc : !gmp.z
}

// CHECK-LABEL: func.func @test_q_trunc_negative
func.func @test_q_trunc_negative() -> !gmp.z {
  // trunc(-7/3) = -2
  // CHECK: gmp.z.constant <-2>
  %q = gmp.q.constant #gmp.q<-7, 3>
  %trunc = gmp.q.trunc %q
  return %trunc : !gmp.z
}

// CHECK-LABEL: func.func @test_q_round_up
func.func @test_q_round_up() -> !gmp.z {
  // round(3/2) = 2 (round away from zero)
  // CHECK: gmp.z.constant <2>
  %q = gmp.q.constant #gmp.q<3, 2>
  %round = gmp.q.round %q
  return %round : !gmp.z
}

// CHECK-LABEL: func.func @test_q_round_down
func.func @test_q_round_down() -> !gmp.z {
  // round(5/4) = 1
  // CHECK: gmp.z.constant <1>
  %q = gmp.q.constant #gmp.q<5, 4>
  %round = gmp.q.round %q
  return %round : !gmp.z
}

//===----------------------------------------------------------------------===//
// Q Conversion Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_to_bigint_fold
func.func @test_q_to_bigint_fold() -> !gmp.z {
  // CHECK: gmp.z.constant <2>
  %q = gmp.q.constant #gmp.q<7, 3>
  %z = gmp.q.to_bigint %q
  return %z : !gmp.z
}

// CHECK-LABEL: func.func @test_q_from_z_fold
func.func @test_q_from_z_fold() -> !gmp.q {
  // CHECK: gmp.q.constant <42, 1>
  %z = gmp.z.constant #gmp.z<42>
  %q = gmp.q.from_z %z
  return %q : !gmp.q
}

//===----------------------------------------------------------------------===//
// Q Special Values Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_neg_inf
func.func @test_q_neg_inf() -> !gmp.q {
  // neg(+inf) = -inf
  // CHECK: gmp.q.constant <-1, 0>
  %q = gmp.q.constant #gmp.q<1, 0>
  %neg = gmp.q.neg %q
  return %neg : !gmp.q
}

// CHECK-LABEL: func.func @test_q_abs_neg_inf
func.func @test_q_abs_neg_inf() -> !gmp.q {
  // abs(-inf) = +inf
  // CHECK: gmp.q.constant <1, 0>
  %q = gmp.q.constant #gmp.q<-1, 0>
  %abs = gmp.q.abs %q
  return %abs : !gmp.q
}

// CHECK-LABEL: func.func @test_q_inv_zero
func.func @test_q_inv_zero() -> !gmp.q {
  // inv(0) = +inf
  // CHECK: gmp.q.constant <1, 0>
  %q = gmp.q.constant #gmp.q<0, 1>
  %inv = gmp.q.inv %q
  return %inv : !gmp.q
}

// CHECK-LABEL: func.func @test_q_inv_inf
func.func @test_q_inv_inf() -> !gmp.q {
  // inv(+inf) = 0
  // CHECK: gmp.q.constant <0, 1>
  %q = gmp.q.constant #gmp.q<1, 0>
  %inv = gmp.q.inv %q
  return %inv : !gmp.q
}

//===----------------------------------------------------------------------===//
// Q Make Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_make_fold
func.func @test_q_make_fold() -> !gmp.q {
  // make(6, 4) canonicalizes to 3/2
  // CHECK: gmp.q.constant <3, 2>
  %num = gmp.z.constant #gmp.z<6>
  %den = gmp.z.constant #gmp.z<4>
  %q = gmp.q.make %num, %den
  return %q : !gmp.q
}

//===----------------------------------------------------------------------===//
// Q Comparison and Classification Operations (non-folding tests)
// Note: These operations don't fold to arith.constant due to MLIR
// limitations with cross-dialect constant materialization, but they
// still work correctly at runtime.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_compare
func.func @test_q_compare(%a: !gmp.q, %b: !gmp.q) -> i32 {
  // CHECK: gmp.q.compare
  %cmp = gmp.q.compare %a, %b
  return %cmp : i32
}

// CHECK-LABEL: func.func @test_q_equal
func.func @test_q_equal(%a: !gmp.q, %b: !gmp.q) -> i1 {
  // CHECK: gmp.q.equal
  %eq = gmp.q.equal %a, %b
  return %eq : i1
}

// CHECK-LABEL: func.func @test_q_classify
func.func @test_q_classify(%q: !gmp.q) -> i32 {
  // CHECK: gmp.q.classify
  %class = gmp.q.classify %q
  return %class : i32
}

// CHECK-LABEL: func.func @test_q_is_real
func.func @test_q_is_real(%q: !gmp.q) -> i1 {
  // CHECK: gmp.q.is_real
  %is_real = gmp.q.is_real %q
  return %is_real : i1
}
