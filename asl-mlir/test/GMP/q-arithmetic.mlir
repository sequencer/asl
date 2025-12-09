// RUN: asl-opt %s | asl-opt | FileCheck %s

// Test Q module operations roundtrip

//===----------------------------------------------------------------------===//
// Q Construction Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_constant
func.func @test_q_constant() -> !gmp.q {
  // CHECK: gmp.q.constant
  %q = gmp.q.constant #gmp.q<1, 2>
  return %q : !gmp.q
}

// CHECK-LABEL: func.func @test_q_constant_inf
func.func @test_q_constant_inf() -> !gmp.q {
  // CHECK: gmp.q.constant
  %q = gmp.q.constant #gmp.q<1, 0>
  return %q : !gmp.q
}

// CHECK-LABEL: func.func @test_q_make
func.func @test_q_make(%num: !gmp.z, %den: !gmp.z) -> !gmp.q {
  // CHECK: gmp.q.make
  %q = gmp.q.make %num, %den
  return %q : !gmp.q
}

//===----------------------------------------------------------------------===//
// Q Component Accessors
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_num
func.func @test_q_num(%q: !gmp.q) -> !gmp.z {
  // CHECK: gmp.q.num
  %num = gmp.q.num %q
  return %num : !gmp.z
}

// CHECK-LABEL: func.func @test_q_den
func.func @test_q_den(%q: !gmp.q) -> !gmp.z {
  // CHECK: gmp.q.den
  %den = gmp.q.den %q
  return %den : !gmp.z
}

//===----------------------------------------------------------------------===//
// Q Arithmetic Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_add
func.func @test_q_add(%a: !gmp.q, %b: !gmp.q) -> !gmp.q {
  // CHECK: gmp.q.add
  %sum = gmp.q.add %a, %b
  return %sum : !gmp.q
}

// CHECK-LABEL: func.func @test_q_sub
func.func @test_q_sub(%a: !gmp.q, %b: !gmp.q) -> !gmp.q {
  // CHECK: gmp.q.sub
  %diff = gmp.q.sub %a, %b
  return %diff : !gmp.q
}

// CHECK-LABEL: func.func @test_q_mul
func.func @test_q_mul(%a: !gmp.q, %b: !gmp.q) -> !gmp.q {
  // CHECK: gmp.q.mul
  %prod = gmp.q.mul %a, %b
  return %prod : !gmp.q
}

// CHECK-LABEL: func.func @test_q_div
func.func @test_q_div(%a: !gmp.q, %b: !gmp.q) -> !gmp.q {
  // CHECK: gmp.q.div
  %quot = gmp.q.div %a, %b
  return %quot : !gmp.q
}

//===----------------------------------------------------------------------===//
// Q Unary Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_neg
func.func @test_q_neg(%q: !gmp.q) -> !gmp.q {
  // CHECK: gmp.q.neg
  %neg = gmp.q.neg %q
  return %neg : !gmp.q
}

// CHECK-LABEL: func.func @test_q_abs
func.func @test_q_abs(%q: !gmp.q) -> !gmp.q {
  // CHECK: gmp.q.abs
  %abs = gmp.q.abs %q
  return %abs : !gmp.q
}

// CHECK-LABEL: func.func @test_q_inv
func.func @test_q_inv(%q: !gmp.q) -> !gmp.q {
  // CHECK: gmp.q.inv
  %inv = gmp.q.inv %q
  return %inv : !gmp.q
}

//===----------------------------------------------------------------------===//
// Q Comparison Operations
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

// CHECK-LABEL: func.func @test_q_lt
func.func @test_q_lt(%a: !gmp.q, %b: !gmp.q) -> i1 {
  // CHECK: gmp.q.lt
  %lt = gmp.q.lt %a, %b
  return %lt : i1
}

// CHECK-LABEL: func.func @test_q_leq
func.func @test_q_leq(%a: !gmp.q, %b: !gmp.q) -> i1 {
  // CHECK: gmp.q.leq
  %leq = gmp.q.leq %a, %b
  return %leq : i1
}

// CHECK-LABEL: func.func @test_q_gt
func.func @test_q_gt(%a: !gmp.q, %b: !gmp.q) -> i1 {
  // CHECK: gmp.q.gt
  %gt = gmp.q.gt %a, %b
  return %gt : i1
}

// CHECK-LABEL: func.func @test_q_geq
func.func @test_q_geq(%a: !gmp.q, %b: !gmp.q) -> i1 {
  // CHECK: gmp.q.geq
  %geq = gmp.q.geq %a, %b
  return %geq : i1
}

//===----------------------------------------------------------------------===//
// Q Classification Operations
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Q Rounding Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_floor
func.func @test_q_floor(%q: !gmp.q) -> !gmp.z {
  // CHECK: gmp.q.floor
  %floor = gmp.q.floor %q
  return %floor : !gmp.z
}

// CHECK-LABEL: func.func @test_q_ceil
func.func @test_q_ceil(%q: !gmp.q) -> !gmp.z {
  // CHECK: gmp.q.ceil
  %ceil = gmp.q.ceil %q
  return %ceil : !gmp.z
}

// CHECK-LABEL: func.func @test_q_trunc
func.func @test_q_trunc(%q: !gmp.q) -> !gmp.z {
  // CHECK: gmp.q.trunc
  %trunc = gmp.q.trunc %q
  return %trunc : !gmp.z
}

// CHECK-LABEL: func.func @test_q_round
func.func @test_q_round(%q: !gmp.q) -> !gmp.z {
  // CHECK: gmp.q.round
  %round = gmp.q.round %q
  return %round : !gmp.z
}

//===----------------------------------------------------------------------===//
// Q Conversion Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_q_to_bigint
func.func @test_q_to_bigint(%q: !gmp.q) -> !gmp.z {
  // CHECK: gmp.q.to_bigint
  %z = gmp.q.to_bigint %q
  return %z : !gmp.z
}

// CHECK-LABEL: func.func @test_q_to_f64
func.func @test_q_to_f64(%q: !gmp.q) -> f64 {
  // CHECK: gmp.q.to_f64
  %f = gmp.q.to_f64 %q
  return %f : f64
}

// CHECK-LABEL: func.func @test_q_from_z
func.func @test_q_from_z(%z: !gmp.z) -> !gmp.q {
  // CHECK: gmp.q.from_z
  %q = gmp.q.from_z %z
  return %q : !gmp.q
}
