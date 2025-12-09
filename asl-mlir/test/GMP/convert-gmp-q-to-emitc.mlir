// RUN: asl-opt %s --convert-gmp-to-emitc | FileCheck %s

// ============================================================================
// Q Constant Operations
// ============================================================================

// CHECK-LABEL: func.func @test_q_constant
func.func @test_q_constant() -> !gmp.q {
  // CHECK: emitc.variable
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_set_str"
  %q = gmp.q.constant #gmp.q<1, 2>
  return %q : !gmp.q
}

// CHECK-LABEL: func.func @test_q_constant_negative
func.func @test_q_constant_negative() -> !gmp.q {
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_set_str"
  %q = gmp.q.constant #gmp.q<-3, 4>
  return %q : !gmp.q
}

// CHECK-LABEL: func.func @test_q_constant_integer
func.func @test_q_constant_integer() -> !gmp.q {
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_set_str"
  %q = gmp.q.constant #gmp.q<5, 1>
  return %q : !gmp.q
}

// ============================================================================
// Q Arithmetic Operations
// ============================================================================

// CHECK-LABEL: func.func @test_q_add
func.func @test_q_add(%a: !gmp.q, %b: !gmp.q) -> !gmp.q {
  // CHECK: emitc.variable
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_add"
  %sum = gmp.q.add %a, %b
  return %sum : !gmp.q
}

// CHECK-LABEL: func.func @test_q_sub
func.func @test_q_sub(%a: !gmp.q, %b: !gmp.q) -> !gmp.q {
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_sub"
  %diff = gmp.q.sub %a, %b
  return %diff : !gmp.q
}

// CHECK-LABEL: func.func @test_q_mul
func.func @test_q_mul(%a: !gmp.q, %b: !gmp.q) -> !gmp.q {
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_mul"
  %prod = gmp.q.mul %a, %b
  return %prod : !gmp.q
}

// CHECK-LABEL: func.func @test_q_div
func.func @test_q_div(%a: !gmp.q, %b: !gmp.q) -> !gmp.q {
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_div"
  %quot = gmp.q.div %a, %b
  return %quot : !gmp.q
}

// ============================================================================
// Q Unary Operations
// ============================================================================

// CHECK-LABEL: func.func @test_q_neg
func.func @test_q_neg(%x: !gmp.q) -> !gmp.q {
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_neg"
  %neg = gmp.q.neg %x
  return %neg : !gmp.q
}

// CHECK-LABEL: func.func @test_q_abs
func.func @test_q_abs(%x: !gmp.q) -> !gmp.q {
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_abs"
  %abs = gmp.q.abs %x
  return %abs : !gmp.q
}

// CHECK-LABEL: func.func @test_q_inv
func.func @test_q_inv(%x: !gmp.q) -> !gmp.q {
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_inv"
  %inv = gmp.q.inv %x
  return %inv : !gmp.q
}

// ============================================================================
// Q Comparison Operations
// ============================================================================

// CHECK-LABEL: func.func @test_q_compare
func.func @test_q_compare(%a: !gmp.q, %b: !gmp.q) -> i32 {
  // CHECK: emitc.call_opaque "mpq_cmp"
  %cmp = gmp.q.compare %a, %b
  return %cmp : i32
}

// CHECK-LABEL: func.func @test_q_equal
func.func @test_q_equal(%a: !gmp.q, %b: !gmp.q) -> i1 {
  // CHECK: emitc.call_opaque "mpq_cmp"
  // CHECK: emitc.cmp eq
  %eq = gmp.q.equal %a, %b
  return %eq : i1
}

// CHECK-LABEL: func.func @test_q_lt
func.func @test_q_lt(%a: !gmp.q, %b: !gmp.q) -> i1 {
  // CHECK: emitc.call_opaque "mpq_cmp"
  // CHECK: emitc.cmp lt
  %lt = gmp.q.lt %a, %b
  return %lt : i1
}

// CHECK-LABEL: func.func @test_q_leq
func.func @test_q_leq(%a: !gmp.q, %b: !gmp.q) -> i1 {
  // CHECK: emitc.call_opaque "mpq_cmp"
  // CHECK: emitc.cmp le
  %le = gmp.q.leq %a, %b
  return %le : i1
}

// CHECK-LABEL: func.func @test_q_gt
func.func @test_q_gt(%a: !gmp.q, %b: !gmp.q) -> i1 {
  // CHECK: emitc.call_opaque "mpq_cmp"
  // CHECK: emitc.cmp gt
  %gt = gmp.q.gt %a, %b
  return %gt : i1
}

// CHECK-LABEL: func.func @test_q_geq
func.func @test_q_geq(%a: !gmp.q, %b: !gmp.q) -> i1 {
  // CHECK: emitc.call_opaque "mpq_cmp"
  // CHECK: emitc.cmp ge
  %ge = gmp.q.geq %a, %b
  return %ge : i1
}

// ============================================================================
// Q Component Accessors
// ============================================================================

// CHECK-LABEL: func.func @test_q_num
func.func @test_q_num(%q: !gmp.q) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpq_numref"
  // CHECK: emitc.call_opaque "mpz_set"
  %num = gmp.q.num %q
  return %num : !gmp.z
}

// CHECK-LABEL: func.func @test_q_den
func.func @test_q_den(%q: !gmp.q) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpq_denref"
  // CHECK: emitc.call_opaque "mpz_set"
  %den = gmp.q.den %q
  return %den : !gmp.z
}

// ============================================================================
// Q Construction from Z
// ============================================================================

// CHECK-LABEL: func.func @test_q_make
func.func @test_q_make(%num: !gmp.z, %den: !gmp.z) -> !gmp.q {
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_numref"
  // CHECK: emitc.call_opaque "mpq_denref"
  // CHECK: emitc.call_opaque "mpz_set"
  // CHECK: emitc.call_opaque "mpz_set"
  // CHECK: emitc.call_opaque "mpq_canonicalize"
  %q = gmp.q.make %num, %den
  return %q : !gmp.q
}

// CHECK-LABEL: func.func @test_q_from_z
func.func @test_q_from_z(%z: !gmp.z) -> !gmp.q {
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_set_z"
  %q = gmp.q.from_z %z
  return %q : !gmp.q
}

// ============================================================================
// Q Rounding Operations
// ============================================================================

// CHECK-LABEL: func.func @test_q_floor
func.func @test_q_floor(%q: !gmp.q) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_fdiv_q"
  %floor = gmp.q.floor %q
  return %floor : !gmp.z
}

// CHECK-LABEL: func.func @test_q_ceil
func.func @test_q_ceil(%q: !gmp.q) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_cdiv_q"
  %ceil = gmp.q.ceil %q
  return %ceil : !gmp.z
}

// CHECK-LABEL: func.func @test_q_trunc
func.func @test_q_trunc(%q: !gmp.q) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_tdiv_q"
  %trunc = gmp.q.trunc %q
  return %trunc : !gmp.z
}

// ============================================================================
// Q Conversion Operations
// ============================================================================

// CHECK-LABEL: func.func @test_q_to_f64
func.func @test_q_to_f64(%q: !gmp.q) -> f64 {
  // CHECK: emitc.call_opaque "mpq_get_d"
  %f = gmp.q.to_f64 %q
  return %f : f64
}

// ============================================================================
// Test Chained Operations
// ============================================================================

// CHECK-LABEL: func.func @test_q_chained_arithmetic
func.func @test_q_chained_arithmetic(%a: !gmp.q, %b: !gmp.q, %c: !gmp.q) -> !gmp.q {
  // CHECK: emitc.call_opaque "mpq_add"
  // CHECK: emitc.call_opaque "mpq_mul"
  %sum = gmp.q.add %a, %b
  %prod = gmp.q.mul %sum, %c
  return %prod : !gmp.q
}

// ============================================================================
// Test Mixed Z and Q Operations
// ============================================================================

// CHECK-LABEL: func.func @test_mixed_z_q
func.func @test_mixed_z_q(%z: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpq_set_z"
  // CHECK: emitc.call_opaque "mpq_numref"
  // CHECK: emitc.call_opaque "mpz_set"
  %q = gmp.q.from_z %z
  %num = gmp.q.num %q
  return %num : !gmp.z
}
