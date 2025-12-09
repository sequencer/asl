// RUN: asl-opt %s --convert-gmp-to-emitc | FileCheck %s

// ============================================================================
// Z Constant Operations
// ============================================================================

// CHECK-LABEL: func.func @test_z_constant
func.func @test_z_constant() -> !gmp.z {
  // CHECK: emitc.variable
  // CHECK: emitc.load
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  %c = gmp.z.constant #gmp.z<42>
  return %c : !gmp.z
}

// CHECK-LABEL: func.func @test_z_constant_negative
func.func @test_z_constant_negative() -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  %c = gmp.z.constant #gmp.z<-123>
  return %c : !gmp.z
}

// CHECK-LABEL: func.func @test_z_constant_large
func.func @test_z_constant_large() -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  %c = gmp.z.constant #gmp.z<123456789012345678901234567890>
  return %c : !gmp.z
}

// CHECK-LABEL: func.func @test_z_from_int
func.func @test_z_from_int(%x: i64) -> !gmp.z {
  // CHECK: emitc.variable
  // CHECK: emitc.load
  // CHECK: emitc.call_opaque "mpz_init_set_si"
  %z = gmp.z.from_int %x : i64
  return %z : !gmp.z
}

// ============================================================================
// Z Arithmetic Operations
// ============================================================================

// CHECK-LABEL: func.func @test_z_add
func.func @test_z_add(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.variable
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_add"
  %sum = gmp.z.add %a, %b
  return %sum : !gmp.z
}

// CHECK-LABEL: func.func @test_z_sub
func.func @test_z_sub(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_sub"
  %diff = gmp.z.sub %a, %b
  return %diff : !gmp.z
}

// CHECK-LABEL: func.func @test_z_mul
func.func @test_z_mul(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_mul"
  %prod = gmp.z.mul %a, %b
  return %prod : !gmp.z
}

// CHECK-LABEL: func.func @test_z_neg
func.func @test_z_neg(%x: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_neg"
  %neg = gmp.z.neg %x
  return %neg : !gmp.z
}

// CHECK-LABEL: func.func @test_z_abs
func.func @test_z_abs(%x: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_abs"
  %abs = gmp.z.abs %x
  return %abs : !gmp.z
}

// CHECK-LABEL: func.func @test_z_succ
func.func @test_z_succ(%x: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_add_ui"
  %next = gmp.z.succ %x
  return %next : !gmp.z
}

// CHECK-LABEL: func.func @test_z_pred
func.func @test_z_pred(%x: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_sub_ui"
  %prev = gmp.z.pred %x
  return %prev : !gmp.z
}

// ============================================================================
// Z Division Operations
// ============================================================================

// CHECK-LABEL: func.func @test_z_div
func.func @test_z_div(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_tdiv_q"
  %quot = gmp.z.div %a, %b
  return %quot : !gmp.z
}

// CHECK-LABEL: func.func @test_z_rem
func.func @test_z_rem(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_tdiv_r"
  %rem = gmp.z.rem %a, %b
  return %rem : !gmp.z
}

// CHECK-LABEL: func.func @test_z_fdiv
func.func @test_z_fdiv(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_fdiv_q"
  %quot = gmp.z.fdiv %a, %b
  return %quot : !gmp.z
}

// CHECK-LABEL: func.func @test_z_frem
func.func @test_z_frem(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_fdiv_r"
  %rem = gmp.z.frem %a, %b
  return %rem : !gmp.z
}

// CHECK-LABEL: func.func @test_z_cdiv
func.func @test_z_cdiv(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_cdiv_q"
  %quot = gmp.z.cdiv %a, %b
  return %quot : !gmp.z
}

// CHECK-LABEL: func.func @test_z_crem
func.func @test_z_crem(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_cdiv_r"
  %rem = gmp.z.crem %a, %b
  return %rem : !gmp.z
}

// CHECK-LABEL: func.func @test_z_ediv
func.func @test_z_ediv(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_fdiv_q"
  %quot = gmp.z.ediv %a, %b
  return %quot : !gmp.z
}

// CHECK-LABEL: func.func @test_z_erem
func.func @test_z_erem(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_fdiv_r"
  %rem = gmp.z.erem %a, %b
  return %rem : !gmp.z
}

// CHECK-LABEL: func.func @test_z_divexact
func.func @test_z_divexact(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_divexact"
  %quot = gmp.z.divexact %a, %b
  return %quot : !gmp.z
}

// ============================================================================
// Z Comparison Operations
// ============================================================================

// CHECK-LABEL: func.func @test_z_compare
func.func @test_z_compare(%a: !gmp.z, %b: !gmp.z) -> i32 {
  // CHECK: emitc.call_opaque "mpz_cmp"
  %cmp = gmp.z.compare %a, %b
  return %cmp : i32
}

// CHECK-LABEL: func.func @test_z_equal
func.func @test_z_equal(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: emitc.call_opaque "mpz_cmp"
  // CHECK: emitc.cmp eq
  %eq = gmp.z.equal %a, %b
  return %eq : i1
}

// CHECK-LABEL: func.func @test_z_lt
func.func @test_z_lt(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: emitc.call_opaque "mpz_cmp"
  // CHECK: emitc.cmp lt
  %lt = gmp.z.lt %a, %b
  return %lt : i1
}

// CHECK-LABEL: func.func @test_z_leq
func.func @test_z_leq(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: emitc.call_opaque "mpz_cmp"
  // CHECK: emitc.cmp le
  %le = gmp.z.leq %a, %b
  return %le : i1
}

// CHECK-LABEL: func.func @test_z_gt
func.func @test_z_gt(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: emitc.call_opaque "mpz_cmp"
  // CHECK: emitc.cmp gt
  %gt = gmp.z.gt %a, %b
  return %gt : i1
}

// CHECK-LABEL: func.func @test_z_geq
func.func @test_z_geq(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: emitc.call_opaque "mpz_cmp"
  // CHECK: emitc.cmp ge
  %ge = gmp.z.geq %a, %b
  return %ge : i1
}

// ============================================================================
// Z Bitwise Operations
// ============================================================================

// CHECK-LABEL: func.func @test_z_logand
func.func @test_z_logand(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_and"
  %and = gmp.z.logand %a, %b
  return %and : !gmp.z
}

// CHECK-LABEL: func.func @test_z_logor
func.func @test_z_logor(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_ior"
  %or = gmp.z.logor %a, %b
  return %or : !gmp.z
}

// CHECK-LABEL: func.func @test_z_logxor
func.func @test_z_logxor(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_xor"
  %xor = gmp.z.logxor %a, %b
  return %xor : !gmp.z
}

// CHECK-LABEL: func.func @test_z_lognot
func.func @test_z_lognot(%x: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_com"
  %not = gmp.z.lognot %x
  return %not : !gmp.z
}

// ============================================================================
// Z Shift Operations
// ============================================================================

// CHECK-LABEL: func.func @test_z_shift_left
func.func @test_z_shift_left(%x: !gmp.z, %n: i64) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_mul_2exp"
  %shl = gmp.z.shift_left %x, %n
  return %shl : !gmp.z
}

// CHECK-LABEL: func.func @test_z_shift_right
func.func @test_z_shift_right(%x: !gmp.z, %n: i64) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_fdiv_q_2exp"
  %shr = gmp.z.shift_right %x, %n
  return %shr : !gmp.z
}

// CHECK-LABEL: func.func @test_z_shift_right_trunc
func.func @test_z_shift_right_trunc(%x: !gmp.z, %n: i64) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_tdiv_q_2exp"
  %shr = gmp.z.shift_right_trunc %x, %n
  return %shr : !gmp.z
}

// ============================================================================
// Z Power Operation
// ============================================================================

// CHECK-LABEL: func.func @test_z_pow
func.func @test_z_pow(%base: !gmp.z, %exp: i64) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_pow_ui"
  %result = gmp.z.pow %base, %exp
  return %result : !gmp.z
}

// ============================================================================
// Z Number-Theoretic Operations
// ============================================================================

// CHECK-LABEL: func.func @test_z_gcd
func.func @test_z_gcd(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_gcd"
  %gcd = gmp.z.gcd %a, %b
  return %gcd : !gmp.z
}

// CHECK-LABEL: func.func @test_z_lcm
func.func @test_z_lcm(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_lcm"
  %lcm = gmp.z.lcm %a, %b
  return %lcm : !gmp.z
}

// CHECK-LABEL: func.func @test_z_sqrt
func.func @test_z_sqrt(%x: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_init"
  // CHECK: emitc.call_opaque "mpz_sqrt"
  %sqrt = gmp.z.sqrt %x
  return %sqrt : !gmp.z
}

// CHECK-LABEL: func.func @test_z_divisible
func.func @test_z_divisible(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: emitc.call_opaque "mpz_divisible_p"
  // CHECK: emitc.cmp ne
  %div = gmp.z.divisible %a, %b
  return %div : i1
}

// CHECK-LABEL: func.func @test_z_congruent
func.func @test_z_congruent(%a: !gmp.z, %b: !gmp.z, %m: !gmp.z) -> i1 {
  // CHECK: emitc.call_opaque "mpz_congruent_p"
  // CHECK: emitc.cmp ne
  %cong = gmp.z.congruent %a, %b, %m
  return %cong : i1
}

// ============================================================================
// Z Bit Operations
// ============================================================================

// CHECK-LABEL: func.func @test_z_testbit
func.func @test_z_testbit(%x: !gmp.z, %n: i64) -> i1 {
  // CHECK: emitc.call_opaque "mpz_tstbit"
  // CHECK: emitc.cmp ne
  %bit = gmp.z.testbit %x, %n
  return %bit : i1
}

// CHECK-LABEL: func.func @test_z_popcount
func.func @test_z_popcount(%x: !gmp.z) -> i64 {
  // CHECK: emitc.call_opaque "mpz_popcount"
  %count = gmp.z.popcount %x
  return %count : i64
}

// CHECK-LABEL: func.func @test_z_numbits
func.func @test_z_numbits(%x: !gmp.z) -> i64 {
  // CHECK: emitc.call_opaque "mpz_sizeinbase"
  %bits = gmp.z.numbits %x
  return %bits : i64
}

// ============================================================================
// Test Chained Operations
// ============================================================================

// CHECK-LABEL: func.func @test_z_chained_arithmetic
func.func @test_z_chained_arithmetic(%a: !gmp.z, %b: !gmp.z, %c: !gmp.z) -> !gmp.z {
  // CHECK: emitc.call_opaque "mpz_add"
  // CHECK: emitc.call_opaque "mpz_mul"
  %sum = gmp.z.add %a, %b
  %prod = gmp.z.mul %sum, %c
  return %prod : !gmp.z
}
