// RUN: asl-opt %s --canonicalize --convert-gmp-to-emitc | FileCheck %s

// ============================================================================
// Test that constant folding happens before EmitC conversion
// ============================================================================

// Z Module Constant Folding Tests
// ============================================================================

// CHECK-LABEL: func.func @fold_z_add_to_emitc
func.func @fold_z_add_to_emitc() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<10>
  %b = gmp.z.constant #gmp.z<20>
  // The add should be folded to constant 30, then converted to EmitC
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_add
  %sum = gmp.z.add %a, %b
  return %sum : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_sub_to_emitc
func.func @fold_z_sub_to_emitc() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<100>
  %b = gmp.z.constant #gmp.z<42>
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_sub
  %diff = gmp.z.sub %a, %b
  return %diff : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_mul_to_emitc
func.func @fold_z_mul_to_emitc() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<7>
  %b = gmp.z.constant #gmp.z<8>
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_mul
  %prod = gmp.z.mul %a, %b
  return %prod : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_neg_to_emitc
func.func @fold_z_neg_to_emitc() -> !gmp.z {
  %x = gmp.z.constant #gmp.z<42>
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_neg
  %neg = gmp.z.neg %x
  return %neg : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_abs_to_emitc
func.func @fold_z_abs_to_emitc() -> !gmp.z {
  %x = gmp.z.constant #gmp.z<-42>
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_abs
  %abs = gmp.z.abs %x
  return %abs : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_div_to_emitc
func.func @fold_z_div_to_emitc() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<100>
  %b = gmp.z.constant #gmp.z<7>
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_tdiv_q
  %quot = gmp.z.div %a, %b
  return %quot : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_compare_to_emitc
func.func @fold_z_compare_to_emitc() -> i32 {
  %a = gmp.z.constant #gmp.z<10>
  %b = gmp.z.constant #gmp.z<20>
  // Comparison of constants should fold to -1 (10 < 20)
  // CHECK: arith.constant -1 : i32
  // CHECK-NOT: mpz_cmp
  %cmp = gmp.z.compare %a, %b
  return %cmp : i32
}

// CHECK-LABEL: func.func @fold_z_equal_to_emitc
func.func @fold_z_equal_to_emitc() -> i1 {
  %a = gmp.z.constant #gmp.z<42>
  %b = gmp.z.constant #gmp.z<42>
  // Equal constants should fold to true
  // CHECK: arith.constant true
  // CHECK-NOT: mpz_cmp
  %eq = gmp.z.equal %a, %b
  return %eq : i1
}

// CHECK-LABEL: func.func @fold_z_gcd_to_emitc
func.func @fold_z_gcd_to_emitc() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<48>
  %b = gmp.z.constant #gmp.z<18>
  // GCD(48, 18) = 6
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_gcd
  %gcd = gmp.z.gcd %a, %b
  return %gcd : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_pow_to_emitc
func.func @fold_z_pow_to_emitc() -> !gmp.z {
  %base = gmp.z.constant #gmp.z<2>
  %c10 = arith.constant 10 : i64
  // 2^10 = 1024
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_pow_ui
  %result = gmp.z.pow %base, %c10
  return %result : !gmp.z
}

// ============================================================================
// Identity/Zero Folding Tests
// ============================================================================

// CHECK-LABEL: func.func @fold_z_add_zero_to_emitc
func.func @fold_z_add_zero_to_emitc(%x: !gmp.z) -> !gmp.z {
  %zero = gmp.z.constant #gmp.z<0>
  // Adding zero should return the original value
  // CHECK-NOT: mpz_add
  // CHECK: return %arg0
  %sum = gmp.z.add %x, %zero
  return %sum : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_mul_one_to_emitc
func.func @fold_z_mul_one_to_emitc(%x: !gmp.z) -> !gmp.z {
  %one = gmp.z.constant #gmp.z<1>
  // Multiplying by one should return the original value
  // CHECK-NOT: mpz_mul
  // CHECK: return %arg0
  %prod = gmp.z.mul %x, %one
  return %prod : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_mul_zero_to_emitc
func.func @fold_z_mul_zero_to_emitc(%x: !gmp.z) -> !gmp.z {
  %zero = gmp.z.constant #gmp.z<0>
  // Multiplying by zero should return zero constant
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_mul
  %prod = gmp.z.mul %x, %zero
  return %prod : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_sub_self_to_emitc
func.func @fold_z_sub_self_to_emitc(%x: !gmp.z) -> !gmp.z {
  // x - x = 0
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_sub
  %diff = gmp.z.sub %x, %x
  return %diff : !gmp.z
}

// ============================================================================
// Q Module Constant Folding Tests
// ============================================================================

// CHECK-LABEL: func.func @fold_q_add_to_emitc
func.func @fold_q_add_to_emitc() -> !gmp.q {
  %a = gmp.q.constant #gmp.q<1, 2>
  %b = gmp.q.constant #gmp.q<1, 4>
  // 1/2 + 1/4 = 3/4
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_set_str"
  // CHECK-NOT: mpq_add
  %sum = gmp.q.add %a, %b
  return %sum : !gmp.q
}

// CHECK-LABEL: func.func @fold_q_mul_to_emitc
func.func @fold_q_mul_to_emitc() -> !gmp.q {
  %a = gmp.q.constant #gmp.q<2, 3>
  %b = gmp.q.constant #gmp.q<3, 4>
  // 2/3 * 3/4 = 1/2
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_set_str"
  // CHECK-NOT: mpq_mul
  %prod = gmp.q.mul %a, %b
  return %prod : !gmp.q
}

// CHECK-LABEL: func.func @fold_q_neg_to_emitc
func.func @fold_q_neg_to_emitc() -> !gmp.q {
  %x = gmp.q.constant #gmp.q<3, 4>
  // -3/4
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_set_str"
  // CHECK-NOT: mpq_neg
  %neg = gmp.q.neg %x
  return %neg : !gmp.q
}

// CHECK-LABEL: func.func @fold_q_inv_to_emitc
func.func @fold_q_inv_to_emitc() -> !gmp.q {
  %x = gmp.q.constant #gmp.q<3, 4>
  // inv(3/4) = 4/3
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_set_str"
  // CHECK-NOT: mpq_inv
  %inv = gmp.q.inv %x
  return %inv : !gmp.q
}

// CHECK-LABEL: func.func @fold_q_equal_to_emitc
func.func @fold_q_equal_to_emitc() -> i1 {
  %a = gmp.q.constant #gmp.q<1, 2>
  %b = gmp.q.constant #gmp.q<2, 4>
  // 1/2 == 2/4 (after canonicalization)
  // CHECK: arith.constant true
  // CHECK-NOT: mpq_cmp
  %eq = gmp.q.equal %a, %b
  return %eq : i1
}

// CHECK-LABEL: func.func @fold_q_floor_to_emitc
func.func @fold_q_floor_to_emitc() -> !gmp.z {
  %x = gmp.q.constant #gmp.q<7, 2>
  // floor(7/2) = 3
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_fdiv_q
  %floor = gmp.q.floor %x
  return %floor : !gmp.z
}

// CHECK-LABEL: func.func @fold_q_ceil_to_emitc
func.func @fold_q_ceil_to_emitc() -> !gmp.z {
  %x = gmp.q.constant #gmp.q<7, 2>
  // ceil(7/2) = 4
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_cdiv_q
  %ceil = gmp.q.ceil %x
  return %ceil : !gmp.z
}

// ============================================================================
// Chained Folding Tests
// ============================================================================

// CHECK-LABEL: func.func @fold_chained_z_ops_to_emitc
func.func @fold_chained_z_ops_to_emitc() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<10>
  %b = gmp.z.constant #gmp.z<20>
  %c = gmp.z.constant #gmp.z<5>
  // (10 + 20) * 5 = 150
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_add
  // CHECK-NOT: mpz_mul
  %sum = gmp.z.add %a, %b
  %prod = gmp.z.mul %sum, %c
  return %prod : !gmp.z
}

// CHECK-LABEL: func.func @fold_chained_q_ops_to_emitc
func.func @fold_chained_q_ops_to_emitc() -> !gmp.q {
  %a = gmp.q.constant #gmp.q<1, 2>
  %b = gmp.q.constant #gmp.q<1, 3>
  %c = gmp.q.constant #gmp.q<1, 4>
  // (1/2 + 1/3) * 1/4 = 5/6 * 1/4 = 5/24
  // CHECK: emitc.call_opaque "mpq_init"
  // CHECK: emitc.call_opaque "mpq_set_str"
  // CHECK-NOT: mpq_add
  // CHECK-NOT: mpq_mul
  %sum = gmp.q.add %a, %b
  %prod = gmp.q.mul %sum, %c
  return %prod : !gmp.q
}

// ============================================================================
// Large Number Folding Tests
// ============================================================================

// CHECK-LABEL: func.func @fold_z_large_add_to_emitc
func.func @fold_z_large_add_to_emitc() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<12345678901234567890>
  %b = gmp.z.constant #gmp.z<98765432109876543210>
  // Large number addition should still fold
  // CHECK: emitc.call_opaque "mpz_init_set_str"
  // CHECK-NOT: mpz_add
  %sum = gmp.z.add %a, %b
  return %sum : !gmp.z
}

// ============================================================================
// Mixed Runtime/Constant Tests
// ============================================================================

// CHECK-LABEL: func.func @partial_fold_z_to_emitc
func.func @partial_fold_z_to_emitc(%x: !gmp.z) -> !gmp.z {
  // Constants are folded, but runtime ops remain
  %a = gmp.z.constant #gmp.z<10>
  %b = gmp.z.constant #gmp.z<20>
  // 10 + 20 = 30 (folded)
  %const_sum = gmp.z.add %a, %b
  // 30 + x (runtime op, not folded)
  // CHECK: emitc.call_opaque "mpz_add"
  %result = gmp.z.add %const_sum, %x
  return %result : !gmp.z
}
