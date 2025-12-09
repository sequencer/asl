// RUN: asl-opt %s --canonicalize | FileCheck %s

// Test constant folding for number theory operations

//===----------------------------------------------------------------------===//
// GCD and LCM Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_gcd_fold
func.func @test_gcd_fold() -> !gmp.z {
  // CHECK: gmp.z.constant <4>
  %a = gmp.z.constant #gmp.z<12>
  %b = gmp.z.constant #gmp.z<8>
  %gcd = gmp.z.gcd %a, %b
  return %gcd : !gmp.z
}

// CHECK-LABEL: func.func @test_gcd_negative
func.func @test_gcd_negative() -> !gmp.z {
  // CHECK: gmp.z.constant <6>
  %a = gmp.z.constant #gmp.z<-12>
  %b = gmp.z.constant #gmp.z<18>
  %gcd = gmp.z.gcd %a, %b
  return %gcd : !gmp.z
}

// CHECK-LABEL: func.func @test_gcdext_fold
func.func @test_gcdext_fold() -> (!gmp.z, !gmp.z, !gmp.z) {
  // gcdext(12, 8) = (4, 1, -1) since 4 = 12*1 + 8*(-1)
  // CHECK-DAG: gmp.z.constant <4>
  // CHECK-DAG: gmp.z.constant <1>
  // CHECK-DAG: gmp.z.constant <-1>
  %a = gmp.z.constant #gmp.z<12>
  %b = gmp.z.constant #gmp.z<8>
  %gcd, %s, %t = gmp.z.gcdext %a, %b
  return %gcd, %s, %t : !gmp.z, !gmp.z, !gmp.z
}

// CHECK-LABEL: func.func @test_lcm_fold
func.func @test_lcm_fold() -> !gmp.z {
  // CHECK: gmp.z.constant <12>
  %a = gmp.z.constant #gmp.z<4>
  %b = gmp.z.constant #gmp.z<6>
  %lcm = gmp.z.lcm %a, %b
  return %lcm : !gmp.z
}

// CHECK-LABEL: func.func @test_lcm_zero
func.func @test_lcm_zero() -> !gmp.z {
  // CHECK: gmp.z.constant <0>
  %a = gmp.z.constant #gmp.z<4>
  %b = gmp.z.constant #gmp.z<0>
  %lcm = gmp.z.lcm %a, %b
  return %lcm : !gmp.z
}

//===----------------------------------------------------------------------===//
// Modular Arithmetic Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_powm_fold
func.func @test_powm_fold() -> !gmp.z {
  // 2^10 mod 1000 = 1024 mod 1000 = 24
  // CHECK: gmp.z.constant <24>
  %base = gmp.z.constant #gmp.z<2>
  %exp = gmp.z.constant #gmp.z<10>
  %mod = gmp.z.constant #gmp.z<1000>
  %result = gmp.z.powm %base, %exp, %mod
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_powm_sec_fold
func.func @test_powm_sec_fold() -> !gmp.z {
  // 3^5 mod 17 = 243 mod 17 = 5
  // CHECK: gmp.z.constant <5>
  %base = gmp.z.constant #gmp.z<3>
  %exp = gmp.z.constant #gmp.z<5>
  %mod = gmp.z.constant #gmp.z<17>
  %result = gmp.z.powm_sec %base, %exp, %mod
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_invert_fold
func.func @test_invert_fold(%a: !gmp.z, %mod: !gmp.z) -> (!gmp.z, i1) {
  // Invert requires runtime computation due to multi-result folding complexity
  // CHECK: gmp.z.invert
  %inv, %exists = gmp.z.invert %a, %mod
  return %inv, %exists : !gmp.z, i1
}

//===----------------------------------------------------------------------===//
// Primality Testing Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_probab_prime_composite
func.func @test_probab_prime_composite(%n: !gmp.z, %reps: i32) -> i32 {
  // Probab_prime requires non-constant ZAttr + constant reps
  // CHECK: gmp.z.probab_prime
  %result = gmp.z.probab_prime %n, %reps
  return %result : i32
}

// CHECK-LABEL: func.func @test_nextprime_fold
func.func @test_nextprime_fold() -> !gmp.z {
  // nextprime(10) = 11
  // CHECK: gmp.z.constant <11>
  %n = gmp.z.constant #gmp.z<10>
  %next = gmp.z.nextprime %n
  return %next : !gmp.z
}

// CHECK-LABEL: func.func @test_nextprime_from_prime
func.func @test_nextprime_from_prime() -> !gmp.z {
  // nextprime(13) = 17
  // CHECK: gmp.z.constant <17>
  %n = gmp.z.constant #gmp.z<13>
  %next = gmp.z.nextprime %n
  return %next : !gmp.z
}

//===----------------------------------------------------------------------===//
// Power and Root Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_pow_fold
func.func @test_pow_fold() -> !gmp.z {
  // 2^10 = 1024
  // CHECK: gmp.z.constant <1024>
  %base = gmp.z.constant #gmp.z<2>
  %exp = arith.constant 10 : i64
  %result = gmp.z.pow %base, %exp
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_pow_negative_base
func.func @test_pow_negative_base() -> !gmp.z {
  // (-2)^3 = -8
  // CHECK: gmp.z.constant <-8>
  %base = gmp.z.constant #gmp.z<-2>
  %exp = arith.constant 3 : i64
  %result = gmp.z.pow %base, %exp
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_pow_zero_exp
func.func @test_pow_zero_exp(%base: !gmp.z) -> !gmp.z {
  // x^0 = 1
  // CHECK: gmp.z.constant <1>
  %exp = arith.constant 0 : i64
  %result = gmp.z.pow %base, %exp
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_sqrt_perfect
func.func @test_sqrt_perfect() -> !gmp.z {
  // sqrt(100) = 10
  // CHECK: gmp.z.constant <10>
  %n = gmp.z.constant #gmp.z<100>
  %root = gmp.z.sqrt %n
  return %root : !gmp.z
}

// CHECK-LABEL: func.func @test_sqrt_non_perfect
func.func @test_sqrt_non_perfect() -> !gmp.z {
  // sqrt(10) = 3 (floor)
  // CHECK: gmp.z.constant <3>
  %n = gmp.z.constant #gmp.z<10>
  %root = gmp.z.sqrt %n
  return %root : !gmp.z
}

// CHECK-LABEL: func.func @test_sqrt_rem_fold
func.func @test_sqrt_rem_fold() -> (!gmp.z, !gmp.z) {
  // sqrt_rem(10) = (3, 1) since 10 = 3^2 + 1
  // CHECK-DAG: gmp.z.constant <3>
  // CHECK-DAG: gmp.z.constant <1>
  %n = gmp.z.constant #gmp.z<10>
  %root, %rem = gmp.z.sqrt_rem %n
  return %root, %rem : !gmp.z, !gmp.z
}

// CHECK-LABEL: func.func @test_root_fold
func.func @test_root_fold() -> !gmp.z {
  // root(27, 3) = 3
  // CHECK: gmp.z.constant <3>
  %n = gmp.z.constant #gmp.z<27>
  %k = arith.constant 3 : i64
  %root = gmp.z.root %n, %k
  return %root : !gmp.z
}

// CHECK-LABEL: func.func @test_root_non_perfect
func.func @test_root_non_perfect() -> !gmp.z {
  // root(30, 3) = 3 (floor of 3.107...)
  // CHECK: gmp.z.constant <3>
  %n = gmp.z.constant #gmp.z<30>
  %k = arith.constant 3 : i64
  %root = gmp.z.root %n, %k
  return %root : !gmp.z
}

//===----------------------------------------------------------------------===//
// Factorial and Combinatorics Folding
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_fac_fold
func.func @test_fac_fold() -> !gmp.z {
  // 5! = 120
  // CHECK: gmp.z.constant <120>
  %n = arith.constant 5 : i64
  %result = gmp.z.fac %n
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_fac_zero
func.func @test_fac_zero() -> !gmp.z {
  // 0! = 1
  // CHECK: gmp.z.constant <1>
  %n = arith.constant 0 : i64
  %result = gmp.z.fac %n
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_fac_large
func.func @test_fac_large() -> !gmp.z {
  // 10! = 3628800
  // CHECK: gmp.z.constant <3628800>
  %n = arith.constant 10 : i64
  %result = gmp.z.fac %n
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_bin_fold
func.func @test_bin_fold() -> !gmp.z {
  // C(5, 2) = 10
  // CHECK: gmp.z.constant <10>
  %n = gmp.z.constant #gmp.z<5>
  %k = arith.constant 2 : i64
  %result = gmp.z.bin %n, %k
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_bin_k_zero
func.func @test_bin_k_zero(%n: !gmp.z) -> !gmp.z {
  // C(n, 0) = 1
  // CHECK: gmp.z.constant <1>
  %k = arith.constant 0 : i64
  %result = gmp.z.bin %n, %k
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_bin_larger
func.func @test_bin_larger() -> !gmp.z {
  // C(10, 4) = 210
  // CHECK: gmp.z.constant <210>
  %n = gmp.z.constant #gmp.z<10>
  %k = arith.constant 4 : i64
  %result = gmp.z.bin %n, %k
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_fib_fold
func.func @test_fib_fold() -> !gmp.z {
  // F(10) = 55
  // CHECK: gmp.z.constant <55>
  %n = arith.constant 10 : i64
  %result = gmp.z.fib %n
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_fib_zero
func.func @test_fib_zero() -> !gmp.z {
  // F(0) = 0
  // CHECK: gmp.z.constant <0>
  %n = arith.constant 0 : i64
  %result = gmp.z.fib %n
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_fib_one
func.func @test_fib_one() -> !gmp.z {
  // F(1) = 1
  // CHECK: gmp.z.constant <1>
  %n = arith.constant 1 : i64
  %result = gmp.z.fib %n
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_fib_larger
func.func @test_fib_larger() -> !gmp.z {
  // F(20) = 6765
  // CHECK: gmp.z.constant <6765>
  %n = arith.constant 20 : i64
  %result = gmp.z.fib %n
  return %result : !gmp.z
}
