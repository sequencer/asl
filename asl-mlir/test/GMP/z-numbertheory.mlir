// RUN: asl-opt %s | asl-opt | FileCheck %s

// Test number theory operations roundtrip

//===----------------------------------------------------------------------===//
// GCD and LCM Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_gcd
func.func @test_z_gcd(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.gcd %arg0, %arg1
  %gcd = gmp.z.gcd %a, %b
  return %gcd : !gmp.z
}

// CHECK-LABEL: func.func @test_z_gcdext
func.func @test_z_gcdext(%a: !gmp.z, %b: !gmp.z) -> (!gmp.z, !gmp.z, !gmp.z) {
  // CHECK: gmp.z.gcdext %arg0, %arg1
  %gcd, %s, %t = gmp.z.gcdext %a, %b
  return %gcd, %s, %t : !gmp.z, !gmp.z, !gmp.z
}

// CHECK-LABEL: func.func @test_z_lcm
func.func @test_z_lcm(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.lcm %arg0, %arg1
  %lcm = gmp.z.lcm %a, %b
  return %lcm : !gmp.z
}

//===----------------------------------------------------------------------===//
// Modular Arithmetic Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_powm
func.func @test_z_powm(%base: !gmp.z, %exp: !gmp.z, %mod: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.powm %arg0, %arg1, %arg2
  %result = gmp.z.powm %base, %exp, %mod
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_powm_sec
func.func @test_z_powm_sec(%base: !gmp.z, %exp: !gmp.z, %mod: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.powm_sec %arg0, %arg1, %arg2
  %result = gmp.z.powm_sec %base, %exp, %mod
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_invert
func.func @test_z_invert(%a: !gmp.z, %mod: !gmp.z) -> (!gmp.z, i1) {
  // CHECK: gmp.z.invert %arg0, %arg1
  %inv, %exists = gmp.z.invert %a, %mod
  return %inv, %exists : !gmp.z, i1
}

//===----------------------------------------------------------------------===//
// Primality Testing Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_probab_prime
func.func @test_z_probab_prime(%n: !gmp.z, %reps: i32) -> i32 {
  // CHECK: gmp.z.probab_prime %arg0, %arg1
  %result = gmp.z.probab_prime %n, %reps
  return %result : i32
}

// CHECK-LABEL: func.func @test_z_nextprime
func.func @test_z_nextprime(%n: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.nextprime %arg0
  %next = gmp.z.nextprime %n
  return %next : !gmp.z
}

//===----------------------------------------------------------------------===//
// Power and Root Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_pow
func.func @test_z_pow(%base: !gmp.z, %exp: i64) -> !gmp.z {
  // CHECK: gmp.z.pow %arg0, %arg1
  %result = gmp.z.pow %base, %exp
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_sqrt
func.func @test_z_sqrt(%n: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.sqrt %arg0
  %root = gmp.z.sqrt %n
  return %root : !gmp.z
}

// CHECK-LABEL: func.func @test_z_sqrt_rem
func.func @test_z_sqrt_rem(%n: !gmp.z) -> (!gmp.z, !gmp.z) {
  // CHECK: gmp.z.sqrt_rem %arg0
  %root, %rem = gmp.z.sqrt_rem %n
  return %root, %rem : !gmp.z, !gmp.z
}

// CHECK-LABEL: func.func @test_z_root
func.func @test_z_root(%n: !gmp.z, %k: i64) -> !gmp.z {
  // CHECK: gmp.z.root %arg0, %arg1
  %root = gmp.z.root %n, %k
  return %root : !gmp.z
}

//===----------------------------------------------------------------------===//
// Factorial and Combinatorics Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_fac
func.func @test_z_fac(%n: i64) -> !gmp.z {
  // CHECK: gmp.z.fac %arg0
  %result = gmp.z.fac %n
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_bin
func.func @test_z_bin(%n: !gmp.z, %k: i64) -> !gmp.z {
  // CHECK: gmp.z.bin %arg0, %arg1
  %result = gmp.z.bin %n, %k
  return %result : !gmp.z
}

// CHECK-LABEL: func.func @test_z_fib
func.func @test_z_fib(%n: i64) -> !gmp.z {
  // CHECK: gmp.z.fib %arg0
  %result = gmp.z.fib %n
  return %result : !gmp.z
}
