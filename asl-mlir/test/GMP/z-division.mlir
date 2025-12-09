// RUN: asl-opt %s | asl-opt | FileCheck %s

// Test division operations roundtrip

// CHECK-LABEL: func.func @test_z_div
func.func @test_z_div(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.div %arg0, %arg1
  %q = gmp.z.div %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @test_z_rem
func.func @test_z_rem(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.rem %arg0, %arg1
  %r = gmp.z.rem %a, %b
  return %r : !gmp.z
}

// CHECK-LABEL: func.func @test_z_div_rem
func.func @test_z_div_rem(%a: !gmp.z, %b: !gmp.z) -> (!gmp.z, !gmp.z) {
  // CHECK: gmp.z.div_rem %arg0, %arg1
  %q, %r = gmp.z.div_rem %a, %b
  return %q, %r : !gmp.z, !gmp.z
}

// CHECK-LABEL: func.func @test_z_fdiv
func.func @test_z_fdiv(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.fdiv %arg0, %arg1
  %q = gmp.z.fdiv %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @test_z_frem
func.func @test_z_frem(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.frem %arg0, %arg1
  %r = gmp.z.frem %a, %b
  return %r : !gmp.z
}

// CHECK-LABEL: func.func @test_z_cdiv
func.func @test_z_cdiv(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.cdiv %arg0, %arg1
  %q = gmp.z.cdiv %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @test_z_crem
func.func @test_z_crem(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.crem %arg0, %arg1
  %r = gmp.z.crem %a, %b
  return %r : !gmp.z
}

// CHECK-LABEL: func.func @test_z_ediv
func.func @test_z_ediv(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.ediv %arg0, %arg1
  %q = gmp.z.ediv %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @test_z_erem
func.func @test_z_erem(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.erem %arg0, %arg1
  %r = gmp.z.erem %a, %b
  return %r : !gmp.z
}

// CHECK-LABEL: func.func @test_z_divexact
func.func @test_z_divexact(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.divexact %arg0, %arg1
  %q = gmp.z.divexact %a, %b
  return %q : !gmp.z
}

// CHECK-LABEL: func.func @test_z_divisible
func.func @test_z_divisible(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: gmp.z.divisible %arg0, %arg1
  %r = gmp.z.divisible %a, %b
  return %r : i1
}

// CHECK-LABEL: func.func @test_z_congruent
func.func @test_z_congruent(%a: !gmp.z, %b: !gmp.z, %m: !gmp.z) -> i1 {
  // CHECK: gmp.z.congruent %arg0, %arg1, %arg2
  %r = gmp.z.congruent %a, %b, %m
  return %r : i1
}
