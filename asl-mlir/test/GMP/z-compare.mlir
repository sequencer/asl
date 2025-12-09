// RUN: asl-opt %s | asl-opt | FileCheck %s

// Test comparison operations roundtrip

// CHECK-LABEL: func.func @test_z_compare
func.func @test_z_compare(%a: !gmp.z, %b: !gmp.z) -> i32 {
  // CHECK: gmp.z.compare %arg0, %arg1
  %cmp = gmp.z.compare %a, %b
  return %cmp : i32
}

// CHECK-LABEL: func.func @test_z_equal
func.func @test_z_equal(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: gmp.z.equal %arg0, %arg1
  %eq = gmp.z.equal %a, %b
  return %eq : i1
}

// CHECK-LABEL: func.func @test_z_lt
func.func @test_z_lt(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: gmp.z.lt %arg0, %arg1
  %lt = gmp.z.lt %a, %b
  return %lt : i1
}

// CHECK-LABEL: func.func @test_z_leq
func.func @test_z_leq(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: gmp.z.leq %arg0, %arg1
  %leq = gmp.z.leq %a, %b
  return %leq : i1
}

// CHECK-LABEL: func.func @test_z_gt
func.func @test_z_gt(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: gmp.z.gt %arg0, %arg1
  %gt = gmp.z.gt %a, %b
  return %gt : i1
}

// CHECK-LABEL: func.func @test_z_geq
func.func @test_z_geq(%a: !gmp.z, %b: !gmp.z) -> i1 {
  // CHECK: gmp.z.geq %arg0, %arg1
  %geq = gmp.z.geq %a, %b
  return %geq : i1
}
