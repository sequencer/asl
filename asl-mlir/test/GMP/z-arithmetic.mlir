// RUN: asl-opt %s | asl-opt | FileCheck %s

// CHECK-LABEL: func.func @test_z_add
func.func @test_z_add(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.add %arg0, %arg1
  %sum = gmp.z.add %a, %b
  return %sum : !gmp.z
}

// CHECK-LABEL: func.func @test_z_sub
func.func @test_z_sub(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.sub %arg0, %arg1
  %diff = gmp.z.sub %a, %b
  return %diff : !gmp.z
}

// CHECK-LABEL: func.func @test_z_mul
func.func @test_z_mul(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.mul %arg0, %arg1
  %prod = gmp.z.mul %a, %b
  return %prod : !gmp.z
}

// CHECK-LABEL: func.func @test_z_neg
func.func @test_z_neg(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.neg %arg0
  %neg = gmp.z.neg %x
  return %neg : !gmp.z
}

// CHECK-LABEL: func.func @test_z_abs
func.func @test_z_abs(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.abs %arg0
  %abs = gmp.z.abs %x
  return %abs : !gmp.z
}

// CHECK-LABEL: func.func @test_z_succ
func.func @test_z_succ(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.succ %arg0
  %next = gmp.z.succ %x
  return %next : !gmp.z
}

// CHECK-LABEL: func.func @test_z_pred
func.func @test_z_pred(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.pred %arg0
  %prev = gmp.z.pred %x
  return %prev : !gmp.z
}
