// RUN: asl-opt %s | asl-opt | FileCheck %s

// Test bitwise operations roundtrip

//===----------------------------------------------------------------------===//
// Bitwise Logical Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_logand
func.func @test_z_logand(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.logand %arg0, %arg1
  %and = gmp.z.logand %a, %b
  return %and : !gmp.z
}

// CHECK-LABEL: func.func @test_z_logor
func.func @test_z_logor(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.logor %arg0, %arg1
  %or = gmp.z.logor %a, %b
  return %or : !gmp.z
}

// CHECK-LABEL: func.func @test_z_logxor
func.func @test_z_logxor(%a: !gmp.z, %b: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.logxor %arg0, %arg1
  %xor = gmp.z.logxor %a, %b
  return %xor : !gmp.z
}

// CHECK-LABEL: func.func @test_z_lognot
func.func @test_z_lognot(%a: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.lognot %arg0
  %not = gmp.z.lognot %a
  return %not : !gmp.z
}

//===----------------------------------------------------------------------===//
// Bit Shift Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_shift_left
func.func @test_z_shift_left(%a: !gmp.z, %count: i64) -> !gmp.z {
  // CHECK: gmp.z.shift_left %arg0, %arg1
  %shifted = gmp.z.shift_left %a, %count
  return %shifted : !gmp.z
}

// CHECK-LABEL: func.func @test_z_shift_right
func.func @test_z_shift_right(%a: !gmp.z, %count: i64) -> !gmp.z {
  // CHECK: gmp.z.shift_right %arg0, %arg1
  %shifted = gmp.z.shift_right %a, %count
  return %shifted : !gmp.z
}

// CHECK-LABEL: func.func @test_z_shift_right_trunc
func.func @test_z_shift_right_trunc(%a: !gmp.z, %count: i64) -> !gmp.z {
  // CHECK: gmp.z.shift_right_trunc %arg0, %arg1
  %shifted = gmp.z.shift_right_trunc %a, %count
  return %shifted : !gmp.z
}

//===----------------------------------------------------------------------===//
// Bit Query Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_testbit
func.func @test_z_testbit(%a: !gmp.z, %index: i64) -> i1 {
  // CHECK: gmp.z.testbit %arg0, %arg1
  %bit = gmp.z.testbit %a, %index
  return %bit : i1
}

// CHECK-LABEL: func.func @test_z_popcount
func.func @test_z_popcount(%a: !gmp.z) -> i64 {
  // CHECK: gmp.z.popcount %arg0
  %count = gmp.z.popcount %a
  return %count : i64
}

// CHECK-LABEL: func.func @test_z_numbits
func.func @test_z_numbits(%a: !gmp.z) -> i64 {
  // CHECK: gmp.z.numbits %arg0
  %bits = gmp.z.numbits %a
  return %bits : i64
}

// CHECK-LABEL: func.func @test_z_trailing_zeros
func.func @test_z_trailing_zeros(%a: !gmp.z) -> i64 {
  // CHECK: gmp.z.trailing_zeros %arg0
  %zeros = gmp.z.trailing_zeros %a
  return %zeros : i64
}

//===----------------------------------------------------------------------===//
// Bit Extraction Operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_z_extract
func.func @test_z_extract(%a: !gmp.z, %lo: i64, %width: i64) -> !gmp.z {
  // CHECK: gmp.z.extract %arg0, %arg1, %arg2
  %bits = gmp.z.extract %a, %lo, %width
  return %bits : !gmp.z
}

// CHECK-LABEL: func.func @test_z_signed_extract
func.func @test_z_signed_extract(%a: !gmp.z, %lo: i64, %width: i64) -> !gmp.z {
  // CHECK: gmp.z.signed_extract %arg0, %arg1, %arg2
  %bits = gmp.z.signed_extract %a, %lo, %width
  return %bits : !gmp.z
}
