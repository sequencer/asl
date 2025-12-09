// RUN: asl-opt %s | asl-opt | FileCheck %s

// CHECK-LABEL: func.func @test_z_constants
func.func @test_z_constants() {
  // CHECK: gmp.z.constant <0>
  %zero = gmp.z.constant #gmp.z<0>

  // CHECK: gmp.z.constant <1>
  %one = gmp.z.constant #gmp.z<1>

  // CHECK: gmp.z.constant <-42>
  %neg = gmp.z.constant #gmp.z<-42>

  // CHECK: gmp.z.constant <12345678901234567890>
  %big = gmp.z.constant #gmp.z<12345678901234567890>

  return
}

// CHECK-LABEL: func.func @test_z_from_int
func.func @test_z_from_int(%arg0: i64) -> !gmp.z {
  // CHECK: gmp.z.from_int %arg0 : i64
  %z = gmp.z.from_int %arg0 : i64
  return %z : !gmp.z
}
