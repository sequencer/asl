// RUN: asl-opt %s | asl-opt | FileCheck %s

// CHECK-LABEL: func.func @test_q_constants
func.func @test_q_constants() {
  // CHECK: gmp.q.constant <1, 2>
  %half = gmp.q.constant #gmp.q<1, 2>

  // CHECK: gmp.q.constant <-3, 4>
  %neg = gmp.q.constant #gmp.q<-3, 4>

  // CHECK: gmp.q.constant <0, 1>
  %zero = gmp.q.constant #gmp.q<0, 1>

  // CHECK: gmp.q.constant <1, 0>
  %inf = gmp.q.constant #gmp.q<1, 0>

  // CHECK: gmp.q.constant <-1, 0>
  %neginf = gmp.q.constant #gmp.q<-1, 0>

  // CHECK: gmp.q.constant <0, 0>
  %undef = gmp.q.constant #gmp.q<0, 0>

  return
}
