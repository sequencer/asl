// RUN: asl-opt %s --canonicalize | FileCheck %s

// Test constant folding for Z arithmetic operations

// CHECK-LABEL: func.func @fold_z_add
func.func @fold_z_add() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<10>
  %b = gmp.z.constant #gmp.z<20>
  // CHECK: gmp.z.constant <30>
  %sum = gmp.z.add %a, %b
  return %sum : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_add_negative
func.func @fold_z_add_negative() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<10>
  %b = gmp.z.constant #gmp.z<-3>
  // CHECK: gmp.z.constant <7>
  %sum = gmp.z.add %a, %b
  return %sum : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_sub
func.func @fold_z_sub() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<100>
  %b = gmp.z.constant #gmp.z<42>
  // CHECK: gmp.z.constant <58>
  %diff = gmp.z.sub %a, %b
  return %diff : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_sub_negative_result
func.func @fold_z_sub_negative_result() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<10>
  %b = gmp.z.constant #gmp.z<25>
  // CHECK: gmp.z.constant <-15>
  %diff = gmp.z.sub %a, %b
  return %diff : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_mul
func.func @fold_z_mul() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<7>
  %b = gmp.z.constant #gmp.z<8>
  // CHECK: gmp.z.constant <56>
  %prod = gmp.z.mul %a, %b
  return %prod : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_mul_negative
func.func @fold_z_mul_negative() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<-5>
  %b = gmp.z.constant #gmp.z<6>
  // CHECK: gmp.z.constant <-30>
  %prod = gmp.z.mul %a, %b
  return %prod : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_neg
func.func @fold_z_neg() -> !gmp.z {
  %x = gmp.z.constant #gmp.z<42>
  // CHECK: gmp.z.constant <-42>
  %neg = gmp.z.neg %x
  return %neg : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_neg_negative
func.func @fold_z_neg_negative() -> !gmp.z {
  %x = gmp.z.constant #gmp.z<-17>
  // CHECK: gmp.z.constant <17>
  %neg = gmp.z.neg %x
  return %neg : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_abs_positive
func.func @fold_z_abs_positive() -> !gmp.z {
  %x = gmp.z.constant #gmp.z<42>
  // CHECK: gmp.z.constant <42>
  %abs = gmp.z.abs %x
  return %abs : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_abs_negative
func.func @fold_z_abs_negative() -> !gmp.z {
  %x = gmp.z.constant #gmp.z<-42>
  // CHECK: gmp.z.constant <42>
  %abs = gmp.z.abs %x
  return %abs : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_succ
func.func @fold_z_succ() -> !gmp.z {
  %x = gmp.z.constant #gmp.z<99>
  // CHECK: gmp.z.constant <100>
  %next = gmp.z.succ %x
  return %next : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_pred
func.func @fold_z_pred() -> !gmp.z {
  %x = gmp.z.constant #gmp.z<100>
  // CHECK: gmp.z.constant <99>
  %prev = gmp.z.pred %x
  return %prev : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_add_zero
func.func @fold_z_add_zero(%x: !gmp.z) -> !gmp.z {
  %zero = gmp.z.constant #gmp.z<0>
  // CHECK: return %arg0
  %sum = gmp.z.add %x, %zero
  return %sum : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_sub_zero
func.func @fold_z_sub_zero(%x: !gmp.z) -> !gmp.z {
  %zero = gmp.z.constant #gmp.z<0>
  // CHECK: return %arg0
  %diff = gmp.z.sub %x, %zero
  return %diff : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_mul_zero
func.func @fold_z_mul_zero(%x: !gmp.z) -> !gmp.z {
  %zero = gmp.z.constant #gmp.z<0>
  // CHECK: gmp.z.constant <0>
  %prod = gmp.z.mul %x, %zero
  return %prod : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_mul_one
func.func @fold_z_mul_one(%x: !gmp.z) -> !gmp.z {
  %one = gmp.z.constant #gmp.z<1>
  // CHECK: return %arg0
  %prod = gmp.z.mul %x, %one
  return %prod : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_sub_self
func.func @fold_z_sub_self(%x: !gmp.z) -> !gmp.z {
  // CHECK: gmp.z.constant <0>
  %diff = gmp.z.sub %x, %x
  return %diff : !gmp.z
}

// Test large number arithmetic
// CHECK-LABEL: func.func @fold_z_add_large
func.func @fold_z_add_large() -> !gmp.z {
  %a = gmp.z.constant #gmp.z<12345678901234567890>
  %b = gmp.z.constant #gmp.z<98765432109876543210>
  // CHECK: gmp.z.constant <111111111011111111100>
  %sum = gmp.z.add %a, %b
  return %sum : !gmp.z
}
