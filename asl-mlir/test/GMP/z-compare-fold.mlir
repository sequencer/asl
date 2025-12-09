// RUN: asl-opt %s --canonicalize | FileCheck %s

// Test constant folding for Z comparison operations

//===----------------------------------------------------------------------===//
// Three-way Comparison
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_compare_less
func.func @fold_z_compare_less() -> i32 {
  %a = gmp.z.constant <5>
  %b = gmp.z.constant <10>
  // CHECK: arith.constant -1 : i32
  %cmp = gmp.z.compare %a, %b
  return %cmp : i32
}

// CHECK-LABEL: func.func @fold_z_compare_equal
func.func @fold_z_compare_equal() -> i32 {
  %a = gmp.z.constant <42>
  %b = gmp.z.constant <42>
  // CHECK: arith.constant 0 : i32
  %cmp = gmp.z.compare %a, %b
  return %cmp : i32
}

// CHECK-LABEL: func.func @fold_z_compare_greater
func.func @fold_z_compare_greater() -> i32 {
  %a = gmp.z.constant <100>
  %b = gmp.z.constant <10>
  // CHECK: arith.constant 1 : i32
  %cmp = gmp.z.compare %a, %b
  return %cmp : i32
}

// CHECK-LABEL: func.func @fold_z_compare_self
func.func @fold_z_compare_self(%x: !gmp.z) -> i32 {
  // CHECK: arith.constant 0 : i32
  %cmp = gmp.z.compare %x, %x
  return %cmp : i32
}

//===----------------------------------------------------------------------===//
// Equality
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_equal_true
func.func @fold_z_equal_true() -> i1 {
  %a = gmp.z.constant <123456789>
  %b = gmp.z.constant <123456789>
  // CHECK: gmp.z.equal
  %eq = gmp.z.equal %a, %b
  return %eq : i1
}

// CHECK-LABEL: func.func @fold_z_equal_false
func.func @fold_z_equal_false() -> i1 {
  %a = gmp.z.constant <100>
  %b = gmp.z.constant <200>
  // CHECK: gmp.z.equal
  %eq = gmp.z.equal %a, %b
  return %eq : i1
}

// CHECK-LABEL: func.func @fold_z_equal_self
func.func @fold_z_equal_self(%x: !gmp.z) -> i1 {
  // CHECK: gmp.z.equal
  %eq = gmp.z.equal %x, %x
  return %eq : i1
}

//===----------------------------------------------------------------------===//
// Less Than
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_lt_true
func.func @fold_z_lt_true() -> i1 {
  %a = gmp.z.constant <5>
  %b = gmp.z.constant <10>
  // CHECK: gmp.z.lt
  %lt = gmp.z.lt %a, %b
  return %lt : i1
}

// CHECK-LABEL: func.func @fold_z_lt_false_equal
func.func @fold_z_lt_false_equal() -> i1 {
  %a = gmp.z.constant <10>
  %b = gmp.z.constant <10>
  // CHECK: gmp.z.lt
  %lt = gmp.z.lt %a, %b
  return %lt : i1
}

// CHECK-LABEL: func.func @fold_z_lt_negative
func.func @fold_z_lt_negative() -> i1 {
  %a = gmp.z.constant <-100>
  %b = gmp.z.constant <-50>
  // -100 < -50 is true
  // CHECK: gmp.z.lt
  %lt = gmp.z.lt %a, %b
  return %lt : i1
}

// CHECK-LABEL: func.func @fold_z_lt_self
func.func @fold_z_lt_self(%x: !gmp.z) -> i1 {
  // x < x is always false
  // CHECK: gmp.z.lt
  %lt = gmp.z.lt %x, %x
  return %lt : i1
}

//===----------------------------------------------------------------------===//
// Less Than or Equal
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_leq_true_less
func.func @fold_z_leq_true_less() -> i1 {
  %a = gmp.z.constant <5>
  %b = gmp.z.constant <10>
  // CHECK: gmp.z.leq
  %leq = gmp.z.leq %a, %b
  return %leq : i1
}

// CHECK-LABEL: func.func @fold_z_leq_true_equal
func.func @fold_z_leq_true_equal() -> i1 {
  %a = gmp.z.constant <10>
  %b = gmp.z.constant <10>
  // CHECK: gmp.z.leq
  %leq = gmp.z.leq %a, %b
  return %leq : i1
}

// CHECK-LABEL: func.func @fold_z_leq_self
func.func @fold_z_leq_self(%x: !gmp.z) -> i1 {
  // x <= x is always true
  // CHECK: gmp.z.leq
  %leq = gmp.z.leq %x, %x
  return %leq : i1
}

//===----------------------------------------------------------------------===//
// Greater Than
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_gt_true
func.func @fold_z_gt_true() -> i1 {
  %a = gmp.z.constant <100>
  %b = gmp.z.constant <10>
  // CHECK: gmp.z.gt
  %gt = gmp.z.gt %a, %b
  return %gt : i1
}

// CHECK-LABEL: func.func @fold_z_gt_self
func.func @fold_z_gt_self(%x: !gmp.z) -> i1 {
  // x > x is always false
  // CHECK: gmp.z.gt
  %gt = gmp.z.gt %x, %x
  return %gt : i1
}

//===----------------------------------------------------------------------===//
// Greater Than or Equal
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_geq_true_greater
func.func @fold_z_geq_true_greater() -> i1 {
  %a = gmp.z.constant <100>
  %b = gmp.z.constant <10>
  // CHECK: gmp.z.geq
  %geq = gmp.z.geq %a, %b
  return %geq : i1
}

// CHECK-LABEL: func.func @fold_z_geq_self
func.func @fold_z_geq_self(%x: !gmp.z) -> i1 {
  // x >= x is always true
  // CHECK: gmp.z.geq
  %geq = gmp.z.geq %x, %x
  return %geq : i1
}

//===----------------------------------------------------------------------===//
// Large Number Comparisons
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_compare_large
func.func @fold_z_compare_large() -> i32 {
  %a = gmp.z.constant <99999999999999999999>
  %b = gmp.z.constant <100000000000000000000>
  // CHECK: arith.constant -1 : i32
  %cmp = gmp.z.compare %a, %b
  return %cmp : i32
}
