// RUN: asl-opt %s --canonicalize | FileCheck %s

// Test constant folding for Z bitwise operations

//===----------------------------------------------------------------------===//
// Bitwise AND
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_logand
func.func @fold_z_logand() -> !gmp.z {
  %a = gmp.z.constant <240>
  %b = gmp.z.constant <170>
  // 240 (0b11110000) & 170 (0b10101010) = 160 (0b10100000)
  // CHECK: gmp.z.constant <160>
  %and = gmp.z.logand %a, %b
  return %and : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_logand_zero
func.func @fold_z_logand_zero(%x: !gmp.z) -> !gmp.z {
  %zero = gmp.z.constant <0>
  // x & 0 -> 0
  // CHECK: gmp.z.constant <0>
  %and = gmp.z.logand %x, %zero
  return %and : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_logand_self
func.func @fold_z_logand_self(%x: !gmp.z) -> !gmp.z {
  // x & x -> x
  // CHECK: return %arg0
  %and = gmp.z.logand %x, %x
  return %and : !gmp.z
}

//===----------------------------------------------------------------------===//
// Bitwise OR
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_logor
func.func @fold_z_logor() -> !gmp.z {
  %a = gmp.z.constant <240>
  %b = gmp.z.constant <15>
  // 240 (0b11110000) | 15 (0b00001111) = 255 (0b11111111)
  // CHECK: gmp.z.constant <255>
  %or = gmp.z.logor %a, %b
  return %or : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_logor_zero
func.func @fold_z_logor_zero(%x: !gmp.z) -> !gmp.z {
  %zero = gmp.z.constant <0>
  // x | 0 -> x
  // CHECK: return %arg0
  %or = gmp.z.logor %x, %zero
  return %or : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_logor_self
func.func @fold_z_logor_self(%x: !gmp.z) -> !gmp.z {
  // x | x -> x
  // CHECK: return %arg0
  %or = gmp.z.logor %x, %x
  return %or : !gmp.z
}

//===----------------------------------------------------------------------===//
// Bitwise XOR
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_logxor
func.func @fold_z_logxor() -> !gmp.z {
  %a = gmp.z.constant <240>
  %b = gmp.z.constant <170>
  // 240 (0b11110000) ^ 170 (0b10101010) = 90 (0b01011010)
  // CHECK: gmp.z.constant <90>
  %xor = gmp.z.logxor %a, %b
  return %xor : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_logxor_zero
func.func @fold_z_logxor_zero(%x: !gmp.z) -> !gmp.z {
  %zero = gmp.z.constant <0>
  // x ^ 0 -> x
  // CHECK: return %arg0
  %xor = gmp.z.logxor %x, %zero
  return %xor : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_logxor_self
func.func @fold_z_logxor_self(%x: !gmp.z) -> !gmp.z {
  // x ^ x -> 0
  // CHECK: gmp.z.constant <0>
  %xor = gmp.z.logxor %x, %x
  return %xor : !gmp.z
}

//===----------------------------------------------------------------------===//
// Bitwise NOT (Complement)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_lognot
func.func @fold_z_lognot() -> !gmp.z {
  %a = gmp.z.constant <0>
  // ~0 = -1 (two's complement)
  // CHECK: gmp.z.constant <-1>
  %not = gmp.z.lognot %a
  return %not : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_lognot_positive
func.func @fold_z_lognot_positive() -> !gmp.z {
  %a = gmp.z.constant <5>
  // ~5 = -6 (two's complement: ~x = -(x+1))
  // CHECK: gmp.z.constant <-6>
  %not = gmp.z.lognot %a
  return %not : !gmp.z
}

//===----------------------------------------------------------------------===//
// Left Shift
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_shift_left
func.func @fold_z_shift_left() -> !gmp.z {
  %a = gmp.z.constant <1>
  %count = arith.constant 8 : i64
  // 1 << 8 = 256
  // CHECK: gmp.z.constant <256>
  %shifted = gmp.z.shift_left %a, %count
  return %shifted : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_shift_left_zero_count
func.func @fold_z_shift_left_zero_count(%x: !gmp.z) -> !gmp.z {
  %count = arith.constant 0 : i64
  // x << 0 -> x
  // CHECK: return %arg0
  %shifted = gmp.z.shift_left %x, %count
  return %shifted : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_shift_left_zero_value
func.func @fold_z_shift_left_zero_value() -> !gmp.z {
  %zero = gmp.z.constant <0>
  %count = arith.constant 10 : i64
  // 0 << n -> 0
  // CHECK: gmp.z.constant <0>
  %shifted = gmp.z.shift_left %zero, %count
  return %shifted : !gmp.z
}

//===----------------------------------------------------------------------===//
// Right Shift (Floor)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_shift_right
func.func @fold_z_shift_right() -> !gmp.z {
  %a = gmp.z.constant <256>
  %count = arith.constant 4 : i64
  // 256 >> 4 = 16
  // CHECK: gmp.z.constant <16>
  %shifted = gmp.z.shift_right %a, %count
  return %shifted : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_shift_right_negative
func.func @fold_z_shift_right_negative() -> !gmp.z {
  %a = gmp.z.constant <-7>
  %count = arith.constant 2 : i64
  // -7 >> 2 with floor = -2 (floor division by 4)
  // CHECK: gmp.z.constant <-2>
  %shifted = gmp.z.shift_right %a, %count
  return %shifted : !gmp.z
}

//===----------------------------------------------------------------------===//
// Right Shift (Truncate)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_shift_right_trunc_negative
func.func @fold_z_shift_right_trunc_negative() -> !gmp.z {
  %a = gmp.z.constant <-7>
  %count = arith.constant 2 : i64
  // -7 >> 2 with truncation = -1 (truncated division by 4)
  // CHECK: gmp.z.constant <-1>
  %shifted = gmp.z.shift_right_trunc %a, %count
  return %shifted : !gmp.z
}

//===----------------------------------------------------------------------===//
// Test Bit
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_testbit_set
func.func @fold_z_testbit_set() -> i1 {
  %a = gmp.z.constant <8>
  %index = arith.constant 3 : i64
  // Bit 3 of 8 (0b1000) is set
  // CHECK: gmp.z.testbit
  %bit = gmp.z.testbit %a, %index
  return %bit : i1
}

// CHECK-LABEL: func.func @fold_z_testbit_clear
func.func @fold_z_testbit_clear() -> i1 {
  %a = gmp.z.constant <8>
  %index = arith.constant 0 : i64
  // Bit 0 of 8 (0b1000) is clear
  // CHECK: gmp.z.testbit
  %bit = gmp.z.testbit %a, %index
  return %bit : i1
}

//===----------------------------------------------------------------------===//
// Population Count
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_popcount
func.func @fold_z_popcount() -> i64 {
  %a = gmp.z.constant <0b11110000>
  // popcount(0b11110000) = 4
  // CHECK: arith.constant 4 : i64
  %count = gmp.z.popcount %a
  return %count : i64
}

// CHECK-LABEL: func.func @fold_z_popcount_zero
func.func @fold_z_popcount_zero() -> i64 {
  %a = gmp.z.constant <0>
  // CHECK: arith.constant 0 : i64
  %count = gmp.z.popcount %a
  return %count : i64
}

//===----------------------------------------------------------------------===//
// Number of Bits
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_numbits
func.func @fold_z_numbits() -> i64 {
  %a = gmp.z.constant <255>
  // 255 requires 8 bits
  // CHECK: arith.constant 8 : i64
  %bits = gmp.z.numbits %a
  return %bits : i64
}

// CHECK-LABEL: func.func @fold_z_numbits_zero
func.func @fold_z_numbits_zero() -> i64 {
  %a = gmp.z.constant <0>
  // 0 requires 0 bits
  // CHECK: arith.constant 0 : i64
  %bits = gmp.z.numbits %a
  return %bits : i64
}

// CHECK-LABEL: func.func @fold_z_numbits_power_of_two
func.func @fold_z_numbits_power_of_two() -> i64 {
  %a = gmp.z.constant <256>
  // 256 = 2^8 requires 9 bits
  // CHECK: arith.constant 9 : i64
  %bits = gmp.z.numbits %a
  return %bits : i64
}

//===----------------------------------------------------------------------===//
// Trailing Zeros
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_trailing_zeros
func.func @fold_z_trailing_zeros() -> i64 {
  %a = gmp.z.constant <24>
  // 24 = 0b11000 has 3 trailing zeros
  // CHECK: arith.constant 3 : i64
  %zeros = gmp.z.trailing_zeros %a
  return %zeros : i64
}

// CHECK-LABEL: func.func @fold_z_trailing_zeros_odd
func.func @fold_z_trailing_zeros_odd() -> i64 {
  %a = gmp.z.constant <7>
  // 7 = 0b111 has 0 trailing zeros
  // CHECK: arith.constant 0 : i64
  %zeros = gmp.z.trailing_zeros %a
  return %zeros : i64
}

//===----------------------------------------------------------------------===//
// Bit Extraction (Unsigned)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_extract
func.func @fold_z_extract() -> !gmp.z {
  %a = gmp.z.constant <0xABCD>
  %lo = arith.constant 4 : i64
  %width = arith.constant 8 : i64
  // Extract bits 4-11 of 0xABCD = 0xBC = 188
  // CHECK: gmp.z.constant <188>
  %bits = gmp.z.extract %a, %lo, %width
  return %bits : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_extract_zero_width
func.func @fold_z_extract_zero_width() -> !gmp.z {
  %a = gmp.z.constant <0xFFFF>
  %lo = arith.constant 0 : i64
  %width = arith.constant 0 : i64
  // Extracting 0 bits gives 0
  // CHECK: gmp.z.constant <0>
  %bits = gmp.z.extract %a, %lo, %width
  return %bits : !gmp.z
}

//===----------------------------------------------------------------------===//
// Bit Extraction (Signed)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fold_z_signed_extract_positive
func.func @fold_z_signed_extract_positive() -> !gmp.z {
  %a = gmp.z.constant <0x7F>
  %lo = arith.constant 0 : i64
  %width = arith.constant 8 : i64
  // Sign bit (bit 7) is 0, so result is positive: 127
  // CHECK: gmp.z.constant <127>
  %bits = gmp.z.signed_extract %a, %lo, %width
  return %bits : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_signed_extract_negative
func.func @fold_z_signed_extract_negative() -> !gmp.z {
  %a = gmp.z.constant <0xFF>
  %lo = arith.constant 0 : i64
  %width = arith.constant 8 : i64
  // Sign bit (bit 7) is 1, sign extend: 0xFF -> -1
  // CHECK: gmp.z.constant <-1>
  %bits = gmp.z.signed_extract %a, %lo, %width
  return %bits : !gmp.z
}

// CHECK-LABEL: func.func @fold_z_signed_extract_partial
func.func @fold_z_signed_extract_partial() -> !gmp.z {
  %a = gmp.z.constant <0xF0>
  %lo = arith.constant 4 : i64
  %width = arith.constant 4 : i64
  // Extract bits 4-7: 0xF = 15, sign bit is 1, sign extend to -1
  // CHECK: gmp.z.constant <-1>
  %bits = gmp.z.signed_extract %a, %lo, %width
  return %bits : !gmp.z
}
