//===- GMPOps.td - GMP operation definitions ---------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the operations for the GMP dialect. Operations are
// organized into two modules following the Zarith convention:
// - Z module: Operations on arbitrary precision integers
// - Q module: Operations on arbitrary precision rationals
//
//===----------------------------------------------------------------------===//

#ifndef GMP_OPS_TD
#define GMP_OPS_TD

include "GMP/GMPDialect.td"
include "GMP/GMPTypes.td"
include "GMP/GMPAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Z Module: Integer Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Z Construction Operations
//===----------------------------------------------------------------------===//

def GMP_ZConstantOp : GMP_Op<"z.constant", [Pure, ConstantLike]> {
  let summary = "Create a constant arbitrary precision integer";
  let description = [{
    Creates a constant arbitrary precision integer from a ZAttr.

    Example:
    ```mlir
    %zero = gmp.z.constant #gmp.z<"0">
    %big = gmp.z.constant #gmp.z<"12345678901234567890">
    ```
  }];

  let arguments = (ins GMP_ZAttr:$value);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$value attr-dict";

  let builders = [
    OpBuilder<(ins "ZAttr":$value), [{
      build($_builder, $_state, $_builder.getType<ZType>(), value);
    }]>
  ];

  let hasFolder = 1;
}

def GMP_ZFromIntOp : GMP_Op<"z.from_int", [Pure]> {
  let summary = "Convert MLIR integer to arbitrary precision integer";
  let description = [{
    Converts an MLIR integer type (any width) to an arbitrary precision
    integer. At compile-time with constant input, this folds to a ZAttr.

    Example:
    ```mlir
    %z = gmp.z.from_int %x : i64
    ```
  }];

  let arguments = (ins AnySignlessInteger:$input);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input)";

  let builders = [
    OpBuilder<(ins "Value":$input), [{
      build($_builder, $_state, $_builder.getType<ZType>(), input);
    }]>
  ];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Basic Arithmetic Operations
//===----------------------------------------------------------------------===//

def GMP_ZAddOp : GMP_Op<"z.add", [Pure, Commutative]> {
  let summary = "Add two arbitrary precision integers";
  let description = [{
    Computes the sum of two arbitrary precision integers.

    Example:
    ```mlir
    %sum = gmp.z.add %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZSubOp : GMP_Op<"z.sub", [Pure]> {
  let summary = "Subtract two arbitrary precision integers";
  let description = [{
    Computes the difference of two arbitrary precision integers (lhs - rhs).

    Example:
    ```mlir
    %diff = gmp.z.sub %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZMulOp : GMP_Op<"z.mul", [Pure, Commutative]> {
  let summary = "Multiply two arbitrary precision integers";
  let description = [{
    Computes the product of two arbitrary precision integers.

    Example:
    ```mlir
    %prod = gmp.z.mul %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Unary Operations
//===----------------------------------------------------------------------===//

def GMP_ZNegOp : GMP_Op<"z.neg", [Pure]> {
  let summary = "Negate an arbitrary precision integer";
  let description = [{
    Computes the negation of an arbitrary precision integer (-x).

    Example:
    ```mlir
    %neg = gmp.z.neg %x
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_ZAbsOp : GMP_Op<"z.abs", [Pure]> {
  let summary = "Absolute value of an arbitrary precision integer";
  let description = [{
    Computes the absolute value of an arbitrary precision integer.

    Example:
    ```mlir
    %abs = gmp.z.abs %x
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_ZSuccOp : GMP_Op<"z.succ", [Pure]> {
  let summary = "Successor of an arbitrary precision integer";
  let description = [{
    Computes the successor (x + 1) of an arbitrary precision integer.

    Example:
    ```mlir
    %next = gmp.z.succ %x
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_ZPredOp : GMP_Op<"z.pred", [Pure]> {
  let summary = "Predecessor of an arbitrary precision integer";
  let description = [{
    Computes the predecessor (x - 1) of an arbitrary precision integer.

    Example:
    ```mlir
    %prev = gmp.z.pred %x
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Division Operations
//===----------------------------------------------------------------------===//

def GMP_ZDivOp : GMP_Op<"z.div", [Pure]> {
  let summary = "Truncated division of arbitrary precision integers";
  let description = [{
    Computes the quotient of lhs / rhs, truncated toward zero.
    This is equivalent to GMP's mpz_tdiv_q.

    Example:
    ```mlir
    %q = gmp.z.div %a, %b  // 7 / 3 = 2, -7 / 3 = -2
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZRemOp : GMP_Op<"z.rem", [Pure]> {
  let summary = "Truncated remainder of arbitrary precision integers";
  let description = [{
    Computes the remainder of lhs / rhs, with sign matching the dividend.
    This is equivalent to GMP's mpz_tdiv_r.

    Example:
    ```mlir
    %r = gmp.z.rem %a, %b  // 7 % 3 = 1, -7 % 3 = -1
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZDivRemOp : GMP_Op<"z.div_rem", [Pure]> {
  let summary = "Truncated division with quotient and remainder";
  let description = [{
    Computes both quotient and remainder of lhs / rhs, truncated toward zero.
    Returns two results: quotient and remainder.

    Example:
    ```mlir
    %q, %r = gmp.z.div_rem %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$quotient, GMP_ZType:$remainder);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZFDivOp : GMP_Op<"z.fdiv", [Pure]> {
  let summary = "Floor division of arbitrary precision integers";
  let description = [{
    Computes the quotient of lhs / rhs, rounded toward negative infinity.
    This is equivalent to GMP's mpz_fdiv_q.

    Example:
    ```mlir
    %q = gmp.z.fdiv %a, %b  // 7 / 3 = 2, -7 / 3 = -3
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZFRemOp : GMP_Op<"z.frem", [Pure]> {
  let summary = "Floor remainder of arbitrary precision integers";
  let description = [{
    Computes the remainder after floor division, with sign matching the divisor.
    This is equivalent to GMP's mpz_fdiv_r.

    Example:
    ```mlir
    %r = gmp.z.frem %a, %b  // 7 % 3 = 1, -7 % 3 = 2
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZCDivOp : GMP_Op<"z.cdiv", [Pure]> {
  let summary = "Ceiling division of arbitrary precision integers";
  let description = [{
    Computes the quotient of lhs / rhs, rounded toward positive infinity.
    This is equivalent to GMP's mpz_cdiv_q.

    Example:
    ```mlir
    %q = gmp.z.cdiv %a, %b  // 7 / 3 = 3, -7 / 3 = -2
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZCRemOp : GMP_Op<"z.crem", [Pure]> {
  let summary = "Ceiling remainder of arbitrary precision integers";
  let description = [{
    Computes the remainder after ceiling division.
    This is equivalent to GMP's mpz_cdiv_r.

    Example:
    ```mlir
    %r = gmp.z.crem %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZEDivOp : GMP_Op<"z.ediv", [Pure]> {
  let summary = "Euclidean division quotient";
  let description = [{
    Computes the Euclidean division quotient where the remainder is always
    non-negative (0 <= r < |divisor|). This follows mathematical conventions.

    Example:
    ```mlir
    %q = gmp.z.ediv %a, %b  // -7 ediv 3 = -3 (so -7 = -3*3 + 2)
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZERemOp : GMP_Op<"z.erem", [Pure]> {
  let summary = "Euclidean division remainder (always non-negative)";
  let description = [{
    Computes the Euclidean remainder, which is always non-negative
    (0 <= result < |divisor|). This follows mathematical conventions.

    Example:
    ```mlir
    %r = gmp.z.erem %a, %b  // -7 erem 3 = 2
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZDivExactOp : GMP_Op<"z.divexact", [Pure]> {
  let summary = "Exact division (assumes divisibility)";
  let description = [{
    Computes lhs / rhs assuming rhs divides lhs exactly. This is faster than
    regular division but produces undefined results if the division is not exact.
    This is equivalent to GMP's mpz_divexact.

    Example:
    ```mlir
    %q = gmp.z.divexact %a, %b  // Only valid when b divides a exactly
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZDivisibleOp : GMP_Op<"z.divisible", [Pure]> {
  let summary = "Test if one integer divides another";
  let description = [{
    Returns true (1) if rhs divides lhs exactly, false (0) otherwise.
    This is equivalent to GMP's mpz_divisible_p.

    Example:
    ```mlir
    %result = gmp.z.divisible %a, %b : i1  // true if b divides a
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZCongruentOp : GMP_Op<"z.congruent", [Pure]> {
  let summary = "Test if two integers are congruent modulo a third";
  let description = [{
    Returns true (1) if lhs ≡ rhs (mod modulus), false (0) otherwise.
    This is equivalent to GMP's mpz_congruent_p.

    Example:
    ```mlir
    %result = gmp.z.congruent %a, %b, %m : i1  // true if a ≡ b (mod m)
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs, GMP_ZType:$modulus);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs `,` $modulus attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Comparison Operations
//===----------------------------------------------------------------------===//

def GMP_ZCompareOp : GMP_Op<"z.compare", [Pure]> {
  let summary = "Three-way comparison of arbitrary precision integers";
  let description = [{
    Compares two arbitrary precision integers and returns:
    - negative value if lhs < rhs
    - zero if lhs == rhs
    - positive value if lhs > rhs
    This is equivalent to GMP's mpz_cmp.

    Example:
    ```mlir
    %cmp = gmp.z.compare %a, %b : i32
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs I32:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZEqualOp : GMP_Op<"z.equal", [Pure, Commutative]> {
  let summary = "Test equality of arbitrary precision integers";
  let description = [{
    Returns true if lhs equals rhs.

    Example:
    ```mlir
    %eq = gmp.z.equal %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZLtOp : GMP_Op<"z.lt", [Pure]> {
  let summary = "Test if lhs is less than rhs";
  let description = [{
    Returns true if lhs < rhs.

    Example:
    ```mlir
    %lt = gmp.z.lt %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZLeqOp : GMP_Op<"z.leq", [Pure]> {
  let summary = "Test if lhs is less than or equal to rhs";
  let description = [{
    Returns true if lhs <= rhs.

    Example:
    ```mlir
    %leq = gmp.z.leq %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZGtOp : GMP_Op<"z.gt", [Pure]> {
  let summary = "Test if lhs is greater than rhs";
  let description = [{
    Returns true if lhs > rhs.

    Example:
    ```mlir
    %gt = gmp.z.gt %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZGeqOp : GMP_Op<"z.geq", [Pure]> {
  let summary = "Test if lhs is greater than or equal to rhs";
  let description = [{
    Returns true if lhs >= rhs.

    Example:
    ```mlir
    %geq = gmp.z.geq %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Bitwise Logical Operations
//===----------------------------------------------------------------------===//

def GMP_ZLogAndOp : GMP_Op<"z.logand", [Pure, Commutative]> {
  let summary = "Bitwise AND of arbitrary precision integers";
  let description = [{
    Computes the bitwise AND of two arbitrary precision integers.
    This is equivalent to GMP's mpz_and.

    Example:
    ```mlir
    %and = gmp.z.logand %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZLogOrOp : GMP_Op<"z.logor", [Pure, Commutative]> {
  let summary = "Bitwise OR of arbitrary precision integers";
  let description = [{
    Computes the bitwise OR of two arbitrary precision integers.
    This is equivalent to GMP's mpz_ior.

    Example:
    ```mlir
    %or = gmp.z.logor %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZLogXorOp : GMP_Op<"z.logxor", [Pure, Commutative]> {
  let summary = "Bitwise XOR of arbitrary precision integers";
  let description = [{
    Computes the bitwise XOR of two arbitrary precision integers.
    This is equivalent to GMP's mpz_xor.

    Example:
    ```mlir
    %xor = gmp.z.logxor %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZLogNotOp : GMP_Op<"z.lognot", [Pure]> {
  let summary = "Bitwise complement of an arbitrary precision integer";
  let description = [{
    Computes the one's complement (bitwise NOT) of an arbitrary precision integer.
    This is equivalent to GMP's mpz_com.

    Example:
    ```mlir
    %not = gmp.z.lognot %x
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Bit Shift Operations
//===----------------------------------------------------------------------===//

def GMP_ZShiftLeftOp : GMP_Op<"z.shift_left", [Pure]> {
  let summary = "Left shift an arbitrary precision integer";
  let description = [{
    Shifts operand left by count bits. Equivalent to multiplying by 2^count.
    This is equivalent to GMP's mpz_mul_2exp.

    Example:
    ```mlir
    %shifted = gmp.z.shift_left %x, %count
    ```
  }];

  let arguments = (ins GMP_ZType:$operand, I64:$count);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand `,` $count attr-dict";

  let hasFolder = 1;
}

def GMP_ZShiftRightOp : GMP_Op<"z.shift_right", [Pure]> {
  let summary = "Arithmetic right shift (floor division by 2^count)";
  let description = [{
    Shifts operand right by count bits using floor division semantics.
    Equivalent to floor division by 2^count.
    This is equivalent to GMP's mpz_fdiv_q_2exp.

    Example:
    ```mlir
    %shifted = gmp.z.shift_right %x, %count
    ```
  }];

  let arguments = (ins GMP_ZType:$operand, I64:$count);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand `,` $count attr-dict";

  let hasFolder = 1;
}

def GMP_ZShiftRightTruncOp : GMP_Op<"z.shift_right_trunc", [Pure]> {
  let summary = "Truncating right shift (truncated division by 2^count)";
  let description = [{
    Shifts operand right by count bits using truncation toward zero.
    Equivalent to truncated division by 2^count.
    This is equivalent to GMP's mpz_tdiv_q_2exp.

    Example:
    ```mlir
    %shifted = gmp.z.shift_right_trunc %x, %count
    ```
  }];

  let arguments = (ins GMP_ZType:$operand, I64:$count);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand `,` $count attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Bit Query Operations
//===----------------------------------------------------------------------===//

def GMP_ZTestBitOp : GMP_Op<"z.testbit", [Pure]> {
  let summary = "Test if a specific bit is set";
  let description = [{
    Returns true if bit at position index is set in operand.
    This is equivalent to GMP's mpz_tstbit.

    Example:
    ```mlir
    %bit = gmp.z.testbit %x, %index
    ```
  }];

  let arguments = (ins GMP_ZType:$operand, I64:$index);
  let results = (outs I1:$result);

  let assemblyFormat = "$operand `,` $index attr-dict";

  let hasFolder = 1;
}

def GMP_ZPopCountOp : GMP_Op<"z.popcount", [Pure]> {
  let summary = "Count the number of set bits";
  let description = [{
    Returns the population count (number of 1 bits) for non-negative integers.
    For negative integers, returns the maximum possible value (ULONG_MAX).
    This is equivalent to GMP's mpz_popcount.

    Example:
    ```mlir
    %count = gmp.z.popcount %x
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs I64:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_ZNumBitsOp : GMP_Op<"z.numbits", [Pure]> {
  let summary = "Count the number of bits needed to represent the value";
  let description = [{
    Returns the number of bits required to represent the absolute value.
    Returns 0 for zero, otherwise returns floor(log2(|x|)) + 1.
    This is equivalent to GMP's mpz_sizeinbase with base 2.

    Example:
    ```mlir
    %bits = gmp.z.numbits %x
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs I64:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_ZTrailingZerosOp : GMP_Op<"z.trailing_zeros", [Pure]> {
  let summary = "Count trailing zero bits";
  let description = [{
    Returns the number of trailing zero bits (index of least significant 1 bit).
    Returns ULONG_MAX for zero.
    This is equivalent to GMP's mpz_scan1 starting from bit 0.

    Example:
    ```mlir
    %zeros = gmp.z.trailing_zeros %x
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs I64:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Bit Extraction Operations
//===----------------------------------------------------------------------===//

def GMP_ZExtractOp : GMP_Op<"z.extract", [Pure]> {
  let summary = "Extract a range of bits as unsigned integer";
  let description = [{
    Extracts width bits starting at position lo from operand.
    The result is treated as an unsigned integer.

    Example:
    ```mlir
    %bits = gmp.z.extract %x, %lo, %width
    ```
  }];

  let arguments = (ins GMP_ZType:$operand, I64:$lo, I64:$width);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand `,` $lo `,` $width attr-dict";

  let hasFolder = 1;
}

def GMP_ZSignedExtractOp : GMP_Op<"z.signed_extract", [Pure]> {
  let summary = "Extract a range of bits as signed integer";
  let description = [{
    Extracts width bits starting at position lo from operand.
    The result is sign-extended based on the highest extracted bit.

    Example:
    ```mlir
    %bits = gmp.z.signed_extract %x, %lo, %width
    ```
  }];

  let arguments = (ins GMP_ZType:$operand, I64:$lo, I64:$width);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand `,` $lo `,` $width attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Q Module: Rational Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Q Construction Operations
//===----------------------------------------------------------------------===//

def GMP_QConstantOp : GMP_Op<"q.constant", [Pure, ConstantLike]> {
  let summary = "Create a constant arbitrary precision rational";
  let description = [{
    Creates a constant arbitrary precision rational from a QAttr.

    Example:
    ```mlir
    %half = gmp.q.constant #gmp.q<"1", "2">
    %inf = gmp.q.constant #gmp.q<"1", "0">  // +infinity
    ```
  }];

  let arguments = (ins GMP_QAttr:$value);
  let results = (outs GMP_QType:$result);

  let assemblyFormat = "$value attr-dict";

  let builders = [
    OpBuilder<(ins "QAttr":$value), [{
      build($_builder, $_state, $_builder.getType<QType>(), value);
    }]>
  ];

  let hasFolder = 1;
}

#endif // GMP_OPS_TD
