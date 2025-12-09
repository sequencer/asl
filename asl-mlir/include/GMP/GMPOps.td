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
// Z Number Theory Operations
//===----------------------------------------------------------------------===//

def GMP_ZGcdOp : GMP_Op<"z.gcd", [Pure, Commutative]> {
  let summary = "Greatest common divisor of two integers";
  let description = [{
    Computes the greatest common divisor of two arbitrary precision integers.
    The result is always non-negative.
    This is equivalent to GMP's mpz_gcd.

    Example:
    ```mlir
    %gcd = gmp.z.gcd %a, %b  // gcd(12, 8) = 4
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZGcdExtOp : GMP_Op<"z.gcdext", [Pure]> {
  let summary = "Extended GCD returning gcd and Bezout coefficients";
  let description = [{
    Computes the extended GCD: returns (g, s, t) such that g = gcd(a, b)
    and g = a*s + b*t. This is equivalent to GMP's mpz_gcdext.

    Example:
    ```mlir
    %g, %s, %t = gmp.z.gcdext %a, %b
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$gcd, GMP_ZType:$s, GMP_ZType:$t);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_ZLcmOp : GMP_Op<"z.lcm", [Pure, Commutative]> {
  let summary = "Least common multiple of two integers";
  let description = [{
    Computes the least common multiple of two arbitrary precision integers.
    The result is always non-negative.
    This is equivalent to GMP's mpz_lcm.

    Example:
    ```mlir
    %lcm = gmp.z.lcm %a, %b  // lcm(4, 6) = 12
    ```
  }];

  let arguments = (ins GMP_ZType:$lhs, GMP_ZType:$rhs);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Modular Arithmetic Operations
//===----------------------------------------------------------------------===//

def GMP_ZPowmOp : GMP_Op<"z.powm", [Pure]> {
  let summary = "Modular exponentiation";
  let description = [{
    Computes (base^exp) mod modulus. This is equivalent to GMP's mpz_powm.
    The exponent can be negative if an inverse exists.

    Example:
    ```mlir
    %result = gmp.z.powm %base, %exp, %mod  // base^exp mod mod
    ```
  }];

  let arguments = (ins GMP_ZType:$base, GMP_ZType:$exp, GMP_ZType:$modulus);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$base `,` $exp `,` $modulus attr-dict";

  let hasFolder = 1;
}

def GMP_ZPowmSecOp : GMP_Op<"z.powm_sec", [Pure]> {
  let summary = "Secure modular exponentiation (constant-time)";
  let description = [{
    Computes (base^exp) mod modulus using a constant-time algorithm
    to prevent timing side-channel attacks. The exponent must be positive
    and the modulus must be odd.
    This is equivalent to GMP's mpz_powm_sec.

    Example:
    ```mlir
    %result = gmp.z.powm_sec %base, %exp, %mod
    ```
  }];

  let arguments = (ins GMP_ZType:$base, GMP_ZType:$exp, GMP_ZType:$modulus);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$base `,` $exp `,` $modulus attr-dict";

  let hasFolder = 1;
}

def GMP_ZInvertOp : GMP_Op<"z.invert", [Pure]> {
  let summary = "Modular multiplicative inverse";
  let description = [{
    Computes the modular multiplicative inverse of operand modulo modulus.
    Returns the inverse x such that (operand * x) mod modulus = 1.
    The second result indicates whether the inverse exists (true) or not (false).
    This is equivalent to GMP's mpz_invert.

    Example:
    ```mlir
    %inv, %exists = gmp.z.invert %a, %mod
    ```
  }];

  let arguments = (ins GMP_ZType:$operand, GMP_ZType:$modulus);
  let results = (outs GMP_ZType:$result, I1:$exists);

  let assemblyFormat = "$operand `,` $modulus attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Primality Testing Operations
//===----------------------------------------------------------------------===//

def GMP_ZProbabPrimeOp : GMP_Op<"z.probab_prime", [Pure]> {
  let summary = "Probabilistic primality test";
  let description = [{
    Performs Miller-Rabin primality test with specified number of rounds.
    Returns:
    - 2 if definitely prime
    - 1 if probably prime
    - 0 if definitely composite
    This is equivalent to GMP's mpz_probab_prime_p.

    Example:
    ```mlir
    %result = gmp.z.probab_prime %n, %reps  // reps = number of test rounds
    ```
  }];

  let arguments = (ins GMP_ZType:$operand, I32:$reps);
  let results = (outs I32:$result);

  let assemblyFormat = "$operand `,` $reps attr-dict";

  let hasFolder = 1;
}

def GMP_ZNextPrimeOp : GMP_Op<"z.nextprime", [Pure]> {
  let summary = "Find next prime number";
  let description = [{
    Finds the next prime greater than operand.
    This is equivalent to GMP's mpz_nextprime.

    Example:
    ```mlir
    %next = gmp.z.nextprime %n  // nextprime(10) = 11
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Power and Root Operations
//===----------------------------------------------------------------------===//

def GMP_ZPowOp : GMP_Op<"z.pow", [Pure]> {
  let summary = "Integer exponentiation";
  let description = [{
    Computes base raised to the power of exp.
    The exponent must be non-negative.
    This is equivalent to GMP's mpz_pow_ui.

    Example:
    ```mlir
    %result = gmp.z.pow %base, %exp  // 2^10 = 1024
    ```
  }];

  let arguments = (ins GMP_ZType:$base, I64:$exp);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$base `,` $exp attr-dict";

  let hasFolder = 1;
}

def GMP_ZSqrtOp : GMP_Op<"z.sqrt", [Pure]> {
  let summary = "Integer square root";
  let description = [{
    Computes the integer square root (floor of the true square root).
    The operand must be non-negative.
    This is equivalent to GMP's mpz_sqrt.

    Example:
    ```mlir
    %root = gmp.z.sqrt %n  // sqrt(10) = 3
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_ZSqrtRemOp : GMP_Op<"z.sqrt_rem", [Pure]> {
  let summary = "Integer square root with remainder";
  let description = [{
    Computes both the integer square root and the remainder.
    Returns (root, rem) where operand = root^2 + rem.
    The operand must be non-negative.
    This is equivalent to GMP's mpz_sqrtrem.

    Example:
    ```mlir
    %root, %rem = gmp.z.sqrt_rem %n  // sqrt_rem(10) = (3, 1)
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs GMP_ZType:$root, GMP_ZType:$remainder);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_ZRootOp : GMP_Op<"z.root", [Pure]> {
  let summary = "Integer nth root";
  let description = [{
    Computes the integer nth root (floor of the true nth root).
    The operand must be non-negative for even n.
    This is equivalent to GMP's mpz_root.

    Example:
    ```mlir
    %root = gmp.z.root %n, %k  // root(27, 3) = 3
    ```
  }];

  let arguments = (ins GMP_ZType:$operand, I64:$n);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand `,` $n attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Z Factorial and Combinatorics Operations
//===----------------------------------------------------------------------===//

def GMP_ZFacOp : GMP_Op<"z.fac", [Pure]> {
  let summary = "Factorial";
  let description = [{
    Computes n! (factorial of n).
    The operand must be non-negative.
    This is equivalent to GMP's mpz_fac_ui.

    Example:
    ```mlir
    %result = gmp.z.fac %n  // fac(5) = 120
    ```
  }];

  let arguments = (ins I64:$n);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$n attr-dict";

  let hasFolder = 1;
}

def GMP_ZBinOp : GMP_Op<"z.bin", [Pure]> {
  let summary = "Binomial coefficient";
  let description = [{
    Computes the binomial coefficient C(n, k) = n! / (k! * (n-k)!).
    This is equivalent to GMP's mpz_bin_ui.

    Example:
    ```mlir
    %result = gmp.z.bin %n, %k  // bin(5, 2) = 10
    ```
  }];

  let arguments = (ins GMP_ZType:$n, I64:$k);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$n `,` $k attr-dict";

  let hasFolder = 1;
}

def GMP_ZFibOp : GMP_Op<"z.fib", [Pure]> {
  let summary = "Fibonacci number";
  let description = [{
    Computes the nth Fibonacci number F(n).
    This is equivalent to GMP's mpz_fib_ui.

    Example:
    ```mlir
    %result = gmp.z.fib %n  // fib(10) = 55
    ```
  }];

  let arguments = (ins I64:$n);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$n attr-dict";

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
    %half = gmp.q.constant #gmp.q<1, 2>
    %inf = gmp.q.constant #gmp.q<1, 0>  // +infinity
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

def GMP_QMakeOp : GMP_Op<"q.make", [Pure]> {
  let summary = "Construct a rational from numerator and denominator";
  let description = [{
    Creates a rational number from a numerator and denominator.
    The result is automatically canonicalized (reduced to lowest terms,
    denominator made positive).

    Example:
    ```mlir
    %q = gmp.q.make %num, %den
    ```
  }];

  let arguments = (ins GMP_ZType:$numerator, GMP_ZType:$denominator);
  let results = (outs GMP_QType:$result);

  let assemblyFormat = "$numerator `,` $denominator attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Q Component Accessors
//===----------------------------------------------------------------------===//

def GMP_QNumOp : GMP_Op<"q.num", [Pure]> {
  let summary = "Get the numerator of a rational";
  let description = [{
    Returns the numerator of the rational number.

    Example:
    ```mlir
    %num = gmp.q.num %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_QDenOp : GMP_Op<"q.den", [Pure]> {
  let summary = "Get the denominator of a rational";
  let description = [{
    Returns the denominator of the rational number.

    Example:
    ```mlir
    %den = gmp.q.den %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Q Basic Arithmetic Operations
//===----------------------------------------------------------------------===//

def GMP_QAddOp : GMP_Op<"q.add", [Pure, Commutative]> {
  let summary = "Add two rationals";
  let description = [{
    Computes the sum of two rational numbers.
    The result is automatically canonicalized.

    Example:
    ```mlir
    %sum = gmp.q.add %a, %b
    ```
  }];

  let arguments = (ins GMP_QType:$lhs, GMP_QType:$rhs);
  let results = (outs GMP_QType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_QSubOp : GMP_Op<"q.sub", [Pure]> {
  let summary = "Subtract two rationals";
  let description = [{
    Computes the difference of two rational numbers (lhs - rhs).
    The result is automatically canonicalized.

    Example:
    ```mlir
    %diff = gmp.q.sub %a, %b
    ```
  }];

  let arguments = (ins GMP_QType:$lhs, GMP_QType:$rhs);
  let results = (outs GMP_QType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_QMulOp : GMP_Op<"q.mul", [Pure, Commutative]> {
  let summary = "Multiply two rationals";
  let description = [{
    Computes the product of two rational numbers.
    The result is automatically canonicalized.

    Example:
    ```mlir
    %prod = gmp.q.mul %a, %b
    ```
  }];

  let arguments = (ins GMP_QType:$lhs, GMP_QType:$rhs);
  let results = (outs GMP_QType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_QDivOp : GMP_Op<"q.div", [Pure]> {
  let summary = "Divide two rationals";
  let description = [{
    Computes the quotient of two rational numbers (lhs / rhs).
    The result is automatically canonicalized.

    Example:
    ```mlir
    %quot = gmp.q.div %a, %b
    ```
  }];

  let arguments = (ins GMP_QType:$lhs, GMP_QType:$rhs);
  let results = (outs GMP_QType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Q Unary Operations
//===----------------------------------------------------------------------===//

def GMP_QNegOp : GMP_Op<"q.neg", [Pure]> {
  let summary = "Negate a rational";
  let description = [{
    Computes the negation of a rational number (-q).

    Example:
    ```mlir
    %neg = gmp.q.neg %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs GMP_QType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_QAbsOp : GMP_Op<"q.abs", [Pure]> {
  let summary = "Absolute value of a rational";
  let description = [{
    Computes the absolute value of a rational number.

    Example:
    ```mlir
    %abs = gmp.q.abs %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs GMP_QType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_QInvOp : GMP_Op<"q.inv", [Pure]> {
  let summary = "Multiplicative inverse of a rational";
  let description = [{
    Computes the multiplicative inverse (reciprocal) of a rational number.
    inv(a/b) = b/a

    Example:
    ```mlir
    %inv = gmp.q.inv %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs GMP_QType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Q Comparison Operations
//===----------------------------------------------------------------------===//

def GMP_QCompareOp : GMP_Op<"q.compare", [Pure]> {
  let summary = "Three-way comparison of rationals";
  let description = [{
    Compares two rationals and returns:
    - negative value if lhs < rhs
    - zero if lhs == rhs
    - positive value if lhs > rhs

    Example:
    ```mlir
    %cmp = gmp.q.compare %a, %b : i32
    ```
  }];

  let arguments = (ins GMP_QType:$lhs, GMP_QType:$rhs);
  let results = (outs I32:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_QEqualOp : GMP_Op<"q.equal", [Pure, Commutative]> {
  let summary = "Test equality of rationals";
  let description = [{
    Returns true if lhs equals rhs.

    Example:
    ```mlir
    %eq = gmp.q.equal %a, %b
    ```
  }];

  let arguments = (ins GMP_QType:$lhs, GMP_QType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_QLtOp : GMP_Op<"q.lt", [Pure]> {
  let summary = "Test if lhs is less than rhs";
  let description = [{
    Returns true if lhs < rhs.

    Example:
    ```mlir
    %lt = gmp.q.lt %a, %b
    ```
  }];

  let arguments = (ins GMP_QType:$lhs, GMP_QType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_QLeqOp : GMP_Op<"q.leq", [Pure]> {
  let summary = "Test if lhs is less than or equal to rhs";
  let description = [{
    Returns true if lhs <= rhs.

    Example:
    ```mlir
    %leq = gmp.q.leq %a, %b
    ```
  }];

  let arguments = (ins GMP_QType:$lhs, GMP_QType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_QGtOp : GMP_Op<"q.gt", [Pure]> {
  let summary = "Test if lhs is greater than rhs";
  let description = [{
    Returns true if lhs > rhs.

    Example:
    ```mlir
    %gt = gmp.q.gt %a, %b
    ```
  }];

  let arguments = (ins GMP_QType:$lhs, GMP_QType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

def GMP_QGeqOp : GMP_Op<"q.geq", [Pure]> {
  let summary = "Test if lhs is greater than or equal to rhs";
  let description = [{
    Returns true if lhs >= rhs.

    Example:
    ```mlir
    %geq = gmp.q.geq %a, %b
    ```
  }];

  let arguments = (ins GMP_QType:$lhs, GMP_QType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Q Classification Operations
//===----------------------------------------------------------------------===//

def GMP_QClassifyOp : GMP_Op<"q.classify", [Pure]> {
  let summary = "Classify a rational value";
  let description = [{
    Classifies a rational value and returns:
    - 0: Real (finite) number
    - 1: Positive infinity
    - 2: Negative infinity
    - 3: Undefined (0/0)

    Example:
    ```mlir
    %class = gmp.q.classify %q : i32
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs I32:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_QIsRealOp : GMP_Op<"q.is_real", [Pure]> {
  let summary = "Test if rational is a real (finite) number";
  let description = [{
    Returns true if the rational is a real (finite) number,
    false if it is infinity or undefined.

    Example:
    ```mlir
    %is_real = gmp.q.is_real %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs I1:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Q Rounding Operations
//===----------------------------------------------------------------------===//

def GMP_QFloorOp : GMP_Op<"q.floor", [Pure]> {
  let summary = "Floor of a rational (round toward negative infinity)";
  let description = [{
    Returns the largest integer not greater than the rational.
    This is the floor function: floor(3/2) = 1, floor(-3/2) = -2.

    Example:
    ```mlir
    %floor = gmp.q.floor %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_QCeilOp : GMP_Op<"q.ceil", [Pure]> {
  let summary = "Ceiling of a rational (round toward positive infinity)";
  let description = [{
    Returns the smallest integer not less than the rational.
    This is the ceiling function: ceil(3/2) = 2, ceil(-3/2) = -1.

    Example:
    ```mlir
    %ceil = gmp.q.ceil %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_QTruncOp : GMP_Op<"q.trunc", [Pure]> {
  let summary = "Truncate a rational (round toward zero)";
  let description = [{
    Returns the integer part of the rational, rounding toward zero.
    trunc(3/2) = 1, trunc(-3/2) = -1.

    Example:
    ```mlir
    %trunc = gmp.q.trunc %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_QRoundOp : GMP_Op<"q.round", [Pure]> {
  let summary = "Round a rational to nearest integer";
  let description = [{
    Rounds to the nearest integer. For values exactly halfway between
    two integers, rounds away from zero (banker's rounding variant).
    round(3/2) = 2, round(-3/2) = -2, round(5/2) = 3.

    Example:
    ```mlir
    %round = gmp.q.round %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Q Conversion Operations
//===----------------------------------------------------------------------===//

def GMP_QToBigIntOp : GMP_Op<"q.to_bigint", [Pure]> {
  let summary = "Convert rational to integer (truncating)";
  let description = [{
    Converts a rational to an integer by truncating toward zero.
    Equivalent to gmp.q.trunc but with explicit conversion semantics.

    Example:
    ```mlir
    %z = gmp.q.to_bigint %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs GMP_ZType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_QToF64Op : GMP_Op<"q.to_f64", [Pure]> {
  let summary = "Convert rational to double precision float";
  let description = [{
    Converts a rational to a 64-bit floating point number.
    The conversion may lose precision.

    Example:
    ```mlir
    %f = gmp.q.to_f64 %q
    ```
  }];

  let arguments = (ins GMP_QType:$operand);
  let results = (outs F64:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

def GMP_QFromZOp : GMP_Op<"q.from_z", [Pure]> {
  let summary = "Convert integer to rational";
  let description = [{
    Converts an arbitrary precision integer to a rational with denominator 1.

    Example:
    ```mlir
    %q = gmp.q.from_z %z
    ```
  }];

  let arguments = (ins GMP_ZType:$operand);
  let results = (outs GMP_QType:$result);

  let assemblyFormat = "$operand attr-dict";

  let hasFolder = 1;
}

#endif // GMP_OPS_TD
