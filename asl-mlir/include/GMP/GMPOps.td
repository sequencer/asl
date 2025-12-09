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
