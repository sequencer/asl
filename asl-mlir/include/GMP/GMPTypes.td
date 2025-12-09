//===- GMPTypes.td - GMP type definitions ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GMP_TYPES_TD
#define GMP_TYPES_TD

include "mlir/IR/AttrTypeBase.td"
include "GMP/GMPDialect.td"

//===----------------------------------------------------------------------===//
// GMP Type Definitions
//===----------------------------------------------------------------------===//

class GMP_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<GMP_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Z Type - Arbitrary Precision Integer
//===----------------------------------------------------------------------===//

def GMP_ZType : GMP_Type<"Z", "z"> {
  let summary = "Arbitrary precision integer type";
  let description = [{
    Represents an unbounded integer (element of Z, the integers).

    At compile-time, constants are represented using MLIR's APInt for
    arbitrary precision arithmetic during constant folding.

    At runtime, this type is lowered to GMP's mpz_t.

    Example:
    ```mlir
    %a = gmp.z.constant #gmp.z<42> : !gmp.z
    %b = gmp.z.add %x, %y : !gmp.z
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Q Type - Arbitrary Precision Rational
//===----------------------------------------------------------------------===//

def GMP_QType : GMP_Type<"Q", "q"> {
  let summary = "Arbitrary precision rational type";
  let description = [{
    Represents an arbitrary precision rational number (element of Q, the
    rationals), stored as a pair of arbitrary precision integers (numerator
    and denominator).

    Rationals are always kept in canonical form:
    - The denominator is always positive
    - The GCD of numerator and denominator is 1
    - Zero is represented as 0/1

    Special values are supported for ASL compatibility:
    - Positive infinity: 1/0
    - Negative infinity: -1/0
    - Undefined: 0/0

    At compile-time, constants are evaluated using APInt arithmetic.
    At runtime, this type is lowered to GMP's mpq_t.

    Example:
    ```mlir
    %half = gmp.q.constant #gmp.q<1, 2> : !gmp.q
    %sum = gmp.q.add %x, %y : !gmp.q
    ```
  }];
}

#endif // GMP_TYPES_TD
