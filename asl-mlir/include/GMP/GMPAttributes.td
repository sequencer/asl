//===- GMPAttributes.td - GMP attribute definitions --------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GMP_ATTRIBUTES_TD
#define GMP_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"
include "GMP/GMPDialect.td"

//===----------------------------------------------------------------------===//
// GMP Attribute Definitions
//===----------------------------------------------------------------------===//

class GMP_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<GMP_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Z Attribute - Arbitrary Precision Integer Constant
//===----------------------------------------------------------------------===//

def GMP_ZAttr : GMP_Attr<"Z", "z"> {
  let summary = "Arbitrary precision integer constant";
  let description = [{
    Represents a compile-time constant arbitrary precision integer.
    The value is stored as a decimal string representation to support
    integers of any size.

    Examples:
    ```mlir
    #gmp.z<0>                          // Zero
    #gmp.z<1>                          // One
    #gmp.z<-42>                        // Negative
    #gmp.z<12345678901234567890>       // Large integer
    ```
  }];

  let parameters = (ins "std::string":$value);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the value as an APInt with sufficient bits.
    llvm::APInt getAPInt() const;

    /// Check if the value is zero.
    bool isZero() const;

    /// Check if the value is one.
    bool isOne() const;

    /// Check if the value is negative.
    bool isNegative() const;
  }];
}

//===----------------------------------------------------------------------===//
// Q Attribute - Arbitrary Precision Rational Constant
//===----------------------------------------------------------------------===//

def GMP_QAttr : GMP_Attr<"Q", "q"> {
  let summary = "Arbitrary precision rational constant";
  let description = [{
    Represents a compile-time constant arbitrary precision rational.
    The value is stored as two decimal strings: numerator and denominator.
    The rational is always stored in canonical form.

    Special values:
    - `#gmp.q<1, 0>` - Positive infinity
    - `#gmp.q<-1, 0>` - Negative infinity
    - `#gmp.q<0, 0>` - Undefined

    Examples:
    ```mlir
    #gmp.q<1, 2>                       // 1/2
    #gmp.q<-3, 4>                      // -3/4
    #gmp.q<0, 1>                       // Zero
    #gmp.q<1, 0>                       // +Infinity
    ```
  }];

  let parameters = (ins "std::string":$numerator, "std::string":$denominator);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get the numerator as an APInt.
    llvm::APInt getNumeratorAPInt() const;

    /// Get the denominator as an APInt.
    llvm::APInt getDenominatorAPInt() const;

    /// Check if the value is zero.
    bool isZero() const;

    /// Check if the value is positive infinity.
    bool isPosInf() const;

    /// Check if the value is negative infinity.
    bool isNegInf() const;

    /// Check if the value is undefined.
    bool isUndef() const;

    /// Check if the value is a real number (not inf or undef).
    bool isReal() const;
  }];
}

#endif // GMP_ATTRIBUTES_TD
