//===- GMPDialect.td - GMP dialect definition --------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GMP_DIALECT_TD
#define GMP_DIALECT_TD

include "mlir/IR/OpBase.td"

def GMP_Dialect : Dialect {
  let name = "gmp";
  let summary = "GMP dialect for arbitrary precision arithmetic";
  let description = [{
    The GMP dialect provides operations and types for arbitrary-precision
    integer (Z) and rational (Q) arithmetic. The API design follows the
    OCaml Zarith library conventions.

    This dialect supports:
    - Arbitrary precision integers (!gmp.z)
    - Arbitrary precision rationals (!gmp.q)
    - Compile-time constant folding via APInt
    - Multiple division modes (truncated, floor, ceiling, Euclidean)
    - Number theory operations (GCD, LCM, primality, etc.)

    At compile-time, operations on constants are evaluated using LLVM's APInt.
    At runtime, the dialect lowers to GMP library calls (mpz_*, mpq_*).
  }];
  let cppNamespace = "::mlir::gmp";

  // Enable constant materialization for folded operations
  let hasConstantMaterializer = 1;

  // Use default attribute and type parsers/printers
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  // Dependent dialects
  let dependentDialects = ["mlir::BuiltinDialect"];

  let extraClassDeclaration = [{
    /// Register all GMP types.
    void registerTypes();

    /// Register all GMP attributes.
    void registerAttributes();
  }];
}

class GMP_Op<string mnemonic, list<Trait> traits = []>
    : Op<GMP_Dialect, mnemonic, traits>;

#endif // GMP_DIALECT_TD
