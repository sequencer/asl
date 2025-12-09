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

def GMP_ZConstantOp : GMP_Op<"z.constant", [Pure]> {
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
// Q Module: Rational Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Q Construction Operations
//===----------------------------------------------------------------------===//

def GMP_QConstantOp : GMP_Op<"q.constant", [Pure]> {
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
