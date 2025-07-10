//===- ASLDialect.td - ASL dialect definition -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_DIALECT
#define ASL_DIALECT

include "mlir/IR/OpBase.td"

def ASL_Dialect : Dialect {
  let name = "asl";
  let summary = "ASL (Architecture Specification Language) dialect";
  let description = [{
    The ASL dialect provides operations and types for representing
    Architecture Specification Language constructs in MLIR.
  }];
  let cppNamespace = "::mlir::asl";
  
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class ASL_Op<string mnemonic, list<Trait> traits = []> :
    Op<ASL_Dialect, mnemonic, traits>;

#endif // ASL_DIALECT
