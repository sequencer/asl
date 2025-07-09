//===- ASLDialect.td - ASL dialect -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_DIALECT
#define ASL_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ASL dialect definition.
//===----------------------------------------------------------------------===//

def ASL_Dialect : Dialect {
    let name = "asl";
    let summary = "A asl out-of-tree MLIR dialect.";
    let description = [{
        This dialect is one-to-one map to Arm Specification Language.
    }];
    let cppNamespace = "::mlir::asl";
}

//===----------------------------------------------------------------------===//
// Base asl operation definition.
//===----------------------------------------------------------------------===//

class ASL_Op<string mnemonic, list<Trait> traits = []> :
        Op<ASL_Dialect, mnemonic, traits>;

#endif // ASL_DIALECT
