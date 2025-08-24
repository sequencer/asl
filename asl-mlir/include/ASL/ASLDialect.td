//===- ASLDialect.td - ASL dialect definition --------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_DIALECT_TD
#define ASL_DIALECT_TD

include "mlir/IR/OpBase.td"

def ASL_Dialect : Dialect {
  let name = "asl";
  let summary = "ASL (Arm Specification Language) dialect";
  let description = [{
    The ASL dialect provides operations and types for representing
    Arm Specification Language constructs in MLIR.
    
    This dialect supports the complete ASL abstract syntax tree including:
    - Expressions (literals, variables, binary/unary operations, calls, etc.)
    - Statements (assignments, control flow, declarations, etc.)  
    - Types (integers, bitvectors, records, arrays, tuples, etc.)
    - Declarations (functions, global storage, type definitions, etc.)
    - L-expressions for assignment targets
    - Pattern matching constructs
    
    The dialect follows the formal semantics specified in the ASL Reference
    Manual and is based on the ASL AST defined in the `asllib` library.
  }];
  let cppNamespace = "::mlir::asl";

  // Enable constant materialization for folded operations
  let hasConstantMaterializer = 1;

  // Use default attribute and type parsers/printers
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  // Enable canonicalization patterns
  let hasCanonicalizer = 1;

  // Dependent dialects for built-in types and operations
  let dependentDialects = ["mlir::BuiltinDialect"];

  let extraClassDeclaration = [{
    /// Register all ASL types.
    void registerTypes();
    
    /// Register all ASL attributes.
    void registerAttributes();
  }];
}

class ASL_Op<string mnemonic, list<Trait> traits = []>
    : Op<ASL_Dialect, mnemonic, traits>;

#endif // ASL_DIALECT_TD
