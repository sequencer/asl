//===- ASLDeclarations.td - ASL declaration operations ----*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_DECLARATIONS_TD
#define ASL_DECLARATIONS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "ASL/ASLDialect.td"
include "ASL/ASLTypes.td"
include "ASL/ASLAttributes.td"

//===----------------------------------------------------------------------===//
// ASL Declaration Operations Base Classes
//===----------------------------------------------------------------------===//

class ASL_DeclOp<string mnemonic, list<Trait> traits = []>
    : ASL_Op<mnemonic, traits> {
  // Declarations typically don't have results, they modify global state
}

//===----------------------------------------------------------------------===//
// Declaration Operations
//===----------------------------------------------------------------------===//

def ASL_FuncDeclOp : ASL_DeclOp<"func", [IsolatedFromAbove]> {
  let summary = "Function declaration";
  let description = [{
    Represents a function declaration with its signature, parameters, and body.
    Corresponds to D_Func in the ASL AST.
    
    This operation declares a function with:
    - Function name and parameters with types
    - Optional return type
    - Function body in the first region
    - Various function attributes (primitive, builtin, etc.)
    - Optional recursion limit
  }];

  let arguments = (ins StrAttr:$name,
      ArrayAttr:$parameters, // Array of DictionaryAttr with identifier and type
      OptionalAttr<TypeAttr>:$return_type, BoolAttr:$primitive,
      ArrayAttr:$args,       // Array of StringAttr for argument names
      ArrayAttr:$args_types, // Array of TypeAttr for argument types (see IR
                             // docs)
      ASL_SubprogramTypeAttr:$subprogram_type,
      OptionalAttr<ASL_FuncQualifierAttr>:$qualifier,
      OptionalAttr<ASL_OverrideInfoAttr>:$override, BoolAttr:$builtin,
      Optional<ASL_IntType>:$recurse_limit);

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    $name `(` $parameters `)` 
    (`->` $return_type^)?
    `primitive` $primitive
    `args` $args
    `args_types` $args_types
    `subprogram_type` $subprogram_type
    (`qualifier` $qualifier^)?
    (`override` $override^)?
    `builtin` $builtin
    (`recurse_limit` $recurse_limit^ `:` type($recurse_limit))?
    attr-dict-with-keyword
    $body
  }];

  let extraClassDeclaration = [{
    /// Verify the function declaration
    ::mlir::LogicalResult verify();
  }];
}

def ASL_GlobalStorageDeclOp : ASL_DeclOp<"global"> {
  let summary = "Global variable declaration";
  let description = [{
    Represents a global variable declaration with initialization.
    Corresponds to D_GlobalStorage in the ASL AST.
    
    This operation declares a global variable with:
    - Declaration keyword (constant, config, let, var, parameter)
    - Variable name and type
    - Initial value
  }];

  let arguments = (ins ASL_GDKAttr:$keyword, StrAttr:$name, TypeAttr:$type,
      AnyType:$initial_value);

  let assemblyFormat = [{
    $keyword $name `:` $type `=` $initial_value attr-dict `:` type($initial_value)
  }];

  let extraClassDeclaration = [{
    /// Verify the global storage declaration
    ::mlir::LogicalResult verify();
  }];
}

def ASL_TypeDeclOp : ASL_DeclOp<"type_decl"> {
  let summary = "Type declaration";
  let description = [{
    Represents a type declaration/alias with optional subtype information.
    Corresponds to D_TypeDecl in the ASL AST.
    
    This operation declares a type with:
    - Type alias name
    - Underlying type
    - Optional subtype definitions with fields
  }];

  let arguments = (ins StrAttr:$identifier, TypeAttr:$type,
      OptionalAttr<DictionaryAttr>:$subtypes // Dictionary with identifier and
                                             // fields
  );

  let assemblyFormat = [{
    $identifier `:` $type (`subtypes` $subtypes^)? attr-dict
  }];

  let extraClassDeclaration = [{
    /// Verify the type declaration
    ::mlir::LogicalResult verify();
  }];
}

def ASL_PragmaDeclOp : ASL_DeclOp<"pragma"> {
  let summary = "Pragma declaration";
  let description = [{
    Represents a pragma declaration with identifier and expression arguments.
    Corresponds to D_Pragma in the ASL AST.
    
    This operation declares a pragma with:
    - Pragma identifier/name
    - Variadic expression arguments
  }];

  let arguments = (ins StrAttr:$identifier, Variadic<AnyType>:$expressions);

  let assemblyFormat = [{
    $identifier `(` $expressions `)` attr-dict `:` type($expressions)
  }];

  let extraClassDeclaration = [{
    /// Verify the pragma declaration
    ::mlir::LogicalResult verify();
  }];
}

#endif // ASL_DECLARATIONS_TD
