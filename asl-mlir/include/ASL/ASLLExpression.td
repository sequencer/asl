//===- ASLLExpression.td - ASL L-expression operations -----*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_LEXPRESSIONS_TD
#define ASL_LEXPRESSIONS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "ASL/ASLDialect.td"
include "ASL/ASLTypes.td"
include "ASL/ASLAttributes.td"

//===----------------------------------------------------------------------===//
// ASL L-Expression Operations Base Classes
//===----------------------------------------------------------------------===//

class ASL_LExprOp<string mnemonic, list<Trait> traits = []>
    : ASL_Op<mnemonic, traits> {
  let results = (outs ASL_LExprType:$result);
}

//===----------------------------------------------------------------------===//
// L-Expression Operations
//===----------------------------------------------------------------------===//

def ASL_LExprDiscardOp : ASL_LExprOp<"lexpr.discard"> {
  let summary = "Discard assignment target";
  let description = [{
    Represents a discarded assignment target (e.g., `-` in tuple destructuring).
    Corresponds to LE_Discard in the ASL AST.
    
    This operation creates an l-expression that discards any value assigned to it.
    It is commonly used in tuple destructuring when some elements are not needed.
  }];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def ASL_LExprVarOp : ASL_LExprOp<"lexpr.var"> {
  let summary = "Variable assignment target";
  let description = [{
    Represents assignment to a variable. The variable name is used to query 
    the local symbol table during parsing and code generation.
    Corresponds to LE_Var in the ASL AST.
    
    This operation creates an l-expression that represents a variable that
    can be assigned to.
  }];

  let arguments = (ins StrAttr:$name);
  let assemblyFormat = "$name attr-dict `:` type($result)";
}

def ASL_LExprSliceOp : ASL_LExprOp<"lexpr.slice"> {
  let summary = "Slice assignment target";
  let description = [{
    Represents a write to a slice of the base l-expression.
    The slice operations define which part of the base to modify.
    Corresponds to LE_Slice in the ASL AST.
    
    This operation creates an l-expression that represents a slice of another
    l-expression that can be assigned to.
  }];

  let arguments = (ins ASL_LExprType:$base, Variadic<ASL_SliceType>:$slices);
  let assemblyFormat =
      "$base `[` $slices `]` attr-dict `:` functional-type(operands, results)";
}

def ASL_LExprSetArrayOp : ASL_LExprOp<"lexpr.set_array"> {
  let summary = "Array element assignment with integer index";
  let description = [{
    Represents assignment to an array element with integer index.
    Takes an l-expression as the base array and a normal value as the index.
    Corresponds to LE_SetArray in the ASL AST.
    
    This operation creates an l-expression that represents an array element
    indexed by an integer that can be assigned to.
  }];

  let arguments = (ins ASL_LExprType:$base, ASL_IntType:$index);
  let assemblyFormat =
      "$base `[` $index `]` attr-dict `:` functional-type(operands, results)";
}

def ASL_LExprSetEnumArrayOp : ASL_LExprOp<"lexpr.set_enum_array"> {
  let summary = "Array element assignment with enum index";
  let description = [{
    Represents assignment to an array element with enumeration index.
    Takes an l-expression as the base array and a normal value as the enum index.
    Corresponds to LE_SetEnumArray in the ASL AST.
    
    This operation creates an l-expression that represents an array element
    indexed by an enumeration value that can be assigned to.
  }];

  let arguments = (ins ASL_LExprType:$base, AnyType:$index);
  let assemblyFormat =
      "$base `[` $index `]` attr-dict `:` functional-type(operands, results)";
}

def ASL_LExprSetFieldOp : ASL_LExprOp<"lexpr.set_field"> {
  let summary = "Record field assignment";
  let description = [{
    Represents assignment to a record field.
    Takes an l-expression as the base record and has a field name attribute.
    Corresponds to LE_SetField in the ASL AST.
    
    This operation creates an l-expression that represents a record field
    that can be assigned to.
  }];

  let arguments = (ins ASL_LExprType:$base, StrAttr:$field_name);
  let assemblyFormat =
      "$base $field_name attr-dict `:` functional-type(operands, results)";
}

def ASL_LExprSetFieldsOp : ASL_LExprOp<"lexpr.set_fields"> {
  let summary = "Multiple record fields assignment with type annotations";
  let description = [{
    Represents assignment to multiple record fields with type annotations.
    Takes an l-expression as the base record, field names, and type annotations.
    Corresponds to LE_SetFields in the ASL AST.
    
    This operation creates an l-expression that represents multiple record fields
    that can be assigned to simultaneously with type annotations for bit-packing.
  }];

  let arguments = (ins ASL_LExprType:$base, ArrayAttr:$field_names,
      ArrayAttr:$annotations);
  let assemblyFormat = "$base $field_names $annotations attr-dict `:` "
                       "functional-type(operands, results)";

  let extraClassDeclaration = [{
    // Verify that field_names contains StringAttrs and annotations contains DictionaryAttrs
    // with 'x' and 'y' IntegerAttrs
    ::mlir::LogicalResult verify();
  }];
}

def ASL_LExprSetCollectionFieldsOp
    : ASL_LExprOp<"lexpr.set_collection_fields"> {
  let summary = "Collection fields assignment with type annotations";
  let description = [{
    Represents assignment to collection fields with type annotations.
    Has collection name, field names, and type annotations as attributes.
    Corresponds to LE_SetCollectionFields in the ASL AST.
    
    This operation creates an l-expression that represents collection fields
    that can be assigned to with type annotations.
  }];

  let arguments = (ins StrAttr:$collection, ArrayAttr:$field_names,
      ArrayAttr:$annotations);
  let assemblyFormat =
      "$collection $field_names $annotations attr-dict `:` type($result)";

  let extraClassDeclaration = [{
    // Verify that field_names contains StringAttrs and annotations contains DictionaryAttrs
    // with 'x' and 'y' IntegerAttrs
    ::mlir::LogicalResult verify();
  }];
}

def ASL_LExprDestructuringOp : ASL_LExprOp<"lexpr.destructuring"> {
  let summary = "Tuple destructuring assignment";
  let description = [{
    Represents tuple destructuring assignment.
    Takes a variadic number of l-expression values as inputs.
    Corresponds to LE_Destructuring in the ASL AST.
    
    This operation creates an l-expression that represents a tuple pattern
    for destructuring assignment, where each element can be assigned to.
  }];

  let arguments = (ins Variadic<ASL_LExprType>:$elements);
  let assemblyFormat =
      "`(` $elements `)` attr-dict `:` functional-type(operands, results)";
}

#endif // ASL_LEXPRESSIONS_TD
