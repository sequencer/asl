//===- ASLTypes.td - ASL type definition -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_TYPE_TD
#define ASL_TYPE_TD

include "mlir/IR/AttrTypeBase.td"
include "ASL/ASLDialect.td"

//===----------------------------------------------------------------------===//
// ASL Type Definitions
//===----------------------------------------------------------------------===//

class ASL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ASL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Integer Type
//===----------------------------------------------------------------------===//

def ASL_IntType : ASL_Type<"Int", "int"> {
  let summary = "ASL integer type with optional constraints";
  let description = [{
    Represents an ASL integer type that can be unconstrained or have various
    constraint kinds applied to it. The constraint is stored as an attribute.
  }];

  let parameters = (ins "ConstraintKindAttr":$constraint);

  let assemblyFormat = "`<` $constraint `>`";

  let extraClassDeclaration = [{
    bool isConstrained() const;
  }];
}

//===----------------------------------------------------------------------===//
// Bitvector Type
//===----------------------------------------------------------------------===//

def ASL_BitsType : ASL_Type<"Bits", "bits"> {
  let summary = "ASL bitvector type with width and optional bitfields";
  let description = [{
    Represents an ASL bitvector type with a specified width and optional
    bitfield definitions. A width of -2 indicates a parameterized type,
    -1 indicates the width is not calculated yet. Both bitfield and width
    should be explicitly converted by E_ATC.
  }];

  let parameters = (ins "IntegerAttr":$width, "ArrayAttr":$bitfields);

  let assemblyFormat = "`<` $width `,` $bitfields `>`";

  let extraClassDeclaration = [{
    bool isParameterized() const;
    unsigned getBitWidth() const;
  }];
}

//===----------------------------------------------------------------------===//
// Real Type
//===----------------------------------------------------------------------===//

def ASL_RealType : ASL_Type<"Real", "real"> {
  let summary = "ASL real number type";
  let description = [{
    Represents an ASL real number type for floating-point values.
  }];
}

//===----------------------------------------------------------------------===//
// String Type
//===----------------------------------------------------------------------===//

def ASL_StringType : ASL_Type<"String", "string"> {
  let summary = "ASL string type";
  let description = [{
    Represents an ASL string type for text values.
  }];
}

//===----------------------------------------------------------------------===//
// Label Type
//===----------------------------------------------------------------------===//

def ASL_LabelType : ASL_Type<"Label", "label"> {
  let summary = "ASL label type";
  let description = [{
    Represents an ASL label type for enumeration values and identifiers.
  }];
}

//===----------------------------------------------------------------------===//
// Boolean Type Note
//===----------------------------------------------------------------------===//

// ASL boolean type is represented by the standard i1 type in MLIR
// See IR documentation: "The `asl.bool` type is represented by the standard
// `i1` type."

//===----------------------------------------------------------------------===//
// Enumeration Type
//===----------------------------------------------------------------------===//

def ASL_EnumType : ASL_Type<"Enum", "enum"> {
  let summary = "ASL enumeration type";
  let description = [{
    Represents an ASL enumeration type with a list of possible string labels.
  }];

  let parameters = (ins "ArrayAttr":$labels);

  let assemblyFormat = "`<` $labels `>`";

  let extraClassDeclaration = [{
    ::llvm::ArrayRef<::mlir::StringAttr> getStringLabels() const;
  }];
}

//===----------------------------------------------------------------------===//
// Tuple Type
//===----------------------------------------------------------------------===//

def ASL_TupleType : ASL_Type<"Tuple", "tuple"> {
  let summary = "ASL tuple type";
  let description = [{
    Represents an ASL tuple type containing multiple element types.
  }];

  let parameters = (ins "ArrayAttr":$types);

  let assemblyFormat = "`<` $types `>`";

  let extraClassDeclaration = [{
    unsigned getNumTypes() const;
    ::mlir::Type getType(unsigned index) const;
  }];
}

//===----------------------------------------------------------------------===//
// Array Type
//===----------------------------------------------------------------------===//

def ASL_ArrayType : ASL_Type<"Array", "array"> {
  let summary = "ASL array type";
  let description = [{
    Represents an ASL array type with an element type and indexing scheme.
  }];

  let parameters = (ins "TypeAttr":$element_type, "ArrayIndexAttr":$index);

  let assemblyFormat = "`<` $element_type `,` $index `>`";
}

//===----------------------------------------------------------------------===//
// Record Type
//===----------------------------------------------------------------------===//

def ASL_RecordType : ASL_Type<"Record", "record"> {
  let summary = "ASL record type";
  let description = [{
    Represents an ASL record type with named fields.
  }];

  let parameters = (ins "ArrayAttr":$fields);

  let assemblyFormat = "`<` $fields `>`";
}

//===----------------------------------------------------------------------===//
// Exception Type
//===----------------------------------------------------------------------===//

def ASL_ExceptionType : ASL_Type<"Exception", "exception"> {
  let summary = "ASL exception type";
  let description = [{
    Represents an ASL exception type with named fields.
  }];

  let parameters = (ins "ArrayAttr":$fields);

  let assemblyFormat = "`<` $fields `>`";
}

//===----------------------------------------------------------------------===//
// Collection Type
//===----------------------------------------------------------------------===//

def ASL_CollectionType : ASL_Type<"Collection", "collection"> {
  let summary = "ASL collection type";
  let description = [{
    Represents an ASL collection type with named fields.
  }];

  let parameters = (ins "ArrayAttr":$fields);

  let assemblyFormat = "`<` $fields `>`";
}

//===----------------------------------------------------------------------===//
// Named Type
//===----------------------------------------------------------------------===//

def ASL_NamedType : ASL_Type<"Named", "named"> {
  let summary = "ASL named type";
  let description = [{
    Represents an ASL named type that references a type alias.
  }];

  let parameters = (ins "StringAttr":$name);

  let assemblyFormat = "`<` $name `>`";
}

//===----------------------------------------------------------------------===//
// Slice Type
//===----------------------------------------------------------------------===//

def ASL_SliceType : ASL_Type<"Slice", "slice"> {
  let summary = "ASL slice descriptor type";
  let description = [{
    Represents an ASL slice descriptor used for bitvector and array slicing
    operations. This type is used by slice operations to describe index ranges.
  }];
}

//===----------------------------------------------------------------------===//
// BitField Type
//===----------------------------------------------------------------------===//

def ASL_BitFieldType : ASL_Type<"BitField", "bitfield"> {
  let summary = "ASL bitfield type";
  let description = [{
    Represents an ASL bitfield type used in bitfield operations and 
    materialization during ATC operations.
    This type is used by bitfield ATC operations to describe bit field
    structures that require runtime materialization.
  }];
}

//===----------------------------------------------------------------------===//
// L-Expression Type
//===----------------------------------------------------------------------===//

def ASL_LExprType : ASL_Type<"LExpr", "lexpr"> {
  let summary = "ASL l-expression type";
  let description = [{
    Represents an ASL l-expression (left-hand side expression) used in
    assignment statements. L-expressions describe memory locations that
    can be assigned to.
  }];
}

#endif // ASL_TYPE_TD