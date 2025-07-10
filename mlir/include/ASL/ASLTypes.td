//===- ASLTypes.td - ASL dialect types -------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_TYPES
#define ASL_TYPES

include "ASLDialect.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// ASL Type Definitions
//===----------------------------------------------------------------------===//

class ASL_Type<string name, list<Trait> traits = []>
    : TypeDef<ASL_Dialect, name, traits> {
}

def ASL_IntType : ASL_Type<"Int"> {
  let summary = "ASL integer type";
  let description = [{
    The ASL integer type represents arbitrary precision integers.
    It can optionally include constraints to restrict the value range.
  }];
  // TODO: add constraint support.
  let typeName = "asl.int";
}

def ASL_BitsType : ASL_Type<"Bits"> {
  let summary = "ASL bitvector type";
  let description = [{
    The ASL bits type represents fixed-width bitvectors.
    The width can be a constant integer or an expression.
    Optionally includes bitfield definitions.
  }];
  
  // TODO: add bitfields support.
  let parameters = (ins 
    "IntegerAttr":$width
  );
  let typeName = "asl.bits";
}

def ASL_RealType : ASL_Type<"Real"> {
  let summary = "ASL real number type";
  let description = [{
    The ASL real type represents exact rational numbers.
    It can optionally include constraints to restrict the value range.
  }];
  let typeName = "asl.real";
}

def ASL_StringType : ASL_Type<"String"> {
  let summary = "ASL string type";
  let description = [{
    The ASL string type represents character strings.
  }];
  let typeName = "asl.string";
}

def ASL_BoolType : ASL_Type<"Bool"> {
  let summary = "ASL boolean type";
  let description = [{
    The ASL boolean type represents logical true/false values.
  }];
  let typeName = "asl.bool";
}

def ASL_EnumType : ASL_Type<"Enum"> {
  let summary = "ASL enumeration type";
  let description = [{
    The ASL enumeration type represents a finite set of named values.
  }];
  
  let parameters = (ins "ArrayAttr":$labels);
  let typeName = "asl.enum";
}

def ASL_TupleType : ASL_Type<"Tuple"> {
  let summary = "ASL tuple type";
  let description = [{
    The ASL tuple type represents ordered collections of heterogeneous elements.
  }];
  
  let parameters = (ins "ArrayAttr":$elementTypes);
  let typeName = "asl.tuple";
}

def ASL_ArrayType : ASL_Type<"Array"> {
  let summary = "ASL array type";
  let description = [{
    The ASL array type represents arrays with element type and index specification.
    Arrays can be indexed by integers or enumeration values.
  }];
  
  let parameters = (ins "Attribute":$indexType, "Type":$elementType);
  let typeName = "asl.array";
}

def ASL_RecordType : ASL_Type<"Record"> {
  let summary = "ASL record type";
  let description = [{
    The ASL record type represents structured data with named fields.
  }];
  
  let parameters = (ins 
    ArrayRefParameter<"Attribute", "field names and types">:$fields
  );
  let typeName = "asl.record";
}


def ASL_ExceptionType : ASL_Type<"Exception"> {
  let summary = "ASL exception type";
  let description = [{
    The ASL exception type represents exception values with named fields.
  }];
  
  let parameters = (ins 
    ArrayRefParameter<"Attribute", "field names and types">:$fields
  );
  let typeName = "asl.exception";
}

def ASL_CollectionType : ASL_Type<"Collection"> {
  let summary = "ASL collection type";
  let description = [{
    The ASL collection type represents collections with named fields,
    similar to records but with different semantics.
  }];
  
  let parameters = (ins 
    ArrayRefParameter<"Attribute", "field names and types">:$fields
  );
  let typeName = "asl.collection";
}

def ASL_NamedType : ASL_Type<"Named"> {
  let summary = "ASL named type";
  let description = [{
    The ASL named type represents type variables and named type references.
    Used for parameterized types and type aliases.
  }];
  
  let parameters = (ins "StringAttr":$name);
  let typeName = "asl.named";
}

def ASL_LExprType : ASL_Type<"LExpr"> {
  let summary = "ASL l-expression type";
  let description = [{
    The ASL l-expression type represents assignable expressions (lvalues).
  }];
  
  let parameters = (ins "Type":$valueType);
  let typeName = "asl.lexpr";
}

// Composite type constraints
def ASLNumericType : AnyTypeOf<[ASL_IntType, ASL_RealType]>;
def ASLIndexableType : AnyTypeOf<[ASL_BitsType, ASL_ArrayType, ASL_StringType]>;
def ASLStructuralType : AnyTypeOf<[ASL_RecordType, ASL_TupleType, ASL_CollectionType]>;

// General ASL type constraint
def ASLType : AnyTypeOf<[
  ASL_IntType, ASL_RealType, ASL_BoolType, ASL_BitsType, ASL_StringType,
  ASL_ArrayType, ASL_TupleType, ASL_RecordType, ASL_CollectionType, ASL_EnumType, 
  ASL_ExceptionType, ASL_NamedType, ASL_LExprType
]>;

#endif // ASL_TYPES
