//===- ASLTypes.td - ASL dialect types --------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_TYPES
#define ASL_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "ASL/ASLDialect.td"

//===----------------------------------------------------------------------===//
// ASL Type Definitions
//===----------------------------------------------------------------------===//

// Base class for all ASL types
class ASL_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ASL_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Primitive Types
//===----------------------------------------------------------------------===//

// Integer type with optional constraints
def ASL_IntType : ASL_Type<"Int", "int"> {
  let summary = "ASL integer type with optional constraints";
  let description = [{
    Represents an ASL integer type that can be:
    - Unconstrained: !asl.int
    - Pending constraint: !asl.int<pending>
    - Parameterized: !asl.int<param, "param_name">
    - Constrained: !asl.int<constraints[exact(42)]>
    - Range constrained: !asl.int<constraints[range(0, 255)]>
  }];
  
  let parameters = (ins
    OptionalParameter<"StringAttr">:$constraintKind,
    OptionalParameter<"ArrayAttr">:$constraints,
    OptionalParameter<"StringAttr">:$parameterName
  );
  
  // Use custom assembly format (to be implemented in C++)
  let hasCustomAssemblyFormat = 1;
  
  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, nullptr, nullptr, nullptr);
    }]>,
    TypeBuilder<(ins "StringRef":$paramName), [{
      auto paramAttr = StringAttr::get($_ctxt, paramName);
      return $_get($_ctxt, nullptr, nullptr, paramAttr);
    }]>
  ];
}

// Real number type
def ASL_RealType : ASL_Type<"Real", "real"> {
  let summary = "ASL real number type";
  let description = "Represents arbitrary-precision rational numbers in ASL";
  let hasCustomAssemblyFormat = 1;
}

// String type
def ASL_StringType : ASL_Type<"String", "string"> {
  let summary = "ASL string type";
  let description = "Represents string values in ASL";
  let hasCustomAssemblyFormat = 1;
}

// Boolean type
def ASL_BoolType : ASL_Type<"Bool", "bool"> {
  let summary = "ASL boolean type";
  let description = "Represents boolean values (true/false) in ASL";
  let hasCustomAssemblyFormat = 1;
}

// Bitvector type with optional bitfields
def ASL_BitsType : ASL_Type<"Bits", "bits"> {
  let summary = "ASL bitvector type with optional bitfields";
  let description = [{
    Represents a bitvector type with configurable width and optional bitfield definitions:
    - Simple: !asl.bits<32>
    - With expression width: !asl.bits<width_expr>
    - With bitfields: !asl.bits<8, [simple("flag", [0:0])]>
  }];
  
  let parameters = (ins
    "Attribute":$widthExpr,
    OptionalParameter<"ArrayAttr">:$bitfields
  );
  
  let assemblyFormat = [{
    `<` $widthExpr (`,` `[` $bitfields^ `]`)? `>`
  }];
  
  let builders = [
    TypeBuilder<(ins "Attribute":$width), [{
      return $_get($_ctxt, width, nullptr);
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Composite Types
//===----------------------------------------------------------------------===//

// Enumeration type
def ASL_EnumType : ASL_Type<"Enum", "enum"> {
  let summary = "ASL enumeration type";
  let description = [{
    Represents an enumeration type with a list of string labels:
    - !asl.enum<["RED", "GREEN", "BLUE"]>
  }];
  
  let parameters = (ins "ArrayAttr":$labels);
  let assemblyFormat = "`<` `[` $labels `]` `>`";
  
  let builders = [
    TypeBuilder<(ins "ArrayRef<StringRef>":$labelNames), [{
      SmallVector<Attribute> attrs;
      for (auto name : labelNames) {
        attrs.push_back(StringAttr::get($_ctxt, name));
      }
      return $_get($_ctxt, ArrayAttr::get($_ctxt, attrs));
    }]>
  ];
}

// Tuple type
def ASL_TupleType : ASL_Type<"Tuple", "tuple"> {
  let summary = "ASL tuple type";
  let description = [{
    Represents a tuple type with ordered elements:
    - Empty: !asl.tuple<>
    - Single: !asl.tuple<!asl.int>
    - Multiple: !asl.tuple<!asl.int, !asl.bool, !asl.real>
  }];
  
  let parameters = (ins OptionalParameter<"ArrayRef<Type>">:$elements);
  let assemblyFormat = "`<` ($elements^)? `>`";
  
  let builders = [
    TypeBuilder<(ins), [{
      return $_get($_ctxt, ArrayRef<Type>{});
    }]>
  ];
}

// Array type  
def ASL_ArrayType : ASL_Type<"Array", "array"> {
  let summary = "ASL array type";
  let description = [{
    Represents an array type with element type and index specification:
    - Length-based: !asl.array<!asl.int, length(10)>
    - Expression-based: !asl.array<!asl.int, length(expr)>
    - Enum-indexed: !asl.array<!asl.real, enum("Color", ["RED", "GREEN", "BLUE"])>
  }];
  
  let parameters = (ins
    "Type":$elementType,
    "Attribute":$indexType
  );
  
  let assemblyFormat = "`<` $elementType `,` $indexType `>`";
}

// Record type
def ASL_RecordType : ASL_Type<"Record", "record"> {
  let summary = "ASL record type";
  let description = [{
    Represents a record type with named fields:
    - !asl.record<"Point", [("x", !asl.real), ("y", !asl.real)]>
  }];
  
  let parameters = (ins
    "StringAttr":$name,
    "ArrayAttr":$fields
  );
  
  let assemblyFormat = "`<` $name `,` `[` $fields `]` `>`";
  
  let builders = [
    TypeBuilder<(ins "StringRef":$name, "ArrayRef<Attribute>":$fieldAttrs), [{
      auto nameAttr = StringAttr::get($_ctxt, name);
      return $_get($_ctxt, nameAttr, ArrayAttr::get($_ctxt, fieldAttrs));
    }]>
  ];
}

// Exception type
def ASL_ExceptionType : ASL_Type<"Exception", "exception"> {
  let summary = "ASL exception type";
  let description = [{
    Represents an exception type with named fields:
    - !asl.exception<"Error", [("code", !asl.int), ("msg", !asl.string)]>
  }];
  
  let parameters = (ins
    "StringAttr":$name,
    "ArrayAttr":$fields
  );
  
  let assemblyFormat = "`<` $name `,` `[` $fields `]` `>`";
  
  let builders = [
    TypeBuilder<(ins "StringRef":$name, "ArrayRef<Attribute>":$fieldAttrs), [{
      auto nameAttr = StringAttr::get($_ctxt, name);
      return $_get($_ctxt, nameAttr, ArrayAttr::get($_ctxt, fieldAttrs));
    }]>
  ];
}

// Collection type
def ASL_CollectionType : ASL_Type<"Collection", "collection"> {
  let summary = "ASL collection type";
  let description = [{
    Represents a collection type with named fields:
    - !asl.collection<[("field1", !asl.int), ("field2", !asl.string)]>
  }];
  
  let parameters = (ins "ArrayAttr":$fields);
  let assemblyFormat = "`<` `[` $fields `]` `>`";
  
  let builders = [
    TypeBuilder<(ins "ArrayRef<Attribute>":$fieldAttrs), [{
      return $_get($_ctxt, ArrayAttr::get($_ctxt, fieldAttrs));
    }]>
  ];
}

// Named type (type alias)
def ASL_NamedType : ASL_Type<"Named", "named"> {
  let summary = "ASL named type reference";
  let description = [{
    Represents a reference to a named type declaration:
    - !asl.named<@MyType>
    - !asl.named<@nested.scope.Type>
  }];
  
  let parameters = (ins "FlatSymbolRefAttr":$name);
  let assemblyFormat = "`<` $name `>`";
}

// L-expression type (for assignment targets)
def ASL_LExprType : ASL_Type<"LExpr", "lexpr"> {
  let summary = "ASL l-expression type";
  let description = [{
    Represents the type of left-hand side expressions used in assignments.
    This is an internal type used by the dialect.
  }];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Type Predicates and Constraints
//===----------------------------------------------------------------------===//

// Predicate for any ASL type
def IsASLType : Type<CPred<"$_self.isa<::mlir::asl::ASLType>()">, 
                     "ASL type">;

// Predicate for ASL integer types
def IsASLIntType : Type<CPred<"$_self.isa<::mlir::asl::IntType>()">,
                        "ASL integer type">;

// Predicate for ASL bits types  
def IsASLBitsType : Type<CPred<"$_self.isa<::mlir::asl::BitsType>()">,
                         "ASL bits type">;

// Predicate for ASL numeric types (int, real)
def IsASLNumericType : Type<Or<[IsASLIntType.predicate,
                                CPred<"$_self.isa<::mlir::asl::RealType>()">]>,
                            "ASL numeric type">;

// Predicate for ASL composite types
def IsASLCompositeType : Type<Or<[
  CPred<"$_self.isa<::mlir::asl::TupleType>()">,
  CPred<"$_self.isa<::mlir::asl::ArrayType>()">,
  CPred<"$_self.isa<::mlir::asl::RecordType>()">,
  CPred<"$_self.isa<::mlir::asl::ExceptionType>()">,
  CPred<"$_self.isa<::mlir::asl::CollectionType>()">
]>, "ASL composite type">;

//===----------------------------------------------------------------------===//
// Type Aliases for Common Usage
//===----------------------------------------------------------------------===//

// Constraint for operands that must be ASL types
def ASLType : Type<IsASLType.predicate, "any ASL type">;
def ASLIntType : Type<IsASLIntType.predicate, "ASL integer type">;
def ASLBitsType : Type<IsASLBitsType.predicate, "ASL bits type">;
def ASLNumericType : Type<IsASLNumericType.predicate, "ASL numeric type">;

#endif // ASL_TYPES
