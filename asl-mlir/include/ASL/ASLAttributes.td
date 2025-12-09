//===- ASLAttributes.td - ASL dialect attributes ----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_ATTRIBUTES_TD
#define ASL_ATTRIBUTES_TD

include "ASL/ASLDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// ASL Attribute Classes
//===----------------------------------------------------------------------===//

class ASL_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ASL_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Enum Attributes
//===----------------------------------------------------------------------===//

// SubprogramType Attribute
def ASL_SubprogramTypeAttr
    : I32EnumAttr<"SubprogramType", "ASL subprogram type",
                  [I32EnumAttrCase<"procedure", 0, "procedure">,
                   I32EnumAttrCase<"function", 1, "function">,
                   I32EnumAttrCase<"getter", 2, "getter">,
                   I32EnumAttrCase<"emptygetter", 3, "emptygetter">,
                   I32EnumAttrCase<"setter", 4, "setter">,
                   I32EnumAttrCase<"emptysetter", 5, "emptysetter">]> {
  let cppNamespace = "::mlir::asl";
}

// ConstraintKind Attribute
def ASL_ConstraintKindTypeAttr
    : I32EnumAttr<"ConstraintKindType", "ASL constraint kind type",
                  [I32EnumAttrCase<"unconstrained", 0, "unconstrained">,
                   I32EnumAttrCase<"constrained", 1, "constrained">,
                   I32EnumAttrCase<"pending", 2, "pending">,
                   I32EnumAttrCase<"parameterized", 3, "parameterized">]> {
  let cppNamespace = "::mlir::asl";
}

// PrecisionLossFlag Attribute
def ASL_PrecisionLossFlagAttr
    : I32EnumAttr<"PrecisionLossFlag", "ASL precision loss flag",
                  [I32EnumAttrCase<"full", 0, "full">,
                   I32EnumAttrCase<"lost", 1, "lost">]> {
  let cppNamespace = "::mlir::asl";
}

// BitField Kind Attribute
def ASL_BitFieldKindAttr : I32EnumAttr<"BitFieldKind", "ASL bitfield kind",
                                       [I32EnumAttrCase<"simple", 0, "simple">,
                                        I32EnumAttrCase<"nested", 1, "nested">,
                                        I32EnumAttrCase<"type", 2, "type">]> {
  let cppNamespace = "::mlir::asl";
}

// Slice Kind Attribute
def ASL_SliceKindAttr : I32EnumAttr<"SliceKind", "ASL slice kind",
                                    [I32EnumAttrCase<"single", 0, "single">,
                                     I32EnumAttrCase<"range", 1, "range">,
                                     I32EnumAttrCase<"length", 2, "length">,
                                     I32EnumAttrCase<"star", 3, "star">]> {
  let cppNamespace = "::mlir::asl";
}

// ArrayIndex Kind Attribute
def ASL_ArrayIndexKindAttr
    : I32EnumAttr<"ArrayIndexKind", "ASL array index kind",
                  [I32EnumAttrCase<"int_type", 0, "int">,
                   I32EnumAttrCase<"enum_type", 1, "enum">]> {
  let cppNamespace = "::mlir::asl";
}

// Local Declaration Keyword Attribute
def ASL_LDKAttr : I32EnumAttr<"LDK", "ASL local declaration keyword",
                              [I32EnumAttrCase<"var", 0, "var">,
                               I32EnumAttrCase<"constant", 1, "constant">,
                               I32EnumAttrCase<"let", 2, "let">]> {
  let cppNamespace = "::mlir::asl";
}

// Global Declaration Keyword Attribute
def ASL_GDKAttr : I32EnumAttr<"GDK", "ASL global declaration keyword",
                              [I32EnumAttrCase<"constant", 0, "constant">,
                               I32EnumAttrCase<"config", 1, "config">,
                               I32EnumAttrCase<"let", 2, "let">,
                               I32EnumAttrCase<"var", 3, "var">]> {
  let cppNamespace = "::mlir::asl";
}

// Local Declaration Item Kind Attribute
def ASL_LDIKindAttr : I32EnumAttr<"LDIKind", "ASL local declaration item kind",
                                  [I32EnumAttrCase<"var", 0, "var">,
                                   I32EnumAttrCase<"tuple", 1, "tuple">]> {
  let cppNamespace = "::mlir::asl";
}

// For Direction Attribute
def ASL_ForDirectionAttr : I32EnumAttr<"ForDirection", "ASL for direction",
                                       [I32EnumAttrCase<"up", 0, "up">,
                                        I32EnumAttrCase<"down", 1, "down">]> {
  let cppNamespace = "::mlir::asl";
}

// Function Qualifier Attribute
def ASL_FuncQualifierAttr
    : I32EnumAttr<"FuncQualifier", "ASL function qualifier",
                  [I32EnumAttrCase<"pure", 0, "pure">,
                   I32EnumAttrCase<"readonly", 1, "readonly">,
                   I32EnumAttrCase<"noreturn", 2, "noreturn">]> {
  let cppNamespace = "::mlir::asl";
}

// Override Info Attribute
def ASL_OverrideInfoAttr
    : I32EnumAttr<"OverrideInfo", "ASL override info",
                  [I32EnumAttrCase<"impdef", 0, "impdef">,
                   I32EnumAttrCase<"implementation", 1, "implementation">]> {
  let cppNamespace = "::mlir::asl";
}

// Version Attribute
def ASL_VersionAttr : I32EnumAttr<"Version", "ASL version",
                                  [I32EnumAttrCase<"v0", 0, "v0">,
                                   I32EnumAttrCase<"v1", 1, "v1">]> {
  let cppNamespace = "::mlir::asl";
}

//===----------------------------------------------------------------------===//
// Complex Attributes
//===----------------------------------------------------------------------===//

// BitVector Mask Attribute
def ASL_BitVectorMaskAttr : ASL_Attr<"BitVectorMask", "bitvector_mask"> {
  let summary = "ASL bitvector mask attribute";
  let description = [{
    Represents a bitvector mask pattern using string serialization.
    Uses mask_to_string for serialization and mask_of_string for parsing.
  }];
  let parameters = (ins "StringAttr":$value);
  let assemblyFormat = "$value";
}

// Constraint Kind Attribute
def ASL_ConstraintKindAttr : ASL_Attr<"ConstraintKind", "constraint_kind"> {
  let summary = "ASL constraint kind attribute";
  let description = [{
    Represents the constraint information for integer types.
  }];
  let parameters = (ins "ConstraintKindTypeAttr":$kind,
      "ArrayAttr":$constraint_list,
      "PrecisionLossFlagAttr":$precision_loss_flag);
  let assemblyFormat =
      "`<` $kind `,` $constraint_list `,` $precision_loss_flag `>`";
}

// Integer Constraint Attribute
def ASL_IntConstraintAttr : ASL_Attr<"IntConstraint", "int_constraint"> {
  let summary = "ASL integer constraint attribute";
  let description = [{
    Represents constraints on integer values (exact or range).
  }];
  let parameters = (ins OptionalParameter<"IntegerAttr">:$exact,
      OptionalParameter<"IntegerAttr">:$lhs,
      OptionalParameter<"IntegerAttr">:$rhs);
  let assemblyFormat = "`<` ($exact^)? (`,` $lhs^)? (`,` $rhs^)? `>`";
}

// BitField Attribute
def ASL_BitFieldAttr : ASL_Attr<"BitField", "bitfield"> {
  let summary = "ASL bitfield attribute";
  let description = [{
    Represents a bitfield definition with identifier, kind, slices,
    and optional nested bitfields or type information.
  }];
  let parameters = (ins "StringAttr":$identifier, "BitFieldKindAttr":$kind,
      "ArrayAttr":$slices, OptionalParameter<"ArrayAttr">:$nested,
      OptionalParameter<"TypeAttr">:$type);
  let assemblyFormat =
      "`<` $identifier `,` $kind `,` $slices (`,` $nested^)? (`,` $type^)? `>`";
}

// Slice Attribute
def ASL_SliceAttr : ASL_Attr<"Slice", "slice"> {
  let summary = "ASL slice attribute";
  let description = [{
    Represents a resolved slice operation with kind-specific parameters.
  }];
  let parameters = (ins "SliceKindAttr":$kind,
      OptionalParameter<"IntegerAttr">:$idx,
      OptionalParameter<"IntegerAttr">:$lhs,
      OptionalParameter<"IntegerAttr">:$rhs,
      OptionalParameter<"IntegerAttr">:$start,
      OptionalParameter<"IntegerAttr">:$factor,
      OptionalParameter<"IntegerAttr">:$length);
  let assemblyFormat = "`<` $kind (`,` $idx^)? (`,` $lhs^)? (`,` $rhs^)? (`,` "
                       "$start^)? (`,` $factor^)? (`,` $length^)? `>`";
}

// Array Index Attribute
def ASL_ArrayIndexAttr : ASL_Attr<"ArrayIndex", "array_index"> {
  let summary = "ASL array index attribute";
  let description = [{
    Represents array indexing scheme, either by expression or enumeration.
  }];
  let parameters = (ins "ArrayIndexKindAttr":$kind,
      OptionalParameter<"IntegerAttr">:$length,
      OptionalParameter<"StringAttr">:$identifier,
      OptionalParameter<"ArrayAttr">:$labels);
  let assemblyFormat =
      "`<` $kind (`,` $length^)? (`,` $identifier^)? (`,` $labels^)? `>`";
}

// Record Field Attribute
def ASL_RecordFieldAttr : ASL_Attr<"RecordField", "record_field"> {
  let summary = "ASL record field attribute";
  let description = [{
    Represents a field in a record-like structure with identifier and type.
  }];
  let parameters = (ins "StringAttr":$identifier, "TypeAttr":$type);
  let assemblyFormat = "`<` $identifier `,` $type `>`";
}

// Local Declaration Item Attribute
def ASL_LDIAttr : ASL_Attr<"LDI", "local_decl_item"> {
  let summary = "ASL local declaration item attribute";
  let description = [{
    Represents a local declaration item, either a single variable or tuple.
  }];
  let parameters = (ins "LDIKindAttr":$kind,
      OptionalParameter<"StringAttr">:$var,
      OptionalParameter<"ArrayAttr">:$tuple);
  let assemblyFormat = "`<` $kind (`,` $var^)? (`,` $tuple^)? `>`";
}

#endif // ASL_ATTRIBUTES_TD