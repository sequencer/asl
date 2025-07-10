//===- ASLAttributes.td - ASL dialect attributes ---------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_ATTRIBUTES
#define ASL_ATTRIBUTES

include "ASLDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// ASL Attribute Definitions
//===----------------------------------------------------------------------===//

class ASL_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ASL_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Enum Attributes
//===----------------------------------------------------------------------===//

def ASL_SubprogramType : I32EnumAttr<"SubprogramType", "ASL subprogram type", [
  I32EnumAttrCase<"ST_Procedure", 0, "procedure">,
  I32EnumAttrCase<"ST_Function", 1, "function">,
  I32EnumAttrCase<"ST_Getter", 2, "getter">,
  I32EnumAttrCase<"ST_EmptyGetter", 3, "empty_getter">,
  I32EnumAttrCase<"ST_Setter", 4, "setter">,
  I32EnumAttrCase<"ST_EmptySetter", 5, "empty_setter">
]> {
  let cppNamespace = "::mlir::asl";
}

def ASL_SubprogramTypeAttr : EnumAttr<ASL_Dialect, ASL_SubprogramType, "subprogram_type">;

def ASL_SubprogramBody : I32EnumAttr<"SubprogramBody", "ASL subprogram body", [
  I32EnumAttrCase<"SB_ASL", 0, "asl">,
  I32EnumAttrCase<"SB_Primitive", 1, "primitive">
]> {
  let cppNamespace = "::mlir::asl";
}

def ASL_SubprogramBodyAttr : EnumAttr<ASL_Dialect, ASL_SubprogramBody, "subprogram_body">;

def ASL_LocalDeclKeyword : I32EnumAttr<"LocalDeclKeyword", "ASL local declaration keyword", [
  I32EnumAttrCase<"LDK_Var", 0, "var">,
  I32EnumAttrCase<"LDK_Constant", 1, "constant">,
  I32EnumAttrCase<"LDK_Let", 2, "let">
]> {
  let cppNamespace = "::mlir::asl";
}

def ASL_LocalDeclKeywordAttr : EnumAttr<ASL_Dialect, ASL_LocalDeclKeyword, "local_decl_keyword">;

def ASL_GlobalDeclKeyword : I32EnumAttr<"GlobalDeclKeyword", "ASL global declaration keyword", [
  I32EnumAttrCase<"GDK_Constant", 0, "constant">,
  I32EnumAttrCase<"GDK_Config", 1, "config">,
  I32EnumAttrCase<"GDK_Let", 2, "let">,
  I32EnumAttrCase<"GDK_Var", 3, "var">
]> {
  let cppNamespace = "::mlir::asl";
}

def ASL_GlobalDeclKeywordAttr : EnumAttr<ASL_Dialect, ASL_GlobalDeclKeyword, "global_decl_keyword">;

def ASL_LoopDirection : I32EnumAttr<"LoopDirection", "ASL loop direction", [
  I32EnumAttrCase<"Up", 0, "up">,
  I32EnumAttrCase<"Down", 1, "down">
]> {
  let cppNamespace = "::mlir::asl";
}

def ASL_LoopDirectionAttr : EnumAttr<ASL_Dialect, ASL_LoopDirection, "loop_direction">;

def ASL_FuncQualifier : I32EnumAttr<"FuncQualifier", "ASL function qualifier", [
  I32EnumAttrCase<"Pure", 0, "pure">,
  I32EnumAttrCase<"Readonly", 1, "readonly">,
  I32EnumAttrCase<"Noreturn", 2, "noreturn">
]> {
  let cppNamespace = "::mlir::asl";
}

def ASL_FuncQualifierAttr : EnumAttr<ASL_Dialect, ASL_FuncQualifier, "func_qualifier">;

def ASL_OverrideInfo : I32EnumAttr<"OverrideInfo", "ASL override info", [
  I32EnumAttrCase<"Impdef", 0, "impdef">,
  I32EnumAttrCase<"Implementation", 1, "implementation">
]> {
  let cppNamespace = "::mlir::asl";
}

def ASL_OverrideInfoAttr : EnumAttr<ASL_Dialect, ASL_OverrideInfo, "override_info">;

def ASL_Version : I32EnumAttr<"Version", "ASL version", [
  I32EnumAttrCase<"V0", 0, "v0">,
  I32EnumAttrCase<"V1", 1, "v1">
]> {
  let cppNamespace = "::mlir::asl";
}

def ASL_VersionAttr : EnumAttr<ASL_Dialect, ASL_Version, "version">;

//===----------------------------------------------------------------------===//
// Simple Attributes
//===----------------------------------------------------------------------===//

def ASL_BitvectorMaskAttr : ASL_Attr<"BitvectorMask", "bitvector_mask"> {
  let summary = "ASL bitvector mask pattern";
  let description = [{
    Represents a bitvector mask pattern with bits, don't-care, and variable positions.
  }];
  
  let parameters = (ins "StringAttr":$pattern);
  let assemblyFormat = "`<` $pattern `>`";
}

#endif // ASL_ATTRIBUTES
