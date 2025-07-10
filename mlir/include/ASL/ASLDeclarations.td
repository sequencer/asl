//===- ASLDeclarations.td - ASL dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_DECLARATIONS
#define ASL_DECLARATIONS

// Add proper function definition operation
def ASL_FuncOp : ASL_Op<"func", [
  FunctionOpInterface, IsolatedFromAbove, Symbol
]> {
  let summary = "ASL function definition";
  let description = [{
    The `asl.func` operation represents ASL function definitions with parameters and body.
    
    Example:
    ```mlir
    asl.func @function_name(%param1: !asl.int, %param2: !asl.real) -> !asl.bool {
      subprogram_type = "ST_Function",
      qualifier = "Pure"
    } {
      // function body
    }
    ```
  }];
  
  let arguments = (ins 
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    ASL_SubprogramTypeAttr:$subprogramType,
    OptionalAttr<ASL_FuncQualifierAttr>:$qualifier,
    OptionalAttr<BoolAttr>:$builtin,
    OptionalAttr<ASL_OverrideInfoAttr>:$override
  );
  let results = (outs);
  let regions = (region AnyRegion:$body);
  
  let hasCustomAssemblyFormat = 1;
}

def ASL_GlobalDeclOp : ASL_Op<"global_decl", [Symbol]> {
  let summary = "ASL global declaration";
  let description = [{
    The `asl.global_decl` operation represents global variable/constant declarations.
    
    Example:
    ```mlir
    asl.global_decl "GDK_Constant", @global_name, !asl.int, %init_value {
      visibility = "public"
    }
    ```
  }];
  
  let arguments = (ins 
    ASL_GlobalDeclKeywordAttr:$keyword,
    SymbolNameAttr:$name,
    TypeAttr:$declType,
    Optional<ASLType>:$initValue,
    OptionalAttr<StrAttr>:$visibility
  );
  let results = (outs);
}


def ASL_TypeDeclOp : ASL_Op<"type_decl", [Symbol]> {
  let summary = "ASL type declaration";
  let description = [{
    The `asl.type_decl` operation represents type declarations and definitions.
    
    Example:
    ```mlir
    asl.type_decl @MyType : !asl.record<"MyType", [#asl.field<"x", !asl.int>]> {
      visibility = "public"
    }
    ```
  }];
  
  let arguments = (ins 
    SymbolNameAttr:$name,
    TypeAttr:$declType,
    OptionalAttr<StrAttr>:$visibility
  );
  let results = (outs);
}

def ASL_PragmaDeclOp : ASL_Op<"pragma_decl", []> {
  let summary = "ASL pragma declaration";
  let description = [{
    The `asl.pragma_decl` operation represents pragma declarations at the top level.
    
    Example:
    ```mlir
    asl.pragma_decl "pragma_name", %arg1, %arg2 : (!asl.int, !asl.string)
    ```
  }];
  
  let arguments = (ins StrAttr:$identifier, Variadic<ASLType>:$args);
  let results = (outs);
}

#endif // ASL_DECLARATIONS
