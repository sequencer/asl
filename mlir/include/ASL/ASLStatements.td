//===- ASLOperations.td - ASL dialect ops ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_STATEMENTS
#define ASL_STATEMENTS

def ASL_LexprDiscardOp : ASL_Op<"lexpr.discard", [Pure]> {
  let summary = "ASL discard l-expression";
  let description = [{
    The `asl.lexpr.discard` operation represents a discarded assignment target.
    
    Example:
    ```mlir
    %discard = asl.lexpr.discard : !asl.lexpr
    ```
  }];
  
  let arguments = (ins);
  let results = (outs ASL_LExprType:$result);
}

def ASL_LexprVarOp : ASL_Op<"lexpr.var", [Pure]> {
  let summary = "ASL variable l-expression";
  let description = [{
    The `asl.lexpr.var` operation represents assignment to a variable.
    
    Example:
    ```mlir
    %var_target = asl.lexpr.var "variable_name" : !asl.lexpr
    ```
  }];
  
  let arguments = (ins StrAttr:$name);
  let results = (outs ASL_LExprType:$result);
}

def ASL_LexprSliceOp : ASL_Op<"lexpr.slice", [Pure]> {
  let summary = "ASL slice l-expression";
  let description = [{
    The `asl.lexpr.slice` operation represents assignment to a slice of an expression.
    
    Example:
    ```mlir
    %slice_target = asl.lexpr.slice %base, [%slice1, %slice2] : !asl.lexpr
    ```
  }];
  
  let arguments = (ins ASL_LExprType:$base, Variadic<ASLType>:$slices);
  let results = (outs ASL_LExprType:$result);
}


def ASL_LexprSetArrayOp : ASL_Op<"lexpr.set_array", [Pure]> {
  let summary = "ASL array element l-expression";
  let description = [{
    The `asl.lexpr.set_array` operation represents assignment to an array element.
    
    Example:
    ```mlir
    %array_target = asl.lexpr.set_array %base, %index : !asl.lexpr
    ```
  }];
  
  let arguments = (ins ASL_LExprType:$base, ASL_IntType:$index);
  let results = (outs ASL_LExprType:$result);
}


def ASL_LexprSetEnumArrayOp : ASL_Op<"lexpr.set_enum_array", [Pure]> {
  let summary = "ASL enum array element l-expression";
  let description = [{
    The `asl.lexpr.set_enum_array` operation represents assignment to an array element with enumeration index.
    
    Example:
    ```mlir
    %enum_array_target = asl.lexpr.set_enum_array %base, %enum_index : !asl.lexpr
    ```
  }];
  
  let arguments = (ins ASL_LExprType:$base, ASL_EnumType:$enumIndex);
  let results = (outs ASL_LExprType:$result);
}


def ASL_LexprSetFieldOp : ASL_Op<"lexpr.set_field", [Pure]> {
  let summary = "ASL field l-expression";
  let description = [{
    The `asl.lexpr.set_field` operation represents assignment to a record field.
    
    Example:
    ```mlir
    %field_target = asl.lexpr.set_field %base, "field_name" : !asl.lexpr
    ```
  }];
  
  let arguments = (ins ASL_LExprType:$base, StrAttr:$fieldName);
  let results = (outs ASL_LExprType:$result);
}

def ASL_LexprSetFieldsOp : ASL_Op<"lexpr.set_fields", [Pure]> {
  let summary = "ASL multiple fields l-expression";
  let description = [{
    The `asl.lexpr.set_fields` operation represents assignment to multiple record fields.
    
    Example:
    ```mlir
    %fields_target = asl.lexpr.set_fields %base, ["f1", "f2"] {types = [(!asl.int, !asl.real)]} : !asl.lexpr
    ```
  }];
  
  let arguments = (ins ASL_LExprType:$base, ArrayAttr:$fieldNames, OptionalAttr<ArrayAttr>:$types);
  let results = (outs ASL_LExprType:$result);
}

def ASL_LexprSetCollectionFieldsOp : ASL_Op<"lexpr.set_collection_fields", [Pure]> {
  let summary = "ASL collection fields l-expression";
  let description = [{
    The `asl.lexpr.set_collection_fields` operation represents assignment to collection fields.
    
    Example:
    ```mlir
    %collection_target = asl.lexpr.set_collection_fields "collection", ["field1"] {types = [(!asl.int)]} : !asl.lexpr
    ```
  }];
  
  let arguments = (ins StrAttr:$collectionName, ArrayAttr:$fieldNames, OptionalAttr<ArrayAttr>:$types);
  let results = (outs ASL_LExprType:$result);
}

def ASL_LexprDestructuringOp : ASL_Op<"lexpr.destructuring", [Pure]> {
  let summary = "ASL tuple destructuring l-expression";
  let description = [{
    The `asl.lexpr.destructuring` operation represents tuple destructuring assignment.
    
    Example:
    ```mlir
    %destructure_target = asl.lexpr.destructuring [%lhs1, %lhs2] : !asl.lexpr
    ```
  }];
  
  let arguments = (ins Variadic<ASL_LExprType>:$targets);
  let results = (outs ASL_LExprType:$result);
}


def ASL_StmtPassOp : ASL_Op<"stmt.pass", []> {
  let summary = "ASL no-operation statement";
  let description = [{
    The `asl.stmt.pass` operation represents a no-operation statement.
    
    Example:
    ```mlir
    asl.stmt.pass
    ```
  }];
  
  let arguments = (ins);
  let results = (outs);
}


def ASL_StmtSeqOp : ASL_Op<"stmt.seq", []> {
  let summary = "ASL sequential composition statement";
  let description = [{
    The `asl.stmt.seq` operation represents sequential composition of statements.
    
    Example:
    ```mlir
    asl.stmt.seq {
      // first statement
    } {
      // second statement
    }
    ```
  }];
  
  let arguments = (ins);
  let results = (outs);
  let regions = (region AnyRegion:$first, AnyRegion:$second);
}


def ASL_StmtDeclOp : ASL_Op<"stmt.decl", []> {
  let summary = "ASL local variable declaration statement";
  let description = [{
    The `asl.stmt.decl` operation represents local variable declaration.
    
    Example:
    ```mlir
    asl.stmt.decl "LDK_Var", "variable_name", %type, %init_value {
      decl_item = "LDI_Var"
    }
    ```
  }];
  
  let arguments = (ins 
    ASL_LocalDeclKeywordAttr:$keyword,
    StrAttr:$name,
    OptionalAttr<TypeAttr>:$declType,
    Optional<ASLType>:$initValue,
    OptionalAttr<StrAttr>:$declItem
  );
  let results = (outs);
}

def ASL_StmtAssignOp : ASL_Op<"stmt.assign", []> {
  let summary = "ASL assignment statement";
  let description = [{
    The `asl.stmt.assign` operation represents assignment to an l-expression.
    
    Example:
    ```mlir
    asl.stmt.assign %lhs, %rhs : !asl.int
    ```
  }];
  
  let arguments = (ins ASL_LExprType:$lhs, ASLType:$rhs);
  let results = (outs);
}


def ASL_StmtCallOp : ASL_Op<"stmt.call", [CallOpInterface, AttrSizedOperandSegments]> {
  let summary = "ASL statement call";
  let description = [{
    The `asl.stmt.call` operation represents function calls with parameters and arguments.
    
    Example:
    ```mlir
    %result = asl.stmt.call @function_name(%param1, %param2)(%arg1, %arg2) : (!asl.int, !asl.real) -> !asl.bool {
      call_type = "ST_Function"
    }
    ```
  }];
  
  let arguments = (ins 
    FlatSymbolRefAttr:$callee,
    Variadic<ASLType>:$params,
    Variadic<ASLType>:$args,
    ASL_SubprogramTypeAttr:$callType
  );
  let results = (outs Variadic<ASLType>:$results);
}


def ASL_StmtReturnOp : ASL_Op<"stmt.return", [Terminator]> {
  let summary = "ASL return statement";
  let description = [{
    The `asl.stmt.return` operation represents a return statement with optional value.
    
    Example:
    ```mlir
    asl.stmt.return           // return without value
    asl.stmt.return %value : !asl.int  // return with value
    ```
  }];
  
  let arguments = (ins Optional<ASLType>:$value);
  let results = (outs);
}

def ASL_StmtCondOp : ASL_Op<"stmt.cond", []> {
  let summary = "ASL conditional statement";
  let description = [{
    The `asl.stmt.cond` operation represents conditional execution with then and else branches.
    
    Example:
    ```mlir
    asl.stmt.cond %condition {
      // then block
    } else {
      // else block
    }
    ```
  }];
  
  let arguments = (ins ASL_BoolType:$condition);
  let results = (outs);
  let regions = (region AnyRegion:$thenRegion, AnyRegion:$elseRegion);
}


def ASL_StmtAssertOp : ASL_Op<"stmt.assert", []> {
  let summary = "ASL assertion statement";
  let description = [{
    The `asl.stmt.assert` operation represents an assertion with a boolean condition.
    
    Example:
    ```mlir
    asl.stmt.assert %condition : !asl.bool
    ```
  }];
  
  let arguments = (ins ASL_BoolType:$condition);
  let results = (outs);
}


def ASL_StmtForOp : ASL_Op<"stmt.for", []> {
  let summary = "ASL for loop statement";
  let description = [{
    The `asl.stmt.for` operation represents a for loop with index variable, bounds, and direction.
    
    Example:
    ```mlir
    asl.stmt.for "index_name", %start, %end {direction = "Up", limit = %limit_expr} {
      // loop body
    }
    ```
  }];
  
  let arguments = (ins 
    StrAttr:$indexName,
    ASL_IntType:$start,
    ASL_IntType:$end,
    ASL_LoopDirectionAttr:$direction,
    Optional<ASL_IntType>:$limit
  );
  let results = (outs);
  let regions = (region AnyRegion:$body);
}


def ASL_StmtWhileOp : ASL_Op<"stmt.while", []> {
  let summary = "ASL while loop statement";
  let description = [{
    The `asl.stmt.while` operation represents a while loop with condition and optional limit.
    
    Example:
    ```mlir
    asl.stmt.while %condition {limit = %limit_expr} {
      // loop body
    }
    ```
  }];
  
  let arguments = (ins ASL_BoolType:$condition, Optional<ASL_IntType>:$limit);
  let results = (outs);
  let regions = (region AnyRegion:$body);
}

def ASL_StmtRepeatOp : ASL_Op<"stmt.repeat", []> {
  let summary = "ASL repeat-until loop statement";
  let description = [{
    The `asl.stmt.repeat` operation represents a repeat-until loop.
    
    Example:
    ```mlir
    asl.stmt.repeat {limit = %limit_expr} {
      // loop body
    } until %condition
    ```
  }];
  
  let arguments = (ins ASL_BoolType:$condition, Optional<ASL_IntType>:$limit);
  let results = (outs);
  let regions = (region AnyRegion:$body);
}


def ASL_StmtThrowOp : ASL_Op<"stmt.throw", [Terminator]> {
  let summary = "ASL throw statement";
  let description = [{
    The `asl.stmt.throw` operation represents throwing an exception.
    
    Example:
    ```mlir
    asl.stmt.throw                                        // implicit throw
    asl.stmt.throw %exception : !asl.exception<"Error", []>  // explicit throw
    ```
  }];
  
  let arguments = (ins Optional<ASL_ExceptionType>:$exception, OptionalAttr<TypeAttr>:$exceptionType);
  let results = (outs);
}


def ASL_StmtTryOp : ASL_Op<"stmt.try", []> {
  let summary = "ASL try-catch statement";
  let description = [{
    The `asl.stmt.try` operation represents try-catch with exception handlers.
    
    Example:
    ```mlir
    asl.stmt.try {
      // protected code
    } catch [
      ("exception_var", !asl.exception<"Error1", []>) : {
        // handler 1
      },
      (none, !asl.exception<"Error2", []>) : {
        // handler 2  
      }
    ] otherwise {
      // otherwise handler
    }
    ```
  }];
  
  let arguments = (ins ArrayAttr:$catchers, OptionalAttr<BoolAttr>:$hasOtherwise);
  let results = (outs);
  let regions = (region AnyRegion:$protectedRegion, AnyRegion:$otherwiseRegion, VariadicRegion<AnyRegion>:$catcherRegions);
}


def ASL_StmtPrintOp : ASL_Op<"stmt.print", []> {
  let summary = "ASL print statement";
  let description = [{
    The `asl.stmt.print` operation represents printing values with optional newline and debug flags.
    
    Example:
    ```mlir
    asl.stmt.print %arg1, %arg2 {newline = true, debug = false} : (!asl.int, !asl.string)
    ```
  }];
  
  let arguments = (ins 
    Variadic<ASLType>:$args,
    DefaultValuedAttr<BoolAttr, "true">:$newline,
    DefaultValuedAttr<BoolAttr, "false">:$debug
  );
  let results = (outs);
}


def ASL_StmtUnreachableOp : ASL_Op<"stmt.unreachable", [Terminator]> {
  let summary = "ASL unreachable statement";
  let description = [{
    The `asl.stmt.unreachable` operation indicates dead code that should never be reached.
    
    Example:
    ```mlir
    asl.stmt.unreachable
    ```
  }];
  
  let arguments = (ins);
  let results = (outs);
}

def ASL_StmtPragmaOp : ASL_Op<"stmt.pragma", []> {
  let summary = "ASL pragma statement";
  let description = [{
    The `asl.stmt.pragma` operation represents pragma statements with tool-specific hints.
    
    Example:
    ```mlir
    asl.stmt.pragma "pragma_name", %arg1, %arg2 : (!asl.int, !asl.string)
    ```
  }];
  
  let arguments = (ins StrAttr:$identifier, Variadic<ASLType>:$args);
  let results = (outs);
}

#endif // ASL_STATEMENTS
