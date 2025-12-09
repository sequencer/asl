//===- ASLStatement.td - ASL statement operations ---------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_STATEMENT_TD
#define ASL_STATEMENT_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "ASL/ASLDialect.td"
include "ASL/ASLTypes.td"
include "ASL/ASLAttributes.td"

//===----------------------------------------------------------------------===//
// ASL Statement Operations Base Classes
//===----------------------------------------------------------------------===//

class ASL_StmtOp<string mnemonic, list<Trait> traits = []>
    : ASL_Op<mnemonic, traits> {
  // Statements have no SSA outputs
}

//===----------------------------------------------------------------------===//
// Statement Operations
//===----------------------------------------------------------------------===//

def ASL_StmtPassOp : ASL_StmtOp<"stmt.pass"> {
  let summary = "No-operation statement";
  let description = [{
    A no-operation statement that performs no action.
    Corresponds to S_Pass in ASL.
  }];

  let assemblyFormat = "attr-dict";
}

def ASL_StmtSeqOp : ASL_StmtOp<"stmt.seq"> {
  let summary = "Sequential composition statement";
  let description = [{
    Sequential composition of two statements.
    Has a block of region with two nested statements.
    Corresponds to S_Seq in ASL.
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

def ASL_StmtDeclOp : ASL_StmtOp<"stmt.decl"> {
  let summary = "Local storage declaration statement";
  let description = [{
    Local storage declaration.
    Has optional initial_value input and attributes: keyword (LDKAttr), 
    item (LDIAttr), and optional type (TypeAttr).
    Corresponds to S_Decl in ASL.
  }];

  let arguments = (ins Optional<AnyType>:$initial_value, ASL_LDKAttr:$keyword,
      ASL_LDIAttr:$item, OptionalAttr<TypeAttr>:$type);
  let assemblyFormat = "$keyword $item (`type` $type^)? (`=` $initial_value^ "
                       "`:` type($initial_value))? attr-dict";
}

def ASL_StmtAssignOp : ASL_StmtOp<"stmt.assign"> {
  let summary = "Assignment statement";
  let description = [{
    Assignment statement. Takes lhs (l-expression) and rhs (value) inputs.
    Corresponds to S_Assign in ASL.
  }];

  let arguments = (ins ASL_LExprType:$lhs, AnyType:$rhs);
  let assemblyFormat =
      "$lhs `=` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_StmtCallOp : ASL_StmtOp<"stmt.call"> {
  let summary = "Procedure call statement";
  let description = [{
    Procedure call without return value.
    Similar to asl.expr.call but for statements.
    Corresponds to S_Call in ASL.
  }];

  let arguments = (ins StrAttr:$name, Variadic<AnyType>:$args,
      I32Attr:$params_size, ASL_SubprogramTypeAttr:$call_type);
  let assemblyFormat = "$name `(` $args `)` `,` $params_size `,` $call_type "
                       "attr-dict `:` functional-type(operands, results)";
}

def ASL_StmtReturnOp : ASL_StmtOp<"stmt.return"> {
  let summary = "Return statement";
  let description = [{
    Return statement with optional expression input.
    Corresponds to S_Return in ASL.
  }];

  let arguments = (ins Optional<AnyType>:$value);
  let assemblyFormat = "($value^)? attr-dict (`:` type($value)^)?";
}

def ASL_StmtCondOp : ASL_StmtOp<"stmt.cond"> {
  let summary = "Conditional statement";
  let description = [{
    Conditional statement (if-then-else).
    Takes condition input, has one region with two blocks for then/else branches.
    Corresponds to S_Cond in ASL.
  }];

  let arguments = (ins I1:$condition);
  let regions = (region SizedRegion<2>:$branches);
  let assemblyFormat = "$condition $branches attr-dict `:` type($condition)";
}

def ASL_StmtAssertOp : ASL_StmtOp<"stmt.assert"> {
  let summary = "Assertion statement";
  let description = [{
    Assertion statement. Takes boolean expression input.
    Corresponds to S_Assert in ASL.
  }];

  let arguments = (ins I1:$condition);
  let assemblyFormat = "$condition attr-dict `:` type($condition)";
}

def ASL_StmtForOp : ASL_StmtOp<"stmt.for"> {
  let summary = "For loop statement";
  let description = [{
    For loop statement.
    Attributes: index_name (StringAttr), direction (ForDirectionAttr).
    Inputs: start, end for loop bounds, optional limit for static limit.
    Body in first region.
    Corresponds to S_For in ASL.
  }];

  let arguments = (ins StrAttr:$index_name, ASL_ForDirectionAttr:$direction,
      ASL_IntType:$start, ASL_IntType:$end, Optional<AnyType>:$limit);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat =
      "$index_name $direction $start `to` $end (`limit` $limit^)? $body "
      "attr-dict `:` functional-type(operands, results)";
}

def ASL_StmtWhileOp : ASL_StmtOp<"stmt.while"> {
  let summary = "While loop statement";
  let description = [{
    While loop with condition input.
    Input: optional limit for static limit.
    Body in first region.
    Corresponds to S_While in ASL.
  }];

  let arguments = (ins I1:$condition, Optional<AnyType>:$limit);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$condition (`limit` $limit^)? $body attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_StmtRepeatOp : ASL_StmtOp<"stmt.repeat"> {
  let summary = "Repeat-until loop statement";
  let description = [{
    Repeat-until loop.
    Input: optional limit for static limit.
    Body in first region.
    Corresponds to S_Repeat in ASL.
  }];

  let arguments = (ins I1:$condition, Optional<AnyType>:$limit);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body `until` $condition (`limit` $limit^)? attr-dict "
                       "`:` functional-type(operands, results)";
}

def ASL_StmtThrowOp : ASL_StmtOp<"stmt.throw"> {
  let summary = "Exception throwing statement";
  let description = [{
    Exception throwing statement.
    Optional expression input and type annotation.
    Corresponds to S_Throw in ASL.
  }];

  let arguments = (ins Optional<AnyType>:$value, OptionalAttr<TypeAttr>:$type);
  let assemblyFormat =
      "($value^)? (`type` $type^)? attr-dict (`:` type($value)^)?";
}

def ASL_StmtTryOp : ASL_StmtOp<"stmt.try"> {
  let summary = "Try-catch statement";
  let description = [{
    Try-catch statement.
    First region contains the protected statement.
    Second region contains catch handlers and an optional otherwise block.
    Corresponds to S_Try in ASL.
  }];

  let regions = (region SizedRegion<1>:$protected, AnyRegion:$handlers);
  let assemblyFormat = "$protected $handlers attr-dict";
}

def ASL_StmtPrintOp : ASL_StmtOp<"stmt.print"> {
  let summary = "Print statement";
  let description = [{
    Print statement.
    Attributes: newline (BoolAttr) and debug (BoolAttr).
    Takes variadic expression inputs.
    Corresponds to S_Print in ASL.
  }];

  let arguments = (ins Variadic<AnyType>:$args, BoolAttr:$newline,
      BoolAttr:$debug);
  let assemblyFormat = "$args `newline` $newline `debug` $debug attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_StmtUnreachableOp : ASL_StmtOp<"stmt.unreachable"> {
  let summary = "Unreachable statement";
  let description = [{
    Marks unreachable code.
    Corresponds to S_Unreachable in ASL.
  }];

  let assemblyFormat = "attr-dict";
}

def ASL_StmtPragmaOp : ASL_StmtOp<"stmt.pragma"> {
  let summary = "Tool-specific pragma statement";
  let description = [{
    Tool-specific pragma.
    Has identifier (StringAttr) attribute and takes variadic expression inputs.
    Corresponds to S_Pragma in ASL.
  }];

  let arguments = (ins StrAttr:$identifier, Variadic<AnyType>:$args);
  let assemblyFormat = "$identifier `(` $args `)` attr-dict `:` "
                       "functional-type(operands, results)";
}

#endif // ASL_STATEMENT_TD
