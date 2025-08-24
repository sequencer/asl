//===- ASLExpressions.td - ASL expression operations -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_EXPRESSIONS_TD
#define ASL_EXPRESSIONS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "ASL/ASLDialect.td"
include "ASL/ASLTypes.td"
include "ASL/ASLAttributes.td"

//===----------------------------------------------------------------------===//
// ASL Expression Operations Base Classes
//===----------------------------------------------------------------------===//

class ASL_ExprOp<string mnemonic, list<Trait> traits = []>
    : ASL_Op<mnemonic, traits#[Pure]> {}

//===----------------------------------------------------------------------===//
// Utility Operations - Slice Operations
//===----------------------------------------------------------------------===//

def ASL_SliceSingleOp : ASL_ExprOp<"slice.single"> {
  let summary = "Single element slice operation";
  let description = [{
    Takes an index input i, returns a slice of length 1 at position i.
    Corresponds to ASL Slice_Single constructor.
  }];

  let arguments = (ins ASL_IntType:$index);
  let results = (outs ASL_SliceType:$result);
  let assemblyFormat =
      "$index attr-dict `:` functional-type(operands, results)";
}

def ASL_SliceRangeOp : ASL_ExprOp<"slice.range"> {
  let summary = "Range slice operation";
  let description = [{
    Takes start index j and end index i, returns slice from position j to i-1 (inclusive).
    Corresponds to ASL Slice_Range constructor.
  }];

  let arguments = (ins ASL_IntType:$start, ASL_IntType:$end);
  let results = (outs ASL_SliceType:$result);
  let assemblyFormat =
      "$start `,` $end attr-dict `:` functional-type(operands, results)";
}

def ASL_SliceLengthOp : ASL_ExprOp<"slice.length"> {
  let summary = "Length-based slice operation";
  let description = [{
    Takes start index i and length n, returns slice of length n starting at position i.
    Corresponds to ASL Slice_Length constructor.
  }];

  let arguments = (ins ASL_IntType:$start, ASL_IntType:$length);
  let results = (outs ASL_SliceType:$result);
  let assemblyFormat =
      "$start `,` $length attr-dict `:` functional-type(operands, results)";
}

def ASL_SliceStarOp : ASL_ExprOp<"slice.star"> {
  let summary = "Star slice operation";
  let description = [{
    Takes factor i and length n, returns slice starting at position i * n with length n.
    Corresponds to ASL Slice_Star constructor.
  }];

  let arguments = (ins ASL_IntType:$factor, ASL_IntType:$length);
  let results = (outs ASL_SliceType:$result);
  let assemblyFormat =
      "$factor `,` $length attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Pattern Matching Operations
//===----------------------------------------------------------------------===//

def ASL_PatternAllOp : ASL_ExprOp<"expr.pattern.all"> {
  let summary = "Wildcard pattern that matches any value";
  let description = [{
    Represents a wildcard pattern that matches any value.
    Corresponds to Pattern_All in ASL.
  }];

  let arguments = (ins AnyType:$expr);
  let results = (outs I1:$result);
  let assemblyFormat = "$expr attr-dict `:` functional-type(operands, results)";
}

def ASL_PatternAnyOp : ASL_ExprOp<"expr.pattern.any"> {
  let summary = "Disjunctive pattern matching";
  let description = [{
    Represents a disjunctive pattern that matches if any of the sub-patterns match.
    The sub-patterns are nested in the first region of this operation.
    Corresponds to Pattern_Any in ASL.
  }];

  let arguments = (ins AnyType:$expr);
  let results = (outs I1:$result);
  let regions = (region SizedRegion<1>:$patterns);
  let assemblyFormat =
      "$expr $patterns attr-dict `:` functional-type(operands, results)";
}

def ASL_PatternGeqOp : ASL_ExprOp<"expr.pattern.geq"> {
  let summary = "Greater than or equal pattern";
  let description = [{
    Matches values greater than or equal to the given threshold expression.
    Corresponds to Pattern_Geq in ASL.
  }];

  let arguments = (ins AnyType:$expr, AnyType:$threshold);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$expr `,` $threshold attr-dict `:` functional-type(operands, results)";
}

def ASL_PatternLeqOp : ASL_ExprOp<"expr.pattern.leq"> {
  let summary = "Less than or equal pattern";
  let description = [{
    Matches values less than or equal to the given threshold expression.
    Corresponds to Pattern_Leq in ASL.
  }];

  let arguments = (ins AnyType:$expr, AnyType:$threshold);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$expr `,` $threshold attr-dict `:` functional-type(operands, results)";
}

def ASL_PatternMaskOp : ASL_ExprOp<"expr.pattern.mask"> {
  let summary = "Bitvector mask pattern";
  let description = [{
    Matches expressions against a bitvector mask pattern using the mask
    stored as a BitVectorMaskAttr attribute.
    Corresponds to Pattern_Mask in ASL.
  }];

  let arguments = (ins ASL_BitsType:$expr, ASL_BitVectorMaskAttr:$mask);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$expr `,` $mask attr-dict `:` functional-type(operands, results)";
}

def ASL_PatternNotOp : ASL_ExprOp<"expr.pattern.not"> {
  let summary = "Negated pattern";
  let description = [{
    Represents the negation of another pattern. The pattern to be inverted
    is nested in the first region of this operation.
    Corresponds to Pattern_Not in ASL.
  }];

  let arguments = (ins AnyType:$expr);
  let results = (outs I1:$result);
  let regions = (region SizedRegion<1>:$pattern);
  let assemblyFormat =
      "$expr $pattern attr-dict `:` functional-type(operands, results)";
}

def ASL_PatternRangeOp : ASL_ExprOp<"expr.pattern.range"> {
  let summary = "Range pattern";
  let description = [{
    Matches values within an inclusive range from lower bound to upper bound.
    Corresponds to Pattern_Range in ASL.
  }];

  let arguments = (ins AnyType:$expr, AnyType:$lower, AnyType:$upper);
  let results = (outs I1:$result);
  let assemblyFormat = "$expr `,` $lower `,` $upper attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_PatternSingleOp : ASL_ExprOp<"expr.pattern.single"> {
  let summary = "Single value pattern";
  let description = [{
    Matches a single specific value using equality comparison.
    Corresponds to Pattern_Single in ASL.
  }];

  let arguments = (ins AnyType:$expr, AnyType:$value);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$expr `,` $value attr-dict `:` functional-type(operands, results)";
}

def ASL_PatternTupleOp : ASL_ExprOp<"expr.pattern.tuple"> {
  let summary = "Tuple pattern";
  let description = [{
    Matches tuple expressions against multiple sub-patterns.
    Sub-patterns are nested in the first region of this operation.
    Corresponds to Pattern_Tuple in ASL.
  }];

  let arguments = (ins AnyType:$expr);
  let results = (outs I1:$result);
  let regions = (region SizedRegion<1>:$patterns);
  let assemblyFormat =
      "$expr $patterns attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Literal Expression Operations
//===----------------------------------------------------------------------===//

def ASL_LiteralIntOp : ASL_ExprOp<"expr.literal.int"> {
  let summary = "Integer literal";
  let description = [{
    Represents a compile-time integer constant. The value is stored as a
    StringAttr using Z.to_string serialization.
    Corresponds to E_Literal(L_Int) in ASL.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs ASL_IntType:$result);
  let assemblyFormat = "$value attr-dict `:` type(results)";
}

def ASL_LiteralBoolOp : ASL_ExprOp<"expr.literal.bool"> {
  let summary = "Boolean literal";
  let description = [{
    Represents a compile-time boolean constant.
    Corresponds to E_Literal(L_Bool) in ASL.
  }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs I1:$result);
  let assemblyFormat = "$value attr-dict `:` type(results)";
}

def ASL_LiteralRealOp : ASL_ExprOp<"expr.literal.real"> {
  let summary = "Real literal";
  let description = [{
    Represents a compile-time real number constant. The value is stored as a
    StringAttr using Q.to_string serialization.
    Corresponds to E_Literal(L_Real) in ASL.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs ASL_RealType:$result);
  let assemblyFormat = "$value attr-dict `:` type(results)";
}

def ASL_LiteralBitvectorOp : ASL_ExprOp<"expr.literal.bitvector"> {
  let summary = "Bitvector literal";
  let description = [{
    Represents a compile-time bitvector constant. The value is stored as a
    StringAttr using Bitvector.to_string serialization.
    Corresponds to E_Literal(L_BitVector) in ASL.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs ASL_BitsType:$result);
  let assemblyFormat = "$value attr-dict `:` type(results)";
}

def ASL_LiteralStringOp : ASL_ExprOp<"expr.literal.string"> {
  let summary = "String literal";
  let description = [{
    Represents a compile-time string constant.
    Corresponds to E_Literal(L_String) in ASL.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs ASL_StringType:$result);
  let assemblyFormat = "$value attr-dict `:` type(results)";
}

def ASL_LiteralLabelOp : ASL_ExprOp<"expr.literal.label"> {
  let summary = "Label literal";
  let description = [{
    Represents a compile-time label constant.
    Corresponds to E_Literal(L_Label) in ASL.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs ASL_LabelType:$result);
  let assemblyFormat = "$value attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// Variable Reference Operations
//===----------------------------------------------------------------------===//

def ASL_VarOp : ASL_ExprOp<"expr.var"> {
  let summary = "Variable reference";
  let description = [{
    Represents variable references. The operation resolves variable names
    to SSA values through local symbol table lookup during parsing.
    Corresponds to E_Var in ASL.
  }];

  let arguments = (ins StrAttr:$name);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$name attr-dict `:` type(results)";
}

//===----------------------------------------------------------------------===//
// Asserted Type Conversion Operations
//===----------------------------------------------------------------------===//

def ASL_AtcOp : ASL_ExprOp<"expr.atc"> {
  let summary = "Asserted Type Conversion";
  let description = [{
    Implements Asserted Type Conversion as specified in ASL Specification TypingRule.ATC.
    This operation is used for compile-time type materialization and constraint checking.
    Corresponds to E_ATC in ASL.
  }];

  let arguments = (ins AnyType:$expr, TypeAttr:$target_type);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$expr `as` $target_type attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_AtcIntOp : ASL_ExprOp<"expr.atc.int"> {
  let summary = "Integer Type Conversion";
  let description = [{
    Integer type conversion that accepts constraint operations.
    Used for constraining integer values.
  }];

  let arguments = (ins AnyType:$expr);
  let results = (outs ASL_IntType:$result);
  let regions = (region SizedRegion<1>:$constraints);
  let assemblyFormat =
      "$expr $constraints attr-dict `:` functional-type(operands, results)";
}

def ASL_AtcIntExactOp : ASL_ExprOp<"expr.atc.int.exact"> {
  let summary = "Exact integer constraint";
  let description = [{
    Constrains the value to exactly match a specified expression.
    For compile-time constants, the exact value can be embedded in the type.
    For runtime expressions, the exact constraint is provided as an operand.
  }];

  let arguments = (ins AnyType:$value, Optional<ASL_IntType>:$exact);
  let results = (outs ASL_IntType:$result);
  let assemblyFormat = [{ 
    $value (`,` $exact^)? attr-dict `:` functional-type(operands, results)
  }];
}

def ASL_AtcIntRangeOp : ASL_ExprOp<"expr.atc.int.range"> {
  let summary = "Range integer constraint";
  let description = [{
    Constrains the value to lie within an inclusive range defined by lhs and rhs bounds.
  }];

  let arguments = (ins AnyType:$expr, ASL_IntType:$lhs, ASL_IntType:$rhs);
  let results = (outs ASL_IntType:$result);
  let assemblyFormat = "$expr `,` $lhs `,` $rhs attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_AtcBitsOp : ASL_ExprOp<"expr.atc.bits"> {
  let summary = "Bitvector Type Conversion";
  let description = [{
    Bitvector type conversion with width and bitfield specifications.
  }];

  let arguments = (ins AnyType:$expr, ASL_IntType:$width, ArrayAttr:$bitfields);
  let results = (outs ASL_BitsType:$result);
  let assemblyFormat = "$expr `,` $width `,` $bitfields attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_AtcBitsBitfieldsSimpleOp
    : ASL_ExprOp<"expr.atc.bits.bitfields.simple"> {
  let summary = "Simple bitfield for bitvector ATC";
  let description = [{
    Simple bitfield that accepts slice operations.
  }];

  let arguments = (ins Variadic<ASL_SliceType>:$slices);
  let results = (outs ASL_BitsType:$result);
  let assemblyFormat =
      "$slices attr-dict `:` functional-type(operands, results)";
}

def ASL_AtcBitsBitfieldsNestedOp
    : ASL_ExprOp<"expr.atc.bits.bitfields.nested"> {
  let summary = "Nested bitfield for bitvector ATC";
  let description = [{
    Nested bitfield that accepts slices and nested bitfields.
  }];

  let arguments = (ins Variadic<ASL_SliceType>:$slices);
  let results = (outs ASL_BitsType:$result);
  let regions = (region SizedRegion<1>:$nested_bitfields);
  let assemblyFormat = "$slices $nested_bitfields attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_AtcBitsBitfieldsTypeOp : ASL_ExprOp<"expr.atc.bits.bitfields.type"> {
  let summary = "Typed bitfield for bitvector ATC";
  let description = [{
    Typed bitfield that accepts slices with explicit TypeAttr.
  }];

  let arguments = (ins Variadic<ASL_SliceType>:$slices, TypeAttr:$type);
  let results = (outs ASL_BitsType:$result);
  let assemblyFormat =
      "$slices `,` $type attr-dict `:` functional-type(operands, results)";
}

def ASL_AtcArrayOp : ASL_ExprOp<"expr.atc.array"> {
  let summary = "Array Type Conversion";
  let description = [{
    Array type conversion with length specification.
  }];

  let arguments = (ins AnyType:$expr, ASL_IntType:$length);
  let results = (outs ASL_ArrayType:$result);
  let assemblyFormat =
      "$expr `,` $length attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Binary Operator Expression Operations
//===----------------------------------------------------------------------===//

def ASL_BinopAndOp : ASL_ExprOp<"expr.binop.and"> {
  let summary = "Bitvector bitwise AND";
  let description =
      [{ Bitvector bitwise AND operation. Corresponds to AND in ASL. }];
  let arguments = (ins ASL_BitsType:$lhs, ASL_BitsType:$rhs);
  let results = (outs ASL_BitsType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopBandOp : ASL_ExprOp<"expr.binop.band"> {
  let summary = "Boolean AND";
  let description = [{ Boolean AND operation. Corresponds to BAND in ASL. }];
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopBeqOp : ASL_ExprOp<"expr.binop.beq"> {
  let summary = "Boolean equivalence";
  let description =
      [{ Boolean equivalence operation. Corresponds to BEQ in ASL. }];
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopBorOp : ASL_ExprOp<"expr.binop.bor"> {
  let summary = "Boolean OR";
  let description = [{ Boolean OR operation. Corresponds to BOR in ASL. }];
  let arguments = (ins I1:$lhs, I1:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopDivOp : ASL_ExprOp<"expr.binop.div"> {
  let summary = "Integer division";
  let description =
      [{ Integer division operation. Corresponds to DIV in ASL. }];
  let arguments = (ins ASL_IntType:$lhs, ASL_IntType:$rhs);
  let results = (outs ASL_IntType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopDivrmOp : ASL_ExprOp<"expr.binop.divrm"> {
  let summary = "Integer division with rounding towards negative infinity";
  let description =
      [{ Integer division with rounding towards negative infinity. Corresponds to DIVRM in ASL. }];
  let arguments = (ins ASL_IntType:$lhs, ASL_IntType:$rhs);
  let results = (outs ASL_IntType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopXorOp : ASL_ExprOp<"expr.binop.xor"> {
  let summary = "Bitvector bitwise exclusive OR";
  let description =
      [{ Bitvector bitwise exclusive OR operation. Corresponds to XOR in ASL. }];
  let arguments = (ins ASL_BitsType:$lhs, ASL_BitsType:$rhs);
  let results = (outs ASL_BitsType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopEqOp : ASL_ExprOp<"expr.binop.eq"> {
  let summary = "Equality comparison";
  let description =
      [{ Equality comparison operation. Corresponds to EQ_OP in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopGtOp : ASL_ExprOp<"expr.binop.gt"> {
  let summary = "Greater than for integers or reals";
  let description =
      [{ Greater than comparison for integers or reals. Corresponds to GT in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopGeqOp : ASL_ExprOp<"expr.binop.geq"> {
  let summary = "Greater than or equal for integers or reals";
  let description =
      [{ Greater than or equal comparison for integers or reals. Corresponds to GEQ in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopImplOp : ASL_ExprOp<"expr.binop.impl"> {
  let summary = "Boolean implication";
  let description =
      [{ Boolean implication operation. Corresponds to IMPL in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopLtOp : ASL_ExprOp<"expr.binop.lt"> {
  let summary = "Less than for integers or reals";
  let description =
      [{ Less than comparison for integers or reals. Corresponds to LT in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopLeqOp : ASL_ExprOp<"expr.binop.leq"> {
  let summary = "Less than or equal for integers or reals";
  let description =
      [{ Less than or equal comparison for integers or reals. Corresponds to LEQ in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopModOp : ASL_ExprOp<"expr.binop.mod"> {
  let summary = "Remainder of integer division";
  let description =
      [{ Remainder of integer division operation. Corresponds to MOD in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs ASL_IntType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopMinusOp : ASL_ExprOp<"expr.binop.minus"> {
  let summary = "Subtraction for integers, reals, or bitvectors";
  let description =
      [{ Subtraction operation for integers, reals, or bitvectors. Corresponds to MINUS in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopMulOp : ASL_ExprOp<"expr.binop.mul"> {
  let summary = "Multiplication for integers, reals, or bitvectors";
  let description =
      [{ Multiplication operation for integers, reals, or bitvectors. Corresponds to MUL in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopNeqOp : ASL_ExprOp<"expr.binop.neq"> {
  let summary = "Non-equality comparison";
  let description =
      [{ Non-equality comparison operation. Corresponds to NEQ in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopOrOp : ASL_ExprOp<"expr.binop.or"> {
  let summary = "Bitvector bitwise OR";
  let description =
      [{ Bitvector bitwise OR operation. Corresponds to OR in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs ASL_BitsType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopPlusOp : ASL_ExprOp<"expr.binop.plus"> {
  let summary = "Addition for integers, reals, or bitvectors";
  let description =
      [{ Addition operation for integers, reals, or bitvectors. Corresponds to PLUS in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopPowOp : ASL_ExprOp<"expr.binop.pow"> {
  let summary = "Exponentiation for integers";
  let description =
      [{ Exponentiation operation for integers. Corresponds to POW in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs ASL_IntType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopRdivOp : ASL_ExprOp<"expr.binop.rdiv"> {
  let summary = "Division for reals";
  let description =
      [{ Division operation for reals. Corresponds to RDIV in ASL. }];
  let arguments = (ins ASL_RealType:$lhs, ASL_RealType:$rhs);
  let results = (outs ASL_RealType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopShlOp : ASL_ExprOp<"expr.binop.shl"> {
  let summary = "Shift left for integers";
  let description =
      [{ Shift left operation for integers. Corresponds to SHL in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs ASL_IntType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopShrOp : ASL_ExprOp<"expr.binop.shr"> {
  let summary = "Shift right for integers";
  let description =
      [{ Shift right operation for integers. Corresponds to SHR in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs ASL_IntType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def ASL_BinopConcatOp : ASL_ExprOp<"expr.binop.concat"> {
  let summary = "Bitvector or string concatenation";
  let description =
      [{ Bitvector or string concatenation operation. Corresponds to CONCAT in ASL. }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Unary Operator Expression Operations
//===----------------------------------------------------------------------===//

def ASL_UnopBnotOp : ASL_ExprOp<"expr.unop.bnot"> {
  let summary = "Boolean inversion";
  let description =
      [{ Boolean inversion operation. Corresponds to BNOT in ASL. }];
  let arguments = (ins I1:$operand);
  let results = (outs I1:$result);
  let assemblyFormat =
      "$operand attr-dict `:` functional-type(operands, results)";
}

def ASL_UnopNegOp : ASL_ExprOp<"expr.unop.neg"> {
  let summary = "Integer or real negation";
  let description =
      [{ Integer or real negation operation. Corresponds to NEG in ASL. }];
  let arguments = (ins AnyType:$operand);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$operand attr-dict `:` functional-type(operands, results)";
}

def ASL_UnopNotOp : ASL_ExprOp<"expr.unop.not"> {
  let summary = "Bitvector bitwise inversion";
  let description =
      [{ Bitvector bitwise inversion operation. Corresponds to NOT in ASL. }];
  let arguments = (ins ASL_BitsType:$operand);
  let results = (outs ASL_BitsType:$result);
  let assemblyFormat =
      "$operand attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Function Call and Control Flow Operations
//===----------------------------------------------------------------------===//

def ASL_CallOp : ASL_ExprOp<"expr.call"> {
  let summary = "Subprogram invocation";
  let description = [{
    Represents a subprogram invocation with return value.
    Corresponds to E_Call in ASL.
  }];

  let arguments = (ins StrAttr:$name, Variadic<AnyType>:$args,
      I32Attr:$params_size, ASL_SubprogramTypeAttr:$call_type);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$name `(` $args `)` `,` $params_size `,` $call_type "
                       "attr-dict `:` functional-type(operands, results)";
}

def ASL_SliceOp : ASL_ExprOp<"expr.slice"> {
  let summary = "Slicing operations";
  let description = [{
    Represents slicing operations on bitvectors and integers.
    Corresponds to E_Slice in ASL.
  }];

  let arguments = (ins AnyType:$base, Variadic<ASL_SliceType>:$slices);
  let results = (outs ASL_BitsType:$result);
  let assemblyFormat =
      "$base `[` $slices `]` attr-dict `:` functional-type(operands, results)";
}

def ASL_CondOp : ASL_ExprOp<"expr.cond"> {
  let summary = "Conditional expressions (ternary operator)";
  let description = [{
    Represents conditional expressions (ternary operator).
    Corresponds to E_Cond in ASL.
  }];

  let arguments = (ins I1:$condition, AnyType:$then_expr, AnyType:$else_expr);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$condition `?` $then_expr `:` $else_expr attr-dict `:` "
                       "functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Data Access Operations
//===----------------------------------------------------------------------===//

def ASL_GetArrayOp : ASL_ExprOp<"expr.get_array"> {
  let summary = "Array access with integer index";
  let description = [{
    Represents array access with integer index.
    Corresponds to E_GetArray in ASL.
  }];

  let arguments = (ins ASL_ArrayType:$base, ASL_IntType:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$base `[` $index `]` attr-dict `:` functional-type(operands, results)";
}

def ASL_GetEnumArrayOp : ASL_ExprOp<"expr.get_enum_array"> {
  let summary = "Array access with enumeration index";
  let description = [{
    Represents array access with enumeration index.
    Corresponds to E_GetEnumArray in ASL.
  }];

  let arguments = (ins ASL_ArrayType:$base, ASL_LabelType:$key);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$base `[` $key `]` attr-dict `:` functional-type(operands, results)";
}

def ASL_GetFieldOp : ASL_ExprOp<"expr.get_field"> {
  let summary = "Record field access";
  let description = [{
    Represents record field access.
    Corresponds to E_GetField in ASL.
  }];

  let arguments = (ins ASL_RecordType:$record, StrAttr:$field_name);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$record `,` $field_name attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_GetFieldsOp : ASL_ExprOp<"expr.get_fields"> {
  let summary = "Accessing multiple record fields";
  let description = [{
    Represents accessing multiple record fields for bit-packing.
    Corresponds to E_GetFields in ASL.
  }];

  let arguments = (ins ASL_RecordType:$record, ArrayAttr:$field_names);
  let results = (outs ASL_BitsType:$result);
  let assemblyFormat = "$record `,` $field_names attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_GetCollectionFieldsOp : ASL_ExprOp<"expr.get_collection_fields"> {
  let summary = "Get collection fields";
  let description = [{
    Represents accessing collection fields.
    Corresponds to E_GetCollectionFields in ASL.
  }];

  let arguments = (ins StrAttr:$collection, ArrayAttr:$field_names);
  let results = (outs ASL_BitsType:$result);
  let assemblyFormat = "$collection `,` $field_names attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_GetItemOp : ASL_ExprOp<"expr.get_item"> {
  let summary = "Tuple element access";
  let description = [{
    Represents tuple element access.
    Corresponds to E_GetItem in ASL.
  }];

  let arguments = (ins ASL_TupleType:$tuple, I32Attr:$index);
  let results = (outs AnyType:$result);
  let assemblyFormat =
      "$tuple `[` $index `]` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Construction Operations
//===----------------------------------------------------------------------===//

def ASL_RecordOp : ASL_ExprOp<"expr.record"> {
  let summary = "Record construction";
  let description = [{
    Represents record construction.
    Corresponds to E_Record in ASL.
  }];

  let arguments = (ins TypeAttr:$record_type, Variadic<AnyType>:$field_values,
      ArrayAttr:$field_names);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$record_type `{` $field_names `:` $field_values `}` "
                       "attr-dict `:` functional-type(operands, results)";
}

def ASL_TupleOp : ASL_ExprOp<"expr.tuple"> {
  let summary = "Tuple construction";
  let description = [{
    Represents tuple construction. Takes variadic inputs for tuple elements.
    Corresponds to E_Tuple in ASL.
  }];

  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs ASL_TupleType:$result);
  let assemblyFormat =
      "`(` $elements `)` attr-dict `:` functional-type(operands, results)";
}

def ASL_ArrayOp : ASL_ExprOp<"expr.array"> {
  let summary = "Array construction";
  let description = [{
    Represents array construction. Takes value and length inputs to create
    an array of the specified length filled with the given value.
    Corresponds to E_Array in ASL.
  }];

  let arguments = (ins AnyType:$value, ASL_IntType:$length);
  let results = (outs ASL_ArrayType:$result);
  let assemblyFormat = "`[` $value `,` $length `]` attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_EnumArrayOp : ASL_ExprOp<"expr.enum_array"> {
  let summary = "Enumeration-indexed array construction";
  let description = [{
    Represents enumeration-indexed array construction.
    Corresponds to E_EnumArray in ASL.
  }];

  let arguments = (ins StrAttr:$enum_name, ArrayAttr:$labels, AnyType:$value);
  let results = (outs ASL_ArrayType:$result);
  let assemblyFormat = "$enum_name `[` $labels `]` `=` $value attr-dict `:` "
                       "functional-type(operands, results)";
}

def ASL_ArbitraryOp : ASL_ExprOp<"expr.arbitrary"> {
  let summary = "Arbitrary value";
  let description = [{
    Represents non-deterministic values.
    Corresponds to E_Arbitrary in ASL.
  }];

  let arguments = (ins TypeAttr:$type);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$type attr-dict `:` functional-type(operands, results)";
}

def ASL_PatternOp : ASL_ExprOp<"expr.pattern"> {
  let summary = "Pattern matching expression";
  let description = [{
    Represents pattern matching operations.
    Corresponds to E_Pattern in ASL.
  }];

  let arguments = (ins AnyType:$expr);
  let results = (outs I1:$result);
  let regions = (region SizedRegion<1>:$pattern);
  let assemblyFormat =
      "$expr `match` $pattern attr-dict `:` functional-type(operands, results)";
}

#endif // ASL_EXPRESSIONS_TD