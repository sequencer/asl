//===- ASLOperations.td - ASL dialect ops ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_OPERATIONS
#define ASL_OPERATIONS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

include "ASLDialect.td"
include "ASLTypes.td"
include "ASLAttributes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

def ASL_BnotOp : ASL_Op<"bnot", [Pure, SameOperandsAndResultType]> {
  let summary = "ASL boolean NOT operation";
  let description = [{
    The `asl.bnot` operation performs boolean inversion.
    
    Example:
    ```mlir
    %result = asl.bnot %operand : !asl.bool
    ```
  }];
  
  let arguments = (ins ASL_BoolType:$operand);
  let results = (outs ASL_BoolType:$result);
}


def ASL_NegOp : ASL_Op<"neg", [Pure, SameOperandsAndResultType]> {
  let summary = "ASL negation operation";
  let description = [{
    The `asl.neg` operation performs integer or real negation.
    
    Example:
    ```mlir
    %result = asl.neg %operand : !asl.int
    ```
  }];
  
  let arguments = (ins ASLNumericType:$operand);
  let results = (outs ASLNumericType:$result);
}


def ASL_NotOp : ASL_Op<"not", [Pure, SameOperandsAndResultType]> {
  let summary = "ASL bitwise NOT operation";
  let description = [{
    The `asl.not` operation performs bitvector bitwise inversion.
    
    Example:
    ```mlir
    %result = asl.not %operand : !asl.bits<32>
    ```
  }];
  
  let arguments = (ins ASL_BitsType:$operand);
  let results = (outs ASL_BitsType:$result);
}


def ASL_AndOp : ASL_Op<"and", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "ASL bitwise AND operation";
  let description = [{
    The `asl.and` operation performs bitvector bitwise AND.
    
    Example:
    ```mlir
    %result = asl.and %lhs, %rhs : !asl.bits<32>
    ```
  }];
  
  let arguments = (ins ASL_BitsType:$lhs, ASL_BitsType:$rhs);
  let results = (outs ASL_BitsType:$result);
}


def ASL_BandOp : ASL_Op<"band", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "ASL boolean AND operation";
  let description = [{
    The `asl.band` operation performs boolean AND.
    
    Example:
    ```mlir
    %result = asl.band %lhs, %rhs : !asl.bool
    ```
  }];
  
  let arguments = (ins ASL_BoolType:$lhs, ASL_BoolType:$rhs);
  let results = (outs ASL_BoolType:$result);
}


def ASL_BeqOp : ASL_Op<"beq", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "ASL boolean equivalence operation";
  let description = [{
    The `asl.beq` operation performs boolean equivalence.
    
    Example:
    ```mlir
    %result = asl.beq %lhs, %rhs : !asl.bool
    ```
  }];
  
  let arguments = (ins ASL_BoolType:$lhs, ASL_BoolType:$rhs);
  let results = (outs ASL_BoolType:$result);
}


def ASL_BorOp : ASL_Op<"bor", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "ASL boolean OR operation";
  let description = [{
    The `asl.bor` operation performs boolean OR.
    
    Example:
    ```mlir
    %result = asl.bor %lhs, %rhs : !asl.bool
    ```
  }];
  
  let arguments = (ins ASL_BoolType:$lhs, ASL_BoolType:$rhs);
  let results = (outs ASL_BoolType:$result);
}


def ASL_DivOp : ASL_Op<"div", [Pure]> {
  let summary = "ASL integer division operation";
  let description = [{
    The `asl.div` operation performs integer division.
    
    Example:
    ```mlir
    %result = asl.div %lhs, %rhs : !asl.int
    ```
  }];
  
  let arguments = (ins ASL_IntType:$lhs, ASL_IntType:$rhs);
  let results = (outs ASL_IntType:$result);
}


def ASL_DivrmOp : ASL_Op<"divrm", [Pure]> {
  let summary = "ASL inexact integer division with rounding";
  let description = [{
    The `asl.divrm` operation performs inexact integer division, rounding towards negative infinity.
    
    Example:
    ```mlir
    %result = asl.divrm %lhs, %rhs : !asl.int
    ```
  }];
  
  let arguments = (ins ASL_IntType:$lhs, ASL_IntType:$rhs);
  let results = (outs ASL_IntType:$result);
}


def ASL_XorOp : ASL_Op<"xor", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "ASL bitwise XOR operation";
  let description = [{
    The `asl.xor` operation performs bitvector bitwise exclusive OR.
    
    Example:
    ```mlir
    %result = asl.xor %lhs, %rhs : !asl.bits<32>
    ```
  }];
  
  let arguments = (ins ASL_BitsType:$lhs, ASL_BitsType:$rhs);
  let results = (outs ASL_BitsType:$result);
}


def ASL_EqOp : ASL_Op<"eq", [Pure, Commutative]> {
  let summary = "ASL equality operation";
  let description = [{
    The `asl.eq` operation performs equality comparison on two base values.
    
    Example:
    ```mlir
    %result = asl.eq %lhs, %rhs : !asl.bool
    ```
  }];
  
  let arguments = (ins ASLType:$lhs, ASLType:$rhs);
  let results = (outs ASL_BoolType:$result);
}


def ASL_GtOp : ASL_Op<"gt", [Pure]> {
  let summary = "ASL greater than operation";
  let description = [{
    The `asl.gt` operation performs greater than comparison for integers or reals.
    
    Example:
    ```mlir
    %result = asl.gt %lhs, %rhs : !asl.int
    ```
  }];
  
  let arguments = (ins ASLNumericType:$lhs, ASLNumericType:$rhs);
  let results = (outs ASL_BoolType:$result);
}

def ASL_GeqOp : ASL_Op<"geq", [Pure]> {
  let summary = "ASL greater than or equal operation";
  let description = [{
    The `asl.geq` operation performs greater than or equal comparison for integers or reals.
    
    Example:
    ```mlir
    %result = asl.geq %lhs, %rhs : !asl.int
    ```
  }];
  
  let arguments = (ins ASLNumericType:$lhs, ASLNumericType:$rhs);
  let results = (outs ASL_BoolType:$result);
}

def ASL_ImplOp : ASL_Op<"impl", [Pure, SameOperandsAndResultType]> {
  let summary = "ASL boolean implication operation";
  let description = [{
    The `asl.impl` operation performs boolean implication.
    
    Example:
    ```mlir
    %result = asl.impl %lhs, %rhs : !asl.bool
    ```
  }];
  
  let arguments = (ins ASL_BoolType:$lhs, ASL_BoolType:$rhs);
  let results = (outs ASL_BoolType:$result);
}

def ASL_LtOp : ASL_Op<"lt", [Pure]> {
  let summary = "ASL less than operation";
  let description = [{
    The `asl.lt` operation performs less than comparison for integers or reals.
    
    Example:
    ```mlir
    %result = asl.lt %lhs, %rhs : !asl.int
    ```
  }];
  
  let arguments = (ins ASLNumericType:$lhs, ASLNumericType:$rhs);
  let results = (outs ASL_BoolType:$result);
}

def ASL_LeqOp : ASL_Op<"leq", [Pure]> {
  let summary = "ASL less than or equal operation";
  let description = [{
    The `asl.leq` operation performs less than or equal comparison for integers or reals.
    
    Example:
    ```mlir
    %result = asl.leq %lhs, %rhs : !asl.int
    ```
  }];
  
  let arguments = (ins ASLNumericType:$lhs, ASLNumericType:$rhs);
  let results = (outs ASL_BoolType:$result);
}

def ASL_ModOp : ASL_Op<"mod", [Pure]> {
  let summary = "ASL modulo operation";
  let description = [{
    The `asl.mod` operation performs remainder of integer division.
    
    Example:
    ```mlir
    %result = asl.mod %lhs, %rhs : !asl.int
    ```
  }];
  
  let arguments = (ins ASL_IntType:$lhs, ASL_IntType:$rhs);
  let results = (outs ASL_IntType:$result);
}

def ASL_MinusOp : ASL_Op<"minus", [Pure, SameOperandsAndResultType]> {
  let summary = "ASL subtraction operation";
  let description = [{
    The `asl.minus` operation performs subtraction for integers, reals, or bitvectors.
    
    Example:
    ```mlir
    %result = asl.minus %lhs, %rhs : !asl.int
    ```
  }];
  
  let arguments = (ins ASLType:$lhs, ASLType:$rhs);
  let results = (outs ASLType:$result);
}

def ASL_MulOp : ASL_Op<"mul", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "ASL multiplication operation";
  let description = [{
    The `asl.mul` operation performs multiplication for integers, reals, or bitvectors.
    
    Example:
    ```mlir
    %result = asl.mul %lhs, %rhs : !asl.int
    ```
  }];
  
  let arguments = (ins ASLType:$lhs, ASLType:$rhs);
  let results = (outs ASLType:$result);
}

def ASL_NeqOp : ASL_Op<"neq", [Pure, Commutative]> {
  let summary = "ASL inequality operation";
  let description = [{
    The `asl.neq` operation performs non-equality comparison on two base values.
    
    Example:
    ```mlir
    %result = asl.neq %lhs, %rhs : !asl.bool
    ```
  }];
  
  let arguments = (ins ASLType:$lhs, ASLType:$rhs);
  let results = (outs ASL_BoolType:$result);
}

def ASL_OrOp : ASL_Op<"or", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "ASL bitwise OR operation";
  let description = [{
    The `asl.or` operation performs bitvector bitwise OR.
    
    Example:
    ```mlir
    %result = asl.or %lhs, %rhs : !asl.bits<32>
    ```
  }];
  
  let arguments = (ins ASL_BitsType:$lhs, ASL_BitsType:$rhs);
  let results = (outs ASL_BitsType:$result);
}

def ASL_PlusOp : ASL_Op<"plus", [Pure, Commutative, SameOperandsAndResultType]> {
  let summary = "ASL addition operation";
  let description = [{
    The `asl.plus` operation performs addition for integers, reals, or bitvectors.
    
    Example:
    ```mlir
    %result = asl.plus %lhs, %rhs : !asl.int
    ```
  }];
  
  let arguments = (ins ASLType:$lhs, ASLType:$rhs);
  let results = (outs ASLType:$result);
}

def ASL_PowOp : ASL_Op<"pow", [Pure]> {
  let summary = "ASL exponentiation operation";
  let description = [{
    The `asl.pow` operation performs exponentiation for integers.
    
    Example:
    ```mlir
    %result = asl.pow %base, %exp : !asl.int
    ```
  }];
  
  let arguments = (ins ASL_IntType:$base, ASL_IntType:$exp);
  let results = (outs ASL_IntType:$result);
}

def ASL_RdivOp : ASL_Op<"rdiv", [Pure]> {
  let summary = "ASL real division operation";
  let description = [{
    The `asl.rdiv` operation performs division for real numbers.
    
    Example:
    ```mlir
    %result = asl.rdiv %lhs, %rhs : !asl.real
    ```
  }];
  
  let arguments = (ins ASL_RealType:$lhs, ASL_RealType:$rhs);
  let results = (outs ASL_RealType:$result);
}


def ASL_ShlOp : ASL_Op<"shl", [Pure]> {
  let summary = "ASL shift left operation";
  let description = [{
    The `asl.shl` operation performs shift left for integers.
    
    Example:
    ```mlir
    %result = asl.shl %value, %amount : !asl.int
    ```
  }];
  
  let arguments = (ins ASL_IntType:$value, ASL_IntType:$amount);
  let results = (outs ASL_IntType:$result);
}


def ASL_ShrOp : ASL_Op<"shr", [Pure]> {
  let summary = "ASL shift right operation";
  let description = [{
    The `asl.shr` operation performs shift right for integers.
    
    Example:
    ```mlir
    %result = asl.shr %value, %amount : !asl.int
    ```
  }];
  
  let arguments = (ins ASL_IntType:$value, ASL_IntType:$amount);
  let results = (outs ASL_IntType:$result);
}

def ASL_ConcatOp : ASL_Op<"concat", [Pure, SameOperandsAndResultType]> {
  let summary = "ASL concatenation operation";
  let description = [{
    The `asl.concat` operation performs bitvector or string concatenation.
    
    Example:
    ```mlir
    %result = asl.concat %lhs, %rhs : !asl.bits<64>
    ```
  }];
  
  let arguments = (ins ASLType:$lhs, ASLType:$rhs);
  let results = (outs ASLType:$result);
}


#endif // ASL_OPERATIONS
