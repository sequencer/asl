//===- ASLExpressions.td - ASL dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_EXPRESSIONS
#define ASL_EXPRESSIONS

include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"

def ASL_LiteralOp : ASL_Op<"literal", [Pure, ConstantLike]> {
  let summary = "ASL literal constant";
  let description = [{
    The `asl.literal` operation represents literal values in ASL, including:
    - Integer literals: `asl.literal 42 : !asl.int`
    - Boolean literals: `asl.literal true : !asl.bool`
    - Real literals: `asl.literal "1/3" : !asl.real`
    - Bitvector literals: `asl.literal 0xFF : !asl.bits<8>`
    - String literals: `asl.literal "hello" : !asl.string`
    - Enum label literals: `asl.literal "RED" : !asl.enum<["RED", "GREEN", "BLUE"]>`
  }];
  
  let arguments = (ins AnyAttr:$value);
  let results = (outs ASLType:$result);
}


def ASL_VarOp : ASL_Op<"var", [Pure]> {
  let summary = "ASL variable reference";
  let description = [{
    The `asl.var` operation represents a reference to a variable by name.
    
    Example:
    ```mlir
    %0 = asl.var "variable_name" : !asl.int
    ```
  }];
  
  let arguments = (ins StrAttr:$name);
  let results = (outs ASLType:$result);
}

def ASL_AtcOp : ASL_Op<"atc", [Pure]> {
  let summary = "ASL asserted type conversion expression";
  let description = [{
    The `asl.atc` operation performs asserted type conversion as an expression,
    converting a value from one ASL type to another with runtime assertion.
    
    Example:
    ```mlir
    %result = asl.atc %value : !asl.int -> !asl.int<#asl.constraints<[#asl.constraint_range<0, 100>]>>
    ```
  }];
  
  let arguments = (ins ASLType:$input, TypeAttr:$targetType);
  let results = (outs ASLType:$result);
}

def ASL_CallOp : ASL_Op<"call", []> {
  let summary = "ASL procedure call statement";
  let description = [{
    The `asl.call` operation represents a procedure call without return value.
    
    Example:
    ```mlir
    asl.call @procedure(%arg1, %arg2) {call_type = "ST_Procedure"}
    ```
  }];
  
  let arguments = (ins 
    FlatSymbolRefAttr:$callee,
    Variadic<ASLType>:$args,
    OptionalAttr<ArrayAttr>:$params,
    ASL_SubprogramTypeAttr:$callType
  );
  let results = (outs);
}

def ASL_SliceOp : ASL_Op<"slice", [Pure]> {
  let summary = "ASL general slice operation";
  let description = [{
    The `asl.slice` operation performs slicing with multiple slice specifications.
    
    Example:
    ```mlir
    %result = asl.slice %base, [%slice1, %slice2] : !asl.bits<16>
    ```
  }];
  
  let arguments = (ins ASLIndexableType:$base, Variadic<ASLType>:$slices);
  let results = (outs ASLType:$result);
}

def ASL_SliceSingleOp : ASL_Op<"slice.single", [Pure]> {
  let summary = "ASL single bit slice operation";
  let description = [{
    The `asl.slice.single` operation takes the base expression and an index,
    returning a slice of length 1 at that position.
    
    Example:
    ```mlir
    %result = asl.slice.single %base, %index : !asl.bits<1>
    ```
  }];
  
  let arguments = (ins ASLIndexableType:$base, ASL_IntType:$index);
  let results = (outs ASLType:$result);
}

def ASL_SliceRangeOp : ASL_Op<"slice.range", [Pure]> {
  let summary = "ASL range slice operation";
  let description = [{
    The `asl.slice.range` operation takes the base expression, a start index i,
    and an end index j, returning the slice from i to j-1.
    
    Example:
    ```mlir
    %result = asl.slice.range %base, %start, %end : !asl.bits<8>
    ```
  }];
  
  let arguments = (ins ASLIndexableType:$base, ASL_IntType:$start, ASL_IntType:$end);
  let results = (outs ASLType:$result);
}

def ASL_SliceLengthOp : ASL_Op<"slice.length", [Pure]> {
  let summary = "ASL length slice operation";
  let description = [{
    The `asl.slice.length` operation takes the base expression, a start index i,
    and a length n, returning a slice of length n starting at i.
    
    Example:
    ```mlir
    %result = asl.slice.length %base, %start, %length : !asl.bits<16>
    ```
  }];
  
  let arguments = (ins ASLIndexableType:$base, ASL_IntType:$start, ASL_IntType:$length);
  let results = (outs ASLType:$result);
}

def ASL_SliceStarOp : ASL_Op<"slice.star", [Pure]> {
  let summary = "ASL star slice operation";
  let description = [{
    The `asl.slice.star` operation takes the base expression, a factor, and a length.
    The start index is computed as factor * length.
    
    Example:
    ```mlir
    %result = asl.slice.star %base, %factor, %length : !asl.bits<8>
    ```
  }];
  
  let arguments = (ins ASLIndexableType:$base, ASL_IntType:$factor, ASL_IntType:$length);
  let results = (outs ASLType:$result);
}

def ASL_CondOp : ASL_Op<"cond", [Pure]> {
  let summary = "ASL conditional expression";
  let description = [{
    The `asl.cond` operation represents a conditional expression that
    selects between two values based on a boolean condition.
    
    Example:
    ```mlir
    %result = asl.cond %condition, %true_val, %false_val : !asl.int
    ```
  }];
  
  let arguments = (ins ASL_BoolType:$condition, ASLType:$trueValue, ASLType:$falseValue);
  let results = (outs ASLType:$result);
}

def ASL_GetArrayOp : ASL_Op<"get_array", [Pure]> {
  let summary = "ASL array element access";
  let description = [{
    The `asl.get_array` operation accesses an array element with integer index.
    
    Example:
    ```mlir
    %result = asl.get_array %array, %index : !asl.int
    ```
  }];
  
  let arguments = (ins ASL_ArrayType:$array, ASL_IntType:$index);
  let results = (outs ASLType:$result);
}

def ASL_GetEnumArrayOp : ASL_Op<"get_enum_array", [Pure]> {
  let summary = "ASL enum array element access";
  let description = [{
    The `asl.get_enum_array` operation accesses an array element with enumeration index.
    
    Example:
    ```mlir
    %result = asl.get_enum_array %array, %enum_index : !asl.int
    ```
  }];
  
  let arguments = (ins ASL_ArrayType:$array, ASL_EnumType:$enumIndex);
  let results = (outs ASLType:$result);
}

def ASL_GetFieldOp : ASL_Op<"get_field", [Pure]> {
  let summary = "ASL single field access";
  let description = [{
    The `asl.get_field` operation accesses a single field from a record.
    
    Example:
    ```mlir
    %result = asl.get_field %record, "field_name" : !asl.int
    ```
  }];
  
  let arguments = (ins ASLStructuralType:$record, StrAttr:$fieldName);
  let results = (outs ASLType:$result);
}

def ASL_GetFieldsOp : ASL_Op<"get_fields", [Pure]> {
  let summary = "ASL multiple fields access";
  let description = [{
    The `asl.get_fields` operation accesses multiple fields from a record,
    returning a tuple of field values.
    
    Example:
    ```mlir
    %result = asl.get_fields %record, ["f1", "f2"] : !asl.tuple<!asl.int, !asl.real>
    ```
  }];
  
  let arguments = (ins ASLStructuralType:$record, ArrayAttr:$fieldNames);
  let results = (outs ASL_TupleType:$result);
}

def ASL_GetCollectionFieldsOp : ASL_Op<"get_collection_fields", [Pure]> {
  let summary = "ASL collection fields access";
  let description = [{
    The `asl.get_collection_fields` operation accesses fields from a collection.
    
    Example:
    ```mlir
    %result = asl.get_collection_fields "collection", ["field1"] : !asl.tuple<!asl.int>
    ```
  }];
  
  let arguments = (ins StrAttr:$collectionName, ArrayAttr:$fieldNames);
  let results = (outs ASL_TupleType:$result);
}

def ASL_GetItemOp : ASL_Op<"get_item", [Pure]> {
  let summary = "ASL tuple element access";
  let description = [{
    The `asl.get_item` operation accesses a tuple element by index.
    
    Example:
    ```mlir
    %result = asl.get_item %tuple, 0 : !asl.int
    ```
  }];
  
  let arguments = (ins ASL_TupleType:$tuple, I32Attr:$index);
  let results = (outs ASLType:$result);
}


def ASL_RecordConstructOp : ASL_Op<"record_construct", [Pure]> {
  let summary = "ASL record construction";
  let description = [{
    The `asl.record_construct` operation constructs a record with named fields.
    
    Example:
    ```mlir
    %record = asl.record_construct(%x, %y) : (!asl.real, !asl.real) -> !asl.record<"Point", [#asl.field<"x", !asl.real>, #asl.field<"y", !asl.real>]> {
      field_names = ["x", "y"]
    }
    ```
  }];
  
  let arguments = (ins Variadic<ASLType>:$fieldValues, ArrayAttr:$fieldNames);
  let results = (outs ASL_RecordType:$result);
}


def ASL_TupleConstructOp : ASL_Op<"tuple_construct", [Pure]> {
  let summary = "ASL tuple construction";
  let description = [{
    The `asl.tuple_construct` operation constructs a tuple from elements.
    
    Example:
    ```mlir
    %tuple = asl.tuple_construct %elem1, %elem2 : !asl.tuple<!asl.int, !asl.real>
    ```
  }];
  
  let arguments = (ins Variadic<ASLType>:$elements);
  let results = (outs ASL_TupleType:$result);
}


def ASL_ArrayConstructOp : ASL_Op<"array_construct", [Pure]> {
  let summary = "ASL array construction";
  let description = [{
    The `asl.array_construct` operation constructs an array with specified length and initial value.
    
    Example:
    ```mlir
    %array = asl.array_construct %length, %value : !asl.array<!asl.int, #asl.array_length<10>>
    ```
  }];
  
  let arguments = (ins ASL_IntType:$length, ASLType:$value);
  let results = (outs ASL_ArrayType:$result);
}


def ASL_EnumArrayConstructOp : ASL_Op<"enum_array_construct", [Pure]> {
  let summary = "ASL enumeration array construction";
  let description = [{
    The `asl.enum_array_construct` operation constructs an enumeration-indexed array.
    
    Example:
    ```mlir
    %array = asl.enum_array_construct %value : !asl.array<!asl.int, #asl.array_enum<"Color", ["RED", "GREEN", "BLUE"]>> {
      enum_name = "Color",
      enum_labels = ["RED", "GREEN", "BLUE"]
    }
    ```
  }];
  
  let arguments = (ins ASLType:$value, StrAttr:$enumName, ArrayAttr:$enumLabels);
  let results = (outs ASL_ArrayType:$result);
}


def ASL_ArbitraryOp : ASL_Op<"arbitrary", [Pure]> {
  let summary = "ASL arbitrary value operation";
  let description = [{
    The `asl.arbitrary` operation represents an arbitrary value of the specified type.
    
    Example:
    ```mlir
    %value = asl.arbitrary : !asl.int
    ```
  }];
  
  let arguments = (ins);
  let results = (outs ASLType:$result);
}


def ASL_PatternAllOp : ASL_Op<"pattern.all", [Pure]> {
  let summary = "ASL wildcard pattern";
  let description = [{
    The `asl.pattern.all` operation represents a wildcard pattern that matches any value.
    
    Example:
    ```mlir
    %match = asl.pattern.all %expr : !asl.bool
    ```
  }];
  
  let arguments = (ins ASLType:$expr);
  let results = (outs ASL_BoolType:$result);
}

def ASL_PatternAnyOp : ASL_Op<"pattern.any", [Pure]> {
  let summary = "ASL disjunctive pattern";
  let description = [{
    The `asl.pattern.any` operation represents a disjunctive pattern that
    matches if any of the sub-patterns match.
    
    Example:
    ```mlir
    %match = asl.pattern.any %expr, [%pattern1, %pattern2] : !asl.bool
    ```
  }];
  
  let arguments = (ins ASLType:$expr, Variadic<ASL_BoolType>:$subPatterns);
  let results = (outs ASL_BoolType:$result);
}

def ASL_PatternGeqOp : ASL_Op<"pattern.geq", [Pure]> {
  let summary = "ASL greater-or-equal pattern";
  let description = [{
    The `asl.pattern.geq` operation matches values greater than or equal to a threshold.
    
    Example:
    ```mlir
    %match = asl.pattern.geq %expr, %threshold : !asl.int
    ```
  }];
  
  let arguments = (ins ASLType:$expr, ASLType:$threshold);
  let results = (outs ASL_BoolType:$result);
}

def ASL_PatternLeqOp : ASL_Op<"pattern.leq", [Pure]> {
  let summary = "ASL less-or-equal pattern";
  let description = [{
    The `asl.pattern.leq` operation matches values less than or equal to a threshold.
    
    Example:
    ```mlir
    %match = asl.pattern.leq %expr, %threshold : !asl.int
    ```
  }];
  
  let arguments = (ins ASLType:$expr, ASLType:$threshold);
  let results = (outs ASL_BoolType:$result);
}

def ASL_PatternMaskOp : ASL_Op<"pattern.mask", [Pure]> {
  let summary = "ASL bitvector mask pattern";
  let description = [{
    The `asl.pattern.mask` operation matches expressions against a bitvector mask pattern.
    
    Example:
    ```mlir
    %match = asl.pattern.mask %expr {mask = #asl.bitvector_mask<"10xx01">} : !asl.bits<6>
    ```
  }];
  
  let arguments = (ins ASL_BitsType:$expr, ASL_BitvectorMaskAttr:$mask);
  let results = (outs ASL_BoolType:$result);
}

def ASL_PatternNotOp : ASL_Op<"pattern.not", [Pure]> {
  let summary = "ASL pattern negation";
  let description = [{
    The `asl.pattern.not` operation represents the negation of another pattern.
    
    Example:
    ```mlir
    %match = asl.pattern.not %expr, %negated_pattern : !asl.int
    ```
  }];
  
  let arguments = (ins ASLType:$expr, ASL_BoolType:$negatedPattern);
  let results = (outs ASL_BoolType:$result);
}

def ASL_PatternRangeOp : ASL_Op<"pattern.range", [Pure]> {
  let summary = "ASL range pattern";
  let description = [{
    The `asl.pattern.range` operation matches values within an inclusive range.
    
    Example:
    ```mlir
    %match = asl.pattern.range %expr, %lower, %upper : !asl.int
    ```
  }];
  
  let arguments = (ins ASLType:$expr, ASLType:$lower, ASLType:$upper);
  let results = (outs ASL_BoolType:$result);
}

def ASL_PatternSingleOp : ASL_Op<"pattern.single", [Pure]> {
  let summary = "ASL single value pattern";
  let description = [{
    The `asl.pattern.single` operation matches a single specific value.
    
    Example:
    ```mlir
    %match = asl.pattern.single %expr, %value : !asl.int
    ```
  }];
  
  let arguments = (ins ASLType:$expr, ASLType:$value);
  let results = (outs ASL_BoolType:$result);
}

def ASL_PatternTupleOp : ASL_Op<"pattern.tuple", [Pure]> {
  let summary = "ASL tuple pattern";
  let description = [{
    The `asl.pattern.tuple` operation destructures and matches tuple expressions.
    
    Example:
    ```mlir
    %match = asl.pattern.tuple %expr, [%pat1, %pat2] : !asl.tuple<!asl.int, !asl.real>
    ```
  }];
  
  let arguments = (ins ASL_TupleType:$expr, Variadic<ASL_BoolType>:$subPatterns);
  let results = (outs ASL_BoolType:$result);
}


#endif // ASL_EXPRESSIONS
