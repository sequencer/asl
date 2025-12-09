= ASL Lowering Passes Specification

This document specifies the transformation passes required to lower ASL dialect to EmitC, based on analysis of the Intel ASL Interpreter's `xform_*.ml` transformation modules.

*Important:* This project uses ARM's herdtools7 asllib as the frontend, which already performs several transformations before generating the typed JSON AST. Passes marked as "Handled by Frontend" below do not need MLIR implementations.

== Table of Contents

+ Overview
+ Frontend vs MLIR Transformations
+ Pass Pipeline
+ Monomorphization Pass
+ Tuple Elimination Pass
+ Getter/Setter Elimination Pass (Handled by Frontend)
+ Bitslice Normalization Pass
+ Bittuple Elimination Pass
+ Case Statement Lowering Pass (Handled by Frontend)
+ Desugar Pass (Partially Handled by Frontend)
+ Constant Propagation Pass
+ Expression Simplification Pass
+ Global Variable Wrapping Pass

#line(length: 100%)

== Overview

=== Purpose

The ASL dialect contains high-level constructs that must be lowered to simpler forms before EmitC translation. Each pass eliminates or simplifies specific language features, eventually producing an IR that maps directly to C constructs.

=== Reference

These passes are based on the Intel ASL Interpreter's transformation modules:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, left, left),
  table.header(
    [*Intel Pass*], [*MLIR Pass*], [*Status*], [*Purpose*],
  ),
  [`xform_mono.ml`], [`--asl-monomorphize`], [Partial (Frontend)], [Specialize parameterized types/functions],
  [`xform_tuples.ml`], [`--asl-eliminate-tuples`], [Required], [Convert tuples to records],
  [`xform_getset.ml`], [`--asl-eliminate-getset`], [Frontend], [Replace getters/setters with calls],
  [`xform_lower.ml`], [`--asl-normalize-slices`], [Required], [Normalize slice representations],
  [`xform_bitslices.ml`], [`--asl-lower-bitslices`], [Required], [Lower bitslice operations],
  [`xform_bittuples.ml`], [`--asl-eliminate-bittuples`], [Required], [Expand bittuple patterns],
  [`xform_case.ml`], [`--asl-lower-case`], [Frontend], [Convert complex case to if-chains],
  [`xform_desugar.ml`], [`--asl-desugar`], [Partial (Frontend)], [Desugar int-bits operations],
  [`xform_constprop.ml`], [`--asl-constprop`], [Required], [Constant propagation],
  [`xform_simplify_expr.ml`], [`--asl-simplify-expr`], [Required], [Algebraic simplification],
  [`xform_wrap.ml`], [`--asl-wrap-globals`], [Optional], [Wrap global access (optional)],
)

#line(length: 100%)

== Frontend vs MLIR Transformations

The herdtools7 asllib frontend performs several transformations during parsing and type-checking. Understanding this division is critical to avoid duplicate work.

=== Transformations Handled by herdtools7 Frontend

The following transformations are performed by `asllib/desugar.ml` and `asllib/typing.ml` before JSON output:

*Case Statement Desugaring* (desugar.ml: `desugar_case_stmt`)
- Complex case statements with pattern matching are converted to conditional chains (S_Cond)
- Mask patterns, range patterns, and multiple alternatives are all lowered to if-else sequences
- The JSON output contains only S_Cond nodes, not S_Case

*Getter/Setter Elimination* (desugar.ml: `desugar_setter`, `desugar_accessor_pair`)
- Accessor pairs (getter/setter definitions) are split into separate read and write functions
- Setter invocations are transformed into read-modify-write sequences
- L-expression setters use the `LE_SetField`, `LE_SetArray`, and `LE_Slice` constructs

*L-Expression Desugaring* (desugar.ml: `desugar_lhs_access`)
- Field access on LHS transformed to `LE_SetField`
- Array access on LHS transformed to `LE_SetArray`
- Slice access on LHS transformed to `LE_Slice`

*Tuple Destructuring* (desugar.ml: `desugar_ldi`)
- Multiple variable declarations split into individual declarations
- LHS tuple destructuring converted to `LE_Destructuring`

*Function Monomorphization* (typing.ml: function renaming)
- Polymorphic function calls are resolved during type-checking
- Functions are renamed with type suffixes for each instantiation
- Type parameters are substituted with concrete types

*Elided Parameter Filling* (desugar.ml: `desugar_elided_parameter`)
- Missing parameters are filled in based on type context

=== Transformations Required in MLIR

The following must still be implemented as MLIR passes:

#table(
  columns: (auto, auto),
  align: (left, left),
  table.header(
    [*Pass*], [*Reason*],
  ),
  [Tuple Elimination], [Frontend handles destructuring but not tuple-to-record type conversion],
  [Bitslice Normalization], [Multiple slice forms (Slice_Single, Slice_Range, Slice_Length, Slice_Star) still present in JSON],
  [Bittuple Elimination], [Multi-element concatenation patterns not handled by frontend],
  [Constant Propagation], [No compile-time evaluation in frontend],
  [Expression Simplification], [No algebraic simplification in frontend],
  [Int-Bits Desugar], [Mixed integer-bitvector operations need lowering (partial frontend handling)],
  [Global Variable Wrapping], [Optional pass for simulator integration],
)

#line(length: 100%)

== Pass Pipeline

=== Recommended Order

The pipeline is simplified compared to Intel's because herdtools7 handles several transformations:

```
ASL Dialect (from JSON import)
    |
    |   [Already done by herdtools7 frontend:]
    |   - Case statement -> if-else chains
    |   - Getter/setter -> read/write functions
    |   - Function monomorphization (partial)
    |   - Tuple destructuring on LHS
    |
    +-> [1. Tuple Elimination] Convert tuple types to records
    |
    +-> [2. Bitslice Normalization] Unify slice representations
    |
    +-> [3. Bittuple Elimination] Expand multi-slice patterns
    |
    +-> [4. Desugar] Lower int-bits operations
    |
    +-> [5. Constant Propagation] Fold constants, eliminate dead code
    |
    +-> [6. Expression Simplification] Algebraic simplification
    |
    +-> [7. ASLToGMP] Lower int/real to GMP dialect
    |
    +-> [8. ASLToEmitC] Lower remaining ASL to EmitC
    |
    +-> [9. GMPToEmitC] Lower GMP to EmitC
    |
    v
EmitC Dialect
```

=== Pass Dependencies

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*Pass*], [*Requires*], [*Enables*],
  ),
  [Tuple Elimination], [None], [Function lowering],
  [Bitslice Normalization], [None], [Bitslice lowering],
  [Bittuple Elimination], [Bitslice normalization], [Expression lowering],
  [Desugar], [None], [Type-consistent operations],
  [Constant Propagation], [Any], [All optimization passes],
  [Expression Simplification], [None], [Constant propagation],
)

#line(length: 100%)

== Monomorphization Pass

*Pass name:* `--asl-monomorphize`

*Reference:* `xform_mono.ml`

*Status:* Partially handled by herdtools7 frontend

=== Frontend Handling

The herdtools7 typing pass (`typing.ml`) performs function monomorphization:
- Polymorphic function calls are resolved during type-checking
- Functions are renamed with type suffixes based on instantiation
- Type parameters in function bodies are substituted with concrete types

=== Remaining Work (If Any)

The MLIR pass may still be needed for:
- Parameterized type definitions that need struct instantiation
- Any cases where the frontend does not fully specialize

Verify by examining the JSON output for your ASL files. If all functions appear with concrete types and mangled names, this pass can be skipped.

=== Transformations (For Reference)

*Parameterized Type Instantiation:*
- Input: `MyType{N}` used with `N=32`
- Output: New type `MyType_32` with parameter substituted

*Parameterized Function Specialization:*
- Input: `func{N}(x : bits(N))` called with `N=64`
- Output: New function `func_64(x : bits(64))`

=== Naming Convention

Specialized names append parameter values with underscores:
- `Type{10, 20}` -> `Type_10_20`
- `func{32}` -> `func_32`

#line(length: 100%)

== Tuple Elimination Pass

*Pass name:* `--asl-eliminate-tuples`

*Reference:* `xform_tuples.ml`

=== Purpose

Converts anonymous tuple types and operations into named record types, which map directly to C structs.

=== Transformations

*Return Type Conversion:*
- Input: `func foo() -> (bits(32), bits(32))`
- Output: `func foo() -> __Return_foo` with `type __Return_foo = record { r0: bits(32), r1: bits(32) }`

*Tuple Return Statements:*
- Input: `return (a, b);`
- Output: `return __Return_foo { r0 = a, r1 = b };`

*Tuple Assignment (parallel):*
- Input: `(x, y) = (a, b);`
- Output: `x = a; y = b;`

*Tuple Assignment (from function):*
- Input: `(x, y) = foo();`
- Output: `let __tmp = foo(); x = __tmp.r0; y = __tmp.r1;`

*Conditional Tuple:*
- Input: `(x, y) = if c then (a, b) else (c, d);`
- Output: `if c then { x = a; y = b; } else { x = c; y = d; }`

=== Generated Record Naming

- Function return tuples: `__Return_<function_name>`
- Anonymous tuples in types: `__Tuple_<type_signature>`

#line(length: 100%)

== Getter/Setter Elimination Pass

*Pass name:* `--asl-eliminate-getset`

*Reference:* `xform_getset.ml`

*Status:* Fully handled by herdtools7 frontend - NO MLIR PASS NEEDED

=== Frontend Handling

The herdtools7 desugar pass (`desugar.ml`) fully handles getter/setter elimination:

*`desugar_accessor_pair`:* Splits accessor pair definitions into separate read and write functions.

*`desugar_setter`:* Transforms setter invocations into read-modify-write sequences using:
- `LE_SetField` for field access on LHS
- `LE_SetArray` for array access on LHS
- `LE_Slice` for slice access on LHS

The JSON AST output contains only regular function calls and the specialized L-expression forms, not getter/setter syntax.

=== Transformations (For Reference)

These transformations are documented for understanding, but are already performed by the frontend:

*Read-Write L-Expression:*
- Input: `X[i] = X[i] + 1;` where X has getter/setter
- Output: Read via getter call, modify, write via setter call

*Write-Only L-Expression:*
- Input: `X[i] = value;`
- Output: Write via setter call

*Read Expression:*
- Input: `y = X[i];`
- Output: Read via getter call

#line(length: 100%)

== Bitslice Normalization Pass

*Pass name:* `--asl-normalize-slices`

*Reference:* `xform_lower.ml`

=== Purpose

Normalizes all bitslice representations to the `Slice_LoWd` (low + width) canonical form.

=== Transformations

ASL supports multiple slice notations:

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*Input Form*], [*Meaning*], [*Normalized Form*],
  ),
  [`x[i]` (Slice_Single)], [Single bit at index i], [`x[i +: 1]`],
  [`x[hi:lo]` (Slice_HiLo)], [Bits from hi down to lo], [`x[lo +: (hi - lo + 1)]`],
  [`x[hi -: wd]` (Slice_HiWd)], [Width bits ending at hi], [`x[(hi - wd + 1) +: wd]`],
  [`x[lo +: wd]` (Slice_LoWd)], [Width bits starting at lo], [(canonical)],
  [`x[i * n, n]` (Slice_Element)], [Element i of width n], [`x[(i * n) +: n]`],
)

=== Complex Expression Handling

When slice bounds contain complex expressions, introduce temporaries to avoid duplication:

- Input: `x[f(y) +: g(z)]`
- Output: `let __lo = f(y); let __wd = g(z); x[__lo +: __wd]`

=== L-Expression Slice Lifting

For nested slice assignments, lift inner slices:

- Input: `x[a +: b][c +: d] = v;`
- Output:
  ```
  let __tmp = x[a +: b];
  __tmp[c +: d] = v;
  x[a +: b] = __tmp;
  ```

#line(length: 100%)

== Bittuple Elimination Pass

*Pass name:* `--asl-eliminate-bittuples`

*Reference:* `xform_bittuples.ml`

=== Purpose

Expands bittuple (multi-element bit concatenation) patterns into explicit slice operations.

=== Transformations

*Bittuple L-Expression Assignment:*
- Input: `[x, y, z] = e;` where x, y, z have widths wx, wy, wz
- Output:
  ```
  let __tmp = e;
  x = __tmp[(wy + wz) +: wx];
  y = __tmp[wz +: wy];
  z = __tmp[0 +: wz];
  ```

*Multiple Slice L-Expression:*
- Input: `x[0 +: 8, 8 +: 8] = e;`
- Output:
  ```
  x[0 +: 8] = e[8 +: 8];
  x[8 +: 8] = e[0 +: 8];
  ```

*Multiple Slice R-Expression:*
- Input: `y = e[0 +: 8, 8 +: 8];`
- Output: `y = e[8 +: 8] : e[0 +: 8];` (concatenation)

*Bittuple Variable Declaration:*
- Input: `let [hi : bits(32), lo : bits(32)] = e;`
- Output:
  ```
  let __tmp = e;
  let hi = __tmp[32 +: 32];
  let lo = __tmp[0 +: 32];
  ```

#line(length: 100%)

== Case Statement Lowering Pass

*Pass name:* `--asl-lower-case`

*Reference:* `xform_case.ml`

*Status:* Fully handled by herdtools7 frontend - NO MLIR PASS NEEDED

=== Frontend Handling

The herdtools7 desugar pass (`desugar.ml: desugar_case_stmt`) fully handles case statement lowering:

- All case statements are converted to conditional chains (S_Cond)
- Mask patterns are lowered to bitwise AND comparisons
- Range patterns are lowered to inequality comparisons
- Multiple alternatives are lowered to OR conditions
- The JSON AST output contains only S_Cond nodes, never S_Case

The JSON importer should map S_Cond directly to `asl.stmt.cond` (if-else) operations.

=== Transformations (For Reference)

These transformations are documented for understanding, but are already performed by the frontend:

*Complex Case to If-Chain:*
- Input: case statement with pattern matching
- Output: Nested S_Cond (if-else) statements

*Mask Pattern Conversion:*
- Input: `when '1xx0'` (x = don't care)
- Output: Bitwise AND with mask, comparison with pattern

*Range Pattern:*
- Input: `when 1..10`
- Output: Greater-or-equal AND less-or-equal comparisons

#line(length: 100%)

== Desugar Pass

*Pass name:* `--asl-desugar`

*Reference:* `xform_desugar.ml`

*Status:* Partially handled by herdtools7 frontend

=== Frontend Handling

The herdtools7 desugar pass handles several transformations:
- Multiple variable declarations split into individual declarations
- Elided parameters filled in based on type context
- Tuple destructuring on LHS converted to LE_Destructuring

=== Remaining Work (MLIR Pass Required)

The following transformations are NOT handled by the frontend and require an MLIR pass:

*Integer-Bitvector Arithmetic:*
- `add_bits_int{N}(x, y)` -> `add_bits(x, cvt_int_bits{N}(y, N))`
- `sub_bits_int{N}(x, y)` -> `sub_bits(x, cvt_int_bits{N}(y, N))`
- `mul_bits_int{N}(x, y)` -> `mul_bits(x, cvt_int_bits{N}(y, N))`

The integer operand must be explicitly converted to a bitvector of matching width before the operation.

=== Purpose

This pass ensures type consistency: all bitvector operations have bitvector operands, simplifying type-based lowering decisions in subsequent passes.

#line(length: 100%)

== Constant Propagation Pass

*Pass name:* `--asl-constprop`

*Reference:* `xform_constprop.ml`

=== Purpose

Evaluates compile-time constant expressions and propagates known values, enabling subsequent optimizations and dead code elimination.

=== Transformations

*Constant Folding:*
- `3 + 5` -> `8`
- `TRUE && FALSE` -> `FALSE`
- `0x0F & 0xF0` -> `0x00`

*Dead Code Elimination:*
- Assertions with constant TRUE condition -> removed
- Unreachable branches (constant FALSE condition) -> removed
- Case branches that cannot match -> removed

*Algebraic Simplifications:*
- `x + 0` -> `x`
- `x * 1` -> `x`
- `x * 0` -> `0`
- `x && TRUE` -> `x`
- `x || FALSE` -> `x`
- `Replicate(x, 1)` -> `x`
- `Replicate(x, 0)` -> `''` (empty bits)
- `'' : x` -> `x` (empty concatenation)

*Loop Unrolling (optional):*
When loop bounds are constants, unroll the entire loop:
- Input: `for i = 0 to 2 do body end`
- Output: `body[i:=0]; body[i:=1]; body[i:=2];`

=== Implementation

Uses abstract interpretation with a lattice tracking:
- Known constant values
- Unknown/dynamic values
- Type information for proper evaluation

#line(length: 100%)

== Expression Simplification Pass

*Pass name:* `--asl-simplify-expr`

*Reference:* `xform_simplify_expr.ml`

=== Purpose

Performs algebraic simplification by converting expressions to polynomial form, combining like terms, and reconstructing simplified expressions.

=== Approach

+ Convert arithmetic expressions to polynomial representation
+ Combine like terms (e.g., `3x + 2x = 5x`)
+ Cancel terms that sum to zero (e.g., `x - x = 0`)
+ Reconstruct simplified expression

=== Handled Operations

- Addition, subtraction, negation
- Multiplication (including distribution)
- Integer constants

=== Safety Constraints

- Function calls are treated as uninterpreted terms (not reordered)
- Only pure expressions are simplified
- Side-effecting expressions preserve evaluation order

#line(length: 100%)

== Global Variable Wrapping Pass

*Pass name:* `--asl-wrap-globals`

*Reference:* `xform_wrap.ml`

=== Purpose

Replaces direct global variable access with function calls, enabling simulator integration where global state can be intercepted.

=== Transformations

*Scalar Variable Read:*
- Input: `x` (global variable reference)
- Output: `x_read()`

*Scalar Variable Write:*
- Input: `x = value;`
- Output: `x_write(value);`

*Array Variable Read:*
- Input: `arr[i]`
- Output: `arr_read(i)`

*Array Variable Write:*
- Input: `arr[i] = value;`
- Output: `arr_write(i, value);`

=== Generated Functions

For each global variable `x`:
- `x_read() -> T` - Returns current value
- `x_write(v : T)` - Updates value

For each global array `arr`:
- `arr_read(i : IndexType) -> ElemType`
- `arr_write(i : IndexType, v : ElemType)`

=== Use Case

This pass is optional and primarily used for simulator integration where:
- Global state represents processor/memory state
- Simulator provides custom implementations of read/write functions
- Enables tracing, breakpoints, or alternate memory models

#line(length: 100%)

== Testing Strategy

=== Per-Pass Tests

Each pass should have dedicated tests in `asl-mlir/test/ASL/Transforms/<pass-name>/`:

+ *Basic functionality* - Core transformations work correctly
+ *Edge cases* - Boundary conditions, empty inputs, nested structures
+ *Preservation* - Untransformed constructs pass through unchanged
+ *Composition* - Pass works correctly after prerequisites

=== Integration Tests

End-to-end tests running the full pipeline:

+ Parse ASL to JSON
+ Import to ASL dialect
+ Run all lowering passes in order
+ Lower to EmitC
+ Generate C code
+ Compile and execute
+ Verify output

#line(length: 100%)

== References

+ *Intel ASL Interpreter*: https://github.com/IntelLabs/asl-interpreter
  - `libASL/xform_*.ml` - Transformation implementations
  - `libASL/backend_c.ml` - C backend expectations

+ *ARM herdtools7 asllib*: https://github.com/herd/herdtools7/tree/master/asllib
  - `asllib/desugar.ml` - Desugaring transformations (case statements, getters/setters, LHS access)
  - `asllib/typing.ml` - Type-checking with function monomorphization
  - `asllib/AST.mli` - AST type definitions
  - This is the frontend used by this project for parsing and type-checking ASL
