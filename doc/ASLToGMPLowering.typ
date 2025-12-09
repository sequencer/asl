= ASL to GMP Dialect Lowering Specification

This document specifies the lowering of ASL dialect integer and real operations to the GMP dialect. The lowering follows patterns from the Intel ASL Interpreter's Zarith-based implementation, adapted for direct GMP usage.

== Table of Contents

+ Overview
+ Type Mapping
+ Constraint-Based Native Integer Optimization
+ Literal Lowering
+ Binary Operations
+ Unary Operations
+ Comparison Operations
+ Type Conversions
+ Division Semantics
+ Shift Operations
+ Pass Implementation
+ Testing Strategy

#line(length: 100%)

== Overview

=== Purpose

The ASL dialect represents arbitrary-precision integers (`asl.int`) and exact rationals (`asl.real`) as specified by ARM's ASL language. The GMP dialect provides a clean abstraction over GMP operations with compile-time constant folding capabilities.

=== Lowering Pipeline

```
ASL Dialect
    |
    +-> [ASLToGMP Pass] Lower integer/real ops to GMP or native ints
    |
    v
GMP Dialect + arith Dialect
    |
    +-> [GMP Canonicalize] Constant folding and algebraic simplifications
    |
    v
GMP Dialect (optimized)
    |
    +-> [GMPToEmitC Pass] Lower to C code with mpz_t/mpq_t calls
    |
    v
EmitC Dialect -> C Code
```

=== Reference: Intel ASL Interpreter

The Intel ASL Interpreter uses OCaml's Zarith library (Z and Q modules) for arbitrary-precision arithmetic. Key reference files:

- `libASL/primops.ml` - Primitive operations semantics
- `libASL/value.ml` - Value representation (`VInt of bigint`, `VReal of real`)

Zarith internally uses GMP for large integers, so the semantic mapping is direct.

#line(length: 100%)

== Type Mapping

=== ASL Types to Target Types

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Type*], [*Target Type*], [*Description*],
  ),
  [`!asl.int<unconstrained>`], [`!gmp.z`], [Unconstrained arbitrary-precision integer],
  [`!asl.int<wellconstrained>`], [`i8`/`i16`/`i32`/`i64` or `!gmp.z`], [*Optimized*: native int if bounds fit <= 64 bits, else GMP],
  [`!asl.int<underconstrained>`], [`!gmp.z`], [Under-constrained integer],
  [`!asl.int<pendingconstrained>`], [`!gmp.z`], [Pending constraint resolution],
  [`!asl.real`], [`!gmp.q`], [Exact rational number],
)

*Note*: Well-constrained integers are the primary target for native integer optimization. See Constraint-Based Native Integer Optimization for details.

=== Constraint Handling

ASL integer constraints are separate from the numeric representation. The GMP dialect handles only the arithmetic; constraint checking occurs at the ASL level before/after operations.

#line(length: 100%)

== Constraint-Based Native Integer Optimization

=== Motivation

GMP operations (`mpz_t`) incur significant overhead compared to native integer operations:
- Memory allocation for each integer
- Function call overhead for every operation
- Cache-unfriendly memory access patterns

For *well-constrained integers* where the type system can prove the value fits within 64 bits (or smaller), we can lower directly to MLIR's native integer types (`i8`, `i16`, `i32`, `i64`) and use `arith` dialect operations instead of GMP.

=== Constraint Analysis

ASL integer types carry constraint information that can be analyzed to determine bounds. Range constraints specify minimum and maximum values; exact constraints specify a single known value.

==== Determining Bit Width from Constraints

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, left, left),
  table.header(
    [*Constraint Range*], [*Required Bits*], [*C Type*], [*MLIR Type*],
  ),
  [`[0, 255]`], [8 (unsigned)], [`uint8_t`], [`i8`],
  [`[-128, 127]`], [8 (signed)], [`int8_t`], [`i8`],
  [`[0, 65535]`], [16 (unsigned)], [`uint16_t`], [`i16`],
  [`[-32768, 32767]`], [16 (signed)], [`int16_t`], [`i16`],
  [`[0, 2^32-1]`], [32 (unsigned)], [`uint32_t`], [`i32`],
  [`[-2^31, 2^31-1]`], [32 (signed)], [`int32_t`], [`i32`],
  [`[0, 2^64-1]`], [64 (unsigned)], [`uint64_t`], [`i64`],
  [`[-2^63, 2^63-1]`], [64 (signed)], [`int64_t`], [`i64`],
  [Larger or unknown], [arbitrary], [`mpz_t`], [`!gmp.z`],
)

=== Type Conversion Logic

The type converter should:
+ Extract constraint bounds from the ASL integer type
+ Check if bounds fit within native integer ranges
+ Select the smallest native integer type that fits, or fall back to `!gmp.z`

For unsigned ranges (min >= 0), check against UINT8_MAX, UINT16_MAX, UINT32_MAX, UINT64_MAX.
For signed ranges (min < 0), check against INT8_MIN/MAX, INT16_MIN/MAX, INT32_MIN/MAX, INT64_MIN/MAX.

=== Native Integer Operation Mapping

When operands are native integers, use `arith` dialect instead of GMP:

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Operation*], [*Native (arith)*], [*GMP*],
  ),
  [`asl.expr.binop.plus`], [`arith.addi`], [`gmp.z.add`],
  [`asl.expr.binop.minus`], [`arith.subi`], [`gmp.z.sub`],
  [`asl.expr.binop.mul`], [`arith.muli`], [`gmp.z.mul`],
  [`asl.expr.binop.div`], [`arith.divsi`], [`gmp.z.div`],
  [`asl.expr.binop.mod`], [`arith.remsi` (with adjustment)], [`gmp.z.frem`],
  [`asl.expr.binop.shl`], [`arith.shli`], [`gmp.z.shift_left`],
  [`asl.expr.binop.shr`], [`arith.shrsi`], [`gmp.z.shift_right`],
  [`asl.expr.unop.neg`], [`arith.subi 0, x`], [`gmp.z.neg`],
  [`asl.expr.binop.eq`], [`arith.cmpi eq`], [`gmp.z.equal`],
  [`asl.expr.binop.lt`], [`arith.cmpi slt`], [`gmp.z.lt`],
)

=== Division Semantics for Native Integers

*Important:* Native integer division in MLIR's `arith` dialect uses truncated division (toward zero), which matches ASL's `DIV`. However, ASL's `MOD` uses floor semantics, requiring adjustment when operand signs differ.

Floor remainder adjustment: When the truncated remainder is non-zero and the signs of remainder and divisor differ, add the divisor to get the floor remainder.

=== Overflow Handling

For operations that might overflow the native integer range:

+ *Conservative approach*: If overflow is possible, use GMP
+ *Checked arithmetic*: Use overflow-checking intrinsics and trap on overflow
+ *Widening*: Perform operation in wider type (e.g., 32-bit operands -> 64-bit multiplication), then truncate

=== Mixed-Width Operations

When operands have different widths, sign-extend or zero-extend the narrower operand to match the wider type before the operation.

=== Pass Option

The optimization can be controlled via pass option: `--asl-to-gmp="enable-native-int-optimization=true"`

=== Limitations

+ *Constraint propagation required*: The optimization depends on accurate constraint information being available. If constraints are unknown or too wide, falls back to GMP.

+ *Operation result bounds*: Must analyze whether operation results stay within native integer range. For example, `a + b` where both are `i32` might overflow.

+ *No runtime checks by default*: The optimization assumes constraints are correct. Runtime overflow is undefined behavior unless checked arithmetic is enabled.

+ *Cross-function boundaries*: Constraint information may not propagate across function calls without interprocedural analysis.

#line(length: 100%)

== Literal Lowering

=== Integer Literals

*ASL Operation:* `asl.expr.literal.int`

The ASL integer literal stores the value as a string (decimal representation) to handle arbitrary precision.

*Lowering:* `asl.expr.literal.int "value"` -> `gmp.z.constant #gmp.z<"value">`

For well-constrained integers that fit in native types, lower to `arith.constant` instead.

=== Real Literals

*ASL Operation:* `asl.expr.literal.real`

ASL real literals are stored as strings in "numerator/denominator" format.

*Lowering:* `asl.expr.literal.real "num/den"` -> `gmp.q.constant #gmp.q<num, den>`

*Parsing Decimal Reals:* For decimal format like "3.14", convert to rational by:
- Concatenate integer and fractional parts: "314"
- Denominator is 10^(fractional digits): 100
- Result: 314/100 (automatically reduced by GMP)

=== Boolean Literals

Boolean literals (`i1`) pass through unchanged - they are not GMP types.

#line(length: 100%)

== Binary Operations

=== Integer Binary Operations

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, left, left),
  table.header(
    [*ASL Operation*], [*GMP Operation*], [*Zarith Equivalent*], [*Notes*],
  ),
  [`asl.expr.binop.plus` (int)], [`gmp.z.add`], [`Z.add`], [Commutative],
  [`asl.expr.binop.minus` (int)], [`gmp.z.sub`], [`Z.sub`], [],
  [`asl.expr.binop.mul` (int)], [`gmp.z.mul`], [`Z.mul`], [Commutative],
  [`asl.expr.binop.div`], [`gmp.z.div`], [`Z.div`], [Truncate toward zero],
  [`asl.expr.binop.divrm`], [`gmp.z.fdiv`], [`Z.fdiv`], [Floor division],
  [`asl.expr.binop.mod`], [`gmp.z.frem`], [floor remainder], [Floor remainder],
  [`asl.expr.binop.pow`], [`gmp.z.pow`], [`Z.pow`], [Exponent must be non-negative],
  [`asl.expr.binop.shl`], [`gmp.z.shift_left`], [`Z.shift_left`], [Left shift],
  [`asl.expr.binop.shr`], [`gmp.z.shift_right`], [`Z.shift_right`], [Arithmetic right shift (floor)],
)

=== Real Binary Operations

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Operation*], [*GMP Operation*], [*Zarith Equivalent*],
  ),
  [`asl.expr.binop.plus` (real)], [`gmp.q.add`], [`Q.add`],
  [`asl.expr.binop.minus` (real)], [`gmp.q.sub`], [`Q.sub`],
  [`asl.expr.binop.mul` (real)], [`gmp.q.mul`], [`Q.mul`],
  [`asl.expr.binop.rdiv`], [`gmp.q.div`], [`Q.div`],
)

=== Type Dispatch

The ASL binary operations use `AnyType` for operands, requiring type-based dispatch during lowering. Check the result type to determine whether to emit GMP operations (for `!asl.int`/`!asl.real`) or leave unchanged (for `!asl.bits`).

#line(length: 100%)

== Unary Operations

=== Integer Unary Operations

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Operation*], [*GMP Operation*], [*Zarith Equivalent*],
  ),
  [`asl.expr.unop.neg` (int)], [`gmp.z.neg`], [`Z.neg`],
)

=== Real Unary Operations

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Operation*], [*GMP Operation*], [*Zarith Equivalent*],
  ),
  [`asl.expr.unop.neg` (real)], [`gmp.q.neg`], [`Q.neg`],
)

#line(length: 100%)

== Comparison Operations

=== Integer Comparisons

All ASL integer comparisons lower to GMP comparison operations that return `i1`:

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Operation*], [*GMP Operation*], [*Semantics*],
  ),
  [`asl.expr.binop.eq` (int)], [`gmp.z.equal`], [`lhs == rhs`],
  [`asl.expr.binop.neq` (int)], [`!gmp.z.equal`], [`lhs != rhs` (negate equal result)],
  [`asl.expr.binop.lt` (int)], [`gmp.z.lt`], [`lhs < rhs`],
  [`asl.expr.binop.leq` (int)], [`gmp.z.leq`], [`lhs <= rhs`],
  [`asl.expr.binop.gt` (int)], [`gmp.z.gt`], [`lhs > rhs`],
  [`asl.expr.binop.geq` (int)], [`gmp.z.geq`], [`lhs >= rhs`],
)

=== Real Comparisons

#table(
  columns: (auto, auto),
  align: (left, left),
  table.header(
    [*ASL Operation*], [*GMP Operation*],
  ),
  [`asl.expr.binop.eq` (real)], [`gmp.q.equal`],
  [`asl.expr.binop.neq` (real)], [`!gmp.q.equal`],
  [`asl.expr.binop.lt` (real)], [`gmp.q.lt`],
  [`asl.expr.binop.leq` (real)], [`gmp.q.leq`],
  [`asl.expr.binop.gt` (real)], [`gmp.q.gt`],
  [`asl.expr.binop.geq` (real)], [`gmp.q.geq`],
)

#line(length: 100%)

== Type Conversions

=== Integer to Real

Converting an ASL integer to real creates a rational with denominator 1.

*Lowering:* `gmp.q.from_z`

=== Real to Integer

Converting a real to integer truncates toward zero.

*Lowering:* `gmp.q.to_bigint` or `gmp.q.trunc`

=== Integer to Bitvector

Converting an integer to a bitvector extracts the low N bits.

*Lowering:* `gmp.z.extract` with offset 0 and width N

*Note:* The actual bitvector type is not a GMP type. The conversion happens at the boundary between GMP and native integer types.

=== Bitvector to Integer

*Unsigned conversion:* Import bitvector value directly as non-negative integer.

*Signed conversion:* Sign-extend based on the highest bit of the bitvector.

#line(length: 100%)

== Division Semantics

Division semantics are critical for correct ASL behavior. The Intel ASL interpreter uses Zarith's division functions which map to GMP as follows:

=== Truncated Division (DIV)

ASL's `DIV` operator truncates toward zero, matching C's integer division:

```
 7 DIV  3 =  2    (7 = 2*3 + 1)
-7 DIV  3 = -2    (-7 = -2*3 + -1)
 7 DIV -3 = -2    (7 = -2*-3 + 1)
-7 DIV -3 =  2    (-7 = 2*-3 + -1)
```

*Zarith:* `Z.div x y`
*GMP:* `mpz_tdiv_q(result, n, d)`
*MLIR:* `gmp.z.div`

=== Floor Division (DIVRM)

ASL's `DIVRM` operator rounds toward negative infinity:

```
 7 DIVRM  3 =  2    (7 = 2*3 + 1)
-7 DIVRM  3 = -3    (-7 = -3*3 + 2)
 7 DIVRM -3 = -3    (7 = -3*-3 + -2)
-7 DIVRM -3 =  2    (-7 = 2*-3 + -1)
```

*Zarith:* `Z.fdiv x y`
*GMP:* `mpz_fdiv_q(result, n, d)`
*MLIR:* `gmp.z.fdiv`

=== Modulo (MOD)

ASL's `MOD` is the floor remainder (sign matches divisor):

```
 7 MOD  3 =  1
-7 MOD  3 =  2
 7 MOD -3 = -2
-7 MOD -3 = -1
```

*Zarith:* `x - y * fdiv(x, y)`
*GMP:* `mpz_fdiv_r(result, n, d)`
*MLIR:* `gmp.z.frem`

#line(length: 100%)

== Shift Operations

=== Left Shift (SHL)

Equivalent to multiplication by 2^n.

*Zarith:* `Z.shift_left x n` where `n` is OCaml `int`
*GMP:* `mpz_mul_2exp(result, op, n)`
*MLIR:* `gmp.z.shift_left`

*Important:* The shift amount in GMP operations is `i64`, but ASL allows arbitrary-precision shift amounts. The lowering must convert the shift amount from `!gmp.z` to `i64`, which may fail at runtime for very large shift amounts.

=== Right Shift (SHR)

ASL's SHR is arithmetic right shift (floor division by 2^n).

*Zarith:* `Z.shift_right x n` (uses floor semantics for negative numbers)
*GMP:* `mpz_fdiv_q_2exp(result, op, n)`
*MLIR:* `gmp.z.shift_right`

#line(length: 100%)

== Pass Implementation

=== Pass Structure

The `ASLToGMPPass` should:
+ Set up a type converter that maps `asl.IntType` -> `gmp.ZType` (or native `IntegerType` with optimization) and `asl.RealType` -> `gmp.QType`
+ Configure conversion target with GMP and arith dialects as legal
+ Mark ASL integer/real operations as illegal (dynamically legal if operating on non-int/real types like bitvectors)
+ Populate rewrite patterns for all operations
+ Apply partial conversion

=== Full Operation Lowering Table

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Operation*], [*GMP Operation*], [*Type Condition*],
  ),
  [`asl.expr.literal.int`], [`gmp.z.constant`], [Always],
  [`asl.expr.literal.real`], [`gmp.q.constant`], [Always],
  [`asl.expr.binop.plus`], [`gmp.z.add`], [Result is `!asl.int`],
  [`asl.expr.binop.plus`], [`gmp.q.add`], [Result is `!asl.real`],
  [`asl.expr.binop.minus`], [`gmp.z.sub`], [Result is `!asl.int`],
  [`asl.expr.binop.minus`], [`gmp.q.sub`], [Result is `!asl.real`],
  [`asl.expr.binop.mul`], [`gmp.z.mul`], [Result is `!asl.int`],
  [`asl.expr.binop.mul`], [`gmp.q.mul`], [Result is `!asl.real`],
  [`asl.expr.binop.div`], [`gmp.z.div`], [Always (int only)],
  [`asl.expr.binop.divrm`], [`gmp.z.fdiv`], [Always (int only)],
  [`asl.expr.binop.mod`], [`gmp.z.frem`], [Always (int only)],
  [`asl.expr.binop.pow`], [`gmp.z.pow`], [Always (int only)],
  [`asl.expr.binop.shl`], [`gmp.z.shift_left`], [Always (int only)],
  [`asl.expr.binop.shr`], [`gmp.z.shift_right`], [Always (int only)],
  [`asl.expr.binop.rdiv`], [`gmp.q.div`], [Always (real only)],
  [`asl.expr.unop.neg`], [`gmp.z.neg`], [Operand is `!asl.int`],
  [`asl.expr.unop.neg`], [`gmp.q.neg`], [Operand is `!asl.real`],
  [`asl.expr.binop.eq`], [`gmp.z.equal`], [Operands are `!asl.int`],
  [`asl.expr.binop.eq`], [`gmp.q.equal`], [Operands are `!asl.real`],
  [`asl.expr.binop.lt`], [`gmp.z.lt`], [Operands are `!asl.int`],
  [`asl.expr.binop.lt`], [`gmp.q.lt`], [Operands are `!asl.real`],
  [`asl.expr.binop.leq`], [`gmp.z.leq`], [Operands are `!asl.int`],
  [`asl.expr.binop.leq`], [`gmp.q.leq`], [Operands are `!asl.real`],
  [`asl.expr.binop.gt`], [`gmp.z.gt`], [Operands are `!asl.int`],
  [`asl.expr.binop.gt`], [`gmp.q.gt`], [Operands are `!asl.real`],
  [`asl.expr.binop.geq`], [`gmp.z.geq`], [Operands are `!asl.int`],
  [`asl.expr.binop.geq`], [`gmp.q.geq`], [Operands are `!asl.real`],
)

#line(length: 100%)

== Testing Strategy

=== Unit Tests

Test files should be placed in `asl-mlir/test/ASL/Transforms/asl-to-gmp/`:

+ *Literal Tests* (`literals.mlir`): Test integer and real literal lowering, including large integers that exceed 64 bits.

+ *Arithmetic Tests* (`arithmetic.mlir`): Test all binary operations for both integer and real types.

+ *Division Semantics Tests* (`division-semantics.mlir`): Verify correct division behavior with constant folding:
  - `7 / 3 = 2` (truncated)
  - `-7 / 3 = -2` (truncated)
  - `-7 DIVRM 3 = -3` (floor)
  - `-7 MOD 3 = 2` (floor remainder)

+ *Native Integer Optimization Tests* (`native-int-opt.mlir`): Test that well-constrained integers lower to native types when optimization is enabled.

=== Integration Tests

End-to-end tests from ASL source to C code through the full pipeline:
`asl-json-backend` -> `asl-opt --asl-to-gmp --gmp-canonicalize --gmp-to-emitc --emitc` -> `gcc -lgmp`

#line(length: 100%)

== Appendix: Zarith to GMP Function Mapping

Complete reference of Zarith functions and their GMP equivalents:

=== Z Module (Integers)

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, left, left),
  table.header(
    [*Zarith Function*], [*Signature*], [*GMP Function*], [*Notes*],
  ),
  [`Z.of_string`], [`string -> t`], [`mpz_set_str`], [Parse decimal string],
  [`Z.to_string`], [`t -> string`], [`mpz_get_str`], [Convert to decimal string],
  [`Z.add`], [`t -> t -> t`], [`mpz_add`], [Addition],
  [`Z.sub`], [`t -> t -> t`], [`mpz_sub`], [Subtraction],
  [`Z.mul`], [`t -> t -> t`], [`mpz_mul`], [Multiplication],
  [`Z.neg`], [`t -> t`], [`mpz_neg`], [Negation],
  [`Z.abs`], [`t -> t`], [`mpz_abs`], [Absolute value],
  [`Z.succ`], [`t -> t`], [`mpz_add_ui(..., 1)`], [Successor (x + 1)],
  [`Z.pred`], [`t -> t`], [`mpz_sub_ui(..., 1)`], [Predecessor (x - 1)],
  [`Z.div`], [`t -> t -> t`], [`mpz_tdiv_q`], [Truncated quotient],
  [`Z.rem`], [`t -> t -> t`], [`mpz_tdiv_r`], [Truncated remainder],
  [`Z.fdiv`], [`t -> t -> t`], [`mpz_fdiv_q`], [Floor quotient],
  [`Z.cdiv`], [`t -> t -> t`], [`mpz_cdiv_q`], [Ceiling quotient],
  [`Z.ediv`], [`t -> t -> t`], [Custom], [Euclidean quotient],
  [`Z.erem`], [`t -> t -> t`], [Custom], [Euclidean remainder],
  [`Z.divexact`], [`t -> t -> t`], [`mpz_divexact`], [Exact division],
  [`Z.pow`], [`t -> int -> t`], [`mpz_pow_ui`], [Exponentiation],
  [`Z.shift_left`], [`t -> int -> t`], [`mpz_mul_2exp`], [Left shift],
  [`Z.shift_right`], [`t -> int -> t`], [`mpz_fdiv_q_2exp`], [Arithmetic right shift],
  [`Z.shift_right_trunc`], [`t -> int -> t`], [`mpz_tdiv_q_2exp`], [Truncating right shift],
  [`Z.logand`], [`t -> t -> t`], [`mpz_and`], [Bitwise AND],
  [`Z.logor`], [`t -> t -> t`], [`mpz_ior`], [Bitwise OR],
  [`Z.logxor`], [`t -> t -> t`], [`mpz_xor`], [Bitwise XOR],
  [`Z.lognot`], [`t -> t`], [`mpz_com`], [Bitwise complement],
  [`Z.equal`], [`t -> t -> bool`], [`mpz_cmp == 0`], [Equality test],
  [`Z.compare`], [`t -> t -> int`], [`mpz_cmp`], [Three-way comparison],
  [`Z.lt`], [`t -> t -> bool`], [`mpz_cmp < 0`], [Less than],
  [`Z.leq`], [`t -> t -> bool`], [`mpz_cmp <= 0`], [Less or equal],
  [`Z.gt`], [`t -> t -> bool`], [`mpz_cmp > 0`], [Greater than],
  [`Z.geq`], [`t -> t -> bool`], [`mpz_cmp >= 0`], [Greater or equal],
  [`Z.gcd`], [`t -> t -> t`], [`mpz_gcd`], [GCD],
  [`Z.lcm`], [`t -> t -> t`], [`mpz_lcm`], [LCM],
  [`Z.sqrt`], [`t -> t`], [`mpz_sqrt`], [Integer square root],
  [`Z.testbit`], [`t -> int -> bool`], [`mpz_tstbit`], [Test bit],
  [`Z.popcount`], [`t -> int`], [`mpz_popcount`], [Population count],
  [`Z.extract`], [`t -> int -> int -> t`], [`mpz_tdiv_r_2exp` + shift], [Extract bits],
)

=== Q Module (Rationals)

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, left, left),
  table.header(
    [*Zarith Function*], [*Signature*], [*GMP Function*], [*Notes*],
  ),
  [`Q.make`], [`Z.t -> Z.t -> t`], [`mpq_set_num/den` + `mpq_canonicalize`], [Construct rational],
  [`Q.num`], [`t -> Z.t`], [`mpq_numref`], [Get numerator],
  [`Q.den`], [`t -> Z.t`], [`mpq_denref`], [Get denominator],
  [`Q.add`], [`t -> t -> t`], [`mpq_add`], [Addition],
  [`Q.sub`], [`t -> t -> t`], [`mpq_sub`], [Subtraction],
  [`Q.mul`], [`t -> t -> t`], [`mpq_mul`], [Multiplication],
  [`Q.div`], [`t -> t -> t`], [`mpq_div`], [Division],
  [`Q.neg`], [`t -> t`], [`mpq_neg`], [Negation],
  [`Q.abs`], [`t -> t`], [`mpq_abs`], [Absolute value],
  [`Q.inv`], [`t -> t`], [`mpq_inv`], [Reciprocal],
  [`Q.compare`], [`t -> t -> int`], [`mpq_cmp`], [Three-way comparison],
  [`Q.equal`], [`t -> t -> bool`], [`mpq_equal`], [Equality test],
  [`Q.of_bigint`], [`Z.t -> t`], [`mpq_set_z`], [Integer to rational],
  [`Q.to_bigint`], [`t -> Z.t`], [`mpz_tdiv_q(num, den)`], [Rational to integer (truncate)],
  [`Q.mul_2exp`], [`t -> int -> t`], [`mpq_mul_2exp`], [Multiply by 2^n],
  [`Q.div_2exp`], [`t -> int -> t`], [`mpq_div_2exp`], [Divide by 2^n],
)

#line(length: 100%)

== References

+ *Intel ASL Interpreter*: https://github.com/alastairreid/asl-interpreter
  - `libASL/primops.ml` - Primitive operation semantics
  - `libASL/value.ml` - Value representation

+ *Zarith Library Documentation*: https://antoinemine.github.io/Zarith/doc/latest/Z.html

+ *GMP Manual*: https://gmplib.org/manual/

+ *ARM ASL Specification*: Architecture Specification Language reference from ARM
