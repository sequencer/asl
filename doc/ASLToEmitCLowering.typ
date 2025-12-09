= ASL to EmitC Dialect Lowering Specification

This document specifies the lowering of ASL dialect operations to MLIR's EmitC dialect for C code generation. It covers all ASL constructs except arbitrary-precision integers and rationals, which are handled by the GMP dialect (see `ASLToGMPLowering.typ`).

== Table of Contents

+ Overview
+ Type Lowering
+ Bitvector Operations
+ Boolean Operations
+ String Operations
+ Control Flow
+ Function Declarations
+ Data Access Operations
+ L-Expressions
+ Exception Handling
+ Global State Management
+ Runtime Support

#line(length: 100%)

== Overview

=== Lowering Pipeline

```
ASL Dialect
    |
    +-> [ASLToGMP Pass] Lower int/real ops to GMP dialect
    |
    v
ASL Dialect (with GMP ops) + GMP Dialect
    |
    +-> [ASLToEmitC Pass] Lower remaining ASL ops to EmitC
    |
    v
EmitC Dialect + GMP Dialect
    |
    +-> [GMPToEmitC Pass] Lower GMP ops to EmitC (GMP function calls)
    |
    v
EmitC Dialect
    |
    +-> [EmitC Translation] Generate C code
    |
    v
C Source Code
```

=== Reference Implementation

- *Intel ASL Interpreter*: `libASL/backend_c.ml` - C backend patterns
- *ARM herdtools7 asllib*: `asllib/Operations.ml` - Operation semantics

#line(length: 100%)

== Type Lowering

=== Type Mapping Table

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Type*], [*EmitC/C Type*], [*Notes*],
  ),
  [`!asl.bits<N>` (N <= 8)], [`uint8_t`], [Fixed-width bitvector],
  [`!asl.bits<N>` (N <= 16)], [`uint16_t`], [Fixed-width bitvector],
  [`!asl.bits<N>` (N <= 32)], [`uint32_t`], [Fixed-width bitvector],
  [`!asl.bits<N>` (N <= 64)], [`uint64_t`], [Fixed-width bitvector],
  [`!asl.bits<N>` (N > 64)], [`asl_bits_N_t`], [Custom struct with word array],
  [`i1`], [`bool`], [C99 `stdbool.h`],
  [`!asl.string`], [`const char*`], [Immutable string literal],
  [`!asl.enum<name>`], [`enum name`], [Named C enum],
  [`!asl.tuple<T1, T2, ...>`], [`struct { T1 item0; T2 item1; ... }`], [Anonymous or typedef'd struct],
  [`!asl.array<T, N>`], [`T[N]` or `struct { T data[N]; }`], [Fixed-size array],
  [`!asl.record<fields>`], [`struct name { ... }`], [Named struct with fields],
  [`!asl.exception`], [`struct { int code; ... }`], [Exception with discriminator],
)

=== Large Bitvector Type Generation

For bitvectors wider than 64 bits, generate a struct type:

*Pattern:* `asl_bits_<N>_t` where N is the bit width

*Structure:*
- `uint64_t words[(N + 63) / 64]` - Array of 64-bit words, little-endian order
- Word 0 contains bits [0:63], word 1 contains bits [64:127], etc.

=== Enum Type Generation

ASL enums lower to C enums with explicit values.

*Input:* `!asl.enum<"MyEnum", ["A", "B", "C"]>`

*Output:*
```c
typedef enum {
    MyEnum_A = 0,
    MyEnum_B = 1,
    MyEnum_C = 2
} MyEnum;
```

=== Tuple Type Generation

Tuples become structs with numbered fields.

*Input:* `!asl.tuple<i32, i64, bool>`

*Output:*
```c
typedef struct {
    int32_t item0;
    int64_t item1;
    bool item2;
} asl_tuple_i32_i64_bool_t;
```

=== Record Type Generation

Records become structs with named fields.

*Input:* `!asl.record<"Point", [("x", i32), ("y", i32)]>`

*Output:*
```c
typedef struct {
    int32_t x;
    int32_t y;
} Point;
```

#line(length: 100%)

== Bitvector Operations

=== Literal Lowering

*ASL:* `asl.expr.literal.bitvector`

Bitvector literals lower to integer constants with appropriate type.

#table(
  columns: (auto, auto),
  align: (left, left),
  table.header(
    [*Width*], [*Lowering*],
  ),
  [N <= 8], [`(uint8_t)value`],
  [N <= 16], [`(uint16_t)value`],
  [N <= 32], [`(uint32_t)value`],
  [N <= 64], [`(uint64_t)value` or `valueULL`],
  [N > 64], [Initialize struct with word array],
)

=== Binary Operations

All bitvector binary operations must apply a mask to ensure the result stays within the declared width.

*Mask computation:* `MASK(w) = (1ULL << w) - 1` for w <= 64

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Operation*], [*C Expression*], [*Notes*],
  ),
  [`asl.expr.binop.and`], [`lhs & rhs`], [No mask needed],
  [`asl.expr.binop.or`], [`lhs | rhs`], [No mask needed],
  [`asl.expr.binop.xor`], [`lhs ^ rhs`], [No mask needed],
  [`asl.expr.binop.plus` (bits)], [`(lhs + rhs) & MASK(w)`], [Wrap on overflow],
  [`asl.expr.binop.minus` (bits)], [`(lhs - rhs) & MASK(w)`], [Wrap on underflow],
  [`asl.expr.binop.mul` (bits)], [`(lhs * rhs) & MASK(w)`], [Wrap on overflow],
  [`asl.expr.binop.concat`], [`(lhs << rhs_width) | rhs`], [Concatenation],
  [`asl.expr.binop.eq` (bits)], [`lhs == rhs`], [Returns bool],
  [`asl.expr.binop.neq` (bits)], [`lhs != rhs`], [Returns bool],
)

=== Unary Operations

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Operation*], [*C Expression*], [*Notes*],
  ),
  [`asl.expr.unop.not` (bits)], [`(~operand) & MASK(w)`], [Bitwise complement],
  [`asl.expr.unop.neg` (bits)], [`(-operand) & MASK(w)`], [Two's complement negation],
)

=== Shift Operations

Shift operations on bitvectors have special semantics:

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Operation*], [*C Expression*], [*Notes*],
  ),
  [`asl.expr.binop.shl` (bits)], [`(lhs << rhs) & MASK(w)`], [Left shift, mask result],
  [`asl.expr.binop.shr` (bits)], [`lhs >> rhs`], [Logical right shift (unsigned)],
  [`asl.expr.binop.asr` (bits)], [Sign-extend then shift], [Arithmetic right shift],
)

*Arithmetic right shift implementation:*
- If MSB is 0: same as logical right shift
- If MSB is 1: `(lhs >> rhs) | (MASK(w) << (w - rhs))`

=== Slice Operations

*ASL:* `asl.expr.slice` - Extract a contiguous range of bits

*Parameters:* base bitvector, low index, width

*C Expression:* `(base >> low) & MASK(width)`

=== Concatenation

*ASL:* `asl.expr.binop.concat` - Concatenate two bitvectors

*C Expression:* `(high << low_width) | low`

*Result width:* `high_width + low_width`

=== Replication

*ASL:* `asl.expr.replicate` - Replicate a bitvector N times

*Implementation:* Loop or unroll to concatenate N copies

=== Large Bitvector Operations (> 64 bits)

For bitvectors wider than 64 bits, operations are performed word-by-word.

*Addition:* Propagate carry through words

*Bitwise ops:* Apply to each word independently

*Shifts:* Multi-word shift with carry between words

*Comparison:* Compare words from most significant to least significant

#line(length: 100%)

== Boolean Operations

=== Literal Lowering

*ASL:* `asl.expr.literal.bool`

#table(
  columns: (auto, auto),
  align: (left, left),
  table.header(
    [*Value*], [*C Expression*],
  ),
  [true], [`true`],
  [false], [`false`],
)

Requires `#include <stdbool.h>`

=== Binary Operations

#table(
  columns: (auto, auto, auto),
  align: (left, left, left),
  table.header(
    [*ASL Operation*], [*C Expression*], [*Notes*],
  ),
  [`asl.expr.binop.band`], [`lhs && rhs`], [Short-circuit AND],
  [`asl.expr.binop.bor`], [`lhs || rhs`], [Short-circuit OR],
  [`asl.expr.binop.beq`], [`lhs == rhs`], [Boolean equality],
  [`asl.expr.binop.impl`], [`!lhs || rhs`], [Implication],
)

=== Unary Operations

#table(
  columns: (auto, auto),
  align: (left, left),
  table.header(
    [*ASL Operation*], [*C Expression*],
  ),
  [`asl.expr.unop.bnot`], [`!operand`],
)

#line(length: 100%)

== String Operations

=== Literal Lowering

*ASL:* `asl.expr.literal.string`

*C Expression:* `"string content"` with proper escaping

=== Comparison

*ASL:* `asl.expr.binop.eq` (string)

*C Expression:* `strcmp(lhs, rhs) == 0`

Requires `#include <string.h>`

#line(length: 100%)

== Control Flow

=== Conditional Statement

*ASL:* `asl.stmt.cond`

*Structure:*
- Condition expression (bool)
- Then block
- Optional else block

*Lowering:* `emitc.if` or direct C `if/else`

=== For Loop

*ASL:* `asl.stmt.for`

*Structure:*
- Loop variable name
- Start expression (integer)
- Direction (up/down)
- End expression (integer)
- Body block

*Lowering considerations:*
- If bounds are well-constrained natives: use C `for` loop with native int
- If bounds are GMP integers: use `mpz_t` loop variable with `mpz_cmp` condition

*Native integer pattern:*
```c
for (int64_t var = start; var <= end; var++) { body }
```

*GMP integer pattern:*
```c
mpz_set(var, start);
while (mpz_cmp(var, end) <= 0) {
    body;
    mpz_add_ui(var, var, 1);
}
```

=== While Loop

*ASL:* `asl.stmt.while`

*Structure:*
- Condition expression (bool)
- Body block

*Lowering:* `while (condition) { body }`

=== Repeat Loop

*ASL:* `asl.stmt.repeat`

*Structure:*
- Body block
- Until condition expression (bool)

*Lowering:* `do { body } while (!(condition));`

=== Return Statement

*ASL:* `asl.stmt.return`

*Lowering:* `return value;` or `return;` for void functions

=== Assert Statement

*ASL:* `asl.stmt.assert`

*Lowering:* `assert(condition);` or custom assertion with message

Requires `#include <assert.h>`

=== Case Statement

*ASL:* `asl.stmt.case`

*Structure:*
- Discriminant expression
- List of (pattern, body) pairs
- Optional otherwise block

*Lowering:* `switch` statement for integer/enum discriminants, or `if/else if` chain for complex patterns

#line(length: 100%)

== Function Declarations

=== Function Signature

*ASL:* `asl.func`

*Structure:*
- Function name
- Parameters with types
- Return type
- Body block

=== Parameter Passing

#table(
  columns: (auto, auto),
  align: (left, left),
  table.header(
    [*ASL Type*], [*C Passing Convention*],
  ),
  [`!asl.bits<N>` (N <= 64)], [By value],
  [`!asl.bits<N>` (N > 64)], [By pointer],
  [`!gmp.z`], [By pointer (mpz_t)],
  [`!gmp.q`], [By pointer (mpq_t)],
  [`!asl.tuple`], [By pointer],
  [`!asl.record`], [By pointer],
  [`!asl.array`], [By pointer],
  [`i1` (bool)], [By value],
)

=== Return Value Handling

#table(
  columns: (auto, auto),
  align: (left, left),
  table.header(
    [*Return Type*], [*C Convention*],
  ),
  [`!asl.bits<N>` (N <= 64)], [Return by value],
  [`!asl.bits<N>` (N > 64)], [Out parameter pointer],
  [`!gmp.z`], [Out parameter pointer],
  [`!gmp.q`], [Out parameter pointer],
  [`!asl.tuple`], [Return by value or out parameter],
  [void], [No return value],
)

=== Context Parameter

All functions receive a context pointer as the first parameter for accessing:
- Global variables
- Exception handling state (jmp_buf)
- Allocated resources

*Pattern:* `ReturnType func_name(module_context* ctx, params...)`

#line(length: 100%)

== Data Access Operations

=== Array Access

*ASL:* `asl.expr.get_array`

*Structure:*
- Base array expression
- Index expression

*Lowering:* `base[index]`

*Index type considerations:*
- Native integer index: direct array access
- GMP integer index: extract to native with `mpz_get_ui`

=== Field Access

*ASL:* `asl.expr.get_field`

*Structure:*
- Base record/struct expression
- Field name

*Lowering:* `base.field` or `base->field` for pointers

=== Tuple Item Access

*ASL:* `asl.expr.get_item`

*Structure:*
- Base tuple expression
- Item index (compile-time constant)

*Lowering:* `base.item<N>` where N is the index

=== Slice Access

*ASL:* `asl.expr.slice`

*Structure:*
- Base bitvector expression
- Low bit index
- Width

*Lowering:* `(base >> low) & MASK(width)`

#line(length: 100%)

== L-Expressions

L-expressions represent assignable locations (left-hand side of assignments).

=== Variable Reference

*ASL:* `asl.lexpr.var`

*Lowering:* Variable name directly

=== Field Assignment

*ASL:* `asl.lexpr.field`

*Lowering:* `base.field = value` or `base->field = value`

=== Array Element Assignment

*ASL:* `asl.lexpr.array`

*Lowering:* `base[index] = value`

=== Tuple Item Assignment

*ASL:* `asl.lexpr.item`

*Lowering:* `base.item<N> = value`

=== Slice Assignment

*ASL:* `asl.lexpr.slice`

Assigning to a slice of a bitvector requires read-modify-write:

*Pattern:*
+ Clear the target bits: `base &= ~(MASK(width) << low)`
+ Set the new bits: `base |= (value & MASK(width)) << low`

=== Concatenation L-Expression

*ASL:* `asl.lexpr.concat`

Assigning to a concatenation of l-expressions distributes the value:

*Pattern:* Split the assigned value and assign parts to each l-expression component

#line(length: 100%)

== Exception Handling

ASL exceptions are implemented using `setjmp`/`longjmp`.

=== Exception Type

Each exception type has a unique integer code.

*Context structure includes:*
- `jmp_buf exception_env` - Jump buffer for exception handling
- `int exception_code` - Current exception code
- Exception-specific data fields

=== Throw Statement

*ASL:* `asl.stmt.throw`

*Lowering:*
+ Set exception code in context
+ Store exception data if any
+ `longjmp(ctx->exception_env, exception_code)`

=== Try-Catch Statement

*ASL:* `asl.stmt.try`

*Structure:*
- Try block
- List of catch clauses (exception type, handler block)
- Optional otherwise clause

*Lowering pattern:*
+ Save current exception environment
+ `if (setjmp(ctx->exception_env) == 0) { try_block }`
+ `else { switch(ctx->exception_code) { catch_handlers } }`
+ Restore previous exception environment

=== Catch Clause Matching

Each catch clause matches against exception type codes.

*Pattern:* `case EXCEPTION_TYPE_CODE: handler; break;`

#line(length: 100%)

== Global State Management

=== Module Context Structure

Each ASL module generates a context structure containing:
- All global variables
- Exception handling state
- Allocated resource tracking

*Pattern:* `typedef struct { globals; jmp_buf exception_env; ... } module_context;`

=== Initialization Function

*Pattern:* `void module_init(module_context* ctx)`

Responsibilities:
- Zero-initialize the context
- Initialize GMP variables (`mpz_init`, `mpq_init`)
- Set up exception environment
- Call per-variable initializers

=== Per-Variable Initializers

For globals with non-trivial initial values:

*Pattern:* `void module_init_varname(module_context* ctx)`

=== Cleanup Function

*Pattern:* `void module_free(module_context* ctx)`

Responsibilities:
- Clear GMP variables (`mpz_clear`, `mpq_clear`)
- Free dynamically allocated memory

=== Variable Access

Global variables are accessed through the context pointer:

*Pattern:* `ctx->variable_name`

#line(length: 100%)

== Runtime Support

=== Required Headers

```c
#include <stdint.h>    // uint8_t, uint16_t, uint32_t, uint64_t, int64_t
#include <stdbool.h>   // bool, true, false
#include <string.h>    // strcmp, memcpy, memset
#include <setjmp.h>    // jmp_buf, setjmp, longjmp
#include <assert.h>    // assert
#include <gmp.h>       // mpz_t, mpq_t (for GMP operations)
```

=== Runtime Helper Functions

The generated code may require helper functions for complex operations:

*Large bitvector operations:*
- `asl_bits_add` - Multi-word addition
- `asl_bits_sub` - Multi-word subtraction
- `asl_bits_and` - Multi-word AND
- `asl_bits_or` - Multi-word OR
- `asl_bits_xor` - Multi-word XOR
- `asl_bits_not` - Multi-word NOT
- `asl_bits_shl` - Multi-word left shift
- `asl_bits_shr` - Multi-word right shift
- `asl_bits_eq` - Multi-word equality
- `asl_bits_slice` - Extract slice from multi-word
- `asl_bits_concat` - Concatenate multi-word bitvectors

*Conversion functions:*
- `asl_bits_to_mpz` - Convert bitvector to GMP integer
- `asl_mpz_to_bits` - Convert GMP integer to bitvector

=== Mask Macros

```c
#define ASL_MASK_8  ((uint8_t)0xFF)
#define ASL_MASK_16 ((uint16_t)0xFFFF)
#define ASL_MASK_32 ((uint32_t)0xFFFFFFFF)
#define ASL_MASK_64 ((uint64_t)0xFFFFFFFFFFFFFFFF)
#define ASL_MASK(w) ((1ULL << (w)) - 1)  // For w <= 64
```

#line(length: 100%)

== Testing Strategy

=== Unit Tests

Test files in `asl-mlir/test/ASL/Transforms/asl-to-emitc/`:

+ *Type Lowering* (`types.mlir`)
  - Verify each ASL type maps to correct C type
  - Test large bitvector struct generation
  - Test enum, tuple, record generation

+ *Bitvector Operations* (`bitvector-ops.mlir`)
  - Test all binary operations with various widths
  - Verify masking behavior
  - Test large bitvector operations

+ *Control Flow* (`control-flow.mlir`)
  - Test if/else lowering
  - Test for loop with native and GMP bounds
  - Test while, repeat loops
  - Test case/switch statements

+ *Function Declarations* (`functions.mlir`)
  - Test parameter passing conventions
  - Test return value handling
  - Test context parameter injection

+ *Exception Handling* (`exceptions.mlir`)
  - Test throw statement
  - Test try-catch blocks
  - Test exception propagation

=== Integration Tests

End-to-end tests compiling ASL to C and executing:

+ Compile ASL source through full pipeline
+ Compile generated C with gcc/clang
+ Execute and verify output
+ Check memory correctness with valgrind/asan

#line(length: 100%)

== References

+ *Intel ASL Interpreter*: https://github.com/alastairreid/asl-interpreter
  - `libASL/backend_c.ml` - C code generation patterns

+ *ARM herdtools7 asllib*: Official ASL reference
  - `asllib/Operations.ml` - Operation semantics

+ *MLIR EmitC Dialect*: https://mlir.llvm.org/docs/Dialects/EmitC/

+ *GMP Manual*: https://gmplib.org/manual/
