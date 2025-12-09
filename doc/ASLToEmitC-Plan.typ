= ASL to EmitC Lowering Plan

== Current State

The `ASLToEmitC.cpp` pass (1291 lines) currently handles:

=== Type Conversions (Implemented)
- `!asl.int` -> `mpz_t` (GMP arbitrary-precision integers)
- `!asl.real` -> `mpq_t` (GMP exact rationals)
- `!asl.bool` (i1) -> `bool`
- `!asl.bits<N>`:
  - N <= 8 -> `uint8_t`
  - N <= 16 -> `uint16_t`
  - N <= 32 -> `uint32_t`
  - N <= 64 -> `uint64_t`
  - N > 64 -> `struct { uint64_t words[N/64+1]; }`
- `!asl.string` -> `const char*`
- `!asl.enum` -> `int` (with generated enum typedefs)
- `!asl.label` -> `int`
- `!asl.tuple` -> C struct with `itemN` fields
- `!asl.named` -> resolves to underlying type

=== Operation Patterns (Implemented)
1. `ConstantInitGlobalStorageDeclOp` - erases (handled via verbatim generation)
2. `LiteralStringOp` -> `emitc.constant`
3. `LiteralBitvectorOp` -> `emitc.constant`
4. `LiteralLabelOp` -> `emitc.constant`
5. `LiteralIntOp` -> `gmp.z.init_set_str` via GMP dialect
6. `LiteralBoolOp` -> `emitc.constant` with `true`/`false`
7. `LiteralRealOp` -> `gmp.q.init` + `gmp.q.set_str` via GMP dialect
8. `TypeDeclOp` - erases (typedef generated via verbatim)
9. `TupleOp` -> `emitc.variable` with field initialization

=== Binary Operations (Phase 2 - Implemented)
==== Integer Operations (GMP)
- `BinopIntAddOp` -> `mpz_add`
- `BinopIntSubOp` -> `mpz_sub`
- `BinopIntMulOp` -> `mpz_mul`
- `BinopDivOp` -> `mpz_tdiv_q` (truncated division)
- `BinopDivrmOp` -> `mpz_fdiv_q` (floor division)
- `BinopModOp` -> `mpz_fdiv_r` (floor remainder)
- `BinopPowOp` -> `mpz_pow_ui`
- `BinopShlOp` -> `mpz_mul_2exp`
- `BinopShrOp` -> `mpz_fdiv_q_2exp`

==== Bitvector Operations (native C with masks)
- `BinopBitsAddOp` -> `+`
- `BinopBitsSubOp` -> `-`
- `BinopBitsMulOp` -> `*`
- `BinopAndOp` -> `&`
- `BinopOrOp` -> `|`
- `BinopXorOp` -> `^`
- `BinopConcatOp` -> shift and OR

==== Real Operations (GMP)
- `BinopRealAddOp` -> `mpq_add`
- `BinopRealSubOp` -> `mpq_sub`
- `BinopRealMulOp` -> `mpq_mul`
- `BinopRdivOp` -> `mpq_div`

==== Boolean Operations (native C)
- `BinopBandOp` -> `&&`
- `BinopBorOp` -> `||`
- `BinopBeqOp` -> `==`
- `BinopImplOp` -> `!lhs || rhs`

==== Comparison Operations
- `BinopEqOp` -> `mpz_cmp(...) == 0`
- `BinopNeqOp` -> `mpz_cmp(...) != 0`
- `BinopLtOp` -> `mpz_cmp(...) < 0`
- `BinopLeqOp` -> `mpz_cmp(...) <= 0`
- `BinopGtOp` -> `mpz_cmp(...) > 0`
- `BinopGeqOp` -> `mpz_cmp(...) >= 0`

=== Code Generation (Implemented)
- Includes: `gmp.h`, `stdint.h`, `stdbool.h`, `string.h`
- Context struct typedef with all global variables
- Per-variable init functions (handles GMP init, tuple recursion)
- Main init function calling all per-variable inits
- Free function with GMP cleanup

== Missing Operations

=== Phase 1: Literals and Simple Expressions (COMPLETED)
See "Operation Patterns (Implemented)" section above.

Remaining:
- *Variable References* (`VarOp`)
  - Strategy: Convert to SSA value lookup or context field access
  - Local vars: direct SSA
  - Global vars: `ctx->varName`

=== Phase 2: Binary Operations (COMPLETED)
See "Binary Operations (Phase 2 - Implemented)" section above.

=== Phase 3: Unary Operations (COMPLETED)
- `UnopBnotOp` -> `!` (boolean)
- `UnopNegOp` -> `mpz_neg` (integer) or `mpq_neg` (real)
- `UnopNotOp` -> `~` with mask (bitvector)

=== Phase 4: Control Flow (IMPLEMENTED)
Note: Testing requires Phase 5 (Function Declarations) to be completed.

==== Implemented Patterns
- `CondOp` -> `emitc.conditional` (ternary operator)
- `StmtCondOp` -> `scf.if` with then/else regions
- `StmtReturnOp` -> `func.return`
- `StmtPassOp` -> erased (no-op)
- `StmtSeqOp` -> inlined body operations
- `StmtAssertOp` -> `assert()` call
- `StmtUnreachableOp` -> `__builtin_unreachable()`
- `StmtWhileOp` -> `scf.while` (without limit support)
- `StmtRepeatOp` -> `scf.while` with do-while semantics (without limit support)
- `StmtForOp` -> stub (GMP integer bounds require complex lowering)

==== Not Yet Implemented
- For loops with GMP bounds (requires conversion to while loop with GMP comparisons)
- Loop limit handling (LimitExceeded exception)

=== Phase 5: Function Declarations
Priority: High

1. *Function Declaration* (`FuncDeclOp`)
   - Generate function signature with converted types
   - Handle parameters and return types
   - Convert body region
   - Add context pointer parameter for globals access

2. *Function Call* (`CallOp`, `StmtCallOp`)
   - Convert argument types
   - Handle return value
   - Pass context pointer

=== Phase 6: Data Structures
Priority: Medium

==== Tuple Operations
- `GetItemOp` -> `.itemN` field access
- `TupleOp` -> struct initialization (already done)

==== Array Operations
- `ArrayOp` -> heap allocation or stack array
- `GetArrayOp` -> indexed access
- `GetEnumArrayOp` -> enum-indexed access
- `EnumArrayOp` -> enum-keyed array construction

==== Record Operations
- `RecordOp` -> struct initialization
- `GetFieldOp` -> `.fieldName` access
- `GetFieldsOp` -> multiple field access for bit-packing

=== Phase 7: Slicing Operations
Priority: Medium

- `SliceSingleOp` -> single bit extraction
- `SliceRangeOp` -> bit range extraction
- `SliceLengthOp` -> length-based extraction
- `SliceStarOp` -> factor-based extraction
- `SliceOp` (expression) -> apply slices to bitvector

=== Phase 8: L-Expressions (Assignment Targets)
Priority: Medium (needed for assignment)

1. *Variable L-Expression* (`LExprVarOp`)
   - Output: lvalue reference to variable

2. *Slice L-Expression* (`LExprSliceOp`)
   - Output: bitfield write

3. *Array Element L-Expression* (`LExprSetArrayOp`)
   - Output: array element lvalue

4. *Field L-Expression* (`LExprSetFieldOp`)
   - Output: struct field lvalue

5. *Destructuring L-Expression* (`LExprDestructuringOp`)
   - Output: tuple element assignments

6. *Discard L-Expression* (`LExprDiscardOp`)
   - Output: nothing (value discarded)

=== Phase 9: Assignment and Declaration
Priority: Medium

1. *Local Declaration* (`StmtDeclOp`)
   - Output: `emitc.variable` with optional initializer

2. *Assignment* (`StmtAssignOp`)
   - Input: l-expression, value
   - Output: store operation

=== Phase 10: Pattern Matching
Priority: Low (complex)

- `PatternAllOp` -> always true
- `PatternSingleOp` -> equality check
- `PatternRangeOp` -> range check
- `PatternGeqOp`, `PatternLeqOp` -> bound check
- `PatternMaskOp` -> bitvector mask check
- `PatternNotOp` -> negation
- `PatternAnyOp` -> disjunction
- `PatternTupleOp` -> element-wise matching
- `PatternOp` -> pattern application

=== Phase 11: Type Conversions (ATC)
Priority: Low (constraint checking)

- `AtcOp` -> type assertion/conversion
- `AtcIntOp` -> integer constraint
- `AtcIntExactOp` -> exact value constraint
- `AtcIntRangeOp` -> range constraint
- `AtcBitsOp` -> bitvector width/bitfield conversion
- `AtcArrayOp` -> array length constraint

=== Phase 12: Exception Handling
Priority: Low (complex runtime)

1. *Throw Statement* (`StmtThrowOp`)
   - Strategy: `setjmp`/`longjmp` based
   - Output: `longjmp` call

2. *Try Statement* (`StmtTryOp`)
   - Output: `setjmp` setup, handler dispatch

=== Phase 13: Miscellaneous
Priority: Low

- `StmtAssertOp` -> runtime assertion (IMPLEMENTED in Phase 4)
- `StmtPrintOp` -> `printf` calls
- `StmtUnreachableOp` -> `__builtin_unreachable()` (IMPLEMENTED in Phase 4)
- `StmtPragmaOp` -> pass-through or ignore
- `ArbitraryOp` -> undefined value (0 or random)
- `PragmaDeclOp` -> ignore

== Architecture Decisions

=== Strategy: Two-Phase Lowering
1. ASL -> ASL + GMP (convert int/real ops to GMP dialect)
2. ASL + GMP -> EmitC (final lowering)

Advantage: Reuse existing GMP->EmitC pass


=== Memory Management

1. *Value Semantics*: ASL uses value semantics
   - All copies are deep copies
   - GMP types need `mpz_init_set` / `mpq_set`

2. *Temporary Variables*: GMP operations need temporaries
   - Create `emitc.variable` for each intermediate result
   - Initialize with `mpz_init` / `mpq_init`
   - Clean up with `mpz_clear` / `mpq_clear` (or rely on scope)

3. *Context Lifetime*: User manages context
   - `asl_init(ctx)` at start
   - `asl_free(ctx)` at end

=== Large Bitvector Handling

For bitvectors > 64 bits:
- Struct with `uint64_t words[]` array
- Operations implemented as helper functions:
  - `asl_bits_and`, `asl_bits_or`, `asl_bits_xor`
  - `asl_bits_add`, `asl_bits_sub`
  - `asl_bits_shift_left`, `asl_bits_shift_right`
- Alternative: Convert to GMP for large bitvectors

== Testing Strategy

1. Unit tests per operation type in `asl-mlir/test/Passes/`
2. FileCheck patterns for expected C output
3. Integration tests compiling generated C code

== Implementation Order

1. Literals (Int, Bool, Real) - foundation
2. Binary ops (arithmetic first, then comparison)
3. Unary ops
4. Control flow (if, for, while)
5. Function decl/call
6. Local variables and assignment
7. Data structure access (tuple, array, record)
8. Slicing
9. Pattern matching
10. ATC (type conversion)
11. Exception handling

== Estimated Scope

- Phase 1-3: ~500 lines (literals + basic ops)
- Phase 4-5: ~400 lines (control flow + functions)
- Phase 6-8: ~600 lines (data structures + l-exprs)
- Phase 9-13: ~500 lines (remaining features)

Total: ~2000 additional lines, bringing ASLToEmitC.cpp to ~3300 lines.

Consider splitting into multiple files:
- `ASLToEmitC.cpp` - pass infrastructure, type converter
- `ASLToEmitCPatterns.cpp` - conversion patterns
- `ASLToEmitCHelpers.cpp` - helper function generation
