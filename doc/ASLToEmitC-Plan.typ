= ASL to EmitC Lowering Plan

== Current State

The `ASLToEmitC.cpp` pass (4289 lines) currently handles:

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
- `StmtForOp` -> `scf.while` with GMP comparisons:
  - Creates loop index variable (mpz_t)
  - Initializes with start value (mpz_init_set)
  - Compares with end value (mpz_cmp)
  - Increments/decrements based on direction (mpz_add_ui/mpz_sub_ui)
  - Cleans up index (mpz_clear)

==== Not Yet Implemented
- Loop limit handling (LimitExceeded exception)
- VarOp resolution for loop index variable (requires name resolution pass)

=== Phase 5: Function Declarations (IMPLEMENTED)

==== Implemented Patterns
- `FuncDeclOp` -> `func.func` with converted types
  - Skips primitive functions (erases them)
  - Converts argument types via type converter
  - Converts return type
  - Uses `inlineRegionBefore` + `convertRegionTypes` for body
- `CallOp` -> `emitc.call_opaque` (returns value)
- `StmtCallOp` -> `emitc.call_opaque` (void, discards result)
- `AtcOp` -> pass-through (type converter handles conversion)

==== Type Materializations
- Source/target materializations handle lvalue-to-value conversion
- Generates `emitc.load` when converting `!emitc.lvalue<T>` to `T`

==== Not Yet Implemented
- Context pointer for global variable access (requires globals analysis)
- Procedure return handling (void functions)

=== Phase 6: Data Structures (IMPLEMENTED)

==== Implemented Patterns
- `TupleOp` -> `emitc.variable` + `emitc.member` for field initialization
  - Uses `emitc.member` to access struct fields
  - Uses `emitc.assign` for simple types, GMP init/set for GMP types
- `GetItemOp` -> `emitc.member` + `emitc.load` for tuple field access
- `RecordOp` -> `emitc.variable` + `emitc.member` for field initialization
- `GetFieldOp` -> `emitc.member` + `emitc.load` for record field access
- `GetArrayOp` -> `emitc.subscript` with `mpz_get_si` for index conversion
- `GetEnumArrayOp` -> `emitc.subscript` with enum cast to int
- `GetFieldsOp` -> concatenates fields via shift and OR for bit-packing
- `ArrayOp` -> `emitc.variable` (simplified, without initialization loop)
- `EnumArrayOp` -> `emitc.variable` (simplified, without initialization)
- `PragmaDeclOp` -> erased (tool-specific hints)
- `GlobalStorageDeclOp` -> erased (handled during context generation)

==== Not Yet Implemented
- Full array initialization with fill value (requires loop generation)
- Full enum array initialization (requires loop over enum values)

=== Phase 7: Slicing Operations (IMPLEMENTED)

==== Slice Type Representation
- `!asl.slice` -> `struct { long start; long length; }` (descriptor)

==== Implemented Patterns
- `SliceSingleOp` -> creates slice descriptor with start=i, length=1
- `SliceRangeOp` -> creates slice descriptor with start=j, length=i-j
- `SliceLengthOp` -> creates slice descriptor with start=i, length=n
- `SliceStarOp` -> creates slice descriptor with start=factor*length, length=n
- `SliceOp` -> applies slices to bitvector:
  - Single slice: `(base >> start) & ((1ULL << length) - 1)`
  - Multiple slices: concatenated via shift and OR

=== Phase 8: L-Expressions (IMPLEMENTED)

==== L-Expression Type Representation
- `!asl.lexpr` -> `void*` (type-erased pointer to assignable location)

==== Implemented Patterns
- `LExprDiscardOp` -> NULL pointer (assignment ignores value)
- `LExprVarOp` -> `&variable_name` (address of named variable)
- `LExprSetFieldOp` -> `&(base->field)` via call_opaque
- `LExprSetArrayOp` -> `&(base[index])` via call_opaque
- `LExprDestructuringOp` -> struct of void* pointers for tuple elements
- `LExprSliceOp` -> slice descriptor with base, start, length
- `LExprSetEnumArrayOp` -> `&(base[enum_index])` with enum-to-int cast
- `LExprSetFieldsOp` -> returns base (bit-packing needs runtime support)
- `LExprSetCollectionFieldsOp` -> NULL (requires runtime support)

=== Phase 9: Assignment and Declaration (IMPLEMENTED)

==== Implemented Patterns
- `StmtDeclOp` -> `emitc.variable` with:
  - GMP types: `mpz_init`/`mpz_init_set` or `mpq_init`/`mpq_set`
  - Simple types: `emitc.assign`
- `StmtAssignOp` -> assignment via:
  - Discard check (NULL pointer skips assignment)
  - GMP types: `mpz_set` or `mpq_set`
  - Simple types: pointer dereference assignment

=== Phase 10: Pattern Matching (IMPLEMENTED)

==== Implemented Patterns
- `PatternAllOp` -> always returns `true`
- `PatternSingleOp` -> equality comparison (mpz_cmp for GMP, == for simple)
- `PatternRangeOp` -> range check: `lower <= expr && expr <= upper`
- `PatternGeqOp` -> greater-or-equal check
- `PatternLeqOp` -> less-or-equal check
- `PatternMaskOp` -> bitvector mask: `(expr & mask) == value`
- `PatternNotOp` -> negates nested pattern result
- `PatternAnyOp` -> OR of all nested pattern results (disjunction)
- `PatternTupleOp` -> AND of all nested pattern results (element-wise matching)
- `PatternOp` -> evaluates nested pattern and returns result

=== Phase 11: Type Conversions (ATC) (IMPLEMENTED)

==== Implemented Patterns
- `AtcOp` -> pass-through (compile-time type assertion)
- `AtcIntOp` -> pass-through (constraint region erased)
- `AtcIntExactOp` -> pass-through (exact constraint compile-time)
- `AtcIntRangeOp` -> pass-through (range constraint compile-time)
- `AtcBitsOp` -> pass-through (width/bitfield specs compile-time)
- `AtcArrayOp` -> pass-through (length constraint compile-time)
- `AtcBitsBitfieldsSimpleOp` -> zero-initialized bitvector
- `AtcBitsBitfieldsNestedOp` -> zero-initialized bitvector
- `AtcBitsBitfieldsTypeOp` -> zero-initialized bitvector

==== Notes
ATC operations are primarily compile-time constraints. At runtime, the type
converter handles the actual type conversion and the operation becomes a
pass-through.

=== Phase 12: Exception Handling (IMPLEMENTED - SIMPLIFIED)

==== Implemented Patterns
- `StmtThrowOp` -> `abort()` call (terminates program)
- `StmtTryOp` -> inlines protected block, ignores handlers

==== Notes
This is a simplified implementation that does not provide true exception handling.
A proper implementation would require:
- Runtime support for setjmp/longjmp
- Exception type discrimination
- Handler dispatch logic

=== Phase 13: Miscellaneous (IMPLEMENTED)

==== Implemented Patterns
- `StmtAssertOp` -> runtime assertion (IMPLEMENTED in Phase 4)
- `StmtUnreachableOp` -> `__builtin_unreachable()` (IMPLEMENTED in Phase 4)
- `StmtPrintOp` -> `printf`/`gmp_printf` with type-based format strings
- `StmtPragmaOp` -> erased (tool-specific hints ignored)
- `ArbitraryOp` -> zero-initialized value (implementation-defined)

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

== Implementation Status

Current implementation: 5055 lines in `ASLToEmitC.cpp`

=== Completed Phases
- Phase 1: Literals and Simple Expressions
- Phase 2: Binary Operations
- Phase 3: Unary Operations
- Phase 4: Control Flow
- Phase 5: Function Declarations
- Phase 6: Data Structures
- Phase 7: Slicing Operations
- Phase 8: L-Expressions
- Phase 9: Assignment and Declaration
- Phase 10: Pattern Matching
- Phase 11: Type Conversions (ATC)
- Phase 12: Exception Handling (simplified)
- Phase 13: Miscellaneous

=== Remaining Work
- Phase 4: Loop limit handling (LimitExceeded exception)
- Phase 5: Context pointer for globals, procedure return
- Phase 6: Full array/enum array initialization with loops
- Phase 8: Full bit-packing support for LExprSetFieldsOp, LExprSetCollectionFieldsOp
- Phase 12: Full setjmp/longjmp exception handling (runtime support)

=== Recently Completed
- VarOp name resolution: VarOpLowering pattern generates `emitc.call_opaque`
  with the variable name. JSONImporter now correctly creates separate VarOps
  for each mutable variable reference instead of reusing initial values.

Consider splitting into multiple files:
- `ASLToEmitC.cpp` - pass infrastructure, type converter
- `ASLToEmitCPatterns.cpp` - conversion patterns
- `ASLToEmitCHelpers.cpp` - helper function generation
