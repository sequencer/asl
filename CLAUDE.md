# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASL-MLIR is a compiler for Arm Specification Language (ASL) that uses LLVM/MLIR infrastructure. The project compiles ASL source code to C via MLIR's EmitC dialect.

## Build System

This project uses Nix for development. Enter the development shell with:
```bash
nix develop
```

### Building asl-mlir
```bash
cd asl-mlir
cmake -B build -G Ninja -DMLIR_DIR=$(nix path-info .#legacyPackages.$(uname -m)-linux.asl-llvm)/lib/cmake/mlir
ninja -C build
```

Or build via Nix:
```bash
nix build .#asl-mlir
```

### Running Tests
```bash
ninja -C asl-mlir/build check-asl-opt
```

Tests use LLVM's `lit` test runner. Test files are `.asl` or `.mlir` files with `// RUN:` directives. Tests are in `asl-mlir/test/`.

## Architecture

### Compilation Pipeline
1. **ASL Source** (.asl) -> **asl-json-backend** (OCaml) -> **JSON AST**
2. **JSON AST** -> **asl-opt** (C++/MLIR) -> **ASL Dialect MLIR**
3. **ASL Dialect** -> **ASLToEmitC pass** -> **EmitC Dialect** -> **C code**

### Key Components

**asl-json-backend** (`asl-json-backend/`): OCaml tool that parses ASL and outputs JSON AST. Uses `asllib` from ARM's herdtools7.

**asl-opt** (`asl-mlir/asl-opt/`): Main compiler driver. Key files:
- `asl-opt.cpp` - CLI entry point with options: `--json-input`, `--canonicalize`, `--run-asl-to-emitc`, `--emitc`
- `JSONImporter.cpp` - Converts JSON AST to ASL dialect MLIR

**ASL Dialect** (`asl-mlir/include/ASL/`, `asl-mlir/lib/ASL/IR/`):
- TableGen definitions (.td) for ops, types, and attributes
- Types: `BitsType`, `IntType`, `RealType`, `StringType`, `TupleType`, `ArrayType`, `RecordType`, `EnumType`, `ExceptionType`
- Operations grouped by: expressions, statements, declarations, l-expressions

**ASLToEmitC Pass** (`asl-mlir/lib/ASL/Pass/ASLToEmitC.cpp`): Lowers ASL dialect to EmitC. Handles:
- Type conversion (bits -> uint, int -> mpz_t, real -> mpq_t)
- Global state encapsulation in `asl_context` struct
- Exception handling via setjmp/longjmp

### Example Usage
```bash
# Full pipeline: ASL -> C
asl-json-backend --no-std input.asl > input.json
asl-opt --canonicalize --run-asl-to-emitc --emitc --json-input input.json
```

## Code Style

- C++: LLVM style (`.clang-format` with LLVM base)
- Naming: CamelCase for classes/enums, camelBack for functions/members/variables
- Use `nix fmt` for Nix and Scala formatting

## TableGen Files

ASL dialect is defined in `asl-mlir/include/ASL/*.td`:
- `ASLDialect.td` - Dialect definition
- `ASLTypes.td` - Type definitions
- `ASLAttributes.td` - Attribute definitions
- `ASLExpressions.td`, `ASLStatement.td`, `ASLDeclarations.td`, `ASLLExpression.td` - Operations

## GMP Dialect Plan

A dedicated MLIR dialect for arbitrary-precision integer operations that maps to GMP at runtime and enables compile-time constant propagation.

### Motivation

ASL integers are unbounded (arbitrary precision), requiring:
1. **Runtime**: GMP `mpz_t` for correct semantics
2. **Compile-time**: Constant folding in MLIR to optimize away known values
3. **Clean separation**: Integer operations distinct from ASL-specific semantics

### Dialect Design

**Dialect name**: `gmp`

**Types**:
```tablegen
def GMP_IntType : TypeDef<GMP_Dialect, "Int", []> {
  let summary = "Arbitrary precision integer type";
  let description = [{
    Represents an unbounded integer. At compile-time, constants are
    represented using MLIR's APInt. At runtime, lowered to GMP mpz_t.
  }];
}
```

**Attributes**:
```tablegen
def GMP_ValueAttr : AttrDef<GMP_Dialect, "Value", []> {
  let summary = "Arbitrary precision integer constant";
  let parameters = (ins "std::string":$value);  // Decimal string representation
}
```

**Operations**:
| Operation | Description | GMP Lowering |
|-----------|-------------|--------------|
| `gmp.constant` | Integer literal | `mpz_init_set_str` |
| `gmp.add` | Addition | `mpz_add` |
| `gmp.sub` | Subtraction | `mpz_sub` |
| `gmp.mul` | Multiplication | `mpz_mul` |
| `gmp.div` | Truncated division | `mpz_tdiv_q` |
| `gmp.fdiv` | Floor division | `mpz_fdiv_q` |
| `gmp.mod` | Floor remainder | `mpz_fdiv_r` |
| `gmp.neg` | Negation | `mpz_neg` |
| `gmp.abs` | Absolute value | `mpz_abs` |
| `gmp.pow` | Exponentiation | `mpz_pow_ui` |
| `gmp.shl` | Shift left | `mpz_mul_2exp` |
| `gmp.shr` | Shift right | `mpz_fdiv_q_2exp` |
| `gmp.cmp_eq` | Equal | `mpz_cmp == 0` |
| `gmp.cmp_ne` | Not equal | `mpz_cmp != 0` |
| `gmp.cmp_lt` | Less than | `mpz_cmp < 0` |
| `gmp.cmp_le` | Less or equal | `mpz_cmp <= 0` |
| `gmp.cmp_gt` | Greater than | `mpz_cmp > 0` |
| `gmp.cmp_ge` | Greater or equal | `mpz_cmp >= 0` |
| `gmp.to_bits` | Convert to bitvector | `mpz_export` |
| `gmp.from_bits` | Convert from bitvector | `mpz_import` |

### Constant Propagation Pass

**Pass name**: `gmp-constant-prop`

The pass performs compile-time evaluation of integer operations when operands are known constants:

```mlir
// Before constant propagation
%0 = gmp.constant "10"
%1 = gmp.constant "20"
%2 = gmp.add %0, %1 : !gmp.int

// After constant propagation
%2 = gmp.constant "30"
```

**Implementation**:
1. Use LLVM's `APInt` for arbitrary precision arithmetic during compilation
2. Implement `fold()` methods on each operation
3. Store constants as decimal strings (portable, no size limit)
4. Canonicalization patterns for algebraic simplifications:
   - `x + 0` → `x`
   - `x * 1` → `x`
   - `x * 0` → `0`
   - `x - x` → `0`
   - `x / 1` → `x`

### Lowering Pipeline

```
ASL Dialect
    │
    ├─→ [Canonicalize] Fold ASL integer literals to gmp.constant
    │
    ▼
GMP Dialect (with constant propagation)
    │
    ├─→ [gmp-constant-prop] Evaluate compile-time constants
    │
    ▼
EmitC Dialect (GMP function calls)
    │
    ▼
C Code
```

### Example Transformation

**Input ASL**:
```
let x = 2 + 3 * 4;  // Should fold to 14 at compile time
let y = x + n;       // Runtime operation with mpz_add
```

**After ASL → GMP lowering**:
```mlir
%c2 = gmp.constant "2"
%c3 = gmp.constant "3"
%c4 = gmp.constant "4"
%t1 = gmp.mul %c3, %c4 : !gmp.int
%t2 = gmp.add %c2, %t1 : !gmp.int
%y = gmp.add %t2, %n : !gmp.int
```

**After constant propagation**:
```mlir
%c14 = gmp.constant "14"
%y = gmp.add %c14, %n : !gmp.int
```

**After EmitC lowering**:
```c
mpz_t c14, y;
mpz_init_set_str(c14, "14", 10);
mpz_init(y);
mpz_add(y, c14, n);
```

### File Structure

```
asl-mlir/
├── include/GMP/
│   ├── GMPDialect.td      # Dialect definition
│   ├── GMPTypes.td        # Type definitions
│   ├── GMPOps.td          # Operation definitions
│   └── GMPPasses.td       # Pass declarations
├── lib/GMP/
│   ├── IR/
│   │   ├── GMPDialect.cpp
│   │   ├── GMPTypes.cpp
│   │   └── GMPOps.cpp     # With fold() implementations
│   └── Transforms/
│       └── ConstantProp.cpp   # Constant propagation pass
└── test/GMP/
    ├── constant-prop.mlir
    └── to-emitc.mlir
```

### Integration with ASL Dialect

1. **ASL type lowering**: `asl.int<constraint>` → `!gmp.int`
2. **ASL operation lowering**: `asl.expr.binop.plus` (int) → `gmp.add`
3. **Literal lowering**: `asl.expr.literal.int` → `gmp.constant`

## ASLToEmitC Lowering Plan

This section documents the plan for lowering ASL dialect operations to EmitC dialect for C code generation. The implementation follows patterns from the Intel ASL Interpreter and ARM's official `asllib` (herdtools7).

### Reference Documentation

- **Intel ASL Interpreter**: https://github.com/IntelLabs/asl-interpreter
  - `libASL/primops.ml` - Primitive operations implementation
  - `libASL/backend_c.ml` - C backend code generation
  - `libASL/xform_*.ml` - AST transformations
- **ARM herdtools7 asllib**: Official ASL reference implementation
  - `asllib/Operations.ml` - Binary/unary operation semantics
  - `asllib/AST.mli` - AST type definitions
  - `asllib/types.mli` - Type algebra

### Type Lowering

| ASL Type | C Type | Notes |
|----------|--------|-------|
| `asl.int` | `mpz_t` (GMP) | Arbitrary precision integers |
| `asl.real` | `mpq_t` (GMP) | Exact rational numbers (p/q) |
| `asl.bits<N>` (N≤8) | `uint8_t` | Fixed-width bitvector |
| `asl.bits<N>` (N≤16) | `uint16_t` | Fixed-width bitvector |
| `asl.bits<N>` (N≤32) | `uint32_t` | Fixed-width bitvector |
| `asl.bits<N>` (N≤64) | `uint64_t` | Fixed-width bitvector |
| `asl.bits<N>` (N>64) | `struct { uint64_t words[(N+63)/64]; }` | Large bitvector |
| `i1` (bool) | `bool` | C99 stdbool.h |
| `asl.string` | `const char*` | Immutable string literals |
| `asl.enum` | `typedef enum` | Named C enum type |
| `asl.tuple` | `typedef struct` | Struct with itemN fields |
| `asl.array` | C array or struct | Indexed by int or enum |
| `asl.record` | `typedef struct` | Named fields |
| `asl.exception` | `struct` with `jmp_buf` | setjmp/longjmp handling |

### Lowering Phases

#### Phase 1: Type Declarations (Implemented)
- [x] Enum type declarations → `typedef enum`
- [x] Tuple type declarations → `typedef struct`
- [ ] Record type declarations → `typedef struct`
- [ ] Exception type declarations → `typedef struct`

#### Phase 2: Global State Management (Implemented)
- [x] Context struct generation (`<module>_context`)
- [x] Per-variable init functions (`<module>_init_<var>`)
- [x] Main init function (`<module>_init`)
- [x] Free function (`<module>_free`)
- [x] GMP initialization (`mpz_init_set_str`, `mpq_init`)
- [x] GMP cleanup (`mpz_clear`, `mpq_clear`)

#### Phase 3: Literal Expressions (Partial)
- [x] `asl.expr.literal.string` → `emitc.constant` with quoted string
- [x] `asl.expr.literal.bitvector` → `emitc.constant` with integer value
- [x] `asl.expr.literal.label` → enum constant reference
- [ ] `asl.expr.literal.int` → `mpz_init_set_str` call
- [ ] `asl.expr.literal.real` → `mpq_set_str` call
- [ ] `asl.expr.literal.bool` → `true`/`false`

#### Phase 4: Binary Operations (Not Started)
Binary operations need type-specific lowering:

**Integer Operations (mpz_t)**:
- `PLUS` → `mpz_add(result, lhs, rhs)`
- `MINUS` → `mpz_sub(result, lhs, rhs)`
- `MUL` → `mpz_mul(result, lhs, rhs)`
- `DIV` → `mpz_tdiv_q(result, lhs, rhs)` (truncate toward zero)
- `DIVRM` → `mpz_fdiv_q(result, lhs, rhs)` (floor division)
- `MOD` → `mpz_fdiv_r(result, lhs, rhs)` (floor remainder)
- `POW` → `mpz_pow_ui(result, lhs, mpz_get_ui(rhs))`
- `SHL` → `mpz_mul_2exp(result, lhs, mpz_get_ui(rhs))`
- `SHR` → `mpz_fdiv_q_2exp(result, lhs, mpz_get_ui(rhs))`
- Comparisons → `mpz_cmp(lhs, rhs)` with relational check

**Real Operations (mpq_t)**:
- `PLUS` → `mpq_add(result, lhs, rhs)`
- `MINUS` → `mpq_sub(result, lhs, rhs)`
- `MUL` → `mpq_mul(result, lhs, rhs)`
- `RDIV` → `mpq_div(result, lhs, rhs)`
- Comparisons → `mpq_cmp(lhs, rhs)` with relational check

**Bitvector Operations (uintN_t)**:
- `AND` → `lhs & rhs`
- `OR` → `lhs | rhs`
- `XOR` → `lhs ^ rhs`
- `PLUS` → `(lhs + rhs) & mask` (wrap at width)
- `MINUS` → `(lhs - rhs) & mask`
- `MUL` → `(lhs * rhs) & mask`
- `CONCAT` → `(lhs << rhs_width) | rhs`

**Boolean Operations**:
- `BAND` → `lhs && rhs`
- `BOR` → `lhs || rhs`
- `BEQ` → `lhs == rhs`
- `IMPL` → `!lhs || rhs`

#### Phase 5: Unary Operations (Not Started)
- `NEG` (int) → `mpz_neg(result, operand)`
- `NEG` (real) → `mpq_neg(result, operand)`
- `NOT` (bits) → `~operand & mask`
- `BNOT` (bool) → `!operand`

#### Phase 6: Control Flow (Not Started)
- `asl.stmt.cond` → `if/else` blocks
- `asl.stmt.for` → `for` loop with `mpz_t` index
- `asl.stmt.while` → `while` loop
- `asl.stmt.repeat` → `do/while` loop
- `asl.stmt.return` → `return` statement

#### Phase 7: Exception Handling (Not Started)
ASL exceptions require `setjmp`/`longjmp` implementation:
- `asl.stmt.throw` → `longjmp(ctx->jmp_buf, exception_code)`
- `asl.stmt.try` → `setjmp` with catch handler dispatch
- Exception types → structs with type discriminator

#### Phase 8: Function Declarations (Not Started)
- `asl.func` → `emitc.func` with context parameter
- Parameter passing for GMP types (by pointer)
- Return value handling

#### Phase 9: Data Access Operations (Not Started)
- `asl.expr.get_array` → array index `base[index]`
- `asl.expr.get_field` → struct field `record.field`
- `asl.expr.get_item` → tuple item `tuple.itemN`
- `asl.expr.slice` → bitvector extraction

#### Phase 10: L-Expressions (Not Started)
- `asl.lexpr.var` → variable reference
- `asl.lexpr.field` → struct field assignment
- `asl.lexpr.array` → array element assignment
- `asl.lexpr.slice` → bitvector slice assignment

### Implementation Patterns

#### GMP Temporary Management
For arithmetic operations, use a temporary result pattern:
```c
mpz_t _tmp_result;
mpz_init(_tmp_result);
mpz_add(_tmp_result, lhs, rhs);
// Use _tmp_result
mpz_clear(_tmp_result);
```

#### Bitvector Width Tracking
Maintain bit width information for proper masking:
```c
#define MASK_N(w) ((1ULL << (w)) - 1)
uint32_t result = (lhs + rhs) & MASK_N(width);
```

#### Large Bitvector Operations
For >64-bit bitvectors, implement word-by-word operations:
```c
void bv_add(uint64_t* result, const uint64_t* a, const uint64_t* b, int words) {
  uint64_t carry = 0;
  for (int i = 0; i < words; i++) {
    __uint128_t sum = (__uint128_t)a[i] + b[i] + carry;
    result[i] = (uint64_t)sum;
    carry = sum >> 64;
  }
}
```

### Required C Runtime Headers
```c
#include <gmp.h>      // mpz_t, mpq_t
#include <stdint.h>   // uint8_t, uint16_t, uint32_t, uint64_t
#include <stdbool.h>  // bool, true, false
#include <string.h>   // strcmp, strlen
#include <setjmp.h>   // jmp_buf, setjmp, longjmp (for exceptions)
```

### Testing Strategy
Test files in `asl-mlir/test/ASL/` with `// RUN:` directives:
1. Type lowering tests - verify correct C types generated
2. Global variable tests - verify init/free functions
3. Expression tests - verify operator lowering
4. Control flow tests - verify statement lowering
5. Integration tests - full ASL programs compiled to C
