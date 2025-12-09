# Array Test Cases Added to TypeLowering.asl

## Summary

Added comprehensive array type test cases to `asl-mlir/test/Passes/TypeLowering.asl` covering various array scenarios as global variables. These tests verify the correct lowering of ASL array types to C code through the EmitC dialect.

## Test Cases Added

### 1. Primitive Bitvector Arrays

**bytes** - Array of 8-bit values
```asl
var bytes: array [16] of bits(8);
```
Expected C initialization:
- Loop from 0 to 15
- Initialize each element to `0u`

**register_file** - Array of 64-bit values (common in CPU modeling)
```asl
var register_file: array [32] of bits(64);
```
Expected C initialization:
- Loop from 0 to 31
- Initialize each element to `0ULL`

### 2. Boolean Arrays

**flags** - Array of boolean flags
```asl
var flags: array [8] of boolean;
```
Expected C initialization:
- Loop from 0 to 7
- Initialize each element to `false`

### 3. GMP Integer Arrays

**bigints** - Array of arbitrary-precision integers
```asl
var bigints: array [10] of integer;
```
Expected C initialization:
- Loop from 0 to 9
- Use `mpz_init_set_str(ctx->bigints[i], "0", 10)` for each element

### 4. GMP Rational Arrays

**rationals** - Array of arbitrary-precision rational numbers
```asl
var rationals: array [5] of real;
```
Expected C initialization:
- Loop from 0 to 4
- Use `mpq_init()`, `mpq_set_str()`, and `mpq_canonicalize()` for each element

### 5. Enumeration-Indexed Arrays

**position** - Array indexed by enumeration type
```asl
type Coord of enumeration {X, Y, Z};
var position: array [Coord] of integer;
```
Expected C initialization:
- Loop from 0 to 2 (3 enum values)
- Initialize each mpz_t element
- Tests type-safe enumeration indexing

### 6. Arrays of Structured Types

**points** - Array of tuples
```asl
var points: array [4] of Point;
```
where `Point = (integer, integer)`

Expected C initialization:
- Loop from 0 to 3
- Initialize both `item0` and `item1` of each tuple element
- Demonstrates nested initialization for complex types

### 7. Multi-Dimensional Arrays

**matrix** - 2D array (3×3 matrix)
```asl
var matrix: array [3] of array [3] of integer;
```
Expected C initialization:
- Nested loops: outer (i: 0 to 2), inner (j: 0 to 2)
- Initialize each `mpz_t` element at `matrix[i][j]`
- Tests multi-dimensional array lowering

### 8. Arrays with Explicit Initialization

**nibbles_init** - Small bitvector array with values
```asl
var nibbles_init: array [4] of bits(4) = [Ones{4} repeated 4];
```
Expected C initialization:
- Loop from 0 to 3
- Initialize each element to `15u` (0xF, all ones in 4 bits)

**counters_init** - Integer array with initial values
```asl
var counters_init: array [6] of integer = [100 repeated 6];
```
Expected C initialization:
- Loop from 0 to 5
- Initialize each element to `"100"` using `mpz_init_set_str`

### 9. Large Arrays

**memory** - Memory bank simulation
```asl
var memory: array [256] of bits(8);
```
Expected C initialization:
- Loop from 0 to 255
- Initialize each byte to `0u`
- Tests scalability of array lowering

### 10. String Arrays

**messages** - Array of strings
```asl
var messages: array [3] of string;
```
Expected C initialization:
- Loop from 0 to 2
- Initialize each element to `""` (empty string)

### 11. Enumeration Arrays

**statuses** - Array of enumeration values
```asl
var statuses: array [5] of Status;
```
where `Status = enumeration {OK, ERROR, PENDING}`

Expected C initialization:
- Loop from 0 to 4
- Initialize each element to `asl_Status_OK` (enum default)

## Test Coverage

The added test cases cover:

1. ✅ **Primitive Types**: bits(8), bits(64), boolean
2. ✅ **GMP Types**: integer (mpz_t), real (mpq_t)
3. ✅ **Indexing**: Integer-indexed and enumeration-indexed
4. ✅ **Element Types**: Primitives, GMP types, tuples, strings, enums
5. ✅ **Dimensions**: 1D and 2D (multi-dimensional)
6. ✅ **Initialization**: Default values and explicit initial values
7. ✅ **Sizes**: Small (3-10 elements) to large (256 elements)
8. ✅ **Real-World Use Cases**: Register files, memory banks, coordinate systems

## Expected Lowering Pattern

All array tests follow the same pattern in C:

```c
static inline void asl_init_<varname>(asl_context* ctx) {
  for (int i = 0; i < SIZE; i++) {
    // Initialize ctx-><varname>[i] based on element type
  }
}
```

For multi-dimensional arrays:
```c
static inline void asl_init_<varname>(asl_context* ctx) {
  for (int i = 0; i < SIZE1; i++) {
    for (int j = 0; j < SIZE2; j++) {
      // Initialize ctx-><varname>[i][j]
    }
  }
}
```

## FileCheck Directives

Each test case includes appropriate CHECK directives to verify:
- Correct function naming
- Proper loop bounds
- Correct initialization calls for the element type
- Nested loops for multi-dimensional arrays
- GMP-specific initialization for arbitrary-precision types

## Testing Instructions

Run the test with:
```bash
cd asl-mlir/test/Passes
asl-json-backend --no-std TypeLowering.asl > /tmp/TypeLowering.json
asl-opt --canonicalize --run-asl-to-emitc --emitc --json-input /tmp/TypeLowering.json | FileCheck TypeLowering.asl
```

Or use lit:
```bash
lit TypeLowering.asl
```

## Related Documentation

These tests correspond to the Array Types section in `doc/Pass.typ` which provides detailed documentation on:
- Array type lowering strategies
- Initialization patterns for different element types
- C code generation for arrays
- Memory management for complex element types
