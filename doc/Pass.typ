#let document_title = "Lowering ASL to EmitC"
#set document(title: document_title, author: "Jiuyang Liu")
#set heading(numbering: "1.1")

This document describes the lowering process from the ASL MLIR Dialect to the EmitC Dialect. The EmitC dialect provides a way to emit C/C++ code from MLIR operations, enabling the generation of efficient C implementations from ASL specifications. This pass performs type conversion, operation lowering, and ensures semantic preservation while translating ASL operations to their C equivalents.

= Overview <overview>

The ASL to EmitC lowering pass transforms high-level ASL operations into EmitC operations that can be directly emitted as C code. The lowering process consists of several key phases:

1. *Type Conversion*: ASL types are converted to C-compatible types through EmitC
2. *Operation Lowering*: ASL operations are replaced with equivalent EmitC operations
3. *Control Flow Conversion*: ASL control flow constructs are mapped to C control flow
4. *Memory Management*: ASL's value semantics are preserved in C's pointer-based model

= Type Lowering <type_lowering>

== Integer Types <int_type_lowering>

ASL integer types (`!asl.int`) are lowered to GMP's `mpz_t` type for arbitrary-precision arithmetic. All integer operations use GMP functions to ensure correctness regardless of value magnitude:

#table(
  columns: 3,
  [ASL Type], [C Type], [Notes],
  [`!asl.int<unconstrained>`], [`mpz_t`], [Arbitrary-precision integer using GMP],
  [`!asl.int<constrained<exact>>`], [`mpz_t`], [Arbitrary-precision integer, may be compile-time constant],
  [`!asl.int<constrained<range>>`], [`mpz_t`], [Arbitrary-precision integer with optional range checking],
)

All integer variables must be initialized with `mpz_init()` and cleaned up with `mpz_clear()`. Constants are created using GMP functions like `mpz_set_si()` for small values or `mpz_set_str()` for large values.

*Rationale:* Using GMP for all integers ensures correctness since ASL integers are unbounded and cannot be safely represented by native C integer types like `intmax_t` in the general case. In the future optimizations, if the operation result is bounded and can be inferred to be less than `INT_MAX`, it can be optimized to use native integer type.

== Real Types <real_type_lowering>

ASL real types (`!asl.real`) represent exact rational numbers and are lowered to GMP's `mpq_t` type for arbitrary-precision rational arithmetic. This ensures that ASL's exact arithmetic semantics are preserved in the generated C code.

#table(
  columns: 3,
  [ASL Type], [C Type], [Notes],
  [`!asl.real`], [`mpq_t`], [Arbitrary-precision rational number using GMP],
)

=== Rational Number Semantics <rational_semantics>

ASL real types differ fundamentally from IEEE 754 floating-point types:

- *Exact Representation*: Real values in ASL represent exact rational numbers (p/q where p and q are integers), not approximations
- *No Rounding Errors*: Operations on reals maintain exactness; `1/3 + 1/6` yields exactly `1/2`, not a floating-point approximation
- *Unbounded Precision*: Both numerator and denominator can grow arbitrarily large
- *Automatic Canonicalization*: Rational values are automatically reduced to lowest terms (e.g., `4/6` becomes `2/3`)

=== GMP Rational Type (`mpq_t`) <mpq_type>

The `mpq_t` type from GMP provides:

- A numerator (`mpz_t`) and denominator (`mpz_t`) pair
- Automatic canonicalization via `mpq_canonicalize()`
- Rich arithmetic operations preserving exactness
- Conversion to/from integers and floating-point (when needed)

=== Initialization and Cleanup <real_init_cleanup>

Real variables require explicit initialization and cleanup:

```c
mpq_t x, y, result;

// Initialize rationals (sets to 0/1)
mpq_init(x);
mpq_init(y);
mpq_init(result);

// ... use the variables ...

// Clean up
mpq_clear(x);
mpq_clear(y);
mpq_clear(result);
```

=== Real Type Operations <real_operations>

Common ASL real operations are lowered to GMP rational functions:

#table(
  columns: 3,
  [ASL Operation], [GMP Function], [Notes],
  [Addition `a + b`], [`mpq_add(result, a, b)`], [Exact rational addition],
  [Subtraction `a - b`], [`mpq_sub(result, a, b)`], [Exact rational subtraction],
  [Multiplication `a * b`], [`mpq_mul(result, a, b)`], [Exact rational multiplication],
  [Division `a / b`], [`mpq_div(result, a, b)`], [Exact rational division (b ≠ 0)],
  [Negation `-a`], [`mpq_neg(result, a)`], [Negates the rational],
  [Comparison `a < b`], [`mpq_cmp(a, b) < 0`], [Returns -1, 0, or 1],
  [Equality `a == b`], [`mpq_equal(a, b)`], [Returns non-zero if equal],
)

=== Creating Rational Constants <real_constants>

Rational constants can be created in several ways:

*From integer literals:*
```c
mpq_t quarter;
mpq_init(quarter);
mpq_set_si(quarter, 1, 4);  // Creates 1/4
mpq_canonicalize(quarter);
```

*From two integers:*
```c
mpq_t fraction;
mpq_init(fraction);

// Set numerator and denominator separately
mpz_t num, den;
mpz_init_set_ui(num, 22);
mpz_init_set_ui(den, 7);
mpq_set_num(fraction, num);
mpq_set_den(fraction, den);
mpq_canonicalize(fraction);

mpz_clear(num);
mpz_clear(den);
```

*From a string:*
```c
mpq_t pi_approx;
mpq_init(pi_approx);
mpq_set_str(pi_approx, "355/113", 10);  // Base 10
mpq_canonicalize(pi_approx);
```

=== Conversion Operations <real_conversions>

Converting between reals and other types:

*Real to Integer (truncation/floor):*
```c
mpq_t rational;
mpz_t integer_result;

// ... initialize and set rational ...

mpz_init(integer_result);
// Floor division: numerator / denominator
mpz_fdiv_q(integer_result, mpq_numref(rational), mpq_denref(rational));
```

*Integer to Real:*
```c
mpz_t integer_value;
mpq_t real_result;

// ... initialize and set integer_value ...

mpq_init(real_result);
mpq_set_z(real_result, integer_value);  // Creates rational with denominator 1
```

*Real to Double (for output/approximation):*
```c
mpq_t rational;
// ... initialize and set rational ...

double approx = mpq_get_d(rational);
printf("Approximate value: %f\n", approx);
```

Note: Converting to floating-point loses the exactness guarantee of rationals.

=== Memory Management Considerations <real_memory>

Rational arithmetic can cause unbounded growth in numerator and denominator sizes:

```c
// Example: repeated division can create large denominators
mpq_t x, half;
mpq_init_set_ui(x, 1, 1);      // x = 1/1
mpq_init_set_ui(half, 1, 2);   // half = 1/2

for (int i = 0; i < 100; i++) {
  mpq_mul(x, x, half);  // x = 1/2^100 after loop
}
// Denominator grows exponentially!

mpq_clear(x);
mpq_clear(half);
```

For long-running computations, consider:
- Periodic canonicalization (though `mpq` operations do this automatically)
- Approximate conversion to bounded representations when exactness is no longer needed
- Careful algorithm design to avoid pathological growth

=== Example: Exact Fraction Arithmetic <real_example>

ASL code with exact rational arithmetic:
```
func harmonic_sum(n: integer) => real
begin
  var sum: real = 0.0;
  var i: integer = 1;
  while i <= n do
    sum = sum + (1.0 / i);  // Exact: 1/1 + 1/2 + 1/3 + ...
    i = i + 1;
  end
  return sum;
end
```

Lowered to C:
```c
void harmonic_sum(mpq_t result, const mpz_t n) {
  mpq_t sum, term, i_rational;
  mpz_t i;
  
  // Initialize
  mpq_init(sum);              // sum = 0/1
  mpq_init(term);
  mpq_init(i_rational);
  mpz_init_set_ui(i, 1);
  
  // While loop: i <= n
  while (mpz_cmp(i, n) <= 0) {
    // term = 1 / i (exact rational division)
    mpq_set_z(i_rational, i);           // Convert i to rational
    mpq_set_ui(term, 1, 1);             // term = 1/1
    mpq_div(term, term, i_rational);    // term = 1/i
    
    // sum = sum + term (exact rational addition)
    mpq_add(sum, sum, term);
    
    // i = i + 1
    mpz_add_ui(i, i, 1);
  }
  
  // Return result
  mpq_set(result, sum);
  
  // Cleanup
  mpq_clear(sum);
  mpq_clear(term);
  mpq_clear(i_rational);
  mpz_clear(i);
}
```

The result is exact: for `n=3`, the function returns exactly `11/6`, not a floating-point approximation.

=== Rationale for GMP Rationals <real_rationale>

Using `mpq_t` for ASL reals ensures:

1. *Semantic Fidelity*: ASL's exact arithmetic is preserved
2. *Correctness*: No accumulation of rounding errors
3. *Predictability*: Results are deterministic and mathematically precise
4. *Compliance*: Matches ASL specification requirements

The performance cost is acceptable for specification-level code where correctness is paramount. Future optimizations could analyze when floating-point approximations are safe and substitute `double` for bounded cases.

== Bitvector Types <bits_type_lowering>

ASL bitvector types (`!asl.bits<width, bitfields>`) are lowered to C types based on width:

#table(
  columns: 3,
  [Width Range], [C Type], [Implementation],
  [1-8 bits], [`uint8_t`], [Direct integer representation],
  [9-16 bits], [`uint16_t`], [Direct integer representation],
  [17-32 bits], [`uint32_t`], [Direct integer representation],
  [33-64 bits], [`uint64_t`], [Direct integer representation],
  [65+ bits], [`struct { uint64_t words[N]; }`], [Array-based representation],
)

=== Bitfield Lowering <bitfield_lowering>

ASL bitvectors support bitfield annotations that define named regions within the bitvector. During lowering, bitfields are handled through accessor functions or macros rather than C struct bitfields, as ASL bitfields have different semantics and can be dynamically positioned.

==== Bitfield Types <bitfield_types>

ASL supports three kinds of bitfields (see `BitFieldAttr` in IR documentation):

1. *Simple Bitfields* (`BitField_Simple`): A named slice of the bitvector
2. *Nested Bitfields* (`BitField_Nested`): A bitfield containing sub-bitfields
3. *Typed Bitfields* (`BitField_Type`): A bitfield with an associated ASL type

==== Bitfield Access Patterns <bitfield_access_patterns>

Bitfield access is lowered to bitwise operations:

*Simple bitfield extraction:*
```c
// ASL: bits(32) with bitfield [15:8] named 'byte1'
uint32_t value = 0xDEADBEEF;

// Access bitfield: value.byte1
// Lowered to:
uint8_t byte1 = (value >> 8) & 0xFF;
```

*Nested bitfield extraction:*
```c
// ASL: bits(32) with bitfield [31:16] named 'upper'
//      and nested bitfield [7:4] within 'upper' named 'nibble'
uint32_t value = 0xDEADBEEF;

// Access nested bitfield: value.upper.nibble
// Lowered to:
uint8_t upper_nibble = (value >> 20) & 0xF;  // Offset adjusted for nesting
```

*Typed bitfield extraction:*
```c
// ASL: bits(32) with bitfield [23:16] of type enum{A,B,C}
uint32_t value = 0xDEADBEEF;

// Access typed bitfield with type annotation
// Lowered to:
enum my_enum { A, B, C };
enum my_enum field = (enum my_enum)((value >> 16) & 0xFF);
```

==== Bitfield Assignment <bitfield_assignment>

Bitfield assignment uses read-modify-write sequences:

*Simple bitfield assignment:*
```c
// ASL: value.byte1 = 0x42;
// Lowered to:
value = (value & ~(0xFF << 8)) | ((uint32_t)0x42 << 8);
```

*Nested bitfield assignment:*
```c
// ASL: value.upper.nibble = 0x5;
// Lowered to:
value = (value & ~(0xF << 20)) | ((uint32_t)0x5 << 20);
```

*Typed bitfield assignment:*
```c
// ASL: value.enum_field = A;
// Lowered to:
value = (value & ~(0xFF << 16)) | (((uint32_t)A) << 16);
```

==== Helper Functions for Bitfields <bitfield_helpers>

For complex bitfield operations, helper functions are generated:

```c
// Extract bitfield from bitvector
static inline uint64_t bitvec_extract_field(uint64_t value, int start, int width) {
  return (value >> start) & ((1ULL << width) - 1);
}

// Insert bitfield into bitvector
static inline uint64_t bitvec_insert_field(uint64_t value, uint64_t field, 
                                           int start, int width) {
  uint64_t mask = ((1ULL << width) - 1) << start;
  return (value & ~mask) | ((field << start) & mask);
}

// For large bitvectors (>64 bits)
typedef struct {
  uint64_t words[N];
} bitvec_large_t;

static inline void bitvec_large_extract_field(uint64_t *result, 
                                               const bitvec_large_t *value,
                                               int start, int width) {
  // Implementation for multi-word extraction
  // ...
}

static inline void bitvec_large_insert_field(bitvec_large_t *value,
                                              const uint64_t *field,
                                              int start, int width) {
  // Implementation for multi-word insertion
  // ...
}
```

==== Bitfield Slice Operations <bitfield_slices>

Bitfield slices are lowered based on slice kind (see `SliceAttr` in IR):

*Single slice* (`Slice_Single`):
```c
// ASL: value[i] where i is a bitfield
// Lowered to:
uint8_t bit = (value >> i) & 1;
```

*Range slice* (`Slice_Range`):
```c
// ASL: value[j:i] where j, i are bitfield boundaries
// Lowered to:
uint32_t slice = (value >> i) & ((1 << (j - i + 1)) - 1);
```

*Length slice* (`Slice_Length`):
```c
// ASL: value[i +: n] where i is start, n is length
// Lowered to:
uint32_t slice = (value >> i) & ((1 << n) - 1);
```

*Star slice* (`Slice_Star`):
```c
// ASL: value[factor * length +: length]
// Lowered to:
int start = factor * length;
uint32_t slice = (value >> start) & ((1 << length) - 1);
```

==== Large Bitvector Bitfields <large_bitvector_bitfields>

For bitvectors larger than 64 bits, bitfields span multiple words:

```c
// Bitvector structure
typedef struct {
  uint64_t words[4];  // For 256-bit bitvector
} bitvec256_t;

// Extract bitfield that may span word boundaries
static inline void bitvec256_extract_field(uint64_t *result,
                                           const bitvec256_t *value,
                                           int start, int width) {
  int start_word = start / 64;
  int start_bit = start % 64;
  int end_word = (start + width - 1) / 64;
  
  if (start_word == end_word) {
    // Field within single word
    *result = (value->words[start_word] >> start_bit) & ((1ULL << width) - 1);
  } else {
    // Field spans multiple words
    int bits_in_first = 64 - start_bit;
    uint64_t low_bits = value->words[start_word] >> start_bit;
    uint64_t high_bits = value->words[end_word] & ((1ULL << (width - bits_in_first)) - 1);
    *result = low_bits | (high_bits << bits_in_first);
  }
}

// Insert bitfield that may span word boundaries
static inline void bitvec256_insert_field(bitvec256_t *value,
                                          uint64_t field,
                                          int start, int width) {
  int start_word = start / 64;
  int start_bit = start % 64;
  int end_word = (start + width - 1) / 64;
  
  if (start_word == end_word) {
    // Field within single word
    uint64_t mask = ((1ULL << width) - 1) << start_bit;
    value->words[start_word] = (value->words[start_word] & ~mask) | 
                                ((field << start_bit) & mask);
  } else {
    // Field spans multiple words
    int bits_in_first = 64 - start_bit;
    uint64_t mask_low = ((1ULL << bits_in_first) - 1) << start_bit;
    uint64_t mask_high = (1ULL << (width - bits_in_first)) - 1;
    
    value->words[start_word] = (value->words[start_word] & ~mask_low) |
                                ((field << start_bit) & mask_low);
    value->words[end_word] = (value->words[end_word] & ~mask_high) |
                              ((field >> bits_in_first) & mask_high);
  }
}
```

==== Bitfield Type Conversion <bitfield_type_conversion>

Typed bitfields require type conversion during access:

```c
// ASL enumeration for bitfield type
enum instruction_type {
  TYPE_ADD = 0,
  TYPE_SUB = 1,
  TYPE_MUL = 2,
  TYPE_DIV = 3
};

// Extract typed bitfield
enum instruction_type get_instruction_type(uint32_t instruction) {
  // Bitfield at [31:29] represents instruction type
  uint32_t raw_value = (instruction >> 29) & 0x7;
  return (enum instruction_type)raw_value;
}

// Insert typed bitfield
uint32_t set_instruction_type(uint32_t instruction, 
                               enum instruction_type type) {
  uint32_t mask = 0x7 << 29;
  return (instruction & ~mask) | (((uint32_t)type << 29) & mask);
}
```

==== Bitfield ATC Operations <bitfield_atc>

The `asl.expr.atc.bits` operation materializes bitfield information during type conversion. This is lowered to inline comments or debug information in the generated C code, as the bitfield structure is captured in the accessor functions:

```c
// ASL: bits(32) {[31:24] opcode, [23:16] rd, [15:8] rs1, [7:0] rs2}
// The bitfield structure is documented but doesn't affect the C type

typedef uint32_t instruction_t;  // Base type

// Accessor functions encode bitfield knowledge
static inline uint8_t instruction_get_opcode(instruction_t insn) {
  return (insn >> 24) & 0xFF;
}

static inline uint8_t instruction_get_rd(instruction_t insn) {
  return (insn >> 16) & 0xFF;
}

static inline uint8_t instruction_get_rs1(instruction_t insn) {
  return (insn >> 8) & 0xFF;
}

static inline uint8_t instruction_get_rs2(instruction_t insn) {
  return insn & 0xFF;
}

static inline instruction_t instruction_set_opcode(instruction_t insn, uint8_t opcode) {
  return (insn & 0x00FFFFFF) | ((uint32_t)opcode << 24);
}

// ... similar setters for other fields
```

== Boolean Types <bool_type_lowering>

ASL boolean types (`!asl.bool`) represent logical truth values and are lowered to C's standard `bool` type from `<stdbool.h>`. This provides a natural and efficient representation for boolean logic in the generated C code.

#table(
  columns: 3,
  [ASL Type], [C Type], [Notes],
  [`!asl.bool`], [`bool`], [Standard C99 boolean type (`true`/`false`)],
)

=== Boolean Type Semantics <bool_semantics>

ASL boolean types have straightforward semantics that map cleanly to C:

- *Two Values*: Only `true` and `false` are valid boolean values
- *Logical Operations*: AND, OR, NOT, XOR operations on booleans
- *Comparison Results*: Comparison operations (`==`, `!=`, `<`, `>`, `<=`, `>=`) produce boolean values
- *Control Flow*: Booleans are used in conditional expressions and loop conditions

=== C Boolean Type (`bool`) <c_bool_type>

The C99 `bool` type from `<stdbool.h>` provides:

- Standard boolean values: `true` (1) and `false` (0)
- Implicit conversion from integer types (0 is `false`, non-zero is `true`)
- Small memory footprint (typically 1 byte)
- Native support for logical operations

=== Boolean Operations <bool_operations>

ASL boolean operations are lowered to C logical operators:

#table(
  columns: 3,
  [ASL Operation], [C Operator], [Notes],
  [Logical AND `a && b`], [`a && b`], [Short-circuit evaluation],
  [Logical OR `a || b`], [`a || b`], [Short-circuit evaluation],
  [Logical NOT `!a`], [`!a`], [Negation],
  [Logical XOR `a ^ b`], [`a != b`], [Exclusive OR (inequality for booleans)],
  [Equality `a == b`], [`a == b`], [Boolean equality],
  [Inequality `a != b`], [`a != b`], [Boolean inequality],
)

=== Boolean Literals <bool_literals>

Boolean literals are lowered directly to C boolean constants:

*ASL boolean literals:*
```asl
let t: boolean = TRUE;
let f: boolean = FALSE;
```

*Lowered to C:*
```c
bool t = true;
bool f = false;
```

=== Boolean Expressions <bool_expressions>

Complex boolean expressions are lowered preserving short-circuit evaluation semantics:

*ASL boolean expression:*
```asl
func check_valid(x: integer, y: integer) => boolean
begin
  return x > 0 && y > 0 && x < 100;
end
```

*Lowered to C:*
```c
bool check_valid(const mpz_t x, const mpz_t y) {
  // Compare GMP integers with constants
  return mpz_cmp_si(x, 0) > 0 && 
         mpz_cmp_si(y, 0) > 0 && 
         mpz_cmp_si(x, 100) < 0;
}
```

=== Conditional Expressions <bool_conditionals>

ASL conditional expressions using booleans map to C's ternary operator or if-statements:

*ASL conditional with boolean:*
```asl
func max(a: integer, b: integer) => integer
begin
  return if a > b then a else b;
end
```

*Lowered to C:*
```c
void max_value(mpz_t result, const mpz_t a, const mpz_t b) {
  bool condition = mpz_cmp(a, b) > 0;
  if (condition) {
    mpz_set(result, a);
  } else {
    mpz_set(result, b);
  }
}
```

=== Boolean to Integer Conversion <bool_to_int_conversion>

When booleans need to be converted to integers (e.g., for array indexing or arithmetic):

*ASL boolean to integer conversion:*
```asl
func bool_to_int(b: boolean) => integer
begin
  return if b then 1 else 0;
end
```

*Lowered to C:*
```c
void bool_to_int(mpz_t result, bool b) {
  mpz_set_ui(result, b ? 1 : 0);
}
```

=== Integer to Boolean Conversion <int_to_bool_conversion>

When integers need to be converted to booleans (following C convention where 0 is false, non-zero is true):

*ASL integer to boolean conversion:*
```asl
func int_to_bool(x: integer) => boolean
begin
  return x != 0;
end
```

*Lowered to C:*
```c
bool int_to_bool(const mpz_t x) {
  return mpz_cmp_si(x, 0) != 0;
}
```

=== Boolean Fields in Structures <bool_in_structures>

When booleans appear in structures or as global state, they use the `bool` type:

*ASL with boolean state:*
```asl
var system_enabled: boolean;
var debug_mode: boolean;
```

*Lowered to C context structure:*
```c
typedef struct asl_context {
  bool system_enabled;
  bool debug_mode;
  // ... other fields
} asl_context_t;

void asl_context_init(asl_context_t* ctx) {
  ctx->system_enabled = false;
  ctx->debug_mode = false;
}
```

=== Boolean Arrays <bool_arrays>

Boolean arrays are lowered to arrays of `bool`:

*ASL boolean array:*
```asl
var flags: array [8] of boolean;
```

*Lowered to C:*
```c
typedef struct asl_context {
  bool flags[8];
  // ... other fields
} asl_context_t;

void asl_context_init(asl_context_t* ctx) {
  for (int i = 0; i < 8; i++) {
    ctx->flags[i] = false;
  }
}
```

=== Rationale for C `bool` Type <bool_rationale>

Using C's `bool` type for ASL booleans ensures:

1. *Semantic Clarity*: Code is self-documenting with explicit boolean types
2. *Type Safety*: C compilers can catch type mismatches involving booleans
3. *Efficiency*: `bool` uses minimal memory (typically 1 byte)
4. *Standard Compliance*: Leverages standard C99 features for portability
5. *Natural Mapping*: ASL boolean semantics align perfectly with C boolean semantics

Unlike integers or rationals which require GMP for correctness, booleans have a finite domain and map directly to C's native boolean type without loss of semantic information.

== String Types <string_type_lowering>

ASL string types (`!asl.string`) represent immutable sequences of characters and are lowered to C's `const char*` type. Strings in ASL are used primarily for error messages, debugging output, and metadata rather than for complex text processing.

#table(
  columns: 3,
  [ASL Type], [C Type], [Notes],
  [`!asl.string`], [`const char*`], [Null-terminated C string, immutable],
)

=== String Type Semantics <string_semantics>

ASL string types have the following characteristics:

- *Immutability*: Strings are immutable values in ASL; operations create new strings rather than modifying existing ones
- *Null-Terminated*: Lowered to standard C null-terminated strings for compatibility
- *ASCII Character Set*: ASL strings consist of printable ASCII characters (decimal 32-126) plus escape sequences for special characters (newline, tab, backslash, double-quote)
- *Static Allocation*: String literals are typically stored in read-only data sections

=== C String Type (`const char*`) <c_string_type>

Using `const char*` provides:

- Direct compatibility with C standard library string functions
- Minimal memory overhead (just a pointer)
- Natural integration with C I/O and formatting functions
- Read-only semantics enforced by `const` qualifier

=== String Literals <string_literals>

String literals are lowered to C string literals:

*ASL string literal:*
```asl
let message: string = "Hello, World!";
let error_msg: string = "Invalid input value";
```

*Lowered to C:*
```c
const char* message = "Hello, World!";
const char* error_msg = "Invalid input value";
```

=== String Operations <string_operations>

Common ASL string operations are lowered to C standard library functions:

#table(
  columns: 3,
  [ASL Operation], [C Function/Operator], [Notes],
  [Concatenation `a ++ b`], [`asprintf()` or buffer], [Allocates new string],
  [Length `length(s)`], [`strlen(s)`], [Returns string length],
  [Equality `a == b`], [`strcmp(a, b) == 0`], [Lexicographic comparison],
  [Substring], [`strncpy()` or pointer arithmetic], [Creates substring],
  [String to integer], [`strtol()` or GMP parsing], [Conversion with validation],
)

=== String Concatenation <string_concatenation>

String concatenation requires dynamic memory allocation since C strings are immutable:

*ASL string concatenation:*
```asl
func format_error(code: integer, msg: string) => string
begin
  return "Error " ++ int_to_string(code) ++ ": " ++ msg;
end
```

*Lowered to C:*
```c
#include <stdio.h>
#include <stdlib.h>

char* format_error(const mpz_t code, const char* msg) {
  char* result;
  char code_str[128];
  
  // Convert integer to string
  gmp_snprintf(code_str, sizeof(code_str), "%Zd", code);
  
  // Allocate and format the result string
  asprintf(&result, "Error %s: %s", code_str, msg);
  
  return result;
}

// Usage requires freeing the allocated string
char* error = format_error(error_code, "Invalid operation");
printf("%s\n", error);
free(error);
```

=== String Memory Management <string_memory_management>

String memory management follows these patterns:

*String Literals*: No allocation or deallocation needed, stored in read-only data section.

```c
const char* literal = "This is a constant string";
// No free() needed for literals
```

*Dynamically Allocated Strings*: Must be explicitly freed after use.

```c
char* dynamic_string = format_error(code, msg);
// ... use dynamic_string ...
free(dynamic_string);
```

*String Parameters*: Functions accepting strings use `const char*` for input.

```c
void process_message(const char* msg) {
  // Function does not take ownership, no free() here
  printf("Processing: %s\n", msg);
}
```

=== String Comparison <string_comparison>

String comparison operations are lowered to C string comparison functions:

*ASL string comparison:*
```asl
func check_command(input: string) => boolean
begin
  return input == "START" || input == "STOP";
end
```

*Lowered to C:*
```c
#include <string.h>

bool check_command(const char* input) {
  return strcmp(input, "START") == 0 || strcmp(input, "STOP") == 0;
}
```

=== String in Structures <string_in_structures>

When strings appear in structures, they are represented as `const char*` fields:

*ASL structure with string:*
```asl
type ErrorInfo = {
  code: integer,
  message: string,
  source: string
};
```

*Lowered to C:*
```c
typedef struct ErrorInfo {
  mpz_t code;
  const char* message;
  const char* source;
} ErrorInfo;

void ErrorInfo_init(ErrorInfo* info, const mpz_t code, 
                    const char* message, const char* source) {
  mpz_init_set(info->code, code);
  info->message = message;  // Assumes ownership semantics are clear
  info->source = source;
}

void ErrorInfo_free(ErrorInfo* info) {
  mpz_clear(info->code);
  // Note: Whether to free message/source depends on ownership policy
  // If ErrorInfo owns the strings:
  // free((void*)info->message);
  // free((void*)info->source);
}
```

=== String Conversion Functions <string_conversion>

Converting between strings and other types:

*Integer to String:*
```c
// Using GMP for arbitrary-precision integers
char* int_to_string(const mpz_t value) {
  char* result;
  gmp_asprintf(&result, "%Zd", value);
  return result;  // Caller must free
}
```

*String to Integer:*
```c
// Parse string to GMP integer
bool string_to_int(mpz_t result, const char* str) {
  int success = mpz_set_str(result, str, 10);
  return success == 0;  // 0 indicates success in GMP
}
```

*Bitvector to String (for debugging):*
```c
char* bits_to_string(uint64_t value, int width) {
  char* result = malloc(width + 1);
  for (int i = width - 1; i >= 0; i--) {
    result[width - 1 - i] = ((value >> i) & 1) ? '1' : '0';
  }
  result[width] = '\0';
  return result;  // Caller must free
}
```

=== String Escape Sequences <string_escape_sequences>

ASL string literals support standard escape sequences which map directly to C:

#table(
  columns: 3,
  [ASL Escape], [C Escape], [Meaning],
  [`\n`], [`\n`], [Newline],
  [`\r`], [`\r`], [Carriage return],
  [`\t`], [`\t`], [Tab],
  [`\\`], [`\\`], [Backslash],
  [`\"`], [`\"`], [Double quote],
  [`\'`], [`\'`], [Single quote],
)

*ASL with escape sequences:*
```asl
let multiline: string = "First line\nSecond line\n";
let quoted: string = "He said \"Hello\"";
```

*Lowered to C:*
```c
const char* multiline = "First line\nSecond line\n";
const char* quoted = "He said \"Hello\"";
```

=== Helper Functions for String Operations <string_helper_functions>

Common string helper functions that may be generated:

```c
// Safe string concatenation with allocation
char* asl_string_concat(const char* a, const char* b) {
  if (!a) a = "";
  if (!b) b = "";
  
  size_t len_a = strlen(a);
  size_t len_b = strlen(b);
  char* result = malloc(len_a + len_b + 1);
  
  if (result) {
    memcpy(result, a, len_a);
    memcpy(result + len_a, b, len_b);
    result[len_a + len_b] = '\0';
  }
  
  return result;
}

// Safe string duplication
char* asl_string_dup(const char* str) {
  if (!str) return NULL;
  return strdup(str);
}

// String substring extraction
char* asl_string_substr(const char* str, size_t start, size_t length) {
  if (!str) return NULL;
  
  size_t str_len = strlen(str);
  if (start >= str_len) return strdup("");
  
  size_t actual_len = (start + length > str_len) ? 
                      (str_len - start) : length;
  
  char* result = malloc(actual_len + 1);
  if (result) {
    memcpy(result, str + start, actual_len);
    result[actual_len] = '\0';
  }
  
  return result;
}
```

=== Rationale for C `const char*` Type <string_rationale>

Using `const char*` for ASL strings ensures:

1. *Standard Compatibility*: Direct use of C standard library string functions
2. *Memory Efficiency*: String literals stored in read-only data sections without duplication
3. *Interoperability*: Easy integration with existing C APIs and libraries
4. *Simplicity*: No complex string object management needed
5. *Performance*: Minimal overhead for passing strings between functions

For ASL specifications, which primarily use strings for error messages and debugging rather than complex text processing, this representation provides the best balance of simplicity, efficiency, and compatibility with the C ecosystem.

== Enumeration Types <enum_type_lowering>

ASL enumeration types (`!asl.enum<labels>`) represent a finite set of named constants called enumeration literals or labels. These are lowered to C enumerations (`enum`) for type-safe, efficient representation of discrete values.

#table(
  columns: 3,
  [ASL Type], [C Type], [Notes],
  [`!asl.enum<[label1, label2, ...]>`], [`enum { label1, label2, ... }`], [Named enumeration type],
  [`!asl.label`], [`int` or enum value], [Individual enumeration literal],
)

=== Enumeration Type Semantics <enum_semantics>

ASL enumeration types have the following characteristics:

- *Named Constants*: Enumeration literals act as global constants that can be compared for equality/inequality
- *No Ordering*: Unlike many languages, ASL enumerations do NOT support ordering comparisons (`<`, `<=`, etc.)
- *Type Safety*: Each enumeration literal has the type of the anonymous enumeration that defined it
- *Array Indexing*: Enumeration literals can be used as indices in enumeration-indexed arrays
- *Global Namespace*: Enumeration literals exist in the same namespace as other declared identifiers (except subprograms)
- *Unique Labels*: Each enumeration literal can be declared in at most one enumeration type declaration

=== C Enumeration Type (`enum`) <c_enum_type>

Using C `enum` types provides:

- Compile-time constants with zero runtime overhead
- Type checking and documentation in the C code
- Automatic integer values assigned sequentially (0, 1, 2, ...)
- Integration with C switch statements for pattern matching
- Standard C compatibility across all compilers

=== Enumeration Type Declaration <enum_declaration>

ASL enumeration types are declared and lowered to C enums:

*ASL enumeration type:*
```asl
type TrafficLight of enumeration {GREEN, ORANGE, RED};
type Direction of enumeration {NORTH, SOUTH, EAST, WEST};
```

*Lowered to C:*
```c
// Named enumeration type
typedef enum TrafficLight {
  TrafficLight_GREEN = 0,
  TrafficLight_ORANGE = 1,
  TrafficLight_RED = 2
} TrafficLight;

typedef enum Direction {
  Direction_NORTH = 0,
  Direction_SOUTH = 1,
  Direction_EAST = 2,
  Direction_WEST = 3
} Direction;
```

*Rationale for prefixing:* Enumeration literals in C share a global namespace within their translation unit, so we prefix each literal with the enum type name to avoid collisions and improve code clarity.

=== Anonymous Enumerations <anonymous_enumerations>

ASL only allows enumeration types in type declarations (anonymous enumerations are not permitted). However, the internal representation uses `!asl.label` type for enumeration literals:

*ASL enumeration literal:*
```asl
type Color of enumeration {RED, GREEN, BLUE};
var current_color: Color = RED;
```

*Lowered to C:*
```c
typedef enum Color {
  Color_RED = 0,
  Color_GREEN = 1,
  Color_BLUE = 2
} Color;

// In context structure
typedef struct asl_context {
  Color current_color;
  // ... other fields
} asl_context;

// Initialization
static inline void asl_init_current_color(asl_context* ctx) {
  ctx->current_color = Color_RED;
}
```

=== Enumeration Operations <enum_operations>

ASL enumeration operations are lowered to C comparison operators:

#table(
  columns: 3,
  [ASL Operation], [C Operator/Expression], [Notes],
  [Equality `a == b`], [`a == b`], [Direct enum comparison],
  [Inequality `a != b`], [`a != b`], [Direct enum comparison],
  [Assignment `x = label`], [`x = EnumType_label`], [Direct assignment],
  [Switch/Case], [`switch (x) { case EnumType_label: ... }`], [Pattern matching],
)

*Note:* Ordering comparisons (`<`, `>`, `<=`, `>=`) are NOT supported in ASL for enumerations and should not be generated in lowered code.

=== Enumeration Literals <enum_literals>

Enumeration literals are represented in ASL IR as `asl.expr.literal.label` operations with `!asl.label` type:

*ASL enumeration literal usage:*
```asl
type Status of enumeration {OK, ERROR, PENDING};

func check_status(s: Status) => boolean
begin
  return s == OK || s == ERROR;
end
```

*Lowered to C:*
```c
typedef enum Status {
  Status_OK = 0,
  Status_ERROR = 1,
  Status_PENDING = 2
} Status;

bool check_status(Status s) {
  return s == Status_OK || s == Status_ERROR;
}
```

=== Enumeration in Pattern Matching <enum_pattern_matching>

ASL case statements with enumeration literals lower to C switch statements:

*ASL pattern matching with enums:*
```asl
type TrafficLight of enumeration {GREEN, ORANGE, RED};

func describe_light(light: TrafficLight) => string
begin
  case light of
    when GREEN => return "Go";
    when ORANGE => return "Caution";
    when RED => return "Stop";
  end
end
```

*Lowered to C:*
```c
typedef enum TrafficLight {
  TrafficLight_GREEN = 0,
  TrafficLight_ORANGE = 1,
  TrafficLight_RED = 2
} TrafficLight;

const char* describe_light(TrafficLight light) {
  switch (light) {
    case TrafficLight_GREEN:
      return "Go";
    case TrafficLight_ORANGE:
      return "Caution";
    case TrafficLight_RED:
      return "Stop";
    default:
      // Unreachable if all cases covered
      return "";
  }
}
```

=== Enumeration-Indexed Arrays <enum_indexed_arrays>

ASL supports arrays indexed by enumeration types, which ensures type-safe array access:

*ASL enumeration-indexed array:*
```asl
type Coord of enumeration {X, Y, Z};
var point: array [Coord] of integer;

func set_coordinate(c: Coord, value: integer)
begin
  point[c] = value;
end
```

*Lowered to C:*
```c
typedef enum Coord {
  Coord_X = 0,
  Coord_Y = 1,
  Coord_Z = 2
} Coord;

// In context structure
typedef struct asl_context {
  mpz_t point[3];  // Array size = number of enum values
  // ... other fields
} asl_context;

void set_coordinate(asl_context* ctx, Coord c, const mpz_t value) {
  // Direct array indexing using enum value (which is an integer)
  mpz_set(ctx->point[c], value);
}

// Initialization
static inline void asl_init_point(asl_context* ctx) {
  for (int i = 0; i < 3; i++) {
    mpz_init_set_ui(ctx->point[i], 0);
  }
}

// Cleanup
void asl_free(asl_context* ctx) {
  for (int i = 0; i < 3; i++) {
    mpz_clear(ctx->point[i]);
  }
  // ... other cleanup
}
```

=== Enumeration Type Conversion <enum_type_conversion>

When lowering ASL enumeration types, special attention is needed for type conversions:

*Enum to Integer* (explicit conversion):
```c
// ASL: let idx: integer = as_int(Color_RED);
// Lowered to:
mpz_t idx;
mpz_init_set_ui(idx, (unsigned long)Color_RED);
```

*Integer to Enum* (with validation):
```c
// ASL may require validation when converting integers to enums
// Lowered to (with bounds checking):
bool int_to_color(Color* result, const mpz_t value) {
  if (mpz_fits_ulong_p(value)) {
    unsigned long val = mpz_get_ui(value);
    if (val <= Color_BLUE) {
      *result = (Color)val;
      return true;
    }
  }
  return false;  // Invalid conversion
}
```

=== Enumeration in Structures <enum_in_structures>

When enumerations appear in structured types, they use the C enum type directly:

*ASL structure with enumeration:*
```asl
type Status of enumeration {IDLE, RUNNING, STOPPED};

type SystemState = {
  status: Status,
  error_code: integer
};
```

*Lowered to C:*
```c
typedef enum Status {
  Status_IDLE = 0,
  Status_RUNNING = 1,
  Status_STOPPED = 2
} Status;

typedef struct SystemState {
  Status status;
  mpz_t error_code;
} SystemState;

void SystemState_init(SystemState* state) {
  state->status = Status_IDLE;
  mpz_init_set_ui(state->error_code, 0);
}

void SystemState_free(SystemState* state) {
  mpz_clear(state->error_code);
}
```

=== Label Type Representation <label_type_representation>

The `!asl.label` type in MLIR IR represents individual enumeration literals. During lowering, these are resolved to their corresponding enum type:

*MLIR IR:*
```mlir
%0 = asl.expr.literal.label "RED" : !asl.label
%1 = asl.expr.var "current_color" : !asl.enum<["RED", "GREEN", "BLUE"]>
%2 = asl.expr.binop.eq %1, %0 : !asl.enum<...>, !asl.label -> i1
```

*Lowered to C:*
```c
// Label literal resolved to enum constant
Color label_RED = Color_RED;

// Variable access
Color current_color = ctx->current_color;

// Comparison
bool result = (current_color == Color_RED);
```

=== Enumeration Naming Conventions <enum_naming>

To ensure generated C code is valid and collision-free:

1. *Type Names*: Keep the original ASL type name (e.g., `TrafficLight`)
2. *Label Prefixing*: Prefix each label with the type name and underscore (e.g., `TrafficLight_GREEN`)
3. *Sanitization*: Replace invalid C identifier characters with underscores
4. *Uniqueness*: The prefixing ensures labels from different enum types don't collide

*Example with sanitization:*
```asl
type My-Status of enumeration {OK-State, Error-State};
```

*Lowered to C:*
```c
typedef enum My_Status {
  My_Status_OK_State = 0,
  My_Status_Error_State = 1
} My_Status;
```

=== Rationale for C `enum` Type <enum_rationale>

Using C `enum` types for ASL enumerations ensures:

1. *Type Safety*: C compilers can detect type mismatches at compile time
2. *Zero Overhead*: Enum constants are compile-time values with no runtime cost
3. *Readability*: Generated C code is self-documenting with meaningful names
4. *Debugging*: Debuggers can display enum names instead of raw integers
5. *Standard Compliance*: C enums are universally supported across all C compilers
6. *Switch Optimization*: Compilers can optimize switch statements on enums efficiently
7. *Semantic Preservation*: ASL's lack of ordering is preserved (C doesn't enforce ordering semantics)

The mapping from ASL enumeration types to C enums maintains all semantic properties while providing efficient, type-safe code generation compatible with the entire C ecosystem.

= Global State Management <global_state_management>

Each MLIR module is lowered to C code with a structured approach to managing global state. This design ensures thread safety, clean initialization, and proper resource management.

== Context Structure Design <context_structure_design>

For each MLIR module (e.g., `foo`), three main components are generated:

1. *Context Structure* (`struct foo_context`): Contains all global state
2. *Initialization Function* (`void foo_init(foo_context*)`): Initializes all global state
3. *Cleanup Function* (`void foo_free(foo_context*)`): Frees allocated resources

=== Context Structure <context_structure>

The context structure aggregates all global variables from the ASL module. When lowering global variables, their definitions are added as fields to this structure:

```c
// For module 'foo' with global variables
typedef struct foo_context {
  // Global variables become struct fields
  uint64_t register_file[32];
  uint32_t program_counter;
  uint8_t status_flags;
  
  // Exception handling state (if needed)
  exception_context_t* current_exception_context;
  
  // Other module-specific global state
  // ...
} foo_context;
```

=== Per-Variable Initialization Functions <per_variable_init>

For each global variable `bar` in the module, an inline initialization function is generated:

```c
// Inline initializer for specific global variable
static inline void foo_init_bar(foo_context* ctx) {
  // Initialize the 'bar' field with its initial value
  ctx->bar = /* initial value */;
}
```

These per-variable initialization functions:
- Are marked `static inline` for efficiency
- Take the context pointer as their only parameter
- Set the initial value for one specific global variable
- Can contain complex initialization logic if needed

=== Main Initialization Function <main_init_function>

The main initialization function `foo_init` calls all per-variable initialization functions:

```c
void foo_init(foo_context* ctx) {
  // Call initializer for each global variable
  foo_init_bar(ctx);
  foo_init_baz(ctx);
  foo_init_qux(ctx);
  // ...
  
  // Initialize exception handling if needed
  ctx->current_exception_context = NULL;
}
```

=== Cleanup Function <cleanup_function>

The cleanup function releases any resources allocated during initialization or execution:

```c
void foo_free(foo_context* ctx) {
  // Free any dynamically allocated resources
  // Clean up exception contexts
  // Reset state if needed
  
  // Note: For simple types, this may be empty
  // Complex types (GMP, large bitvectors) require explicit cleanup
}
```

=== Design Rationale <design_rationale>

This three-component design provides several benefits:

*Thread Safety*: Each thread can maintain its own `foo_context` instance, eliminating shared mutable state.

*Composability*: Multiple instances of the same module can coexist (e.g., multi-core simulation).

*Clean Initialization*: Separating per-variable initializers from the main init function:
- Improves code organization and readability
- Allows the compiler to inline initialization code effectively
- Makes it easier to maintain initialization order dependencies
- Facilitates separate compilation and testing

*Resource Management*: Explicit initialization and cleanup functions make resource lifetimes clear.

*Testing*: Easy to create, initialize, use, and destroy context instances in tests.

=== Example: Complete Module Lowering <complete_module_example>

ASL module:
```
var R : bits(64);
```

MLIR module:
```mlir
module {
  %0 = asl.expr.literal.bitvector "'0000000000000000000000000000000000000000000000000000000000000000'" : !asl.bits<-1 : i64, []>
  asl.global var "R" : !asl.bits<64 : i64, []> = %0 : !asl.bits<-1 : i64, []>
}
```

Lowered to C:
```c
// Context structure with all global state
typedef struct foo_context {
  uint64_t R;
} foo_context;

// Per-variable initializers
static inline void foo_init_R(foo_context* ctx) {
    ctx->R = 0ULL;
}

// Main initialization function
void foo_init(foo_context* ctx) {
  foo_init_R(ctx);
}

// Cleanup function
void foo_free(foo_context* ctx) {
  // No dynamic allocations in this example
  // In more complex cases, this would free resources
}

// Usage example
int main() {
  foo_context ctx;
  foo_init(&ctx);
  
  // User logic
  
  foo_free(&ctx);
  return 0;
}
```

== Simple Function <example_func>

ASL:
```
func square(x: integer) => integer
begin
  return x * x;
end
```

Lowered to C (EmitC):
```c
void square(mpz_t* result, const mpz_t x) {
  mpz_mul(*result, x, x);
}
```

== Bitvector Slicing <example_slice>

ASL:
```
let value: bits(32) = 0xDEADBEEF;
let slice: bits(8) = value[15:8];
```

Lowered to C:
```c
uint32_t value = 0xDEADBEEFu;
uint8_t slice = (uint8_t)((value >> 8) & 0xFFu);
```

== Pattern Matching <example_pattern>

ASL:
```
case value of
  when 0 => action1();
  when 1..10 => action2();
  otherwise => action3();
end
```

Lowered to C:
```c
if (value == 0) {
  action1();
} else if (value >= 1 && value <= 10) {
  action2();
} else {
  action3();
}
```

== Global Variable Access <example_global_access>

*Note:* See @global_state_management for the complete design of the context structure pattern.

ASL with global variables:
```
var register_file: array [32] of bits(64);
let ZERO_REGISTER: integer = 0;

func read_register(idx: integer) => bits(64)
begin
  if idx == ZERO_REGISTER then
    return Zeros(64);
  else
    return register_file[idx];
  end
end

func write_register(idx: integer, value: bits(64))
begin
  if idx != ZERO_REGISTER then
    register_file[idx] = value;
  end
end
```

Lowered to C with state struct:
```c
// Global state struct
typedef struct asl_global_state {
  uint64_t register_file[32];
} asl_global_state_t;

// Constant (not in state struct) - using GMP for consistency
// use asl_ as prefix to avoid collision 
static mpz_t asl_ZERO_REGISTER;

// Module initialization function (called once at program start)
void asl_module_init(void) {
  mpz_init_set_ui(asl_ZERO_REGISTER, 0);
}

// State initialization function (called for each context instance)
void asl_global_state_init(asl_global_state_t* state) {
  for (int i = 0; i < 32; i++) {
    state->register_file[i] = 0ULL;
  }
}

// Functions with state parameter
void read_register(asl_global_state_t* state, uint64_t idx, mpz_t* result) {
  if (idx == 0) {  // ZERO_REGISTER comparison
    mpz_set_ui(*result, 0);
  } else {
    // Read from register file (assuming idx is valid)
    mpz_set_ui(*result, state->register_file[idx]);
  }
}

void write_register(asl_global_state_t* state, uint64_t idx, const mpz_t value) {
  if (idx != 0) {  // ZERO_REGISTER is read-only
    // Convert mpz_t to uint64_t and write to register file
    state->register_file[idx] = mpz_get_ui(value);
  }
}

// Usage example
int main() {
  // Initialize module-level constants
  asl_module_init();
  
  asl_global_state_t cpu_state;
  asl_global_state_init(&cpu_state);
  
  // Use mpz_t for register values
  mpz_t reg_value, read_value;
  mpz_init_set_ui(reg_value, 0x1234567890ABCDEFULL);
  mpz_init(read_value);
  
  // Write to register 5
  write_register(&cpu_state, 5, reg_value);
  
  // Read from register 5
  read_register(&cpu_state, 5, read_value);
  
  mpz_clear(reg_value);
  mpz_clear(read_value);
  
  return 0;
}
```

== Thread-Safe Execution <example_thread_safe>

Multiple threads with separate state:
```c
// Thread function
void* cpu_thread(void* arg) {
  asl_global_state_t* state = (asl_global_state_t*)arg;
  
  // Each thread has its own state, no races
  for (int i = 0; i < 1000; i++) {
    uint64_t pc = read_register(state, PC_REGISTER);
    uint32_t insn = fetch_instruction(state, pc);
    execute_instruction(state, insn);
  }
  
  return NULL;
}

int main() {
  // Create separate state for each thread
  asl_global_state_t core0_state, core1_state;
  asl_global_state_init(&core0_state);
  asl_global_state_init(&core1_state);
  
  pthread_t thread0, thread1;
  pthread_create(&thread0, NULL, cpu_thread, &core0_state);
  pthread_create(&thread1, NULL, cpu_thread, &core1_state);
  
  pthread_join(thread0, NULL);
  pthread_join(thread1, NULL);
  
  return 0;
}
```

== Real Number Operations <example_real>

ASL with exact rational arithmetic:
```
func compute_fraction(a: integer, b: integer) => real
begin
  return a / b;  // Exact rational division
end

func test_real() => integer
begin
  let x: real = compute_fraction(1, 3);    // 1/3 exactly
  let y: real = compute_fraction(1, 6);    // 1/6 exactly
  let z: real = x + y;                     // 1/3 + 1/6 = 1/2 exactly
  return CONVERT_INT(z * 10);              // Returns 5
end
```

Lowered to C using GMP:
```c
#include <gmp.h>

void compute_fraction(mpq_t result, const mpz_t a, const mpz_t b) {
  mpq_set_z(result, a);
  mpz_t temp_denom;
  mpz_init_set(temp_denom, b);
  mpq_set_den(result, temp_denom);
  mpz_clear(temp_denom);
  mpq_canonicalize(result);
}

void test_real(mpz_t* result) {
  mpq_t x, y, z, ten, temp;
  mpz_t one, three, six, ten_int;
  
  // Initialize all rationals and integers
  mpq_init(x);
  mpq_init(y);
  mpq_init(z);
  mpq_init(ten);
  mpq_init(temp);
  mpz_init_set_ui(one, 1);
  mpz_init_set_ui(three, 3);
  mpz_init_set_ui(six, 6);
  mpz_init_set_ui(ten_int, 10);
  
  // Compute x = 1/3
  compute_fraction(x, one, three);
  
  // Compute y = 1/6
  compute_fraction(y, one, six);
  
  // Compute z = x + y = 1/2
  mpq_add(z, x, y);
  
  // Compute z * 10
  mpq_set_si(ten, 10, 1);
  mpq_mul(temp, z, ten);
  
  // Convert to integer (floor division)
  mpz_fdiv_q(*result, mpq_numref(temp), mpq_denref(temp));
  
  // Cleanup
  mpq_clear(x);
  mpq_clear(y);
  mpq_clear(z);
  mpq_clear(ten);
  mpq_clear(temp);
  mpz_clear(one);
  mpz_clear(three);
  mpz_clear(six);
  mpz_clear(ten_int);
  
  // result now contains 5
}
```

=== GMP Linking Requirements <gmp_linking>

Programs using the lowered ASL code with `real` types must link against the GMP library:

```bash
# Compilation
gcc -c asl_generated.c -o asl_generated.o

# Linking
gcc asl_generated.o -lgmp -o program

# Or with pkg-config
gcc asl_generated.c $(pkg-config --cflags --libs gmp) -o program
```

The generated C code includes appropriate headers:
```c
#include <gmp.h>      // For GMP arbitrary-precision integers (mpz_t) and rationals (mpq_t)
#include <stdint.h>   // For fixed-width integer types (bitvectors)
#include <stdbool.h>  // For bool
```

= Exception Lowering

== Exception Types <exception_type_lowering>

ASL exception types (`!asl.exception<fields>`) are lowered to C structs for exception data and use setjmp/longjmp for control flow. This approach provides the closest semantics to ASL exceptions with non-local control flow.

=== Exception Context Structure <exception_context>

The lowering uses C standard library `setjmp`/`longjmp` for exception handling. The exception context is stored in the global state struct for thread safety:

```c
#include <setjmp.h>
#include <stdbool.h>

// Exception context for setjmp/longjmp
typedef struct asl_exception_context {
  jmp_buf jump_buffer;
  bool exception_active;
  int exception_type;
  void* exception_data;
  struct asl_exception_context* prev_context;  // For nested try-catch
} asl_exception_context_t;

// Global state includes exception context
typedef struct asl_global_state {
  // ... other global variables ...
  
  // Exception handling state
  asl_exception_context_t* current_exception_context;
} asl_global_state_t;
```

=== Exception Type Definitions <exception_types>

Exception types are represented as enums with associated data structures:

```c
// Exception types as enums
enum asl_exception_type {
  ASL_EXCEPTION_NONE = 0,
  ASL_EXCEPTION_UNPREDICTABLE,
  ASL_EXCEPTION_SEE,
  ASL_EXCEPTION_UNDEFINED,
  ASL_EXCEPTION_CONSTRAINT_ERROR,
  ASL_EXCEPTION_USER_DEFINED  // User-defined exceptions start from here
};

// Built-in exception data structures
typedef struct {
  const char* message;
} asl_exception_unpredictable_t;

typedef struct {
  const char* description;
} asl_exception_see_t;

typedef struct {
  const char* reason;
} asl_exception_undefined_t;

typedef struct {
  const char* constraint;
  mpz_t value;
} asl_exception_constraint_error_t;

// Example user-defined exception structure
typedef struct {
  const char* name;
  uint64_t pc;
  uint32_t instruction;
} asl_exception_decode_error_t;
```

=== Helper Functions <exception_helpers>

Helper functions manage exception context and throwing, taking the global state as parameter:

```c
// Throw exception
static inline void asl_throw_exception(asl_global_state_t* state, 
                                       int type, void* data) {
  if (state->current_exception_context) {
    state->current_exception_context->exception_active = true;
    state->current_exception_context->exception_type = type;
    state->current_exception_context->exception_data = data;
    longjmp(state->current_exception_context->jump_buffer, 1);
  } else {
    // Uncaught exception - terminate program
    fprintf(stderr, "Uncaught exception of type %d\n", type);
    if (data) free(data);
    abort();
  }
}
```

=== Exception Statement Lowering <exception_stmt_lowering>

==== Throw Statement Lowering <stmt_throw_lowering>

The `asl.stmt.throw` operation is lowered to allocate exception data and call the throw helper with state:

```c
// ASL: throw Unpredictable;
asl_exception_unpredictable_t* exc_data = 
  malloc(sizeof(asl_exception_unpredictable_t));
exc_data->message = "Unpredictable behavior";
asl_throw_exception(state, ASL_EXCEPTION_UNPREDICTABLE, exc_data);

// ASL: throw ConstraintError("Value out of range", value);
asl_exception_constraint_error_t* exc_data = 
  malloc(sizeof(asl_exception_constraint_error_t));
exc_data->constraint = "Value out of range";
mpz_init_set(exc_data->value, value);
asl_throw_exception(state, ASL_EXCEPTION_CONSTRAINT_ERROR, exc_data);
```

==== Try-Catch Statement Lowering <stmt_try_lowering>

The `asl.stmt.try` operation is lowered to setjmp/longjmp-based structured exception handling with state management:

```c
// Setup exception context
asl_exception_context_t exc_ctx;
exc_ctx.prev_context = state->current_exception_context;
state->current_exception_context = &exc_ctx;

// Set jump point
if (setjmp(exc_ctx.jump_buffer) == 0) {
  // Try block
  protected_code(state);
} else {
  // Exception was thrown
  switch (exc_ctx.exception_type) {
    case ASL_EXCEPTION_UNPREDICTABLE: {
      asl_exception_unpredictable_t* data = 
        (asl_exception_unpredictable_t*)exc_ctx.exception_data;
      // Handle unpredictable exception
      handle_unpredictable(state, data);
      break;
    }
    case ASL_EXCEPTION_CONSTRAINT_ERROR: {
      asl_exception_constraint_error_t* data = 
        (asl_exception_constraint_error_t*)exc_ctx.exception_data;
      // Handle constraint error
      handle_constraint_error(state, data);
      // Cleanup mpz_t in exception data
      mpz_clear(data->value);
      break;
    }
    default:
      // Otherwise block or re-throw
      state->current_exception_context = exc_ctx.prev_context;
      if (exc_ctx.prev_context) {
        longjmp(exc_ctx.prev_context->jump_buffer, 1);
      } else {
        fprintf(stderr, "Unhandled exception type: %d\n", 
                exc_ctx.exception_type);
        abort();
      }
  }
  
  // Cleanup exception data
  if (exc_ctx.exception_data) {
    free(exc_ctx.exception_data);
    exc_ctx.exception_data = NULL;
  }
}

// Restore previous context
state->current_exception_context = exc_ctx.prev_context;
```

=== Exception Type Structure Lowering <exception_type_struct>

Exception types with fields are lowered to C structs:

```c
// ASL exception type definition:
// exception DecodeError of {
//   pc: bits(64),
//   instruction: bits(32),
//   reason: string
// };

typedef struct {
  uint64_t pc;
  uint32_t instruction;
  const char* reason;
} asl_exception_decode_error_t;

// Exception type enum entry
#define ASL_EXCEPTION_DECODE_ERROR (ASL_EXCEPTION_USER_DEFINED + 1)

// Throwing the exception
void throw_decode_error(asl_global_state_t* state, 
                        uint64_t pc, uint32_t insn, const char* reason) {
  asl_exception_decode_error_t* exc = 
    malloc(sizeof(asl_exception_decode_error_t));
  exc->pc = pc;
  exc->instruction = insn;
  exc->reason = reason;
  asl_throw_exception(state, ASL_EXCEPTION_DECODE_ERROR, exc);
}

// Catching the exception
if (exc_ctx.exception_type == ASL_EXCEPTION_DECODE_ERROR) {
  asl_exception_decode_error_t* data = 
    (asl_exception_decode_error_t*)exc_ctx.exception_data;
  printf("Decode error at PC 0x%lx: %s\n", data->pc, data->reason);
  free(data);
}
```

= Exception Handling Examples <example_exceptions>

=== Simple Exception Throw and Catch <example_simple_exception>

ASL code with exception handling:
```
exception DivideByZero;

func safe_divide(a: integer, b: integer) => integer
begin
  if b == 0 then
    throw DivideByZero;
  end
  return a / b;
end

func test_divide(a: integer, b: integer) => integer
begin
  try
    return safe_divide(a, b);
  catch
    when DivideByZero =>
      return 0;
  end
end
```

Lowered to C using setjmp/longjmp with global state:
```c
#include <setjmp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

// Exception type definitions
enum asl_exception_type {
  ASL_EXCEPTION_NONE = 0,
  ASL_EXCEPTION_DIVIDE_BY_ZERO = 1
};

typedef struct asl_exception_context {
  jmp_buf jump_buffer;
  bool exception_active;
  int exception_type;
  void* exception_data;
  struct asl_exception_context* prev_context;
} asl_exception_context_t;

typedef struct asl_global_state {
  // Exception handling state
  asl_exception_context_t* current_exception_context;
  
  // Other global state would be here
} asl_global_state_t;

static inline void asl_throw_exception(asl_global_state_t* state,
                                       int type, void* data) {
  if (state->current_exception_context) {
    state->current_exception_context->exception_active = true;
    state->current_exception_context->exception_type = type;
    state->current_exception_context->exception_data = data;
    longjmp(state->current_exception_context->jump_buffer, 1);
  } else {
    fprintf(stderr, "Uncaught exception of type %d\n", type);
    abort();
  }
}

// Initialize global state
void asl_global_state_init(asl_global_state_t* state) {
  state->current_exception_context = NULL;
}

// Function implementations
intmax_t safe_divide(asl_global_state_t* state, intmax_t a, intmax_t b) {
  if (b == 0) {
    asl_throw_exception(state, ASL_EXCEPTION_DIVIDE_BY_ZERO, NULL);
  }
  return a / b;
}

intmax_t test_divide(asl_global_state_t* state, intmax_t a, intmax_t b) {
  // Setup exception context
  asl_exception_context_t exc_ctx = {0};
  exc_ctx.prev_context = state->current_exception_context;
  state->current_exception_context = &exc_ctx;
  
  intmax_t result = 0;
  
  // Set jump point
  if (setjmp(exc_ctx.jump_buffer) == 0) {
    // Try block
    result = safe_divide(state, a, b);
  } else {
    // Exception was thrown
    if (exc_ctx.exception_type == ASL_EXCEPTION_DIVIDE_BY_ZERO) {
      // Catch DivideByZero
      result = 0;
    } else {
      // Re-throw unhandled exception
      state->current_exception_context = exc_ctx.prev_context;
      if (exc_ctx.prev_context) {
        longjmp(exc_ctx.prev_context->jump_buffer, 1);
      }
    }
  }
  
  // Restore previous context
  state->current_exception_context = exc_ctx.prev_context;
  return result;
}
```

=== Exception with Data Fields <example_exception_with_data>

ASL code with exception carrying data:
```
exception ValidationError of {
  field_name: string,
  expected: integer,
  actual: integer
};

func validate_range(name: string, value: integer, min: integer, max: integer)
begin
  if value < min || value > max then
    throw ValidationError {
      field_name = name,
      expected = max,
      actual = value
    };
  end
end

func process_value(value: integer) => integer
begin
  try
    validate_range("input", value, 0, 100);
    return value * 2;
  catch
    when ValidationError => ve =>
      print("Validation failed for ", ve.field_name);
      print("Expected <= ", ve.expected, ", got ", ve.actual);
      return -1;
  end
end
```

Lowered to C:
```c
#include <setjmp.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

// Exception definitions
enum asl_exception_type {
  ASL_EXCEPTION_NONE = 0,
  ASL_EXCEPTION_VALIDATION_ERROR = 1
};

typedef struct {
  const char* field_name;
  mpz_t expected;
  mpz_t actual;
} asl_exception_validation_error_t;

typedef struct asl_exception_context {
  jmp_buf jump_buffer;
  bool exception_active;
  int exception_type;
  void* exception_data;
  struct asl_exception_context* prev_context;
} asl_exception_context_t;

typedef struct asl_global_state {
  asl_exception_context_t* current_exception_context;
} asl_global_state_t;

static inline void asl_throw_exception(asl_global_state_t* state,
                                       int type, void* data) {
  if (state->current_exception_context) {
    state->current_exception_context->exception_active = true;
    state->current_exception_context->exception_type = type;
    state->current_exception_context->exception_data = data;
    longjmp(state->current_exception_context->jump_buffer, 1);
  } else {
    fprintf(stderr, "Uncaught exception of type %d\n", type);
    if (data) free(data);
    abort();
  }
}

// Function implementations
void validate_range(asl_global_state_t* state, const char* name, 
                    const mpz_t value, const mpz_t min, const mpz_t max) {
  if (mpz_cmp(value, min) < 0 || mpz_cmp(value, max) > 0) {
    asl_exception_validation_error_t* exc = 
      malloc(sizeof(asl_exception_validation_error_t));
    exc->field_name = name;
    mpz_init_set(exc->expected, max);
    mpz_init_set(exc->actual, value);
    asl_throw_exception(state, ASL_EXCEPTION_VALIDATION_ERROR, exc);
  }
}

void process_value(mpz_t* result, asl_global_state_t* state, const mpz_t value) {
  // Setup exception context
  asl_exception_context_t exc_ctx = {0};
  exc_ctx.prev_context = state->current_exception_context;
  state->current_exception_context = &exc_ctx;
  
  mpz_set_ui(*result, 0);
  
  if (setjmp(exc_ctx.jump_buffer) == 0) {
    // Try block
    mpz_t min, max;
    mpz_init_set_ui(min, 0);
    mpz_init_set_ui(max, 100);
    validate_range(state, "input", value, min, max);
    mpz_mul_ui(*result, value, 2);
    mpz_clear(min);
    mpz_clear(max);
  } else {
    // Exception was thrown
    if (exc_ctx.exception_type == ASL_EXCEPTION_VALIDATION_ERROR) {
      // Catch ValidationError
      asl_exception_validation_error_t* ve = 
        (asl_exception_validation_error_t*)exc_ctx.exception_data;
      
      printf("Validation failed for %s\n", ve->field_name);
      gmp_printf("Expected <= %Zd, got %Zd\n", ve->expected, ve->actual);
      
      mpz_clear(ve->expected);
      mpz_clear(ve->actual);
      free(ve);
      exc_ctx.exception_data = NULL;
      mpz_set_si(*result, -1);
    } else {
      // Re-throw
      state->current_exception_context = exc_ctx.prev_context;
      if (exc_ctx.prev_context) {
        longjmp(exc_ctx.prev_context->jump_buffer, 1);
      }
    }
  }
  
  // Restore context
  state->current_exception_context = exc_ctx.prev_context;
  // result is already set via output parameter
}
```

=== Nested Try-Catch Blocks <example_nested_try_catch>

ASL code with nested exception handling:
```
exception OuterError;
exception InnerError of { code: integer };

func inner_function(x: integer) => integer
begin
  if x < 0 then
    throw InnerError { code = x };
  end
  return x * 2;
end

func outer_function(x: integer) => integer
begin
  try
    let y = inner_function(x);
    if y > 100 then
      throw OuterError;
    end
    return y;
  catch
    when InnerError => ie =>
      print("Inner error with code: ", ie.code);
      return 0;
  end
end

func main_function(x: integer) => integer
begin
  try
    return outer_function(x);
  catch
    when OuterError =>
      print("Outer error caught");
      return -1;
    otherwise =>
      print("Unknown error");
      return -2;
  end
end
```

Lowered to C:
```c
// Exception types
enum asl_exception_type {
  ASL_EXCEPTION_NONE = 0,
  ASL_EXCEPTION_OUTER_ERROR = 1,
  ASL_EXCEPTION_INNER_ERROR = 2
};

typedef struct {
  intmax_t code;
} asl_exception_inner_error_t;

// ... (exception context and state definitions as before) ...

void inner_function(asl_global_state_t* state, mpz_t* result, const mpz_t x) {
  if (mpz_cmp_si(x, 0) < 0) {
    asl_exception_inner_error_t* exc = 
      malloc(sizeof(asl_exception_inner_error_t));
    exc->code = mpz_get_si(x);  // Convert mpz_t to intmax_t for exception code
    asl_throw_exception(state, ASL_EXCEPTION_INNER_ERROR, exc);
  }
  mpz_mul_ui(*result, x, 2);
}

void outer_function(asl_global_state_t* state, mpz_t* result, const mpz_t x) {
  asl_exception_context_t exc_ctx = {0};
  exc_ctx.prev_context = state->current_exception_context;
  state->current_exception_context = &exc_ctx;
  
  mpz_set_ui(*result, 0);
  
  if (setjmp(exc_ctx.jump_buffer) == 0) {
    // Try block
    mpz_t y;
    mpz_init(y);
    inner_function(state, &y, x);
    if (mpz_cmp_ui(y, 100) > 0) {
      mpz_clear(y);
      asl_throw_exception(state, ASL_EXCEPTION_OUTER_ERROR, NULL);
    }
    mpz_set(*result, y);
    mpz_clear(y);
  } else {
    // Exception was thrown
    if (exc_ctx.exception_type == ASL_EXCEPTION_INNER_ERROR) {
      // Catch InnerError
      asl_exception_inner_error_t* ie = 
        (asl_exception_inner_error_t*)exc_ctx.exception_data;
      printf("Inner error with code: %jd\n", ie->code);
      free(ie);
      mpz_set_ui(*result, 0);
    } else {
      // Re-throw (not caught here)
      state->current_exception_context = exc_ctx.prev_context;
      if (exc_ctx.prev_context) {
        longjmp(exc_ctx.prev_context->jump_buffer, 1);
      }
    }
  }
  
  state->current_exception_context = exc_ctx.prev_context;
  // result is already set via output parameter
}

void main_function(asl_global_state_t* state, mpz_t* result, const mpz_t x) {
  asl_exception_context_t exc_ctx = {0};
  exc_ctx.prev_context = state->current_exception_context;
  state->current_exception_context = &exc_ctx;
  
  mpz_set_ui(*result, 0);
  
  if (setjmp(exc_ctx.jump_buffer) == 0) {
    // Try block
    outer_function(state, result, x);
  } else {
    // Exception was thrown
    if (exc_ctx.exception_type == ASL_EXCEPTION_OUTER_ERROR) {
      // Catch OuterError
      printf("Outer error caught\n");
      mpz_set_si(*result, -1);
    } else {
      // Otherwise clause
      printf("Unknown error\n");
      mpz_set_si(*result, -2);
    }
    
    if (exc_ctx.exception_data) {
      free(exc_ctx.exception_data);
    }
  }
  
  state->current_exception_context = exc_ctx.prev_context;
  // result is already set via output parameter
}
```

=== Exception Handling with Global State <example_exception_global_state>

ASL code with exceptions accessing global state:
```
exception StateError of { current_state: integer };

var system_state: integer = 0;

func update_state(new_state: integer)
begin
  if new_state < 0 || new_state > 10 then
    throw StateError { current_state = system_state };
  end
  system_state = new_state;
end

func safe_update(new_value: integer) => integer
begin
  try
    update_state(new_value);
    return system_state;
  catch
    when StateError => se =>
      print("State error, current: ", se.current_state);
      return se.current_state;
  end
end
```

Lowered to C with integrated exception and global state:
```c
// Global state with both system variables and exception context
typedef struct asl_global_state {
  // System state variables
  intmax_t system_state;
  
  // Exception handling state
  asl_exception_context_t* current_exception_context;
} asl_global_state_t;

// Exception definitions
typedef struct {
  intmax_t current_state;
} asl_exception_state_error_t;

#define ASL_EXCEPTION_STATE_ERROR 1

// ... (exception context and throw helper as before) ...

void asl_global_state_init(asl_global_state_t* state) {
  state->system_state = 0;
  state->current_exception_context = NULL;
}

void update_state(asl_global_state_t* state, intmax_t new_state) {
  if (new_state < 0 || new_state > 10) {
    asl_exception_state_error_t* exc = 
      malloc(sizeof(asl_exception_state_error_t));
    exc->current_state = state->system_state;
    asl_throw_exception(state, ASL_EXCEPTION_STATE_ERROR, exc);
  }
  state->system_state = new_state;
}

intmax_t safe_update(asl_global_state_t* state, intmax_t new_value) {
  asl_exception_context_t exc_ctx = {0};
  exc_ctx.prev_context = state->current_exception_context;
  state->current_exception_context = &exc_ctx;
  
  intmax_t result = 0;
  
  if (setjmp(exc_ctx.jump_buffer) == 0) {
    // Try block
    update_state(state, new_value);
    result = state->system_state;
  } else {
    // Exception was thrown
    if (exc_ctx.exception_type == ASL_EXCEPTION_STATE_ERROR) {
      // Catch StateError
      asl_exception_state_error_t* se = 
        (asl_exception_state_error_t*)exc_ctx.exception_data;
      printf("State error, current: %jd\n", se->current_state);
      result = se->current_state;
      free(se);
    } else {
      // Re-throw
      state->current_exception_context = exc_ctx.prev_context;
      if (exc_ctx.prev_context) {
        longjmp(exc_ctx.prev_context->jump_buffer, 1);
      }
    }
  }
  
  state->current_exception_context = exc_ctx.prev_context;
  return result;
}
```
