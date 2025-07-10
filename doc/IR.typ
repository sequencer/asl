#let document_title = "MLIR Dialect for ASL"
#set document(title: document_title, author: "Jiuyang Liu")
#set heading(numbering: "1.1")

= #document_title
== Introduction

This document describes the design of an MLIR dialect for the ASL (Arm Specification Language). The goal is to provide a lower-level representation of ASL code that can be used for analysis, optimization, and code generation. This dialect aims to capture the semantics of ASL as faithfully as possible, while also being amenable to MLIR's transformation infrastructure. The design is based on the ASL AST defined in the `asllib` library.

=== AST Metadata Mapping
Information from the ASL AST's `'a annotated` record is preserved in the MLIR representation to ensure full traceability.

==== Source Location
The `pos_start` and `pos_end` fields from the AST are translated into MLIR `Location` objects. Every generated operation will be tagged with the location of the AST node it originates from.

==== Version Mapping
The `version` field (`V0` or `V1`) is preserved by attaching a custom `asl.version` attribute to each operation.

#block(
  stroke: luma(180),
  inset: 8pt,
  radius: 4pt,
)[
  An `E_Binop` node from a V1 source file at `file.asl:10:5` would be converted to an `asl.binop` operation with both location and version attributes:
  ```mlir
  %add = asl.binop "PLUS" %x_val, %c2 : !asl.int loc("file.asl":10:5) {asl.version = 1}
  ```
]

== Operations Mapping
This section maps ASL AST expressions, statements, and declarations to `asl` dialect operations.

=== Operations
- *`E_Unop`*: Each unary operator maps to a specific MLIR operation:
  - `BNOT`: `asl.bnot` (Boolean inversion).
  - `NEG`: `asl.neg` (Integer or real negation).
  - `NOT`: `asl.not` (Bitvector bitwise inversion).
- *`E_Binop`*: Each binary operator maps to a specific MLIR operation:
  - `AND`: `asl.and` (Bitvector bitwise AND).
  - `BAND`: `asl.band` (Boolean AND).
  - `BEQ`: `asl.beq` (Boolean equivalence).
  - `BOR`: `asl.bor` (Boolean OR).
  - `DIV`: `asl.div` (Integer division).
  - `DIVRM`: `asl.divrm` (Inexact integer division, rounding towards negative infinity).
  - `XOR`: `asl.xor` (Bitvector bitwise exclusive OR).
  - `EQ_OP`: `asl.eq` (Equality on two base values).
  - `GT`: `asl.gt` (Greater than for integers or reals).
  - `GEQ`: `asl.geq` (Greater than or equal for integers or reals).
  - `IMPL`: `asl.impl` (Boolean implication).
  - `LT`: `asl.lt` (Less than for integers or reals).
  - `LEQ`: `asl.leq` (Less than or equal for integers or reals).
  - `MOD`: `asl.mod` (Remainder of integer division).
  - `MINUS`: `asl.minus` (Subtraction for integers, reals, or bitvectors).
  - `MUL`: `asl.mul` (Multiplication for integers, reals, or bitvectors).
  - `NEQ`: `asl.neq` (Non-equality on two base values).
  - `OR`: `asl.or` (Bitvector bitwise OR).
  - `PLUS`: `asl.plus` (Addition for integers, reals, or bitvectors).
  - `POW`: `asl.pow` (Exponentiation for integers).
  - `RDIV`: `asl.rdiv` (Division for reals).
  - `SHL`: `asl.shl` (Shift left for integers).
  - `SHR`: `asl.shr` (Shift right for integers).
  - `CONCAT`: `asl.concat` (Bitvector or string concatenation).
=== Parsed Values
- *`E_Literal`*: `asl.literal`. The operation's `value` attribute stores the literal value, with the attribute kind depending on the literal type:
  - `L_Int`: Stored as `IntegerAttr` with type `!asl.int`.
  - `L_Bool`: Stored as `BoolAttr` with type `!asl.bool`.
  - `L_Real`: Stored as `StringAttr` with type `!asl.real`. The OCaml type `Q.t` represents an arbitrary-precision rational number. To preserve its full precision, it is serialized to a string (e.g., `"1/3"`). Downstream tools can then parse this string back into a GMP-compatible rational number data structure without loss of information.
  - `L_BitVector`: Stored as `IntegerAttr` with type `!asl.bits`.
  - `L_String`: Stored as `StringAttr` with type `!asl.string`.
  - `L_Label`: Stored as `StringAttr` (the label's name) with the corresponding `!asl.enum` type.
=== Expressions
- *`E_Literal`*: Mapped to `asl.literal`. See `Parsed Values` for details.
- *`E_Var`*: Mapped to `asl.var`.
- *`E_ATC`*: Mapped to `asl.atc` (Asserted Type Conversion).
- *`E_Binop`*: Mapped to specific binary operations like `asl.plus`, `asl.minus`, etc. See `Operations` for a full list.
- *`E_Unop`*: Mapped to specific unary operations like `asl.not`. See `Operations` for a full list.
- *`E_Call`*: Mapped to `asl.call`. This operation represents a subprogram invocation, capturing the callee's name, type parameters (`params`), arguments (`args`), and the kind of subprogram (`call_type`). See `Call Mapping` for more details.
- *`E_Slice`*: Mapped to a family of `asl.slice.*` operations. See `Slice Mapping` for more details.
- *`E_Cond`*: Mapped to `asl.cond` (as an expression with results).
- *`E_GetArray`*: Mapped to `asl.get_array`.
- *`E_GetEnumArray`*: Mapped to `asl.get_enum_array`.
- *`E_GetField`*: Mapped to `asl.get_field`.
- *`E_GetFields`*: Mapped to `asl.get_fields`, which returns a tuple of field values.
- *`E_GetCollectionFields`*: Mapped to `asl.get_collection_fields`.
- *`E_GetItem`*: Mapped to `asl.get_item`.
- *`E_Record`*: Mapped to `asl.record_construct`.
- *`E_Tuple`*: Mapped to `asl.tuple_construct`.
- *`E_Array`*: Mapped to `asl.array_construct`.
- *`E_EnumArray`*: Mapped to `asl.enum_array_construct`.
- *`E_Arbitrary`*: Mapped to `asl.arbitrary_expr`.
- *`E_Pattern`*: Mapped to `asl.pattern`.

=== Pattern Mapping
The `E_Pattern` expression is mapped to a family of `asl.pattern.*` operations, each corresponding to a variant of the `pattern_desc` type in the ASL AST. All pattern operations take the expression to be matched as their first operand.
- *`Pattern_All`*: Mapped to `asl.pattern.all`. Represents a wildcard pattern that matches any value.
  - `asl.pattern.all %expr`
- *`Pattern_Any`*: Mapped to `asl.pattern.any`. Represents a disjunctive pattern that matches if any of the sub-patterns match.
  - `asl.pattern.any %expr, [sub-patterns...]`
- *`Pattern_Geq`*: Mapped to `asl.pattern.geq`. Matches values greater than or equal to the given threshold expression.
  - `asl.pattern.geq %expr, %threshold`
- *`Pattern_Leq`*: Mapped to `asl.pattern.leq`. Matches values less than or equal to the given threshold expression.
  - `asl.pattern.leq %expr, %threshold`
- *`Pattern_Mask`*: Mapped to `asl.pattern.mask`. Matches expressions against a bitvector mask pattern using the mask stored as an attribute.
  - `asl.pattern.mask %expr {mask = bitvector_mask_attr}`
- *`Pattern_Not`*: Mapped to `asl.pattern.not`. Represents the negation of another pattern.
  - `asl.pattern.not %expr, [negated-pattern]`
- *`Pattern_Range`*: Mapped to `asl.pattern.range`. Matches values within an inclusive range from lower bound to upper bound.
  - `asl.pattern.range %expr, %lower, %upper`
- *`Pattern_Single`*: Mapped to `asl.pattern.single`. Matches a single specific value using equality comparison.
  - `asl.pattern.single %expr, %value`
- *`Pattern_Tuple`*: Mapped to `asl.pattern.tuple`. Destructures and matches tuple expressions against multiple sub-patterns.
  - `asl.pattern.tuple %expr, [sub-patterns...]`

=== Slice Mapping
The `E_Slice` expression is mapped to a family of `asl.slice.*` operations, each corresponding to a variant of the `slice` type in the ASL AST. All slice operations take the base expression to be sliced as their first operand.
- *`Slice_Single`*: Mapped to `asl.slice.single`. It takes the base expression and an index `i`, returning a slice of length 1 at that position.
  - `asl.slice.single %base, %i`
- *`Slice_Range`*: Mapped to `asl.slice.range`. It takes the base expression, a start index `i`, and an end index `j`, returning the slice from `i` to `j-1`.
  - `asl.slice.range %base, %i, %j`
- *`Slice_Length`*: Mapped to `asl.slice.length`. It takes the base expression, a start index `i`, and a length `n`, returning a slice of length `n` starting at `i`.
  - `asl.slice.length %base, %i, %n`
- *`Slice_Star`*: Mapped to `asl.slice.star`. It takes the base expression, a factor, and a length. The start index is computed as `factor * length`.
  - `asl.slice.star %base, %factor, %length`

=== Call Mapping
The `call` record from the ASL AST is mapped to the `asl.call` operation.
- *`name`*: The callee, represented as a `SymbolRefAttr`.
- *`params`*: The type parameters for the call, which can be stored as an array attribute.
- *`args`*: The SSA values for the function arguments.
- *`call_type`*: Stored as a string attribute on the operation (see `Subprogram Type Mapping`).
== Type System Mapping
This section details how the type system from `AST.mli` is mapped to MLIR types within the `asl` dialect.

- *`T_Int`*: Mapped to `!asl.int<constraint>`, where the constraint parameter captures the `constraint_kind` from the AST:
  - `UnConstrained`: No constraint, represented as `!asl.int`.
  - `WellConstrained`: Constrained to specific values or ranges, represented as `!asl.int<constraints>` where constraints is an array of:
    - `Constraint_Exact`: An exact value constraint.
    - `Constraint_Range`: An inclusive range constraint with lower and upper bounds.
  - `PendingConstrained`: A constraint to be inferred during type-checking, represented as `!asl.int<pending>`.
  - `Parameterized`: A parameterized integer type with a unique identifier, represented as `!asl.int<param, identifier>`.
- *`T_Bits`*: Mapped to `!asl.bits<width, bitfields>`, where:
  - `width` is an expression defining the bit width.
  - `bitfields` is an optional array of bitfield definitions, each being one of:
    - `BitField_Simple`: A simple bitfield with name and slice list.
    - `BitField_Nested`: A bitfield with name, slice list, and nested bitfields.
    - `BitField_Type`: A bitfield with name, slice list, and explicit type annotation.
- *`T_Real`*: Mapped to `!asl.real`.
- *`T_String`*: Mapped to `!asl.string`.
- *`T_Bool`*: Mapped to `!asl.bool`.
- *`T_Enum`*: Mapped to `!asl.enum<labels>`, where `labels` is an array of strings.
- *`T_Tuple`*: Mapped to `!asl.tuple<elements>`, where `elements` is a list of MLIR types.
- *`T_Array`*: Mapped to `!asl.array<element_type, index_type>`. The `index_type` captures the `array_index` from the AST.
- *`T_Record`*: Mapped to `!asl.record<name, fields>`, representing a collection of named fields.
- *`T_Exception`*: Mapped to `!asl.exception<name, fields>`, structurally similar to records.
- *`T_Collection`*: Mapped to `!asl.collection<fields>`.
- *`T_Named`*: Mapped to `!asl.named<name>`, where `name` is a `SymbolRefAttr` to a type declaration. This allows for type synonyms.

== L-Expressions and Statements
=== L-Expressions
L-expressions represent the left-hand side of assignments and are mapped to a family of `asl.lexpr.*` operations.

- *`LE_Discard`*: Mapped to `asl.lexpr.discard`. Represents a discarded assignment target (e.g., `_` in tuple destructuring).
- *`LE_Var`*: Mapped to `asl.lexpr.var`. Represents assignment to a variable.
- *`LE_Slice`*: Mapped to `asl.lexpr.slice`. Represents assignment to a slice of an expression.
- *`LE_SetArray`*: Mapped to `asl.lexpr.set_array`. Represents assignment to an array element with integer index.
- *`LE_SetEnumArray`*: Mapped to `asl.lexpr.set_enum_array`. Represents assignment to an array element with enumeration index (typed AST only).
- *`LE_SetField`*: Mapped to `asl.lexpr.set_field`. Represents assignment to a record field.
- *`LE_SetFields`*: Mapped to `asl.lexpr.set_fields`. Represents assignment to multiple record fields with type annotations.
- *`LE_SetCollectionFields`*: Mapped to `asl.lexpr.set_collection_fields`. Represents assignment to collection fields with type annotations.
- *`LE_Destructuring`*: Mapped to `asl.lexpr.destructuring`. Represents tuple destructuring assignment.

=== Statements
Statements are mapped to a family of `asl.stmt.*` operations and control flow operations.
- *`S_Pass`*: Mapped to `asl.stmt.pass`. A no-operation statement.
- *`S_Seq`*: Mapped to `asl.stmt.seq`. Sequential composition of two statements.
- *`S_Decl`*: Mapped to `asl.stmt.decl`. Local variable declaration with:
  - Declaration keyword (`LDK_Var`, `LDK_Constant`, `LDK_Let`) stored as string attribute
  - Declaration item (variable name or tuple destructuring)
    The `local_decl_item` variants are mapped as follows:
    - `LDI_Var`: Single variable declaration
    - `LDI_Tuple`: Tuple destructuring declaration
  - Optional type annotation
  - Optional initial value expression

- *`S_Assign`*: Mapped to `asl.stmt.assign`. Assignment statement taking an l-expression and an expression.
- *`S_Call`*: Mapped to `asl.stmt.call`. Procedure call statement (similar to `asl.call` but without return value).
- *`S_Return`*: Mapped to `asl.stmt.return`. Return statement with optional expression.
- *`S_Cond`*: Mapped to `asl.stmt.cond`. Conditional statement with condition, then-branch, and else-branch.
- *`S_Assert`*: Mapped to `asl.stmt.assert`. Assertion statement with boolean expression.
- *`S_For`*: Mapped to `asl.stmt.for`. For loop with:
  - Index variable name
  - Start expression
  - Direction (`Up` or `Down`) as string attribute
  - End expression
  - Loop body statement
  - Optional limit expression
- *`S_While`*: Mapped to `asl.stmt.while`. While loop with condition, optional limit expression, and body.
- *`S_Repeat`*: Mapped to `asl.stmt.repeat`. Repeat-until loop with body, condition, and optional limit expression.
- *`S_Throw`*: Mapped to `asl.stmt.throw`. Throw statement with optional expression and type annotation.
- *`S_Try`*: Mapped to `asl.stmt.try`. Try-catch statement with:
  - Protected statement
  - List of catchers (exception handlers)
  - Optional otherwise clause
  Catchers are represented as structured attributes containing:
  - Optional exception variable name
  - Guard type
  - Handler statement
- *`S_Print`*: Mapped to `asl.stmt.print`. Print statement with:
  - List of argument expressions
  - `newline` boolean attribute indicating if a newline should be added
  - `debug` boolean attribute indicating if this is a debug print
- *`S_Unreachable`*: Mapped to `asl.stmt.unreachable`. Unreachable statement indicating dead code.
- *`S_Pragma`*: Mapped to `asl.stmt.pragma`. Pragma statement with identifier and expression list for tool-specific hints.

=== Case Alternatives and Pattern Matching
Case alternatives (`case_alt`) are used in pattern matching contexts and are mapped to `asl.case_alt` operations with:
- Pattern to match against
- Optional where clause (guard expression)
- Statement to execute on match

=== Catcher
Catchers are exception handlers used in try-catch statements. Each catcher from the ASL AST is represented as a structured attribute or region containing:
- Optional exception variable name (identifier option): The name to bind the caught exception to, if specified
- Guard type (ty): The type of exception this catcher handles
- Handler statement (stmt): The statement to execute when this catcher matches the thrown exception

Catchers are used within `asl.stmt.try` operations to define how different types of exceptions should be handled.

== Declarations
- *`D_Func`*: Mapped to `asl.func`, compatible with MLIR's `func.func`. The `func` record from the ASL AST contains the following fields:
  - `name`: Function identifier, mapped to the symbol name of the operation.
  - `parameters`: Type parameters as `(identifier * ty option) list`, stored as attributes containing parameter names and optional type constraints.
  - `args`: Function arguments as `typed_identifier list`, mapped to the function's input arguments with their types.
  - `body`: Subprogram body of type `subprogram_body`:
    - `SB_ASL`: Normal ASL statement body, mapped to the function's body region.
    - `SB_Primitive`: Primitive function with boolean indicating side effects, represented as an attribute.
  - `return_type`: Optional return type, mapped to the function's result types.
  - `subprogram_type`: The kind of subprogram (`ST_Procedure`, `ST_Function`, etc.), stored as a string attribute.
  - `recurse_limit`: Optional recursion limit expression, stored as an attribute.
  - `qualifier`: Optional function qualifier, stored as string attribute:
    - `Pure`: Function does not read or modify mutable state.
    - `Readonly`: Function can read but not modify mutable state.
    - `Noreturn`: Function always terminates by exception or unreachable.
  - `override`: Optional override information, stored as string attribute:
    - `Impdef`: Function can be overridden.
    - `Implementation`: Function overrides a corresponding `Impdef` function.
  - `builtin`: Boolean flag indicating builtin functions that receive special treatment during parameter checking, stored as a boolean attribute.
- *`D_GlobalStorage`*: Mapped to `asl.global`. The `global_decl` record from the ASL AST contains the following fields:
  - `keyword`: Declaration keyword stored as string attribute:
    - `GDK_Constant`: Constant global storage
    - `GDK_Config`: Configuration variable that can be set at runtime
    - `GDK_Let`: Immutable binding evaluated once
    - `GDK_Var`: Mutable global variable
  - `name`: Global identifier, mapped to the symbol name of the operation
  - `ty`: Optional type annotation for the global storage
  - `initial_value`: Optional initial value expression for the global storage
- *`D_TypeDecl`*: Mapped to `asl.type_decl`. Type declaration with:
  - `identifier`: The name of the type being declared, mapped to the symbol name
  - `ty`: The type definition being aliased or defined
  - `(identifier * field list) option`: Optional enumeration information containing:
    - Enumeration name (identifier)
    - List of enumeration fields with their types
    This third parameter allows type declarations to also define enumerations with associated data.
- *`D_Pragma`*: Mapped to `asl.pragma`. Global pragma declaration with:
  - `identifier`: The pragma name/directive, stored as a string attribute
  - `expr list`: List of expressions providing arguments to the pragma, stored as operands
  Global pragmas provide tool-specific hints at the top level of ASL files and can be used by analysis or compilation tools that need AST-level guidance.
=== Subprogram Type Mapping
The `subprogram_type` from the ASL AST is mapped to a string attribute on the `asl.call` operation. The possible values for this attribute are:
- `ST_Procedure`: A subprogram without return type, called from a statement
- `ST_Function`: A subprogram with a return type, called from an expression
- `ST_Getter`: A special function called with a syntax similar to slices
- `ST_EmptyGetter`: A special function called with a syntax similar to a variable (relevant only for V0)
- `ST_Setter`: A special procedure called with a syntax similar to slice assignment
- `ST_EmptySetter`: A special procedure called with a syntax similar to an assignment to a variable (relevant only for V0)


== MLIR Type System Design

This section details the complete type system design for the ASL dialect in MLIR. Each ASL type is mapped to a corresponding MLIR type with appropriate parameters and attributes.

=== Base Types

==== Integer Types
- `!asl.int` - Unconstrained integer type
- `!asl.int<pending>` - Integer with pending constraint inference
- `!asl.int<param, "param_name">` - Parameterized integer with identifier
- `!asl.int<constraints[exact(42)]>` - Integer constrained to exact value 42
- `!asl.int<constraints[range(0, 255)]>` - Integer constrained to range [0, 255]
- `!asl.int<constraints[exact(1), range(10, 20)]>` - Integer constrained to value 1 OR range [10, 20]

==== Bitvector Types
- `!asl.bits<32>` - 32-bit bitvector without bitfields
- `!asl.bits<width_expr>` - Bitvector with expression-defined width
- `!asl.bits<32, [simple("flag", [0:0]), nested("data", [7:1], [simple("high", [7:4]), simple("low", [3:1])])]>` - Bitvector with bitfield definitions

==== Primitive Types
- `!asl.real` - Real number type
- `!asl.string` - String type
- `!asl.bool` - Boolean type

=== Composite Types

==== Enumeration Types
- `!asl.enum<["RED", "GREEN", "BLUE"]>` - Enumeration with three labels
- `!asl.enum<[]>` - Empty enumeration (error case)

==== Tuple Types
- `!asl.tuple<>` - Empty tuple
- `!asl.tuple<!asl.int, !asl.bool>` - Tuple of integer and boolean
- `!asl.tuple<!asl.int, !asl.tuple<!asl.real, !asl.string>>` - Nested tuple

==== Array Types
- `!asl.array<!asl.int, length(10)>` - Array of 10 integers
- `!asl.array<!asl.bits<32>, length(expr)>` - Array with expression-defined length
- `!asl.array<!asl.real, enum("Color", ["RED", "GREEN", "BLUE"])>` - Array indexed by enumeration

==== Record Types
- `!asl.record<"Point", [("x", !asl.real), ("y", !asl.real)]>` - Point record with x,y fields
- `!asl.record<"Empty", []>` - Empty record
- `!asl.record<"Nested", [("inner", !asl.record<"Inner", [("value", !asl.int)]>)]>` - Nested record

==== Exception Types
- `!asl.exception<"DivByZero", []>` - Exception without fields
- `!asl.exception<"ParseError", [("line", !asl.int), ("message", !asl.string)]>` - Exception with fields

==== Collection Types
- `!asl.collection<[("field1", !asl.int), ("field2", !asl.string)]>` - Collection with named fields

=== Special Types

==== Named Types
- `!asl.named<@MyType>` - Reference to type declaration named "MyType"
- `!asl.named<@nested.scope.Type>` - Reference to nested type declaration

=== Type Attributes and Parameters

==== Constraint Attributes
```mlir
#asl.constraint_exact<42 : i64>
#asl.constraint_range<0 : i64, 255 : i64>
#asl.constraints<[#asl.constraint_exact<1>, #asl.constraint_range<10, 20>]>
```

==== Bitfield Attributes
```mlir
#asl.bitfield_simple<"name", [#asl.slice_single<0>, #asl.slice_range<7, 1>]>
#asl.bitfield_nested<"name", [#asl.slice_range<15, 8>], [#asl.bitfield_simple<"sub", [#asl.slice_single<0>]>]>
#asl.bitfield_type<"name", [#asl.slice_range<31, 16>], !asl.int>
```

==== Array Index Attributes
```mlir
#asl.array_length_expr<expr_attr>
#asl.array_length_enum<"Color", ["RED", "GREEN", "BLUE"]>
```

=== Type Verification Rules

==== Constraint Verification
- Exact constraints must have statically evaluable expressions
- Range constraints must have lower bound ≤ upper bound
- Parameterized types must reference valid parameter identifiers

==== Bitfield Verification
- Slice ranges must be within bitvector width bounds
- Nested bitfield slices must be subsets of parent slice
- No overlapping bitfield slices at the same nesting level

==== Array Verification
- Length expressions must evaluate to positive integers
- Enumeration arrays must reference valid enumeration types
- Element types must be well-formed

==== Record/Exception Verification
- Field names must be unique within the record/exception
- Field types must be well-formed
- Records and exceptions with same name must have identical field signatures

=== Type Conversion Operations

==== Implicit Conversions
- Integer constraint widening (more specific to less specific)
- Parameterized type instantiation
- Named type resolution

==== Explicit Conversions (ATC - Asserted Type Conversion)
- `asl.atc %value : !asl.int -> !asl.int<constraints[range(0, 100)]>`
- `asl.atc %value : !asl.bits<32> -> !asl.bits<32, [simple("flag", [0:0])]>`

=== Type Examples

==== Complete Function Type
```mlir
asl.func @example(
  %x: !asl.int<constraints[range(0, 255)]>,
  %data: !asl.bits<32, [simple("flag", [0:0]), simple("value", [31:1])]>,
  %point: !asl.record<"Point", [("x", !asl.real), ("y", !asl.real)]>
) -> !asl.tuple<!asl.bool, !asl.array<!asl.int, length(10)>>
```

==== Global Storage with Complex Type
```mlir
asl.global @lookup_table : !asl.array<!asl.record<"Entry", [("key", !asl.string), ("value", !asl.int)]>, enum("Priority", ["LOW", "MEDIUM", "HIGH"])>
```

==== Type Declaration Examples
```mlir
asl.type_decl @CustomInt : !asl.int<constraints[range(0, 1000)]>
asl.type_decl @StatusBits : !asl.bits<8, [simple("error", [0:0]), simple("ready", [1:1]), simple("data", [7:2])]>
asl.type_decl @Result : !asl.record<"Result", [("success", !asl.bool), ("data", !asl.named<@CustomInt>)]>
```


== MLIR IR Design

This section provides the complete MLIR IR design specification for implementing the ASL dialect. It covers all operations, types, attributes, and their syntax.

=== Dialect Namespace
- Dialect namespace: `asl`
- All types prefixed with `!asl.`
- All operations prefixed with `asl.`
- All attributes prefixed with `#asl.`

=== Complete Type System

==== Primitive Types
```mlir
!asl.int                                    // Unconstrained integer
!asl.int<pending>                          // Pending constraint inference
!asl.int<param, "param_id">                // Parameterized integer
!asl.int<constraints[exact(42)]>           // Exact value constraint
!asl.int<constraints[range(0, 255)]>       // Range constraint
!asl.int<constraints[exact(1), range(10, 20)]>  // Multiple constraints (union)

!asl.real                                  // Real number type
!asl.string                               // String type  
!asl.bool                                 // Boolean type

!asl.bits<32>                             // 32-bit bitvector
!asl.bits<width_expr>                     // Expression-defined width
!asl.bits<8, [simple("flag", [0:0])]>     // With simple bitfield
!asl.bits<32, [nested("ctrl", [7:0], [simple("en", [0:0])])]>  // Nested bitfields
!asl.bits<16, [typed("data", [15:8], !asl.int)]>  // Typed bitfield
```

==== Composite Types
```mlir
!asl.enum<["RED", "GREEN", "BLUE"]>       // Enumeration type

!asl.tuple<>                              // Empty tuple
!asl.tuple<!asl.int>                      // Single element tuple
!asl.tuple<!asl.int, !asl.bool, !asl.real>  // Multi-element tuple

!asl.array<!asl.int, length(10)>          // Fixed-length array
!asl.array<!asl.int, length(expr)>        // Expression-length array  
!asl.array<!asl.real, enum("Color", ["RED", "GREEN", "BLUE"])>  // Enum-indexed array

!asl.record<"Point", [("x", !asl.real), ("y", !asl.real)]>  // Record type
!asl.exception<"Error", [("code", !asl.int), ("msg", !asl.string)]>  // Exception type
!asl.collection<[("field1", !asl.int), ("field2", !asl.string)]>  // Collection type

!asl.named<@TypeName>                     // Named type reference
```

=== Complete Attribute System

==== Constraint Attributes
```mlir
#asl.constraint_exact<42 : i64>
#asl.constraint_range<0 : i64, 255 : i64>
#asl.constraints<[#asl.constraint_exact<1>, #asl.constraint_range<10, 20>]>
```

==== Bitfield Attributes
```mlir
#asl.slice_single<0>
#asl.slice_range<7, 1>
#asl.bitfield_simple<"name", [#asl.slice_single<0>]>
#asl.bitfield_nested<"name", [#asl.slice_range<7, 0>], [#asl.bitfield_simple<"sub", [#asl.slice_single<0>]>]>
#asl.bitfield_typed<"name", [#asl.slice_range<15, 8>], !asl.int>
```

==== Array Index Attributes
```mlir
#asl.array_length<expr_attr>
#asl.array_enum<"Color", ["RED", "GREEN", "BLUE"]>
```

==== Field Attributes
```mlir
#asl.field<"name", !asl.int>
#asl.fields<[#asl.field<"x", !asl.real>, #asl.field<"y", !asl.real>]>
```

=== Complete Operation Set

==== Literal Operations
```mlir
%0 = asl.literal 42 : !asl.int                           // Integer literal
%1 = asl.literal true : !asl.bool                        // Boolean literal  
%2 = asl.literal "1/3" : !asl.real                       // Real literal (as string)
%3 = asl.literal 0xFF : !asl.bits<8>                     // Bitvector literal
%4 = asl.literal "hello" : !asl.string                   // String literal
%5 = asl.literal "RED" : !asl.enum<["RED", "GREEN", "BLUE"]>  // Enum label literal
```

==== Variable Operations
```mlir
%0 = asl.var "variable_name" : !asl.int                  // Variable reference
```

==== Arithmetic Operations
```mlir
%0 = asl.plus %lhs, %rhs : !asl.int                     // Addition
%1 = asl.minus %lhs, %rhs : !asl.int                    // Subtraction
%2 = asl.mul %lhs, %rhs : !asl.int                      // Multiplication
%3 = asl.div %lhs, %rhs : !asl.int                      // Integer division
%4 = asl.mod %lhs, %rhs : !asl.int                      // Modulo
%5 = asl.pow %base, %exp : !asl.int                     // Exponentiation
%6 = asl.rdiv %lhs, %rhs : !asl.real                    // Real division
%7 = asl.divrm %lhs, %rhs : !asl.int                    // Division with rounding
%8 = asl.neg %operand : !asl.int                        // Negation
```

==== Bitwise Operations
```mlir
%0 = asl.and %lhs, %rhs : !asl.bits<32>                 // Bitwise AND
%1 = asl.or %lhs, %rhs : !asl.bits<32>                  // Bitwise OR
%2 = asl.xor %lhs, %rhs : !asl.bits<32>                 // Bitwise XOR
%3 = asl.not %operand : !asl.bits<32>                   // Bitwise NOT
%4 = asl.shl %value, %amount : !asl.int                 // Shift left
%5 = asl.shr %value, %amount : !asl.int                 // Shift right
%6 = asl.concat %lhs, %rhs : !asl.bits<64>              // Concatenation
```

==== Logical Operations  
```mlir
%0 = asl.band %lhs, %rhs : !asl.bool                    // Boolean AND
%1 = asl.bor %lhs, %rhs : !asl.bool                     // Boolean OR
%2 = asl.bnot %operand : !asl.bool                      // Boolean NOT
%3 = asl.beq %lhs, %rhs : !asl.bool                     // Boolean equivalence
%4 = asl.impl %lhs, %rhs : !asl.bool                    // Boolean implication
```

==== Comparison Operations
```mlir
%0 = asl.eq %lhs, %rhs : !asl.bool                      // Equality
%1 = asl.neq %lhs, %rhs : !asl.bool                     // Inequality
%2 = asl.lt %lhs, %rhs : !asl.bool                      // Less than
%3 = asl.leq %lhs, %rhs : !asl.bool                     // Less or equal
%4 = asl.gt %lhs, %rhs : !asl.bool                      // Greater than
%5 = asl.geq %lhs, %rhs : !asl.bool                     // Greater or equal
```

==== Slice Operations
```mlir
%0 = asl.slice.single %base, %index : !asl.bits<1>      // Single bit slice
%1 = asl.slice.range %base, %start, %end : !asl.bits<8>  // Range slice
%2 = asl.slice.length %base, %start, %len : !asl.bits<16>  // Length slice
%3 = asl.slice.star %base, %factor, %len : !asl.bits<8>  // Star slice
```

==== Array/Record Access Operations
```mlir
%0 = asl.get_array %array, %index : !asl.int            // Array element access
%1 = asl.get_enum_array %array, %enum_index : !asl.int  // Enum array access
%2 = asl.get_field %record, "field_name" : !asl.int     // Single field access
%3 = asl.get_fields %record, ["f1", "f2"] : !asl.tuple<!asl.int, !asl.real>  // Multiple fields
%4 = asl.get_collection_fields "collection", ["field1"] : !asl.tuple<!asl.int>  // Collection fields
%5 = asl.get_item %tuple, 0 : !asl.int                  // Tuple element access
```

==== Construction Operations
```mlir
%0 = asl.record_construct : !asl.record<"Point", [("x", !asl.real), ("y", !asl.real)]> {
  "x" = %x_val,
  "y" = %y_val
}
%1 = asl.tuple_construct %elem1, %elem2 : !asl.tuple<!asl.int, !asl.real>
%2 = asl.array_construct %length, %value : !asl.array<!asl.int, length(10)>
%3 = asl.enum_array_construct "Color", ["RED", "GREEN", "BLUE"], %value : !asl.array<!asl.int, enum("Color", ["RED", "GREEN", "BLUE"])>
```

==== Function Call Operations
```mlir
%0 = asl.call @function_name(%param1, %param2)(%arg1, %arg2) : (!asl.int, !asl.real) -> !asl.bool {
  call_type = "ST_Function"
}
%1:2 = asl.call @multi_return(%arg) : (!asl.int) -> (!asl.int, !asl.bool) {
  call_type = "ST_Function"
}
```

==== Type Conversion Operations
```mlir
%0 = asl.atc %value : !asl.int -> !asl.int<constraints[range(0, 100)]>  // Asserted type conversion
```

==== Conditional Operations
```mlir
%0 = asl.cond %condition, %true_val, %false_val : !asl.int  // Conditional expression
```

==== Pattern Operations
```mlir
%0 = asl.pattern.all %expr : !asl.bool                   // Wildcard pattern
%1 = asl.pattern.any %expr, [%pattern1, %pattern2] : !asl.bool  // Disjunctive pattern
%2 = asl.pattern.geq %expr, %threshold : !asl.bool       // Greater-equal pattern
%3 = asl.pattern.leq %expr, %threshold : !asl.bool       // Less-equal pattern
%4 = asl.pattern.mask %expr {mask = #asl.bitvector_mask<"10xx01">} : !asl.bool  // Mask pattern
%5 = asl.pattern.not %expr, %sub_pattern : !asl.bool     // Negation pattern
%6 = asl.pattern.range %expr, %lower, %upper : !asl.bool  // Range pattern
%7 = asl.pattern.single %expr, %value : !asl.bool        // Single value pattern
%8 = asl.pattern.tuple %expr, [%pat1, %pat2] : !asl.bool  // Tuple pattern
```

==== Arbitrary Value Operations
```mlir
%0 = asl.arbitrary : !asl.int                           // Arbitrary value of type
```

=== L-Expression Operations

==== L-Expression Types
```mlir
%0 = asl.lexpr.discard : !asl.lexpr                     // Discard target
%1 = asl.lexpr.var "variable_name" : !asl.lexpr         // Variable target
%2 = asl.lexpr.slice %base, [%slice1, %slice2] : !asl.lexpr  // Slice target
%3 = asl.lexpr.set_array %base, %index : !asl.lexpr     // Array element target
%4 = asl.lexpr.set_enum_array %base, %enum_index : !asl.lexpr  // Enum array target
%5 = asl.lexpr.set_field %base, "field_name" : !asl.lexpr  // Field target
%6 = asl.lexpr.set_fields %base, ["f1", "f2"] {types = [(!asl.int, !asl.real)]} : !asl.lexpr  // Multiple fields
%7 = asl.lexpr.set_collection_fields "collection", ["field1"] {types = [(!asl.int)]} : !asl.lexpr  // Collection fields
%8 = asl.lexpr.destructuring [%lhs1, %lhs2] : !asl.lexpr  // Tuple destructuring
```

=== Statement Operations

==== Basic Statements
```mlir
asl.stmt.pass                                           // No-operation
asl.stmt.seq %first, %second                           // Sequential composition
asl.stmt.assign %lhs, %rhs                             // Assignment
asl.stmt.call @procedure(%arg1, %arg2) {call_type = "ST_Procedure"}  // Procedure call
asl.stmt.return                                        // Return without value
asl.stmt.return %value                                 // Return with value
```

==== Declaration Statements
```mlir
asl.stmt.decl "LDK_Var", "variable_name", %type, %init_value  // Variable declaration
asl.stmt.decl "LDK_Let", ["x", "y", "z"], %tuple_type, %init_tuple  // Tuple declaration
asl.stmt.decl "LDK_Constant", "const_name", %type       // Constant declaration
```

==== Control Flow Statements
```mlir
asl.stmt.cond %condition {
  // then block
} else {
  // else block
}

asl.stmt.for "index_name", %start, %end {direction = "Up", limit = %limit_expr} {
  // loop body
}

asl.stmt.while %condition {limit = %limit_expr} {
  // loop body  
}

asl.stmt.repeat {limit = %limit_expr} {
  // loop body
} until %condition

asl.stmt.assert %condition                             // Assertion
```

==== Exception Handling Statements
```mlir
asl.stmt.throw                                         // Implicit throw
asl.stmt.throw %exception : !asl.exception<"Error", []>  // Explicit throw

asl.stmt.try {
  // protected code
} catch [
  ("exception_var", !asl.exception<"Error1", []>) : {
    // handler 1
  },
  (none, !asl.exception<"Error2", []>) : {
    // handler 2
  }
] otherwise {
  // otherwise handler
}
```

==== Utility Statements
```mlir
asl.stmt.print %arg1, %arg2 {newline = true, debug = false}  // Print statement
asl.stmt.unreachable                                   // Unreachable statement
asl.stmt.pragma "pragma_name", %arg1, %arg2           // Pragma statement
```

=== Declaration Operations

==== Function Declarations
```mlir
asl.func @function_name(
  %param1: !asl.int {param_name = "width"},
  %arg1: !asl.int,
  %arg2: !asl.real
) -> !asl.bool {
  subprogram_type = "ST_Function",
  qualifier = "Pure",
  override = "Impdef", 
  builtin = false,
  recurse_limit = %limit_expr
} {
  // function body
  asl.stmt.return %result
}

asl.func @procedure_name(%arg: !asl.int) {
  subprogram_type = "ST_Procedure",
  body_type = "SB_Primitive",
  side_effects = true
}
```

==== Global Storage Declarations  
```mlir
asl.global @global_var : !asl.int {
  keyword = "GDK_Var",
  initial_value = %init_expr
}

asl.global @constant : !asl.int {
  keyword = "GDK_Constant", 
  initial_value = %const_value
}

asl.global @config : !asl.string {
  keyword = "GDK_Config"
}
```

==== Type Declarations
```mlir
asl.type_decl @CustomInt : !asl.int<constraints[range(0, 1000)]>

asl.type_decl @StatusBits : !asl.bits<8, [
  simple("error", [0:0]), 
  simple("ready", [1:1]), 
  simple("data", [7:2])
]>

asl.type_decl @ColorEnum : !asl.enum<["RED", "GREEN", "BLUE"]> {
  enum_name = "Color",
  enum_fields = [("RED", []), ("GREEN", []), ("BLUE", [])]
}
```

==== Pragma Declarations
```mlir
asl.pragma "optimization_hint", %param1, %param2      // Global pragma
```

=== Module Structure
```mlir
module @asl_module {
  // Type declarations
  asl.type_decl @MyInt : !asl.int<constraints[range(0, 255)]>
  
  // Global storage
  asl.global @counter : !asl.int {keyword = "GDK_Var", initial_value = %zero}
  
  // Functions
  asl.func @add(%a: !asl.int, %b: !asl.int) -> !asl.int {
    subprogram_type = "ST_Function"
  } {
    %result = asl.plus %a, %b : !asl.int
    asl.stmt.return %result
  }
  
  // Main program (if applicable)
  asl.func @main() {
    subprogram_type = "ST_Procedure"
  } {
    // program statements
    asl.stmt.pass
  }
}
```

This comprehensive IR design provides the complete specification for implementing the ASL MLIR dialect, covering all AST constructs with their corresponding MLIR representations.
