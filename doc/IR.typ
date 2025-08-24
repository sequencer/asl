#let document_title = "MLIR Dialect for ASL"
#set document(title: document_title, author: "Jiuyang Liu")
#set heading(numbering: "1.1")

This document describes the design of an MLIR dialect for the Arm Specification Language (ASL). The dialect provides a lower-level intermediate representation of ASL code that enables analysis, optimization, and code generation while preserving ASL's semantic properties. The design is based on the ASL Abstract Syntax Tree (AST) defined in the `asllib` library and follows the formal semantics specified in the ASL Reference Manual.

= Operations <operations>

== Type <type>
See `ty`
```ml
and type_desc =
  | T_Int of constraint_kind
  | T_Bits of expr * bitfield list
  | T_Real
  | T_String
  | T_Bool
  | T_Enum of identifier list
  | T_Tuple of ty list
  | T_Array of array_index * ty
  | T_Record of field list
  | T_Exception of field list
  | T_Collection of field list
  | T_Named of identifier
```

#table(
  columns: 2,
  [ASL Type], [MLIR Type],
  [`integer`], [`!asl.int<constraint_kind>`],
  [`bits(N)`], [`!asl.bits<width, bitfields>`],
  [`real`], [`!asl.real`],
  [`string`], [`!asl.string`],
  [`boolean`], [`i1`],
  [`enumeration`], [`!asl.enum<labels>`],
  [`tuple`], [`!asl.tuple<types>`],
  [`array`], [`!asl.array<element_type, index>`],
  [`record`], [`!asl.record<fields>`],
  [`exception`], [`!asl.exception<fields>`],
  [`collection`], [`!asl.collection<fields>`],
  [Named Type], [`!asl.named<name>`],
  [Slice Descriptor], [`!asl.slice`],
  [L-Expression], [`!asl.lexpr`],
)

=== Integer Type <t_int>
The `asl.int` type is defined as `!asl.int<constraint>`, where `constraint` is a `ConstraintKindAttr` (see @constraint_kind).

During the parsing procedure, it requires materialization when depending on the function parameter, which type is also `asl.int`, but the `ConstraintKindAttr.kind` is `Parameterized`, and should be explicitly converted by `E_ATC` (see @e_atc_int) when parsing. It will be represented with `asl.expr.atc.int.exact` (see @e_atc_int_exact) or `asl.expr.atc.int.range` (see @e_atc_int_range) operations depending on the constraint kind, the function parameter is the SSA inputs to these two operations.
When not depending on the Parameterized Integer, and being constrained, it should always parsed to `!asl.int<constrained>` type, where `constrained` is a `ConstraintKindAttr` (see @constraint_kind).
Notice, `PendingConstrained` will be forbidden due to we are parsing the typed AST.

=== Bitvector Type <t_bits>
The `asl.bits` type is defined as `!asl.bits<width, bitfields>`, where `width` is an `IntegerAttr` and `bitfields` is an `ArrayAttr` of `BitFieldAttr` (see @bitfield).

During the parsing procedure, it requires materialization when depending on the function parameter, which type is `asl.int`, but the `ConstraintKindAttr.Kind` is `Parameterized`, and should be explicitly converted by `E_ATC` (see @e_atc_int) when parsing. It will be represented with `asl.expr.atc.bits` operation (see @e_atc_bits) with `width` and array of `bitfield` as SSA inputs.
While `asl.expr.atc.bits.bitfields` has three variants and return `!asl.bitfield` type:
- `asl.expr.atc.bits.bitfields.simple`: accepts `slice` (`!asl.slice`) operations;
- `asl.expr.atc.bits.bitfields.nested`: accepts variadic of inputs `slices`(`!asl.slice`) and `e_atc_bits` (`!asl.bits`) as nested bits type, the type should be declared with the ATC operation, the input to the ATC operation is ignored;
- `asl.expr.atc.bits.bitfields.type`: accepts variadic of inputs `slices`(`!asl.slice`) with explicit `TypeAttr`;

=== Real Type <t_real>
The `asl.real` type is `!asl.real` and has no parameters.

=== String Type <t_string>
The `asl.string` type is `!asl.string` and has no parameters.

=== Boolean Type <t_bool>
The `asl.bool` type is represented by the standard `i1` type.

=== Enumeration Type <t_enum>
The `asl.enum` type is `!asl.enum<labels>`, where `labels` is an `ArrayAttr` of `StringAttr`.

=== Tuple Type <t_tuple>
The `asl.tuple` type is `!asl.tuple<types>`, where `types` is an `ArrayAttr` of `TypeAttr`.

=== Array Type <t_array>
The `asl.array` type is `!asl.array<element_type, index>`, where `element_type` is a `TypeAttr` and `index` is an `ArrayIndexAttr` (see @array_index). It should be explicitly converted by `E_ATC` (see @e_atc_array).

=== Record Type <t_record>
The `asl.record` type is `!asl.record<fields>`, where `fields` is an `ArrayAttr` of `RecordFieldAttr`.

=== Exception Type <t_exception>
The `asl.exception` type is `!asl.exception<fields>`, where `fields` is an `ArrayAttr` of `RecordFieldAttr`.

=== Collection Type <t_collection>
The `asl.collection` type is `!asl.collection<fields>`, where `fields` is an `ArrayAttr` of `RecordFieldAttr`.

=== Named Type <t_named>
The `asl.named` type is `!asl.named<name>`, where `name` is a `StringAttr`.

=== BitField Type <t_bitfield>
The `asl.bitfield` type is `!asl.bitfield` and represents a bitfield structure used in bitfield operations and materialization during ATC operations. This type is produced by bitfield ATC operations that require materialization of bitfield structures.

== Utility Operations <utility_operations>
These operations support materialization and intermediate representation needs.

=== Slice Operations <slice_op>
Slice operations appear in both static and dynamic contexts within the ASL AST. These operations represent the generic expression of bitvector and array slicing:

#table(columns: 5,
[ASL Constructor], [MLIR Operation], [MLIR Operands], [MLIR Results], [Description],
[`Slice_Single`], [`asl.slice.single`], [`!asl.int`], [`!asl.slice`], [Takes an index input `i`, returns a slice of length 1 at position `i`],
[`Slice_Range`], [`asl.slice.range`], [`!asl.int`, `!asl.int`], [`!asl.slice`], [Takes start index `j` and end index `i`, returns slice from position `j` to `i-1` (inclusive)],
[`Slice_Length`], [`asl.slice.length`], [`!asl.int`, `!asl.int`], [`!asl.slice`], [Takes start index `i` and length `n`, returns slice of length `n` starting at position `i`],
[`Slice_Star`], [`asl.slice.star`], [`!asl.int`, `!asl.int`], [`!asl.slice`], [Takes factor `i` and length `n`, returns slice starting at position `i * n` with length `n`]
)

== Pattern Matching <e_pattern>
`asl.expr.pattern` represents pattern matching. Takes `expr` as input , and variations version: 
```ml
and pattern_desc =
  | Pattern_All
  | Pattern_Any of pattern list
  | Pattern_Geq of expr
  | Pattern_Leq of expr
  | Pattern_Mask of Bitvector.mask
  | Pattern_Not of pattern
  | Pattern_Range of expr * expr (* lower -> upper, included *)
  | Pattern_Single of expr
  | Pattern_Tuple of pattern list
```
The result of a pattern matching expression is a boolean value.

=== Wildcard Pattern <pattern_all>
`asl.expr.pattern.all` represents a wildcard pattern that matches any value. Takes `expr` as input and returns an `i1`.

=== Disjunctive Pattern <pattern_any>
`asl.expr.pattern.any` represents a disjunctive pattern that matches if any of the sub-patterns match, the sub-patterns are nested in the first block of the first region of this operation. Takes `expr` as input and returns an `i1`.

=== Greater Than or Equal Pattern <pattern_geq>
`asl.expr.pattern.geq` matches values greater than or equal to the given threshold expression. Takes `expr`(`!asl.int` or `!asl.real`) and `threshold`(`!asl.int` or `!asl.real`) as inputs and returns an `i1`.

=== Less Than or Equal Pattern <pattern_leq>
`asl.expr.pattern.leq` matches values less than or equal to the given threshold expression. Has `expr`(`!asl.int` or `!asl.real`) and `threshold`(`!asl.int` or `!asl.real`) as inputs and returns an `i1`. 

=== Bitvector Mask Pattern <pattern_mask>
`asl.expr.pattern.mask` matches expressions against a bitvector mask pattern using the mask stored as an `BitVectorMaskAttr` as attribute, has `expr`(`!asl.bits`) as input and returns an `i1`.

=== Negated Pattern <pattern_not>
`asl.expr.pattern.not` represents the negation of another pattern, the pattern need to be inverted is nested in the first block of the first region of this operation, has `expr` as input and returns an `i1`.

=== Range Pattern <pattern_range>
`asl.expr.pattern.range` matches values within an inclusive range from lower bound to upper bound. Has `lower`(`!asl.int` or `!asl.real`) and `upper`(`!asl.int` or `!asl.real`) and `expr`(`!asl.int` or `!asl.real`) as inputs and returns an `i1`.

=== Single Value Pattern <pattern_single>
`asl.expr.pattern.single` matches a single specific value using equality comparison. Has `value` as inputs for the static comparison value and returns an `i1`.

=== Tuple Pattern <pattern_tuple>
`asl.expr.pattern.tuple` matches tuple expressions against multiple sub-patterns, sub-patterns are nested in the first block of the first region of this operation, has `value` as inputs for the static comparison value and returns an `i1`.

== Expression Operations <expression>
Expression operations correspond to the `expr_desc` type from the ASL AST:

```ml
type expr_desc =
  | E_Literal of literal
  | E_Var of identifier
  | E_ATC of expr * ty  (** Asserted type conversion *)
  | E_Binop of binop * expr * expr
  | E_Unop of unop * expr
  | E_Call of call
  | E_Slice of expr * slice list
  | E_Cond of expr * expr * expr
  | E_GetArray of expr * expr
  | E_GetEnumArray of expr * expr
  | E_GetField of expr * identifier
  | E_GetFields of expr * identifier list
  | E_GetCollectionFields of identifier * identifier list
  | E_GetItem of expr * int
  | E_Record of ty * (identifier * expr) list
  | E_Tuple of expr list
  | E_Array of { length : expr; value : expr }
  | E_EnumArray of { enum : identifier; labels : identifier list; value : expr }
  | E_Arbitrary of ty
  | E_Pattern of expr * pattern
```

=== Literal Expressions <e_literal>

`asl.expr.literal` operations represent compile-time constant values. The literal value is stored as an attribute, with the attribute type determined by the literal's ASL type:

#table(
columns: 6,
[ASL Type], [OCaml Type], [MLIR Operation], [MLIR Result Type], [Serialization], [Deserialization],
[`L_Int`], [#link("https://github.com/ocaml/Zarith/blob/9c9097567d54d79cf3afaa3beda6db53dbf025dd/z.mli")[`Z.t`]], [`asl.expr.literal.int`], [`!asl.int`], [#link("https://github.com/ocaml/Zarith/blob/9c9097567d54d79cf3afaa3beda6db53dbf025dd/z.mli#L361")[`Z.to_string`]], [#link("https://github.com/ocaml/Zarith/blob/9c9097567d54d79cf3afaa3beda6db53dbf025dd/z.mli#L101")[`Z.of_string`]],
[`L_Bool`], [`bool`], [`asl.expr.literal.bool`], [`i1`], [`BoolAttr`], [`BoolAttr`],
[`L_Real`], [#link("https://github.com/ocaml/Zarith/blob/9c9097567d54d79cf3afaa3beda6db53dbf025dd/q.mli")[`Q.t`]], [`asl.expr.literal.real`], [`!asl.real`], [#link("https://github.com/ocaml/Zarith/blob/9c9097567d54d79cf3afaa3beda6db53dbf025dd/q.mli#L164")[`Q.to_string`]], [#link("https://github.com/ocaml/Zarith/blob/9c9097567d54d79cf3afaa3beda6db53dbf025dd/q.mli#L")[`Q.of_string`]],
[`L_BitVector`], [#link("https://github.com/herd/herdtools7/blob/54d92b2af30deb11fefe4595b8922fddf6734dce/asllib/bitvector.mli#L49")[`Bitvector.t`]], [`asl.expr.literal.bitvector`], [`!asl.bits`], [#link("https://github.com/herd/herdtools7/blob/54d92b2af30deb11fefe4595b8922fddf6734dce/asllib/bitvector.mli#L80")[`Bitvector.to_string`]], [#link("https://github.com/herd/herdtools7/blob/54d92b2af30deb11fefe4595b8922fddf6734dce/asllib/bitvector.mli#L49")[`Bitvector.of_string`]],
[`L_String`], [`string`], [`asl.expr.literal.string`], [`!asl.string`], [`StringAttr`], [`StringAttr`],
[`L_Label`], [`string`], [`asl.expr.literal.label`], [`!asl.label`], [`StringAttr`], [`StringAttr`],
)

Note: `L_Int` and `L_BitVector` literals are unconstrained by default. To apply type constraints, use the Asserted Type Conversion operation (@e_atc).

=== Variable Reference <e_var>
`asl.expr.var` represents variable references. The operation resolves variable names to SSA values through local symbol table lookup during parsing. It has a `StringAttr` attribute for the variable identifier. The result type is the type of the variable.

=== Asserted Type Conversion <e_atc>
`asl.expr.atc` implements Asserted Type Conversion as specified in ASL Specification `TypingRule.ATC`. This operation is used for compile-time type materialization and constraint checking. It takes an expression aIt takes an expression andnd a type attribute, and returns a value of the given type.

==== Integer Type Conversion <e_atc_int>
`asl.expr.atc.int` accepts a list of constraint operations:

===== Exact Constraint <e_atc_int_exact>
`asl.expr.atc.int.exact` constrains the value to exactly match a compile-time constant.

===== Range Constraint <e_atc_int_range>
`asl.expr.atc.int.range` constrains the value to lie within an inclusive range defined by `lhs`(`!asl.int`) and `rhs`(`!asl.int`) bounds.

==== Bitvector Type Conversion <e_atc_bits>
`asl.expr.atc.bits` has two operands, returns a `!asl.bits` type:
- `width`(`!asl.int`): expression defining the bitvector size
- `bitfields`: `ArrayAttr` of `BitFieldAttr` defining bitfield layout (see @bitfield)

===== Bitfield Definitions <e_atc_bits_bitfields>
`asl.expr.atc.bits.bitfields` supports three variants:
- `asl.expr.atc.bits.bitfields.simple`: accepts `slice`(`!asl.slice`) operations;
- `asl.expr.atc.bits.bitfields.nested`: accepts variadic of inputs `slices`(`!asl.slice`) and `e_atc_bits` (`!asl.bits`) as nested bits type;
- `asl.expr.atc.bits.bitfields.type`: accepts variadic of inputs `slices`(`!asl.slice`) and the optional `type` input when requires materialization, if not requiring materialization, the `type` should be `TypeAttr`;

==== Array Type Conversion <e_atc_array>
`asl.expr.atc.array` takes a `length`(`!asl.int`) operand for array size specification using `ArrayIndexAttr`, and the optional `element_type` input when requires materialization, if not requiring materialization, the `element_type` should be `TypeAttr`;

=== Binary Operator Expressions <e_binop>
`asl.expr.binop` represents binary operator expressions:
#table(
columns: 4,
  [Operator], [MLIR Operation], [MLIR Operands], [MLIR Results],
  [AND], [`asl.expr.binop.and`], [`!asl.bits`, `!asl.bits`], [`!asl.bits`],
  [BAND], [`asl.expr.binop.band`], [`i1`, `i1`], [`i1`],
  [BEQ], [`asl.expr.binop.beq`], [`i1`, `i1`], [`i1`],
  [BOR], [`asl.expr.binop.bor`], [`i1`, `i1`], [`i1`],
  [DIV], [`asl.expr.binop.div`], [`!asl.int`, `!asl.int`], [`!asl.int`],
  [DIVRM], [`asl.expr.binop.divrm`], [`!asl.int`, `!asl.int`], [`!asl.int`],
  [XOR], [`asl.expr.binop.xor`], [`!asl.bits`, `!asl.bits`], [`!asl.bits`],
  [EQ_OP], [`asl.expr.binop.eq`], [`any`, `any`], [`i1`],
  [GT], [`asl.expr.binop.gt`], [`!asl.int`, `!asl.int`], [`i1`],
  [GEQ], [`asl.expr.binop.geq`], [`!asl.int`, `!asl.int`], [`i1`],
  [IMPL], [`asl.expr.binop.impl`], [`i1`, `i1`], [`i1`],
  [LT], [`asl.expr.binop.lt`], [`!asl.int`, `!asl.int`], [`i1`],
  [LEQ], [`asl.expr.binop.leq`], [`!asl.int`, `!asl.int`], [`i1`],
  [MOD], [`asl.expr.binop.mod`], [`!asl.int`, `!asl.int`], [`!asl.int`],
  [MINUS], [`asl.expr.binop.minus`], [`any`, `any`], [`any`],
  [MUL], [`asl.expr.binop.mul`], [`any`, `any`], [`any`],
  [NEQ], [`asl.expr.binop.neq`], [`any`, `any`], [`i1`],
  [OR], [`asl.expr.binop.or`], [`!asl.bits`, `!asl.bits`], [`!asl.bits`],
  [PLUS], [`asl.expr.binop.plus`], [`any`, `any`], [`any`],
  [POW], [`asl.expr.binop.pow`], [`!asl.int`, `!asl.int`], [`!asl.int`],
  [RDIV], [`asl.expr.binop.rdiv`], [`!asl.real`, `!asl.real`], [`!asl.real`],
  [SHL], [`asl.expr.binop.shl`], [`!asl.int`, `!asl.int`], [`!asl.int`],
  [SHR], [`asl.expr.binop.shr`], [`!asl.int`, `!asl.int`], [`!asl.int`],
  [CONCAT], [`asl.expr.binop.concat`], [`any`, `any`], [`any`],
)

=== Unary Operator Expressions <e_unop>
`asl.expr.unop` represents unary operator expressions:
#table(columns: 4,
  [Operator], [MLIR Operation], [MLIR Operands], [MLIR Results],
  [BNOT], [`asl.expr.unop.bnot`], [`i1`], [`i1`],
  [NEG], [`asl.expr.unop.neg`], [`!asl.int`], [`!asl.int`],
  [NOT], [`asl.expr.unop.not`], [`!asl.bits`], [`!asl.bits`],
)

=== Subprogram Invocation <e_call>
`asl.expr.call` represents a subprogram invocation with return value. It has `name` (`StringAttr`) for the callee, variadic `args` and `params` inputs, a `params_size` `IntegerAttr` for size of `params`, and `call_type` (`SubprogramTypeAttr`) for the subprogram kind. The result type is the return type of the callee.

=== Slicing Operations <e_slice>
`asl.expr.slice` represents slicing operations on bitvectors and integers. It takes a `base` input (`!asl.bits` or `!asl.int`) and has `slice` attributes (see @slice_op). The result is of type `!asl.bits`.

=== Conditional Expressions <e_cond>
`asl.expr.cond` represents conditional expressions (ternary operator). Takes three inputs: `condition` (`i1`), `then_expr`, and `else_expr`. The `then_expr` and `else_expr` must have the same type, which is the result type of the operation.

=== Array Access with Integer Index <e_getarray>
`asl.expr.get_array` represents array access with integer index. Takes `base` (`!asl.array`) and `index` (`!asl.int`) inputs. The result type is the element type of the array.

=== Array Access with Enumeration Index <e_getenumarray>
`asl.expr.get_enum_array` represents array access with enumeration index. Takes `base` (`!asl.array`) and `key` (`!asl.label`) inputs. The result type is the element type of the array.

=== Record Field Access <e_getfield>
`asl.expr.get_field` represents record field access. Takes `record` (`!asl.record`) input and has `field_name` (`StringAttr`) attribute. The result type is the type of the field.

=== Accessing Multiple Record Fields <e_getfields>
`asl.expr.get_fields` represents accessing multiple record fields for bit-packing. Takes `record` (`!asl.record`) input and has `field_names` (`ArrayAttr` of `StringAttr`) attribute. The result is of type `!asl.bits`.

=== Tuple Element Access <e_getitem>
`asl.expr.get_item` represents tuple element access. Takes tuple (`!asl.tuple`) input and has `index` (`IntegerAttr`) attribute. The result type is the type of the element at the given index.

=== Record Construction <e_record>
`asl.expr.record` represents record construction. Has `record_type` (`TypeAttr`) attribute for record type and takes field values as inputs with corresponding `field_names` (`ArrayAttr` of `StringAttr`) attribute. The result is of the given record type.

=== Tuple Construction <e_tuple>
`asl.expr.tuple` represents tuple construction. Takes variadic inputs for tuple elements. The result is a `!asl.tuple` of the types of the elements.

=== Array Construction <e_array>
`asl.expr.array` represents array construction. Taking `value` and `length` as inputs to create an array of the specified length filled with the given value. The result is a `!asl.array`.

=== Enumeration-Indexed Array Construction <e_enumarray>
`asl.expr.enum_array` represents enumeration-indexed array construction. Has `enum` (`StringAttr`) and `labels` (`ArrayAttr` of `StringAttr`) attributes, takes `value` input. The result is a `!asl.array`.

=== Arbitrary Value <e_arbitrary>
`asl.expr.arbitrary` represents non-deterministic values. Has `type` (`TypeAttr`) attribute to specify the type of the arbitrary value. The result is of the given type.

== L-Expressions <l_expressions>
See `lexpr`. The left-hand side of assignments, has one result value.
```ml
type lexpr_desc =
  | LE_Discard
  | LE_Var of identifier
  | LE_Slice of lexpr * slice list
  | LE_SetArray of lexpr * expr
  | LE_SetEnumArray of lexpr * expr
  | LE_SetField of lexpr * identifier
  | LE_SetFields of lexpr * identifier list * (int * int) list
  | LE_SetCollectionFields of identifier * identifier list * (int * int) list
  | LE_Destructuring of lexpr list
```
#table(
  columns: 4,
  [ASL Constructor], [MLIR Operation], [MLIR Operands], [MLIR Results],
  [`LE_Discard`], [`asl.lexpr.discard`], [], [`!asl.lexpr`],
  [`LE_Var`], [`asl.lexpr.var`], [], [`!asl.lexpr`],
  [`LE_Slice`], [`asl.lexpr.slice`], [`!asl.lexpr`, `variadic<!asl.slice>`], [`!asl.lexpr`],
  [`LE_SetArray`], [`asl.lexpr.set_array`], [`!asl.lexpr`, `any`], [`!asl.lexpr`],
  [`LE_SetEnumArray`], [`asl.lexpr.set_enum_array`], [`!asl.lexpr`, `any`], [`!asl.lexpr`],
  [`LE_SetField`], [`asl.lexpr.set_field`], [`!asl.lexpr`], [`!asl.lexpr`],
  [`LE_SetFields`], [`asl.lexpr.set_fields`], [`!asl.lexpr`], [`!asl.lexpr`],
  [`LE_SetCollectionFields`], [`asl.lexpr.set_collection_fields`], [], [`!asl.lexpr`],
  [`LE_Destructuring`], [`asl.lexpr.destructuring`], [`variadic<!asl.lexpr>`], [`!asl.lexpr`],
)
=== Discard L-Expression <le_discard>
Mapped to `asl.lexpr.discard` represents a discarded assignment target (e.g., `-` in tuple destructuring). It takes no inputs and returns a `!asl.lexpr` type.
=== Variable L-Expression <le_var>
Mapped to `asl.lexpr.var` used to represents assignment to a variable, it has one `name` (`StringAttr`) attribute mapping to the variable, which is used to query local symbol. It takes no inputs and returns a `!asl.lexpr` type.
=== Slice L-Expression <le_slice>
Mapped to `asl.lexpr.slice`, represents a write to an array given by the l-expression `base` at index. It takes `base` (`!asl.lexpr`) and variadic `slices` (`!asl.slice`) as inputs (see @slice_op), and returns a `!asl.lexpr` type.
=== Set Array Element L-Expression <le_setarray>
Mapped to `asl.lexpr.set_array`. Represents assignment to an array element with integer index. It takes two inputs: `base` (`!asl.lexpr`) and `index` (`!asl.int`), and returns a `!asl.lexpr` type.
=== Set Enum Array Element L-Expression <le_setenumarray>
Mapped to `asl.lexpr.set_enum_array`. Represents assignment to an array element with enum index. It takes two inputs: `base` (`!asl.lexpr`), and `index` (`StringAttr`), and returns a `!asl.lexpr` type.
=== Set Field L-Expression <le_setfield>
Mapped to `asl.lexpr.set_field`. Represents assignment to a record field. It takes one `base` (`!asl.lexpr`) input, and has `field_name` (`StringAttr`) attribute for the field identifier. The result is of type `!asl.lexpr`.
=== Set Multiple Fields L-Expression <le_setfields>
Mapped to `asl.lexpr.set_fields`. Represents assignment to multiple record fields with type annotations. It takes a `base` (`!asl.lexpr`) input, and has two attributes: `field_names` (`ArrayAttr` of `StringAttr`) and `annotations` (`ArrayAttr` of `DictionaryAttr` with `x` and `y` as `IntegerAttr`) for type annotations. The result is of type `!asl.lexpr`.
=== Set Collection Fields L-Expression <le_setcollectionfields>
Mapped to `asl.lexpr.set_collection_fields`. Represents assignment to collection fields with type annotations. It has three attributes: `collection` (`StringAttr`), `field_names` (`ArrayAttr` of `StringAttr`), and `annotations` (`ArrayAttr` of `DictionaryAttr` with `x` and `y` as `IntegerAttr`) for type annotations. It takes no inputs and returns a `!asl.lexpr` type.
=== Tuple Destructuring L-Expression <le_destructuring>
Mapped to `asl.lexpr.destructuring`. Represents tuple destructuring assignment. It takes a variadic number of l-expressions (`!asl.lexpr`) as inputs, and returns a `!asl.lexpr` type.

== Statement <statement>
See `stmt`, operations with no SSA outputs.
```ml
type stmt_desc =
  | S_Pass
  | S_Seq of stmt * stmt
  | S_Decl of local_decl_keyword * local_decl_item * ty option * expr option
  | S_Assign of lexpr * expr
  | S_Call of call
  | S_Return of expr option
  | S_Cond of expr * stmt * stmt
  | S_Assert of expr
  | S_For of {
      index_name : identifier;
      start_e : expr;
      dir : for_direction;
      end_e : expr;
      body : stmt;
      limit : expr option;
    }
  | S_While of expr * expr option * stmt
  | S_Repeat of stmt * expr * expr option
  | S_Throw of (expr * ty option) option
  | S_Try of stmt * catcher list * stmt option
  | S_Print of { args : expr list; newline : bool; debug : bool }
  | S_Unreachable
  | S_Pragma of identifier * expr list
```
#table(
  columns: 3,
  [ASL Constructor], [MLIR Operation], [MLIR Operands],
  [`S_Pass`], [`asl.stmt.pass`], [],
  [`S_Seq`], [`asl.stmt.seq`], [],
  [`S_Decl`], [`asl.stmt.decl`], [`optional<any>`],
  [`S_Assign`], [`asl.stmt.assign`], [`!asl.lexpr`, `any`],
  [`S_Call`], [`asl.stmt.call`], [`variadic<any>`, `variadic<any>`],
  [`S_Return`], [`asl.stmt.return`], [`optional<any>`],
  [`S_Cond`], [`asl.stmt.cond`], [`i1`],
  [`S_Assert`], [`asl.stmt.assert`], [`i1`],
  [`S_For`], [`asl.stmt.for`], [`!asl.int`, `!asl.int`, `optional<...>`],
  [`S_While`], [`asl.stmt.while`], [`i1`, `optional<...>`],
  [`S_Repeat`], [`asl.stmt.repeat`], [`i1`, `optional<...>`],
  [`S_Throw`], [`asl.stmt.throw`], [`optional<any>`],
  [`S_Try`], [`asl.stmt.try`], [],
  [`S_Print`], [`asl.stmt.print`], [`variadic<any>`],
  [`S_Unreachable`], [`asl.stmt.unreachable`], [],
  [`S_Pragma`], [`asl.stmt.pragma`], [`variadic<any>`],
)

=== No-Operation Statement <s_pass>
Mapped to `asl.stmt.pass`. A no-operation statement.

=== Sequential Composition Statement <s_seq>
Mapped to `asl.stmt.seq`. Sequential composition of two statements.
Has a block of region with two nested statements.

=== Local Storage Declaration Statement <s_decl>
Mapped to `asl.stmt.decl`. Local storage declaration.
Has optional `initial_value` input and attributes: `keyword` (`LDKAttr`), `item` (`LDIAttr`), and optional `type` (`TypeAttr`).

=== Assignment Statement <s_assign>
Mapped to `asl.stmt.assign`. Assignment statement. Takes `lhs` (l-expression) and `rhs` (value) inputs.

=== Procedure Call Statement <s_call>
Mapped to `asl.stmt.call`. Procedure call without return value.
Similar to @e_call but for statements.

=== Return Statement <s_return>
Mapped to `asl.stmt.return`. Return statement with optional expression input.

=== Conditional Statement <s_cond>
Mapped to `asl.stmt.cond`. Conditional statement (if-then-else).
Takes `condition` input, has one region with two blocks for then/else branches.

=== Assertion Statement <s_assert>
Mapped to `asl.stmt.assert`. Assertion statement. Takes boolean expression input.

=== For Loop Statement <s_for>
Mapped to `asl.stmt.for`. For loop statement.
Attributes: `index_name` (`StringAttr`), `direction` (`ForDirectionAttr`).
Inputs: `start`, `end` for loop bounds, optional `limit` for static limit.
Body in first region.

=== While Loop Statement <s_while>
Mapped to `asl.stmt.while`. While loop with condition input.
Input: optional `limit` for static limit.
Body in first region.

=== Repeat-Until Loop Statement <s_repeat>
Mapped to `asl.stmt.repeat`. Repeat-until loop.
Input: optional `limit` for static limit.
Body in first region.

=== Exception Throwing Statement <s_throw>
Mapped to `asl.stmt.throw`. Exception throwing statement.
Optional expression input and type annotation.

=== Try-Catch Statement <s_try>
Mapped to `asl.stmt.try`. Try-catch statement.
First region contains protected statement.
Second region contains catch handlers and optional otherwise block.

=== Print Statement <s_print>
Mapped to `asl.stmt.print`. Print statement.
Attributes: `newline` (`BoolAttr`) and `debug` (`BoolAttr`).
Takes variadic expression inputs.

=== Unreachable Statement <s_unreachable>
Mapped to `asl.stmt.unreachable`. Marks unreachable code.

=== Tool-specific Pragma Statement <s_pragma>
Mapped to `asl.stmt.pragma`. Tool-specific pragma.
Has `identifier` (`StringAttr`) attribute and takes variadic expression inputs;

== Declaration <declaration>
See `decl`,
```ml
type decl_desc =
  | D_Func of func
  | D_GlobalStorage of global_decl
  | D_TypeDecl of identifier * ty * (identifier * field list) option
  | D_Pragma of identifier * expr list
```
#table(
  columns: 3,
  [ASL Constructor], [MLIR Operation], [MLIR Operands],
  [`D_Func`], [`asl.func`], [`optional<any>`],
  [`D_GlobalStorage`], [`asl.global`], [`any`],
  [`D_TypeDecl`], [`asl.type_decl`], [],
  [`D_Pragma`], [`asl.pragma`], [`variadic<any>`],
)

=== Function Declaration <d_func>
`D_Func` node encapsulates a func record, which has the following attributes:
- `name` (`StringAttr`) for the function name
- `parameters` (`ArrayAttr` of `DictionaryAttr` with `identifier` (`StringAttr`) and `type` (`TypeAttr`))
- `return_type` (optional `TypeAttr`) to indicate the return type
- `primitive` (`BoolAttr`) to indicate if the function is primitive
- `args` (`ArrayAttr` of `StringAttr`) for function argument names, with types in `BlockArgument`
- `args_types` (`ArrayAttr` of `TypeAttr`) for function argument types
- `subprogram_type` (`SubprogramTypeAttr`)
- `qualifier` (optional `FuncQualifierAttr`)
- `override` (optional `OverrideInfoAttr`)
- `builtin` (`BoolAttr`), treated specially when checking parameters at call sites
It has an optional input:
- `recurse_limit`(`!asl.int`) to indicate the maximum static recursion limit

The function body is in the first block of the first region.

=== Global Variable Declaration <d_globalstorage>
`D_GlobalStorage` node declares a global variable record, which has the following attributes:
- `keyword` (`GDKAttr`)
- `name` (`StringAttr`)
- `type` (`TypeAttr`)
It takes `initial_value` as SSA input.

=== Type Declaration <d_typedecl>
`D_TypeDecl` node declares type information with the following attributes:
- `identifier` (`StringAttr`) for the type alias name
- `type` (`TypeAttr`) for the underlying type
- `subtypes` (optional `DictionaryAttr` with `identifier` (`StringAttr`) and `fields` (`ArrayAttr` of `DictionaryAttr` with `identifier` (`StringAttr`) and `type` (`TypeAttr`)))

=== Pragma Declaration <d_pragma>
`D_Pragma` node declares pragma information with the following attributes:
- `identifier` (`StringAttr`) for the pragma name
It takes variadic expression inputs.

= Attributes <attr>
== SubprogramTypeAttr (`subprogram_type`) <subprogram_type>

Stored in `StrEnumAttr`:
#table(columns: 3,
[`procedure`],[`ST_Procedure`], [A procedure is a subprogram without return type, called from a statement.],
[`function`],[`ST_Function`], [A function is a subprogram with a return type, called from an expression.],
[`getter`],[`ST_Getter`], [A getter is a special function called with a syntax similar to slices.],
[`emptygetter`],[`ST_EmptyGetter`], [An empty getter is a special function called with a syntax similar to a variable. This is relevant only for V0.],
[`setter`],[`ST_Setter`], [A setter is a special procedure called with a syntax similar to slice assignment.],
[`emptysetter`],[`ST_EmptySetter`], [An empty setter is a special procedure called with a syntax similar to an assignment to a variable. This is relevant only for V0.],
)

== BitVectorMask Attribute <bitvector_mask>
`BitVectorMaskAttr` is defined with `StringAttr`, it use #link("https://github.com/herd/herdtools7/blob/54d92b2af30deb11fefe4595b8922fddf6734dce/asllib/bitvector.mli#L209")[mask_to_string] to serialize and #link("https://github.com/herd/herdtools7/blob/54d92b2af30deb11fefe4595b8922fddf6734dce/asllib/bitvector.mli#L199C5-L199C19")[mask_of_string] to parse.

== ConstraintKind Attribute <constraint_kind>
`ConstraintKindAttr` is defined with `DictionaryAttr` with the following elements:

=== Kind (`kind`) <constraint_kind_kind>
Stored in `StrEnumAttr`:
#table(columns: 3,
[Enum Name], [AST Name], [Description],
[`unconstrained`],[`UnConstrained`], [The normal, unconstrained integer type],
[`constrained`],[`WellConstrained`], [An integer type constrained from ASL syntax],
[`pending`],[`PendingConstrained`], [An integer type whose constraint will be inferred during type-checking],
[`parameterized`],[`Parameterized`], [A parameterized integer with unique identifier],
)

=== Constraint List (`constraint_list`) <constraint_kind_constraint_list>
Stored in `ArrayAttr`, the contents is be `IntConstraintAttr`, see @int_constraint

=== Precision Flag (`precision_loss_flag`) <constraint_kind_precision_flag>
Stored in `PrecisionLossFlagAttr`

== IntConstraint Attribute <int_constraint>
`IntConstraintAttr` is defined with `DictionaryAttr` with the following elements:

=== Exact `exact` <int_constraint_exact>
It is matched from `Constraint_Exact`, represent exactly a value, as given by a statically evaluable expression. 
The expressing of defining the operations is store as `IntegerAttr`, binding to the operation.

=== RHS (`rhs`) <int_constraint_rhs>
It is matched from `Constraint_Range`, In the inclusive range of these two statically evaluable values.
The expressing of defining the operations is store as `IntegerAttr`, binding to the operation.

=== LHS (`lhs`) <int_constraint_lhs>
It is matched from `Constraint_Range`, In the inclusive range of these two statically evaluable values.
The expressing of defining the operations is store as `IntegerAttr`, binding to the operation.

== PrecisionLossFlagAttr <precision_loss_flag>
`PrecisionLossFlagAttr` is defined with `StrEnumAttr`, indicates if any precision loss occurred.
#table(columns: 3,
[Enum Name], [AST Name], [Description],
[`full`],[`Precision_Full`], [No loss of precision],
[`lost`],[`Precision_Lost`], [A loss of precision comes with a list of warnings that can explain why the loss of precision happened(the warning is ignored for now.)]
)

== BitField Kind Attribute <bitfield>
`BitFieldAttr` is defined with `DictionaryAttr` with the following elements:

=== Identifier (`identifier`) <bitfield_identifier>
Stored in `StringAttr`, the name of this bit field.

=== Kind (`kind`) <bitfield_kind>
Stored in `StrEnumAttr`:
#table(columns: 3,
[Enum Name], [AST Name], [Description],
[`simple`],[`BitField_Simple`],[A name and its corresponding slice],
[`nested`],[`BitField_Nested`],[A name, its corresponding slice and some nested bitfields.],
[`type`],[`BitField_Type`],[A name, its corresponding slice and the type of the bitfield.],
)

=== Slices (`slices`) <bitfield_slices>
Stored in `ArrayAttr`, the contents are `SliceAttr`, see @slice

=== Nested (`nested`) <bitfield_nested>
Stored in `ArrayAttr`, the contents are `BitFieldAttr`, only applied when `kind` is `nested`, see @bitfield.

=== Type (`type`) <bitfield_type>
Stored in `TypeAttr`, represent the type of this bitfield.

== Slice Attribute <slice>
`SliceAttr` is defined with `DictionaryAttr` with the following elements, after @slice_op been resolved at compile time, it will be converted to this this attribute.

=== Kind (`kind`) <slice_kind>
Stored in `StrEnumAttr`:
#table(columns: 3,
[`single`],[`Slice_Single`], [`Slice_Single(i)` is the slice of length $1$ at position $i$.],
[`range`],[`Slice_Range`], [`Slice_Range(j, i)` denotes the slice from $i$ to $j - 1$.],
[`length`],[`Slice_Length`], [`Slice_Length(i, n)` denotes the slice starting at $i$ of length $n$],
[`star`],[`Slice_Star`], [`Slice_Start(factor, length)` denotes the slice starting at $#text("factor") * #text("length")$ of length $n$.]
)

=== Index (`idx`) <slice_idx>
Stored in `IntegerAttr`, only for `single`

=== LHS (`lhs`) <slice_lhs>
Stored in `IntegerAttr`, only for `range`

=== RHS (`rhs`) <slice_rhs>
Stored in `IntegerAttr`, only for `range`

=== Start (`start`) <slice_start>
Stored in `IntegerAttr`, only for `length`

=== Factor (`factor`) <slice_factor>
Stored in `IntegerAttr`, only for `star`

=== Length (`length`) <slice_length>
Stored in `IntegerAttr`, for `length` and `star`

== ArrayIndexAttr <array_index>
`ArrayIndexAttr` is defined with `DictionaryAttr` with the following elements:

=== Kind (`kind`) <array_index_kind>
#table(columns: 3,
[`int_type`],[`ArrayLength_Expr`], [An integer expression giving the length of the array.],
[`enum_type`],[`ArrayLength_Enum`], [An enumeration name and its list of labels.],
)

=== Length (`length`) <array_index_length>
An integer expression giving the length of the array. Stored in `IntegerAttr`.

=== Identifier (`identifier`) <array_index_identifier>
An enumeration name and its list of labels. Stored in `StringAttr`, only applied for `enum`.

=== Labels (`labels`) <array_index_labels>
Stored in `ArrayAttr`, the contents are `StringAttr`.

== RecordFieldAttr (`record_field`) <record_field>
`RecordFieldAttr` is a field of a record-like structure, defined with `DictionaryAttr` with the following elements:

=== Identifier (`identifier`) <record_field_identifier>
An name its field. Stored in `StringAttr`

=== Type (`type`) <record_field_type>
Stored in `TypeAttr`, represent the type of this field.

== LDKAttr (`local_decl_keyword`)
Stored in `StrEnumAttr`:
#table(columns: 3,
[`var`],[`LDK_Var`], [Mutable local storage declaration],
[`constant`],[`LDK_Constant`], [Constant storage declaration, must be known at compile time],
[`let`],[`LDK_Let`], [Immutable local storage declaration],
)

== GDKAttr (`global_decl_keyword`)
Stored in `StrEnumAttr`:
#table(columns: 3,
[`constant`],[`GDK_Constant`], [Constant storage declaration, must be known at compile time],
[`config`],[`GDK_Config`], [A configurable implementation-specific value that can be overridden by external tools],
[`let`],[`GDK_Let`], [Immutable global storage declaration],
[`var`],[`GDK_Var`], [Mutable global storage declaration],
)

== LDIAttr (`local_decl_item`) <local_decl_item>

=== Kind (`kind`) <local_decl_item_kind>
#table(columns: 3,
[`var`],[`LDI_Var`], [`LDI_Var` is the variable declaration of the variable `x`],
[`tuple`],[`LDI_Tuple`], [`LDI_Tuple` is the tuple declarations of names],
)

=== Var (`var`) <local_decl_item_var>
The name of the local declaration item. Stored in `StringAttr`.

=== Tuple (`tuple`) <local_decl_item_tuple>
The names of the local declaration item. Stored in `ArrayAttr` of `StringAttr`.

== ForDirectionAttr (`for_direction`)
The direction of for iteration, Stored in `StrEnumAttr`:
#table(columns: 3,
[`up`],[`Up`], [The step is constant $+1$],
[`down`],[`Down`], [The step is constant $-1$],
)

== FuncQualifierAttr (`func_qualifier`) <func_qualifier>
Stored in `StrEnumAttr`:
#table(columns: 3,
[`pure`],[`Pure`], [A `pure` subprogram does not read or modify mutable state. It can be called in types.],
[`readonly`],[`Readonly`], [A `readonly` subprogram can read mutable state but not modify it. It can be called in assertions.],
[`noreturn`],[`Noreturn`], [A `noreturn` subprogram always terminates by a thrown exception or calling `Unreachable`.],
)

== OverrideInfoAttr (`override_info`) <override_info>
Stored in `StrEnumAttr`:
#table(columns: 3,
[Enum Name], [AST Name], [Description],
[`impdef`],[`Impdef`], [A subprogram which can be overridden],
[`implementation`],[`Implementation`], [A subprogram which overrides a corresponding `impdef` subprogram]
)

== VersionAttr (`version`) <version>
The `version` field (`V0` or `V1`) is preserved by attaching a custom `asl.version` attribute to each operation.
Stored in `StrEnumAttr`:
#table(columns: 3,
[Enum Name], [AST Name], [Description],
[`v0`],[`V0`], [Legacy version],
[`v1`],[`V1`], [Current version]
)
