#let document_title = "ASL Dialect Design Review"
#set document(title: document_title, author: "Jiuyang Liu")
#set heading(numbering: "1.1")
#set text(font: "New Computer Modern", size: 11pt)

= ASL Dialect Design Review

This document analyzes the ASL dialect implementation against the herdtools7 ASL specification, identifying design considerations and recommendations.

== Review Methodology

The review compares:
- herdtools7 ASL specification (Types.tex, Expressions.tex, Statements.tex, Slicing.tex, RelationsOnTypes.tex)
- herdtools7 AST.mli at pinned commit `d7d6bdd24f8680c4abf2df3a3e54a9d98494321e`
- Current ASLRational.typ documentation
- ASL dialect TableGen definitions
- JSON backend implementation

== JSON Backend Coverage

The JSON backend (`asl-json-backend`) correctly serializes all AST constructs from the pinned herdtools7 version.

=== Binary Operators - Complete

All 24 binary operators from the pinned herdtools7 version are handled:

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, center),
  [*Category*], [*Operators*], [*Status*],
  [Arithmetic], [`PLUS`, `MINUS`, `MUL`, `DIV`, `DIVRM`, `MOD`, `POW`, `RDIV`], [OK],
  [Bitwise], [`AND`, `OR`, `XOR`, `SHL`, `SHR`], [OK],
  [Boolean], [`BAND`, `BOR`, `BEQ`, `IMPL`], [OK],
  [Comparison], [`EQ_OP`, `NEQ`, `LT`, `LEQ`, `GT`, `GEQ`], [OK],
  [Other], [`CONCAT`], [OK],
)

Note: `BIC` (bit clear) is not present in the pinned herdtools7 version. It was added in a later commit.

=== Unary Operators - Complete

#table(
  columns: (auto, auto, auto),
  inset: 6pt,
  align: (left, left, center),
  [*Operator*], [*Description*], [*Status*],
  [`BNOT`], [Boolean inversion], [OK],
  [`NEG`], [Integer/real negation], [OK],
  [`NOT`], [Bitvector bitwise inversion], [OK],
)

=== Types - Complete

All type constructors are handled: `T_Int`, `T_Bits`, `T_Real`, `T_String`, `T_Bool`, `T_Enum`, `T_Tuple`, `T_Array`, `T_Record`, `T_Exception`, `T_Collection`, `T_Named`.

=== Statements - Complete

All statement types are handled: `S_Pass`, `S_Seq`, `S_Decl`, `S_Assign`, `S_Call`, `S_Return`, `S_Cond`, `S_Assert`, `S_For`, `S_While`, `S_Repeat`, `S_Throw`, `S_Try`, `S_Print`, `S_Unreachable`, `S_Pragma`.

=== Expressions - Complete

All expression types are handled: `E_Literal`, `E_Var`, `E_ATC`, `E_Binop`, `E_Unop`, `E_Call`, `E_Slice`, `E_Cond`, `E_GetArray`, `E_GetEnumArray`, `E_GetField`, `E_GetFields`, `E_GetCollectionFields`, `E_GetItem`, `E_Record`, `E_Tuple`, `E_Array`, `E_EnumArray`, `E_Arbitrary`, `E_Pattern`.

== Semantic Issues

The following issues relate to semantic interpretation in the MLIR dialect, not JSON serialization.

=== High Severity

==== Integer Division Semantics

*Problem:* The ASL spec defines `DIV` as exact integer division, while `DIVRM` provides floor division toward negative infinity. The dialect documentation does not clarify these semantics.

*herdtools7 AST.mli:*
```ocaml
| `DIV      (** Integer division *)
| `DIVRM    (** Inexact integer division, with rounding towards negative infinity *)
```

*Recommendation:* Document that:
- `DIV` performs exact integer division (result undefined if not divisible)
- `DIVRM` performs floor division (rounds toward negative infinity)

==== Short-Circuit Semantics

*Problem:* ASL specifies short-circuit evaluation for `BAND`, `BOR`, and `IMPL`. The current `Pure` trait on expression operations does not capture lazy evaluation semantics.

*Recommendation:* Either:
- Add regions for short-circuit evaluation (similar to `scf.if`)
- Document that short-circuit semantics are handled at lowering
- Add a `ShortCircuit` trait or attribute

==== Bitvector vs Integer Arithmetic

*Problem:* ASL bitvector arithmetic wraps around (unsigned modular arithmetic), while integer arithmetic has unbounded precision. The dialect uses `AnyType` for arithmetic operations without documenting type-dependent behavior.

*Recommendation:* Document type-dependent semantics:
- `!asl.int`: Arbitrary precision, no overflow
- `!asl.bits`: Fixed-width, unsigned wraparound

=== Medium Severity

==== Type Satisfaction (Frontend-Handled)

*Problem:* ASL has complex type satisfaction rules distinct from structural equality. Named types use identity-based comparison with explicit subtype declarations.

*Current status:* The herdtools7 frontend performs all type checking before JSON serialization. The MLIR dialect receives well-typed AST.

*Recommendation:* Document that type satisfaction is verified by the herdtools7 frontend, not the MLIR dialect.

==== Symbolically Evaluable Expressions (Frontend-Handled)

*Problem:* ASL requires certain expressions (array lengths, bitvector widths, constraints) to be "symbolically evaluable" - involving only immutable values.

*Current status:* The herdtools7 frontend verifies symbolic evaluability during type checking.

*Recommendation:* Document that symbolic evaluability is verified by the herdtools7 frontend.

==== Loop Limit Semantics

*Problem:* ASL loops have optional limits with specific semantics: evaluated once, decremented each iteration, raises `LimitExceeded` at zero.

*herdtools7 AST:*
```ocaml
| S_For of { ...; limit: expr option }
| S_While of expr * expr option * stmt
| S_Repeat of stmt * expr * expr option
```

*Recommendation:* Document loop limit semantics and `LimitExceeded` exception behavior.

=== Low Severity

==== Execution Graphs - Out of Scope

*Problem:* The ASL formal semantics use execution graphs with `aslpo`, `aslctrl`, `asldata` edges for memory model analysis.

*Current status:* The herdtools7 AST does not include execution graph types. This is a semantic interpretation layer above the AST.

*Recommendation:* Document that execution graph analysis is out of scope for the current dialect.

==== Real Number Representation

*Problem:* ASL reals are "mathematical rational numbers with no bounds on precision." The mapping to runtime representation is not documented.

*Recommendation:* Document that `!asl.real` maps to `!gmp.q` (arbitrary-precision rational) for lowering.

==== Valueless Exceptions

*Problem:* ASL exceptions can be valueless (marked with `-`). The dialect models this as empty field lists.

*Recommendation:* Document that valueless exceptions use empty `fields` array.

==== Collection Type Constraints

*Problem:* ASL collections are global-only with bitvector fields exclusively. These constraints are not enforced in the dialect.

*Current status:* The herdtools7 frontend enforces these constraints.

*Recommendation:* Document that collection constraints are frontend-verified.

==== Mixed Integer/Real Operations

*Problem:* ASL supports multiplication between integers and rationals. Type coercion semantics are undocumented.

*Recommendation:* Document that integer operands are promoted to rational for mixed-type multiplication.

== Summary

#table(
  columns: (auto, auto, auto, auto),
  inset: 6pt,
  align: (left, center, left, left),
  [*Issue*], [*Severity*], [*Layer*], [*Action*],
  [Division semantics], [High], [Dialect], [Document exact vs floor],
  [Short-circuit], [High], [Dialect], [Add trait or document],
  [Bitvector wraparound], [High], [Dialect], [Document type semantics],
  [Type satisfaction], [Medium], [Frontend], [Document frontend handles],
  [Symbolically evaluable], [Medium], [Frontend], [Document frontend handles],
  [Loop limits], [Medium], [Dialect], [Document semantics],
  [Execution graphs], [Low], [N/A], [Document out of scope],
  [Real representation], [Low], [Dialect], [Document GMP mapping],
  [Valueless exceptions], [Low], [Dialect], [Document empty fields],
  [Collection constraints], [Low], [Frontend], [Document frontend handles],
  [Mixed int/real], [Low], [Dialect], [Document coercion],
)

== Conclusion

=== JSON Backend

The JSON backend is complete and correctly serializes all AST constructs from the pinned herdtools7 version (`d7d6bdd24f8680c4abf2df3a3e54a9d98494321e`).

=== MLIR Dialect

The high severity issues affect semantic correctness:

+ *Division semantics* - `DIV` is exact division, `DIVRM` is floor division
+ *Short-circuit evaluation* - `BAND`, `BOR`, `IMPL` have lazy evaluation
+ *Bitvector wraparound* - Bitvector arithmetic wraps, integer does not

These require documentation or dialect changes for correct ARM specification modeling.

=== Frontend-Handled Features

Several concerns are addressed by the herdtools7 frontend before JSON serialization:
- Type satisfaction and subtyping
- Symbolic evaluability verification
- Collection type constraints

The MLIR dialect can rely on the frontend for these guarantees.
