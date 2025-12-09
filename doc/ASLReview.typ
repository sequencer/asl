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

==== Valueless Exceptions (RESOLVED)

*Problem:* ASL exceptions can be valueless (marked with `-`). The dialect models this as empty field lists.

*Resolution:* Documented in ASLRational.typ section "Exception Type":
- Valueless exceptions use empty `fields` array: `!asl.exception<[]>`

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
  [*Issue*], [*Severity*], [*Layer*], [*Status*],
  [Division semantics], [High], [Dialect], [RESOLVED],
  [Short-circuit], [High], [Dialect], [RESOLVED],
  [Bitvector wraparound], [High], [Dialect], [RESOLVED],
  [Type satisfaction], [Medium], [Frontend], [Documented],
  [Symbolically evaluable], [Medium], [Frontend], [Documented],
  [Loop limits], [Medium], [Dialect], [RESOLVED],
  [Execution graphs], [Low], [N/A], [Out of scope],
  [Real representation], [Low], [Dialect], [RESOLVED],
  [Valueless exceptions], [Low], [Dialect], [RESOLVED],
  [Collection constraints], [Low], [Frontend], [Documented],
  [Mixed int/real], [Low], [Dialect], [Open],
)

== Conclusion

=== JSON Backend

The JSON backend is complete and correctly serializes all AST constructs from the pinned herdtools7 version (`d7d6bdd24f8680c4abf2df3a3e54a9d98494321e`).

=== MLIR Dialect

==== Resolved Issues

+ *Division semantics* - Documented in ASLRational.typ and TableGen
+ *Short-circuit evaluation* - Documented lowering responsibility in ASLRational.typ
+ *Bitvector wraparound* - Split into type-specific operations (int/bits/real)

All high severity issues have been resolved.

=== Frontend-Handled Features

Several concerns are addressed by the herdtools7 frontend before JSON serialization:
- Type satisfaction and subtyping
- Symbolic evaluability verification
- Collection type constraints

The MLIR dialect can rely on the frontend for these guarantees.
