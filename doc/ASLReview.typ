#let document_title = "ASL Dialect Design Review"
#set document(title: document_title, author: "Jiuyang Liu")
#set heading(numbering: "1.1")
#set text(font: "New Computer Modern", size: 11pt)

= ASL Dialect Design Review

This document analyzes the ASL dialect implementation against the official herdtools7 ASL specification, identifying design pitfalls and recommendations.

== Review Methodology

The review compares:
- herdtools7 ASL specification (Types.tex, Expressions.tex, Statements.tex, Slicing.tex, RelationsOnTypes.tex)
- herdtools7 Operations.ml implementation
- Current ASLRational.typ documentation
- ASL dialect TableGen definitions

== High Severity Issues

=== Integer Division Semantics Mismatch

*Problem:* The ASL spec defines `DIV` as exact division (returns `None` if not divisible), but the dialect documentation describes `asl.expr.binop.div` as integer division without clarifying which semantics apply.

*herdtools7 spec:*
- `DIV` - exact division only (fails if not exact)
- `DIVRM` - floor division toward negative infinity

*Current dialect:*
- `asl.expr.binop.div` - "Integer division" (ambiguous)
- `asl.expr.binop.divrm` - "Integer division with rounding towards negative infinity"

*Recommendation:* Document that `DIV` in ASL requires exact division. Add validation or explicit failure handling for non-exact division cases.

=== Short-Circuit Semantics Not Captured

*Problem:* ASL specifies short-circuit evaluation for `&&`, `||`, and `==>`. The current `Pure` trait on all expression operations may not properly capture these semantics.

*herdtools7 spec:*
```
Special short-circuit semantics for &&, ||, and ==>
```

*Current dialect:* `ASL_BinopBandOp`, `ASL_BinopBorOp`, `ASL_BinopImplOp` are all marked `Pure` without special handling.

*Recommendation:* Either:
- Add regions for short-circuit evaluation (like `scf.if`)
- Document that short-circuit semantics are handled at the lowering level
- Add a `ShortCircuit` trait or attribute

=== Bitvector Operations vs Integer Operations

*Problem:* ASL specifies bitvector ADD/SUB as "unsigned arithmetic with wraparound" but integer ADD/SUB have no bounds. The dialect uses `any` types for many operations.

*herdtools7 spec:*
```
Bit vectors: ADD, SUB (unsigned arithmetic with wraparound)
Integer: ADD, SUB, MUL, DIV (exact), DIVRM, MOD, POW, SHL, SHR
```

*Current dialect:*
```tablegen
def ASL_BinopPlusOp : ASL_ExprOp<"expr.binop.plus"> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
```

*Recommendation:* Either:
- Add type-specific operations (`asl.expr.binop.bits.add`, `asl.expr.binop.int.add`)
- Add verifiers to check type compatibility
- Document type-dependent semantics clearly

== Medium Severity Issues

=== Missing BIC (Bit Clear) Operation

*Problem:* The ASL spec includes `BIC` (bitwise AND with complement), but this operation is not present in the dialect.

*herdtools7 spec:*
```
BIC - AND with complement (a AND NOT b)
```

*Current dialect:* Missing `asl.expr.binop.bic`

*Recommendation:* Add `BIC` operation or document that it should be lowered to `AND(a, NOT(b))` during parsing.

=== Type Satisfaction vs Type Equality

*Problem:* ASL has a complex type satisfaction relation distinct from type equality. Named types have identity-based comparison. The dialect does not model this.

*herdtools7 spec:*
```
Named types maintaining strict identity-based incompatibility
except through explicit supertype relationships
Type satisfaction: A type satisfies another if it can be used
where the second type is expected
Subtype satisfaction: Stricter than simple subtyping
```

*Current dialect:* `ASL_NamedType` has an optional `resolved_type` but no subtype hierarchy or satisfaction checking.

*Recommendation:* Add:
- Subtype declaration operations
- Type satisfaction verification
- Or document these are checked by herdtools7 frontend only

=== Symbolically Evaluable Expressions Not Distinguished

*Problem:* ASL requires certain expressions (array lengths, bitvector widths, constraint expressions) to be "symbolically evaluable" - only involving immutable values. The dialect does not distinguish these.

*herdtools7 spec:*
```
An expression is symbolically evaluable if its evaluation
only involves immutable values
Expressions appearing in integer constraints must be both
symbolically evaluable and constrained integer types
```

*Current dialect:* No distinction between symbolically evaluable and runtime expressions.

*Recommendation:* Add a trait or marker for operations that must be symbolically evaluable, enabling compile-time verification.

=== Loop Limit Semantics Incomplete

*Problem:* ASL loops have mandatory limits with specific decrement semantics. The current dialect has optional limits without specifying behavior.

*herdtools7 spec:*
```
Limit checking: Evaluates limit once, decrements each iteration;
raises LimitExceeded at zero
```

*Current dialect:*
```tablegen
let arguments = (ins ASL_IntType:$start, ASL_IntType:$end, optional<...>:$limit);
```

*Recommendation:* Either:
- Make limit mandatory (per ASL spec)
- Document default limit behavior
- Add `LimitExceeded` exception handling

== Low Severity Issues

=== Execution Graph / Side Effects Not Modeled

*Problem:* ASL's formal semantics use execution graphs with `aslpo`, `aslctrl`, `asldata` edges for memory model analysis. The current dialect has no representation for these.

*herdtools7 spec:*
```
Graph composition uses parallel (||) and ordered operators
to track dependencies
Edge types: aslpo (program order), aslctrl (control flow),
asldata (data dependencies)
```

*Current dialect:* No execution graph representation.

*Recommendation:* If the goal is formal verification or memory model analysis, add:
- An execution graph type or attribute
- Operations for graph composition
- Alternatively, document that this is out of scope

=== Real Number Representation Unclear

*Problem:* ASL specifies reals as "mathematical rational numbers with no bounds on precision or magnitude." The dialect uses `!asl.real` but does not specify representation.

*herdtools7 spec:*
```
Real Type: Mathematical rational numbers with no bounds
on precision or magnitude
```

*Current dialect:* `!asl.real` with no parameters, literals stored as string (Q.to_string).

*GMP dialect:* Has `!gmp.q` for rationals.

*Recommendation:* Document that `!asl.real` is implemented as arbitrary-precision rational (Q) not floating-point, and how it maps to `!gmp.q`.

=== Exception Type Structure

*Problem:* ASL exceptions can be either records with fields or valueless (marked with `-`). The current dialect only models the record case.

*herdtools7 spec:*
```
Exception Types: Similar to records; carry values in fields
or marked with - for valueless exceptions
```

*Current dialect:*
```tablegen
def ASL_ExceptionType : ASL_Type<"Exception", "exception"> {
  let parameters = (ins ArrayAttr:$fields);
}
```

*Recommendation:* Add support for valueless exceptions (empty fields or special marker).

=== Collection Type Constraints

*Problem:* ASL collections are global-only with bitvector fields exclusively. These constraints are not enforced.

*herdtools7 spec:*
```
Collection Types: Global-only structured types with
bitvector fields exclusively
```

*Current dialect:* `ASL_CollectionType` has generic `ArrayAttr:$fields` without bitvector constraint.

*Recommendation:* Add verifier to ensure collection fields are bitvector types.

=== Missing E_GetCollectionFields Implementation Details

*Problem:* The documentation mentions this operation but the L-expression counterpart `LE_SetCollectionFields` takes no inputs, which seems inconsistent.

*Current dialect:*
```
[LE_SetCollectionFields], [asl.lexpr.set_collection_fields], [], [!asl.lexpr],
```

*Recommendation:* Review whether collection field operations should take inputs or rely entirely on attributes.

=== Mixed Integer/Real Multiplication Not Documented

*Problem:* ASL supports multiplication between integers and rationals. The dialect does not document this cross-type operation.

*herdtools7 spec:*
```
Mixed int/real: Multiplication between integers and rationals
```

*Current dialect:* `asl.expr.binop.mul` uses `AnyType` but semantics are undocumented.

*Recommendation:* Document mixed-type operation semantics and expected type coercion.

== Summary

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  align: (left, center, left),
  [*Issue*], [*Severity*], [*Action*],
  [Division semantics], [High], [Document exact vs floor],
  [Short-circuit], [High], [Add trait/region],
  [Bitvector wraparound], [High], [Add type-specific ops],
  [Missing BIC], [Medium], [Add op or document lowering],
  [Type satisfaction], [Medium], [Add verification],
  [Symbolically evaluable], [Medium], [Add trait],
  [Loop limits], [Medium], [Make mandatory],
  [Execution graphs], [Low], [Document scope],
  [Real representation], [Low], [Document mapping],
  [Valueless exceptions], [Low], [Add support],
  [Collection constraints], [Low], [Add verifier],
  [Collection fields], [Low], [Review consistency],
  [Mixed int/real], [Low], [Document semantics],
)

== Conclusion

The most critical issues are:

+ *Division semantics* - `DIV` in ASL is exact division, not truncated division
+ *Short-circuit evaluation* - Boolean operators have lazy evaluation semantics
+ *Bitvector wraparound* - Bitvector arithmetic wraps, integer arithmetic does not

These affect correctness when modeling ARM architecture specifications. The medium severity issues around type satisfaction and symbolically evaluable expressions affect compile-time verification capabilities.

The low severity issues are mostly documentation gaps that should be addressed for completeness but do not affect functional correctness.
