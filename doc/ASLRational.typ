#set document(title: "Rationale for the ASL Dialect")

= Rationale for the ASL Dialect

The primary motivation for creating the ASL (Arm Specification Language) dialect is the need for an executable model of the RISC-V architecture.
Such a model is crucial for rigorously analyzing and implementing behaviors, including undefined behaviors (UB), which are often underspecified in traditional documentation.

The source of truth for the ASL used in this project is derived from `herdtools7`, ensuring that our dialect is based on a well-established and validated foundation.

== Frontend

The frontend is responsible for converting ASL source code into the ASL MLIR dialect.
Given that ASL is under active development, we leverage the `herdtools7/asllib` library to handle the complexities of the frontend.

- *Parsing and Typing:* The `herdtools7/asllib` library performs all parsing, type checking, and other frontend analyses, producing a well-formed, typed AST.
- *MLIR Integration:* The interaction between our OCaml-based tooling and the MLIR C++ framework is facilitated by JSON serialization.

The design of the ASL dialect's IR is guided by the principle of simplicity and directness.

- *Minimal Validation:* Since all type checking and semantic validation are handled by the `herdtools7/asllib` frontend, the ASL dialect itself requires minimal validation logic.
This assumes that the incoming IR is already correct, simplifying the implementation of operation verifiers.
- *Type Materialization:* The dialect includes mechanisms for type materialization, the herdtools7 typing doesn't support type parameters, the type AST includes none literal expressions, which need to be fold in the MLIR.

== Backend
The ASL dialect is designed to be a central IR and being able to lower to multiple backends:

- *C Code Generation:*: The dialect can be lowered to the `EmitC` dialect, enabling the generation of portable C code for architecture specification.
- *High-Performance Execution:*: For performance-critical use cases, the ASL dialect can be lowered to LLVM IR.
This allows for Just-In-Time (JIT) or Ahead-Of-Time (AOT) compilation, resulting in a high-performance executable model of the RISC-V specification.
- *Formal Verification:* The dialect should be able to lower to MLIR SMT Dialect for formal verification.

