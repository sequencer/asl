#ifndef ASL_PASSES
#define ASL_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ASL Dialect Passes
//===----------------------------------------------------------------------===//

def ASLToEmitC : Pass<"asl-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Lower ASL dialect to EmitC dialect";
  let description = [{
    This pass lowers ASL dialect operations to EmitC dialect operations,
    enabling C code generation from ASL specifications.

    The lowering process performs:
    1. Type conversion from ASL types to C-compatible types
    2. Operation lowering with semantic preservation
    3. Control flow conversion to structured C constructs
    4. Memory management for value semantics
    5. Exception handling using setjmp/longjmp
    6. Global state encapsulation for thread safety

    Type Lowering:
    - `!asl.int` -> `intmax_t` (with optional GMP for large integers)
    - `!asl.bits<N>` -> `uintN_t` or multi-word structs for N > 64
    - `!asl.real` -> GMP `mpq_t` for exact rational arithmetic
    - `!asl.bool` -> `i1` (C `bool`)
    - `!asl.string` -> `const char*` or `char*`
    - `!asl.enum` -> C `enum`
    - `!asl.tuple` -> C `struct`
    - `!asl.array` -> C array
    - `!asl.record` -> C `struct`
    - `!asl.exception` -> C `struct` with setjmp/longjmp

    Operation Lowering:
    - Arithmetic operations -> inline operators or GMP calls
    - Bitfield access -> masking and shifting operations
    - Control flow -> if/else, loops with EmitC operations
    - Function calls -> direct calls with state parameter
    - Exception handling -> setjmp/longjmp based control flow

    Global State Management:
    - All mutable globals wrapped in `asl_global_state_t` struct
    - State pointer passed as first parameter to non-pure functions
    - Exception context stored in global state for thread safety
    - Thread-local storage for exception handling state

    Example:
    ```mlir
    // Input ASL
    asl.func @compute(%arg0: !asl.int) -> !asl.int {
      %0 = asl.expr.binop.plus %arg0, %arg0 : !asl.int
      asl.stmt.return %0 : !asl.int
    }

    // Output EmitC
    emitc.func @compute(%state: !emitc.ptr<!emitc.opaque<"asl_global_state_t">>,
                        %arg0: !emitc.opaque<"intmax_t">) 
        -> !emitc.opaque<"intmax_t"> {
      %0 = emitc.add %arg0, %arg0 : !emitc.opaque<"intmax_t">
      emitc.return %0 : !emitc.opaque<"intmax_t">
    }
    ```
  }];

  let constructor = "mlir::asl::createASLToEmitCPass()";

  let options = [
    Option<"useGMPForIntegers", "use-gmp-integers", "bool",
           /*default=*/"false",
           "Use GMP for all integer operations (for correctness with large integers)">,
    Option<"generateDebugInfo", "debug-info", "bool",
           /*default=*/"false",
           "Generate debug information and comments in output">,
  ];

  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::cf::ControlFlowDialect",
  ];
}

#endif // ASL_PASSES
