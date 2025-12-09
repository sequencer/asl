//===- GMPPasses.td - GMP Dialect Passes -----------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines passes for the GMP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GMP_PASSES_TD
#define GMP_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// GMP Dialect Passes
//===----------------------------------------------------------------------===//

def ConvertGMPToEmitC : Pass<"convert-gmp-to-emitc", "::mlir::ModuleOp"> {
  let summary = "Convert GMP dialect to EmitC dialect";
  let description = [{
    This pass lowers GMP dialect operations to EmitC dialect operations,
    enabling C code generation using the GMP library.

    The lowering process converts:
    - `!gmp.z` type to `mpz_t` (GMP arbitrary precision integer)
    - `!gmp.q` type to `mpq_t` (GMP arbitrary precision rational)
    - Z module operations to `mpz_*` function calls
    - Q module operations to `mpq_*` function calls

    Type Lowering:
    - `!gmp.z` -> `mpz_t` (GMP arbitrary precision integer)
    - `!gmp.q` -> `mpq_t` (GMP arbitrary precision rational)

    Z Module Operation Lowering:
    - `gmp.z.constant` -> `mpz_init_set_str` or inline constant
    - `gmp.z.add` -> `mpz_add`
    - `gmp.z.sub` -> `mpz_sub`
    - `gmp.z.mul` -> `mpz_mul`
    - `gmp.z.div` -> `mpz_tdiv_q` (truncated division)
    - `gmp.z.rem` -> `mpz_tdiv_r` (truncated remainder)
    - `gmp.z.neg` -> `mpz_neg`
    - `gmp.z.abs` -> `mpz_abs`
    - `gmp.z.compare` -> `mpz_cmp`
    - `gmp.z.logand` -> `mpz_and`
    - `gmp.z.logor` -> `mpz_ior`
    - `gmp.z.logxor` -> `mpz_xor`
    - `gmp.z.lognot` -> `mpz_com`
    - `gmp.z.shift_left` -> `mpz_mul_2exp`
    - `gmp.z.shift_right` -> `mpz_fdiv_q_2exp`
    - `gmp.z.pow` -> `mpz_pow_ui`
    - ... and more

    Q Module Operation Lowering:
    - `gmp.q.constant` -> `mpq_init` + `mpq_set_*`
    - `gmp.q.add` -> `mpq_add`
    - `gmp.q.sub` -> `mpq_sub`
    - `gmp.q.mul` -> `mpq_mul`
    - `gmp.q.div` -> `mpq_div`
    - `gmp.q.neg` -> `mpq_neg`
    - `gmp.q.abs` -> `mpq_abs`
    - `gmp.q.inv` -> `mpq_inv`
    - `gmp.q.compare` -> `mpq_cmp`
    - `gmp.q.floor` -> custom implementation using `mpz_fdiv_q`
    - `gmp.q.ceil` -> custom implementation using `mpz_cdiv_q`
    - ... and more

    Example:
    ```mlir
    // Input GMP dialect
    %a = gmp.z.constant #gmp.z<42>
    %b = gmp.z.constant #gmp.z<10>
    %c = gmp.z.add %a, %b

    // Output EmitC dialect (conceptual)
    emitc.call_opaque "mpz_init_set_str"(%a_var, "42", 10)
    emitc.call_opaque "mpz_init_set_str"(%b_var, "10", 10)
    emitc.call_opaque "mpz_add"(%c_var, %a_var, %b_var)
    ```
  }];

  let constructor = "mlir::gmp::createConvertGMPToEmitCPass()";

  let dependentDialects = [
    "::mlir::emitc::EmitCDialect",
    "::mlir::func::FuncDialect",
    "::mlir::arith::ArithDialect",
  ];
}

#endif // GMP_PASSES_TD
