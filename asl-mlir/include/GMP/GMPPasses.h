//===- GMPPasses.h - GMP Dialect Passes ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for GMP dialect transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef GMP_GMPPASSES_H
#define GMP_GMPPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace gmp {

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

/// Creates a pass to lower GMP dialect to EmitC dialect.
std::unique_ptr<Pass> createConvertGMPToEmitCPass();

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "GMP/GMPPasses.h.inc"

} // namespace gmp
} // namespace mlir

#endif // GMP_GMPPASSES_H
