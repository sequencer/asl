//===- ASLPasses.h - ASL Dialect Passes ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for ASL dialect transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef ASL_ASLPASSES_H
#define ASL_ASLPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace asl {

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

/// Creates a pass to lower ASL dialect to EmitC dialect.
std::unique_ptr<Pass> createASLToEmitCPass();

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "ASL/ASLPasses.h.inc"

} // namespace asl
} // namespace mlir

#endif // ASL_ASLPASSES_H
