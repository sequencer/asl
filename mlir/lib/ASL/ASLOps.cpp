//===- ASLOps.cpp - ASL dialect ops -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASL/ASLOps.h"
#include "ASL/ASLDialect.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "ASL/ASLOps.cpp.inc"
