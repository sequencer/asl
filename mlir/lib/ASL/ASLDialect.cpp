//===- ASLDialect.cpp - ASL dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASL/ASLDialect.h"
#include "ASL/ASLOps.h"

using namespace mlir;
using namespace mlir::asl;

//===----------------------------------------------------------------------===//
// ASL dialect.
//===----------------------------------------------------------------------===//

void ASLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ASL/ASLOps.cpp.inc"
      >();
}
