//===- ASLDialect.cpp - ASL dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASL/ASLDialect.h"
#include "ASL/ASLAttributes.h"
#include "ASL/ASLOps.h"
#include "ASL/ASLTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

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
  registerAttributes();
  registerTypes();
}

void ASLDialect::getCanonicalizationPatterns(
    RewritePatternSet &patterns) const {
  // TODO: Add canonicalization patterns when needed
}

Operation *ASLDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  // TODO: Implement constant materialization when needed
  return nullptr;
}

// Removed parse/print stubs for attributes and types to avoid duplicate
// definitions. Implementations are provided in attribute/type source files.

#include "ASL/ASLDialect.cpp.inc"
#include "ASL/ASLEnums.cpp.inc"
