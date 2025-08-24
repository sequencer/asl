//===- ASLOps.cpp - ASL dialect types ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "ASL/ASLOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::asl;

#define GET_OP_CLASSES
#include "ASL/ASLOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Operation verify method implementations
//===----------------------------------------------------------------------===//

LogicalResult FuncDeclOp::verify() {
  // TODO: Add verification logic for function declarations
  return success();
}

LogicalResult PragmaDeclOp::verify() {
  // TODO: Add verification logic for pragma declarations
  return success();
}

LogicalResult TypeDeclOp::verify() {
  // TODO: Add verification logic for type declarations
  return success();
}

LogicalResult LExprSetFieldsOp::verify() {
  // TODO: Add verification logic for LExpr set fields operation
  return success();
}

LogicalResult LExprSetCollectionFieldsOp::verify() {
  // TODO: Add verification logic for LExpr set collection fields operation
  return success();
}

LogicalResult GlobalStorageDeclOp::verify() {
  // TODO: Add verification logic for global storage declarations
  return success();
}
