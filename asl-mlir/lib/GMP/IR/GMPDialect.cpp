//===- GMPDialect.cpp - GMP dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GMP/GMPDialect.h"
#include "GMP/GMPAttributes.h"
#include "GMP/GMPOps.h"
#include "GMP/GMPTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::gmp;

//===----------------------------------------------------------------------===//
// GMP dialect.
//===----------------------------------------------------------------------===//

void GMPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GMP/GMPOps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
}

Operation *GMPDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (auto zAttr = dyn_cast<ZAttr>(value)) {
    if (isa<ZType>(type))
      return builder.create<ZConstantOp>(loc, type, zAttr);
  }
  if (auto qAttr = dyn_cast<QAttr>(value)) {
    if (isa<QType>(type))
      return builder.create<QConstantOp>(loc, type, qAttr);
  }
  return nullptr;
}

#include "GMP/GMPDialect.cpp.inc"
