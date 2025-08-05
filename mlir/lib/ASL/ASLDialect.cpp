//===- ASLDialect.cpp - ASL dialect -----------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASL/ASLDialect.h"
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

void ASLDialect::initialize() {}

void ASLDialect::getCanonicalizationPatterns(
    RewritePatternSet &patterns) const {
  // TODO: Add canonicalization patterns when needed
}

Operation *ASLDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  // TODO: Implement constant materialization when needed
  return nullptr;
}

Attribute ASLDialect::parseAttribute(DialectAsmParser &parser,
                                     Type type) const {
  // TODO: Implement attribute parsing when custom attributes are added
  return Attribute();
}

void ASLDialect::printAttribute(Attribute attr,
                                DialectAsmPrinter &printer) const {
  // TODO: Implement attribute printing when custom attributes are added
}

Type ASLDialect::parseType(DialectAsmParser &parser) const {
  // TODO: Implement type parsing when custom types are added
  return Type();
}

void ASLDialect::printType(Type type, DialectAsmPrinter &printer) const {
  // TODO: Implement type printing when custom types are added
}

#include "ASL/ASLDialect.cpp.inc"
#include "ASL/ASLEnums.cpp.inc"
