//===- ASLAttributes.cpp - ASL dialect attributes ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASL/ASLAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::asl;

//===----------------------------------------------------------------------===//
// Attribute Storage and Uniquing
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "ASL/ASLAttributes.cpp.inc"

//===----------------------------------------------------------------------===//
// ASL Dialect Attribute Registration
//===----------------------------------------------------------------------===//

void ASLDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ASL/ASLAttributes.cpp.inc"
      >();
}