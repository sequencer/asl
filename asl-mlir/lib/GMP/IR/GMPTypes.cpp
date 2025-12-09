//===- GMPTypes.cpp - GMP types ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GMP/GMPTypes.h"
#include "GMP/GMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::gmp;

//===----------------------------------------------------------------------===//
// GMP Types Registration
//===----------------------------------------------------------------------===//

void GMPDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "GMP/GMPTypes.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "GMP/GMPTypes.cpp.inc"
