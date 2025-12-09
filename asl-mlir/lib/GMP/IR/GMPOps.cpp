//===- GMPOps.cpp - GMP operations ------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GMP/GMPOps.h"
#include "GMP/GMPAttributes.h"
#include "GMP/GMPDialect.h"
#include "GMP/GMPTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"

using namespace mlir;
using namespace mlir::gmp;

//===----------------------------------------------------------------------===//
// ZConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ZConstantOp::fold(FoldAdaptor adaptor) {
  // Constants fold to themselves
  return getValue();
}

//===----------------------------------------------------------------------===//
// ZFromIntOp
//===----------------------------------------------------------------------===//

OpFoldResult ZFromIntOp::fold(FoldAdaptor adaptor) {
  // If the input is a constant integer, fold to a ZAttr
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getInput())) {
    llvm::APInt value = intAttr.getValue();
    llvm::SmallString<32> str;
    value.toStringSigned(str);
    return ZAttr::get(getContext(), std::string(str));
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult QConstantOp::fold(FoldAdaptor adaptor) {
  // Constants fold to themselves
  return getValue();
}

#define GET_OP_CLASSES
#include "GMP/GMPOps.cpp.inc"
