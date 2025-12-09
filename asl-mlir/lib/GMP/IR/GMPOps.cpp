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
#include "llvm/ADT/SmallString.h"
#include <gmp.h>

using namespace mlir;
using namespace mlir::gmp;

//===----------------------------------------------------------------------===//
// Helper class for RAII mpz_t management
//===----------------------------------------------------------------------===//

namespace {
/// RAII wrapper for mpz_t to ensure proper initialization and cleanup.
class MPZValue {
public:
  MPZValue() { mpz_init(value); }
  explicit MPZValue(const std::string &str) {
    mpz_init_set_str(value, str.c_str(), 10);
  }
  ~MPZValue() { mpz_clear(value); }

  // Non-copyable
  MPZValue(const MPZValue &) = delete;
  MPZValue &operator=(const MPZValue &) = delete;

  mpz_t &get() { return value; }
  const mpz_t &get() const { return value; }

  std::string toString() const {
    char *str = mpz_get_str(nullptr, 10, value);
    std::string result(str);
    free(str);
    return result;
  }

private:
  mpz_t value;
};
} // namespace

//===----------------------------------------------------------------------===//
// ZConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ZConstantOp::fold(FoldAdaptor adaptor) {
  // Return the constant value for use by other operations' fold methods.
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
// ZAddOp
//===----------------------------------------------------------------------===//

OpFoldResult ZAddOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_add(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x + 0 -> x
  if (rhsAttr && rhsAttr.isZero())
    return getLhs();
  if (lhsAttr && lhsAttr.isZero())
    return getRhs();

  return {};
}

//===----------------------------------------------------------------------===//
// ZSubOp
//===----------------------------------------------------------------------===//

OpFoldResult ZSubOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_sub(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x - 0 -> x
  if (rhsAttr && rhsAttr.isZero())
    return getLhs();

  // x - x -> 0
  if (getLhs() == getRhs())
    return ZAttr::get(getContext(), "0");

  return {};
}

//===----------------------------------------------------------------------===//
// ZMulOp
//===----------------------------------------------------------------------===//

OpFoldResult ZMulOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_mul(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x * 0 -> 0
  if ((lhsAttr && lhsAttr.isZero()) || (rhsAttr && rhsAttr.isZero()))
    return ZAttr::get(getContext(), "0");

  // x * 1 -> x
  if (rhsAttr && rhsAttr.isOne())
    return getLhs();
  if (lhsAttr && lhsAttr.isOne())
    return getRhs();

  return {};
}

//===----------------------------------------------------------------------===//
// ZNegOp
//===----------------------------------------------------------------------===//

OpFoldResult ZNegOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    MPZValue val(attr.getValue());
    MPZValue result;
    mpz_neg(result.get(), val.get());
    return ZAttr::get(getContext(), result.toString());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ZAbsOp
//===----------------------------------------------------------------------===//

OpFoldResult ZAbsOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    MPZValue val(attr.getValue());
    MPZValue result;
    mpz_abs(result.get(), val.get());
    return ZAttr::get(getContext(), result.toString());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ZSuccOp
//===----------------------------------------------------------------------===//

OpFoldResult ZSuccOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    MPZValue val(attr.getValue());
    MPZValue result;
    mpz_add_ui(result.get(), val.get(), 1);
    return ZAttr::get(getContext(), result.toString());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ZPredOp
//===----------------------------------------------------------------------===//

OpFoldResult ZPredOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    MPZValue val(attr.getValue());
    MPZValue result;
    mpz_sub_ui(result.get(), val.get(), 1);
    return ZAttr::get(getContext(), result.toString());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult QConstantOp::fold(FoldAdaptor adaptor) {
  // Return the constant value for use by other operations' fold methods.
  return getValue();
}

#define GET_OP_CLASSES
#include "GMP/GMPOps.cpp.inc"
