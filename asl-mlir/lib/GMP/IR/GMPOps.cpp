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
// ZDivOp (Truncated division)
//===----------------------------------------------------------------------===//

OpFoldResult ZDivOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && !rhsAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_tdiv_q(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x / 1 -> x
  if (rhsAttr && rhsAttr.isOne())
    return getLhs();

  // 0 / x -> 0 (when x != 0)
  if (lhsAttr && lhsAttr.isZero())
    return lhsAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZRemOp (Truncated remainder)
//===----------------------------------------------------------------------===//

OpFoldResult ZRemOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && !rhsAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_tdiv_r(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x % 1 -> 0
  if (rhsAttr && rhsAttr.isOne())
    return ZAttr::get(getContext(), "0");

  // 0 % x -> 0 (when x != 0)
  if (lhsAttr && lhsAttr.isZero())
    return lhsAttr;

  // x % x -> 0
  if (getLhs() == getRhs())
    return ZAttr::get(getContext(), "0");

  return {};
}

//===----------------------------------------------------------------------===//
// ZDivRemOp (Truncated division with quotient and remainder)
//===----------------------------------------------------------------------===//

LogicalResult ZDivRemOp::fold(FoldAdaptor adaptor,
                              SmallVectorImpl<OpFoldResult> &results) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && !rhsAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue quotient, remainder;
    mpz_tdiv_qr(quotient.get(), remainder.get(), lhs.get(), rhs.get());
    results.push_back(ZAttr::get(getContext(), quotient.toString()));
    results.push_back(ZAttr::get(getContext(), remainder.toString()));
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ZFDivOp (Floor division)
//===----------------------------------------------------------------------===//

OpFoldResult ZFDivOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && !rhsAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_fdiv_q(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x / 1 -> x
  if (rhsAttr && rhsAttr.isOne())
    return getLhs();

  // 0 / x -> 0 (when x != 0)
  if (lhsAttr && lhsAttr.isZero())
    return lhsAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZFRemOp (Floor remainder)
//===----------------------------------------------------------------------===//

OpFoldResult ZFRemOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && !rhsAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_fdiv_r(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x % 1 -> 0
  if (rhsAttr && rhsAttr.isOne())
    return ZAttr::get(getContext(), "0");

  // 0 % x -> 0 (when x != 0)
  if (lhsAttr && lhsAttr.isZero())
    return lhsAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZCDivOp (Ceiling division)
//===----------------------------------------------------------------------===//

OpFoldResult ZCDivOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && !rhsAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_cdiv_q(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x / 1 -> x
  if (rhsAttr && rhsAttr.isOne())
    return getLhs();

  // 0 / x -> 0 (when x != 0)
  if (lhsAttr && lhsAttr.isZero())
    return lhsAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZCRemOp (Ceiling remainder)
//===----------------------------------------------------------------------===//

OpFoldResult ZCRemOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && !rhsAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_cdiv_r(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x % 1 -> 0
  if (rhsAttr && rhsAttr.isOne())
    return ZAttr::get(getContext(), "0");

  // 0 % x -> 0 (when x != 0)
  if (lhsAttr && lhsAttr.isZero())
    return lhsAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZEDivOp (Euclidean division quotient)
//===----------------------------------------------------------------------===//

OpFoldResult ZEDivOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && !rhsAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    // Euclidean division: remainder is always non-negative
    // For positive divisor, this is floor division
    // For negative divisor, this is ceiling division
    if (mpz_sgn(rhs.get()) >= 0) {
      mpz_fdiv_q(result.get(), lhs.get(), rhs.get());
    } else {
      mpz_cdiv_q(result.get(), lhs.get(), rhs.get());
    }
    return ZAttr::get(getContext(), result.toString());
  }

  // x / 1 -> x
  if (rhsAttr && rhsAttr.isOne())
    return getLhs();

  // 0 / x -> 0 (when x != 0)
  if (lhsAttr && lhsAttr.isZero())
    return lhsAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZERemOp (Euclidean remainder - always non-negative)
//===----------------------------------------------------------------------===//

OpFoldResult ZERemOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && !rhsAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    // Euclidean remainder: always non-negative (0 <= r < |d|)
    // For positive divisor, this is floor remainder
    // For negative divisor, this is ceiling remainder (but we want it positive)
    if (mpz_sgn(rhs.get()) >= 0) {
      mpz_fdiv_r(result.get(), lhs.get(), rhs.get());
    } else {
      mpz_cdiv_r(result.get(), lhs.get(), rhs.get());
    }
    return ZAttr::get(getContext(), result.toString());
  }

  // x % 1 -> 0
  if (rhsAttr && rhsAttr.isOne())
    return ZAttr::get(getContext(), "0");

  // 0 % x -> 0 (when x != 0)
  if (lhsAttr && lhsAttr.isZero())
    return lhsAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZDivExactOp (Exact division)
//===----------------------------------------------------------------------===//

OpFoldResult ZDivExactOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && !rhsAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_divexact(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x / 1 -> x
  if (rhsAttr && rhsAttr.isOne())
    return getLhs();

  // 0 / x -> 0 (when x != 0)
  if (lhsAttr && lhsAttr.isZero())
    return lhsAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZDivisibleOp (Divisibility test)
//===----------------------------------------------------------------------===//

OpFoldResult ZDivisibleOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && !rhsAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    bool divisible = mpz_divisible_p(lhs.get(), rhs.get()) != 0;
    return BoolAttr::get(getContext(), divisible);
  }

  // 0 is divisible by any non-zero number
  if (lhsAttr && lhsAttr.isZero())
    return BoolAttr::get(getContext(), true);

  // x is divisible by 1
  if (rhsAttr && rhsAttr.isOne())
    return BoolAttr::get(getContext(), true);

  // x is divisible by x
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), true);

  return {};
}

//===----------------------------------------------------------------------===//
// ZCongruentOp (Congruence test)
//===----------------------------------------------------------------------===//

OpFoldResult ZCongruentOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());
  auto modAttr = dyn_cast_or_null<ZAttr>(adaptor.getModulus());

  if (lhsAttr && rhsAttr && modAttr && !modAttr.isZero()) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue mod(modAttr.getValue());
    bool congruent = mpz_congruent_p(lhs.get(), rhs.get(), mod.get()) != 0;
    return BoolAttr::get(getContext(), congruent);
  }

  // x ≡ x (mod m) for any m
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), true);

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
