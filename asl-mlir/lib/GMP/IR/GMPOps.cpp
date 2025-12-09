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
#include <cmath>
#include <gmp.h>
#include <limits>

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

  // neg(neg(x)) -> x
  if (auto negOp = getOperand().getDefiningOp<ZNegOp>())
    return negOp.getOperand();

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

  // abs(abs(x)) -> abs(x)
  if (auto absOp = getOperand().getDefiningOp<ZAbsOp>())
    return getOperand();

  // abs(neg(x)) -> abs(x)
  if (auto negOp = getOperand().getDefiningOp<ZNegOp>()) {
    // Return a new abs op on the inner operand (this is not a direct fold,
    // but we can canonicalize by rebuilding)
    // Actually, we just return the inner operand to let abs be applied to it
    // We can't really fold this directly, but we could use a rewrite pattern.
    // For now, skip this optimization since it requires creating new ops.
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
// ZCompareOp (Three-way comparison)
//===----------------------------------------------------------------------===//

OpFoldResult ZCompareOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    int cmp = mpz_cmp(lhs.get(), rhs.get());
    // Normalize to -1, 0, 1
    int32_t result = (cmp > 0) ? 1 : (cmp < 0) ? -1 : 0;
    return IntegerAttr::get(IntegerType::get(getContext(), 32), result);
  }

  // x compare x -> 0
  if (getLhs() == getRhs())
    return IntegerAttr::get(IntegerType::get(getContext(), 32), 0);

  return {};
}

//===----------------------------------------------------------------------===//
// ZEqualOp
//===----------------------------------------------------------------------===//

OpFoldResult ZEqualOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    bool equal = mpz_cmp(lhs.get(), rhs.get()) == 0;
    return BoolAttr::get(getContext(), equal);
  }

  // x == x -> true
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), true);

  return {};
}

//===----------------------------------------------------------------------===//
// ZLtOp
//===----------------------------------------------------------------------===//

OpFoldResult ZLtOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    bool lt = mpz_cmp(lhs.get(), rhs.get()) < 0;
    return BoolAttr::get(getContext(), lt);
  }

  // x < x -> false
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), false);

  return {};
}

//===----------------------------------------------------------------------===//
// ZLeqOp
//===----------------------------------------------------------------------===//

OpFoldResult ZLeqOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    bool leq = mpz_cmp(lhs.get(), rhs.get()) <= 0;
    return BoolAttr::get(getContext(), leq);
  }

  // x <= x -> true
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), true);

  return {};
}

//===----------------------------------------------------------------------===//
// ZGtOp
//===----------------------------------------------------------------------===//

OpFoldResult ZGtOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    bool gt = mpz_cmp(lhs.get(), rhs.get()) > 0;
    return BoolAttr::get(getContext(), gt);
  }

  // x > x -> false
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), false);

  return {};
}

//===----------------------------------------------------------------------===//
// ZGeqOp
//===----------------------------------------------------------------------===//

OpFoldResult ZGeqOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    bool geq = mpz_cmp(lhs.get(), rhs.get()) >= 0;
    return BoolAttr::get(getContext(), geq);
  }

  // x >= x -> true
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), true);

  return {};
}

//===----------------------------------------------------------------------===//
// ZLogAndOp (Bitwise AND)
//===----------------------------------------------------------------------===//

OpFoldResult ZLogAndOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_and(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x & 0 -> 0
  if ((lhsAttr && lhsAttr.isZero()) || (rhsAttr && rhsAttr.isZero()))
    return ZAttr::get(getContext(), "0");

  // x & x -> x
  if (getLhs() == getRhs())
    return getLhs();

  return {};
}

//===----------------------------------------------------------------------===//
// ZLogOrOp (Bitwise OR)
//===----------------------------------------------------------------------===//

OpFoldResult ZLogOrOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_ior(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x | 0 -> x
  if (rhsAttr && rhsAttr.isZero())
    return getLhs();
  if (lhsAttr && lhsAttr.isZero())
    return getRhs();

  // x | x -> x
  if (getLhs() == getRhs())
    return getLhs();

  return {};
}

//===----------------------------------------------------------------------===//
// ZLogXorOp (Bitwise XOR)
//===----------------------------------------------------------------------===//

OpFoldResult ZLogXorOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_xor(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // x ^ 0 -> x
  if (rhsAttr && rhsAttr.isZero())
    return getLhs();
  if (lhsAttr && lhsAttr.isZero())
    return getRhs();

  // x ^ x -> 0
  if (getLhs() == getRhs())
    return ZAttr::get(getContext(), "0");

  return {};
}

//===----------------------------------------------------------------------===//
// ZLogNotOp (Bitwise complement)
//===----------------------------------------------------------------------===//

OpFoldResult ZLogNotOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    MPZValue val(attr.getValue());
    MPZValue result;
    mpz_com(result.get(), val.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // lognot(lognot(x)) -> x
  if (auto notOp = getOperand().getDefiningOp<ZLogNotOp>())
    return notOp.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// ZShiftLeftOp
//===----------------------------------------------------------------------===//

OpFoldResult ZShiftLeftOp::fold(FoldAdaptor adaptor) {
  auto opAttr = dyn_cast_or_null<ZAttr>(adaptor.getOperand());
  auto countAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getCount());

  if (opAttr && countAttr) {
    MPZValue val(opAttr.getValue());
    MPZValue result;
    uint64_t count = countAttr.getInt();
    mpz_mul_2exp(result.get(), val.get(), count);
    return ZAttr::get(getContext(), result.toString());
  }

  // x << 0 -> x
  if (countAttr && countAttr.getInt() == 0)
    return getOperand();

  // 0 << n -> 0
  if (opAttr && opAttr.isZero())
    return opAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZShiftRightOp (Floor division by 2^n)
//===----------------------------------------------------------------------===//

OpFoldResult ZShiftRightOp::fold(FoldAdaptor adaptor) {
  auto opAttr = dyn_cast_or_null<ZAttr>(adaptor.getOperand());
  auto countAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getCount());

  if (opAttr && countAttr) {
    MPZValue val(opAttr.getValue());
    MPZValue result;
    uint64_t count = countAttr.getInt();
    mpz_fdiv_q_2exp(result.get(), val.get(), count);
    return ZAttr::get(getContext(), result.toString());
  }

  // x >> 0 -> x
  if (countAttr && countAttr.getInt() == 0)
    return getOperand();

  // 0 >> n -> 0
  if (opAttr && opAttr.isZero())
    return opAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZShiftRightTruncOp (Truncated division by 2^n)
//===----------------------------------------------------------------------===//

OpFoldResult ZShiftRightTruncOp::fold(FoldAdaptor adaptor) {
  auto opAttr = dyn_cast_or_null<ZAttr>(adaptor.getOperand());
  auto countAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getCount());

  if (opAttr && countAttr) {
    MPZValue val(opAttr.getValue());
    MPZValue result;
    uint64_t count = countAttr.getInt();
    mpz_tdiv_q_2exp(result.get(), val.get(), count);
    return ZAttr::get(getContext(), result.toString());
  }

  // x >> 0 -> x
  if (countAttr && countAttr.getInt() == 0)
    return getOperand();

  // 0 >> n -> 0
  if (opAttr && opAttr.isZero())
    return opAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZTestBitOp
//===----------------------------------------------------------------------===//

OpFoldResult ZTestBitOp::fold(FoldAdaptor adaptor) {
  auto opAttr = dyn_cast_or_null<ZAttr>(adaptor.getOperand());
  auto indexAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getIndex());

  if (opAttr && indexAttr) {
    MPZValue val(opAttr.getValue());
    uint64_t index = indexAttr.getInt();
    bool bit = mpz_tstbit(val.get(), index) != 0;
    return BoolAttr::get(getContext(), bit);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ZPopCountOp
//===----------------------------------------------------------------------===//

OpFoldResult ZPopCountOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    MPZValue val(attr.getValue());
    mp_bitcnt_t count = mpz_popcount(val.get());
    // Note: for negative numbers, mpz_popcount returns ULONG_MAX
    return IntegerAttr::get(IntegerType::get(getContext(), 64), count);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ZNumBitsOp
//===----------------------------------------------------------------------===//

OpFoldResult ZNumBitsOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    MPZValue val(attr.getValue());
    // mpz_sizeinbase returns 1 for 0, but we want 0 for 0
    if (mpz_sgn(val.get()) == 0)
      return IntegerAttr::get(IntegerType::get(getContext(), 64), 0);
    size_t bits = mpz_sizeinbase(val.get(), 2);
    return IntegerAttr::get(IntegerType::get(getContext(), 64), bits);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ZTrailingZerosOp
//===----------------------------------------------------------------------===//

OpFoldResult ZTrailingZerosOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    MPZValue val(attr.getValue());
    // mpz_scan1 returns ULONG_MAX if no 1 bit found (i.e., for 0)
    mp_bitcnt_t zeros = mpz_scan1(val.get(), 0);
    return IntegerAttr::get(IntegerType::get(getContext(), 64), zeros);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ZExtractOp (Unsigned bit extraction)
//===----------------------------------------------------------------------===//

OpFoldResult ZExtractOp::fold(FoldAdaptor adaptor) {
  auto opAttr = dyn_cast_or_null<ZAttr>(adaptor.getOperand());
  auto loAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getLo());
  auto widthAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getWidth());

  if (opAttr && loAttr && widthAttr) {
    MPZValue val(opAttr.getValue());
    uint64_t lo = loAttr.getInt();
    uint64_t width = widthAttr.getInt();

    if (width == 0)
      return ZAttr::get(getContext(), "0");

    MPZValue result;
    // Shift right by lo bits, then mask with (2^width - 1)
    mpz_fdiv_q_2exp(result.get(), val.get(), lo);

    MPZValue mask;
    mpz_set_ui(mask.get(), 1);
    mpz_mul_2exp(mask.get(), mask.get(), width);
    mpz_sub_ui(mask.get(), mask.get(), 1);

    mpz_and(result.get(), result.get(), mask.get());
    return ZAttr::get(getContext(), result.toString());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ZSignedExtractOp (Signed bit extraction)
//===----------------------------------------------------------------------===//

OpFoldResult ZSignedExtractOp::fold(FoldAdaptor adaptor) {
  auto opAttr = dyn_cast_or_null<ZAttr>(adaptor.getOperand());
  auto loAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getLo());
  auto widthAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getWidth());

  if (opAttr && loAttr && widthAttr) {
    MPZValue val(opAttr.getValue());
    uint64_t lo = loAttr.getInt();
    uint64_t width = widthAttr.getInt();

    if (width == 0)
      return ZAttr::get(getContext(), "0");

    MPZValue result;
    // Shift right by lo bits
    mpz_fdiv_q_2exp(result.get(), val.get(), lo);

    // Mask with (2^width - 1)
    MPZValue mask;
    mpz_set_ui(mask.get(), 1);
    mpz_mul_2exp(mask.get(), mask.get(), width);
    mpz_sub_ui(mask.get(), mask.get(), 1);
    mpz_and(result.get(), result.get(), mask.get());

    // Check if sign bit is set (bit at position width-1)
    if (mpz_tstbit(result.get(), width - 1)) {
      // Sign extend: subtract 2^width
      MPZValue twoToWidth;
      mpz_set_ui(twoToWidth.get(), 1);
      mpz_mul_2exp(twoToWidth.get(), twoToWidth.get(), width);
      mpz_sub(result.get(), result.get(), twoToWidth.get());
    }

    return ZAttr::get(getContext(), result.toString());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// Z Number Theory Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ZGcdOp
//===----------------------------------------------------------------------===//

OpFoldResult ZGcdOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_gcd(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // gcd(x, 0) -> |x|
  if (rhsAttr && rhsAttr.isZero())
    return ZAttr::get(getContext(),
                      lhsAttr
                          ? (lhsAttr.isNegative() ? lhsAttr.getValue().substr(1)
                                                  : lhsAttr.getValue())
                          : "0");
  if (lhsAttr && lhsAttr.isZero())
    return ZAttr::get(getContext(),
                      rhsAttr
                          ? (rhsAttr.isNegative() ? rhsAttr.getValue().substr(1)
                                                  : rhsAttr.getValue())
                          : "0");

  // gcd(x, x) -> |x|
  if (getLhs() == getRhs())
    return getLhs();

  return {};
}

//===----------------------------------------------------------------------===//
// ZGcdExtOp
//===----------------------------------------------------------------------===//

LogicalResult ZGcdExtOp::fold(FoldAdaptor adaptor,
                              SmallVectorImpl<OpFoldResult> &results) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue gcd, s, t;
    mpz_gcdext(gcd.get(), s.get(), t.get(), lhs.get(), rhs.get());
    results.push_back(ZAttr::get(getContext(), gcd.toString()));
    results.push_back(ZAttr::get(getContext(), s.toString()));
    results.push_back(ZAttr::get(getContext(), t.toString()));
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// ZLcmOp
//===----------------------------------------------------------------------===//

OpFoldResult ZLcmOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<ZAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr) {
    MPZValue lhs(lhsAttr.getValue());
    MPZValue rhs(rhsAttr.getValue());
    MPZValue result;
    mpz_lcm(result.get(), lhs.get(), rhs.get());
    return ZAttr::get(getContext(), result.toString());
  }

  // lcm(x, 0) -> 0
  if ((lhsAttr && lhsAttr.isZero()) || (rhsAttr && rhsAttr.isZero()))
    return ZAttr::get(getContext(), "0");

  // lcm(x, 1) -> |x|
  if (rhsAttr && rhsAttr.isOne())
    return getLhs();
  if (lhsAttr && lhsAttr.isOne())
    return getRhs();

  return {};
}

//===----------------------------------------------------------------------===//
// Z Modular Arithmetic Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ZPowmOp
//===----------------------------------------------------------------------===//

OpFoldResult ZPowmOp::fold(FoldAdaptor adaptor) {
  auto baseAttr = dyn_cast_or_null<ZAttr>(adaptor.getBase());
  auto expAttr = dyn_cast_or_null<ZAttr>(adaptor.getExp());
  auto modAttr = dyn_cast_or_null<ZAttr>(adaptor.getModulus());

  if (baseAttr && expAttr && modAttr && !modAttr.isZero()) {
    MPZValue base(baseAttr.getValue());
    MPZValue exp(expAttr.getValue());
    MPZValue mod(modAttr.getValue());
    MPZValue result;
    mpz_powm(result.get(), base.get(), exp.get(), mod.get());
    return ZAttr::get(getContext(), result.toString());
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ZPowmSecOp
//===----------------------------------------------------------------------===//

OpFoldResult ZPowmSecOp::fold(FoldAdaptor adaptor) {
  auto baseAttr = dyn_cast_or_null<ZAttr>(adaptor.getBase());
  auto expAttr = dyn_cast_or_null<ZAttr>(adaptor.getExp());
  auto modAttr = dyn_cast_or_null<ZAttr>(adaptor.getModulus());

  if (baseAttr && expAttr && modAttr && !modAttr.isZero()) {
    MPZValue base(baseAttr.getValue());
    MPZValue exp(expAttr.getValue());
    MPZValue mod(modAttr.getValue());
    // mpz_powm_sec requires positive exp and odd modulus
    // Check modulus is odd
    if (mpz_sgn(exp.get()) > 0 && mpz_odd_p(mod.get())) {
      MPZValue result;
      mpz_powm_sec(result.get(), base.get(), exp.get(), mod.get());
      return ZAttr::get(getContext(), result.toString());
    }
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ZInvertOp
//===----------------------------------------------------------------------===//

LogicalResult ZInvertOp::fold(FoldAdaptor adaptor,
                              SmallVectorImpl<OpFoldResult> &results) {
  auto opAttr = dyn_cast_or_null<ZAttr>(adaptor.getOperand());
  auto modAttr = dyn_cast_or_null<ZAttr>(adaptor.getModulus());

  if (opAttr && modAttr && !modAttr.isZero()) {
    MPZValue op(opAttr.getValue());
    MPZValue mod(modAttr.getValue());
    MPZValue result;
    int exists = mpz_invert(result.get(), op.get(), mod.get());
    results.push_back(ZAttr::get(getContext(), result.toString()));
    results.push_back(BoolAttr::get(getContext(), exists != 0));
    return success();
  }

  return failure();
}

//===----------------------------------------------------------------------===//
// Z Primality Testing Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ZProbabPrimeOp
//===----------------------------------------------------------------------===//

OpFoldResult ZProbabPrimeOp::fold(FoldAdaptor adaptor) {
  auto opAttr = dyn_cast_or_null<ZAttr>(adaptor.getOperand());
  auto repsAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getReps());

  if (opAttr && repsAttr) {
    MPZValue val(opAttr.getValue());
    int reps = repsAttr.getInt();
    int result = mpz_probab_prime_p(val.get(), reps);
    return IntegerAttr::get(IntegerType::get(getContext(), 32), result);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// ZNextPrimeOp
//===----------------------------------------------------------------------===//

OpFoldResult ZNextPrimeOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    MPZValue val(attr.getValue());
    MPZValue result;
    mpz_nextprime(result.get(), val.get());
    return ZAttr::get(getContext(), result.toString());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Z Power and Root Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ZPowOp
//===----------------------------------------------------------------------===//

OpFoldResult ZPowOp::fold(FoldAdaptor adaptor) {
  auto baseAttr = dyn_cast_or_null<ZAttr>(adaptor.getBase());
  auto expAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getExp());

  if (baseAttr && expAttr) {
    MPZValue base(baseAttr.getValue());
    uint64_t exp = expAttr.getInt();
    MPZValue result;
    mpz_pow_ui(result.get(), base.get(), exp);
    return ZAttr::get(getContext(), result.toString());
  }

  // x^0 -> 1
  if (expAttr && expAttr.getInt() == 0)
    return ZAttr::get(getContext(), "1");

  // x^1 -> x
  if (expAttr && expAttr.getInt() == 1)
    return getBase();

  // 0^n -> 0 (for n > 0)
  if (baseAttr && baseAttr.isZero())
    return baseAttr;

  // 1^n -> 1
  if (baseAttr && baseAttr.isOne())
    return baseAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// ZSqrtOp
//===----------------------------------------------------------------------===//

OpFoldResult ZSqrtOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    MPZValue val(attr.getValue());
    // Only fold for non-negative values
    if (mpz_sgn(val.get()) >= 0) {
      MPZValue result;
      mpz_sqrt(result.get(), val.get());
      return ZAttr::get(getContext(), result.toString());
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ZSqrtRemOp
//===----------------------------------------------------------------------===//

LogicalResult ZSqrtRemOp::fold(FoldAdaptor adaptor,
                               SmallVectorImpl<OpFoldResult> &results) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    MPZValue val(attr.getValue());
    // Only fold for non-negative values
    if (mpz_sgn(val.get()) >= 0) {
      MPZValue root, rem;
      mpz_sqrtrem(root.get(), rem.get(), val.get());
      results.push_back(ZAttr::get(getContext(), root.toString()));
      results.push_back(ZAttr::get(getContext(), rem.toString()));
      return success();
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ZRootOp
//===----------------------------------------------------------------------===//

OpFoldResult ZRootOp::fold(FoldAdaptor adaptor) {
  auto opAttr = dyn_cast_or_null<ZAttr>(adaptor.getOperand());
  auto nAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getN());

  if (opAttr && nAttr) {
    MPZValue val(opAttr.getValue());
    uint64_t n = nAttr.getInt();
    // For even n, operand must be non-negative
    if (n > 0 && (n % 2 == 1 || mpz_sgn(val.get()) >= 0)) {
      MPZValue result;
      mpz_root(result.get(), val.get(), n);
      return ZAttr::get(getContext(), result.toString());
    }
  }

  // x root 1 -> x
  if (nAttr && nAttr.getInt() == 1)
    return getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// Z Factorial and Combinatorics Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ZFacOp
//===----------------------------------------------------------------------===//

OpFoldResult ZFacOp::fold(FoldAdaptor adaptor) {
  if (auto nAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getN())) {
    int64_t n = nAttr.getInt();
    if (n >= 0) {
      MPZValue result;
      mpz_fac_ui(result.get(), static_cast<uint64_t>(n));
      return ZAttr::get(getContext(), result.toString());
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// ZBinOp
//===----------------------------------------------------------------------===//

OpFoldResult ZBinOp::fold(FoldAdaptor adaptor) {
  auto nAttr = dyn_cast_or_null<ZAttr>(adaptor.getN());
  auto kAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getK());

  if (nAttr && kAttr) {
    MPZValue n(nAttr.getValue());
    uint64_t k = kAttr.getInt();
    MPZValue result;
    mpz_bin_ui(result.get(), n.get(), k);
    return ZAttr::get(getContext(), result.toString());
  }

  // C(n, 0) -> 1
  if (kAttr && kAttr.getInt() == 0)
    return ZAttr::get(getContext(), "1");

  return {};
}

//===----------------------------------------------------------------------===//
// ZFibOp
//===----------------------------------------------------------------------===//

OpFoldResult ZFibOp::fold(FoldAdaptor adaptor) {
  if (auto nAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getN())) {
    int64_t n = nAttr.getInt();
    if (n >= 0) {
      MPZValue result;
      mpz_fib_ui(result.get(), static_cast<uint64_t>(n));
      return ZAttr::get(getContext(), result.toString());
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Q Module Operations
//===----------------------------------------------------------------------===//

namespace {
/// RAII wrapper for mpq_t to ensure proper initialization and cleanup.
class MPQValue {
public:
  MPQValue() { mpq_init(value); }
  MPQValue(const std::string &num, const std::string &den) {
    mpq_init(value);
    mpz_set_str(mpq_numref(value), num.c_str(), 10);
    mpz_set_str(mpq_denref(value), den.c_str(), 10);
    // Canonicalize - but only if denominator is not zero
    if (mpz_sgn(mpq_denref(value)) != 0)
      mpq_canonicalize(value);
  }
  ~MPQValue() { mpq_clear(value); }

  // Non-copyable
  MPQValue(const MPQValue &) = delete;
  MPQValue &operator=(const MPQValue &) = delete;

  mpq_t &get() { return value; }
  const mpq_t &get() const { return value; }

  std::string numToString() const {
    char *str = mpz_get_str(nullptr, 10, mpq_numref(value));
    std::string result(str);
    free(str);
    return result;
  }

  std::string denToString() const {
    char *str = mpz_get_str(nullptr, 10, mpq_denref(value));
    std::string result(str);
    free(str);
    return result;
  }

  bool isZero() const {
    return mpz_sgn(mpq_numref(value)) == 0 && mpz_sgn(mpq_denref(value)) != 0;
  }

  bool isPosInf() const {
    return mpz_sgn(mpq_numref(value)) > 0 && mpz_sgn(mpq_denref(value)) == 0;
  }

  bool isNegInf() const {
    return mpz_sgn(mpq_numref(value)) < 0 && mpz_sgn(mpq_denref(value)) == 0;
  }

  bool isUndef() const {
    return mpz_sgn(mpq_numref(value)) == 0 && mpz_sgn(mpq_denref(value)) == 0;
  }

  bool isReal() const { return mpz_sgn(mpq_denref(value)) != 0; }

private:
  mpq_t value;
};
} // namespace

/// Helper to create a QAttr from an MPQValue
static QAttr createQAttr(MLIRContext *ctx, const MPQValue &val) {
  return QAttr::get(ctx, val.numToString(), val.denToString());
}

//===----------------------------------------------------------------------===//
// QConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult QConstantOp::fold(FoldAdaptor adaptor) {
  // Return the constant value for use by other operations' fold methods.
  return getValue();
}

//===----------------------------------------------------------------------===//
// QMakeOp
//===----------------------------------------------------------------------===//

OpFoldResult QMakeOp::fold(FoldAdaptor adaptor) {
  auto numAttr = dyn_cast_or_null<ZAttr>(adaptor.getNumerator());
  auto denAttr = dyn_cast_or_null<ZAttr>(adaptor.getDenominator());

  if (numAttr && denAttr) {
    MPQValue result(numAttr.getValue(), denAttr.getValue());
    return createQAttr(getContext(), result);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// QNumOp
//===----------------------------------------------------------------------===//

OpFoldResult QNumOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    return ZAttr::get(getContext(), attr.getNumerator());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QDenOp
//===----------------------------------------------------------------------===//

OpFoldResult QDenOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    return ZAttr::get(getContext(), attr.getDenominator());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QAddOp
//===----------------------------------------------------------------------===//

OpFoldResult QAddOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<QAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<QAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && lhsAttr.isReal() && rhsAttr.isReal()) {
    MPQValue lhs(lhsAttr.getNumerator(), lhsAttr.getDenominator());
    MPQValue rhs(rhsAttr.getNumerator(), rhsAttr.getDenominator());
    MPQValue result;
    mpq_add(result.get(), lhs.get(), rhs.get());
    return createQAttr(getContext(), result);
  }

  // q + 0 -> q
  if (rhsAttr && rhsAttr.isZero())
    return getLhs();
  if (lhsAttr && lhsAttr.isZero())
    return getRhs();

  return {};
}

//===----------------------------------------------------------------------===//
// QSubOp
//===----------------------------------------------------------------------===//

OpFoldResult QSubOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<QAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<QAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && lhsAttr.isReal() && rhsAttr.isReal()) {
    MPQValue lhs(lhsAttr.getNumerator(), lhsAttr.getDenominator());
    MPQValue rhs(rhsAttr.getNumerator(), rhsAttr.getDenominator());
    MPQValue result;
    mpq_sub(result.get(), lhs.get(), rhs.get());
    return createQAttr(getContext(), result);
  }

  // q - 0 -> q
  if (rhsAttr && rhsAttr.isZero())
    return getLhs();

  // q - q -> 0
  if (getLhs() == getRhs())
    return QAttr::get(getContext(), "0", "1");

  return {};
}

//===----------------------------------------------------------------------===//
// QMulOp
//===----------------------------------------------------------------------===//

OpFoldResult QMulOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<QAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<QAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && lhsAttr.isReal() && rhsAttr.isReal()) {
    MPQValue lhs(lhsAttr.getNumerator(), lhsAttr.getDenominator());
    MPQValue rhs(rhsAttr.getNumerator(), rhsAttr.getDenominator());
    MPQValue result;
    mpq_mul(result.get(), lhs.get(), rhs.get());
    return createQAttr(getContext(), result);
  }

  // q * 0 -> 0 (for real q)
  if (lhsAttr && lhsAttr.isReal() && rhsAttr && rhsAttr.isZero())
    return rhsAttr;
  if (rhsAttr && rhsAttr.isReal() && lhsAttr && lhsAttr.isZero())
    return lhsAttr;

  // q * 1 -> q
  if (rhsAttr && rhsAttr.getNumerator() == "1" &&
      rhsAttr.getDenominator() == "1")
    return getLhs();
  if (lhsAttr && lhsAttr.getNumerator() == "1" &&
      lhsAttr.getDenominator() == "1")
    return getRhs();

  return {};
}

//===----------------------------------------------------------------------===//
// QDivOp
//===----------------------------------------------------------------------===//

OpFoldResult QDivOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<QAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<QAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && lhsAttr.isReal() && rhsAttr.isReal() &&
      !rhsAttr.isZero()) {
    MPQValue lhs(lhsAttr.getNumerator(), lhsAttr.getDenominator());
    MPQValue rhs(rhsAttr.getNumerator(), rhsAttr.getDenominator());
    MPQValue result;
    mpq_div(result.get(), lhs.get(), rhs.get());
    return createQAttr(getContext(), result);
  }

  // q / 1 -> q
  if (rhsAttr && rhsAttr.getNumerator() == "1" &&
      rhsAttr.getDenominator() == "1")
    return getLhs();

  // 0 / q -> 0 (for non-zero real q)
  if (lhsAttr && lhsAttr.isZero() && rhsAttr && rhsAttr.isReal() &&
      !rhsAttr.isZero())
    return lhsAttr;

  return {};
}

//===----------------------------------------------------------------------===//
// QNegOp
//===----------------------------------------------------------------------===//

OpFoldResult QNegOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    if (attr.isReal()) {
      MPQValue val(attr.getNumerator(), attr.getDenominator());
      MPQValue result;
      mpq_neg(result.get(), val.get());
      return createQAttr(getContext(), result);
    }
    // Handle special values: neg(+inf) = -inf, neg(-inf) = +inf
    if (attr.isPosInf())
      return QAttr::get(getContext(), "-1", "0");
    if (attr.isNegInf())
      return QAttr::get(getContext(), "1", "0");
  }

  // neg(neg(q)) -> q
  if (auto negOp = getOperand().getDefiningOp<QNegOp>())
    return negOp.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// QAbsOp
//===----------------------------------------------------------------------===//

OpFoldResult QAbsOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    if (attr.isReal()) {
      MPQValue val(attr.getNumerator(), attr.getDenominator());
      MPQValue result;
      mpq_abs(result.get(), val.get());
      return createQAttr(getContext(), result);
    }
    // abs(+inf) = abs(-inf) = +inf
    if (attr.isPosInf() || attr.isNegInf())
      return QAttr::get(getContext(), "1", "0");
  }

  // abs(abs(q)) -> abs(q)
  if (auto absOp = getOperand().getDefiningOp<QAbsOp>())
    return getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// QInvOp
//===----------------------------------------------------------------------===//

OpFoldResult QInvOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    if (attr.isReal() && !attr.isZero()) {
      MPQValue val(attr.getNumerator(), attr.getDenominator());
      MPQValue result;
      mpq_inv(result.get(), val.get());
      return createQAttr(getContext(), result);
    }
    // inv(0) = +inf
    if (attr.isZero())
      return QAttr::get(getContext(), "1", "0");
    // inv(+inf) = inv(-inf) = 0
    if (attr.isPosInf() || attr.isNegInf())
      return QAttr::get(getContext(), "0", "1");
  }

  // inv(inv(q)) -> q
  if (auto invOp = getOperand().getDefiningOp<QInvOp>())
    return invOp.getOperand();

  return {};
}

//===----------------------------------------------------------------------===//
// QCompareOp
//===----------------------------------------------------------------------===//

OpFoldResult QCompareOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<QAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<QAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && lhsAttr.isReal() && rhsAttr.isReal()) {
    MPQValue lhs(lhsAttr.getNumerator(), lhsAttr.getDenominator());
    MPQValue rhs(rhsAttr.getNumerator(), rhsAttr.getDenominator());
    int cmp = mpq_cmp(lhs.get(), rhs.get());
    int32_t result = (cmp > 0) ? 1 : (cmp < 0) ? -1 : 0;
    return IntegerAttr::get(IntegerType::get(getContext(), 32), result);
  }

  // q compare q -> 0
  if (getLhs() == getRhs())
    return IntegerAttr::get(IntegerType::get(getContext(), 32), 0);

  return {};
}

//===----------------------------------------------------------------------===//
// QEqualOp
//===----------------------------------------------------------------------===//

OpFoldResult QEqualOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<QAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<QAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && lhsAttr.isReal() && rhsAttr.isReal()) {
    MPQValue lhs(lhsAttr.getNumerator(), lhsAttr.getDenominator());
    MPQValue rhs(rhsAttr.getNumerator(), rhsAttr.getDenominator());
    bool equal = mpq_equal(lhs.get(), rhs.get()) != 0;
    return BoolAttr::get(getContext(), equal);
  }

  // q == q -> true
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), true);

  return {};
}

//===----------------------------------------------------------------------===//
// QLtOp
//===----------------------------------------------------------------------===//

OpFoldResult QLtOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<QAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<QAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && lhsAttr.isReal() && rhsAttr.isReal()) {
    MPQValue lhs(lhsAttr.getNumerator(), lhsAttr.getDenominator());
    MPQValue rhs(rhsAttr.getNumerator(), rhsAttr.getDenominator());
    bool lt = mpq_cmp(lhs.get(), rhs.get()) < 0;
    return BoolAttr::get(getContext(), lt);
  }

  // q < q -> false
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), false);

  return {};
}

//===----------------------------------------------------------------------===//
// QLeqOp
//===----------------------------------------------------------------------===//

OpFoldResult QLeqOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<QAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<QAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && lhsAttr.isReal() && rhsAttr.isReal()) {
    MPQValue lhs(lhsAttr.getNumerator(), lhsAttr.getDenominator());
    MPQValue rhs(rhsAttr.getNumerator(), rhsAttr.getDenominator());
    bool leq = mpq_cmp(lhs.get(), rhs.get()) <= 0;
    return BoolAttr::get(getContext(), leq);
  }

  // q <= q -> true
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), true);

  return {};
}

//===----------------------------------------------------------------------===//
// QGtOp
//===----------------------------------------------------------------------===//

OpFoldResult QGtOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<QAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<QAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && lhsAttr.isReal() && rhsAttr.isReal()) {
    MPQValue lhs(lhsAttr.getNumerator(), lhsAttr.getDenominator());
    MPQValue rhs(rhsAttr.getNumerator(), rhsAttr.getDenominator());
    bool gt = mpq_cmp(lhs.get(), rhs.get()) > 0;
    return BoolAttr::get(getContext(), gt);
  }

  // q > q -> false
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), false);

  return {};
}

//===----------------------------------------------------------------------===//
// QGeqOp
//===----------------------------------------------------------------------===//

OpFoldResult QGeqOp::fold(FoldAdaptor adaptor) {
  auto lhsAttr = dyn_cast_or_null<QAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_or_null<QAttr>(adaptor.getRhs());

  if (lhsAttr && rhsAttr && lhsAttr.isReal() && rhsAttr.isReal()) {
    MPQValue lhs(lhsAttr.getNumerator(), lhsAttr.getDenominator());
    MPQValue rhs(rhsAttr.getNumerator(), rhsAttr.getDenominator());
    bool geq = mpq_cmp(lhs.get(), rhs.get()) >= 0;
    return BoolAttr::get(getContext(), geq);
  }

  // q >= q -> true
  if (getLhs() == getRhs())
    return BoolAttr::get(getContext(), true);

  return {};
}

//===----------------------------------------------------------------------===//
// QClassifyOp
//===----------------------------------------------------------------------===//

OpFoldResult QClassifyOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    int32_t result;
    if (attr.isReal())
      result = 0; // Real
    else if (attr.isPosInf())
      result = 1; // +Inf
    else if (attr.isNegInf())
      result = 2; // -Inf
    else
      result = 3; // Undef
    return IntegerAttr::get(IntegerType::get(getContext(), 32), result);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QIsRealOp
//===----------------------------------------------------------------------===//

OpFoldResult QIsRealOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    return BoolAttr::get(getContext(), attr.isReal());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QFloorOp
//===----------------------------------------------------------------------===//

OpFoldResult QFloorOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    if (attr.isReal()) {
      MPZValue num(attr.getNumerator());
      MPZValue den(attr.getDenominator());
      MPZValue result;
      mpz_fdiv_q(result.get(), num.get(), den.get());
      return ZAttr::get(getContext(), result.toString());
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QCeilOp
//===----------------------------------------------------------------------===//

OpFoldResult QCeilOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    if (attr.isReal()) {
      MPZValue num(attr.getNumerator());
      MPZValue den(attr.getDenominator());
      MPZValue result;
      mpz_cdiv_q(result.get(), num.get(), den.get());
      return ZAttr::get(getContext(), result.toString());
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QTruncOp
//===----------------------------------------------------------------------===//

OpFoldResult QTruncOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    if (attr.isReal()) {
      MPZValue num(attr.getNumerator());
      MPZValue den(attr.getDenominator());
      MPZValue result;
      mpz_tdiv_q(result.get(), num.get(), den.get());
      return ZAttr::get(getContext(), result.toString());
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QRoundOp
//===----------------------------------------------------------------------===//

OpFoldResult QRoundOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    if (attr.isReal()) {
      MPZValue num(attr.getNumerator());
      MPZValue den(attr.getDenominator());

      // Round away from zero for halfway cases
      // result = floor((num + sign(num) * den/2) / den)
      MPZValue halfDen, adjustedNum, result;
      mpz_fdiv_q_ui(halfDen.get(), den.get(), 2);

      if (mpz_sgn(num.get()) >= 0) {
        mpz_add(adjustedNum.get(), num.get(), halfDen.get());
      } else {
        mpz_sub(adjustedNum.get(), num.get(), halfDen.get());
      }
      mpz_tdiv_q(result.get(), adjustedNum.get(), den.get());
      return ZAttr::get(getContext(), result.toString());
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QToBigIntOp
//===----------------------------------------------------------------------===//

OpFoldResult QToBigIntOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    if (attr.isReal()) {
      MPZValue num(attr.getNumerator());
      MPZValue den(attr.getDenominator());
      MPZValue result;
      mpz_tdiv_q(result.get(), num.get(), den.get());
      return ZAttr::get(getContext(), result.toString());
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QToF64Op
//===----------------------------------------------------------------------===//

OpFoldResult QToF64Op::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<QAttr>(adaptor.getOperand())) {
    if (attr.isReal()) {
      MPQValue val(attr.getNumerator(), attr.getDenominator());
      double result = mpq_get_d(val.get());
      return FloatAttr::get(Float64Type::get(getContext()), result);
    }
    // +inf -> infinity
    if (attr.isPosInf())
      return FloatAttr::get(Float64Type::get(getContext()),
                            std::numeric_limits<double>::infinity());
    // -inf -> -infinity
    if (attr.isNegInf())
      return FloatAttr::get(Float64Type::get(getContext()),
                            -std::numeric_limits<double>::infinity());
    // undef -> NaN
    if (attr.isUndef())
      return FloatAttr::get(Float64Type::get(getContext()),
                            std::numeric_limits<double>::quiet_NaN());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// QFromZOp
//===----------------------------------------------------------------------===//

OpFoldResult QFromZOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<ZAttr>(adaptor.getOperand())) {
    return QAttr::get(getContext(), attr.getValue(), "1");
  }
  return {};
}

#define GET_OP_CLASSES
#include "GMP/GMPOps.cpp.inc"
