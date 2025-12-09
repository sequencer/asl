//===- GMPAttributes.cpp - GMP attributes -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GMP/GMPAttributes.h"
#include "GMP/GMPDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::gmp;

//===----------------------------------------------------------------------===//
// GMP Attributes Registration
//===----------------------------------------------------------------------===//

void GMPDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "GMP/GMPAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ZAttr Implementation
//===----------------------------------------------------------------------===//

llvm::APInt ZAttr::getAPInt() const {
  llvm::StringRef str = getValue();
  bool isNegative = str.starts_with("-");
  if (isNegative)
    str = str.drop_front(1);

  // Calculate required bits: roughly 3.32 bits per decimal digit + 1 for sign
  unsigned numBits = str.size() * 4 + 1;
  if (numBits < 64)
    numBits = 64;

  llvm::APInt result(numBits, str, 10);
  if (isNegative)
    result.negate();
  return result;
}

bool ZAttr::isZero() const { return getValue() == "0"; }

bool ZAttr::isOne() const { return getValue() == "1"; }

bool ZAttr::isNegative() const {
  return llvm::StringRef(getValue()).starts_with("-");
}

//===----------------------------------------------------------------------===//
// QAttr Implementation
//===----------------------------------------------------------------------===//

llvm::APInt QAttr::getNumeratorAPInt() const {
  llvm::StringRef str = getNumerator();
  bool isNegative = str.starts_with("-");
  if (isNegative)
    str = str.drop_front(1);

  unsigned numBits = str.size() * 4 + 1;
  if (numBits < 64)
    numBits = 64;

  llvm::APInt result(numBits, str, 10);
  if (isNegative)
    result.negate();
  return result;
}

llvm::APInt QAttr::getDenominatorAPInt() const {
  llvm::StringRef str = getDenominator();
  unsigned numBits = str.size() * 4 + 1;
  if (numBits < 64)
    numBits = 64;
  return llvm::APInt(numBits, str, 10);
}

bool QAttr::isZero() const {
  return getNumerator() == "0" && getDenominator() != "0";
}

bool QAttr::isPosInf() const {
  return !llvm::StringRef(getNumerator()).starts_with("-") &&
         getNumerator() != "0" && getDenominator() == "0";
}

bool QAttr::isNegInf() const {
  return llvm::StringRef(getNumerator()).starts_with("-") &&
         getDenominator() == "0";
}

bool QAttr::isUndef() const {
  return getNumerator() == "0" && getDenominator() == "0";
}

bool QAttr::isReal() const { return getDenominator() != "0"; }

//===----------------------------------------------------------------------===//
// ZAttr Parse/Print
//===----------------------------------------------------------------------===//

/// Parse a signed integer as a string (handles negative numbers)
static FailureOr<std::string> parseSignedInteger(AsmParser &parser) {
  std::string value;

  // Check for optional negative sign
  bool isNegative = false;
  if (succeeded(parser.parseOptionalMinus())) {
    isNegative = true;
  }

  // Parse the integer value
  APInt intVal;
  if (parser.parseInteger(intVal)) {
    return failure();
  }

  // Convert to string
  llvm::SmallString<32> str;
  intVal.toStringSigned(str);
  value = std::string(str);

  if (isNegative && !value.empty() && value[0] != '-') {
    value = "-" + value;
  }

  return value;
}

Attribute ZAttr::parse(AsmParser &parser, Type odsType) {
  if (parser.parseLess())
    return {};

  auto valueOrErr = parseSignedInteger(parser);
  if (failed(valueOrErr)) {
    parser.emitError(parser.getCurrentLocation(), "expected integer value");
    return {};
  }

  if (parser.parseGreater())
    return {};

  return ZAttr::get(parser.getContext(), *valueOrErr);
}

void ZAttr::print(AsmPrinter &printer) const {
  printer << "<" << getValue() << ">";
}

//===----------------------------------------------------------------------===//
// QAttr Parse/Print
//===----------------------------------------------------------------------===//

Attribute QAttr::parse(AsmParser &parser, Type odsType) {
  if (parser.parseLess())
    return {};

  auto numOrErr = parseSignedInteger(parser);
  if (failed(numOrErr)) {
    parser.emitError(parser.getCurrentLocation(), "expected numerator");
    return {};
  }

  if (parser.parseComma())
    return {};

  auto denOrErr = parseSignedInteger(parser);
  if (failed(denOrErr)) {
    parser.emitError(parser.getCurrentLocation(), "expected denominator");
    return {};
  }

  if (parser.parseGreater())
    return {};

  return QAttr::get(parser.getContext(), *numOrErr, *denOrErr);
}

void QAttr::print(AsmPrinter &printer) const {
  printer << "<" << getNumerator() << ", " << getDenominator() << ">";
}

#define GET_ATTRDEF_CLASSES
#include "GMP/GMPAttributes.cpp.inc"
