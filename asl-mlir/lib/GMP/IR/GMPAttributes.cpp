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

  // Calculate required bits: log2(10) ≈ 3.32 bits per decimal digit.
  // We need one extra bit for sign to ensure the high bit is 0 for positive
  // numbers, allowing correct sign extension later.
  unsigned numBits = str.size() * 4 + 2; // +2 for safety margin
  if (numBits < 65)
    numBits = 65; // Minimum 65 bits to handle 64-bit values plus sign

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

/// Parse a signed integer as a string (handles negative numbers,
/// arbitrary precision, and binary/hex literals like 0b1010 and 0xFF).
static FailureOr<std::string> parseSignedInteger(AsmParser &parser) {
  std::string value;

  // Check for optional negative sign
  bool isNegative = false;
  if (succeeded(parser.parseOptionalMinus())) {
    isNegative = true;
  }

  // Try to parse binary literal (0b...) which MLIR lexer doesn't handle natively.
  // The lexer will tokenize "0b1010" as integer "0" followed by identifier "b1010".
  APInt intVal;

  // First, try normal integer parsing (handles decimal and hex 0x...)
  OptionalParseResult intResult = parser.parseOptionalInteger(intVal);
  if (intResult.has_value()) {
    if (failed(*intResult)) {
      return failure();
    }
    // Check if this is the start of a binary literal: we parsed "0" and next is "b..."
    if (intVal == 0) {
      // Try to parse identifier starting with 'b' for binary literal
      llvm::StringRef binStr;
      if (succeeded(parser.parseOptionalKeyword(&binStr)) &&
          binStr.size() > 0 && (binStr[0] == 'b' || binStr[0] == 'B')) {
        // It's a binary literal like 0b1010
        llvm::StringRef digits = binStr.drop_front(1);
        if (digits.empty()) {
          parser.emitError(parser.getCurrentLocation(),
                           "expected binary digits after '0b'");
          return failure();
        }
        // Validate all characters are 0 or 1
        for (char c : digits) {
          if (c != '0' && c != '1') {
            parser.emitError(parser.getCurrentLocation(),
                             "invalid binary digit");
            return failure();
          }
        }
        // Parse as binary
        unsigned numBits = digits.size() + 1;
        if (numBits < 65)
          numBits = 65;
        intVal = APInt(numBits, digits, 2);
      }
    }
  } else {
    // parseOptionalInteger didn't find an integer, this is an error
    parser.emitError(parser.getCurrentLocation(), "expected integer value");
    return failure();
  }

  // Convert to decimal string - use unsigned string since we handle sign
  // separately. This normalizes binary (0b...) and hex (0x...) to decimal.
  llvm::SmallString<128> str;
  intVal.toStringUnsigned(str, 10);
  value = std::string(str);

  if (isNegative) {
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
