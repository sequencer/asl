//===- ASLTypes.cpp - ASL dialect types ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ASL/ASLTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::asl;

//===----------------------------------------------------------------------===//
// ASL dialect type registration
//===----------------------------------------------------------------------===//

void ASLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ASL/ASLTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Generated type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "ASL/ASLTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Type method implementations
//===----------------------------------------------------------------------===//

bool IntType::isConstrained() const {
  return getConstraint().getKind() !=
         ConstraintKindTypeAttr::get(getContext(),
                                     ConstraintKindType::unconstrained);
}

bool BitsType::isParameterized() const { return getWidth().getInt() == -1; }

unsigned BitsType::getBitWidth() const { return getWidth().getInt(); }

::llvm::ArrayRef<::mlir::StringAttr> EnumType::getStringLabels() const {
  auto labels = getLabels();
  return llvm::ArrayRef<StringAttr>(
      reinterpret_cast<const StringAttr *>(labels.begin()), labels.size());
}

unsigned asl::TupleType::getNumTypes() const { return getTypes().size(); }

::mlir::Type asl::TupleType::getType(unsigned index) const {
  auto types = getTypes();
  if (index >= types.size())
    return {};
  return llvm::cast<TypeAttr>(types[index]).getValue();
}
