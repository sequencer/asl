//===- GMPAttributes.h - GMP attributes -------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GMP_GMPATTRIBUTES_H
#define GMP_GMPATTRIBUTES_H

#include "mlir/IR/Attributes.h"
#include "llvm/ADT/APInt.h"

#define GET_ATTRDEF_CLASSES
#include "GMP/GMPAttributes.h.inc"

#endif // GMP_GMPATTRIBUTES_H
