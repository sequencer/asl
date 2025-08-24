//===- ASLAttributes.h - ASL dialect attributes ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_ATTRIBUTES_H
#define ASL_ATTRIBUTES_H

#include "ASL/ASLDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"

#define GET_ATTRDEF_CLASSES
#include "ASL/ASLAttributes.h.inc"

#endif // ASL_ATTRIBUTES_H