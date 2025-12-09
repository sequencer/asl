//===- GMPOps.h - GMP operations --------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef GMP_GMPOPS_H
#define GMP_GMPOPS_H

#include "GMP/GMPAttributes.h"
#include "GMP/GMPDialect.h"
#include "GMP/GMPTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "GMP/GMPOps.h.inc"

#endif // GMP_GMPOPS_H
