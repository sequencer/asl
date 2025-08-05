//===- ASLOp.h - ASL dialect operation --------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_ASLOP_H
#define ASL_ASLOP_H

#include "ASL/ASLAttributes.h"
#include "ASL/ASLDialect.h"
#include "ASL/ASLTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "ASL/ASLOps.h.inc"

#endif // ASL_ASLOP_H
