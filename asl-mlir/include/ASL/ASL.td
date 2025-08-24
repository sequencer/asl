//===- ASL.td - ASL dialect definition ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_TD
#define ASL_TD
// Dialect
include "ASLDialect.td"
// Attributes
include "ASLAttributes.td"
// Type
include "ASLTypes.td"
// Operations
include "ASLExpressions.td"
include "ASLLExpression.td"
include "ASLStatement.td"
include "ASLDeclarations.td"

#endif // ASL_TD