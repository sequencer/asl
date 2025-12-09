//===- JSONImporter.h - Import ASL JSON IR ----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASL_JSON_IMPORTER_H
#define ASL_JSON_IMPORTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Error.h"
#include <string>

namespace mlir {
class OpBuilder;
}

namespace mlir::asl {

/// Import an ASL JSON file (conforming to the schema in doc/Json.typ) into a
/// newly created MLIR module. On failure returns an llvm::Error with a message.
llvm::Expected<mlir::ModuleOp> importJSONFile(MLIRContext &ctx,
                                              llvm::StringRef filePath);

} // namespace mlir::asl

#endif // ASL_JSON_IMPORTER_H
