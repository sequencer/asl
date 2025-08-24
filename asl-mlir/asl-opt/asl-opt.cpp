//===- asl-opt.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "ASL/ASLDialect.h"
#include "JSONImporter.h"

#include "llvm/Support/MemoryBuffer.h"

static llvm::cl::opt<std::string>
    jsonInput("json-input", llvm::cl::desc("Path to ASL JSON file to import"),
              llvm::cl::value_desc("filename"));

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::asl::ASLDialect>();
  mlir::registerAllDialects(registry);

  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "ASL optimizer driver\n");

  mlir::MLIRContext context(registry);

  if (!jsonInput.empty()) {
    auto moduleOrErr = mlir::asl::importJSONFile(context, jsonInput);
    if (!moduleOrErr) {
      llvm::errs() << "Failed to import JSON: "
                   << llvm::toString(moduleOrErr.takeError()) << "\n";
      return 1;
    }
    // Print the imported module and exit (skip normal mlir-opt flow)
    moduleOrErr->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ASL optimizer driver\n", registry));
}
