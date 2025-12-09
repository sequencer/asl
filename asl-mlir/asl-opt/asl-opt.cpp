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
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "ASL/ASLDialect.h"
#include "ASL/ASLPasses.h"
#include "GMP/GMPDialect.h"
#include "JSONImporter.h"

#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/Support/MemoryBuffer.h"

static llvm::cl::opt<std::string>
    jsonInput("json-input", llvm::cl::desc("Path to ASL JSON file to import"),
              llvm::cl::value_desc("filename"));

static llvm::cl::opt<bool> runCanonicalization(
    "json-canonicalize",
    llvm::cl::desc("Run canonicalization after JSON import"),
    llvm::cl::init(false));

static llvm::cl::opt<bool>
    runASLToEmitC("run-asl-to-emitc",
                  llvm::cl::desc("Run the ASL to EmitC lowering pass"),
                  llvm::cl::init(false));

static llvm::cl::opt<bool>
    emitC("emitc", llvm::cl::desc("Translate EmitC dialect to C code"),
          llvm::cl::init(false));

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::asl::registerASLPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::asl::ASLDialect>();
  registry.insert<mlir::gmp::GMPDialect>();
  mlir::registerAllDialects(registry);

  // Check if --json-input is present before full parsing
  bool hasJsonInput = false;
  for (int i = 1; i < argc; ++i) {
    if (llvm::StringRef(argv[i]).starts_with("--json-input"))
      hasJsonInput = true;
  }

  // If no JSON input, use standard mlir-opt main
  if (!hasJsonInput) {
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "ASL optimizer driver\n", registry));
  }

  // JSON input mode - use custom handling
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv, "ASL optimizer driver\n");

  mlir::MLIRContext context(registry);

  auto moduleOrErr = mlir::asl::importJSONFile(context, jsonInput);
  if (!moduleOrErr) {
    llvm::errs() << "Failed to import JSON: "
                 << llvm::toString(moduleOrErr.takeError()) << "\n";
    return 1;
  }

  mlir::OwningOpRef<mlir::ModuleOp> module = std::move(*moduleOrErr);

  // Optionally run canonicalization after import
  if (runCanonicalization) {
    mlir::PassManager pm(&context);
    pm.addPass(mlir::createCanonicalizerPass());

    if (mlir::failed(pm.run(module.get()))) {
      llvm::errs() << "Failed to run canonicalization pass\n";
      return 1;
    }
  }

  // Optionally run the ASL to EmitC lowering pass
  if (runASLToEmitC) {
    mlir::PassManager pm(&context);
    pm.addPass(mlir::asl::createASLToEmitCPass());

    if (mlir::failed(pm.run(module.get()))) {
      llvm::errs() << "Failed to run ASL to EmitC pass\n";
      return 1;
    }
  }

  // Optionally translate EmitC to C code
  if (emitC) {
    if (mlir::failed(mlir::emitc::translateToCpp(module.get(), llvm::outs()))) {
      llvm::errs() << "Failed to translate to C++\n";
      return 1;
    }
    return 0;
  }

  // Print the module and exit
  module->print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
