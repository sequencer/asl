//===- ASLToEmitC.cpp - Lower ASL to EmitC dialect -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ASLToEmitC pass, which lowers ASL dialect operations
// to EmitC dialect operations for C code generation.
//
// Design: Global State Management
// ================================
// For each MLIR module, we generate three main components:
// 1. Context Structure (struct foo_context): Contains all global state
// 2. Initialization Function (void foo_init(foo_context*)): Initializes all
//    global state
// 3. Cleanup Function (void foo_free(foo_context*)): Frees allocated resources
//
// For each global variable 'bar', we generate:
// - A field in foo_context
// - An inline initializer function: void foo_init_bar(foo_context* ctx)
// - The foo_init function calls all per-variable initializers
//
//===----------------------------------------------------------------------===//

#include "ASL/ASLDialect.h"
#include "ASL/ASLOps.h"
#include "ASL/ASLPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {
namespace asl {

#define GEN_PASS_DECL_ASLTOEMITC
#define GEN_PASS_DEF_ASLTOEMITC
#include "ASL/ASLPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class ASLToEmitCTypeConverter : public TypeConverter {
public:
  ASLToEmitCTypeConverter(MLIRContext *context) : context(context) {
    // Pass through already legal types (EmitC types, integer types, etc.)
    addConversion([](Type type) { return type; });

    // Convert ASL bits type to appropriate C integer type
    addConversion([this](asl::BitsType type) -> std::optional<Type> {
      return convertBitsType(type);
    });

    // Convert ASL integer type to GMP mpz_t
    addConversion([this](asl::IntType type) -> std::optional<Type> {
      return convertIntType(type);
    });

    // Convert ASL real type to GMP mpq_t
    addConversion([this](asl::RealType type) -> std::optional<Type> {
      return convertRealType(type);
    });

    // Convert MLIR i1 type (representing ASL bool) to C bool
    // Note: ASL boolean type is represented by the standard i1 type in MLIR
    addConversion([this](IntegerType type) -> std::optional<Type> {
      if (type.getWidth() == 1) {
        return convertBoolType(type);
      }
      return std::nullopt;
    });

    // TODO: Add more ASL type conversions
    // - !asl.string -> emitc.opaque<"const char*">
  }

private:
  MLIRContext *context;

  // Convert ASL integer type to GMP mpz_t
  Type convertIntType(asl::IntType type) {
    // All ASL integers are converted to GMP's mpz_t for arbitrary precision
    // This ensures correctness since ASL integers are unbounded
    return emitc::OpaqueType::get(context, "mpz_t");
  }

  // Convert ASL real type to GMP mpq_t
  Type convertRealType(asl::RealType type) {
    // All ASL reals are converted to GMP's mpq_t for exact rational arithmetic
    // This ensures correctness since ASL reals represent exact rational numbers
    // (p/q where p and q are integers), not floating-point approximations
    return emitc::OpaqueType::get(context, "mpq_t");
  }

  // Convert i1 type (representing ASL bool) to C bool
  Type convertBoolType(IntegerType type) {
    // ASL boolean types are represented by MLIR's i1 type and converted to
    // C's standard bool type from <stdbool.h>
    // This provides a natural and efficient representation for boolean logic
    // Unlike integers or rationals which require GMP for correctness, booleans
    // have a finite domain and map directly to C's native boolean type without
    // loss of semantic information
    return emitc::OpaqueType::get(context, "bool");
  }

  // Convert ASL bits type to C integer type based on width
  Type convertBitsType(asl::BitsType type) {
    auto widthAttr = llvm::dyn_cast_or_null<IntegerAttr>(type.getWidth());
    if (!widthAttr) {
      mlir::emitError(UnknownLoc::get(context))
          << "bitvector width must be an integer attribute";
      return Type();
    }

    int64_t width = widthAttr.getInt();

    // Error on invalid width values
    if (width <= 0) {
      mlir::emitError(UnknownLoc::get(context))
          << "invalid bitvector width: " << width
          << " (width must be positive)";
      return Type();
    }

    // Select appropriate C type based on bit width
    if (width <= 8)
      return emitc::OpaqueType::get(context, "uint8_t");
    else if (width <= 16)
      return emitc::OpaqueType::get(context, "uint16_t");
    else if (width <= 32)
      return emitc::OpaqueType::get(context, "uint32_t");
    else if (width <= 64)
      return emitc::OpaqueType::get(context, "uint64_t");
    else {
      // For large bitvectors (>64 bits), generate struct with uint64_t array
      // Calculate number of 64-bit words needed
      int64_t numWords = (width + 63) / 64;
      std::string structType =
          "struct { uint64_t words[" + std::to_string(numWords) + "]; }";
      return emitc::OpaqueType::get(context, structType);
    }
  }
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Generate a sanitized identifier for C
static std::string sanitizeIdentifier(StringRef name) {
  std::string result = name.str();
  // Replace invalid C identifier characters with underscores
  for (char &c : result) {
    if (!llvm::isAlnum(c) && c != '_')
      c = '_';
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

// Convert asl.global.const_init operation to EmitC struct field and initializer
struct ConstantInitGlobalStorageDeclOpLowering
    : public OpConversionPattern<asl::ConstantInitGlobalStorageDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::ConstantInitGlobalStorageDeclOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For now, we'll collect global variable information
    // The actual struct generation happens in the pass
    // This pattern just removes the global decl op

    // TODO: Store global variable metadata for struct generation
    // For now, just erase the op - the pass will handle code generation
    rewriter.eraseOp(op);
    return success();
  }
};

// Convert asl.expr.literal.bitvector to emitc.constant
struct LiteralBitvectorOpLowering
    : public OpConversionPattern<asl::LiteralBitvectorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::LiteralBitvectorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto bitsType = llvm::dyn_cast<asl::BitsType>(op.getType());
    if (!bitsType)
      return failure();

    // Get the literal value
    StringRef literal = op.getValue();

    // Parse the bitvector literal (format: '0101...')
    // Remove quotes and convert to integer
    if (literal.size() < 2 || literal.front() != '\'' || literal.back() != '\'')
      return failure();

    literal = literal.drop_front().drop_back();

    // Convert binary string to integer value
    uint64_t value = 0;
    for (char c : literal) {
      value = value * 2 + (c == '1' ? 1 : 0);
    }

    // Convert the type
    Type convertedType = getTypeConverter()->convertType(bitsType);
    if (!convertedType)
      return failure();

    // Create EmitC constant with the appropriate C suffix
    std::string valueStr;
    if (auto opaqueType = llvm::dyn_cast<emitc::OpaqueType>(convertedType)) {
      StringRef typeName = opaqueType.getValue();
      if (typeName == "uint8_t")
        valueStr = std::to_string(value) + "u";
      else if (typeName == "uint16_t")
        valueStr = std::to_string(value) + "u";
      else if (typeName == "uint32_t")
        valueStr = std::to_string(value) + "u";
      else if (typeName == "uint64_t")
        valueStr = std::to_string(value) + "ULL";
      else
        valueStr = std::to_string(value);
    } else {
      valueStr = std::to_string(value);
    }

    auto constantOp = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), convertedType,
        emitc::OpaqueAttr::get(rewriter.getContext(), valueStr));

    rewriter.replaceOp(op, constantOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ASLToEmitCPass : public impl::ASLToEmitCBase<ASLToEmitCPass> {
  using Base = impl::ASLToEmitCBase<ASLToEmitCPass>;

  ASLToEmitCPass() = default;
  ASLToEmitCPass(const ASLToEmitCPass &) = default;

  // Explicitly provide constructor for options to work around tablegen bug
  ASLToEmitCPass(const ASLToEmitCOptions &options) : Base() {
    useGMPForIntegers = options.useGMPForIntegers;
    generateDebugInfo = options.generateDebugInfo;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitc::EmitCDialect, func::FuncDialect, arith::ArithDialect,
                    scf::SCFDialect, cf::ControlFlowDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Set up type converter
    ASLToEmitCTypeConverter typeConverter(context);

    // Set up conversion target
    ConversionTarget target(*context);
    target.addLegalDialect<emitc::EmitCDialect, func::FuncDialect,
                           arith::ArithDialect, scf::SCFDialect,
                           cf::ControlFlowDialect>();
    target.addIllegalDialect<asl::ASLDialect>();

    // Set up rewrite patterns
    RewritePatternSet patterns(context);

    // Add conversion patterns
    patterns.add<ConstantInitGlobalStorageDeclOpLowering,
                 LiteralBitvectorOpLowering>(typeConverter, context);

    // Collect global variables before conversion for context struct generation
    SmallVector<asl::ConstantInitGlobalStorageDeclOp> globalVars;
    module.walk([&](asl::ConstantInitGlobalStorageDeclOp op) {
      globalVars.push_back(op);
    });

    // Generate context structure, init, and free functions
    if (!globalVars.empty()) {
      generateGlobalContext(module, globalVars, typeConverter);
    }

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

private:
  // Generate the context structure and init/free functions for global variables
  void generateGlobalContext(
      ModuleOp module,
      ArrayRef<asl::ConstantInitGlobalStorageDeclOp> globalVars,
      ASLToEmitCTypeConverter &typeConverter) {
    OpBuilder builder(module.getContext());
    Location loc = module.getLoc();

    // Get module name for context struct name (default to "asl")
    std::string moduleName = "asl";
    if (auto nameAttr = module->getAttrOfType<StringAttr>("sym_name")) {
      moduleName = sanitizeIdentifier(nameAttr.getValue());
    }

    std::string contextTypeName = moduleName + "_context";

    // Set insertion point at the start of module
    builder.setInsertionPointToStart(module.getBody());

    // 0. Generate necessary includes
    generateIncludes(builder, loc);

    // 1. Generate typedef struct definition using emitc.verbatim
    generateContextStruct(builder, loc, contextTypeName, globalVars,
                          typeConverter);

    // 2. Generate per-variable inline init functions
    for (auto globalVar : globalVars) {
      generatePerVariableInit(builder, loc, moduleName, contextTypeName,
                              globalVar, typeConverter);
    }

    // 3. Generate main init function
    generateMainInitFunction(builder, loc, moduleName, contextTypeName,
                             globalVars);

    // 4. Generate free function
    generateFreeFunction(builder, loc, moduleName, contextTypeName, globalVars,
                         typeConverter);
  }

  // Generate necessary C includes
  void generateIncludes(OpBuilder &builder, Location loc) {
    // Add gmp.h for GMP arbitrary-precision integers (mpz_t) and rationals
    // (mpq_t)
    builder.create<emitc::VerbatimOp>(loc, "#include <gmp.h>");
    // Add stdint.h for uint8_t, uint16_t, uint32_t, uint64_t types (bitvectors)
    builder.create<emitc::VerbatimOp>(loc, "#include <stdint.h>");
    // Add stdbool.h for bool type
    builder.create<emitc::VerbatimOp>(loc, "#include <stdbool.h>");
  }

  // Generate the context struct typedef
  void generateContextStruct(
      OpBuilder &builder, Location loc, StringRef contextTypeName,
      ArrayRef<asl::ConstantInitGlobalStorageDeclOp> globalVars,
      ASLToEmitCTypeConverter &typeConverter) {
    std::string structDef = "typedef struct " + contextTypeName.str() + " {\n";

    // Add fields for each global variable
    for (auto globalVar : globalVars) {
      Type varType = globalVar.getType();
      Type convertedType = typeConverter.convertType(varType);

      std::string cType;
      if (auto opaqueType = llvm::dyn_cast<emitc::OpaqueType>(convertedType)) {
        cType = opaqueType.getValue().str();
      } else {
        cType = "uint64_t"; // fallback
      }

      std::string varName = sanitizeIdentifier(globalVar.getName());
      structDef += "  " + cType + " " + varName + ";\n";
    }

    structDef += "} " + contextTypeName.str() + ";";

    // Create verbatim op for struct definition
    builder.create<emitc::VerbatimOp>(loc, structDef);
  }

  // Generate per-variable inline init function
  void generatePerVariableInit(OpBuilder &builder, Location loc,
                               StringRef moduleName, StringRef contextTypeName,
                               asl::ConstantInitGlobalStorageDeclOp globalVar,
                               ASLToEmitCTypeConverter &typeConverter) {
    std::string varName = sanitizeIdentifier(globalVar.getName());
    std::string initFuncName = moduleName.str() + "_init_" + varName;

    // Get type information
    Type varType = globalVar.getType();
    Type convertedType = typeConverter.convertType(varType);

    // Check if this is a GMP type
    bool isGMPInt = false;
    bool isGMPRational = false;
    if (auto opaqueType = llvm::dyn_cast<emitc::OpaqueType>(convertedType)) {
      isGMPInt = (opaqueType.getValue() == "mpz_t");
      isGMPRational = (opaqueType.getValue() == "mpq_t");
    }

    // Get the constant initial value from the attribute
    StringRef literal = globalVar.getInitialValue();

    // Parse the bitvector literal to get the bits
    SmallVector<uint64_t> words;
    std::string initialValue = "0"; // default for simple types

    if (literal.size() >= 2 && literal.front() == '\'' &&
        literal.back() == '\'') {
      literal = literal.drop_front().drop_back();

      // Check if this is a large bitvector (struct with words array)
      bool isLargeStruct = false;
      int64_t numWords = 0;
      if (auto opaqueType = llvm::dyn_cast<emitc::OpaqueType>(convertedType)) {
        StringRef typeName = opaqueType.getValue();
        if (typeName.starts_with("struct { uint64_t words[")) {
          isLargeStruct = true;
          // Extract number of words from type string
          // Format: "struct { uint64_t words[N]; }"
          size_t start = typeName.find('[') + 1;
          size_t end = typeName.find(']');
          if (start != std::string::npos && end != std::string::npos) {
            std::string numWordsStr = typeName.slice(start, end).str();
            numWords = std::stoll(numWordsStr);
          }
        }
      }

      if (isLargeStruct && numWords > 0) {
        // Parse bits into multiple 64-bit words
        // Bits are stored MSB first in the literal, but we need to distribute
        // them into words where word[0] contains the least significant bits
        words.resize(numWords, 0);

        int bitIndex = literal.size() - 1; // Start from LSB
        for (int wordIdx = 0; wordIdx < numWords && bitIndex >= 0; wordIdx++) {
          uint64_t wordValue = 0;
          // Fill up to 64 bits for this word
          for (int bitInWord = 0; bitInWord < 64 && bitIndex >= 0;
               bitInWord++, bitIndex--) {
            if (literal[bitIndex] == '1') {
              wordValue |= (1ULL << bitInWord);
            }
          }
          words[wordIdx] = wordValue;
        }
      } else {
        // For simple types, convert to single integer value
        uint64_t value = 0;
        for (char c : literal) {
          value = value * 2 + (c == '1' ? 1 : 0);
        }

        // Get the appropriate suffix based on type
        if (auto opaqueType =
                llvm::dyn_cast<emitc::OpaqueType>(convertedType)) {
          StringRef typeName = opaqueType.getValue();
          if (typeName == "uint8_t" || typeName == "uint16_t" ||
              typeName == "uint32_t")
            initialValue = std::to_string(value) + "u";
          else if (typeName == "uint64_t")
            initialValue = std::to_string(value) + "ULL";
          else
            initialValue = std::to_string(value);
        } else {
          initialValue = std::to_string(value);
        }
      }
    }

    // Generate the inline init function
    std::string initFunc = "static inline void " + initFuncName + "(" +
                           contextTypeName.str() + "* ctx) {\n";

    if (isGMPInt) {
      // For GMP integer types, use mpz_init_set_str for initialization
      // The initial value is stored as a decimal string (e.g., "0", "42",
      // "12345") Use mpz_init_set_str to handle arbitrary-precision integers
      std::string decimalValue = literal.str();

      // Remove quotes if present (shouldn't be for integer literals)
      if (decimalValue.size() >= 2 && decimalValue.front() == '\'' &&
          decimalValue.back() == '\'') {
        // This is a bitvector literal being used for an integer - convert it
        StringRef bits = literal.drop_front().drop_back();
        uint64_t value = 0;
        for (char c : bits) {
          value = value * 2 + (c == '1' ? 1 : 0);
        }
        decimalValue = std::to_string(value);
      }

      // Initialize mpz_t with the decimal string value
      // mpz_init_set_str(mpz_t rop, const char *str, int base)
      // base 10 for decimal integers
      initFunc += "  mpz_init_set_str(ctx->" + varName + ", \"" + decimalValue +
                  "\", 10);\n";
    } else if (isGMPRational) {
      // For GMP rational types, use mpq_init and mpq_set_str for initialization
      // The initial value is stored as a string (e.g., "0", "1/2", "3.14")
      std::string rationalValue = literal.str();

      // Initialize mpq_t with the rational string value
      // mpq_set_str(mpq_t rop, const char *str, int base)
      // The string can be:
      // - An integer: "42" -> 42/1
      // - A fraction: "22/7" -> 22/7
      // - A decimal: "3.14" -> 314/100 (after parsing)

      // For now, we support integer and fraction formats directly
      // If the value looks like a decimal (contains '.'), we need to convert it
      if (rationalValue.find('.') != std::string::npos) {
        // Decimal format - need to convert to fraction
        // For example: "3.14" should become "314/100"
        size_t dotPos = rationalValue.find('.');
        std::string intPart = rationalValue.substr(0, dotPos);
        std::string fracPart = rationalValue.substr(dotPos + 1);

        // Calculate denominator (10^number_of_decimal_places)
        uint64_t denominator = 1;
        for (size_t i = 0; i < fracPart.size(); i++) {
          denominator *= 10;
        }

        // Calculate numerator
        uint64_t numerator = std::stoull(intPart + fracPart);

        rationalValue =
            std::to_string(numerator) + "/" + std::to_string(denominator);
      }

      initFunc += "  mpq_init(ctx->" + varName + ");\n";
      initFunc += "  mpq_set_str(ctx->" + varName + ", \"" + rationalValue +
                  "\", 10);\n";
      initFunc += "  mpq_canonicalize(ctx->" + varName + ");\n";
    } else if (!words.empty()) {
      // For large bitvectors, initialize each word individually
      for (size_t i = 0; i < words.size(); i++) {
        initFunc += "  ctx->" + varName + ".words[" + std::to_string(i) +
                    "] = " + std::to_string(words[i]) + "ULL;\n";
      }
    } else {
      // For simple types, direct assignment
      // Special handling for bool: emit 'false' or 'true' instead of 0/1
      bool isBool = false;
      if (auto opaqueType = llvm::dyn_cast<emitc::OpaqueType>(convertedType)) {
        isBool = (opaqueType.getValue() == "bool");
      }
      if (isBool) {
        // Use the literal value if available, otherwise default to false
        std::string boolValue = "false";
        // Accept '0', '1', or string literal 'true'/'false' from IR
        if (literal == "1" || literal == "true" || literal == "TRUE")
          boolValue = "true";
        else if (literal == "0" || literal == "false" || literal == "FALSE")
          boolValue = "false";
        initFunc += "  ctx->" + varName + " = " + boolValue + ";\n";
      } else {
        initFunc += "  ctx->" + varName + " = " + initialValue + ";\n";
      }
    }

    initFunc += "}";

    builder.create<emitc::VerbatimOp>(loc, initFunc);
  }

  // Generate main init function that calls all per-variable inits
  void generateMainInitFunction(
      OpBuilder &builder, Location loc, StringRef moduleName,
      StringRef contextTypeName,
      ArrayRef<asl::ConstantInitGlobalStorageDeclOp> globalVars) {
    std::string initFunc = "void " + moduleName.str() + "_init(" +
                           contextTypeName.str() + "* ctx) {\n";

    // Call each per-variable init function
    for (auto globalVar : globalVars) {
      std::string varName = sanitizeIdentifier(globalVar.getName());
      std::string initFuncName = moduleName.str() + "_init_" + varName;
      initFunc += "  " + initFuncName + "(ctx);\n";
    }

    initFunc += "}";

    builder.create<emitc::VerbatimOp>(loc, initFunc);
  }

  // Generate free function for cleanup
  void generateFreeFunction(
      OpBuilder &builder, Location loc, StringRef moduleName,
      StringRef contextTypeName,
      ArrayRef<asl::ConstantInitGlobalStorageDeclOp> globalVars,
      ASLToEmitCTypeConverter &typeConverter) {
    std::string freeFunc = "void " + moduleName.str() + "_free(" +
                           contextTypeName.str() + "* ctx) {\n";

    // Cleanup GMP types and other dynamically allocated resources
    bool needsCleanup = false;

    for (auto globalVar : globalVars) {
      Type varType = globalVar.getType();
      Type convertedType = typeConverter.convertType(varType);

      // Check if this is a GMP integer type that needs cleanup
      if (auto opaqueType = llvm::dyn_cast<emitc::OpaqueType>(convertedType)) {
        StringRef typeName = opaqueType.getValue();
        std::string varName = sanitizeIdentifier(globalVar.getName());

        if (typeName == "mpz_t") {
          // GMP integer requires mpz_clear
          freeFunc += "  mpz_clear(ctx->" + varName + ");\n";
          needsCleanup = true;
        } else if (typeName == "mpq_t") {
          // GMP rational requires mpq_clear
          freeFunc += "  mpq_clear(ctx->" + varName + ");\n";
          needsCleanup = true;
        }
        // TODO: Add cleanup for other dynamically allocated types
      }
    }

    if (!needsCleanup) {
      freeFunc += "  // No dynamic allocations to clean up\n";
    }

    freeFunc += "}";

    builder.create<emitc::VerbatimOp>(loc, freeFunc);
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation Function
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createASLToEmitCPass() {
  return std::make_unique<ASLToEmitCPass>();
}

} // namespace asl
} // namespace mlir
