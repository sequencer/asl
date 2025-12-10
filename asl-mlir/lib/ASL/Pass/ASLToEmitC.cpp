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

    // Convert ASL string type to C const char*
    addConversion([this](asl::StringType type) -> std::optional<Type> {
      return convertStringType(type);
    });

    // Convert ASL enum type to C enum (represented as opaque type)
    addConversion([this](asl::EnumType type) -> std::optional<Type> {
      return convertEnumType(type);
    });

    // Convert ASL named type by resolving to underlying type
    addConversion([this](asl::NamedType type) -> std::optional<Type> {
      return convertNamedType(type);
    });

    // Convert ASL label type to C enum value (int)
    addConversion([this](asl::LabelType type) -> std::optional<Type> {
      return convertLabelType(type);
    });

    // Convert ASL tuple type to C struct
    addConversion([this](asl::TupleType type) -> std::optional<Type> {
      return convertTupleType(type);
    });

    // Convert ASL slice type to C struct { long start; long length; }
    addConversion([this](asl::SliceType type) -> std::optional<Type> {
      return convertSliceType(type);
    });

    // Add source materialization for converting back from converted types
    addSourceMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() != 1)
        return Value();
      if (inputs[0].getType() == resultType)
        return inputs[0];
      // Handle lvalue to value conversion
      if (auto lvalueType = llvm::dyn_cast<emitc::LValueType>(inputs[0].getType())) {
        if (lvalueType.getValueType() == resultType) {
          return builder.create<emitc::LoadOp>(loc, resultType, inputs[0]);
        }
      }
      return Value();
    });

    // Add target materialization for converting to target types
    addTargetMaterialization([](OpBuilder &builder, Type resultType,
                                ValueRange inputs, Location loc,
                                Type originalType) -> Value {
      if (inputs.size() != 1)
        return Value();
      if (inputs[0].getType() == resultType)
        return inputs[0];
      // Handle lvalue to value conversion
      if (auto lvalueType = llvm::dyn_cast<emitc::LValueType>(inputs[0].getType())) {
        if (lvalueType.getValueType() == resultType) {
          return builder.create<emitc::LoadOp>(loc, resultType, inputs[0]);
        }
      }
      return Value();
    });
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

  // Convert ASL string type to C const char*
  Type convertStringType(asl::StringType type) {
    // ASL string types are converted to C's const char* for immutable strings
    // This provides:
    // - Direct compatibility with C standard library string functions
    // - Minimal memory overhead (just a pointer)
    // - Natural integration with C I/O and formatting functions
    // - Read-only semantics enforced by const qualifier
    // ASL strings consist of printable ASCII characters (decimal 32-126) plus
    // escape sequences for special characters (newline, tab, backslash,
    // double-quote)
    return emitc::OpaqueType::get(context, "const char*");
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

  // Convert ASL enum type to C enum (represented as opaque type)
  Type convertEnumType(asl::EnumType type) {
    // Generate enum type name from labels
    // For now, use a generic enum type - the actual typedef will be generated
    // when we see the type declaration
    // We represent it as the sanitized enum type name

    // Extract labels to create a deterministic type name
    auto labels = type.getStringLabels();
    if (labels.empty()) {
      return emitc::OpaqueType::get(context, "int");
    }

    // Create a type name based on first label (will be refined by type decl)
    // For anonymous enums, use int
    return emitc::OpaqueType::get(context, "int");
  }

  // Convert ASL named type by resolving to underlying type
  Type convertNamedType(asl::NamedType type) {
    // If the named type has a resolved type attribute, convert that
    if (type.getResolvedType()) {
      Type resolvedType = type.getResolvedType().getValue();
      if (auto enumType = llvm::dyn_cast<asl::EnumType>(resolvedType)) {
        // Use the name for the enum type with asl_ prefix
        std::string enumTypeName =
            "asl_" + sanitizeIdentifier(type.getName().str());
        return emitc::OpaqueType::get(context, enumTypeName);
      } else if (auto tupleType =
                     llvm::dyn_cast<asl::TupleType>(resolvedType)) {
        // Use the name for the tuple type (struct typedef) with asl_ prefix
        std::string tupleTypeName =
            "asl_" + sanitizeIdentifier(type.getName().str());
        return emitc::OpaqueType::get(context, tupleTypeName);
      }
      // Recursively convert the resolved type
      return convertType(resolvedType);
    }

    // Otherwise, assume it's an enum type and use the name with asl_ prefix
    std::string typeName = "asl_" + sanitizeIdentifier(type.getName().str());
    return emitc::OpaqueType::get(context, typeName);
  }

  // Convert ASL label type to C enum value (int)
  Type convertLabelType(asl::LabelType type) {
    // Label literals will be resolved to specific enum values
    // The type itself is just an int
    return emitc::OpaqueType::get(context, "int");
  }

  // Convert ASL tuple type to C struct
  Type convertTupleType(asl::TupleType type) {
    // ASL tuple types are lowered to C structures (struct) to maintain type
    // information and enable efficient element access.
    //
    // Each tuple element is stored as a struct field named "itemN" where N is
    // the zero-based index. This provides:
    // - Type safety with distinct field types
    // - Efficient memory layout with sequential element storage
    // - Natural mapping to C's type system
    // - Compatibility with C calling conventions
    // - Debugger support for inspecting tuple contents
    //
    // Tuple types must contain at least two elements (single-element tuples
    // are not valid in ASL).

    auto types = type.getTypes();
    if (types.size() < 2) {
      mlir::emitError(UnknownLoc::get(context))
          << "tuple must contain at least two elements";
      return Type();
    }

    // Build the struct type string: "struct { type0 item0; type1 item1; ... }"
    std::string structType = "struct { ";

    for (size_t i = 0; i < types.size(); ++i) {
      if (i > 0)
        structType += " ";

      // Get the type attribute and extract the type
      auto typeAttr = llvm::dyn_cast<TypeAttr>(types[i]);
      if (!typeAttr) {
        mlir::emitError(UnknownLoc::get(context))
            << "tuple element " << i << " is not a type attribute";
        return Type();
      }

      Type elementType = typeAttr.getValue();

      // Recursively convert the element type
      Type convertedElementType = convertType(elementType);
      if (!convertedElementType) {
        mlir::emitError(UnknownLoc::get(context))
            << "failed to convert tuple element type " << i;
        return Type();
      }

      // Get the C type string
      std::string cType;
      if (auto opaqueType =
              llvm::dyn_cast<emitc::OpaqueType>(convertedElementType)) {
        cType = opaqueType.getValue().str();
      } else {
        mlir::emitError(UnknownLoc::get(context))
            << "converted tuple element type " << i << " is not an opaque type";
        return Type();
      }

      // Add field: "type itemN;"
      structType += cType + " item" + std::to_string(i) + ";";
    }

    structType += " }";

    return emitc::OpaqueType::get(context, structType);
  }

  // Convert ASL slice type to C struct
  Type convertSliceType(asl::SliceType type) {
    // ASL slice descriptors are used to describe index ranges for bitvector
    // and array slicing operations. We represent them as a simple struct
    // with start position and length fields.
    //
    // The struct contains:
    // - start: the starting bit position (long to handle GMP values)
    // - length: the number of bits to extract (long)
    //
    // This representation is sufficient for all slice variants:
    // - SliceSingle(i): start=i, length=1
    // - SliceRange(j, i): start=j, length=i-j (caller computes)
    // - SliceLength(i, n): start=i, length=n
    // - SliceStar(i, n): start=i*n, length=n (caller computes)
    return emitc::OpaqueType::get(context, "struct { long start; long length; }");
  }
};

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

// Convert asl.expr.literal.string to emitc.constant
struct LiteralStringOpLowering
    : public OpConversionPattern<asl::LiteralStringOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::LiteralStringOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto stringType = llvm::dyn_cast<asl::StringType>(op.getType());
    if (!stringType)
      return failure();

    // Get the string literal value
    StringRef value = op.getValue();

    // Convert the type to const char*
    Type convertedType = getTypeConverter()->convertType(stringType);
    if (!convertedType)
      return failure();

    // Create EmitC constant with the string literal
    // The value is already in the correct format (quoted string)
    // EmitC will emit it as: const char* var = "string value";
    std::string escapedValue = value.str();

    // Ensure the string is properly quoted
    if (escapedValue.empty() || escapedValue.front() != '"') {
      escapedValue = "\"" + escapedValue + "\"";
    }

    auto constantOp = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), convertedType,
        emitc::OpaqueAttr::get(rewriter.getContext(), escapedValue));

    rewriter.replaceOp(op, constantOp.getResult());
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

// Convert asl.expr.literal.label to emitc.constant with enum value
struct LiteralLabelOpLowering
    : public OpConversionPattern<asl::LiteralLabelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::LiteralLabelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the label value
    StringRef label = op.getValue();

    // Convert the type
    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();

    // The label will be used as an enum constant
    // We need to find the enum type this label belongs to
    // For now, we'll emit it as a reference to the enum constant
    // The actual enum type name will be determined from context

    // Create EmitC constant with the label as an identifier
    // This will be emitted as: EnumType_LABEL
    // The enum type prefix will be added during global var initialization
    auto constantOp = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), convertedType,
        emitc::OpaqueAttr::get(rewriter.getContext(), label.str()));

    rewriter.replaceOp(op, constantOp.getResult());
    return success();
  }
};

// Convert asl.expr.literal.int to GMP mpz_t initialization
// ASL integers are arbitrary-precision, so we use GMP's mpz_t type.
// This pattern creates a temporary mpz_t variable and initializes it
// with mpz_init_set_str using base 10.
struct LiteralIntOpLowering : public OpConversionPattern<asl::LiteralIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::LiteralIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Get the integer value as string
    StringRef valueStr = op.getValue();

    // Convert the type (should be mpz_t)
    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();

    // Create a temporary mpz_t variable
    auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
    auto lvalueType = emitc::LValueType::get(mpzType);
    auto varOp = rewriter.create<emitc::VariableOp>(
        loc, lvalueType, emitc::OpaqueAttr::get(context, ""));

    // Load the variable to get the value for GMP function calls
    auto loadedVar =
        rewriter.create<emitc::LoadOp>(loc, mpzType, varOp.getResult());

    // Create string constant for the value
    std::string quotedValue = "\"" + valueStr.str() + "\"";
    auto strConstant = rewriter.create<emitc::ConstantOp>(
        loc, emitc::OpaqueType::get(context, "const char*"),
        emitc::OpaqueAttr::get(context, quotedValue));

    // Create base constant (10 for decimal)
    auto baseConstant = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(10));

    // Call mpz_init_set_str(var, str, base)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_init_set_str",
        ValueRange{loadedVar.getResult(), strConstant.getResult(),
                   baseConstant.getResult()},
        nullptr, nullptr);

    // Replace the op with the lvalue variable
    rewriter.replaceOp(op, varOp.getResult());
    return success();
  }
};

// Convert asl.expr.literal.bool to emitc.constant with true/false
// ASL boolean literals map directly to C's bool type from stdbool.h
struct LiteralBoolOpLowering : public OpConversionPattern<asl::LiteralBoolOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::LiteralBoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert the type (should be bool)
    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();

    // Get the boolean value
    bool value = op.getValue();

    // Create EmitC constant with "true" or "false"
    std::string valueStr = value ? "true" : "false";
    auto constantOp = rewriter.create<emitc::ConstantOp>(
        op.getLoc(), convertedType,
        emitc::OpaqueAttr::get(rewriter.getContext(), valueStr));

    rewriter.replaceOp(op, constantOp.getResult());
    return success();
  }
};

// Convert asl.expr.literal.real to GMP mpq_t initialization
// ASL reals are exact rationals (p/q), so we use GMP's mpq_t type.
// The value string can be in various formats:
// - Decimal: "3.14" -> converted to fraction
// - Fraction: "22/7"
// - Integer: "42" -> 42/1
struct LiteralRealOpLowering : public OpConversionPattern<asl::LiteralRealOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::LiteralRealOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Get the real value as string
    StringRef valueStr = op.getValue();

    // Convert the type (should be mpq_t)
    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();

    // Create a temporary mpq_t variable
    auto mpqType = emitc::OpaqueType::get(context, "mpq_t");
    auto lvalueType = emitc::LValueType::get(mpqType);
    auto varOp = rewriter.create<emitc::VariableOp>(
        loc, lvalueType, emitc::OpaqueAttr::get(context, ""));

    // Load the variable to get the value for GMP function calls
    auto loadedVar =
        rewriter.create<emitc::LoadOp>(loc, mpqType, varOp.getResult());

    // Initialize the mpq_t
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "mpq_init",
                                         ValueRange{loadedVar.getResult()},
                                         nullptr, nullptr);

    // Convert decimal format to fraction if needed
    std::string rationalValue = valueStr.str();
    if (rationalValue.find('.') != std::string::npos) {
      // Decimal format - convert to fraction
      // e.g., "3.14" -> "314/100"
      size_t dotPos = rationalValue.find('.');
      std::string intPart = rationalValue.substr(0, dotPos);
      std::string fracPart = rationalValue.substr(dotPos + 1);

      // Calculate denominator (10^number_of_decimal_places)
      std::string denominator = "1";
      for (size_t i = 0; i < fracPart.size(); i++) {
        denominator += "0";
      }

      // Combine integer and fractional parts for numerator
      std::string numerator = intPart + fracPart;

      // Remove leading zeros from numerator (but keep at least one digit)
      size_t firstNonZero = numerator.find_first_not_of('0');
      if (firstNonZero != std::string::npos && firstNonZero > 0) {
        numerator = numerator.substr(firstNonZero);
      } else if (firstNonZero == std::string::npos) {
        numerator = "0";
      }

      rationalValue = numerator + "/" + denominator;
    }

    // Create string constant for the value
    std::string quotedValue = "\"" + rationalValue + "\"";
    auto strConstant = rewriter.create<emitc::ConstantOp>(
        loc, emitc::OpaqueType::get(context, "const char*"),
        emitc::OpaqueAttr::get(context, quotedValue));

    // Create base constant (10 for decimal)
    auto baseConstant = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(10));

    // Call mpq_set_str(var, str, base)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_set_str",
        ValueRange{loadedVar.getResult(), strConstant.getResult(),
                   baseConstant.getResult()},
        nullptr, nullptr);

    // Canonicalize the rational (reduce to lowest terms)
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "mpq_canonicalize",
                                         ValueRange{loadedVar.getResult()},
                                         nullptr, nullptr);

    // Replace the op with the lvalue variable
    rewriter.replaceOp(op, varOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Binary Operation Lowering Patterns
//===----------------------------------------------------------------------===//

// Helper: Create a temporary mpz_t variable and initialize it
static Value createTempMpzVar(ConversionPatternRewriter &rewriter, Location loc,
                              MLIRContext *context) {
  auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
  auto lvalueType = emitc::LValueType::get(mpzType);
  auto varOp = rewriter.create<emitc::VariableOp>(
      loc, lvalueType, emitc::OpaqueAttr::get(context, ""));
  auto loadedVar =
      rewriter.create<emitc::LoadOp>(loc, mpzType, varOp.getResult());
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "mpz_init",
                                       ValueRange{loadedVar.getResult()},
                                       nullptr, nullptr);
  return varOp.getResult();
}

// Helper: Create a temporary mpq_t variable and initialize it
static Value createTempMpqVar(ConversionPatternRewriter &rewriter, Location loc,
                              MLIRContext *context) {
  auto mpqType = emitc::OpaqueType::get(context, "mpq_t");
  auto lvalueType = emitc::LValueType::get(mpqType);
  auto varOp = rewriter.create<emitc::VariableOp>(
      loc, lvalueType, emitc::OpaqueAttr::get(context, ""));
  auto loadedVar =
      rewriter.create<emitc::LoadOp>(loc, mpqType, varOp.getResult());
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "mpq_init",
                                       ValueRange{loadedVar.getResult()},
                                       nullptr, nullptr);
  return varOp.getResult();
}

// Helper: Load a GMP value from an lvalue for passing to GMP functions
static Value loadGmpValue(ConversionPatternRewriter &rewriter, Location loc,
                          Value value) {
  if (auto lvalueType = llvm::dyn_cast<emitc::LValueType>(value.getType())) {
    return rewriter.create<emitc::LoadOp>(loc, lvalueType.getValueType(), value);
  }
  return value;
}

// Integer binary operations: add, sub, mul
// These lower to GMP mpz_* functions
template <typename OpTy>
struct IntBinaryOpLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  IntBinaryOpLowering(const TypeConverter &converter, MLIRContext *context,
                      StringRef gmpFunc)
      : OpConversionPattern<OpTy>(converter, context), gmpFuncName(gmpFunc) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Create temporary mpz_t variable for result
    Value resultVar = createTempMpzVar(rewriter, loc, context);

    // Load values for GMP function call
    auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
    Value loadedResult = rewriter.create<emitc::LoadOp>(loc, mpzType, resultVar);
    Value loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    Value loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Call the GMP function: mpz_func(result, lhs, rhs)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, gmpFuncName,
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }

private:
  std::string gmpFuncName;
};

// Integer division operations with different rounding modes
struct BinopDivOpLowering : public OpConversionPattern<asl::BinopDivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    Value resultVar = createTempMpzVar(rewriter, loc, context);
    auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
    Value loadedResult = rewriter.create<emitc::LoadOp>(loc, mpzType, resultVar);
    Value loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    Value loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // ASL DIV is truncated division (toward zero): mpz_tdiv_q
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_tdiv_q",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// DIVRM is floor division (toward negative infinity)
struct BinopDivrmOpLowering : public OpConversionPattern<asl::BinopDivrmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopDivrmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    Value resultVar = createTempMpzVar(rewriter, loc, context);
    auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
    Value loadedResult = rewriter.create<emitc::LoadOp>(loc, mpzType, resultVar);
    Value loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    Value loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Floor division: mpz_fdiv_q
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_fdiv_q",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// MOD operation (remainder)
struct BinopModOpLowering : public OpConversionPattern<asl::BinopModOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopModOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    Value resultVar = createTempMpzVar(rewriter, loc, context);
    auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
    Value loadedResult = rewriter.create<emitc::LoadOp>(loc, mpzType, resultVar);
    Value loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    Value loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // ASL MOD uses floor remainder semantics: mpz_fdiv_r
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_fdiv_r",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Power operation
struct BinopPowOpLowering : public OpConversionPattern<asl::BinopPowOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopPowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    Value resultVar = createTempMpzVar(rewriter, loc, context);
    auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
    Value loadedResult = rewriter.create<emitc::LoadOp>(loc, mpzType, resultVar);
    Value loadedBase = loadGmpValue(rewriter, loc, adaptor.getLhs());

    // The exponent needs to be converted to unsigned long
    // First get the exponent value (it should be a small integer)
    Value loadedExp = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Convert exponent to unsigned long using mpz_get_ui
    auto ulongType = emitc::OpaqueType::get(context, "unsigned long");
    auto expUlong = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{ulongType}, "mpz_get_ui", ValueRange{loadedExp}, nullptr,
        nullptr);

    // mpz_pow_ui(result, base, exp)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_pow_ui",
        ValueRange{loadedResult, loadedBase, expUlong.getResult(0)}, nullptr,
        nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Shift left operation
struct BinopShlOpLowering : public OpConversionPattern<asl::BinopShlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopShlOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    Value resultVar = createTempMpzVar(rewriter, loc, context);
    auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
    Value loadedResult = rewriter.create<emitc::LoadOp>(loc, mpzType, resultVar);
    Value loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    Value loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Convert shift amount to unsigned long
    auto ulongType = emitc::OpaqueType::get(context, "unsigned long");
    auto shiftAmount = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{ulongType}, "mpz_get_ui", ValueRange{loadedRhs}, nullptr,
        nullptr);

    // mpz_mul_2exp(result, op, shift) - multiply by 2^shift
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_mul_2exp",
        ValueRange{loadedResult, loadedLhs, shiftAmount.getResult(0)}, nullptr,
        nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Shift right operation
struct BinopShrOpLowering : public OpConversionPattern<asl::BinopShrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopShrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    Value resultVar = createTempMpzVar(rewriter, loc, context);
    auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
    Value loadedResult = rewriter.create<emitc::LoadOp>(loc, mpzType, resultVar);
    Value loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    Value loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Convert shift amount to unsigned long
    auto ulongType = emitc::OpaqueType::get(context, "unsigned long");
    auto shiftAmount = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{ulongType}, "mpz_get_ui", ValueRange{loadedRhs}, nullptr,
        nullptr);

    // mpz_fdiv_q_2exp(result, op, shift) - floor division by 2^shift
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_fdiv_q_2exp",
        ValueRange{loadedResult, loadedLhs, shiftAmount.getResult(0)}, nullptr,
        nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Real binary operations: add, sub, mul
// These lower to GMP mpq_* functions
template <typename OpTy>
struct RealBinaryOpLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  RealBinaryOpLowering(const TypeConverter &converter, MLIRContext *context,
                       StringRef gmpFunc)
      : OpConversionPattern<OpTy>(converter, context), gmpFuncName(gmpFunc) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Create temporary mpq_t variable for result
    Value resultVar = createTempMpqVar(rewriter, loc, context);

    // Load values for GMP function call
    auto mpqType = emitc::OpaqueType::get(context, "mpq_t");
    Value loadedResult = rewriter.create<emitc::LoadOp>(loc, mpqType, resultVar);
    Value loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    Value loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Call the GMP function: mpq_func(result, lhs, rhs)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, gmpFuncName,
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }

private:
  std::string gmpFuncName;
};

// Real division (RDIV)
struct BinopRdivOpLowering : public OpConversionPattern<asl::BinopRdivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopRdivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    Value resultVar = createTempMpqVar(rewriter, loc, context);
    auto mpqType = emitc::OpaqueType::get(context, "mpq_t");
    Value loadedResult = rewriter.create<emitc::LoadOp>(loc, mpqType, resultVar);
    Value loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    Value loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_div",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Bitvector binary operations: add, sub, mul, and, or, xor
// For bitvectors <= 64 bits, use native C operators
// The result is masked to the appropriate bit width
template <typename OpTy>
struct BitsBinaryOpLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  BitsBinaryOpLowering(const TypeConverter &converter, MLIRContext *context,
                       StringRef cOperator)
      : OpConversionPattern<OpTy>(converter, context), cOp(cOperator) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    Type convertedType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return failure();

    // Get the bit width from the result type for masking
    int64_t bitWidth = 0;
    if (auto bitsType =
            llvm::dyn_cast<asl::BitsType>(op.getResult().getType())) {
      bitWidth = bitsType.getWidth().getInt();
    }

    // For bitvectors > 64 bits, we would need special handling
    // For now, only handle <= 64 bits
    if (bitWidth > 64) {
      return rewriter.notifyMatchFailure(
          op, "large bitvector operations not yet supported");
    }

    // Create the binary operation using emitc.expression or call_opaque
    // Use emitc.call_opaque with the operator as the callee
    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{convertedType}, cOp,
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, nullptr, nullptr);

    // Apply mask if needed (for non-power-of-2 widths)
    if (bitWidth > 0 && bitWidth < 64 && (bitWidth & (bitWidth - 1)) != 0) {
      // Need to mask the result to bitWidth bits
      uint64_t mask = (1ULL << bitWidth) - 1;
      std::string maskStr = std::to_string(mask);
      if (bitWidth > 32)
        maskStr += "ULL";
      else
        maskStr += "u";

      auto maskConstant = rewriter.create<emitc::ConstantOp>(
          loc, convertedType,
          emitc::OpaqueAttr::get(context, maskStr));

      auto maskedResult = rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{convertedType}, "&",
          ValueRange{result.getResult(0), maskConstant.getResult()}, nullptr,
          nullptr);

      rewriter.replaceOp(op, maskedResult.getResult(0));
    } else {
      rewriter.replaceOp(op, result.getResult(0));
    }

    return success();
  }

private:
  std::string cOp;
};

// Bitvector concatenation
struct BinopConcatOpLowering : public OpConversionPattern<asl::BinopConcatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();

    // Get the bit width of the RHS for shifting
    int64_t rhsWidth = 0;
    if (auto rhsBitsType = llvm::dyn_cast<asl::BitsType>(op.getRhs().getType())) {
      rhsWidth = rhsBitsType.getWidth().getInt();
    }

    if (rhsWidth <= 0 || rhsWidth > 64) {
      return rewriter.notifyMatchFailure(op, "invalid RHS width for concat");
    }

    // concat(lhs, rhs) = (lhs << rhsWidth) | rhs
    // First shift lhs left by rhsWidth
    std::string shiftStr = std::to_string(rhsWidth);
    auto shiftConstant = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(rhsWidth));

    auto shiftedLhs = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{convertedType}, "<<",
        ValueRange{adaptor.getLhs(), shiftConstant.getResult()}, nullptr,
        nullptr);

    // Then OR with rhs
    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{convertedType}, "|",
        ValueRange{shiftedLhs.getResult(0), adaptor.getRhs()}, nullptr,
        nullptr);

    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

// Boolean binary operations: band, bor, beq, impl
// These use native C operators
struct BinopBandOpLowering : public OpConversionPattern<asl::BinopBandOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopBandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();

    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{convertedType}, "&&",
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, nullptr, nullptr);

    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

struct BinopBorOpLowering : public OpConversionPattern<asl::BinopBorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopBorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();

    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{convertedType}, "||",
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, nullptr, nullptr);

    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

struct BinopBeqOpLowering : public OpConversionPattern<asl::BinopBeqOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopBeqOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();

    // Boolean equivalence: lhs == rhs
    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{convertedType}, "==",
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, nullptr, nullptr);

    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

struct BinopImplOpLowering : public OpConversionPattern<asl::BinopImplOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::BinopImplOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();

    // Implication: lhs => rhs is equivalent to !lhs || rhs
    auto notLhs = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{convertedType}, "!", ValueRange{adaptor.getLhs()},
        nullptr, nullptr);

    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{convertedType}, "||",
        ValueRange{notLhs.getResult(0), adaptor.getRhs()}, nullptr, nullptr);

    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Unary Operation Lowering Patterns (Phase 3)
//===----------------------------------------------------------------------===//

// Boolean NOT: !operand
struct UnopBnotOpLowering : public OpConversionPattern<asl::UnopBnotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::UnopBnotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type convertedType = getTypeConverter()->convertType(op.getType());
    if (!convertedType)
      return failure();

    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{convertedType}, "!", ValueRange{adaptor.getOperand()},
        nullptr, nullptr);

    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

// Integer negation: mpz_neg(result, operand)
struct UnopNegIntOpLowering : public OpConversionPattern<asl::UnopNegOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::UnopNegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Only handle integer types
    if (!llvm::isa<asl::IntType>(op.getOperand().getType()))
      return failure();

    // Create temporary mpz_t variable for result
    Value resultVar = createTempMpzVar(rewriter, loc, context);

    // Load values for GMP function call
    auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
    Value loadedResult = rewriter.create<emitc::LoadOp>(loc, mpzType, resultVar);
    Value loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // Call mpz_neg(result, operand)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_neg",
        ValueRange{loadedResult, loadedOperand}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Real negation: mpq_neg(result, operand)
struct UnopNegRealOpLowering : public OpConversionPattern<asl::UnopNegOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::UnopNegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Only handle real types
    if (!llvm::isa<asl::RealType>(op.getOperand().getType()))
      return failure();

    // Create temporary mpq_t variable for result
    Value resultVar = createTempMpqVar(rewriter, loc, context);

    // Load values for GMP function call
    auto mpqType = emitc::OpaqueType::get(context, "mpq_t");
    Value loadedResult = rewriter.create<emitc::LoadOp>(loc, mpqType, resultVar);
    Value loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // Call mpq_neg(result, operand)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_neg",
        ValueRange{loadedResult, loadedOperand}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Bitvector NOT: ~operand
struct UnopNotOpLowering : public OpConversionPattern<asl::UnopNotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::UnopNotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    Type convertedType = getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return failure();

    // Get the bit width from the result type for masking
    int64_t bitWidth = 0;
    if (auto bitsType = llvm::dyn_cast<asl::BitsType>(op.getResult().getType())) {
      bitWidth = bitsType.getWidth().getInt();
    }

    // For bitvectors > 64 bits, we would need special handling
    if (bitWidth > 64) {
      return rewriter.notifyMatchFailure(
          op, "large bitvector NOT not yet supported");
    }

    // Create the NOT operation using emitc.call_opaque
    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{convertedType}, "~",
        ValueRange{adaptor.getOperand()}, nullptr, nullptr);

    // Apply mask to clear high bits (for non-native sizes)
    if (bitWidth > 0 && bitWidth < 64) {
      uint64_t mask = (1ULL << bitWidth) - 1;
      std::string maskStr = std::to_string(mask);
      if (bitWidth > 32)
        maskStr += "ULL";
      else
        maskStr += "u";

      auto maskConstant = rewriter.create<emitc::ConstantOp>(
          loc, convertedType,
          emitc::OpaqueAttr::get(context, maskStr));

      auto maskedResult = rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{convertedType}, "&",
          ValueRange{result.getResult(0), maskConstant.getResult()}, nullptr,
          nullptr);

      rewriter.replaceOp(op, maskedResult.getResult(0));
    } else {
      rewriter.replaceOp(op, result.getResult(0));
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Phase 4: Control Flow Lowering Patterns
//===----------------------------------------------------------------------===//

// Conditional expression: condition ? then_expr : else_expr
struct CondOpLowering : public OpConversionPattern<asl::CondOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::CondOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type convertedType = getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return failure();

    // Use emitc.conditional for ternary operation
    auto result = rewriter.create<emitc::ConditionalOp>(
        loc, convertedType, adaptor.getCondition(), adaptor.getThenExpr(),
        adaptor.getElseExpr());

    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

// Pass statement: no-op, just erase
struct StmtPassOpLowering : public OpConversionPattern<asl::StmtPassOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::StmtPassOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Pass is a no-op, just erase it
    rewriter.eraseOp(op);
    return success();
  }
};

// Return statement: func.return or emitc equivalent
struct StmtReturnOpLowering : public OpConversionPattern<asl::StmtReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::StmtReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getValue()) {
      // Return with value
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getValue());
    } else {
      // Return without value
      rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    }
    return success();
  }
};

// Sequence statement: inline the body operations
struct StmtSeqOpLowering : public OpConversionPattern<asl::StmtSeqOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::StmtSeqOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Inline the body region at the current position
    Region &bodyRegion = op.getBody();
    if (bodyRegion.empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    // Move operations from body block to parent
    Block &bodyBlock = bodyRegion.front();
    rewriter.inlineBlockBefore(&bodyBlock, op);
    rewriter.eraseOp(op);
    return success();
  }
};

// Conditional statement: if-then-else
struct StmtCondOpLowering : public OpConversionPattern<asl::StmtCondOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::StmtCondOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Create scf.if with then/else regions
    auto ifOp = rewriter.create<scf::IfOp>(loc, adaptor.getCondition(),
                                           /*withElseRegion=*/true);

    // Move the then block contents
    Region &thenRegion = op.getBranches();
    if (thenRegion.hasOneBlock()) {
      Block &aslThenBlock = thenRegion.front();
      rewriter.inlineBlockBefore(&aslThenBlock, &ifOp.getThenRegion().front(),
                                 ifOp.getThenRegion().front().begin());
    }

    // Move the else block contents (second block in branches region)
    if (thenRegion.getBlocks().size() > 1) {
      Block &aslElseBlock = *std::next(thenRegion.begin());
      rewriter.inlineBlockBefore(&aslElseBlock, &ifOp.getElseRegion().front(),
                                 ifOp.getElseRegion().front().begin());
    }

    // Add scf.yield terminators if needed
    for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
      Block &block = region->front();
      if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>()) {
        rewriter.setInsertionPointToEnd(&block);
        rewriter.create<scf::YieldOp>(loc);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Assert statement: runtime assertion
struct StmtAssertOpLowering : public OpConversionPattern<asl::StmtAssertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::StmtAssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Generate assert() call
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "assert", ValueRange{adaptor.getCondition()}, nullptr,
        nullptr);

    rewriter.eraseOp(op);
    return success();
  }
};

// Unreachable statement: __builtin_unreachable()
struct StmtUnreachableOpLowering
    : public OpConversionPattern<asl::StmtUnreachableOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::StmtUnreachableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Generate __builtin_unreachable() call
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{},
                                         "__builtin_unreachable", ValueRange{},
                                         nullptr, nullptr);

    rewriter.eraseOp(op);
    return success();
  }
};

// For loop: for index = start to/downto end [limit L] do body
// Loops with GMP bounds are complex - for now, emit a warning and fail
// A full implementation would use while loops with GMP comparisons
struct StmtForOpLowering : public OpConversionPattern<asl::StmtForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::StmtForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For loops with GMP integer bounds require complex lowering
    // For now, fail with a clear message - this needs a dedicated lowering
    // that converts GMP integers to native types for loop control
    return rewriter.notifyMatchFailure(
        op, "for loops with GMP integer bounds not yet supported - "
            "requires conversion to while loop with GMP comparisons");
  }
};

// While loop: while condition [limit L] do body
struct StmtWhileOpLowering : public OpConversionPattern<asl::StmtWhileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::StmtWhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Check if limit is specified - if so, we need special handling
    if (adaptor.getLimit()) {
      return rewriter.notifyMatchFailure(
          op, "while loops with limit not yet supported");
    }

    // Create scf.while operation
    // The condition is evaluated before each iteration
    auto whileOp = rewriter.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});

    // Set up the "before" region (condition check)
    Block *beforeBlock = rewriter.createBlock(&whileOp.getBefore());
    rewriter.setInsertionPointToEnd(beforeBlock);
    rewriter.create<scf::ConditionOp>(loc, adaptor.getCondition(), ValueRange{});

    // Set up the "after" region (body)
    Block *afterBlock = rewriter.createBlock(&whileOp.getAfter());

    // Move the body operations
    Region &bodyRegion = op.getBody();
    if (!bodyRegion.empty()) {
      Block &bodyBlock = bodyRegion.front();
      rewriter.inlineBlockBefore(&bodyBlock, afterBlock, afterBlock->begin());
    }

    // Add yield at end of after block
    rewriter.setInsertionPointToEnd(afterBlock);
    rewriter.create<scf::YieldOp>(loc);

    rewriter.eraseOp(op);
    return success();
  }
};

// Repeat-until loop: repeat body until condition [limit L]
struct StmtRepeatOpLowering : public OpConversionPattern<asl::StmtRepeatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::StmtRepeatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Check if limit is specified - if so, we need special handling
    if (adaptor.getLimit()) {
      return rewriter.notifyMatchFailure(
          op, "repeat loops with limit not yet supported");
    }

    // Create scf.while operation for do-while semantics
    // The body executes at least once, then condition is checked
    auto whileOp = rewriter.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});

    // Set up the "before" region (body + condition check)
    Block *beforeBlock = rewriter.createBlock(&whileOp.getBefore());

    // Move the body operations to before block
    Region &bodyRegion = op.getBody();
    if (!bodyRegion.empty()) {
      Block &bodyBlock = bodyRegion.front();
      rewriter.inlineBlockBefore(&bodyBlock, beforeBlock, beforeBlock->begin());
    }

    // Add condition check at end - note: repeat-until continues while NOT cond
    rewriter.setInsertionPointToEnd(beforeBlock);
    // NOT the condition since repeat-until exits when condition is TRUE
    auto boolType = emitc::OpaqueType::get(rewriter.getContext(), "bool");
    auto negatedCond = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{boolType}, "!",
        ValueRange{adaptor.getCondition()}, nullptr, nullptr);
    rewriter.create<scf::ConditionOp>(loc, negatedCond.getResult(0),
                                      ValueRange{});

    // Set up the "after" region (empty, just yield)
    Block *afterBlock = rewriter.createBlock(&whileOp.getAfter());
    rewriter.setInsertionPointToEnd(afterBlock);
    rewriter.create<scf::YieldOp>(loc);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Phase 5: Function Declaration Lowering Patterns
//===----------------------------------------------------------------------===//

// Function declaration: asl.func -> func.func
// This pattern converts the function signature and lets the dialect conversion
// framework handle the body operations
struct FuncDeclOpLowering : public OpConversionPattern<asl::FuncDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::FuncDeclOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Skip primitive functions (they have no body to convert)
    if (op.getPrimitive()) {
      rewriter.eraseOp(op);
      return success();
    }

    // Get function name
    StringRef funcName = op.getName();

    // Convert argument types
    SmallVector<Type> argTypes;
    for (auto typeAttr : op.getArgsTypes()) {
      Type aslType = mlir::cast<TypeAttr>(typeAttr).getValue();
      Type convertedType = getTypeConverter()->convertType(aslType);
      if (!convertedType)
        return rewriter.notifyMatchFailure(op, "failed to convert arg type");
      argTypes.push_back(convertedType);
    }

    // Convert return type
    SmallVector<Type> resultTypes;
    if (auto aslRetType = op.getReturnType()) {
      Type convertedRetType = getTypeConverter()->convertType(*aslRetType);
      if (!convertedRetType)
        return rewriter.notifyMatchFailure(op, "failed to convert return type");
      resultTypes.push_back(convertedRetType);
    }

    // Create function type
    auto funcType = FunctionType::get(context, argTypes, resultTypes);

    // Create func.func operation
    auto funcOp = rewriter.create<func::FuncOp>(loc, funcName, funcType);

    // Convert function body using inlineRegionBefore to preserve operations
    // The dialect conversion framework will then convert the operations in the
    // region
    Region &aslBody = op.getBody();
    if (!aslBody.empty()) {
      // Move the region to the new function
      rewriter.inlineRegionBefore(aslBody, funcOp.getBody(),
                                  funcOp.getBody().end());

      // Update block argument types
      // The type converter will handle block argument conversion
      if (failed(rewriter.convertRegionTypes(&funcOp.getBody(),
                                             *getTypeConverter()))) {
        return failure();
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Function call expression: asl.expr.call -> emitc.call_opaque or func.call
struct CallOpLowering : public OpConversionPattern<asl::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Convert result type
    Type convertedType = getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return failure();

    // Get function name
    StringRef funcName = op.getName();

    // Create call using emitc.call_opaque for now
    // This allows calling external C functions as well as converted functions
    auto callOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{convertedType}, funcName, adaptor.getArgs(), nullptr,
        nullptr);

    rewriter.replaceOp(op, callOp.getResult(0));
    return success();
  }
};

// Procedure call statement: asl.stmt.call -> emitc.call_opaque (void)
struct StmtCallOpLowering : public OpConversionPattern<asl::StmtCallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::StmtCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get procedure name
    StringRef procName = op.getName();

    // Create void call using emitc.call_opaque
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, procName,
                                         adaptor.getArgs(), nullptr, nullptr);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Type Conversion (ATC) Lowering Patterns
//===----------------------------------------------------------------------===//

// Generic ATC: just pass through the value with type conversion
struct AtcOpLowering : public OpConversionPattern<asl::AtcOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::AtcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // ATC is essentially a type assertion/conversion
    // For now, just pass through the value - the type converter handles
    // the actual type conversion
    rewriter.replaceOp(op, adaptor.getExpr());
    return success();
  }
};

// VarOp: variable reference - convert to use the value directly
struct VarOpLowering : public OpConversionPattern<asl::VarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::VarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For local variables (within functions), the variable reference
    // should have been replaced by the actual SSA value during function
    // body conversion. For now, just fail - this pattern should not
    // be reached for properly converted code.
    return rewriter.notifyMatchFailure(
        op, "VarOp should be resolved during function body conversion");
  }
};

// Comparison operations for integers (using GMP)
template <typename OpTy>
struct IntCompareOpLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  IntCompareOpLowering(const TypeConverter &converter, MLIRContext *context,
                       StringRef cmpOp)
      : OpConversionPattern<OpTy>(converter, context), compareOp(cmpOp) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Check if operands are integer types (GMP)
    Type lhsType = op.getLhs().getType();
    bool isIntCompare = llvm::isa<asl::IntType>(lhsType);

    Type convertedType =
        this->getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return failure();

    if (isIntCompare) {
      // Use mpz_cmp for comparison
      Value loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
      Value loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

      auto cmpResult = rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{rewriter.getI32Type()}, "mpz_cmp",
          ValueRange{loadedLhs, loadedRhs}, nullptr, nullptr);

      // Create comparison with 0
      auto zeroConstant = rewriter.create<emitc::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

      auto result = rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{convertedType}, compareOp,
          ValueRange{cmpResult.getResult(0), zeroConstant.getResult()}, nullptr,
          nullptr);

      rewriter.replaceOp(op, result.getResult(0));
    } else {
      // For non-GMP types (bitvectors, booleans), use direct comparison
      auto result = rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{convertedType}, compareOp,
          ValueRange{adaptor.getLhs(), adaptor.getRhs()}, nullptr, nullptr);

      rewriter.replaceOp(op, result.getResult(0));
    }

    return success();
  }

private:
  std::string compareOp;
};

// Convert asl.type_decl operation to EmitC typedef (for enums)
struct TypeDeclOpLowering : public OpConversionPattern<asl::TypeDeclOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::TypeDeclOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // This pattern just removes the type decl op
    // The actual typedef generation happens in the pass
    rewriter.eraseOp(op);
    return success();
  }
};

// Convert asl.expr.tuple operation to EmitC struct initialization
struct TupleOpLowering : public OpConversionPattern<asl::TupleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::TupleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Get the tuple type and convert it
    auto tupleType = llvm::dyn_cast<asl::TupleType>(op.getType());
    if (!tupleType)
      return failure();

    Type convertedType = getTypeConverter()->convertType(tupleType);
    if (!convertedType)
      return failure();

    // Get converted element operands
    auto elements = adaptor.getElements();
    if (elements.size() != tupleType.getTypes().size())
      return failure();

    // Create a variable to hold the tuple (emitc.variable requires LValueType)
    auto lvalueType = emitc::LValueType::get(convertedType);
    auto varOp = rewriter.create<emitc::VariableOp>(
        op.getLoc(), lvalueType,
        emitc::OpaqueAttr::get(rewriter.getContext(), ""));

    // For each element, initialize the corresponding field
    // For GMP types we need proper initialization, for other types direct
    // assignment
    for (size_t i = 0; i < elements.size(); ++i) {
      std::string fieldName = "item" + std::to_string(i);

      // Get element type - tupleType.getTypes() returns ArrayAttr
      auto typeAttr = mlir::cast<TypeAttr>(tupleType.getTypes()[i]);
      Type elementType = typeAttr.getValue();
      Type convertedElementType = getTypeConverter()->convertType(elementType);

      // Build field access using emitc.member: tuple.itemN
      auto fieldLvalueType = emitc::LValueType::get(convertedElementType);
      auto fieldAccessOp = rewriter.create<emitc::MemberOp>(
          op.getLoc(), fieldLvalueType, fieldName, varOp.getResult());

      // For GMP types (mpz_t, mpq_t), we need to call init and set functions
      // For GMP, we load to get pointer and pass that to GMP functions
      if (auto opaqueType =
              llvm::dyn_cast<emitc::OpaqueType>(convertedElementType)) {
        StringRef typeName = opaqueType.getValue();

        if (typeName == "mpz_t") {
          // Load to get value for GMP function
          auto loadedField = rewriter.create<emitc::LoadOp>(
              op.getLoc(), convertedElementType, fieldAccessOp.getResult());
          SmallVector<Value, 2> initSetArgs = {loadedField, elements[i]};
          rewriter.create<emitc::CallOpaqueOp>(op.getLoc(), TypeRange{},
                                               "mpz_init_set", initSetArgs,
                                               nullptr, nullptr);
        } else if (typeName == "mpq_t") {
          auto loadedField = rewriter.create<emitc::LoadOp>(
              op.getLoc(), convertedElementType, fieldAccessOp.getResult());
          SmallVector<Value, 1> initArgs = {loadedField};
          rewriter.create<emitc::CallOpaqueOp>(
              op.getLoc(), TypeRange{}, "mpq_init", initArgs, nullptr, nullptr);
          SmallVector<Value, 2> setArgs = {loadedField, elements[i]};
          rewriter.create<emitc::CallOpaqueOp>(
              op.getLoc(), TypeRange{}, "mpq_set", setArgs, nullptr, nullptr);
        } else {
          // For simple types: use emitc.assign
          rewriter.create<emitc::AssignOp>(op.getLoc(), fieldAccessOp,
                                           elements[i]);
        }
      } else {
        // For non-opaque simple types: use emitc.assign
        rewriter.create<emitc::AssignOp>(op.getLoc(), fieldAccessOp,
                                         elements[i]);
      }
    }

    rewriter.replaceOp(op, varOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Phase 6: Data Structure Access Lowering Patterns
//===----------------------------------------------------------------------===//

// Tuple element access: tuple.itemN
struct GetItemOpLowering : public OpConversionPattern<asl::GetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::GetItemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get the index
    int32_t index = op.getIndex();
    std::string fieldName = "item" + std::to_string(index);

    // Get converted result type
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // Build field access using emitc.member: tuple.itemN
    // The result is an lvalue that will be loaded by materialization if needed
    auto fieldLvalueType = emitc::LValueType::get(convertedResultType);
    auto fieldAccessOp = rewriter.create<emitc::MemberOp>(
        loc, fieldLvalueType, fieldName, adaptor.getTuple());

    // Load the field value
    auto loadedValue = rewriter.create<emitc::LoadOp>(
        loc, convertedResultType, fieldAccessOp.getResult());

    rewriter.replaceOp(op, loadedValue.getResult());
    return success();
  }
};

// Record construction: struct initialization
struct RecordOpLowering : public OpConversionPattern<asl::RecordOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::RecordOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get converted result type
    Type convertedType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // Create a variable to hold the record (emitc.variable requires LValueType)
    auto lvalueType = emitc::LValueType::get(convertedType);
    auto varOp = rewriter.create<emitc::VariableOp>(
        loc, lvalueType, emitc::OpaqueAttr::get(rewriter.getContext(), ""));

    // Get field names and values
    ArrayAttr fieldNames = op.getFieldNames();
    auto fieldValues = adaptor.getFieldValues();

    // For each field, initialize it
    for (size_t i = 0; i < fieldValues.size(); ++i) {
      StringRef fieldName =
          mlir::cast<StringAttr>(fieldNames[i]).getValue();

      // Get converted element type
      Type convertedElementType = fieldValues[i].getType();

      // Build field access using emitc.member: record.fieldName
      auto fieldLvalueType = emitc::LValueType::get(convertedElementType);
      auto fieldAccessOp = rewriter.create<emitc::MemberOp>(
          loc, fieldLvalueType, fieldName, varOp.getResult());

      // Handle GMP types vs simple types
      if (auto opaqueType =
              llvm::dyn_cast<emitc::OpaqueType>(convertedElementType)) {
        StringRef typeName = opaqueType.getValue();

        if (typeName == "mpz_t") {
          auto loadedField = rewriter.create<emitc::LoadOp>(
              loc, convertedElementType, fieldAccessOp.getResult());
          SmallVector<Value, 2> initSetArgs = {loadedField, fieldValues[i]};
          rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "mpz_init_set",
                                               initSetArgs, nullptr, nullptr);
        } else if (typeName == "mpq_t") {
          auto loadedField = rewriter.create<emitc::LoadOp>(
              loc, convertedElementType, fieldAccessOp.getResult());
          SmallVector<Value, 1> initArgs = {loadedField};
          rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "mpq_init",
                                               initArgs, nullptr, nullptr);
          SmallVector<Value, 2> setArgs = {loadedField, fieldValues[i]};
          rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "mpq_set",
                                               setArgs, nullptr, nullptr);
        } else {
          rewriter.create<emitc::AssignOp>(loc, fieldAccessOp, fieldValues[i]);
        }
      } else {
        // Simple types: use emitc.assign
        rewriter.create<emitc::AssignOp>(loc, fieldAccessOp, fieldValues[i]);
      }
    }

    rewriter.replaceOp(op, varOp.getResult());
    return success();
  }
};

// Record field access: record.fieldName
struct GetFieldOpLowering : public OpConversionPattern<asl::GetFieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::GetFieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    StringRef fieldName = op.getFieldName();

    // Get converted result type
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // Build field access using emitc.member: record.fieldName
    auto fieldLvalueType = emitc::LValueType::get(convertedResultType);
    auto fieldAccessOp = rewriter.create<emitc::MemberOp>(
        loc, fieldLvalueType, fieldName, adaptor.getRecord());

    // Load the field value
    auto loadedValue = rewriter.create<emitc::LoadOp>(
        loc, convertedResultType, fieldAccessOp.getResult());

    rewriter.replaceOp(op, loadedValue.getResult());
    return success();
  }
};

// Array element access: array[index]
struct GetArrayOpLowering : public OpConversionPattern<asl::GetArrayOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::GetArrayOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Get converted result type
    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!convertedResultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // For GMP index, we need to convert to native int first
    Value index = adaptor.getIndex();
    if (auto lvalueType = llvm::dyn_cast<emitc::LValueType>(index.getType())) {
      auto mpzType = emitc::OpaqueType::get(rewriter.getContext(), "mpz_t");
      index = rewriter.create<emitc::LoadOp>(loc, mpzType, index);
    }

    // Convert GMP integer index to native long
    auto longType = emitc::OpaqueType::get(rewriter.getContext(), "long");
    auto nativeIndex = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{longType}, "mpz_get_si", ValueRange{index}, nullptr,
        nullptr);

    // Build array access: array[index]
    auto subscriptOp = rewriter.create<emitc::SubscriptOp>(
        loc, convertedResultType, adaptor.getBase(), nativeIndex.getResult(0));

    rewriter.replaceOp(op, subscriptOp.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Phase 7: Slicing Operations
//===----------------------------------------------------------------------===//

// Helper to convert GMP integer to native long
static Value gmpToLong(ConversionPatternRewriter &rewriter, Location loc,
                       Value gmpValue) {
  MLIRContext *context = rewriter.getContext();

  // Load if lvalue
  if (auto lvalueType = llvm::dyn_cast<emitc::LValueType>(gmpValue.getType())) {
    auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
    gmpValue = rewriter.create<emitc::LoadOp>(loc, mpzType, gmpValue);
  }

  // Convert to long
  auto longType = emitc::OpaqueType::get(context, "long");
  auto result = rewriter.create<emitc::CallOpaqueOp>(
      loc, TypeRange{longType}, "mpz_get_si", ValueRange{gmpValue}, nullptr,
      nullptr);
  return result.getResult(0);
}

// Helper to create a slice descriptor struct
static Value createSliceDescriptor(ConversionPatternRewriter &rewriter,
                                   Location loc, Value start, Value length) {
  MLIRContext *context = rewriter.getContext();
  auto sliceType =
      emitc::OpaqueType::get(context, "struct { long start; long length; }");
  auto lvalueType = emitc::LValueType::get(sliceType);

  // Create variable for the slice descriptor
  auto varOp = rewriter.create<emitc::VariableOp>(
      loc, lvalueType, emitc::OpaqueAttr::get(context, ""));

  // Set start field
  auto longType = emitc::OpaqueType::get(context, "long");
  auto startLvalue = emitc::LValueType::get(longType);
  auto startField =
      rewriter.create<emitc::MemberOp>(loc, startLvalue, "start", varOp);
  rewriter.create<emitc::AssignOp>(loc, startField, start);

  // Set length field
  auto lengthField =
      rewriter.create<emitc::MemberOp>(loc, startLvalue, "length", varOp);
  rewriter.create<emitc::AssignOp>(loc, lengthField, length);

  return varOp.getResult();
}

// SliceSingle: creates slice of length 1 at position i
struct SliceSingleOpLowering : public OpConversionPattern<asl::SliceSingleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::SliceSingleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Convert index from GMP to long
    Value start = gmpToLong(rewriter, loc, adaptor.getIndex());

    // Length is 1
    auto longType = emitc::OpaqueType::get(context, "long");
    auto lengthOne = rewriter.create<emitc::ConstantOp>(
        loc, longType, emitc::OpaqueAttr::get(context, "1"));

    // Create slice descriptor
    Value sliceDesc = createSliceDescriptor(rewriter, loc, start, lengthOne);

    rewriter.replaceOp(op, sliceDesc);
    return success();
  }
};

// SliceRange: creates slice from position j to i-1 (inclusive)
struct SliceRangeOpLowering : public OpConversionPattern<asl::SliceRangeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::SliceRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Convert start (j) and end (i) from GMP to long
    Value startVal = gmpToLong(rewriter, loc, adaptor.getStart());
    Value endVal = gmpToLong(rewriter, loc, adaptor.getEnd());

    // Length = end - start (i - j gives length from j to i-1)
    auto longType = emitc::OpaqueType::get(context, "long");
    auto lengthVal = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{longType}, "-", ValueRange{endVal, startVal}, nullptr,
        nullptr);

    // Create slice descriptor with start=j, length=i-j
    Value sliceDesc =
        createSliceDescriptor(rewriter, loc, startVal, lengthVal.getResult(0));

    rewriter.replaceOp(op, sliceDesc);
    return success();
  }
};

// SliceLength: creates slice of length n starting at position i
struct SliceLengthOpLowering : public OpConversionPattern<asl::SliceLengthOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::SliceLengthOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // Convert start and length from GMP to long
    Value startVal = gmpToLong(rewriter, loc, adaptor.getStart());
    Value lengthVal = gmpToLong(rewriter, loc, adaptor.getLength());

    // Create slice descriptor
    Value sliceDesc = createSliceDescriptor(rewriter, loc, startVal, lengthVal);

    rewriter.replaceOp(op, sliceDesc);
    return success();
  }
};

// SliceStar: creates slice at position factor*length with given length
struct SliceStarOpLowering : public OpConversionPattern<asl::SliceStarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::SliceStarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Convert factor and length from GMP to long
    Value factorVal = gmpToLong(rewriter, loc, adaptor.getFactor());
    Value lengthVal = gmpToLong(rewriter, loc, adaptor.getLength());

    // Compute start = factor * length
    auto longType = emitc::OpaqueType::get(context, "long");
    auto startVal = rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{longType}, "*", ValueRange{factorVal, lengthVal},
        nullptr, nullptr);

    // Create slice descriptor
    Value sliceDesc =
        createSliceDescriptor(rewriter, loc, startVal.getResult(0), lengthVal);

    rewriter.replaceOp(op, sliceDesc);
    return success();
  }
};

// SliceOp: applies slices to extract bits from a bitvector
struct SliceOpLowering : public OpConversionPattern<asl::SliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(asl::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Get converted result type
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // Get the base bitvector value
    Value base = adaptor.getBase();

    // Handle lvalue base
    if (auto lvalueType = llvm::dyn_cast<emitc::LValueType>(base.getType())) {
      base = rewriter.create<emitc::LoadOp>(loc, lvalueType.getValueType(),
                                            base);
    }

    // Get the slices
    auto slices = adaptor.getSlices();

    if (slices.empty()) {
      // No slices - just return the base value
      rewriter.replaceOp(op, base);
      return success();
    }

    // For a single slice, extract bits: (base >> start) & ((1ULL << length) - 1)
    // For multiple slices, concatenate them
    auto longType = emitc::OpaqueType::get(context, "long");
    auto sliceType =
        emitc::OpaqueType::get(context, "struct { long start; long length; }");

    Value result;

    for (size_t i = 0; i < slices.size(); ++i) {
      Value slice = slices[i];

      // Load slice if lvalue
      if (auto lvalueType = llvm::dyn_cast<emitc::LValueType>(slice.getType())) {
        slice = rewriter.create<emitc::LoadOp>(loc, sliceType, slice);
      }

      // Extract start and length from slice descriptor
      auto sliceLvalue = emitc::LValueType::get(sliceType);

      // We need to create a variable to hold the slice for member access
      auto tempVar = rewriter.create<emitc::VariableOp>(
          loc, sliceLvalue, emitc::OpaqueAttr::get(context, ""));
      rewriter.create<emitc::AssignOp>(loc, tempVar, slice);

      auto startLvalue = emitc::LValueType::get(longType);
      auto startField =
          rewriter.create<emitc::MemberOp>(loc, startLvalue, "start", tempVar);
      auto start =
          rewriter.create<emitc::LoadOp>(loc, longType, startField);

      auto lengthField =
          rewriter.create<emitc::MemberOp>(loc, startLvalue, "length", tempVar);
      auto length =
          rewriter.create<emitc::LoadOp>(loc, longType, lengthField);

      // Extract bits: (base >> start) & ((1ULL << length) - 1)
      // First: shifted = base >> start
      auto shifted = rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{resultType}, ">>", ValueRange{base, start}, nullptr,
          nullptr);

      // Create mask: (1ULL << length) - 1
      auto oneULL = rewriter.create<emitc::ConstantOp>(
          loc, resultType, emitc::OpaqueAttr::get(context, "1ULL"));
      auto maskShifted = rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{resultType}, "<<", ValueRange{oneULL, length}, nullptr,
          nullptr);
      auto oneForMask = rewriter.create<emitc::ConstantOp>(
          loc, resultType, emitc::OpaqueAttr::get(context, "1"));
      auto mask = rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{resultType}, "-",
          ValueRange{maskShifted.getResult(0), oneForMask}, nullptr, nullptr);

      // Apply mask: shifted & mask
      auto extracted = rewriter.create<emitc::CallOpaqueOp>(
          loc, TypeRange{resultType}, "&",
          ValueRange{shifted.getResult(0), mask.getResult(0)}, nullptr,
          nullptr);

      if (i == 0) {
        result = extracted.getResult(0);
      } else {
        // For multiple slices, shift and OR to concatenate
        // result = (result << length) | extracted
        auto shiftedResult = rewriter.create<emitc::CallOpaqueOp>(
            loc, TypeRange{resultType}, "<<", ValueRange{result, length},
            nullptr, nullptr);
        result = rewriter.create<emitc::CallOpaqueOp>(
                     loc, TypeRange{resultType}, "|",
                     ValueRange{shiftedResult.getResult(0),
                                extracted.getResult(0)},
                     nullptr, nullptr)
                     .getResult(0);
      }
    }

    rewriter.replaceOp(op, result);
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

    // Add conversion patterns for literals and declarations
    patterns.add<ConstantInitGlobalStorageDeclOpLowering,
                 LiteralStringOpLowering, LiteralBitvectorOpLowering,
                 LiteralLabelOpLowering, LiteralIntOpLowering,
                 LiteralBoolOpLowering, LiteralRealOpLowering,
                 TypeDeclOpLowering, TupleOpLowering>(typeConverter, context);

    // Add data structure access patterns (Phase 6)
    patterns.add<GetItemOpLowering, RecordOpLowering, GetFieldOpLowering,
                 GetArrayOpLowering>(typeConverter, context);

    // Add integer binary operation patterns
    patterns.add<IntBinaryOpLowering<asl::BinopIntAddOp>>(typeConverter, context,
                                                          "mpz_add");
    patterns.add<IntBinaryOpLowering<asl::BinopIntSubOp>>(typeConverter, context,
                                                          "mpz_sub");
    patterns.add<IntBinaryOpLowering<asl::BinopIntMulOp>>(typeConverter, context,
                                                          "mpz_mul");
    patterns.add<BinopDivOpLowering, BinopDivrmOpLowering, BinopModOpLowering,
                 BinopPowOpLowering, BinopShlOpLowering, BinopShrOpLowering>(
        typeConverter, context);

    // Add real binary operation patterns
    patterns.add<RealBinaryOpLowering<asl::BinopRealAddOp>>(typeConverter,
                                                            context, "mpq_add");
    patterns.add<RealBinaryOpLowering<asl::BinopRealSubOp>>(typeConverter,
                                                            context, "mpq_sub");
    patterns.add<RealBinaryOpLowering<asl::BinopRealMulOp>>(typeConverter,
                                                            context, "mpq_mul");
    patterns.add<BinopRdivOpLowering>(typeConverter, context);

    // Add bitvector binary operation patterns
    patterns.add<BitsBinaryOpLowering<asl::BinopBitsAddOp>>(typeConverter,
                                                            context, "+");
    patterns.add<BitsBinaryOpLowering<asl::BinopBitsSubOp>>(typeConverter,
                                                            context, "-");
    patterns.add<BitsBinaryOpLowering<asl::BinopBitsMulOp>>(typeConverter,
                                                            context, "*");
    patterns.add<BitsBinaryOpLowering<asl::BinopAndOp>>(typeConverter, context,
                                                        "&");
    patterns.add<BitsBinaryOpLowering<asl::BinopOrOp>>(typeConverter, context,
                                                       "|");
    patterns.add<BitsBinaryOpLowering<asl::BinopXorOp>>(typeConverter, context,
                                                        "^");
    patterns.add<BinopConcatOpLowering>(typeConverter, context);

    // Add boolean binary operation patterns
    patterns.add<BinopBandOpLowering, BinopBorOpLowering, BinopBeqOpLowering,
                 BinopImplOpLowering>(typeConverter, context);

    // Add comparison operation patterns
    patterns.add<IntCompareOpLowering<asl::BinopEqOp>>(typeConverter, context,
                                                       "==");
    patterns.add<IntCompareOpLowering<asl::BinopNeqOp>>(typeConverter, context,
                                                        "!=");
    patterns.add<IntCompareOpLowering<asl::BinopLtOp>>(typeConverter, context,
                                                       "<");
    patterns.add<IntCompareOpLowering<asl::BinopLeqOp>>(typeConverter, context,
                                                        "<=");
    patterns.add<IntCompareOpLowering<asl::BinopGtOp>>(typeConverter, context,
                                                       ">");
    patterns.add<IntCompareOpLowering<asl::BinopGeqOp>>(typeConverter, context,
                                                        ">=");

    // Add unary operation patterns (Phase 3)
    patterns.add<UnopBnotOpLowering, UnopNegIntOpLowering, UnopNegRealOpLowering,
                 UnopNotOpLowering>(typeConverter, context);

    // Add control flow patterns (Phase 4)
    patterns.add<CondOpLowering, StmtPassOpLowering, StmtReturnOpLowering,
                 StmtSeqOpLowering, StmtCondOpLowering, StmtAssertOpLowering,
                 StmtUnreachableOpLowering, StmtForOpLowering,
                 StmtWhileOpLowering, StmtRepeatOpLowering>(typeConverter,
                                                            context);

    // Add function declaration patterns (Phase 5)
    patterns.add<FuncDeclOpLowering, CallOpLowering, StmtCallOpLowering>(
        typeConverter, context);

    // Add type conversion patterns (ATC)
    patterns.add<AtcOpLowering>(typeConverter, context);

    // Add slicing patterns (Phase 7)
    patterns.add<SliceSingleOpLowering, SliceRangeOpLowering,
                 SliceLengthOpLowering, SliceStarOpLowering, SliceOpLowering>(
        typeConverter, context);

    // Collect type declarations before conversion for typedef generation
    SmallVector<asl::TypeDeclOp> typeDecls;
    module.walk([&](asl::TypeDeclOp op) { typeDecls.push_back(op); });

    // Collect global variables before conversion for context struct generation
    SmallVector<asl::ConstantInitGlobalStorageDeclOp> globalVars;
    module.walk([&](asl::ConstantInitGlobalStorageDeclOp op) {
      globalVars.push_back(op);
    });

    // Generate enum typedefs and context structure
    if (!typeDecls.empty() || !globalVars.empty()) {
      generateGlobalContext(module, typeDecls, globalVars, typeConverter);
    }

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

private:
  // Generate enum typedefs, context structure and init/free functions
  void generateGlobalContext(
      ModuleOp module, ArrayRef<asl::TypeDeclOp> typeDecls,
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

    // 1. Generate typedefs for type declarations (enums and tuples)
    for (auto typeDecl : typeDecls) {
      generateEnumTypedef(builder, loc, typeDecl);
      generateTupleTypedef(builder, loc, typeDecl, typeConverter);
    }

    // 2. Generate typedef struct definition using emitc.verbatim
    generateContextStruct(builder, loc, contextTypeName, globalVars,
                          typeConverter);

    // 2. Generate per-variable inline init functions
    for (auto globalVar : globalVars) {
      generatePerVariableInit(builder, loc, moduleName, contextTypeName,
                              globalVar, typeConverter, module);
    }

    // 3. Generate main init function
    generateMainInitFunction(builder, loc, moduleName, contextTypeName,
                             globalVars);

    // 4. Generate free function
    generateFreeFunction(builder, loc, moduleName, contextTypeName, globalVars,
                         typeConverter);
  }

  // Generate enum typedef from type declaration
  void generateEnumTypedef(OpBuilder &builder, Location loc,
                           asl::TypeDeclOp typeDecl) {
    // Only generate typedef for enum types
    Type declType = typeDecl.getType();
    auto enumType = llvm::dyn_cast<asl::EnumType>(declType);
    if (!enumType)
      return;

    // Get the type name with asl_ prefix to avoid collisions
    std::string typeName =
        "asl_" + sanitizeIdentifier(typeDecl.getIdentifier());

    // Generate enum typedef
    std::string enumDef = "typedef enum " + typeName + " {\n";

    // Add enum constants
    auto labels = enumType.getStringLabels();
    for (size_t i = 0; i < labels.size(); ++i) {
      std::string labelName = sanitizeIdentifier(labels[i].str());
      enumDef += "  " + typeName + "_" + labelName + " = " + std::to_string(i);
      if (i < labels.size() - 1) {
        enumDef += ",\n";
      } else {
        enumDef += "\n";
      }
    }

    enumDef += "} " + typeName + ";";

    // Create verbatim op for enum typedef
    builder.create<emitc::VerbatimOp>(loc, enumDef);
  }

  // Generate tuple typedef from type declaration
  void generateTupleTypedef(OpBuilder &builder, Location loc,
                            asl::TypeDeclOp typeDecl,
                            ASLToEmitCTypeConverter &typeConverter) {
    // Only generate typedef for tuple types
    Type declType = typeDecl.getType();
    auto tupleType = llvm::dyn_cast<asl::TupleType>(declType);
    if (!tupleType)
      return;

    // Get the type name with asl_ prefix to avoid collisions
    std::string typeName =
        "asl_" + sanitizeIdentifier(typeDecl.getIdentifier());

    // Generate struct typedef
    std::string structDef = "typedef struct " + typeName + " {\n";

    // Add struct fields
    auto types = tupleType.getTypes();
    for (size_t i = 0; i < types.size(); ++i) {
      auto typeAttr = mlir::cast<TypeAttr>(types[i]);
      Type elemType = typeAttr.getValue();

      // Convert the element type
      Type convertedElemType = typeConverter.convertType(elemType);
      if (!convertedElemType) {
        mlir::emitError(loc) << "failed to convert tuple element type " << i;
        return;
      }

      // Get the C type string
      std::string cType;
      if (auto opaqueType =
              llvm::dyn_cast<emitc::OpaqueType>(convertedElemType)) {
        cType = opaqueType.getValue().str();
      } else {
        mlir::emitError(loc)
            << "converted tuple element type " << i << " is not an opaque type";
        return;
      }

      // Add field: "  type itemN;\n"
      structDef += "  " + cType + " item" + std::to_string(i) + ";\n";
    }

    structDef += "} " + typeName + ";";

    // Create verbatim op for struct typedef
    builder.create<emitc::VerbatimOp>(loc, structDef);
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
    // Add string.h for string operations (strcmp, strlen, etc.)
    builder.create<emitc::VerbatimOp>(loc, "#include <string.h>");
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

  // Helper function to recursively generate initialization code for a field
  void generateFieldInit(std::string &initFunc, const std::string &fieldPath,
                         Type aslType, const std::string &value,
                         ASLToEmitCTypeConverter &typeConverter,
                         ModuleOp module) {
    // Resolve named types first - keep resolving until we get to the actual
    // type
    Type resolvedType = aslType;
    while (auto namedType = llvm::dyn_cast<asl::NamedType>(resolvedType)) {
      if (namedType.getResolvedType()) {
        resolvedType = namedType.getResolvedType().getValue();
      } else {
        // Named type without resolved type - look up the type declaration in
        // the module
        StringRef typeName = namedType.getName();
        bool found = false;

        // Search for type declaration in the module
        module.walk([&](asl::TypeDeclOp typeDecl) {
          if (typeDecl.getIdentifier() == typeName) {
            // Found the type declaration, get its type
            TypeAttr typeAttr = typeDecl.getTypeAttr();
            resolvedType = typeAttr.getValue();
            found = true;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

        if (!found) {
          // Cannot find type declaration - generate error
          initFunc += "  // ERROR: Cannot find type declaration for '" +
                      typeName.str() + "'\n";
          initFunc += "  " + fieldPath + " = " + value + "; // INVALID\n";
          return;
        }
      }
    }

    // Check if this is a tuple type - handle recursively
    if (auto tupleType = llvm::dyn_cast<asl::TupleType>(resolvedType)) {
      ArrayAttr types = tupleType.getTypes();

      // Parse tuple value: "(val1, val2, ...)"
      SmallVector<std::string> tupleValues;
      if (value.size() >= 2 && value.front() == '(' && value.back() == ')') {
        std::string content = value.substr(1, value.size() - 2);

        int parenDepth = 0;
        size_t start = 0;
        for (size_t i = 0; i <= content.size(); ++i) {
          if (i < content.size()) {
            if (content[i] == '(')
              parenDepth++;
            else if (content[i] == ')')
              parenDepth--;
          }

          if ((i == content.size()) || (content[i] == ',' && parenDepth == 0)) {
            std::string val = content.substr(start, i - start);
            // Trim whitespace
            size_t first = val.find_first_not_of(" \t\n\r");
            size_t last = val.find_last_not_of(" \t\n\r");
            if (first != std::string::npos) {
              val = val.substr(first, last - first + 1);
            } else {
              val = "";
            }
            tupleValues.push_back(val);
            start = i + 1;
          }
        }
      }

      // Recursively initialize each field
      for (size_t i = 0; i < types.size(); ++i) {
        auto typeAttr = mlir::cast<TypeAttr>(types[i]);
        Type elemType = typeAttr.getValue();
        std::string fieldName = "item" + std::to_string(i);
        std::string elemValue = (i < tupleValues.size()) ? tupleValues[i] : "";

        // Recursive call for this field
        generateFieldInit(initFunc, fieldPath + "." + fieldName, elemType,
                          elemValue, typeConverter, module);
      }
      return;
    }

    // For non-tuple types, generate the appropriate initialization
    Type convertedType = typeConverter.convertType(resolvedType);

    if (auto opaqueType = llvm::dyn_cast<emitc::OpaqueType>(convertedType)) {
      StringRef typeName = opaqueType.getValue();

      if (typeName == "mpz_t") {
        // Initialize GMP integer
        std::string val = value.empty() ? "0" : value;
        initFunc +=
            "  mpz_init_set_str(" + fieldPath + ", \"" + val + "\", 10);\n";
      } else if (typeName == "mpq_t") {
        // Initialize GMP rational
        std::string val = value.empty() ? "0" : value;
        initFunc += "  mpq_init(" + fieldPath + ");\n";
        initFunc += "  mpq_set_str(" + fieldPath + ", \"" + val + "\", 10);\n";
        initFunc += "  mpq_canonicalize(" + fieldPath + ");\n";
      } else if (typeName == "bool") {
        // Initialize boolean
        std::string boolValue = "false";
        if (value == "true" || value == "TRUE" || value == "1") {
          boolValue = "true";
        }
        initFunc += "  " + fieldPath + " = " + boolValue + ";\n";
      } else if (typeName == "const char*") {
        // Initialize string
        if (value.empty()) {
          initFunc += "  " + fieldPath + " = \"\";\n";
        } else {
          std::string strValue = value;
          if (strValue.front() != '"') {
            strValue = "\"" + strValue + "\"";
          }
          initFunc += "  " + fieldPath + " = " + strValue + ";\n";
        }
      } else if (typeName.starts_with("uint") || typeName.starts_with("int")) {
        // Initialize integer types
        std::string val = value.empty() ? "0" : value;
        initFunc += "  " + fieldPath + " = " + val + ";\n";
      } else {
        // For other types (enums, etc.)
        std::string val = value.empty() ? "0" : value;
        initFunc += "  " + fieldPath + " = " + val + ";\n";
      }
    } else if (auto intType = llvm::dyn_cast<IntegerType>(convertedType)) {
      // Handle MLIR integer types (i1, i8, i16, i32, i64)
      if (intType.getWidth() == 1) {
        // i1 is bool
        std::string boolValue = "false";
        if (value == "true" || value == "TRUE" || value == "1") {
          boolValue = "true";
        }
        initFunc += "  " + fieldPath + " = " + boolValue + ";\n";
      } else {
        std::string val = value.empty() ? "0" : value;
        initFunc += "  " + fieldPath + " = " + val + ";\n";
      }
    } else {
      // Fallback
      std::string val = value.empty() ? "0" : value;
      initFunc += "  " + fieldPath + " = " + val + ";\n";
    }
  }

  // Generate per-variable inline init function
  void generatePerVariableInit(OpBuilder &builder, Location loc,
                               StringRef moduleName, StringRef contextTypeName,
                               asl::ConstantInitGlobalStorageDeclOp globalVar,
                               ASLToEmitCTypeConverter &typeConverter,
                               ModuleOp module) {
    std::string varName = sanitizeIdentifier(globalVar.getName());
    std::string initFuncName = moduleName.str() + "_init_" + varName;

    // Get type information
    Type varType = globalVar.getType();
    Type convertedType = typeConverter.convertType(varType);

    // Check if this is a GMP type, string type, enum type, or tuple type
    bool isGMPInt = false;
    bool isGMPRational = false;
    bool isString = false;
    bool isEnum = false;
    bool isTuple = false;
    std::string enumTypeName;

    // First, resolve named types to check if they're tuples or enums
    Type resolvedVarType = varType;
    if (auto namedType = llvm::dyn_cast<asl::NamedType>(varType)) {
      if (namedType.getResolvedType()) {
        resolvedVarType = namedType.getResolvedType().getValue();
      }
    }
    isTuple = llvm::isa<asl::TupleType>(resolvedVarType);
    isEnum = llvm::isa<asl::EnumType>(resolvedVarType);

    if (auto opaqueType = llvm::dyn_cast<emitc::OpaqueType>(convertedType)) {
      StringRef typeName = opaqueType.getValue();
      isGMPInt = (typeName == "mpz_t");
      isGMPRational = (typeName == "mpq_t");
      isString = (typeName == "const char*");
      // Also check if converted type is an anonymous struct (for tuples without
      // names). But exclude large bitvector structs which have a words array.
      if (!isTuple) {
        isTuple = typeName.starts_with("struct {") &&
                  !typeName.contains("words[");
      }
      // Store the enum type name for generating the enum constant
      if (isEnum) {
        enumTypeName = typeName.str();
      }
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
    } else if (isString) {
      // For string types, assign the string literal directly
      // The literal value should already be a properly quoted string
      std::string stringValue = literal.str();

      // Ensure the string is properly quoted
      if (stringValue.empty() || stringValue.front() != '"') {
        // If not quoted, add quotes
        stringValue = "\"" + stringValue + "\"";
      }

      // Assign the string literal (stored in read-only data section)
      initFunc += "  ctx->" + varName + " = " + stringValue + ";\n";
    } else if (isEnum) {
      // For enum types, initialize with the enum constant
      // The literal should be the label name (e.g., "OK", "ERROR")
      std::string labelName = sanitizeIdentifier(literal.str());
      std::string enumConstant = enumTypeName + "_" + labelName;
      initFunc += "  ctx->" + varName + " = " + enumConstant + ";\n";
    } else if (isTuple) {
      // For tuple types, use the recursive helper function
      std::string fieldPath = "ctx->" + varName;
      generateFieldInit(initFunc, fieldPath, varType, literal.str(),
                        typeConverter, module);
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
        // Note: const char* (strings) don't need cleanup as they point to
        // string literals in read-only data sections
        // TODO: Add cleanup for dynamically allocated strings if needed
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
