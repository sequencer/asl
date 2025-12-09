//===- ConvertGMPToEmitC.cpp - Convert GMP to EmitC dialect ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ConvertGMPToEmitC pass, which lowers GMP dialect
// operations to EmitC dialect operations for C code generation using the
// GMP library.
//
//===----------------------------------------------------------------------===//

#include "GMP/GMPAttributes.h"
#include "GMP/GMPDialect.h"
#include "GMP/GMPOps.h"
#include "GMP/GMPPasses.h"
#include "GMP/GMPTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"

namespace mlir {
namespace gmp {

#define GEN_PASS_DECL_CONVERTGMPTOEMITC
#define GEN_PASS_DEF_CONVERTGMPTOEMITC
#include "GMP/GMPPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

class GMPToEmitCTypeConverter : public TypeConverter {
public:
  GMPToEmitCTypeConverter(MLIRContext *ctx) : context(ctx) {
    // Pass through already legal types
    addConversion([](Type type) { return type; });

    // Convert GMP Z type to lvalue<mpz_t>
    addConversion([this](ZType type) -> std::optional<Type> {
      auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
      return emitc::LValueType::get(mpzType);
    });

    // Convert GMP Q type to lvalue<mpq_t>
    addConversion([this](QType type) -> std::optional<Type> {
      auto mpqType = emitc::OpaqueType::get(context, "mpq_t");
      return emitc::LValueType::get(mpqType);
    });
  }

private:
  MLIRContext *context;
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Create a temporary mpz_t variable (returns lvalue type)
static Value createTempMpzVar(ConversionPatternRewriter &rewriter, Location loc,
                              MLIRContext *context) {
  auto mpzType = emitc::OpaqueType::get(context, "mpz_t");
  auto lvalueType = emitc::LValueType::get(mpzType);
  // Use empty initializer - GMP types will be initialized with mpz_init
  return rewriter.create<emitc::VariableOp>(
      loc, lvalueType, emitc::OpaqueAttr::get(context, ""));
}

// Create a temporary mpq_t variable (returns lvalue type)
static Value createTempMpqVar(ConversionPatternRewriter &rewriter, Location loc,
                              MLIRContext *context) {
  auto mpqType = emitc::OpaqueType::get(context, "mpq_t");
  auto lvalueType = emitc::LValueType::get(mpqType);
  // Use empty initializer - GMP types will be initialized with mpq_init
  return rewriter.create<emitc::VariableOp>(
      loc, lvalueType, emitc::OpaqueAttr::get(context, ""));
}

// Load a value from an lvalue (for passing to GMP functions)
// GMP types are arrays that decay to pointers, so this load gives us the value
// that can be passed to GMP functions
static Value loadGmpValue(ConversionPatternRewriter &rewriter, Location loc,
                          Value lvalue) {
  // If it's already not an lvalue, return as-is
  auto lvalueType = dyn_cast<emitc::LValueType>(lvalue.getType());
  if (!lvalueType)
    return lvalue;
  return rewriter.create<emitc::LoadOp>(loc, lvalueType.getValueType(), lvalue);
}

// Create an mpz_init call
static void createMpzInit(ConversionPatternRewriter &rewriter, Location loc,
                          Value var) {
  Value loadedVar = loadGmpValue(rewriter, loc, var);
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "mpz_init",
                                       ValueRange{loadedVar}, nullptr, nullptr);
}

// Create an mpq_init call
static void createMpqInit(ConversionPatternRewriter &rewriter, Location loc,
                          Value var) {
  Value loadedVar = loadGmpValue(rewriter, loc, var);
  rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "mpq_init",
                                       ValueRange{loadedVar}, nullptr, nullptr);
}

//===----------------------------------------------------------------------===//
// Z Module Conversion Patterns
//===----------------------------------------------------------------------===//

// Convert gmp.z.constant to mpz_init_set_str call
struct ZConstantOpLowering : public OpConversionPattern<ZConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Create temporary mpz_t variable
    auto resultVar = createTempMpzVar(rewriter, loc, context);
    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);

    // Get the string value from the ZAttr
    auto zAttr = op.getValue();
    std::string valueStr = zAttr.getValue();

    // Create mpz_init_set_str call
    auto strAttr = rewriter.create<emitc::ConstantOp>(
        loc, emitc::OpaqueType::get(context, "const char*"),
        emitc::OpaqueAttr::get(context, "\"" + valueStr + "\""));
    auto baseAttr = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(10));

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_init_set_str",
        ValueRange{loadedResult, strAttr.getResult(), baseAttr.getResult()},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Convert gmp.z.from_int to mpz_init_set_si/ui call
struct ZFromIntOpLowering : public OpConversionPattern<ZFromIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZFromIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Create temporary mpz_t variable
    auto resultVar = createTempMpzVar(rewriter, loc, context);
    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);

    // Check input type width to determine if we use signed or unsigned
    auto inputType = cast<IntegerType>(op.getInput().getType());
    bool isSigned = inputType.isSigned() || inputType.isSignless();

    // Create mpz_init_set_si or mpz_init_set_ui call
    std::string funcName = isSigned ? "mpz_init_set_si" : "mpz_init_set_ui";
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, funcName,
        ValueRange{loadedResult, adaptor.getInput()}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Generic pattern for binary Z operations (add, sub, mul)
template <typename OpTy>
struct ZBinaryOpLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  ZBinaryOpLowering(TypeConverter &converter, MLIRContext *context,
                    StringRef gmpFunc)
      : OpConversionPattern<OpTy>(converter, context), gmpFuncName(gmpFunc) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Create temporary mpz_t variable for result
    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    // Load all values for GMP function call
    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

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

// Generic pattern for unary Z operations (neg, abs)
template <typename OpTy>
struct ZUnaryOpLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  ZUnaryOpLowering(TypeConverter &converter, MLIRContext *context,
                   StringRef gmpFunc)
      : OpConversionPattern<OpTy>(converter, context), gmpFuncName(gmpFunc) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Create temporary mpz_t variable for result
    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    // Load all values for GMP function call
    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // Call the GMP function: mpz_func(result, operand)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, gmpFuncName, ValueRange{loadedResult, loadedOperand},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }

private:
  std::string gmpFuncName;
};

// Z division operations - special handling for different division modes
struct ZDivOpLowering : public OpConversionPattern<ZDivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Use truncated division: mpz_tdiv_q
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_tdiv_q",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZRemOpLowering : public OpConversionPattern<ZRemOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZRemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Use truncated remainder: mpz_tdiv_r
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_tdiv_r",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZFDivOpLowering : public OpConversionPattern<ZFDivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZFDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Use floor division: mpz_fdiv_q
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_fdiv_q",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZFRemOpLowering : public OpConversionPattern<ZFRemOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZFRemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Use floor remainder: mpz_fdiv_r
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_fdiv_r",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZCDivOpLowering : public OpConversionPattern<ZCDivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZCDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Use ceiling division: mpz_cdiv_q
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_cdiv_q",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZCRemOpLowering : public OpConversionPattern<ZCRemOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZCRemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Use ceiling remainder: mpz_cdiv_r
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_cdiv_r",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Euclidean division (same as floor division for positive divisor)
struct ZEDivOpLowering : public OpConversionPattern<ZEDivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZEDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Euclidean division uses floor division
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_fdiv_q",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZERemOpLowering : public OpConversionPattern<ZERemOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZERemOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Euclidean remainder uses floor remainder
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_fdiv_r",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZDivExactOpLowering : public OpConversionPattern<ZDivExactOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZDivExactOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Use exact division: mpz_divexact
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_divexact",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Z comparison operations
struct ZCompareOpLowering : public OpConversionPattern<ZCompareOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZCompareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Call mpz_cmp which returns int (-1, 0, or 1)
    auto cmpOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, rewriter.getI32Type(), "mpz_cmp", ValueRange{loadedLhs, loadedRhs},
        nullptr, nullptr);

    rewriter.replaceOp(op, cmpOp.getResult(0));
    return success();
  }
};

// Z relational comparison lowering - generic template
template <typename OpTy>
struct ZRelationalOpLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  ZRelationalOpLowering(TypeConverter &converter, MLIRContext *context,
                        StringRef cmpOp)
      : OpConversionPattern<OpTy>(converter, context), compOp(cmpOp) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // Call mpz_cmp
    auto cmpOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, rewriter.getI32Type(), "mpz_cmp", ValueRange{loadedLhs, loadedRhs},
        nullptr, nullptr);

    // Create constant 0 for comparison
    auto zero = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    // Compare result with 0 using the appropriate operator
    auto result = rewriter.create<emitc::CmpOp>(
        loc, rewriter.getI1Type(),
        emitc::CmpPredicateAttr::get(context, getCmpPredicate()),
        cmpOp.getResult(0), zero.getResult());

    rewriter.replaceOp(op, result.getResult());
    return success();
  }

private:
  std::string compOp;

  emitc::CmpPredicate getCmpPredicate() const {
    if (compOp == "eq")
      return emitc::CmpPredicate::eq;
    if (compOp == "lt")
      return emitc::CmpPredicate::lt;
    if (compOp == "le")
      return emitc::CmpPredicate::le;
    if (compOp == "gt")
      return emitc::CmpPredicate::gt;
    if (compOp == "ge")
      return emitc::CmpPredicate::ge;
    return emitc::CmpPredicate::eq;
  }
};

// Z bitwise operations
struct ZLogAndOpLowering : public OpConversionPattern<ZLogAndOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZLogAndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_and",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZLogOrOpLowering : public OpConversionPattern<ZLogOrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZLogOrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_ior",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZLogXorOpLowering : public OpConversionPattern<ZLogXorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZLogXorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_xor",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZLogNotOpLowering : public OpConversionPattern<ZLogNotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZLogNotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_com", ValueRange{loadedResult, loadedOperand},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Z shift operations
struct ZShiftLeftOpLowering : public OpConversionPattern<ZShiftLeftOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZShiftLeftOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // mpz_mul_2exp(result, operand, count)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_mul_2exp",
        ValueRange{loadedResult, loadedOperand, adaptor.getCount()}, nullptr,
        nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZShiftRightOpLowering : public OpConversionPattern<ZShiftRightOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZShiftRightOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // mpz_fdiv_q_2exp for floor division right shift (preserves sign correctly)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_fdiv_q_2exp",
        ValueRange{loadedResult, loadedOperand, adaptor.getCount()}, nullptr,
        nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZShiftRightTruncOpLowering
    : public OpConversionPattern<ZShiftRightTruncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZShiftRightTruncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // mpz_tdiv_q_2exp for truncated division right shift
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_tdiv_q_2exp",
        ValueRange{loadedResult, loadedOperand, adaptor.getCount()}, nullptr,
        nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Z power operation
struct ZPowOpLowering : public OpConversionPattern<ZPowOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZPowOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedBase = loadGmpValue(rewriter, loc, adaptor.getBase());

    // mpz_pow_ui(result, base, exp)
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_pow_ui",
        ValueRange{loadedResult, loadedBase, adaptor.getExp()}, nullptr,
        nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Z succ/pred operations
struct ZSuccOpLowering : public OpConversionPattern<ZSuccOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZSuccOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // mpz_add_ui(result, operand, 1)
    auto one = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_add_ui",
        ValueRange{loadedResult, loadedOperand, one.getResult()}, nullptr,
        nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZPredOpLowering : public OpConversionPattern<ZPredOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZPredOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // mpz_sub_ui(result, operand, 1)
    auto one = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_sub_ui",
        ValueRange{loadedResult, loadedOperand, one.getResult()}, nullptr,
        nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Z divisibility test
struct ZDivisibleOpLowering : public OpConversionPattern<ZDivisibleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZDivisibleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    // mpz_divisible_p returns non-zero if lhs is divisible by rhs
    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, rewriter.getI32Type(), "mpz_divisible_p",
        ValueRange{loadedLhs, loadedRhs}, nullptr, nullptr);

    // Convert non-zero to i1 (bool)
    auto zero = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto boolResult = rewriter.create<emitc::CmpOp>(
        loc, rewriter.getI1Type(),
        emitc::CmpPredicateAttr::get(rewriter.getContext(),
                                     emitc::CmpPredicate::ne),
        result.getResult(0), zero.getResult());

    rewriter.replaceOp(op, boolResult.getResult());
    return success();
  }
};

// Z congruence test
struct ZCongruentOpLowering : public OpConversionPattern<ZCongruentOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZCongruentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());
    auto loadedModulus = loadGmpValue(rewriter, loc, adaptor.getModulus());

    // mpz_congruent_p returns non-zero if lhs ≡ rhs (mod modulus)
    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, rewriter.getI32Type(), "mpz_congruent_p",
        ValueRange{loadedLhs, loadedRhs, loadedModulus}, nullptr, nullptr);

    // Convert non-zero to i1 (bool)
    auto zero = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto boolResult = rewriter.create<emitc::CmpOp>(
        loc, rewriter.getI1Type(),
        emitc::CmpPredicateAttr::get(rewriter.getContext(),
                                     emitc::CmpPredicate::ne),
        result.getResult(0), zero.getResult());

    rewriter.replaceOp(op, boolResult.getResult());
    return success();
  }
};

// Z GCD and LCM
struct ZGcdOpLowering : public OpConversionPattern<ZGcdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZGcdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_gcd",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct ZLcmOpLowering : public OpConversionPattern<ZLcmOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZLcmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_lcm",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Z sqrt
struct ZSqrtOpLowering : public OpConversionPattern<ZSqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZSqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_sqrt", ValueRange{loadedResult, loadedOperand},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Z bit operations
struct ZTestBitOpLowering : public OpConversionPattern<ZTestBitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZTestBitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // mpz_tstbit returns 0 or 1
    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, rewriter.getI32Type(), "mpz_tstbit",
        ValueRange{loadedOperand, adaptor.getIndex()}, nullptr, nullptr);

    // Convert to i1
    auto zero = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
    auto boolResult = rewriter.create<emitc::CmpOp>(
        loc, rewriter.getI1Type(),
        emitc::CmpPredicateAttr::get(rewriter.getContext(),
                                     emitc::CmpPredicate::ne),
        result.getResult(0), zero.getResult());

    rewriter.replaceOp(op, boolResult.getResult());
    return success();
  }
};

struct ZPopCountOpLowering : public OpConversionPattern<ZPopCountOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZPopCountOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, rewriter.getI64Type(), "mpz_popcount", ValueRange{loadedOperand},
        nullptr, nullptr);

    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

struct ZNumBitsOpLowering : public OpConversionPattern<ZNumBitsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ZNumBitsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // mpz_sizeinbase returns size_t
    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, rewriter.getI64Type(), "mpz_sizeinbase", ValueRange{loadedOperand},
        nullptr, nullptr);

    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Q Module Conversion Patterns
//===----------------------------------------------------------------------===//

// Convert gmp.q.constant to mpq_init + mpq_set_str/mpq_set_si calls
struct QConstantOpLowering : public OpConversionPattern<QConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    // Create temporary mpq_t variable
    auto resultVar = createTempMpqVar(rewriter, loc, context);
    createMpqInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);

    // Get the numerator and denominator from the QAttr
    auto qAttr = op.getValue();
    std::string numStr = qAttr.getNumerator();
    std::string denStr = qAttr.getDenominator();

    // Create fraction string "num/den"
    std::string fracStr = numStr + "/" + denStr;

    // Create mpq_set_str call
    auto strAttr = rewriter.create<emitc::ConstantOp>(
        loc, emitc::OpaqueType::get(context, "const char*"),
        emitc::OpaqueAttr::get(context, "\"" + fracStr + "\""));
    auto baseAttr = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(10));

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_set_str",
        ValueRange{loadedResult, strAttr.getResult(), baseAttr.getResult()},
        nullptr, nullptr);

    // Canonicalize the rational
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "mpq_canonicalize",
                                         ValueRange{loadedResult}, nullptr,
                                         nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Convert gmp.q.make to construct Q from Z numerator and denominator
struct QMakeOpLowering : public OpConversionPattern<QMakeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QMakeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpqVar(rewriter, loc, context);
    createMpqInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedNum = loadGmpValue(rewriter, loc, adaptor.getNumerator());
    auto loadedDen = loadGmpValue(rewriter, loc, adaptor.getDenominator());

    // mpq_set_num and mpq_set_den
    // First get pointers to numerator and denominator of result
    auto numRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_ptr"), "mpq_numref",
        ValueRange{loadedResult}, nullptr, nullptr);
    auto denRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_ptr"), "mpq_denref",
        ValueRange{loadedResult}, nullptr, nullptr);

    // Set numerator and denominator
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_set", ValueRange{numRef.getResult(0), loadedNum},
        nullptr, nullptr);
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_set", ValueRange{denRef.getResult(0), loadedDen},
        nullptr, nullptr);

    // Canonicalize
    rewriter.create<emitc::CallOpaqueOp>(loc, TypeRange{}, "mpq_canonicalize",
                                         ValueRange{loadedResult}, nullptr,
                                         nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Q component accessors
struct QNumOpLowering : public OpConversionPattern<QNumOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QNumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // Get numerator reference and copy it
    auto numRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_srcptr"), "mpq_numref",
        ValueRange{loadedOperand}, nullptr, nullptr);
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_set",
        ValueRange{loadedResult, numRef.getResult(0)}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct QDenOpLowering : public OpConversionPattern<QDenOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QDenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // Get denominator reference and copy it
    auto denRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_srcptr"), "mpq_denref",
        ValueRange{loadedOperand}, nullptr, nullptr);
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_set",
        ValueRange{loadedResult, denRef.getResult(0)}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Q arithmetic operations
struct QAddOpLowering : public OpConversionPattern<QAddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpqVar(rewriter, loc, context);
    createMpqInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_add",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct QSubOpLowering : public OpConversionPattern<QSubOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QSubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpqVar(rewriter, loc, context);
    createMpqInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_sub",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct QMulOpLowering : public OpConversionPattern<QMulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpqVar(rewriter, loc, context);
    createMpqInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_mul",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct QDivOpLowering : public OpConversionPattern<QDivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QDivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpqVar(rewriter, loc, context);
    createMpqInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_div",
        ValueRange{loadedResult, loadedLhs, loadedRhs}, nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Q unary operations
struct QNegOpLowering : public OpConversionPattern<QNegOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QNegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpqVar(rewriter, loc, context);
    createMpqInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_neg", ValueRange{loadedResult, loadedOperand},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct QAbsOpLowering : public OpConversionPattern<QAbsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QAbsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpqVar(rewriter, loc, context);
    createMpqInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_abs", ValueRange{loadedResult, loadedOperand},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct QInvOpLowering : public OpConversionPattern<QInvOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QInvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpqVar(rewriter, loc, context);
    createMpqInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_inv", ValueRange{loadedResult, loadedOperand},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Q comparison operations
struct QCompareOpLowering : public OpConversionPattern<QCompareOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QCompareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    auto cmpOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, rewriter.getI32Type(), "mpq_cmp", ValueRange{loadedLhs, loadedRhs},
        nullptr, nullptr);

    rewriter.replaceOp(op, cmpOp.getResult(0));
    return success();
  }
};

// Q relational operations - using similar pattern to Z
template <typename OpTy>
struct QRelationalOpLowering : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  QRelationalOpLowering(TypeConverter &converter, MLIRContext *context,
                        StringRef cmpOp)
      : OpConversionPattern<OpTy>(converter, context), compOp(cmpOp) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto loadedLhs = loadGmpValue(rewriter, loc, adaptor.getLhs());
    auto loadedRhs = loadGmpValue(rewriter, loc, adaptor.getRhs());

    auto cmpOp = rewriter.create<emitc::CallOpaqueOp>(
        loc, rewriter.getI32Type(), "mpq_cmp", ValueRange{loadedLhs, loadedRhs},
        nullptr, nullptr);

    auto zero = rewriter.create<emitc::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    auto result = rewriter.create<emitc::CmpOp>(
        loc, rewriter.getI1Type(),
        emitc::CmpPredicateAttr::get(context, getCmpPredicate()),
        cmpOp.getResult(0), zero.getResult());

    rewriter.replaceOp(op, result.getResult());
    return success();
  }

private:
  std::string compOp;

  emitc::CmpPredicate getCmpPredicate() const {
    if (compOp == "eq")
      return emitc::CmpPredicate::eq;
    if (compOp == "lt")
      return emitc::CmpPredicate::lt;
    if (compOp == "le")
      return emitc::CmpPredicate::le;
    if (compOp == "gt")
      return emitc::CmpPredicate::gt;
    if (compOp == "ge")
      return emitc::CmpPredicate::ge;
    return emitc::CmpPredicate::eq;
  }
};

// Q rounding operations
struct QFloorOpLowering : public OpConversionPattern<QFloorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QFloorOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // Get numerator and denominator
    auto numRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_srcptr"), "mpq_numref",
        ValueRange{loadedOperand}, nullptr, nullptr);
    auto denRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_srcptr"), "mpq_denref",
        ValueRange{loadedOperand}, nullptr, nullptr);

    // Use floor division: mpz_fdiv_q
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_fdiv_q",
        ValueRange{loadedResult, numRef.getResult(0), denRef.getResult(0)},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct QCeilOpLowering : public OpConversionPattern<QCeilOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QCeilOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    auto numRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_srcptr"), "mpq_numref",
        ValueRange{loadedOperand}, nullptr, nullptr);
    auto denRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_srcptr"), "mpq_denref",
        ValueRange{loadedOperand}, nullptr, nullptr);

    // Use ceiling division: mpz_cdiv_q
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_cdiv_q",
        ValueRange{loadedResult, numRef.getResult(0), denRef.getResult(0)},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct QTruncOpLowering : public OpConversionPattern<QTruncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QTruncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    auto numRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_srcptr"), "mpq_numref",
        ValueRange{loadedOperand}, nullptr, nullptr);
    auto denRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_srcptr"), "mpq_denref",
        ValueRange{loadedOperand}, nullptr, nullptr);

    // Use truncated division: mpz_tdiv_q
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_tdiv_q",
        ValueRange{loadedResult, numRef.getResult(0), denRef.getResult(0)},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

// Q conversion operations
struct QToBigIntOpLowering : public OpConversionPattern<QToBigIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QToBigIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpzVar(rewriter, loc, context);
    createMpzInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    auto numRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_srcptr"), "mpq_numref",
        ValueRange{loadedOperand}, nullptr, nullptr);
    auto denRef = rewriter.create<emitc::CallOpaqueOp>(
        loc, emitc::OpaqueType::get(context, "mpz_srcptr"), "mpq_denref",
        ValueRange{loadedOperand}, nullptr, nullptr);

    // Truncated division for conversion to integer
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpz_tdiv_q",
        ValueRange{loadedResult, numRef.getResult(0), denRef.getResult(0)},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

struct QToF64OpLowering : public OpConversionPattern<QToF64Op> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QToF64Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    auto result = rewriter.create<emitc::CallOpaqueOp>(
        loc, rewriter.getF64Type(), "mpq_get_d", ValueRange{loadedOperand},
        nullptr, nullptr);

    rewriter.replaceOp(op, result.getResult(0));
    return success();
  }
};

struct QFromZOpLowering : public OpConversionPattern<QFromZOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(QFromZOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = rewriter.getContext();

    auto resultVar = createTempMpqVar(rewriter, loc, context);
    createMpqInit(rewriter, loc, resultVar);

    auto loadedResult = loadGmpValue(rewriter, loc, resultVar);
    auto loadedOperand = loadGmpValue(rewriter, loc, adaptor.getOperand());

    // mpq_set_z sets the rational to z/1
    rewriter.create<emitc::CallOpaqueOp>(
        loc, TypeRange{}, "mpq_set_z", ValueRange{loadedResult, loadedOperand},
        nullptr, nullptr);

    rewriter.replaceOp(op, resultVar);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ConvertGMPToEmitCPass
    : public impl::ConvertGMPToEmitCBase<ConvertGMPToEmitCPass> {
  using Base = impl::ConvertGMPToEmitCBase<ConvertGMPToEmitCPass>;

  ConvertGMPToEmitCPass() = default;
  ConvertGMPToEmitCPass(const ConvertGMPToEmitCPass &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<emitc::EmitCDialect, func::FuncDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Set up type converter
    GMPToEmitCTypeConverter typeConverter(context);

    // Add source/target materialization for the type converter
    // This handles function argument conversions
    typeConverter.addSourceMaterialization(
        [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
          return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
              .getResult(0);
        });
    typeConverter.addTargetMaterialization(
        [](OpBuilder &builder, Type type, ValueRange inputs, Location loc) {
          return builder.create<UnrealizedConversionCastOp>(loc, type, inputs)
              .getResult(0);
        });

    // Set up conversion target
    ConversionTarget target(*context);
    target.addLegalDialect<emitc::EmitCDialect, func::FuncDialect,
                           arith::ArithDialect>();
    target.addIllegalDialect<GMPDialect>();

    // Mark func.func as dynamically legal - legal if it has no GMP types
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      auto funcType = op.getFunctionType();
      for (Type inputType : funcType.getInputs()) {
        if (isa<ZType, QType>(inputType))
          return false;
      }
      for (Type resultType : funcType.getResults()) {
        if (isa<ZType, QType>(resultType))
          return false;
      }
      return true;
    });

    // Mark func.return as dynamically legal
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      for (Type operandType : op.getOperandTypes()) {
        if (isa<ZType, QType>(operandType))
          return false;
      }
      return true;
    });

    // Set up rewrite patterns
    RewritePatternSet patterns(context);

    // Add function type conversion patterns
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    // Z module patterns
    patterns.add<ZConstantOpLowering>(typeConverter, context);
    patterns.add<ZFromIntOpLowering>(typeConverter, context);
    patterns.add<ZBinaryOpLowering<ZAddOp>>(typeConverter, context, "mpz_add");
    patterns.add<ZBinaryOpLowering<ZSubOp>>(typeConverter, context, "mpz_sub");
    patterns.add<ZBinaryOpLowering<ZMulOp>>(typeConverter, context, "mpz_mul");
    patterns.add<ZUnaryOpLowering<ZNegOp>>(typeConverter, context, "mpz_neg");
    patterns.add<ZUnaryOpLowering<ZAbsOp>>(typeConverter, context, "mpz_abs");
    patterns.add<ZDivOpLowering>(typeConverter, context);
    patterns.add<ZRemOpLowering>(typeConverter, context);
    patterns.add<ZFDivOpLowering>(typeConverter, context);
    patterns.add<ZFRemOpLowering>(typeConverter, context);
    patterns.add<ZCDivOpLowering>(typeConverter, context);
    patterns.add<ZCRemOpLowering>(typeConverter, context);
    patterns.add<ZEDivOpLowering>(typeConverter, context);
    patterns.add<ZERemOpLowering>(typeConverter, context);
    patterns.add<ZDivExactOpLowering>(typeConverter, context);
    patterns.add<ZCompareOpLowering>(typeConverter, context);
    patterns.add<ZRelationalOpLowering<ZEqualOp>>(typeConverter, context, "eq");
    patterns.add<ZRelationalOpLowering<ZLtOp>>(typeConverter, context, "lt");
    patterns.add<ZRelationalOpLowering<ZLeqOp>>(typeConverter, context, "le");
    patterns.add<ZRelationalOpLowering<ZGtOp>>(typeConverter, context, "gt");
    patterns.add<ZRelationalOpLowering<ZGeqOp>>(typeConverter, context, "ge");
    patterns.add<ZLogAndOpLowering>(typeConverter, context);
    patterns.add<ZLogOrOpLowering>(typeConverter, context);
    patterns.add<ZLogXorOpLowering>(typeConverter, context);
    patterns.add<ZLogNotOpLowering>(typeConverter, context);
    patterns.add<ZShiftLeftOpLowering>(typeConverter, context);
    patterns.add<ZShiftRightOpLowering>(typeConverter, context);
    patterns.add<ZShiftRightTruncOpLowering>(typeConverter, context);
    patterns.add<ZPowOpLowering>(typeConverter, context);
    patterns.add<ZSuccOpLowering>(typeConverter, context);
    patterns.add<ZPredOpLowering>(typeConverter, context);
    patterns.add<ZDivisibleOpLowering>(typeConverter, context);
    patterns.add<ZCongruentOpLowering>(typeConverter, context);
    patterns.add<ZGcdOpLowering>(typeConverter, context);
    patterns.add<ZLcmOpLowering>(typeConverter, context);
    patterns.add<ZSqrtOpLowering>(typeConverter, context);
    patterns.add<ZTestBitOpLowering>(typeConverter, context);
    patterns.add<ZPopCountOpLowering>(typeConverter, context);
    patterns.add<ZNumBitsOpLowering>(typeConverter, context);

    // Q module patterns
    patterns.add<QConstantOpLowering>(typeConverter, context);
    patterns.add<QMakeOpLowering>(typeConverter, context);
    patterns.add<QNumOpLowering>(typeConverter, context);
    patterns.add<QDenOpLowering>(typeConverter, context);
    patterns.add<QAddOpLowering>(typeConverter, context);
    patterns.add<QSubOpLowering>(typeConverter, context);
    patterns.add<QMulOpLowering>(typeConverter, context);
    patterns.add<QDivOpLowering>(typeConverter, context);
    patterns.add<QNegOpLowering>(typeConverter, context);
    patterns.add<QAbsOpLowering>(typeConverter, context);
    patterns.add<QInvOpLowering>(typeConverter, context);
    patterns.add<QCompareOpLowering>(typeConverter, context);
    patterns.add<QRelationalOpLowering<QEqualOp>>(typeConverter, context, "eq");
    patterns.add<QRelationalOpLowering<QLtOp>>(typeConverter, context, "lt");
    patterns.add<QRelationalOpLowering<QLeqOp>>(typeConverter, context, "le");
    patterns.add<QRelationalOpLowering<QGtOp>>(typeConverter, context, "gt");
    patterns.add<QRelationalOpLowering<QGeqOp>>(typeConverter, context, "ge");
    patterns.add<QFloorOpLowering>(typeConverter, context);
    patterns.add<QCeilOpLowering>(typeConverter, context);
    patterns.add<QTruncOpLowering>(typeConverter, context);
    patterns.add<QToBigIntOpLowering>(typeConverter, context);
    patterns.add<QToF64OpLowering>(typeConverter, context);
    patterns.add<QFromZOpLowering>(typeConverter, context);

    // Apply conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Creation Function
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createConvertGMPToEmitCPass() {
  return std::make_unique<ConvertGMPToEmitCPass>();
}

} // namespace gmp
} // namespace mlir
