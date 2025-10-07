//===- ASLCanonicalization.cpp - ASL Canonicalization Patterns -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements canonicalization patterns for ASL dialect operations.
//
//===----------------------------------------------------------------------===//

#include "ASL/ASLOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::asl;

//===----------------------------------------------------------------------===//
// Canonicalization Pattern Infrastructure
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Binary Operation Canonicalization Patterns
//===----------------------------------------------------------------------===//

// TODO: Add constant folding patterns for binary operations
// Example patterns that could be added:
// - Fold operations with constant operands (e.g., x + 0 = x, x * 1 = x)
// - Algebraic simplifications (e.g., x - x = 0, x / x = 1)
// - Associativity/commutativity reordering for better CSE

//===----------------------------------------------------------------------===//
// Unary Operation Canonicalization Patterns
//===----------------------------------------------------------------------===//

/// Fold double negation: -(-x) = x
struct UnopNegDoubleNegation : public OpRewritePattern<UnopNegOp> {
  using OpRewritePattern<UnopNegOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnopNegOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the operand is also a negation operation
    auto innerNeg = op.getOperand().getDefiningOp<UnopNegOp>();
    if (!innerNeg)
      return failure();

    // Replace -(-x) with x
    rewriter.replaceOp(op, innerNeg.getOperand());
    return success();
  }
};

/// Fold double boolean inversion: !(!x) = x
struct UnopBnotDoubleNegation : public OpRewritePattern<UnopBnotOp> {
  using OpRewritePattern<UnopBnotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnopBnotOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the operand is also a boolean inversion operation
    auto innerBnot = op.getOperand().getDefiningOp<UnopBnotOp>();
    if (!innerBnot)
      return failure();

    // Replace !(!x) with x
    rewriter.replaceOp(op, innerBnot.getOperand());
    return success();
  }
};

/// Fold double bitwise inversion: ~(~x) = x
struct UnopNotDoubleNegation : public OpRewritePattern<UnopNotOp> {
  using OpRewritePattern<UnopNotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnopNotOp op,
                                PatternRewriter &rewriter) const override {
    // Check if the operand is also a bitwise inversion operation
    auto innerNot = op.getOperand().getDefiningOp<UnopNotOp>();
    if (!innerNot)
      return failure();

    // Replace ~(~x) with x
    rewriter.replaceOp(op, innerNot.getOperand());
    return success();
  }
};

// TODO: Add more constant folding patterns for unary operations

//===----------------------------------------------------------------------===//
// Global Storage Declaration Canonicalization Patterns
//===----------------------------------------------------------------------===//

/// Fold constant literal initial values into ConstantInitGlobalStorageDeclOp
/// This pattern matches GlobalStorageDeclOp with a literal operation as the
/// initial value and folds the literal into an attribute.
struct GlobalStorageDeclConstantFolder
    : public OpRewritePattern<GlobalStorageDeclOp> {
  using OpRewritePattern<GlobalStorageDeclOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GlobalStorageDeclOp op,
                                PatternRewriter &rewriter) const override {
    Value initialValue = op.getInitialValue();
    Operation *definingOp = initialValue.getDefiningOp();

    if (!definingOp)
      return failure();

    // Extract the constant value from various literal operations
    StringAttr constantValue;

    if (auto bitvectorLit = dyn_cast<LiteralBitvectorOp>(definingOp)) {
      constantValue = bitvectorLit.getValueAttr();
    } else if (auto intLit = dyn_cast<LiteralIntOp>(definingOp)) {
      constantValue = intLit.getValueAttr();
    } else if (auto boolLit = dyn_cast<LiteralBoolOp>(definingOp)) {
      // Convert BoolAttr to StringAttr
      bool boolValue = boolLit.getValue();
      constantValue = rewriter.getStringAttr(boolValue ? "true" : "false");
    } else if (auto realLit = dyn_cast<LiteralRealOp>(definingOp)) {
      constantValue = realLit.getValueAttr();
    } else if (auto stringLit = dyn_cast<LiteralStringOp>(definingOp)) {
      constantValue = stringLit.getValueAttr();
    } else {
      // Not a literal operation we can fold
      return failure();
    }

    // Create the new ConstantInitGlobalStorageDeclOp with the folded constant
    rewriter.replaceOpWithNewOp<ConstantInitGlobalStorageDeclOp>(
        op, op.getKeywordAttr(), op.getNameAttr(), op.getTypeAttr(),
        constantValue);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Type Conversion Canonicalization Patterns
//===----------------------------------------------------------------------===//

// TODO: Add patterns for type conversion operations
// Example patterns:
// - Remove redundant type conversions (e.g., atc(atc(x, T), T) = atc(x, T))
// - Fold type conversions with constants

//===----------------------------------------------------------------------===//
// Pattern Matching Canonicalization Patterns
//===----------------------------------------------------------------------===//

// TODO: Add patterns for pattern matching operations
// Example patterns:
// - Simplify pattern.not(pattern.not(x)) = x
// - Optimize range patterns with constant bounds

//===----------------------------------------------------------------------===//
// L-Expression Canonicalization Patterns
//===----------------------------------------------------------------------===//

// TODO: Add patterns for L-expression operations
// Example patterns:
// - Simplify nested field access patterns
// - Optimize slice operations with constant indices

} // namespace

//===----------------------------------------------------------------------===//
// Populate Canonicalization Patterns
//===----------------------------------------------------------------------===//

void mlir::asl::populateASLCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  // Unary operation patterns
  patterns.add<UnopNegDoubleNegation, UnopBnotDoubleNegation,
               UnopNotDoubleNegation>(patterns.getContext());

  // Global storage declaration patterns
  patterns.add<GlobalStorageDeclConstantFolder>(patterns.getContext());

  // TODO: Add more canonicalization patterns as they are implemented
  // Binary operation patterns:
  // - patterns.add<BinopPlusIdentity>(patterns.getContext());
  // - patterns.add<BinopMulIdentity>(patterns.getContext());
  // - patterns.add<BinopZeroAbsorption>(patterns.getContext());

  // Type conversion patterns:
  // - patterns.add<RedundantAtcElimination>(patterns.getContext());
  // - patterns.add<AtcConstantFolding>(patterns.getContext());

  // Pattern matching patterns:
  // - patterns.add<PatternNotDoubleNegation>(patterns.getContext());
  // - patterns.add<PatternRangeConstantFolding>(patterns.getContext());
}
