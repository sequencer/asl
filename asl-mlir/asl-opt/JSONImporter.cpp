//===- JSONImporter.cpp - Import ASL JSON IR --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "JSONImporter.h"

#include "ASL/ASLAttributes.h"
#include "ASL/ASLDialect.h"
#include "ASL/ASLOps.h"
#include "ASL/ASLTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
// Added for std::function used in bitfield parsing
#include <functional>

using namespace mlir;
using namespace mlir::asl;

namespace {
struct JSONImporter {
  MLIRContext &ctx;
  OpBuilder builder;
  Location loc;
  // Simple symbol table stack mapping identifier -> Value
  SmallVector<llvm::StringMap<Value>, 8> scopes;
  // Pointer to current JSON value context (if any) for richer error messages.
  const llvm::json::Value *currentContext = nullptr;
  // Current function return type context (if any) used for inserting ATC on
  // return.
  TypeAttr currentReturnTypeAttr = nullptr;
  // If the return type is Bits, this holds the width value SSA to feed into
  // atc.bits.
  Value currentReturnBitsWidth;
  // If the return type is Array, this holds the length value SSA to feed into
  // atc.array.
  Value currentReturnArrayLength;

  // RAII helper to set currentContext.
  struct ContextSetter {
    JSONImporter &imp;
    const llvm::json::Value *prev;
    ContextSetter(JSONImporter &imp, const llvm::json::Value *v)
        : imp(imp), prev(imp.currentContext) {
      imp.currentContext = v;
    }
    ~ContextSetter() { imp.currentContext = prev; }
  };
  // RAII helper to set current function return type.
  struct ReturnTypeSetter {
    JSONImporter &imp;
    TypeAttr prev;
    ReturnTypeSetter(JSONImporter &imp, TypeAttr t)
        : imp(imp), prev(imp.currentReturnTypeAttr) {
      imp.currentReturnTypeAttr = t;
    }
    ~ReturnTypeSetter() { imp.currentReturnTypeAttr = prev; }
  };

  JSONImporter(MLIRContext &ctx)
      : ctx(ctx), builder(&ctx), loc(builder.getUnknownLoc()) {
    // Ensure the ASL dialect is loaded before we attempt to create any
    // ASL attributes or types programmatically (enum attrs, etc.).
    ctx.loadDialect<ASLDialect>();
    pushScope();
  }

  void pushScope() { scopes.emplace_back(); }
  void popScope() {
    if (!scopes.empty())
      scopes.pop_back();
  }
  void bind(StringRef name, Value v) {
    if (!scopes.empty())
      scopes.back()[name] = v;
  }
  Value lookup(StringRef name) {
    for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
      auto f = it->find(name);
      if (f != it->end())
        return f->second;
    }
    return Value();
  }

  //===----------------------------------------------------------------------//
  // Helpers
  //===----------------------------------------------------------------------//

  template <typename T>
  llvm::Error makeError(const Twine &msg, const T &detail) {
    std::string ctxDump;
    if (currentContext) {
      llvm::raw_string_ostream os(ctxDump);
      os << "\nContext JSON: " << *currentContext; // dump JSON value
    }
    return llvm::make_error<llvm::StringError>(
        (msg + detail + Twine(ctxDump)).str(), llvm::inconvertibleErrorCode());
  }
  llvm::Error makeError(const Twine &msg) {
    std::string ctxDump;
    if (currentContext) {
      llvm::raw_string_ostream os(ctxDump);
      os << "\nContext JSON: " << *currentContext;
    }
    return llvm::make_error<llvm::StringError>((msg + Twine(ctxDump)).str(),
                                               llvm::inconvertibleErrorCode());
  }

  // Get required string property
  llvm::Expected<StringRef> getString(const llvm::json::Object &obj,
                                      StringRef key) {
    if (auto v = obj.getString(key))
      return *v;
    return makeError("missing string field: ", key);
  }

  // Get optional string (returns Empty string if absent)
  std::optional<StringRef> getOptString(const llvm::json::Object &obj,
                                        StringRef key) {
    if (auto v = obj.getString(key))
      return *v;
    return std::nullopt;
  }

  // Get required object
  llvm::Expected<const llvm::json::Object *>
  getObject(const llvm::json::Value &v) {
    if (auto *o = v.getAsObject())
      return o;
    return makeError("expected object value");
  }

  // Map subprogram type JSON string (ST_*) to enum attr keyword
  SubprogramType mapSubprogramType(StringRef s) {
    return llvm::StringSwitch<SubprogramType>(s)
        .Case("ST_Procedure", SubprogramType::procedure)
        .Case("ST_Function", SubprogramType::function)
        .Case("ST_Getter", SubprogramType::getter)
        .Case("ST_EmptyGetter", SubprogramType::emptygetter)
        .Case("ST_Setter", SubprogramType::setter)
        .Case("ST_EmptySetter", SubprogramType::emptysetter)
        .Default(SubprogramType::procedure);
  }

  FuncQualifier mapFuncQualifier(StringRef s) {
    return llvm::StringSwitch<FuncQualifier>(s)
        .Case("Pure", FuncQualifier::pure)
        .Case("Readonly", FuncQualifier::readonly)
        .Case("Noreturn", FuncQualifier::noreturn)
        .Default(FuncQualifier::pure);
  }

  OverrideInfo mapOverrideInfo(StringRef s) {
    return llvm::StringSwitch<OverrideInfo>(s)
        .Case("Impdef", OverrideInfo::impdef)
        .Case("Implementation", OverrideInfo::implementation)
        .Default(OverrideInfo::impdef);
  }

  GDK mapGlobalDeclKeyword(StringRef s) {
    return llvm::StringSwitch<GDK>(s)
        .Case("GDK_Constant", GDK::constant)
        .Case("GDK_Config", GDK::config)
        .Case("GDK_Let", GDK::let)
        .Case("GDK_Var", GDK::var)
        .Default(GDK::var);
  }

  StringRef mapForDirection(StringRef s) {
    return llvm::StringSwitch<StringRef>(s)
        .Case("Up", "up")
        .Case("Down", "down")
        .Default("up");
  }
  asl::ForDirectionAttr mapForDirectionAttr(StringRef s) {
    auto dirStr = mapForDirection(s);
    auto dirEnum = llvm::StringSwitch<asl::ForDirection>(dirStr)
                       .Case("up", asl::ForDirection::up)
                       .Case("down", asl::ForDirection::down)
                       .Default(asl::ForDirection::up);
    return asl::ForDirectionAttr::get(&ctx, dirEnum);
  }

  // Local decl keyword mapping (LDK_*)
  asl::LDKAttr mapLocalDeclKeyword(StringRef s) {
    auto k = llvm::StringSwitch<asl::LDK>(s)
                 .Case("LDK_Var", asl::LDK::var)
                 .Case("LDK_Constant", asl::LDK::constant)
                 .Case("LDK_Let", asl::LDK::let)
                 .Default(asl::LDK::var);
    return asl::LDKAttr::get(&ctx, k);
  }
  // Build LDIAttr from decl_item object
  llvm::Expected<asl::LDIAttr>
  buildLocalDeclItem(const llvm::json::Object &obj) {
    auto kind = obj.getString("type");
    if (!kind)
      return makeError("local decl item missing type");
    if (*kind == "LDI_Var") {
      auto name = obj.getString("name");
      if (!name)
        return makeError("LDI_Var missing name");
      return asl::LDIAttr::get(&ctx,
                               asl::LDIKindAttr::get(&ctx, asl::LDIKind::var),
                               builder.getStringAttr(*name), nullptr);
    } else if (*kind == "LDI_Tuple") {
      SmallVector<Attribute> names;
      if (auto *arrV = obj.get("names")) {
        if (auto *arr = arrV->getAsArray()) {
          for (auto &n : *arr) {
            if (auto ns = n.getAsString())
              names.push_back(builder.getStringAttr(*ns));
          }
        }
      }
      auto tupleAttr = builder.getArrayAttr(names);
      return asl::LDIAttr::get(&ctx,
                               asl::LDIKindAttr::get(&ctx, asl::LDIKind::tuple),
                               nullptr, tupleAttr);
    }
    return makeError("unsupported local decl item type: ", *kind);
  }

  // Helper builders for default unconstrained integer type
  asl::ConstraintKindTypeAttr getUnconstrainedKind() {
    return asl::ConstraintKindTypeAttr::get(
        &ctx, asl::ConstraintKindType::unconstrained);
  }
  asl::PrecisionLossFlagAttr getFullPrecision() {
    return asl::PrecisionLossFlagAttr::get(&ctx, asl::PrecisionLossFlag::full);
  }
  asl::ConstraintKindAttr buildUnconstrainedConstraint() {
    return asl::ConstraintKindAttr::get(&ctx, getUnconstrainedKind(),
                                        builder.getArrayAttr({}),
                                        getFullPrecision());
  }
  asl::IntType getDefaultIntType() {
    return asl::IntType::get(&ctx, buildUnconstrainedConstraint());
  }
  // Helper: parameterized generic integer type used for template parameters
  asl::ConstraintKindAttr buildParameterizedConstraint() {
    return asl::ConstraintKindAttr::get(
        &ctx,
        asl::ConstraintKindTypeAttr::get(
            &ctx, asl::ConstraintKindType::parameterized),
        builder.getArrayAttr({}), getFullPrecision());
  }
  asl::IntType getParameterizedIntType() {
    return asl::IntType::get(&ctx, buildParameterizedConstraint());
  }
  asl::BitsType getDefaultBitsType() {
    return asl::BitsType::get(&ctx, builder.getI64IntegerAttr(-1),
                              builder.getArrayAttr({}));
  }

  // Helper: Check if a type depends on parameterized integers
  bool typeUsesParameterizedIntegers(Type type) {
    if (auto intType = dyn_cast<asl::IntType>(type)) {
      auto constraintAttr = intType.getConstraint();
      return constraintAttr.getKind().getValue() ==
             asl::ConstraintKindType::parameterized;
    }
    if (auto bitsType = dyn_cast<asl::BitsType>(type)) {
      // For bits type, we need to check if width depends on parameters
      // This is determined at the JSON level - if width is not a literal
      // expression, then it likely depends on parameters
      return true; // Conservative: assume bits type could depend on parameters
    }
    if (auto arrayType = dyn_cast<asl::ArrayType>(type)) {
      // Check if element type or index depends on parameters
      if (typeUsesParameterizedIntegers(arrayType.getElementType().getValue()))
        return true;
      // Array length could also depend on parameters
      return true; // Conservative: assume array type could depend on parameters
    }
    if (auto tupleType = dyn_cast<asl::TupleType>(type)) {
      // Check if any element type depends on parameters
      for (auto typeAttr : tupleType.getTypes()) {
        if (auto typeAttrCast = dyn_cast<TypeAttr>(typeAttr)) {
          if (typeUsesParameterizedIntegers(typeAttrCast.getValue()))
            return true;
        }
      }
    }
    // Add more type checks as needed
    return false;
  }

  // Helper: Check if a type depends on parameterized integers by examining JSON
  bool
  typeUsesParameterizedIntegersFromJSON(const llvm::json::Value *typeJson) {
    if (!typeJson)
      return false;

    auto *obj = typeJson->getAsObject();
    if (!obj)
      return false;

    auto kind = obj->getString("type");
    if (!kind)
      return false;

    StringRef k = *kind;

    if (k == "T_Bits") {
      // Check if width is non-literal (depends on parameters)
      if (auto *wV = obj->get("width")) {
        if (auto *wObj = wV->getAsObject()) {
          auto wKind = wObj->getString("type");
          // If width is not a literal, it depends on parameters
          return !(wKind && *wKind == "E_Literal");
        }
      }
    } else if (k == "T_Array") {
      // Check if element type depends on parameters
      if (auto *elemV = obj->get("element_type")) {
        if (typeUsesParameterizedIntegersFromJSON(elemV))
          return true;
      }
      // Check if index depends on parameters
      if (auto *idxV = obj->get("index")) {
        if (auto *idxObj = idxV->getAsObject()) {
          if (auto idxType = idxObj->getString("type")) {
            if (*idxType == "ArrayLength_Expr") {
              if (auto *eV = idxObj->get("expr")) {
                if (auto *eObj = eV->getAsObject()) {
                  auto eKind = eObj->getString("type");
                  // If length expression is not a literal, it depends on
                  // parameters
                  return !(eKind && *eKind == "E_Literal");
                }
              }
            }
          }
        }
      }
    } else if (k == "T_Int") {
      // Check constraint kind
      if (auto *cV = obj->get("constraint_kind")) {
        if (auto *cObj = cV->getAsObject()) {
          auto cKind = cObj->getString("type");
          if (cKind && *cKind == "Parameterized") {
            return true;
          }
        }
      }
    }
    // Add more type checks as needed
    return false;
  }

  //===----------------------------------------------------------------------//
  // Importer implementation
  //===----------------------------------------------------------------------//

  // Parse Type object from JSON (complete implementation)
  llvm::Expected<Type> parseType(const llvm::json::Value &v) {
    ContextSetter guard(*this, &v);
    const auto *obj = v.getAsObject();
    if (!obj)
      return makeError("type entry not object");
    auto kind = obj->getString("type");
    if (!kind)
      return makeError("missing type discriminant");
    StringRef k = *kind;

    // Integer type (with optional constraint descriptor)
    if (k == "T_Int") {
      asl::ConstraintKindAttr constraintAttr = buildUnconstrainedConstraint();
      // Updated to read "constraint_kind" which encodes a variant object with a
      // discriminant field named "type" matching one of:
      //   UnConstrained | WellConstrained | PendingConstrained | Parameterized
      if (auto *cV = obj->get("constraint_kind")) {
        if (auto *cObj = cV->getAsObject()) {
          auto cKind = cObj->getString("type");
          if (cKind) {
            SmallVector<Attribute> intConstraints;
            asl::ConstraintKindType kindEnum =
                asl::ConstraintKindType::unconstrained;
            asl::PrecisionLossFlag plEnum =
                asl::PrecisionLossFlag::full; // backend currently drops
                                              // precision flag
            if (*cKind == "UnConstrained") {
              kindEnum = asl::ConstraintKindType::unconstrained;
            } else if (*cKind == "WellConstrained") {
              kindEnum = asl::ConstraintKindType::constrained;
              if (auto *listV = cObj->get("constraints"))
                if (auto *arr = listV->getAsArray()) {
                  for (auto &cEntry : *arr) {
                    if (auto *ceObj = cEntry.getAsObject()) {
                      auto ceType = ceObj->getString("type");
                      if (!ceType)
                        continue;
                      if (*ceType == "Constraint_Exact") {
                        if (auto *eV = ceObj->get("expr")) {
                          if (auto *eObj = eV->getAsObject()) {
                            // For exact constraints, we need to mark this as
                            // constrained regardless of whether it's literal or
                            // not
                            if (auto et = eObj->getString("type")) {
                              if (*et == "E_Literal") {
                                if (auto *litV = eObj->get("literal"))
                                  if (auto *litObj = litV->getAsObject()) {
                                    if (auto lt = litObj->getString("type"))
                                      if (*lt == "L_Int")
                                        if (auto valStr =
                                                litObj->getString("value")) {
                                          // Parse as signed 64-bit integer to
                                          // avoid APInt sign-extension issues
                                          // from minimal bit-widths.
                                          int64_t sval = 0;
                                          if (!valStr->getAsInteger(10, sval))
                                            intConstraints.push_back(
                                                asl::IntConstraintAttr::get(
                                                    &ctx,
                                                    builder.getI64IntegerAttr(
                                                        sval),
                                                    nullptr, nullptr));
                                        }
                                  }
                              } else {
                                // Non-literal exact constraint - create a
                                // placeholder constraint so the type appears as
                                // constrained and gets picked up by the
                                // function argument ATC processing
                                intConstraints.push_back(
                                    asl::IntConstraintAttr::get(
                                        &ctx, nullptr, nullptr, nullptr));
                              }
                            }
                          }
                        }
                      } else if (*ceType == "Constraint_Range") {
                        // Parse start / end similarly
                        IntegerAttr lhsAttr = nullptr, rhsAttr = nullptr;
                        auto parseEndPoint =
                            [&](const char *key) -> IntegerAttr {
                          if (auto *pV = ceObj->get(key))
                            if (auto *pObj = pV->getAsObject())
                              if (auto pt = pObj->getString("type"))
                                if (*pt == "E_Literal")
                                  if (auto *litV = pObj->get("literal"))
                                    if (auto *litObj = litV->getAsObject())
                                      if (auto lt = litObj->getString("type"))
                                        if (*lt == "L_Int")
                                          if (auto valStr =
                                                  litObj->getString("value")) {
                                            int64_t sval = 0;
                                            if (!valStr->getAsInteger(10, sval))
                                              return builder.getI64IntegerAttr(
                                                  sval);
                                          }
                          return IntegerAttr();
                        };
                        lhsAttr = parseEndPoint("start");
                        rhsAttr = parseEndPoint("end");
                        intConstraints.push_back(asl::IntConstraintAttr::get(
                            &ctx, nullptr, lhsAttr, rhsAttr));
                      }
                    }
                  }
                }
            } else if (*cKind == "PendingConstrained") {
              return makeError("PendingConstrained types are forbidden in "
                               "typed AST parsing");
            } else if (*cKind == "Parameterized") {
              kindEnum = asl::ConstraintKindType::parameterized;
              // parameter name ignored for now
            }
            constraintAttr = asl::ConstraintKindAttr::get(
                &ctx, asl::ConstraintKindTypeAttr::get(&ctx, kindEnum),
                builder.getArrayAttr(intConstraints),
                asl::PrecisionLossFlagAttr::get(&ctx, plEnum));
          }
        }
      }
      return asl::IntType::get(&ctx, constraintAttr);
    }

    if (k == "T_Real")
      return asl::RealType::get(&ctx);
    if (k == "T_String")
      return asl::StringType::get(&ctx);
    if (k == "T_Bool")
      return builder.getI1Type();

    if (k == "T_Named") {
      auto name = obj->getString("name");
      if (!name)
        return makeError("named type missing name");
      return asl::NamedType::get(&ctx, builder.getStringAttr(*name));
    }

    if (k == "T_Bits") {
      // Helper lambdas.
      auto getIntFromLiteralExpr =
          [&](const llvm::json::Value *exprV) -> std::optional<int64_t> {
        if (!exprV)
          return std::nullopt;
        auto *eObj = exprV->getAsObject();
        if (!eObj)
          return std::nullopt;
        auto et = eObj->getString("type");
        if (!et || *et != "E_Literal")
          return std::nullopt;
        auto *litV = eObj->get("literal");
        if (!litV)
          return std::nullopt;
        auto *litObj = litV->getAsObject();
        if (!litObj)
          return std::nullopt;
        auto lt = litObj->getString("type");
        if (!lt || *lt != "L_Int")
          return std::nullopt;
        if (auto valStr = litObj->getString("value")) {
          llvm::APInt ap;
          if (!valStr->getAsInteger(10, ap))
            return ap.getSExtValue();
        }
        return std::nullopt;
      };
      auto parseSliceAttr = [&](const llvm::json::Value &sv) -> asl::SliceAttr {
        auto *sObj = sv.getAsObject();
        if (!sObj)
          return {};
        auto st = sObj->getString("type");
        if (!st)
          return {};
        auto kindEnum = llvm::StringSwitch<asl::SliceKind>(*st)
                            .Case("Slice_Single", asl::SliceKind::single)
                            .Case("Slice_Range", asl::SliceKind::range)
                            .Case("Slice_Length", asl::SliceKind::length)
                            .Case("Slice_Star", asl::SliceKind::star)
                            .Default(asl::SliceKind::single);
        IntegerAttr idx = nullptr, lhs = nullptr, rhs = nullptr,
                    start = nullptr, factor = nullptr, lengthAttr = nullptr;
        if (kindEnum == asl::SliceKind::single) {
          auto v = getIntFromLiteralExpr(sObj->get("expr"));
          if (v)
            idx = builder.getI64IntegerAttr(*v);
        } else if (kindEnum == asl::SliceKind::range) {
          auto l = getIntFromLiteralExpr(sObj->get("start"));
          auto r = getIntFromLiteralExpr(sObj->get("end"));
          if (l)
            lhs = builder.getI64IntegerAttr(*l);
          if (r)
            rhs = builder.getI64IntegerAttr(*r);
        } else if (kindEnum == asl::SliceKind::length) {
          auto stV = getIntFromLiteralExpr(sObj->get("start"));
          auto ln = getIntFromLiteralExpr(sObj->get("length"));
          if (stV)
            start = builder.getI64IntegerAttr(*stV);
          if (ln)
            lengthAttr = builder.getI64IntegerAttr(*ln);
        } else if (kindEnum == asl::SliceKind::star) {
          auto stV = getIntFromLiteralExpr(sObj->get("start"));
          auto step = getIntFromLiteralExpr(sObj->get("step"));
          if (stV)
            start = builder.getI64IntegerAttr(*stV);
          if (step)
            factor = builder.getI64IntegerAttr(*step);
        }
        return asl::SliceAttr::get(&ctx,
                                   asl::SliceKindAttr::get(&ctx, kindEnum), idx,
                                   lhs, rhs, start, factor, lengthAttr);
      };
      std::function<asl::BitFieldAttr(const llvm::json::Value &)>
          parseBitField =
              [&](const llvm::json::Value &bv) -> asl::BitFieldAttr {
        auto *bObj = bv.getAsObject();
        if (!bObj)
          return {};
        auto typeStr = bObj->getString("type");
        if (!typeStr)
          return {};
        auto nameStr = bObj->getString("name");
        if (!nameStr)
          return {};
        SmallVector<Attribute> sliceAttrs;
        if (auto *sv = bObj->get("slices"))
          if (auto *sArr = sv->getAsArray())
            for (auto &s : *sArr) {
              auto sa = parseSliceAttr(s);
              if (sa)
                sliceAttrs.push_back(sa);
            }
        auto slicesAttr = builder.getArrayAttr(sliceAttrs);
        asl::BitFieldKind kindEnum =
            llvm::StringSwitch<asl::BitFieldKind>(*typeStr)
                .Case("BitField_Simple", asl::BitFieldKind::simple)
                .Case("BitField_Nested", asl::BitFieldKind::nested)
                .Case("BitField_Type", asl::BitFieldKind::type)
                .Default(asl::BitFieldKind::simple);
        ArrayAttr nestedAttr = nullptr;
        TypeAttr typeAttr = nullptr;
        if (kindEnum == asl::BitFieldKind::nested) {
          SmallVector<Attribute> nestedVec;
          if (auto *nv = bObj->get("nested"))
            if (auto *nArr = nv->getAsArray())
              for (auto &nf : *nArr) {
                auto nbf = parseBitField(nf);
                if (nbf)
                  nestedVec.push_back(nbf);
              }
          nestedAttr = builder.getArrayAttr(nestedVec);
        } else if (kindEnum == asl::BitFieldKind::type) {
          if (auto *ftV = bObj->get("field_type")) {
            auto fty = parseType(*ftV);
            if (fty)
              typeAttr = TypeAttr::get(*fty);
          }
        }
        return asl::BitFieldAttr::get(
            &ctx, builder.getStringAttr(*nameStr),
            asl::BitFieldKindAttr::get(&ctx, kindEnum), slicesAttr, nestedAttr,
            typeAttr);
      };
      int64_t width = -1;
      if (auto *widthV = obj->get("width")) {
        if (auto *wObj = widthV->getAsObject()) {
          if (auto wType = wObj->getString("type")) {
            if (*wType == "E_Literal") {
              if (auto *lit = wObj->get("literal")) {
                if (auto *litObj = lit->getAsObject()) {
                  if (auto litType = litObj->getString("type");
                      litType && *litType == "L_Int") {
                    if (auto val = litObj->getString("value")) {
                      llvm::APInt ap;
                      if (!val->getAsInteger(10, ap))
                        width = ap.getSExtValue();
                    }
                  }
                }
              }
            }
          }
        }
      }
      SmallVector<Attribute> bitfieldAttrs;
      if (auto *bfV = obj->get("bitfields"))
        if (auto *arr = bfV->getAsArray())
          for (auto &bf : *arr) {
            auto bfa = parseBitField(bf);
            if (bfa)
              bitfieldAttrs.push_back(bfa);
          }
      auto bitfields = builder.getArrayAttr(bitfieldAttrs);
      return asl::BitsType::get(&ctx, builder.getI64IntegerAttr(width),
                                bitfields);
    }

    if (k == "T_Array") {
      auto *indexV = obj->get("index");
      auto *elemV = obj->get("element_type");
      if (!indexV || !elemV)
        return makeError("array type missing components");
      auto elemType = parseType(*elemV);
      if (!elemType)
        return elemType.takeError();
      int64_t length = -1;
      StringAttr enumName;
      ArrayAttr enumLabels;
      if (auto *idxObj = indexV->getAsObject()) {
        if (auto idxType = idxObj->getString("type")) {
          if (*idxType == "ArrayLength_Expr") {
            if (auto *exprV = idxObj->get("expr"))
              if (auto *eObj = exprV->getAsObject()) {
                if (auto etype = eObj->getString("type");
                    etype && *etype == "E_Literal") {
                  if (auto *litV = eObj->get("literal"))
                    if (auto *litObj = litV->getAsObject()) {
                      if (auto ltype = litObj->getString("type");
                          ltype && *ltype == "L_Int") {
                        if (auto sval = litObj->getString("value")) {
                          llvm::APInt ap;
                          if (!sval->getAsInteger(10, ap))
                            length = ap.getSExtValue();
                        }
                      }
                    }
                }
              }
          } else if (*idxType == "ArrayLength_Enum") {
            auto en = idxObj->getString("enum");
            if (en)
              enumName = builder.getStringAttr(*en);
            SmallVector<Attribute> labels;
            if (auto *labsV = idxObj->get("labels"))
              if (auto *arr = labsV->getAsArray())
                for (auto &lab : *arr)
                  if (auto ls = lab.getAsString())
                    labels.push_back(builder.getStringAttr(*ls));
            enumLabels = builder.getArrayAttr(labels);
            length = labels.size();
          }
        }
      }
      asl::ArrayIndexAttr idxAttr;
      if (enumName)
        idxAttr = asl::ArrayIndexAttr::get(
            &ctx,
            asl::ArrayIndexKindAttr::get(&ctx, asl::ArrayIndexKind::enum_type),
            builder.getI64IntegerAttr(length), enumName, enumLabels);
      else
        idxAttr = asl::ArrayIndexAttr::get(
            &ctx,
            asl::ArrayIndexKindAttr::get(&ctx, asl::ArrayIndexKind::int_type),
            builder.getI64IntegerAttr(length), nullptr, nullptr);
      return asl::ArrayType::get(&ctx, TypeAttr::get(*elemType), idxAttr);
    }

    if (k == "T_Tuple") {
      SmallVector<Attribute> typeAttrs;
      if (auto *typesV = obj->get("types"))
        if (auto *arr = typesV->getAsArray())
          for (auto &entry : *arr) {
            auto t = parseType(entry);
            if (!t)
              return t.takeError();
            typeAttrs.push_back(TypeAttr::get(*t));
          }
      return asl::TupleType::get(&ctx, builder.getArrayAttr(typeAttrs));
    }

    if (k == "T_Record" || k == "T_Exception" || k == "T_Collection") {
      SmallVector<Attribute> fields;
      if (auto *fieldsV = obj->get("fields"))
        if (auto *arr = fieldsV->getAsArray())
          for (auto &f : *arr)
            if (auto *fObj = f.getAsObject()) {
              auto fname = fObj->getString("name");
              auto ftypeV = fObj->get("field_type");
              if (!fname || !ftypeV)
                return makeError("record-like field missing components");
              auto ftype = parseType(*ftypeV);
              if (!ftype)
                return ftype.takeError();
              fields.push_back(asl::RecordFieldAttr::get(
                  &ctx, builder.getStringAttr(*fname), TypeAttr::get(*ftype)));
            }
      auto fieldArr = builder.getArrayAttr(fields);
      if (k == "T_Record")
        return asl::RecordType::get(&ctx, fieldArr);
      if (k == "T_Exception")
        return asl::ExceptionType::get(&ctx, fieldArr);
      return asl::CollectionType::get(&ctx, fieldArr);
    }

    if (k == "T_Enum") {
      SmallVector<Attribute> labels;
      if (auto *lV = obj->get("labels"))
        if (auto *arr = lV->getAsArray())
          for (auto &lab : *arr)
            if (auto ls = lab.getAsString())
              labels.push_back(builder.getStringAttr(*ls));
      return asl::EnumType::get(&ctx, builder.getArrayAttr(labels));
    }

    return makeError("unsupported type kind: ", k);
  }

  // Parse literal -> produce Value via ops
  llvm::Expected<Value> parseLiteral(const llvm::json::Object &obj) {
    auto ltype = obj.getString("type");
    if (!ltype)
      return makeError("literal missing type");
    StringRef t = *ltype;
    if (t == "L_Int") {
      auto valStr = obj.getString("value");
      if (!valStr)
        return makeError("int literal missing value");
      auto op = builder.create<asl::LiteralIntOp>(
          loc, getDefaultIntType(), builder.getStringAttr(*valStr));
      return op.getResult();
    } else if (t == "L_Bool") {
      bool b = false;
      if (auto val = obj.get("value"))
        if (auto ob = val->getAsBoolean())
          b = *ob;
      auto op = builder.create<asl::LiteralBoolOp>(loc, builder.getI1Type(),
                                                   builder.getBoolAttr(b));
      return op.getResult();
    } else if (t == "L_String") {
      auto valStr = obj.getString("value");
      if (!valStr)
        return makeError("string literal missing value");
      auto op = builder.create<asl::LiteralStringOp>(
          loc, asl::StringType::get(&ctx), builder.getStringAttr(*valStr));
      return op.getResult();
    } else if (t == "L_Real") {
      auto valStr = obj.getString("value");
      if (!valStr)
        return makeError("real literal missing value");
      auto op = builder.create<asl::LiteralRealOp>(
          loc, asl::RealType::get(&ctx), builder.getStringAttr(*valStr));
      return op.getResult();
    } else if (t == "L_BitVector") {
      auto valueStr = obj.getString("value");
      if (!valueStr)
        return makeError("bitvector literal missing value");
      auto bitsTy = asl::BitsType::get(&ctx, builder.getI64IntegerAttr(-1),
                                       builder.getArrayAttr({}));
      auto op = builder.create<asl::LiteralBitvectorOp>(
          loc, bitsTy, builder.getStringAttr(*valueStr));
      return op.getResult();
    } else if (t == "L_Label") {
      // Backend encodes label literal under key "value" not "name".
      auto val = obj.getString("value");
      if (!val)
        return makeError("label literal missing value");
      auto op = builder.create<asl::LiteralLabelOp>(
          loc, asl::LabelType::get(&ctx), builder.getStringAttr(*val));
      return op.getResult();
    }
    return makeError("unsupported literal kind: ", t);
  }

  // Parse pattern object from JSON
  llvm::Expected<Value> parsePattern(const llvm::json::Value &v) {
    ContextSetter guard(*this, &v);
    auto *obj = v.getAsObject();
    if (!obj)
      return makeError("pattern not object");
    auto kind = obj->getString("type");
    if (!kind)
      return makeError("pattern missing type");
    StringRef k = *kind;

    if (k == "Pattern_All") {
      // Pattern_All needs an expression to match against
      return makeError("Pattern_All not supported in this context");
    } else if (k == "Pattern_Any") {
      return makeError("Pattern_Any not supported in this context");
    } else if (k == "Pattern_Single") {
      auto *exprV = obj->get("expr");
      if (!exprV)
        return makeError("Pattern_Single missing expr");
      auto expr = parseExpr(*exprV);
      if (!expr)
        return expr.takeError();
      // This should probably be handled differently in a pattern context
      return makeError("Pattern_Single not supported as standalone value");
    } else if (k == "Pattern_Range") {
      return makeError("Pattern_Range not supported in this context");
    } else if (k == "Pattern_Geq") {
      return makeError("Pattern_Geq not supported in this context");
    } else if (k == "Pattern_Leq") {
      return makeError("Pattern_Leq not supported in this context");
    } else if (k == "Pattern_Mask") {
      return makeError("Pattern_Mask not supported in this context");
    } else if (k == "Pattern_Not") {
      return makeError("Pattern_Not not supported in this context");
    } else if (k == "Pattern_Tuple") {
      return makeError("Pattern_Tuple not supported in this context");
    }
    return makeError("unsupported pattern kind: ", k);
  }

  // Parse pattern object from JSON in the context of pattern matching
  llvm::Expected<Value> parsePatternInRegion(const llvm::json::Value &v,
                                             Value matchExpr) {
    ContextSetter guard(*this, &v);
    auto *obj = v.getAsObject();
    if (!obj)
      return makeError("pattern not object");
    auto kind = obj->getString("type");
    if (!kind)
      return makeError("pattern missing type");
    StringRef k = *kind;

    if (k == "Pattern_All") {
      auto op = builder.create<asl::PatternAllOp>(loc, builder.getI1Type(),
                                                  matchExpr);
      return op.getResult();
    } else if (k == "Pattern_Any") {
      // Create PatternAnyOp with nested patterns in a region
      auto anyOp = builder.create<asl::PatternAnyOp>(loc, builder.getI1Type(),
                                                     matchExpr);
      Region &patternsRegion = anyOp.getPatterns();
      patternsRegion.push_back(new Block);

      auto prevInsertionPoint = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(&patternsRegion.back());

      if (auto *patternsV = obj->get("patterns")) {
        if (auto *arr = patternsV->getAsArray()) {
          for (auto &p : *arr) {
            auto patternResult = parsePatternInRegion(p, matchExpr);
            if (!patternResult) {
              builder.restoreInsertionPoint(prevInsertionPoint);
              return patternResult.takeError();
            }
          }
        }
      }

      builder.restoreInsertionPoint(prevInsertionPoint);
      return anyOp.getResult();
    } else if (k == "Pattern_Single") {
      auto *exprV = obj->get("expr");
      if (!exprV)
        return makeError("Pattern_Single missing expr");
      auto expr = parseExpr(*exprV);
      if (!expr)
        return expr.takeError();
      auto op = builder.create<asl::PatternSingleOp>(loc, builder.getI1Type(),
                                                     matchExpr, *expr);
      return op.getResult();
    } else if (k == "Pattern_Range") {
      auto *startV = obj->get("start");
      auto *endV = obj->get("end");
      if (!startV || !endV)
        return makeError("Pattern_Range missing bounds");
      auto start = parseExpr(*startV);
      if (!start)
        return start.takeError();
      auto end = parseExpr(*endV);
      if (!end)
        return end.takeError();
      auto op = builder.create<asl::PatternRangeOp>(loc, builder.getI1Type(),
                                                    matchExpr, *start, *end);
      return op.getResult();
    } else if (k == "Pattern_Geq") {
      auto *boundV = obj->get("expr");
      if (!boundV)
        return makeError("Pattern_Geq missing expr");
      auto bound = parseExpr(*boundV);
      if (!bound)
        return bound.takeError();
      auto op = builder.create<asl::PatternGeqOp>(loc, builder.getI1Type(),
                                                  matchExpr, *bound);
      return op.getResult();
    } else if (k == "Pattern_Leq") {
      auto *boundV = obj->get("expr");
      if (!boundV)
        return makeError("Pattern_Leq missing expr");
      auto bound = parseExpr(*boundV);
      if (!bound)
        return bound.takeError();
      auto op = builder.create<asl::PatternLeqOp>(loc, builder.getI1Type(),
                                                  matchExpr, *bound);
      return op.getResult();
    } else if (k == "Pattern_Mask") {
      auto maskStr = obj->getString("mask");
      if (!maskStr)
        return makeError("Pattern_Mask missing mask");
      auto maskAttr =
          asl::BitVectorMaskAttr::get(&ctx, builder.getStringAttr(*maskStr));
      auto op = builder.create<asl::PatternMaskOp>(loc, builder.getI1Type(),
                                                   matchExpr, maskAttr);
      return op.getResult();
    } else if (k == "Pattern_Not") {
      auto *patternV = obj->get("pattern");
      if (!patternV)
        return makeError("Pattern_Not missing pattern");
      auto pattern = parsePatternInRegion(*patternV, matchExpr);
      if (!pattern)
        return pattern.takeError();
      auto op =
          builder.create<asl::PatternNotOp>(loc, builder.getI1Type(), *pattern);
      return op.getResult();
    } else if (k == "Pattern_Tuple") {
      // PatternTupleOp uses regions, not operands for patterns
      auto tupleOp = builder.create<asl::PatternTupleOp>(
          loc, builder.getI1Type(), matchExpr);
      Region &patternsRegion = tupleOp.getPatterns();
      patternsRegion.push_back(new Block);

      auto prevInsertionPoint = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(&patternsRegion.back());

      if (auto *patternsV = obj->get("patterns")) {
        if (auto *arr = patternsV->getAsArray()) {
          for (auto &p : *arr) {
            auto patternResult = parsePatternInRegion(p, matchExpr);
            if (!patternResult) {
              builder.restoreInsertionPoint(prevInsertionPoint);
              return patternResult.takeError();
            }
            // Pattern is built in the region
          }
        }
      }

      builder.restoreInsertionPoint(prevInsertionPoint);
      return tupleOp.getResult();
    }
    return makeError("unsupported pattern kind: ", k);
  }

  // Parse l-expression object from JSON
  llvm::Expected<Value> parseLExpr(const llvm::json::Value &v) {
    ContextSetter guard(*this, &v);
    auto *obj = v.getAsObject();
    if (!obj)
      return makeError("lexpr not object");
    auto kind = obj->getString("type");
    if (!kind)
      return makeError("lexpr missing type");
    StringRef k = *kind;

    if (k == "LE_Discard") {
      auto op =
          builder.create<asl::LExprDiscardOp>(loc, asl::LExprType::get(&ctx));
      return op.getResult();
    } else if (k == "LE_Var") {
      auto name = obj->getString("name");
      if (!name)
        return makeError("LE_Var missing name");
      auto op = builder.create<asl::LExprVarOp>(loc, asl::LExprType::get(&ctx),
                                                builder.getStringAttr(*name));
      return op.getResult();
    } else if (k == "LE_Slice") {
      auto *baseV = obj->get("lexpr");
      if (!baseV)
        return makeError("LE_Slice missing lexpr");
      auto base = parseLExpr(*baseV);
      if (!base)
        return base.takeError();
      SmallVector<Value> sliceVals;
      if (auto *sv = obj->get("slices")) {
        if (auto *arr = sv->getAsArray()) {
          for (auto &s : *arr) {
            auto *sObj = s.getAsObject();
            if (!sObj)
              return makeError("slice entry not object");
            auto st = sObj->getString("type");
            if (!st)
              return makeError("slice missing type");
            if (*st == "Slice_Single") {
              auto *eV = sObj->get("expr");
              if (!eV)
                return makeError("Slice_Single missing expr");
              auto e = parseExpr(*eV);
              if (!e)
                return e.takeError();
              auto sOp = builder.create<asl::SliceSingleOp>(
                  loc, asl::SliceType::get(&ctx), *e);
              sliceVals.push_back(sOp.getResult());
            } else if (*st == "Slice_Range") {
              auto *startV = sObj->get("start");
              auto *endV = sObj->get("end");
              if (!startV || !endV)
                return makeError("Slice_Range missing bounds");
              auto start = parseExpr(*startV);
              if (!start)
                return start.takeError();
              auto end = parseExpr(*endV);
              if (!end)
                return end.takeError();
              auto sOp = builder.create<asl::SliceRangeOp>(
                  loc, asl::SliceType::get(&ctx), *start, *end);
              sliceVals.push_back(sOp.getResult());
            } else if (*st == "Slice_Length") {
              auto *startV = sObj->get("start");
              auto *lenV = sObj->get("length");
              if (!startV || !lenV)
                return makeError("Slice_Length missing components");
              auto start = parseExpr(*startV);
              if (!start)
                return start.takeError();
              auto len = parseExpr(*lenV);
              if (!len)
                return len.takeError();
              auto sOp = builder.create<asl::SliceLengthOp>(
                  loc, asl::SliceType::get(&ctx), *start, *len);
              sliceVals.push_back(sOp.getResult());
            } else if (*st == "Slice_Star") {
              auto *startV = sObj->get("start");
              auto *step = sObj->get("step");
              if (!startV || !step)
                return makeError("Slice_Star missing components");
              auto start = parseExpr(*startV);
              if (!start)
                return start.takeError();
              auto stepVal = parseExpr(*step);
              if (!stepVal)
                return stepVal.takeError();
              auto sOp = builder.create<asl::SliceStarOp>(
                  loc, asl::SliceType::get(&ctx), *start, *stepVal);
              sliceVals.push_back(sOp.getResult());
            } else {
              return makeError("unknown slice variant: ", *st);
            }
          }
        }
      }
      auto op = builder.create<asl::LExprSliceOp>(
          loc, asl::LExprType::get(&ctx), *base, ValueRange(sliceVals));
      return op.getResult();
    } else if (k == "LE_SetArray") {
      auto *baseV = obj->get("lexpr");
      auto *indexV = obj->get("index");
      if (!baseV || !indexV)
        return makeError("LE_SetArray missing components");
      auto base = parseLExpr(*baseV);
      if (!base)
        return base.takeError();
      auto index = parseExpr(*indexV);
      if (!index)
        return index.takeError();
      auto op = builder.create<asl::LExprSetArrayOp>(
          loc, asl::LExprType::get(&ctx), *base, *index);
      return op.getResult();
    } else if (k == "LE_SetEnumArray") {
      auto *baseV = obj->get("lexpr");
      auto *indexV = obj->get("index");
      if (!baseV || !indexV)
        return makeError("LE_SetEnumArray missing components");
      auto base = parseLExpr(*baseV);
      if (!base)
        return base.takeError();
      auto index = parseExpr(*indexV);
      if (!index)
        return index.takeError();
      auto op = builder.create<asl::LExprSetEnumArrayOp>(
          loc, asl::LExprType::get(&ctx), *base, *index);
      return op.getResult();
    } else if (k == "LE_SetField") {
      auto *baseV = obj->get("lexpr");
      auto fieldName = obj->getString("field");
      if (!baseV || !fieldName)
        return makeError("LE_SetField missing components");
      auto base = parseLExpr(*baseV);
      if (!base)
        return base.takeError();
      auto op = builder.create<asl::LExprSetFieldOp>(
          loc, asl::LExprType::get(&ctx), *base,
          builder.getStringAttr(*fieldName));
      return op.getResult();
    } else if (k == "LE_SetFields") {
      auto *baseV = obj->get("lexpr");
      if (!baseV)
        return makeError("LE_SetFields missing lexpr");
      auto base = parseLExpr(*baseV);
      if (!base)
        return base.takeError();
      SmallVector<Attribute> fieldNames;
      SmallVector<Attribute> annotations;
      if (auto *fieldsV = obj->get("fields")) {
        if (auto *arr = fieldsV->getAsArray()) {
          for (auto &f : *arr) {
            if (auto fs = f.getAsString())
              fieldNames.push_back(builder.getStringAttr(*fs));
          }
        }
      }
      if (auto *annotV = obj->get("annotations")) {
        if (auto *arr = annotV->getAsArray()) {
          for (auto &a : *arr) {
            if (auto *aObj = a.getAsObject()) {
              auto xVal = aObj->getInteger("x");
              auto yVal = aObj->getInteger("y");
              if (xVal && yVal) {
                auto annot = builder.getDictionaryAttr(
                    {{builder.getStringAttr("x"),
                      builder.getI32IntegerAttr(*xVal)},
                     {builder.getStringAttr("y"),
                      builder.getI32IntegerAttr(*yVal)}});
                annotations.push_back(annot);
              }
            }
          }
        }
      }
      auto op = builder.create<asl::LExprSetFieldsOp>(
          loc, asl::LExprType::get(&ctx), *base,
          builder.getArrayAttr(fieldNames), builder.getArrayAttr(annotations));
      return op.getResult();
    } else if (k == "LE_SetCollectionFields") {
      auto collection = obj->getString("collection");
      if (!collection)
        return makeError("LE_SetCollectionFields missing collection");
      SmallVector<Attribute> fieldNames;
      SmallVector<Attribute> annotations;
      if (auto *fieldsV = obj->get("fields")) {
        if (auto *arr = fieldsV->getAsArray()) {
          for (auto &f : *arr) {
            if (auto fs = f.getAsString())
              fieldNames.push_back(builder.getStringAttr(*fs));
          }
        }
      }
      if (auto *annotV = obj->get("annotations")) {
        if (auto *arr = annotV->getAsArray()) {
          for (auto &a : *arr) {
            if (auto *aObj = a.getAsObject()) {
              auto xVal = aObj->getInteger("x");
              auto yVal = aObj->getInteger("y");
              if (xVal && yVal) {
                auto annot = builder.getDictionaryAttr(
                    {{builder.getStringAttr("x"),
                      builder.getI32IntegerAttr(*xVal)},
                     {builder.getStringAttr("y"),
                      builder.getI32IntegerAttr(*yVal)}});
                annotations.push_back(annot);
              }
            }
          }
        }
      }
      auto op = builder.create<asl::LExprSetCollectionFieldsOp>(
          loc, asl::LExprType::get(&ctx), builder.getStringAttr(*collection),
          builder.getArrayAttr(fieldNames), builder.getArrayAttr(annotations));
      return op.getResult();
    } else if (k == "LE_Destructuring") {
      SmallVector<Value> lexprs;
      if (auto *lexprsV = obj->get("lexprs")) {
        if (auto *arr = lexprsV->getAsArray()) {
          for (auto &le : *arr) {
            auto lexpr = parseLExpr(le);
            if (!lexpr)
              return lexpr.takeError();
            lexprs.push_back(*lexpr);
          }
        }
      }
      auto op = builder.create<asl::LExprDestructuringOp>(
          loc, asl::LExprType::get(&ctx), ValueRange(lexprs));
      return op.getResult();
    }
    return makeError("unsupported lexpr kind: ", k);
  }

  llvm::Expected<Value> parseExpr(const llvm::json::Value &v) {
    ContextSetter guard(*this, &v);
    auto *obj = v.getAsObject();
    if (!obj)
      return makeError("expr not object");
    auto kind = obj->getString("type");
    if (!kind)
      return makeError("expr missing type");
    StringRef k = *kind;
    if (k == "E_Literal") {
      auto *litV = obj->get("literal");
      if (!litV)
        return makeError("literal expr missing literal");
      auto *litObj = litV->getAsObject();
      if (!litObj)
        return makeError("literal node not object");
      return parseLiteral(*litObj);
    } else if (k == "E_Var") {
      auto name = obj->getString("name");
      if (!name)
        return makeError("var expr missing name");
      if (Value existing = lookup(*name))
        return existing;
      auto op = builder.create<asl::VarOp>(loc, getDefaultIntType(),
                                           builder.getStringAttr(*name));
      bind(*name, op.getResult());
      return op.getResult();
    } else if (k == "E_Binop") {
      auto opName = obj->getString("op");
      if (!opName)
        return makeError("binop missing op");
      auto *leftV = obj->get("left");
      auto *rightV = obj->get("right");
      if (!leftV || !rightV)
        return makeError("binop missing operands");
      auto lhs = parseExpr(*leftV);
      if (!lhs)
        return lhs.takeError();
      auto rhs = parseExpr(*rightV);
      if (!rhs)
        return rhs.takeError();
      Value result;
      StringRef bop = *opName;
      auto intTy = getDefaultIntType();
      // Infer a reasonable result type for polymorphic binops.
      Type lhsTy = (*lhs).getType();
      Type rhsTy = (*rhs).getType();
      auto chooseSameOrPromote = [&]() -> Type {
        // If operand types match, keep that type.
        if (lhsTy == rhsTy)
          return lhsTy;
        // If any operand is real, promote to real.
        if (llvm::isa<asl::RealType>(lhsTy) || llvm::isa<asl::RealType>(rhsTy))
          return asl::RealType::get(&ctx);
        // If both operands are bits, keep bits (fallback to lhs type).
        if (llvm::isa<asl::BitsType>(lhsTy) && llvm::isa<asl::BitsType>(rhsTy))
          return lhsTy;
        // Default to unconstrained int.
        return intTy;
      };
#define BUILD_BIN(OPCLS, RTYPE)                                                \
  result = builder.create<asl::OPCLS>(loc, RTYPE, *lhs, *rhs).getResult();     \
  break;
      do {
        if (bop == "PLUS") {
          BUILD_BIN(BinopPlusOp, chooseSameOrPromote());
        }
        if (bop == "MINUS") {
          BUILD_BIN(BinopMinusOp, chooseSameOrPromote());
        }
        if (bop == "MUL") {
          BUILD_BIN(BinopMulOp, chooseSameOrPromote());
        }
        if (bop == "DIV") {
          BUILD_BIN(BinopDivOp, intTy);
        }
        if (bop == "DIVRM") {
          BUILD_BIN(BinopDivrmOp, intTy);
        }
        if (bop == "MOD") {
          BUILD_BIN(BinopModOp, intTy);
        }
        if (bop == "POW") {
          BUILD_BIN(BinopPowOp, intTy);
        }
        if (bop == "SHL") {
          BUILD_BIN(BinopShlOp, intTy);
        }
        if (bop == "SHR") {
          BUILD_BIN(BinopShrOp, intTy);
        }
        if (bop == "AND") {
          // Bitwise AND is defined over bitvectors; result must be BitsType.
          BUILD_BIN(BinopAndOp, getDefaultBitsType());
        }
        if (bop == "OR") {
          // Bitwise OR is defined over bitvectors; result must be BitsType.
          BUILD_BIN(BinopOrOp, getDefaultBitsType());
        }
        if (bop == "XOR") {
          // Bitwise XOR is defined over bitvectors; result must be BitsType.
          BUILD_BIN(BinopXorOp, getDefaultBitsType());
        }
        if (bop == "BAND") {
          BUILD_BIN(BinopBandOp, builder.getI1Type());
        }
        if (bop == "BOR") {
          BUILD_BIN(BinopBorOp, builder.getI1Type());
        }
        if (bop == "BEQ") {
          BUILD_BIN(BinopBeqOp, builder.getI1Type());
        }
        if (bop == "EQ_OP") {
          BUILD_BIN(BinopEqOp, builder.getI1Type());
        }
        if (bop == "NEQ") {
          BUILD_BIN(BinopNeqOp, builder.getI1Type());
        }
        if (bop == "GT") {
          BUILD_BIN(BinopGtOp, builder.getI1Type());
        }
        if (bop == "GEQ") {
          BUILD_BIN(BinopGeqOp, builder.getI1Type());
        }
        if (bop == "LT") {
          BUILD_BIN(BinopLtOp, builder.getI1Type());
        }
        if (bop == "LEQ") {
          BUILD_BIN(BinopLeqOp, builder.getI1Type());
        }
        if (bop == "IMPL") {
          BUILD_BIN(BinopImplOp, builder.getI1Type());
        }
        if (bop == "RDIV") {
          BUILD_BIN(BinopRdivOp, asl::RealType::get(&ctx));
        }
        if (bop == "CONCAT") {
          BUILD_BIN(BinopConcatOp, chooseSameOrPromote());
        }
        return makeError("unsupported binop: ", bop);
      } while (false);
#undef BUILD_BIN
      return result;
    } else if (k == "E_Unop") {
      auto opName = obj->getString("op");
      if (!opName)
        return makeError("unop missing op");
      auto sub = obj->get("expr");
      if (!sub)
        return makeError("unop missing expr");
      auto val = parseExpr(*sub);
      if (!val)
        return val.takeError();
      StringRef uop = *opName;
      if (uop == "NEG") {
        return builder.create<asl::UnopNegOp>(loc, getDefaultIntType(), *val)
            .getResult();
      } else if (uop == "BNOT") {
        return builder.create<asl::UnopBnotOp>(loc, builder.getI1Type(), *val)
            .getResult();
      } else if (uop == "NOT") {
        // NOT is defined over bitvectors (see ASL_UnopNotOp in
        // ASLExpressions.td) Previously this incorrectly used an IntType result
        // causing a failed cast to BitsType during op construction.
        return builder.create<asl::UnopNotOp>(loc, getDefaultBitsType(), *val)
            .getResult();
      }
      return makeError("unsupported unop: ", uop);
    } else if (k == "E_Slice") {
      auto *baseV = obj->get("expr");
      if (!baseV)
        return makeError("slice missing expr");
      auto base = parseExpr(*baseV);
      if (!base)
        return base.takeError();
      SmallVector<Value> sliceVals;
      if (auto *sv = obj->get("slices")) {
        if (auto *arr = sv->getAsArray()) {
          for (auto &s : *arr) {
            auto *sObj = s.getAsObject();
            if (!sObj)
              return makeError("slice entry not object");
            auto st = sObj->getString("type");
            if (!st)
              return makeError("slice missing type");
            if (*st == "Slice_Single") {
              auto *eV = sObj->get("expr");
              if (!eV)
                return makeError("Slice_Single missing expr");
              auto e = parseExpr(*eV);
              if (!e)
                return e.takeError();
              auto sOp = builder.create<asl::SliceSingleOp>(
                  loc, asl::SliceType::get(&ctx), *e);
              sliceVals.push_back(sOp.getResult());
            } else if (*st == "Slice_Range") {
              auto *startV = sObj->get("start");
              auto *endV = sObj->get("end");
              if (!startV || !endV)
                return makeError("Slice_Range missing bounds");
              auto start = parseExpr(*startV);
              if (!start)
                return start.takeError();
              auto end = parseExpr(*endV);
              if (!end)
                return end.takeError();
              auto sOp = builder.create<asl::SliceRangeOp>(
                  loc, asl::SliceType::get(&ctx), *start, *end);
              sliceVals.push_back(sOp.getResult());
            } else if (*st == "Slice_Length") {
              auto *startV = sObj->get("start");
              auto *lenV = sObj->get("length");
              if (!startV || !lenV)
                return makeError("Slice_Length missing components");
              auto start = parseExpr(*startV);
              if (!start)
                return start.takeError();
              auto len = parseExpr(*lenV);
              if (!len)
                return len.takeError();
              auto sOp = builder.create<asl::SliceLengthOp>(
                  loc, asl::SliceType::get(&ctx), *start, *len);
              sliceVals.push_back(sOp.getResult());
            } else if (*st == "Slice_Star") {
              auto *startV = sObj->get("start");
              auto *stepV = sObj->get("step");
              if (!startV || !stepV)
                return makeError("Slice_Star missing components");
              auto start = parseExpr(*startV);
              if (!start)
                return start.takeError();
              auto step = parseExpr(*stepV);
              if (!step)
                return step.takeError();
              auto sOp = builder.create<asl::SliceStarOp>(
                  loc, asl::SliceType::get(&ctx), *start, *step);
              sliceVals.push_back(sOp.getResult());
            } else {
              return makeError("unknown slice variant: ", *st);
            }
          }
        }
      }
      auto sliceOp = builder.create<asl::SliceOp>(loc, getDefaultBitsType(),
                                                  *base, ValueRange(sliceVals));
      return sliceOp.getResult();
    } else if (k == "E_Call") {
      auto *callV = obj->get("call");
      if (!callV)
        return makeError("call expr missing call object");
      auto *cObj = callV->getAsObject();
      if (!cObj)
        return makeError("call object not obj");
      auto name = cObj->getString("name");
      if (!name)
        return makeError("call missing name");
      SmallVector<Value> args;
      int paramsSize = 0;
      if (auto *paramsV = cObj->get("params"))
        if (auto *arr = paramsV->getAsArray()) {
          for (auto &p : *arr) {
            auto pv = parseExpr(p);
            if (!pv)
              return pv.takeError();
            args.push_back(*pv);
            ++paramsSize;
          }
        }
      if (auto *argsV = cObj->get("args"))
        if (auto *arr = argsV->getAsArray())
          for (auto &a : *arr) {
            auto av = parseExpr(a);
            if (!av)
              return av.takeError();
            args.push_back(*av);
          }
      auto callTypeStr = cObj->getString("call_type");
      SubprogramType callType = SubprogramType::procedure;
      if (callTypeStr)
        callType = mapSubprogramType(*callTypeStr);
      auto callOp = builder.create<asl::CallOp>(
          loc, getDefaultIntType(), builder.getStringAttr(*name),
          ValueRange(args), builder.getI32IntegerAttr(paramsSize),
          asl::SubprogramTypeAttr::get(&ctx, callType));
      return callOp.getResult();
    } else if (k == "E_Cond") {
      auto *condV = obj->get("condition");
      auto *trueV = obj->get("then_expr");
      auto *falseV = obj->get("else_expr");
      if (!condV || !trueV || !falseV)
        return makeError("cond expression missing components");
      auto cond = parseExpr(*condV);
      if (!cond)
        return cond.takeError();
      auto trueVal = parseExpr(*trueV);
      if (!trueVal)
        return trueVal.takeError();
      auto falseVal = parseExpr(*falseV);
      if (!falseVal)
        return falseVal.takeError();
      auto op = builder.create<asl::CondOp>(loc, getDefaultIntType(), *cond,
                                            *trueVal, *falseVal);
      return op.getResult();
    } else if (k == "E_ATC") {
      auto *exprV = obj->get("expr");
      auto *typeV = obj->get("target_type");
      if (!exprV || !typeV)
        return makeError("E_ATC missing components");
      auto expr = parseExpr(*exprV);
      if (!expr)
        return expr.takeError();
      auto type = parseType(*typeV);
      if (!type)
        return type.takeError();

      // Handle specialized ATC for integer types with constraints
      if (auto intTy = dyn_cast<asl::IntType>(*type)) {
        // Parse constraints directly from the JSON type to handle expressions
        if (auto *tObj = typeV->getAsObject()) {
          if (auto *cV = tObj->get("constraint_kind")) {
            if (auto *cObj = cV->getAsObject()) {
              auto cKind = cObj->getString("type");

              // Handle Parameterized types - these don't have constraints array
              if (cKind && *cKind == "Parameterized") {
                // For parameterized types, use generic ATC since they reference
                // parameters
                auto op = builder.create<asl::AtcOp>(loc, *type, *expr,
                                                     TypeAttr::get(*type));
                return op.getResult();
              }

              // Handle WellConstrained types - these have a constraints array
              if (cKind && *cKind == "WellConstrained") {
                if (auto *listV = cObj->get("constraints")) {
                  if (auto *arr = listV->getAsArray()) {
                    for (auto &cEntry : *arr) {
                      if (auto *ceObj = cEntry.getAsObject()) {
                        auto ceType = ceObj->getString("type");
                        if (!ceType)
                          continue;

                        if (*ceType == "Constraint_Exact") {
                          if (auto *eV = ceObj->get("expr")) {
                            auto exactExpr = parseExpr(*eV);
                            if (exactExpr) {
                              return builder
                                  .create<asl::AtcIntExactOp>(loc, intTy, *expr,
                                                              *exactExpr)
                                  .getResult();
                            }
                          }
                        } else if (*ceType == "Constraint_Range") {
                          if (auto *startV = ceObj->get("start")) {
                            if (auto *endV = ceObj->get("end")) {
                              // Check if any bound contains non-literal
                              // expressions
                              auto hasNonLiteralBounds =
                                  [](const llvm::json::Value *v) {
                                    if (auto *obj = v->getAsObject()) {
                                      auto type = obj->getString("type");
                                      return type && *type != "E_Literal";
                                    }
                                    return false;
                                  };

                              // Use specialized range ATC if ANY bound involves
                              // parameter expressions Only use generic ATC when
                              // ALL bounds are literal constants
                              bool startNonLiteral =
                                  hasNonLiteralBounds(startV);
                              bool endNonLiteral = hasNonLiteralBounds(endV);

                              if (startNonLiteral || endNonLiteral) {
                                // At least one bound is non-literal, use
                                // specialized range ATC
                                auto startExpr = parseExpr(*startV);
                                auto endExpr = parseExpr(*endV);
                                if (startExpr && endExpr) {
                                  return builder
                                      .create<asl::AtcIntRangeOp>(
                                          loc, intTy, *expr, *startExpr,
                                          *endExpr)
                                      .getResult();
                                }
                              }
                              // If both bounds are literals, fall through to
                              // generic ATC
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Prefer specialized ATC for bits only when the width is a non-literal
      // expression. If width is a literal, directly use generic ATC with the
      // attribute-encoded BitsType.
      if (auto bitsTy = dyn_cast<asl::BitsType>(*type)) {
        Value widthSSA;
        if (auto *tObj = typeV->getAsObject()) {
          if (auto tk = tObj->getString("type"); tk && *tk == "T_Bits") {
            if (auto *wV = tObj->get("width")) {
              if (auto *wObj = wV->getAsObject()) {
                auto wKind = wObj->getString("type");
                // Only parse to SSA if not a literal.
                if (!(wKind && *wKind == "E_Literal")) {
                  auto w = parseExpr(*wV);
                  if (!w)
                    return w.takeError();
                  widthSSA = *w;
                }
              }
            }
          }
        }
        if (widthSSA) {
          // For now, use the static bitfields from the type
          // TODO: Implement dynamic bitfield materialization when operations
          // support it
          return builder
              .create<asl::AtcBitsOp>(loc, bitsTy, *expr, widthSSA,
                                      bitsTy.getBitfields())
              .getResult();
        }
        auto op =
            builder.create<asl::AtcOp>(loc, *type, *expr, TypeAttr::get(*type));
        return op.getResult();
      }
      // Prefer specialized ATC for arrays only when the length is a non-literal
      // expression. If length is a literal or enum-sized, use generic ATC.
      if (auto arrayTy = dyn_cast<asl::ArrayType>(*type)) {
        Value lengthSSA;
        if (auto *tObj = typeV->getAsObject()) {
          if (auto tk = tObj->getString("type"); tk && *tk == "T_Array") {
            if (auto *idxV = tObj->get("index")) {
              if (auto *idxObj = idxV->getAsObject()) {
                if (auto idxType = idxObj->getString("type")) {
                  if (*idxType == "ArrayLength_Expr") {
                    if (auto *eV = idxObj->get("expr")) {
                      if (auto *eObj = eV->getAsObject()) {
                        auto eKind = eObj->getString("type");
                        if (!(eKind && *eKind == "E_Literal")) {
                          auto len = parseExpr(*eV);
                          if (!len)
                            return len.takeError();
                          lengthSSA = *len;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        if (lengthSSA) {
          // Current AtcArrayOp doesn't support element type materialization
          // Use the operation as defined
          return builder.create<asl::AtcArrayOp>(loc, arrayTy, *expr, lengthSSA)
              .getResult();
        }
        auto op =
            builder.create<asl::AtcOp>(loc, *type, *expr, TypeAttr::get(*type));
        return op.getResult();
      }
      auto op =
          builder.create<asl::AtcOp>(loc, *type, *expr, TypeAttr::get(*type));
      return op.getResult();
    } else if (k == "E_GetArray") {
      // Accept 'base', 'array', and legacy 'expr' for the base expression.
      const llvm::json::Value *baseV = obj->get("base");
      if (!baseV)
        baseV = obj->get("array");
      if (!baseV)
        baseV = obj->get("expr");
      const llvm::json::Value *indexV = obj->get("index");
      if (!baseV || !indexV)
        return makeError("E_GetArray missing components");
      auto base = parseExpr(*baseV);
      if (!base)
        return base.takeError();
      auto index = parseExpr(*indexV);
      if (!index)
        return index.takeError();
      auto op = builder.create<asl::GetArrayOp>(loc, getDefaultIntType(), *base,
                                                *index);
      return op.getResult();
    } else if (k == "E_GetEnumArray") {
      // Accept 'base', 'array', and legacy 'expr' for the base expression.
      const llvm::json::Value *baseV = obj->get("base");
      if (!baseV)
        baseV = obj->get("array");
      if (!baseV)
        baseV = obj->get("expr");
      // Accept both 'index' and 'enum_value' for the index expression.
      const llvm::json::Value *indexV = obj->get("index");
      if (!indexV)
        indexV = obj->get("enum_value");
      if (!baseV || !indexV)
        return makeError("E_GetEnumArray missing components");
      auto base = parseExpr(*baseV);
      if (!base)
        return base.takeError();
      auto index = parseExpr(*indexV);
      if (!index)
        return index.takeError();
      auto op = builder.create<asl::GetEnumArrayOp>(loc, getDefaultIntType(),
                                                    *base, *index);
      return op.getResult();
    } else if (k == "E_GetField") {
      // Accept both 'base' and legacy 'expr' for the base expression.
      const llvm::json::Value *baseV = obj->get("base");
      if (!baseV)
        baseV = obj->get("expr");
      auto fieldName = obj->getString("field");
      if (!baseV || !fieldName)
        return makeError("E_GetField missing components");
      auto base = parseExpr(*baseV);
      if (!base)
        return base.takeError();
      auto op = builder.create<asl::GetFieldOp>(
          loc, getDefaultIntType(), *base, builder.getStringAttr(*fieldName));
      return op.getResult();
    } else if (k == "E_GetFields") {
      // Accept both 'base' and legacy 'expr' for the base expression.
      const llvm::json::Value *baseV = obj->get("base");
      if (!baseV)
        baseV = obj->get("expr");
      if (!baseV)
        return makeError("E_GetFields missing base");
      auto base = parseExpr(*baseV);
      if (!base)
        return base.takeError();
      SmallVector<Attribute> fieldNames;
      if (auto *fieldsV = obj->get("fields")) {
        if (auto *arr = fieldsV->getAsArray()) {
          for (auto &f : *arr) {
            if (auto fs = f.getAsString())
              fieldNames.push_back(builder.getStringAttr(*fs));
          }
        }
      }
      auto op = builder.create<asl::GetFieldsOp>(
          loc, getDefaultBitsType(), *base, builder.getArrayAttr(fieldNames));
      return op.getResult();
    } else if (k == "E_GetCollectionFields") {
      auto collection = obj->getString("collection");
      if (!collection)
        return makeError("E_GetCollectionFields missing collection");
      SmallVector<Attribute> fieldNames;
      if (auto *fieldsV = obj->get("fields")) {
        if (auto *arr = fieldsV->getAsArray()) {
          for (auto &f : *arr) {
            if (auto fs = f.getAsString())
              fieldNames.push_back(builder.getStringAttr(*fs));
          }
        }
      }
      auto op = builder.create<asl::GetCollectionFieldsOp>(
          loc, getDefaultBitsType(), builder.getStringAttr(*collection),
          builder.getArrayAttr(fieldNames));
      return op.getResult();
    } else if (k == "E_GetItem") {
      // Accept both 'base' and legacy 'expr'/'tuple' for the base expression.
      const llvm::json::Value *baseV = obj->get("base");
      if (!baseV)
        baseV = obj->get("expr");
      if (!baseV)
        baseV = obj->get("tuple");
      auto itemIndex = obj->getInteger("index");
      if (!baseV || !itemIndex)
        return makeError("E_GetItem missing components");
      auto base = parseExpr(*baseV);
      if (!base)
        return base.takeError();
      auto op =
          builder.create<asl::GetItemOp>(loc, getDefaultIntType(), *base,
                                         builder.getI32IntegerAttr(*itemIndex));
      return op.getResult();
    } else if (k == "E_Record") {
      auto *typeV = obj->get("record_type");
      if (!typeV)
        return makeError("E_Record missing record_type");
      auto recordType = parseType(*typeV);
      if (!recordType)
        return recordType.takeError();
      SmallVector<Value> values;
      SmallVector<Attribute> fieldNames;
      if (auto *fieldsV = obj->get("fields")) {
        if (auto *arr = fieldsV->getAsArray()) {
          for (auto &f : *arr) {
            if (auto *fObj = f.getAsObject()) {
              auto name = fObj->getString("name");
              auto *valueV = fObj->get("value");
              if (name && valueV) {
                fieldNames.push_back(builder.getStringAttr(*name));
                auto value = parseExpr(*valueV);
                if (!value)
                  return value.takeError();
                values.push_back(*value);
              }
            }
          }
        }
      }
      auto op = builder.create<asl::RecordOp>(
          loc, *recordType, TypeAttr::get(*recordType), ValueRange(values),
          builder.getArrayAttr(fieldNames));
      return op.getResult();
    } else if (k == "E_Tuple") {
      SmallVector<Value> values;
      if (auto *elementsV = obj->get("elements")) {
        if (auto *arr = elementsV->getAsArray()) {
          for (auto &e : *arr) {
            auto value = parseExpr(e);
            if (!value)
              return value.takeError();
            values.push_back(*value);
          }
        }
      }
      SmallVector<Type> types;
      SmallVector<Attribute> typeAttrs;
      for (auto v : values) {
        types.push_back(v.getType());
        typeAttrs.push_back(TypeAttr::get(v.getType()));
      }
      auto tupleType =
          asl::TupleType::get(&ctx, builder.getArrayAttr(typeAttrs));
      auto op =
          builder.create<asl::TupleOp>(loc, tupleType, ValueRange(values));
      return op.getResult();
    } else if (k == "E_Array") {
      auto *lengthV = obj->get("length");
      auto *valueV = obj->get("value");
      if (!lengthV || !valueV)
        return makeError("E_Array missing components");
      auto length = parseExpr(*lengthV);
      if (!length)
        return length.takeError();
      auto value = parseExpr(*valueV);
      if (!value)
        return value.takeError();
      auto arrayType = asl::ArrayType::get(
          &ctx, TypeAttr::get(value->getType()),
          asl::ArrayIndexAttr::get(
              &ctx,
              asl::ArrayIndexKindAttr::get(&ctx, asl::ArrayIndexKind::int_type),
              builder.getI64IntegerAttr(-1), nullptr, nullptr));
      auto op = builder.create<asl::ArrayOp>(loc, arrayType, *length, *value);
      return op.getResult();
    } else if (k == "E_EnumArray") {
      auto enumName = obj->getString("enum");
      auto *labelsV = obj->get("labels");
      auto *valueV = obj->get("value");
      if (!enumName || !labelsV || !valueV)
        return makeError("E_EnumArray missing components");
      SmallVector<Attribute> labels;
      if (auto *arr = labelsV->getAsArray()) {
        for (auto &l : *arr) {
          if (auto ls = l.getAsString())
            labels.push_back(builder.getStringAttr(*ls));
        }
      }
      auto value = parseExpr(*valueV);
      if (!value)
        return value.takeError();
      auto arrayType = asl::ArrayType::get(
          &ctx, TypeAttr::get(value->getType()),
          asl::ArrayIndexAttr::get(&ctx,
                                   asl::ArrayIndexKindAttr::get(
                                       &ctx, asl::ArrayIndexKind::enum_type),
                                   builder.getI64IntegerAttr(-1),
                                   builder.getStringAttr(*enumName),
                                   builder.getArrayAttr(labels)));
      auto op = builder.create<asl::EnumArrayOp>(
          loc, arrayType, builder.getStringAttr(*enumName),
          builder.getArrayAttr(labels), *value);
      return op.getResult();
    } else if (k == "E_Arbitrary") {
      // Accept both 'arbitrary_type' and 'target_type' for the type field.
      auto *typeV = obj->get("arbitrary_type");
      if (!typeV)
        typeV = obj->get("target_type");
      if (!typeV)
        return makeError("E_Arbitrary missing arbitrary_type");
      auto arbitraryType = parseType(*typeV);
      if (!arbitraryType)
        return arbitraryType.takeError();
      auto op = builder.create<asl::ArbitraryOp>(loc, *arbitraryType,
                                                 TypeAttr::get(*arbitraryType));
      return op.getResult();
    } else if (k == "E_Pattern") {
      auto *exprV = obj->get("expr");
      auto *patternV = obj->get("pattern");
      if (!exprV || !patternV)
        return makeError("E_Pattern missing expr or pattern");

      auto expr = parseExpr(*exprV);
      if (!expr)
        return expr.takeError();

      // Create pattern op with region
      auto patternOp =
          builder.create<asl::PatternOp>(loc, builder.getI1Type(), *expr);
      Region &patternRegion = patternOp.getPattern();

      // Build pattern inside the region
      patternRegion.push_back(new Block);
      auto prevInsertionPoint = builder.saveInsertionPoint();
      builder.setInsertionPointToStart(&patternRegion.back());

      auto patternResult = parsePatternInRegion(*patternV, *expr);
      if (!patternResult) {
        builder.restoreInsertionPoint(prevInsertionPoint);
        return patternResult.takeError();
      }

      // No need for explicit yield in ASL
      builder.restoreInsertionPoint(prevInsertionPoint);

      return patternOp.getResult();
    }
    return makeError("unsupported expr kind: ", k);
  }

  // Parse statement object from JSON (top-level switch on "type" value)
  llvm::Error parseAndEmitStmt(const llvm::json::Value &v) {
    ContextSetter guard(*this, &v);
    auto *obj = v.getAsObject();
    if (!obj)
      return makeError("stmt not object");
    // asl-json-backend emit local declarations without a statement discriminant
    // string \"type\":\"S_Decl\"; instead the key
    // \"type\" may hold the declared variable's type object (JSON object not a
    // string). Detect this by checking that 'type' is either missing or not a
    // string while both 'keyword' and 'decl_item' keys are present.
    StringRef k; // statement kind discriminant
    if (auto kindStr = obj->getString("type")) {
      k = *kindStr; // Normal case: string discriminant.
    } else if ((obj->get("keyword") && obj->get("decl_item"))) {
      k = "S_Decl"; // Fallback: implicit decl statement.
    } else {
      return makeError("stmt missing type");
    }
    if (k == "S_Pragma") {
      auto name = obj->getString("name");
      if (!name)
        return makeError("pragma stmt missing name");
      SmallVector<Value> args;
      if (auto *argsV = obj->get("args"))
        if (auto *arr = argsV->getAsArray())
          for (auto &a : *arr) {
            auto av = parseExpr(a);
            if (!av)
              return av.takeError();
            args.push_back(*av);
          }
      builder.create<asl::StmtPragmaOp>(loc, builder.getStringAttr(*name),
                                        ValueRange(args));
      return llvm::Error::success();
    } else if (k == "S_Unreachable") {
      builder.create<asl::StmtUnreachableOp>(loc);
      return llvm::Error::success();
    } else if (k == "S_Seq") {
      auto *firstV = obj->get("first");
      auto *secondV = obj->get("second");
      if (!firstV || !secondV)
        return makeError("seq missing components");
      // Do not create asl.stmt.seq; emit statements directly in order.
      if (auto err = parseAndEmitStmt(*firstV))
        return err;
      if (auto err = parseAndEmitStmt(*secondV))
        return err;
      return llvm::Error::success();
    } else if (k == "S_Decl") {
      auto keywordStr = obj->getString("keyword");
      auto *itemV = obj->get("decl_item");
      if (!keywordStr || !itemV)
        return makeError("decl missing keyword/item");
      auto *itemObj = itemV->getAsObject();
      if (!itemObj)
        return makeError("decl_item not object");
      auto ldkAttr = mapLocalDeclKeyword(*keywordStr);
      auto ldiAttr = buildLocalDeclItem(*itemObj);
      if (!ldiAttr)
        return ldiAttr.takeError();
      TypeAttr typeAttr = nullptr;
      const llvm::json::Value *declTypeJson = nullptr;
      if (auto *dtV = obj->get("type")) { // key is "type" in backend JSON
        if (!dtV->getAsNull()) {
          declTypeJson = dtV;
          auto t = parseType(*dtV);
          if (!t)
            return t.takeError();
          typeAttr = TypeAttr::get(*t);
        }
      }
      Value initVal;
      if (auto *ivV = obj->get("init")) {
        if (!ivV->getAsNull()) {
          auto iv = parseExpr(*ivV);
          if (!iv)
            return iv.takeError();
          initVal = *iv;
        }
      }
      // If declared type is present and initial value type differs, insert an
      // ATC.
      if (typeAttr && initVal && initVal.getType() != typeAttr.getValue()) {
        if (auto bitsTy = dyn_cast<asl::BitsType>(typeAttr.getValue())) {
          // Specialized bits ATC when width is non-literal.
          Value widthVal;
          if (declTypeJson) {
            if (auto *tObj = declTypeJson->getAsObject()) {
              if (auto tStr = tObj->getString("type");
                  tStr && *tStr == "T_Bits") {
                if (auto *wV = tObj->get("width")) {
                  if (auto *wObj = wV->getAsObject()) {
                    auto wKind = wObj->getString("type");
                    // Only parse to SSA if not a literal.
                    if (!(wKind && *wKind == "E_Literal")) {
                      auto w = parseExpr(*wV);
                      if (!w)
                        return w.takeError();
                      widthVal = *w;
                    }
                  }
                }
              }
            }
          }
          if (widthVal)
            initVal =
                builder
                    .create<asl::AtcBitsOp>(loc, bitsTy, initVal, widthVal,
                                            bitsTy.getBitfields())
                    .getResult();
          else
            initVal = builder
                          .create<asl::AtcOp>(loc, typeAttr.getValue(), initVal,
                                              typeAttr)
                          .getResult();
        } else if (auto arrayTy =
                       dyn_cast<asl::ArrayType>(typeAttr.getValue())) {
          // Specialized array ATC when length is non-literal.
          Value lengthVal;
          if (declTypeJson) {
            if (auto *tObj = declTypeJson->getAsObject()) {
              if (auto tStr = tObj->getString("type");
                  tStr && *tStr == "T_Array") {
                if (auto *idxV = tObj->get("index")) {
                  if (auto *idxObj = idxV->getAsObject()) {
                    if (auto idxType = idxObj->getString("type")) {
                      if (*idxType == "ArrayLength_Expr") {
                        if (auto *eV = idxObj->get("expr")) {
                          if (auto *eObj = eV->getAsObject()) {
                            auto eKind = eObj->getString("type");
                            if (!(eKind && *eKind == "E_Literal")) {
                              auto len = parseExpr(*eV);
                              if (!len)
                                return len.takeError();
                              lengthVal = *len;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          if (lengthVal)
            initVal =
                builder
                    .create<asl::AtcArrayOp>(loc, arrayTy, initVal, lengthVal)
                    .getResult();
          else
            initVal = builder
                          .create<asl::AtcOp>(loc, typeAttr.getValue(), initVal,
                                              typeAttr)
                          .getResult();
        } else {
          initVal = builder
                        .create<asl::AtcOp>(loc, typeAttr.getValue(), initVal,
                                            typeAttr)
                        .getResult();
        }
      }
      auto decl = builder.create<asl::StmtDeclOp>(loc, initVal, ldkAttr,
                                                  *ldiAttr, typeAttr);
      // Bind names
      auto itemKindAttr = (*ldiAttr).getKind();
      if (itemKindAttr) {
        switch (itemKindAttr.getValue()) {
        case asl::LDIKind::var: {
          if (auto varName = (*ldiAttr).getVar())
            bind(varName.getValue(), initVal);
          break;
        }
        case asl::LDIKind::tuple: {
          if (auto tupleArr = (*ldiAttr).getTuple()) {
            for (auto attr : tupleArr)
              if (auto sa = dyn_cast<StringAttr>(attr))
                bind(sa.getValue(), Value());
          }
          break;
        }
        default:
          break;
        }
      }
      (void)decl;
      return llvm::Error::success();
    } else if (k == "S_Call") {
      auto *callV = obj->get("call");
      if (!callV)
        return makeError("S_Call missing call");
      auto *cObj = callV->getAsObject();
      if (!cObj)
        return makeError("call not object");
      auto name = cObj->getString("name");
      if (!name)
        return makeError("call missing name");
      SmallVector<Value> args;
      int paramsSize = 0;
      if (auto *paramsV = cObj->get("params"))
        if (auto *arr = paramsV->getAsArray()) {
          for (auto &p : *arr) {
            auto pv = parseExpr(p);
            if (!pv)
              return pv.takeError();
            args.push_back(*pv);
            ++paramsSize;
          }
        }
      if (auto *argsV = cObj->get("args"))
        if (auto *arr = argsV->getAsArray())
          for (auto &a : *arr) {
            auto av = parseExpr(a);
            if (!av)
              return av.takeError();
            args.push_back(*av);
          }
      auto callTypeStr = cObj->getString("call_type");
      SubprogramType callType = SubprogramType::procedure;
      if (callTypeStr)
        callType = mapSubprogramType(*callTypeStr);
      builder.create<asl::StmtCallOp>(
          loc, builder.getStringAttr(*name), ValueRange(args),
          builder.getI32IntegerAttr(paramsSize),
          asl::SubprogramTypeAttr::get(&ctx, callType));
      return llvm::Error::success();
    } else if (k == "S_Pass") {
      builder.create<asl::StmtPassOp>(loc);
      return llvm::Error::success();
    } else if (k == "S_Assign") {
      auto *lhsV = obj->get("lhs");
      auto *rhsV = obj->get("rhs");
      if (!lhsV || !rhsV)
        return makeError("S_Assign missing components");
      auto lhs = parseLExpr(*lhsV);
      if (!lhs)
        return lhs.takeError();
      auto rhs = parseExpr(*rhsV);
      if (!rhs)
        return rhs.takeError();
      builder.create<asl::StmtAssignOp>(loc, *lhs, *rhs);
      return llvm::Error::success();
    } else if (k == "S_Return") {
      auto *valV = obj->get("expr");
      if (!valV)
        return makeError("return missing expr");
      Value returnValue;
      // Handle null expressions (void returns)
      if (!valV->getAsNull()) {
        auto rv = parseExpr(*valV);
        if (!rv)
          return rv.takeError();
        returnValue = *rv;
        // If we know the function's return type, insert ATC for:
        // 1. Type mismatches (existing logic)
        // 2. Parameterized types that need materialization (new logic)
        if (currentReturnTypeAttr) {
          bool needsATC =
              returnValue.getType() != currentReturnTypeAttr.getValue();

          // For parameterized types, always insert ATC to materialize the type
          if (!needsATC &&
              (currentReturnBitsWidth || currentReturnArrayLength)) {
            needsATC = true;
          }

          if (needsATC) {
            if (auto bitsTy =
                    dyn_cast<asl::BitsType>(currentReturnTypeAttr.getValue())) {
              if (currentReturnBitsWidth)
                returnValue =
                    builder
                        .create<asl::AtcBitsOp>(loc, bitsTy, returnValue,
                                                currentReturnBitsWidth,
                                                bitsTy.getBitfields())
                        .getResult();
              else
                returnValue = builder
                                  .create<asl::AtcOp>(loc, bitsTy, returnValue,
                                                      currentReturnTypeAttr)
                                  .getResult();
            } else if (auto arrayTy = dyn_cast<asl::ArrayType>(
                           currentReturnTypeAttr.getValue())) {
              if (currentReturnArrayLength)
                returnValue =
                    builder
                        .create<asl::AtcArrayOp>(loc, arrayTy, returnValue,
                                                 currentReturnArrayLength)
                        .getResult();
              else
                returnValue = builder
                                  .create<asl::AtcOp>(loc, arrayTy, returnValue,
                                                      currentReturnTypeAttr)
                                  .getResult();
            } else {
              returnValue =
                  builder
                      .create<asl::AtcOp>(loc, currentReturnTypeAttr.getValue(),
                                          returnValue, currentReturnTypeAttr)
                      .getResult();
            }
          }
        }
      }
      builder.create<asl::StmtReturnOp>(loc, returnValue);
      return llvm::Error::success();
    } else if (k == "S_Cond") {
      auto *condV = obj->get("condition");
      auto *thenV = obj->get("then_stmt");
      auto *elseV = obj->get("else_stmt");
      if (!condV || !thenV)
        return makeError("S_Cond missing components");
      auto cond = parseExpr(*condV);
      if (!cond)
        return cond.takeError();
      auto condOp = builder.create<asl::StmtCondOp>(loc, *cond);
      Region &branchesRegion = condOp.getBranches();
      branchesRegion.push_back(new Block); // then branch
      builder.setInsertionPointToStart(&branchesRegion.back());
      if (auto err = parseAndEmitStmt(*thenV))
        return err;

      branchesRegion.push_back(new Block); // else branch
      if (elseV && !elseV->getAsNull()) {
        builder.setInsertionPointToStart(&branchesRegion.back());
        if (auto err = parseAndEmitStmt(*elseV))
          return err;
      } else {
        // Empty else block if no else statement
        builder.setInsertionPointToStart(&branchesRegion.back());
      }
      builder.setInsertionPointAfter(condOp);
      return llvm::Error::success();
    } else if (k == "S_Assert") {
      auto *exprV = obj->get("expr");
      if (!exprV)
        return makeError("S_Assert missing expr");
      auto expr = parseExpr(*exprV);
      if (!expr)
        return expr.takeError();
      builder.create<asl::StmtAssertOp>(loc, *expr);
      return llvm::Error::success();
    } else if (k == "S_For") {
      auto indexName = obj->getString("index");
      // Some JSON variants use "index_name" instead of "index".
      if (!indexName)
        indexName = obj->getString("index_name");
      auto *startV = obj->get("start");
      auto *endV = obj->get("end");
      auto *bodyV = obj->get("body");
      auto directionStr = obj->getString("direction");
      if (!indexName || !startV || !endV || !bodyV)
        return makeError("S_For missing components");
      auto start = parseExpr(*startV);
      if (!start)
        return start.takeError();
      auto end = parseExpr(*endV);
      if (!end)
        return end.takeError();
      auto direction = mapForDirectionAttr(directionStr ? *directionStr : "up");
      Value limit;
      if (auto *limitV = obj->get("limit")) {
        if (!limitV->getAsNull()) {
          auto limitExpr = parseExpr(*limitV);
          if (!limitExpr)
            return limitExpr.takeError();
          limit = *limitExpr;
        }
      }
      auto forOp =
          builder.create<asl::StmtForOp>(loc, builder.getStringAttr(*indexName),
                                         direction, *start, *end, limit);
      Region &bodyRegion = forOp.getBody();
      bodyRegion.push_back(new Block);
      builder.setInsertionPointToStart(&bodyRegion.back());
      if (auto err = parseAndEmitStmt(*bodyV))
        return err;
      builder.setInsertionPointAfter(forOp);
      return llvm::Error::success();
    } else if (k == "S_While") {
      auto *condV = obj->get("condition");
      auto *bodyV = obj->get("body");
      if (!condV || !bodyV)
        return makeError("S_While missing components");
      auto cond = parseExpr(*condV);
      if (!cond)
        return cond.takeError();
      Value limit;
      if (auto *limitV = obj->get("limit")) {
        if (!limitV->getAsNull()) {
          auto limitExpr = parseExpr(*limitV);
          if (!limitExpr)
            return limitExpr.takeError();
          limit = *limitExpr;
        }
      }
      auto whileOp = builder.create<asl::StmtWhileOp>(loc, *cond, limit);
      Region &bodyRegion = whileOp.getBody();
      bodyRegion.push_back(new Block);
      builder.setInsertionPointToStart(&bodyRegion.back());
      if (auto err = parseAndEmitStmt(*bodyV))
        return err;
      builder.setInsertionPointAfter(whileOp);
      return llvm::Error::success();
    } else if (k == "S_Repeat") {
      auto *bodyV = obj->get("body");
      auto *condV = obj->get("condition");
      if (!bodyV || !condV)
        return makeError("S_Repeat missing components");
      auto cond = parseExpr(*condV);
      if (!cond)
        return cond.takeError();
      Value limit;
      if (auto *limitV = obj->get("limit")) {
        if (!limitV->getAsNull()) {
          auto limitExpr = parseExpr(*limitV);
          if (!limitExpr)
            return limitExpr.takeError();
          limit = *limitExpr;
        }
      }
      auto repeatOp = builder.create<asl::StmtRepeatOp>(loc, *cond, limit);
      Region &bodyRegion = repeatOp.getBody();
      bodyRegion.push_back(new Block);
      builder.setInsertionPointToStart(&bodyRegion.back());
      if (auto err = parseAndEmitStmt(*bodyV))
        return err;
      builder.setInsertionPointAfter(repeatOp);
      return llvm::Error::success();
    } else if (k == "S_Throw") {
      Value expr;
      TypeAttr throwType;
      // Per schema, S_Throw encodes its payload under key "value":
      //   null, or { expr: <Expr>, type: <Type|null> }
      if (auto *valueV = obj->get("value")) {
        if (!valueV->getAsNull()) {
          if (auto *valObj = valueV->getAsObject()) {
            if (auto *exprV = valObj->get("expr")) {
              if (!exprV->getAsNull()) {
                auto throwExpr = parseExpr(*exprV);
                if (!throwExpr)
                  return throwExpr.takeError();
                expr = *throwExpr;
              }
            }
            if (auto *typeV = valObj->get("type")) {
              if (!typeV->getAsNull()) {
                auto type = parseType(*typeV);
                if (!type)
                  return type.takeError();
                throwType = TypeAttr::get(*type);
              }
            }
          } else {
            return makeError("S_Throw 'value' not object");
          }
        }
      }
      builder.create<asl::StmtThrowOp>(loc, expr, throwType);
      return llvm::Error::success();
    } else if (k == "S_Try") {
      // Accept multiple encodings: prefer 'stmt', fallback to 'body'.
      const llvm::json::Value *protV = obj->get("stmt");
      if (!protV || protV->getAsNull())
        if (auto *alt = obj->get("body"))
          protV = alt;
      auto tryOp = builder.create<asl::StmtTryOp>(loc);
      // Protected region
      Region &protectedRegion = tryOp.getProtected();
      protectedRegion.push_back(new Block);
      builder.setInsertionPointToStart(&protectedRegion.back());
      if (protV && !protV->getAsNull()) {
        if (auto err = parseAndEmitStmt(*protV))
          return err;
      }

      // Handlers region: parse 'catchers' (array of handlers) and optional
      // 'otherwise'. We lower each handler into its own block. To ensure the
      // MLIR printer emits an explicit block label (which FileCheck expects),
      // we also create a leading empty block.
      Region &handlersRegion = tryOp.getHandlers();
      // Leading empty block to force label printing even if a single handler.
      handlersRegion.push_back(new Block);

      // Collect handler metadata to attach as attributes on the try op so we
      // preserve exception variable names and types.
      SmallVector<Attribute> handlerTypeAttrs;
      SmallVector<Attribute> handlerNameAttrs;

      auto emitHandlerBody =
          [&](const llvm::json::Value &bodyVal) -> llvm::Error {
        // Each handler/otherwise gets its own block.
        handlersRegion.push_back(new Block);
        builder.setInsertionPointToStart(&handlersRegion.back());
        return parseAndEmitStmt(bodyVal);
      };

      // Parse catchers
      if (auto *catchersV = obj->get("catchers")) {
        if (auto *arr = catchersV->getAsArray()) {
          for (auto &cv : *arr) {
            if (auto *cObj = cv.getAsObject()) {
              // Name (may be null)
              if (auto name = cObj->getString("name"))
                handlerNameAttrs.push_back(builder.getStringAttr(*name));
              else
                handlerNameAttrs.push_back(builder.getStringAttr(""));
              // Exception type
              if (auto *exnTyV = cObj->get("exception_type")) {
                auto exnTy = parseType(*exnTyV);
                if (!exnTy)
                  return exnTy.takeError();
                handlerTypeAttrs.push_back(TypeAttr::get(*exnTy));
              }
              // Accept both 'stmt' and legacy 'body' for handler body.
              const llvm::json::Value *stmtV = cObj->get("stmt");
              if (!stmtV || stmtV->getAsNull())
                if (auto *alt = cObj->get("body"))
                  stmtV = alt;
              if (stmtV && !stmtV->getAsNull()) {
                if (auto err = emitHandlerBody(*stmtV))
                  return err;
              }
            }
          }
        }
      }
      // Parse 'otherwise' handler if present (record empty name and null type?)
      if (auto *otherwiseV = obj->get("otherwise")) {
        if (!otherwiseV->getAsNull()) {
          // Encode name as empty string and omit a type entry for otherwise.
          handlerNameAttrs.push_back(builder.getStringAttr(""));
          if (auto err = emitHandlerBody(*otherwiseV))
            return err;
        }
      }

      // Attach captured handler metadata as attributes for visibility.
      if (!handlerTypeAttrs.empty())
        tryOp->setAttr("handler_types", builder.getArrayAttr(handlerTypeAttrs));
      if (!handlerNameAttrs.empty())
        tryOp->setAttr("handler_names", builder.getArrayAttr(handlerNameAttrs));

      builder.setInsertionPointAfter(tryOp);
      return llvm::Error::success();
    } else if (k == "S_Print") {
      SmallVector<Value> args;
      bool newline = false;
      bool debug = false;
      if (auto *argsV = obj->get("args")) {
        if (auto *arr = argsV->getAsArray()) {
          for (auto &a : *arr) {
            auto arg = parseExpr(a);
            if (!arg)
              return arg.takeError();
            args.push_back(*arg);
          }
        }
      }
      if (auto newlineVal = obj->getBoolean("newline"))
        newline = *newlineVal;
      if (auto debugVal = obj->getBoolean("debug"))
        debug = *debugVal;
      builder.create<asl::StmtPrintOp>(loc, ValueRange(args),
                                       builder.getBoolAttr(newline),
                                       builder.getBoolAttr(debug));
      return llvm::Error::success();
    }
    // Gracefully ignore unknown statements instead of failing import.
    return llvm::Error::success();
  }

  // Build function body region from subprogram body JSON (only SB_ASL with
  // single stmt sequence simplified)
  llvm::Error buildFunctionBody(Region &region,
                                const llvm::json::Object &bodyObj) {
    // New scope for function body
    pushScope();
    auto res = [&]() -> llvm::Error {
      auto bType = bodyObj.getString("type");
      if (!bType)
        return makeError("subprogram body missing type");
      StringRef t = *bType;
      region.push_back(new Block);
      builder.setInsertionPointToStart(&region.back());
      if (t == "SB_ASL") {
        auto *stmtV = bodyObj.get("stmt");
        if (!stmtV)
          return makeError("SB_ASL missing stmt");
        if (auto err = parseAndEmitStmt(*stmtV))
          return err;
        return llvm::Error::success();
      } else if (t == "SB_Primitive") {
        return llvm::Error::success();
      }
      return makeError("unsupported subprogram body type: ", t);
    }();
    popScope();
    return res;
  }
  llvm::Error importFunc(const llvm::json::Object &obj) {
    // Push scope for parameters and arguments.
    pushScope();
    auto name = obj.getString("name");
    if (!name)
      return makeError("function missing name");
    // Parse template-like parameters ("parameters" JSON field).
    SmallVector<Attribute> paramsAttr;
    SmallVector<StringAttr> tmplParamNames; // collect for entry materialization
    SmallVector<Type> tmplParamTypes; // collect types for entry materialization
    if (auto *paramsV = obj.get("parameters")) {
      if (auto *arr = paramsV->getAsArray()) {
        for (auto &pv : *arr) {
          if (auto *pObj = pv.getAsObject()) {
            auto paramName = pObj->getString("name");
            if (!paramName)
              return makeError("parameter missing name");
            // Default parameter type is a parameterized integer, not plain
            // unconstrained.
            Type paramType = getParameterizedIntType();
            if (auto *ptV = pObj->get("type")) {
              if (!ptV->getAsNull()) {
                auto t = parseType(*ptV);
                if (!t)
                  return t.takeError();
                paramType = *t;
              }
            }
            NamedAttribute pair[] = {
                builder.getNamedAttr("identifier",
                                     builder.getStringAttr(*paramName)),
                builder.getNamedAttr("type", TypeAttr::get(paramType))};
            paramsAttr.push_back(DictionaryAttr::get(&ctx, pair));
            // Defer materialization of template parameter; create VarOp in
            // function entry.
            tmplParamNames.push_back(builder.getStringAttr(*paramName));
            tmplParamTypes.push_back(paramType);
          }
        }
      }
    }

    // Parse value arguments ("args" JSON field). We record just the names in
    // the op attribute (to avoid changing the op definition) but we now also
    // collect their types so we can create block arguments instead of emitting
    // placeholder VarOps inside the body (fixes TODO 1: use block args & keep
    // correct types).
    SmallVector<Attribute> argsAttr;  // names only for op attribute
    SmallVector<Type> argTypes;       // types for block arguments
    SmallVector<StringAttr> argNames; // parallel names for binding
    SmallVector<Attribute>
        argTypeAttrVec; // Added: TypeAttr list for args_types
    // New: capture original arg type JSON objects for later ATC materialization
    SmallVector<const llvm::json::Value *, 8> argTypeJsons;
    if (auto *argsV = obj.get("args")) {
      if (auto *arr = argsV->getAsArray()) {
        for (auto &av : *arr) {
          if (auto *aObj = av.getAsObject()) {
            auto an = aObj->getString("name");
            if (!an)
              return makeError("arg missing name");
            argsAttr.push_back(builder.getStringAttr(*an));
            Type argTy = getDefaultIntType();
            const llvm::json::Value *capturedTypeJson = nullptr;
            if (auto *atV = aObj->get("type")) {
              if (!atV->getAsNull()) {
                auto t = parseType(*atV);
                if (!t)
                  return t.takeError();
                argTy = *t;
                capturedTypeJson =
                    atV; // save for potential non-literal width/length
              }
            }
            argTypes.push_back(argTy);
            argNames.push_back(builder.getStringAttr(*an));
            argTypeAttrVec.push_back(TypeAttr::get(argTy)); // collect TypeAttr
            argTypeJsons.push_back(capturedTypeJson);
          }
        }
      }
    }

    // Return type optional.
    TypeAttr retTypeAttr = nullptr;
    const llvm::json::Value *retTypeJsonCapture = nullptr;
    bool retTypeDependsOnParameters = false;
    if (auto *rtV = obj.get("return_type")) {
      if (!rtV->getAsNull()) {
        auto pt = parseType(*rtV);
        if (!pt)
          return pt.takeError();
        retTypeAttr = TypeAttr::get(*pt);
        retTypeJsonCapture = rtV;

        // Check if return type depends on parameterized integers
        retTypeDependsOnParameters = typeUsesParameterizedIntegersFromJSON(rtV);

        // If return type depends on parameters, don't set it as an attribute
        // Instead, it will be materialized via ATC in return statements
        if (retTypeDependsOnParameters) {
          // Keep retTypeAttr for internal use but don't pass to FuncDeclOp
        }
      }
    }

    bool builtin = false;
    if (auto *bV = obj.get("builtin"))
      if (auto ob = bV->getAsBoolean())
        builtin = *ob;

    // Body object (required)
    const llvm::json::Value *bodyV = obj.get("body");
    if (!bodyV)
      return makeError("function missing body");
    const llvm::json::Object *bodyObj = bodyV->getAsObject();
    if (!bodyObj)
      return makeError("body not object");

    bool primitive = false;
    if (auto bt = bodyObj->getString("type"))
      primitive = (*bt == "SB_Primitive");

    // Subprogram type attr
    auto subType = obj.getString("subprogram_type");
    if (!subType)
      return makeError("function missing subprogram_type");
    auto subAttr =
        asl::SubprogramTypeAttr::get(&ctx, mapSubprogramType(*subType));

    // Qualifier optional
    asl::FuncQualifierAttr qualifierAttr = nullptr;
    if (auto qual = obj.getString("qualifier"))
      qualifierAttr =
          asl::FuncQualifierAttr::get(&ctx, mapFuncQualifier(*qual));

    // Override optional
    asl::OverrideInfoAttr overrideAttr = nullptr;
    if (auto ov = obj.getString("override"))
      overrideAttr = asl::OverrideInfoAttr::get(&ctx, mapOverrideInfo(*ov));

    auto funcOp = builder.create<asl::FuncDeclOp>(
        loc, builder.getStringAttr(*name), builder.getArrayAttr(paramsAttr),
        retTypeAttr, builder.getBoolAttr(primitive),
        builder.getArrayAttr(argsAttr),
        builder.getArrayAttr(argTypeAttrVec), // new args_types attribute
        subAttr, qualifierAttr, overrideAttr, builder.getBoolAttr(builtin),
        Value());

    if (!primitive) {
      // Build entry block with block arguments for value args.
      ReturnTypeSetter __retGuard(*this, retTypeAttr);
      currentReturnBitsWidth = Value();
      currentReturnArrayLength = Value();
      Region &region = funcOp.getBody();
      region.push_back(new Block);
      Block *entry = &region.back();
      if (!argTypes.empty())
        entry->addArguments(argTypes,
                            SmallVector<Location>(argTypes.size(), loc));
      // Insert parameter VarOps at the start of entry so they are inside the
      // function.
      builder.setInsertionPointToStart(entry);
      for (auto it : llvm::enumerate(tmplParamNames)) {
        auto pVar = builder.create<asl::VarOp>(loc, tmplParamTypes[it.index()],
                                               it.value());
        bind(it.value().getValue(), pVar.getResult());
      }
      // If return type is Bits, precompute its width value from JSON
      // return_type only when the width is not a literal expression.
      if (retTypeAttr && isa<asl::BitsType>(retTypeAttr.getValue())) {
        if (retTypeJsonCapture) {
          if (auto *rtObj = retTypeJsonCapture->getAsObject()) {
            if (auto rtKind = rtObj->getString("type");
                rtKind && *rtKind == "T_Bits") {
              if (auto *wV = rtObj->get("width")) {
                if (auto *wObj = wV->getAsObject()) {
                  auto wKind = wObj->getString("type");
                  // Only parse to SSA if not a literal.
                  if (!(wKind && *wKind == "E_Literal")) {
                    auto w = parseExpr(*wV);
                    if (!w)
                      return w.takeError();
                    currentReturnBitsWidth = *w;
                  }
                }
              }
            }
          }
        }
      }
      // If return type is Array, precompute its length value from JSON
      // return_type only when the length is not a literal expression.
      if (retTypeAttr && isa<asl::ArrayType>(retTypeAttr.getValue())) {
        if (retTypeJsonCapture) {
          if (auto *rtObj = retTypeJsonCapture->getAsObject()) {
            if (auto rtKind = rtObj->getString("type");
                rtKind && *rtKind == "T_Array") {
              if (auto *idxV = rtObj->get("index")) {
                if (auto *idxObj = idxV->getAsObject()) {
                  if (auto idxType = idxObj->getString("type")) {
                    if (*idxType == "ArrayLength_Expr") {
                      if (auto *eV = idxObj->get("expr")) {
                        if (auto *eObj = eV->getAsObject()) {
                          auto eKind = eObj->getString("type");
                          if (!(eKind && *eKind == "E_Literal")) {
                            auto len = parseExpr(*eV);
                            if (!len)
                              return len.takeError();
                            currentReturnArrayLength = *len;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      // Before binding, materialize asserted type conversions for arguments
      // whose types depend on non-literal parameters (e.g., bits(N),
      // array(length_expr)). Emit ATCs for inference and bind names to the
      // coerced values so subsequent uses reference the annotated type.
      for (size_t idx = 0; idx < argNames.size(); ++idx) {
        Value sourceVal = entry->getArgument(static_cast<unsigned>(idx));
        Type argTy = argTypes[idx];
        const llvm::json::Value *typeJson =
            idx < argTypeJsons.size() ? argTypeJsons[idx] : nullptr;

        // If this argument had an explicit type annotation in the JSON, emit
        // an ATC op to reflect that annotation regardless of literal or
        // non-literal parameters.
        Value coerced = sourceVal;
        if (typeJson) {
          // Int with dynamic range constraint -> atc.int.range
          if (auto intTy = dyn_cast<asl::IntType>(argTy)) {
            if (auto *tObj = typeJson->getAsObject()) {
              if (auto tk = tObj->getString("type"); tk && *tk == "T_Int") {
                if (auto *ckV = tObj->get("constraint_kind")) {
                  if (auto *ckObj = ckV->getAsObject()) {
                    if (auto ckKind = ckObj->getString("type");
                        ckKind && *ckKind == "WellConstrained") {
                      if (auto *constraintsV = ckObj->get("constraints")) {
                        if (auto *constraintsArr = constraintsV->getAsArray()) {
                          for (auto &constraintVal : *constraintsArr) {
                            if (auto *constraintObj =
                                    constraintVal.getAsObject()) {
                              if (auto constraintType =
                                      constraintObj->getString("type");
                                  constraintType &&
                                  *constraintType == "Constraint_Exact") {
                                // Handle exact constraints with non-literal
                                // expressions
                                if (auto *exprV = constraintObj->get("expr")) {
                                  if (auto *exprObj = exprV->getAsObject()) {
                                    if (auto exprKind =
                                            exprObj->getString("type");
                                        exprKind && *exprKind != "E_Literal") {
                                      // Non-literal exact constraint, use
                                      // AtcIntExactOp with expression
                                      auto exactExpr = parseExpr(*exprV);
                                      if (!exactExpr)
                                        return exactExpr.takeError();
                                      coerced = builder
                                                    .create<asl::AtcIntExactOp>(
                                                        loc, intTy, sourceVal,
                                                        *exactExpr)
                                                    .getResult();
                                      break; // Found an exact constraint, stop
                                             // processing other constraints
                                    }
                                  }
                                }
                              } else if (constraintType &&
                                         *constraintType ==
                                             "Constraint_Range") {
                                // Check if range bounds are non-literal
                                // (variables)
                                bool hasNonLiteralBounds = false;
                                Value startSSA, endSSA;

                                if (auto *startV =
                                        constraintObj->get("start")) {
                                  auto start = parseExpr(*startV);
                                  if (!start)
                                    return start.takeError();
                                  startSSA = *start;

                                  if (auto *startObj = startV->getAsObject()) {
                                    if (auto startKind =
                                            startObj->getString("type");
                                        startKind &&
                                        *startKind != "E_Literal") {
                                      hasNonLiteralBounds = true;
                                    }
                                  }
                                }

                                if (auto *endV = constraintObj->get("end")) {
                                  auto end = parseExpr(*endV);
                                  if (!end)
                                    return end.takeError();
                                  endSSA = *end;

                                  if (auto *endObj = endV->getAsObject()) {
                                    if (auto endKind =
                                            endObj->getString("type");
                                        endKind && *endKind != "E_Literal") {
                                      hasNonLiteralBounds = true;
                                    }
                                  }
                                }

                                if (hasNonLiteralBounds && startSSA && endSSA) {
                                  coerced = builder
                                                .create<asl::AtcIntRangeOp>(
                                                    loc, intTy, sourceVal,
                                                    startSSA, endSSA)
                                                .getResult();
                                  break; // Found a range constraint, stop
                                         // processing other constraints
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }

            // If no specific range constraint with variables, fall back to
            // generic ATC
            if (coerced == sourceVal) {
              coerced = builder
                            .create<asl::AtcOp>(loc, intTy, sourceVal,
                                                TypeAttr::get(intTy))
                            .getResult();
            }
          }
          // Bits with dynamic width expression -> atc.bits
          else if (auto bitsTy = dyn_cast<asl::BitsType>(argTy)) {
            Value widthSSA; // only set when width not a literal
            if (auto *tObj = typeJson->getAsObject()) {
              if (auto tk = tObj->getString("type"); tk && *tk == "T_Bits") {
                if (auto *wV = tObj->get("width")) {
                  if (auto *wObj = wV->getAsObject()) {
                    auto wKind = wObj->getString("type");
                    // Only parse to SSA if not a literal.
                    if (!(wKind && *wKind == "E_Literal")) {
                      if (auto w = parseExpr(*wV))
                        widthSSA = *w;
                      else
                        return w.takeError();
                    }
                  }
                }
              }
            }
            if (widthSSA) {
              coerced =
                  builder
                      .create<asl::AtcBitsOp>(loc, bitsTy, sourceVal, widthSSA,
                                              bitsTy.getBitfields())
                      .getResult();
            } else {
              // Literal width or unconstrained -> generic atc to target bitsTy
              coerced = builder
                            .create<asl::AtcOp>(loc, bitsTy, sourceVal,
                                                TypeAttr::get(bitsTy))
                            .getResult();
            }
          } else if (auto arrayTy = dyn_cast<asl::ArrayType>(argTy)) {
            // Array with dynamic length expression -> atc.array
            Value lengthSSA; // only set when length not a literal
            if (auto *tObj = typeJson->getAsObject()) {
              if (auto tk = tObj->getString("type"); tk && *tk == "T_Array") {
                if (auto *idxV = tObj->get("index")) {
                  if (auto *idxObj = idxV->getAsObject()) {
                    if (auto idxType = idxObj->getString("type")) {
                      if (*idxType == "ArrayLength_Expr") {
                        if (auto *eV = idxObj->get("expr")) {
                          if (auto *eObj = eV->getAsObject()) {
                            auto eKind = eObj->getString("type");
                            if (!(eKind && *eKind == "E_Literal")) {
                              auto len = parseExpr(*eV);
                              if (!len)
                                return len.takeError();
                              lengthSSA = *len;
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
            if (lengthSSA) {
              coerced = builder
                            .create<asl::AtcArrayOp>(loc, arrayTy, sourceVal,
                                                     lengthSSA)
                            .getResult();
            } else {
              // Literal length or enum length -> generic atc to target arrayTy
              coerced = builder
                            .create<asl::AtcOp>(loc, arrayTy, sourceVal,
                                                TypeAttr::get(arrayTy))
                            .getResult();
            }
          } else {
            // Other annotated types -> generic atc
            coerced = builder
                          .create<asl::AtcOp>(loc, argTy, sourceVal,
                                              TypeAttr::get(argTy))
                          .getResult();
          }
        }
        // Bind the argument name to the coerced value so later uses (e.g.,
        // returns) reference the ATC result.
        bind(argNames[idx].getValue(), coerced);
      }
      // Now parse body statement(s).
      auto bType = bodyObj->getString("type");
      if (!bType)
        return makeError("subprogram body missing type");
      if (*bType == "SB_ASL") {
        if (auto *stmtV = bodyObj->get("stmt")) {
          if (auto err = parseAndEmitStmt(*stmtV))
            return err;
        } else {
          return makeError("SB_ASL missing stmt");
        }
      } else if (*bType == "SB_Primitive") {
        // Nothing to emit for primitive; leave empty body.
      } else {
        return makeError("unsupported subprogram body type: ", *bType);
      }
    }

    // Reset insertion point after function so subsequent decls are at module
    // scope.
    builder.setInsertionPointAfter(funcOp);
    popScope();
    return llvm::Error::success();
  }

  llvm::Error importGlobal(const llvm::json::Object &obj) {
    auto name = obj.getString("name");
    if (!name)
      return makeError("global missing name");
    auto kw = obj.getString("keyword");
    if (!kw)
      return makeError("global missing keyword");
    auto kwAttr = asl::GDKAttr::get(&ctx, mapGlobalDeclKeyword(*kw));
    // Type required if not null
    Type type = getDefaultIntType();
    if (auto *tV = obj.get("type")) {
      if (!tV->getAsNull()) {
        auto parsed = parseType(*tV);
        if (!parsed)
          return parsed.takeError();
        type = *parsed;
      }
    }
    // Initial value optional; fallback to literal int 0
    Value initVal;
    if (auto *ivV = obj.get("initial_value")) {
      if (!ivV->getAsNull()) {
        auto val = parseExpr(*ivV);
        if (!val)
          return val.takeError();
        initVal = *val;
      }
    }
    if (!initVal) {
      auto zero = builder.create<asl::LiteralIntOp>(loc, getDefaultIntType(),
                                                    builder.getStringAttr("0"));
      initVal = zero.getResult();
    }
    builder.create<asl::GlobalStorageDeclOp>(loc, kwAttr,
                                             builder.getStringAttr(*name),
                                             TypeAttr::get(type), initVal);
    return llvm::Error::success();
  }

  llvm::Error importTypeDecl(const llvm::json::Object &obj) {
    auto name = obj.getString("name");
    if (!name)
      return makeError("type decl missing name");
    auto *tdV = obj.get("type_def");
    if (!tdV)
      return makeError("type decl missing type_def");
    auto ty = parseType(*tdV);
    if (!ty)
      return ty.takeError();
    // dependencies ignored for now
    builder.create<asl::TypeDeclOp>(loc, builder.getStringAttr(*name),
                                    TypeAttr::get(*ty), DictionaryAttr());
    return llvm::Error::success();
  }

  //===----------------------------------------------------------------------//
  // Top-level import
  //===----------------------------------------------------------------------//

  llvm::Expected<ModuleOp> import(llvm::StringRef filePath) {
    auto fileOrErr = llvm::MemoryBuffer::getFile(filePath);
    if (!fileOrErr)
      return llvm::make_error<llvm::StringError>(
          "failed to open file", llvm::inconvertibleErrorCode());
    auto buffer = fileOrErr->get();
    auto value = llvm::json::parse(buffer->getBuffer());
    // Set context to root value during import.
    if (value)
      currentContext = &*value;

    if (!value)
      return llvm::make_error<llvm::StringError>(
          "failed to parse JSON", llvm::inconvertibleErrorCode());

    auto *obj = value->getAsObject();
    if (!obj || obj->getString("type") != "ASL_AST")
      return llvm::make_error<llvm::StringError>(
          "root object must have type = ASL_AST",
          llvm::inconvertibleErrorCode());

    ModuleOp module = ModuleOp::create(loc);
    builder.setInsertionPointToStart(module.getBody());

    if (auto decls = obj->getArray("declarations")) {
      for (const auto &declVal : *decls) {
        if (auto err = importDecl(declVal))
          return std::move(err);
      }
    }
    return module;
  }

  llvm::Error importDecl(const llvm::json::Value &val) {
    ContextSetter guard(*this, &val);
    const auto *obj = val.getAsObject();
    if (!obj)
      return llvm::make_error<llvm::StringError>(
          "declaration is not an object", llvm::inconvertibleErrorCode());
    auto typeStr = obj->getString("type");
    if (!typeStr)
      return llvm::make_error<llvm::StringError>(
          "missing declaration type", llvm::inconvertibleErrorCode());

    if (*typeStr == "D_Pragma") {
      auto name = obj->getString("name");
      if (!name)
        return llvm::make_error<llvm::StringError>(
            "pragma missing name", llvm::inconvertibleErrorCode());
      builder.create<asl::PragmaDeclOp>(loc, builder.getStringAttr(*name),
                                        ValueRange{});
      return llvm::Error::success();
    } else if (*typeStr == "D_Func") {
      auto *funcV = obj->get("func");
      if (!funcV)
        return makeError("function decl missing func object");
      auto *fObj = funcV->getAsObject();
      if (!fObj)
        return makeError("func not object");
      return importFunc(*fObj);
    } else if (*typeStr == "D_GlobalStorage") {
      auto *gV = obj->get("global_decl");
      if (!gV)
        return makeError("global storage missing global_decl");
      auto *gObj = gV->getAsObject();
      if (!gObj)
        return makeError("global_decl not object");
      return importGlobal(*gObj);
    } else if (*typeStr == "D_TypeDecl") {
      return importTypeDecl(*obj);
    }
    return llvm::make_error<llvm::StringError>(
        ("unsupported declaration type: " + *typeStr).str(),
        llvm::inconvertibleErrorCode());
  }
};
} // namespace

llvm::Expected<ModuleOp> mlir::asl::importJSONFile(MLIRContext &ctx,
                                                   llvm::StringRef filePath) {
  JSONImporter importer(ctx);
  return importer.import(filePath);
}
