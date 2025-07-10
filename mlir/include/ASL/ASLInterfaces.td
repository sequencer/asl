#ifndef ASL_INTERFACES
#define ASL_INTERFACES

include "mlir/IR/OpBase.td"
include "ASL/ASLAttributes.td"

def ASL_AnnotatedInterface : OpInterface<"AnnotatedInterface"> {
  let description = [{
    Interface for ASL operations that require version annotation and location tracking.
    Operations implementing this interface must have a version attribute and implement
    the location interface.
  }];

  let methods = [
    InterfaceMethod<
      "Get the version attribute",
      "mlir::asl::VersionAttr", "getVersion", (ins)
    >,
    InterfaceMethod<
      "Get the location",
      "mlir::Location", "getLoc", (ins)
    >,
  ];

  let extraClassDeclaration = [{
    /// Verify that the operation has required version attribute and location
    static mlir::LogicalResult verifyAnnotatedInterface(mlir::Operation *op) {
      // Check for version attribute
      auto versionAttr = op->getAttrOfType<mlir::asl::VersionAttr>("version");
      if (!versionAttr) {
        return op->emitOpError("must have a 'version' attribute");
      }
      
      return mlir::success();
    }
  }];

  let verify = [{
    return verifyAnnotatedInterface($_op);
  }];
}

// Constraint to ensure ops have version attribute
def HasVersionAttr : AttrConstraint<
  CPred<"$_self.isa<mlir::asl::VersionAttr>()">,
  "version must be an ASL version attribute">;

// Trait for operations that must implement AnnotatedInterface
def ASL_AnnotatedOp : NativeOpTrait<"AnnotatedOp">;

#endif // ASL_INTERFACES
