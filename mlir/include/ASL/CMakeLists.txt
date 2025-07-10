set(LLVM_TARGET_DEFINITIONS ASLDialect.td)
mlir_tablegen(ASLOpsDialect.h.inc -gen-dialect-decls -dialect=asl)
mlir_tablegen(ASLOpsDialect.cpp.inc -gen-dialect-defs -dialect=asl)

set(LLVM_TARGET_DEFINITIONS ASLOps.td)
mlir_tablegen(ASLOps.h.inc -gen-op-decls)
mlir_tablegen(ASLOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS ASLTypes.td)
mlir_tablegen(ASLTypes.h.inc -gen-typedef-decls -dialect=asl)
mlir_tablegen(ASLTypes.cpp.inc -gen-typedef-defs -dialect=asl)

set(LLVM_TARGET_DEFINITIONS ASLAttributes.td)
mlir_tablegen(ASLAttributes.h.inc -gen-attrdef-decls -dialect=asl)
mlir_tablegen(ASLAttributes.cpp.inc -gen-attrdef-defs -dialect=asl)

set(LLVM_TARGET_DEFINITIONS ASLInterfaces.td)
mlir_tablegen(ASLInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ASLInterfaces.cpp.inc -gen-op-interface-defs)

add_public_tablegen_target(MLIRASLOpsIncGen)
