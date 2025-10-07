set(LLVM_TARGET_DEFINITIONS ASL.td)

mlir_tablegen(ASLEnums.h.inc -gen-enum-decls)
mlir_tablegen(ASLEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ASLOps.h.inc -gen-op-decls)
mlir_tablegen(ASLOps.cpp.inc -gen-op-defs)
mlir_tablegen(ASLAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=asl)
mlir_tablegen(ASLAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=asl)
mlir_tablegen(ASLTypes.h.inc -gen-typedef-decls -typedefs-dialect=asl)
mlir_tablegen(ASLTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=asl)
mlir_tablegen(ASLDialect.h.inc -gen-dialect-decls -dialect=asl)
mlir_tablegen(ASLDialect.cpp.inc -gen-dialect-defs -dialect=asl)

add_public_tablegen_target(ASLIncGen)

set(LLVM_TARGET_DEFINITIONS ASLPasses.td)

mlir_tablegen(ASLPasses.h.inc -gen-pass-decls -name ASL)
mlir_tablegen(ASLPasses.capi.h.inc -gen-pass-capi-header --prefix ASL)
mlir_tablegen(ASLPasses.capi.cpp.inc -gen-pass-capi-impl --prefix ASL)

add_public_tablegen_target(ASLPassIncGen)
