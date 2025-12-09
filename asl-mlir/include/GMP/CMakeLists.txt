set(LLVM_TARGET_DEFINITIONS GMP.td)

mlir_tablegen(GMPOps.h.inc -gen-op-decls)
mlir_tablegen(GMPOps.cpp.inc -gen-op-defs)
mlir_tablegen(GMPAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=gmp)
mlir_tablegen(GMPAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=gmp)
mlir_tablegen(GMPTypes.h.inc -gen-typedef-decls -typedefs-dialect=gmp)
mlir_tablegen(GMPTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=gmp)
mlir_tablegen(GMPDialect.h.inc -gen-dialect-decls -dialect=gmp)
mlir_tablegen(GMPDialect.cpp.inc -gen-dialect-defs -dialect=gmp)

add_public_tablegen_target(GMPIncGen)
