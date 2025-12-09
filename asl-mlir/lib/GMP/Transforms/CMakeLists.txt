add_mlir_dialect_library(MLIRGMPTransforms
  ConvertGMPToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/GMP

  DEPENDS
  GMPIncGen
  GMPPassIncGen

  LINK_LIBS PUBLIC
  MLIRGMP
  MLIRIR
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
)
