add_mlir_dialect_library(MLIRASLPass
  ASLToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ASL

  DEPENDS
  ASLIncGen
  ASLPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRControlFlowDialect
  MLIRASL
)
