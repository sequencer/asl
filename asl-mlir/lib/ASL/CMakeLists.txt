# Only TableGen generation for now - no C++ source files yet
add_mlir_dialect_library(MLIRASL
  ASLDialect.cpp
  ASLAttributes.cpp
  ASLTypes.cpp
  ASLOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ASL

  DEPENDS
  ASLIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
