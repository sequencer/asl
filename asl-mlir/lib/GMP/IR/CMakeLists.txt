add_mlir_dialect_library(MLIRGMP
  GMPDialect.cpp
  GMPAttributes.cpp
  GMPTypes.cpp
  GMPOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/GMP

  DEPENDS
  GMPIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)
