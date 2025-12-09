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

target_include_directories(obj.MLIRGMP PRIVATE ${GMP_INCLUDE_DIR})
target_link_libraries(obj.MLIRGMP PRIVATE ${GMP_LIBRARY})
