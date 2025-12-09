get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(asl-opt
  asl-opt.cpp
  JSONImporter.cpp

  DEPENDS
  ASLIncGen
  ASLPassIncGen
  GMPIncGen
  GMPPassIncGen
)

llvm_update_compile_flags(asl-opt)

target_link_libraries(asl-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRASL
  MLIRGMP
  MLIRASLPass
  MLIRGMPTransforms
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  MLIRTargetCpp
  ${GMP_LIBRARY}
)
