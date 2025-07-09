get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRASL
        )
add_llvm_executable(asl-opt asl-opt.cpp)

llvm_update_compile_flags(asl-opt)
target_link_libraries(asl-opt PRIVATE ${LIBS})
