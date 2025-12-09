configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Provide extra substitutions for tool directories

set(ASL_OPT_TEST_DEPENDS
        FileCheck count not
        asl-opt
        )

add_lit_testsuite(check-asl-opt "Running the asl-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${ASL_OPT_TEST_DEPENDS}
        )
set_target_properties(check-asl-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(ASL_OPT ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${ASL_OPT_TEST_DEPENDS})

# GMP differential testing target (requires Python 3 with gmpy2)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
    add_custom_target(check-gmp-differential
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/GMP/differential-test.py
            --asl-opt $<TARGET_FILE:asl-opt>
            --num-tests 100
        DEPENDS asl-opt
        COMMENT "Running GMP differential tests against Python/gmpy2"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(check-gmp-differential PROPERTIES FOLDER "Tests")
endif()
