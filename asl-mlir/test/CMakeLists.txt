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

add_lit_testsuites(ASL_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${ASL_OPT_TEST_DEPENDS})
