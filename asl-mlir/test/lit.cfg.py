# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'ASL'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

config.suffixes = ['.mlir', '.asl']

config.test_source_root = os.path.dirname(__file__)

config.test_exec_root = os.path.join(config.standalone_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))

llvm_config.with_system_environment(['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])
llvm_config.with_environment('PATH', config.ajb_tools_dir, append_path=True)
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

llvm_config.use_default_substitutions()
tools = ['asl-json-backend', 'asl-opt']
tool_dirs = [config.asl_tools_dir, config.llvm_tools_dir, config.ajb_tools_dir]

llvm_config.add_tool_substitutions(tools, tool_dirs)