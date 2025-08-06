---
description: 'Description of the custom chat mode.'
tools: ['changes', 'codebase', 'editFiles', 'extensions', 'fetch', 'findTestFiles', 'githubRepo', 'problems', 'runCommands', 'runTasks', 'search', 'searchResults', 'terminalLastCommand', 'terminalSelection', 'testFailure', 'usages', 'vscodeAPI',]
---
Build the MLIR project in the `mlir` folder using `cmake -G Ninja -B build && ninja -C build > compile.log 2>&1`; then you should check the `compile.log` whenever you think you finished the job, please remove the build dir and rerun the build command to check if `build/bin/asl-opt` exist, if not please continue fix.

Documentation is located in `doc` folder, please refer to `IR.typ` to see how to design IR.