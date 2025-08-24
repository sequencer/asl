---
description: 'Description of the custom chat mode.'
tools: ['changes', 'codebase', 'editFiles', 'extensions', 'fetch', 'findTestFiles', 'githubRepo', 'problems', 'runCommands', 'runTasks', 'search', 'searchResults', 'terminalLastCommand', 'terminalSelection', 'testFailure', 'usages', 'vscodeAPI',]
---
Build the MLIR project in the `mlir` folder using `cmake -G Ninja -DLLVM_EXTERNAL_LIT=$(which lit) -B build && ninja -C build`.

To test the ASL MLIR, you can run the ninja target `check-asl-opt` via `ninja -C build check-asl-opt`.

Documentation is located in `doc` folder, please refer to `IR.typ` to see how to design IR.