// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pragma my_tool_pragma1;
pragma other_tool_op '0010', 123;
func my_function_with_tool_pragmas()
begin
    pass;
end;
