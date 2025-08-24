// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pragma asl_pragma1;
pragma asl_op 1, "start";
func my_function_with_pragma()
begin
    pass;
end;
