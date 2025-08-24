// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func internal_function(x : integer)
begin
    pragma implementation_hidden x + 1;
end;
