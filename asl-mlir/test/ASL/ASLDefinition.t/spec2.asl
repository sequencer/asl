// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

var COUNT: integer;

func ColdReset()
begin
    COUNT = 0;
end;

func Step()
begin
    assert COUNT >= 0;
    COUNT = COUNT + 1;
    assert COUNT > 0;
end;
