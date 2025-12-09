// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

let      LET_ALLOWED_NUMS_A                   = 8;

func positive4()
begin
    let testA : integer {LET_ALLOWED_NUMS_A}     = 8;
end;

