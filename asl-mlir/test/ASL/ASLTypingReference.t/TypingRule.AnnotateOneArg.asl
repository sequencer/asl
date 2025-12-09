// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({
func arguments(b: boolean, i: integer, r: real)
begin

    - = b;

    - = i;

    - = r;

end;

