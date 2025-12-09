// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({
func increment(x: integer) => integer
begin

    return x + 1;

end;

func increment(r: real) => real
begin

    return r + 1.0;

end;
