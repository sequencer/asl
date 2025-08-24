// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func declare{N}(b: bits(N)) => bits(2 * N)
begin
    var b1: bits(2 * N);
    return b1;
end;

let exp: integer{4, 8} = 4;
func decl() => bits(exp)
begin
    var b: bits(exp);
    return b;
end;
