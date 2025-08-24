// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type A of integer;
func main() => integer
begin
    var b = ARBITRARY: boolean;
    var a : array[[5]] of integer;
    var c : array[[5]] of A;
    var x = if (b) then a else c;
    return 0;
end;

