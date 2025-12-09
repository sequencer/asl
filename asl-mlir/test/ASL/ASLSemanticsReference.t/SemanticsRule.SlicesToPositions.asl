// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main() => integer
begin
    var bv = '1010 0011';
    assert bv[1+:3, 7:5] == '001 101';
    return 0;
end;
