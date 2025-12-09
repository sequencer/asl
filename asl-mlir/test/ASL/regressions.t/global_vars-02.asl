// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

var x : integer = 5;
 
var y = x as integer{1..7};
 
func main() => integer
begin
    assert y == 5;

    return 0;
end;
