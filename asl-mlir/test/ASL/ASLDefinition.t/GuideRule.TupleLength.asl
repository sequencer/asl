// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main() => integer
begin
    var x = (5, TRUE);
    var y : (integer, boolean) = (5, TRUE);
    var z = (5);
    var w : integer = (5);
    return 0;
end;
