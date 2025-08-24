// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  var i: integer = 0;
  repeat
    i = i + 1;
    println i;
  until (i >= 10) looplimit 5;

  return 0;
end;

