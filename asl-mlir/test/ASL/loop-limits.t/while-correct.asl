// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  var i: integer = 0;
  while (i < 10) looplimit 20 do
    i = i + 1;
  end;

  return 0;
end;
