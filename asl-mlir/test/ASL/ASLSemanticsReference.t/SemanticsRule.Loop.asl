// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  var i: integer = 0;

  while i <= 3 looplimit 4 do
    assert i <= 3;
    i = i + 1;
  end;

  return 0;
end;
