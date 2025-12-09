// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  for i = 0 to 3 do
    assert i <= 3;
  end;

  return 0;
end;
