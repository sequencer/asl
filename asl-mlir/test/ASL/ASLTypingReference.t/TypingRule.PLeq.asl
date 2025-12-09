// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  assert 3 IN { <= 42 };
  assert 3.0 IN { <= 42.0 };
  return 0;
end;
