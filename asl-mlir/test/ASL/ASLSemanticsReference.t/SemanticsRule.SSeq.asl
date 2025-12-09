// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  let x = 3;
  let y = x + 1;

  assert x == 3 && y == 4;

  return 0;
end;
