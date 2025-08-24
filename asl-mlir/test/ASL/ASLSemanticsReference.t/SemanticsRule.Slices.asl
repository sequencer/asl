// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  let x = '000 00 1 00';
  assert x[2, 7:5, 0+:3] == '1 000 100';
  return 0;
end;
