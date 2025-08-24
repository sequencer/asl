// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  let A = Zeros{4};
  let b = A[UInt('1')*2+:2];

  return 0;
end;
