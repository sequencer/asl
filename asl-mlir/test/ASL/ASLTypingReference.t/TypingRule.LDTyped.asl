// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyT of integer;

readonly func foo (t: MyT) => integer
begin
  return t as integer;
end;

func main () => integer
begin
  let x: MyT = 42;
  var z: MyT;

  assert foo (x) == 42;
  assert foo (z) == 0;

  return 0;
end;
