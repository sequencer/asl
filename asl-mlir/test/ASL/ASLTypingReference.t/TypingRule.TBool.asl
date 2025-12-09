// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyType of boolean;

func foo (x: boolean) => boolean
begin
  return FALSE ==> x;
end;

func main () => integer
begin
  var x: boolean;

  x = TRUE;
  x = foo (x as boolean);
  
  let y: boolean = x && x;

  assert x as boolean == x;

  return 0;
end;
