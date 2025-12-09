// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyType of string; // An alias of string

func foo(x: string) => string
begin
  return x;
end;

func main() => integer
begin
  var x: string = "foo";
  assert x as string == x;
  x = foo(x as string);
  let y: string = x;
  return 0;
end;
