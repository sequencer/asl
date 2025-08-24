// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func foo {N} (x: bits(N))
begin
  let y = 5 DIV (N + 1);
end;

func main () => integer
begin
  foo {1}('1');
  foo {0}('');
  return 0;
end;

