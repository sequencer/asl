// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func foo(x: integer {0..10}, y: integer {0..10})
begin
  let z = Zeros{64};
  println z[x:y];
end;

func main() => integer
begin
    foo(4, 2);
    foo(2, 4);
    return 0;
end;
