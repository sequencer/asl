// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type E of exception {-};

func throwing () => integer
begin
  throw E {-};
end;

let X: integer = throwing ();

func main () => integer
begin
  println (X);

  return 0;
end;
