// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

var X: integer = 0;
var Y: integer = 0;

func set_and_return_X () => integer
begin
  X = 2;
  return 3;
end;

func set_and_return_Y () => integer
begin
  Y = 2;
  return 3;
end;

func main () => integer
begin
  let y = set_and_return_X () + set_and_return_Y ();

  return 0;
end;

