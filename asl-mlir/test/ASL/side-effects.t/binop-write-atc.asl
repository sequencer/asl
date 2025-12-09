// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

var X: integer = 0;

func performs_atc () => integer
begin
  return (1 as integer {2});
end;

func main () => integer
begin
  let y = performs_atc () + X;

  return 0;
end;

