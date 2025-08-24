// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin
  var arr: array[[5]] of integer;

  // Legal
  arr[[2]] = 0;

  // Illegal
  let x = arr[[14]];

  return 1;
end;

