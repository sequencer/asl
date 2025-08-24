// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

config N : integer = 0;
func main () => integer begin
  return 0;
end;
