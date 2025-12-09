// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func elide_empty_argument_list()
begin
  let x : bits(64) = Zeros{};
end;

func main() => integer
begin
  return 0;
end;
