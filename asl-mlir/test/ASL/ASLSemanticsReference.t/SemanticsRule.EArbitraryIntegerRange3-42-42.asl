// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main () => integer
begin

  let x = ARBITRARY:integer {3, 42};
  assert x==42;

  return 0;
end; 
