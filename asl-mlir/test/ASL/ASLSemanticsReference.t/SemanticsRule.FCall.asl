// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

readonly func foo (x : integer) => integer
begin

  return x + 1;

end;

func bar (x : integer) 
begin

  assert x == 3; 

end;

func main () => integer
begin

  assert foo(2) == 3;
  bar(3);

  return 0;
end;
