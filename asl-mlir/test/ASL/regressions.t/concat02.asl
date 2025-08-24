// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main() => integer
begin
  let a = '01';
  let b = '10';
  let a5 = '1' :: a :: a;
  let b14 = Replicate{14}(b);
  let b15 = '0' :: b14;
  let x20 = b15 :: a5;
  let a15 = '1' :: Replicate{14}(a);
  let b5 = '0' :: b :: b;
  let y20 = b5 :: a15;
  assert x20 == y20;
  return 0;
end;
