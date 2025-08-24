// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyArrayType of array [[3]] of integer;

var my_array : MyArrayType;

func main () => integer
begin
  println my_array[[3]];
  return 0;
end;
