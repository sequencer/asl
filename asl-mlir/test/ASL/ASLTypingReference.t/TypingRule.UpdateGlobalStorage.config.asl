// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

config i: integer = 1;
config r: real = 1.0;
config s: string = "hello";
config b: boolean = TRUE;
config bv: bits(8) = Zeros{8};

type Color of enumeration {RED, GREEN, BLUE};
config c: Color = RED;
