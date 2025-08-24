// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

constant PI = 3.14;
var PC: bits(32) = Zeros{32};
config MaxIrq: integer = 480;
var Regs: array[[16]] of bits(32);

func main() => integer
begin
    return 0;
end;
