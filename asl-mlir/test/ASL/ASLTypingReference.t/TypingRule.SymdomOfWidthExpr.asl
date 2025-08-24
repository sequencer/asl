// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

func main() => integer
begin
    let x = ARBITRARY: integer{0..1000};
    var bv : bits(2 * x) = Zeros{x} :: Zeros{x};
    return 0;
end;
