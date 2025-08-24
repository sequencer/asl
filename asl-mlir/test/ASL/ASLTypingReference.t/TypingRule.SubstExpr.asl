// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

func plus{N}(x: bits(N + 2), z: integer{0..N}) => bits(N + 2)
begin
    return x + z;
end;

func main() => integer
begin
    var bv1 = Zeros{64};
    let z = 40;
    - = plus{z + 22}(bv1, z);
    return 0;
end;
