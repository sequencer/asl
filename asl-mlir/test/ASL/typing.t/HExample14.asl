// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func Reverse{N}(word : bits(N), M : integer{1..N}) => bits(N)
begin
    return Zeros{N};
end;

func main () => integer
begin
  let c = 8 << UInt (ARBITRARY: bits(2)); // integer {8, 16, 32, 64}
  let bv = Zeros{c};
  let res = Reverse{c}(bv, 8);

  return 0;
end;


