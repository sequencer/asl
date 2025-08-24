// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func f0 {N} (x: bits(N)) => bits(N)
begin
  return x;
end;

func f1 {M} (x: bits(M)) => integer {0..M}
begin
  return Len(x) as integer {0..M};
end;

func f2 {L}() => bits(L)
begin
  return Zeros{L};
end;

func main() => integer
begin
  let x: bits(4) = f0{}('0000');
  let y: integer {0..5} = f1{5}('11111');
  let z: bits(6) = f2 {6};

  return 0;
end;


