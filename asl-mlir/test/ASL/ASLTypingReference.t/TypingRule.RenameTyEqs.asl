// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

pure func Ones{N}() => bits(N)
begin
  return NOT Zeros{N};
end;

func FlipPrefix{N}(x: bits(N), y: integer{0..N}) => bits(N)
begin
    return x XOR (Zeros{N-y} :: Ones{y});
end;

func main() => integer
begin
    let bv = '1001 0011';
    let bv_flipped = FlipPrefix{8}(bv, 4);
    assert bv_flipped == '1001 1100';
    // The next statement in comment is illegal,
    // since integer{9} does not type-satisfy integer{0..8}.
    // let bv_flipped = FlipPrefix{8}(bv, 9);
    return 0;
end;
