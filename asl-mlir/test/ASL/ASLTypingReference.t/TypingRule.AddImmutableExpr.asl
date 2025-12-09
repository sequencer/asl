// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{M}() => bits(M)
begin

  return 0[M-1:0];

end;

constant N = 15;

func main() => integer
begin

    let x: integer{1..2 * N} = 1; // The static environment remembers x = 1

    let t: integer{x..x + 1} = 2;

    let k: integer{5, 7, 8, 64} = 64 as integer{5, 7, 64};

    // The static environment remembers k = 64
    let sub_k: integer{5, 64} = 64 as integer{5, 64};

    // The static environment remembers sub_k = 64
    let y: bits(k) = Zeros{64} as bits(sub_k);

    return 0;

end;























