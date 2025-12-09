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

func foo{N}(bv: bits(N))
begin
    var x : bits(N DIV 2 + 1) = bv[N DIV 2:0];
    var y : bits(N DIV 2 + N DIV 2) = bv;
    var z : bits(3 * N + N) = Ones{4 * N};
    // The following statement in comment is illegal, since the current equivalence
    // test cannot determine that `N*N` is equal to `N^2`.
    // var - : bits(N^2) = Ones{N * N};
end;

constant FOUR = 4;

func main() => integer
begin
    var bv: bits(2^FOUR) = Zeros{FOUR*FOUR};
    return 0;
end;
