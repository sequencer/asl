// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

func widCheck{N, M}(b: bits(M)) => bits(N)
begin
    if (N == M) then
        // b has the type bits(M), but we know from the previous line that N==M
        // so it is safe to assert that b is within the domain of bits(N).
        // The resulting type of the asserting type conversion expression is bits(N),
        // matching the required return type.
        return b as bits(N);
    else
        return Zeros{N};
    end;
end;
