// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

func parameterized_base_value{N}(x: bits(N))
begin
    // Legal: produces 0[:N].
    var constrained_bits_base: bits(N);

    // Legal.
    var constrained_bits_init: bits(N) = Zeros{N};
end;
