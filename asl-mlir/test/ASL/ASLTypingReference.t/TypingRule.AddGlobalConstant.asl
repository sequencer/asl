// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin

  return 0[N-1:0];

end;

constant FOUR = 4;
// The static environment binds FOUR to 4.

pure func foo(v: integer {0..100}) => integer {0..100}
begin

    return v;

end;


constant x = 32;

constant z: integer {0..100} = foo(x);

// The static environment binds z to 32.
constant gbv: bits(32) = Zeros{z};

func main() => integer
begin

    var bv: bits(2^FOUR) = Zeros{FOUR * FOUR};

    return 0;

end;


