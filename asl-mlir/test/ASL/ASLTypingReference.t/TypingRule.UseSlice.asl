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

constant FIVE = 5;
constant SEVEN = 7;

func main() => integer
begin
    var bv = Ones{64};
    - = bv[FIVE]; // { Other(FIVE) }
    - = bv[SEVEN  : FIVE]; // { Other(FIVE), Other(SEVEN) }
    - = bv[SEVEN +: FIVE]; // { Other(FIVE), Other(SEVEN) }
    - = bv[SEVEN *: FIVE]; // { Other(FIVE), Other(SEVEN) }
    return 0;
end;
