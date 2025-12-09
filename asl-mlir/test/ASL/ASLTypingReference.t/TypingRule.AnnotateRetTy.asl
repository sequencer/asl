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


func flip{N}(x: bits(N)) => bits(N)
begin

    return x XOR Ones{N};

end;


func proc()
begin

    pass;

end;


func main() => integer
begin

    var bv = Zeros{64};

    bv = flip{64}(bv);

    proc();

    return 0;

end;

