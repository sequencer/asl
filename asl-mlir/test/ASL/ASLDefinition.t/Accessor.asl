// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

// Underlying storage element, R
var R : array [[32]] of bits(64);

// Accessor, X
accessor X(regno: integer{0..31}) <=> value: bits(64)
begin
  readonly getter
    if regno == 31 then
      return Zeros{64};
    else
      return R[[regno]];
    end;
  end;

  setter
    if regno != 31 then
      R[[regno]] = value;
    end;
  end;
end;

func main() => integer
begin
  var x = X(16);
  X(15) = x;
  return 0;
end;
