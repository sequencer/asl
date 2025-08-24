// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

pure func Zeros{N}() => bits(N)
begin
  return 0[N-1:0];
end;

func foo{N}(x: bits(N)) => bit
begin
    return x[0];
end;

config LIMIT1: integer = 2;
config LIMIT2: integer{1, 2, 3, 4, 5, 6, 7, 8, 9, 10} = 7;

func bar() => integer{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
begin
    var ret: integer = 1;
    while ret < LIMIT1 do
        ret = ret + ret * 2;
    end;
    return ret as integer{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
end;

func main() => integer
begin
    let N = bar();
    let M = LIMIT2;
    let x = Zeros{N};
    let y = Zeros{M};
    let z = foo{M+N}(x :: y);
    return 0;
end;
