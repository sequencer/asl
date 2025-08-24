// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func f{N}(x: bits(N)) => bits(N + 1)
begin
    return x :: '0';
end;

func main() => integer
begin
    var inputA: bits(4);
    var outputA = f{4}(inputA); // an invocation of f: bits(4)=>bits(5)
    let widthB: integer {8,16} = if (ARBITRARY: boolean) then 8 else 16;
    var inputB: bits(widthB);
    var outputB = f{widthB}(inputB); // an invocation of
                                     // f: bits(widthB)=>bits(widthB + 1)
                                     // outputB is of type bits(widthB + 1)
    return 0;
end;
