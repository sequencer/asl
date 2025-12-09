// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func conditionalExpression{N: integer {1,2}}(argT: bits(1), argF: bits(2)) => bits(N)
begin
    return (if (N == 1) then
        argT as bits(N) // asserting type conversion is required
    else
        argF as bits(N)); // asserting type conversion is required
end;
