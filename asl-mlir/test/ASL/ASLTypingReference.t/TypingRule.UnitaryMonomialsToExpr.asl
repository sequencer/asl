// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main() => integer
begin
                        // the expressions corresponding
                        // to the rhs monomials
    var x : integer;
    var y : integer;
    - = x ^ 0 * y;      // y
    - = x ^ 1 * y;      // x * y
    - = x ^ 2 * y;      // x * x * y
    - = x ^ 3 * y;      // x^3 * y
    return 0;
end;
