// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func main() => integer
begin
    let x: integer{0..2^64 - 1} = ARBITRARY: integer{0..2^64 - 1};
    let offset = 10;
    var x_offset: integer{x..x + 10} = x as integer{x..x + offset};
    return 0;
end;
