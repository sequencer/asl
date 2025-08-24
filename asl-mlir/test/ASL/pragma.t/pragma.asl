// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func test() => integer
begin
    pragma p1 "with arg";
    pragma p2;
    pragma p3 "multi arg", TRUE;
    return 1;
end;

pragma p1;
pragma p2 TRUE;
pragma p3 TRUE, 2 + 2, test();

func main() => integer
begin
    return 0;
end;
