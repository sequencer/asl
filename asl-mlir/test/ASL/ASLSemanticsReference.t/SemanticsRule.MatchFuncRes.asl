// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func proc()
begin
    return;
end;

func returns_values() => integer
begin
    return 5;
end;

func main() => integer
begin
    proc();
    - = returns_values();
    return 0;
end;
