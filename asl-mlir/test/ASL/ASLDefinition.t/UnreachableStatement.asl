// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func diagnostic_assertion(condition: boolean, should_check: boolean, message: string)
begin
    if should_check && !condition then
        println "diagnostic assertion failed: ", message;
        unreachable;
    end;
end;

func main() => integer
begin
    diagnostic_assertion(FALSE, TRUE, "example message");
    return 0;
end;
