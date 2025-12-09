// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

// Function 'Add'
func Add{N}(x: bits(N), y: bits(N)) => bits(N)
begin
    return x + y;
end;

var Counter: integer = 0;

// Procedure 'IncrementCounter'
func IncrementCounter(inc: integer)
begin
    Counter = Counter + inc;
    return;
end;
