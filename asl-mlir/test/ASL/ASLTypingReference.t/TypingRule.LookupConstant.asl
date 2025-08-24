// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type Color of enumeration {RED, GREEN, BLUE};

constant WORD_SIZE = 64;

func main() => integer
begin
    var c : Color; // Initialization requires looking up the constant RED.
    var bv1: bits(WORD_SIZE); // Requires looking up WORD_SIZE.
    return 0;
end;
