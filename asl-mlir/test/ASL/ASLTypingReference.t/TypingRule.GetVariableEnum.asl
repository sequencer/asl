// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type Key of enumeration {One, Two, Three};
type SubKey subtypes Key;

func main() => integer
begin            // The right-hand-side expression is | Reason:
    var x = 5;   // Not an enumeration variable       | not a variable expression
    var y = x;   // Not an enumeration variable       | x is integer-typed
    var z = One; // An enumeration variable           | the underlying type is Key
    return 0;
end;
