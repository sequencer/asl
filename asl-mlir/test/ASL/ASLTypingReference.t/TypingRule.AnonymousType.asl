// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type T1 of integer; // the named type `T1` whose structure is integer

type T2 of integer; // the named type `T2` whose structure is integer

type pairT of (integer, T1); // the named type `pairT` whose structure
                             // is (integer, integer)

func tsub01()
begin

    var dataT1: T1;

    var pair: pairT = (1,dataT1);
    // legal since right hand side has anonymous, non-primitive type (integer, T1)

    let dataAsInt: integer = dataT1;

    pair = (1, dataAsInt);

    // legal since right hand side has anonymous, primitive type (integer, integer)
    let dataT2: T2 = 10;

end;
