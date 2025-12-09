// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyRecord of record{-}; // { Other(MyRecord) }

var g : MyRecord; // { Other(g) }

func main() => integer // { Subprogram(main) }
begin
    return 0;
end;
