// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

// any width bitvector may be passed to sub
func sub{N}(arg : bits(N))
begin
    pass;
end;
func positive4(w: integer{1, 2, 3})
begin
    sub{12}(Zeros{12});
    sub{w}(Zeros{w});
end;
