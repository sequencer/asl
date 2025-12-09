// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func Addition{N}(addend : bits(N), esize : integer{8,16,32,64}) => bits(N)
begin
    assert N == esize * 2 * 2;
    var result = Zeros {N};
    var e = Zeros {esize};

    for i = 0 to 1 do
        for j = 0 to 1 do
            e = Ones{esize};
        end;
        result[(2*i) *: esize] = e;
    end;
    return result;
end;
