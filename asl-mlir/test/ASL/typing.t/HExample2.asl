// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({


type ExtractType of enumeration {PLUS, COLON, ELEM, FUNC, ARBITRARY_OP, EXTEND};
func Example{size}(op_type : ExtractType) => bits(8*size)
begin
    var register : bits(8*size) = Ones{8*size};
    case op_type of
        when PLUS =>
            return register[0 +: 8*size];

        when COLON =>
            return register[8*size-1 : 0];

        when ELEM =>
            return register[0 *: 8*size];

        when FUNC =>
            return register_read{size};

        when ARBITRARY_OP =>
            return ARBITRARY: bits(8*size);

        when EXTEND =>
            if size == 32 then
                register[31:0] = Ones{32};
            else
                register = ZeroExtend{8*size}('0');
            end;
            return register;
    end;
end;

func register_read{size}() => bits(8*size)
begin
    return Ones{8*size};
end;
