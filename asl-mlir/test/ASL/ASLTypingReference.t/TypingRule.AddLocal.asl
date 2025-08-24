// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyException of exception{-};

func foo{N}(bv: bits(N))
begin
    pass;
end;

func main() => integer
begin

    try

        var x: integer;

        for i = 0 to 10 do

            x = x + 1;

        end;

    catch
        when exn: MyException => pass;
    end;

    return 0;

end;

