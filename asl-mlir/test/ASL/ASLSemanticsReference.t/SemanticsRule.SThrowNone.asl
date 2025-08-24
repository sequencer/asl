// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyExceptionType of exception{ a: integer };

func main () => integer
begin

  try
    try 
      throw MyExceptionType { a = 42 };
    catch
      when MyExceptionType => throw;
      otherwise => assert FALSE;
    end;
    assert FALSE;

  catch
    when exn: MyExceptionType =>
      assert exn.a == 42; 
    otherwise => assert FALSE;
  end;

  return 0;
end;
