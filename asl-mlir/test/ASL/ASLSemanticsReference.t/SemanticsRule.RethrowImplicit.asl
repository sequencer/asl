// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyExceptionType of exception {msg: string};

func main () => integer
begin
  try
     try
       throw MyExceptionType{msg="Exception value A"}; // exception value A
       assert FALSE;
     catch
       when e: MyExceptionType =>
         println e.msg;
         throw; // Implicitly re-throwing exception value A
     end;
  catch
    when e: MyExceptionType =>
        println e.msg;
        assert TRUE;
  end;
  return 0;
end;
