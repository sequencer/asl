// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyExceptionType1 of exception{-};
type MyExceptionType2 of exception{-};

func main () => integer
begin

     try
       throw MyExceptionType1 {-};
       assert FALSE;
     catch
       when MyExceptionType2 =>
         assert FALSE;
       otherwise =>
         println "Otherwise";
     end;

  return 0;
end;
