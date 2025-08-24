// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

type MyException of exception{-};

func throwing_func()
begin
    throw MyException{-};
end;

func non_throwing_func()
begin
    pass;
end;

func main() => integer
begin
    non_throwing_func();
    try
        throwing_func();
    catch
        when MyException => pass;
    end;
    return 0;
end;
