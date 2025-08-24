// RUN: asl-json-backend %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func myfunction( s : string)
begin
   case s of
       when "hello" =>
           println "helloworld";
        otherwise =>
           return;
    end;
end;

func main () => integer
begin
  myfunction ("Hello");
  myfunction ("helloworld\n");

  return 0;
end;
