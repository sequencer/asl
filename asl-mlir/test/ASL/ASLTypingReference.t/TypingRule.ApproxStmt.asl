// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --json-input %t.json | FileCheck %s

// CHECK: "builtin.module"() ({

func falls_through()
begin
    pass;
    var x : integer = 5;
    x = 7;
    assert x == 7;
    println x;
    pragma require_positive x;
end;

func returns_value() => integer
begin
    return 42;
end;

type invalid_state of exception{-};
func throws_exception() => integer
begin
    throw invalid_state{-};
end;

func sequencing1() => integer
begin
    // The abstract configuration is determined by the first statement.
    throw invalid_state{-};
    var x = 5;
end;

func sequencing2() => integer
begin
    // The abstract configuration is determined by the second statement.
    pass;
    return 5;
end;

func conditional(flag: boolean) => integer
begin
    // The abstract configuration is determined by "joining"
    // the abstract configurations of each of the statements
    // comprising the conditional. statement.
    if flag then
        return 5;
    else
        throw invalid_state{-};
    end;
end;

func while_loop(flag: boolean) => integer
begin
    // The loop is conservatively treated as not terminating
    // by returning a value, throwing an exception or executing unreachable.
    while (flag) looplimit 2^128 do
        pass;
    end;
    return 0;
end;

func for_loop(upper_limit: integer) => integer
begin
    // The loop is conservatively treated as not terminating
    // by returning a value, throwing an exception or executing unreachable.
    for i = 0 to upper_limit do
        pass;
    end;
    return 0;
end;

func repeat_loop(upper_limit: integer) => integer
begin
    // The loop is conservatively treated as not terminating
    // by returning a value, throwing an exception or executing unreachable.
    repeat
        pass;
    until TRUE looplimit 2^128;
    return 0;
end;

func throwing_function() => integer
begin
    try
        return repeat_loop(1000);
    catch
        when invalid_state => return 0;
        otherwise => return 1;
    end;
end;

func main() => integer
begin
    return 0;
end;
