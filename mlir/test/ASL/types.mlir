// RUN: asl-opt %s | asl-opt | FileCheck %s

// Test basic ASL types with attributes

module {
  // CHECK-LABEL: func.func @test_basic_types
  func.func @test_basic_types() {
    // Test integer types with constraints
    %int_unconstrained = "test.value"() : () -> !asl.int
    %int_constrained = "test.value"() : () -> !asl.int<#asl.constraints<[#asl.constraint_exact<42>]>>
    %int_range = "test.value"() : () -> !asl.int<#asl.constraints<[#asl.constraint_range<0, 255>]>>
    
    // Test bitvector types
    %bits_simple = "test.value"() : () -> !asl.bits<32>
    %bits_with_fields = "test.value"() : () -> !asl.bits<8, [#asl.bitfield_simple<"flag", [#asl.slice_single<0>]>]>
    
    // Test composite types
    %enum_type = "test.value"() : () -> !asl.enum<["RED", "GREEN", "BLUE"]>
    %tuple_type = "test.value"() : () -> !asl.tuple<!asl.int, !asl.bool>
    %array_type = "test.value"() : () -> !asl.array<!asl.int, #asl.array_length<10>>
    
    // Test structural types
    %record_type = "test.value"() : () -> !asl.record<"Point", [#asl.field<"x", !asl.real>, #asl.field<"y", !asl.real>]>
    
    return
  }

  // CHECK-LABEL: func.func @test_complex_types
  func.func @test_complex_types() {
    // Test complex nested types
    %complex_record = "test.value"() : () -> !asl.record<"ComplexStruct", [
      #asl.field<"id", !asl.int<#asl.constraints<[#asl.constraint_range<1, 1000>]>>>,
      #asl.field<"flags", !asl.bits<8, [
        #asl.bitfield_simple<"enable", [#asl.slice_single<0>]>,
        #asl.bitfield_simple<"priority", [#asl.slice_range<7, 4>]>
      ]>>,
      #asl.field<"data", !asl.array<!asl.real, #asl.array_length<16>>>
    ]>
    
    return
  }
}
