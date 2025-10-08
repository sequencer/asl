// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --canonicalize --run-asl-to-emitc --emitc --json-input %t.json | FileCheck %s

// CHECK: #include <gmp.h>
// CHECK: #include <stdint.h>
// CHECK: #include <stdbool.h>
// CHECK: #include <string.h>
// CHECK: typedef struct asl_context {
// CHECK:   uint8_t Bits1;
// CHECK:   uint16_t Bits9;
// CHECK:   uint32_t Bits17;
// CHECK:   uint64_t Bits33;
// CHECK:   struct { uint64_t words[2]; } Bits65;
// CHECK:   uint8_t Bits1_init;
// CHECK:   uint16_t Bits9_init;
// CHECK:   uint32_t Bits17_init;
// CHECK:   uint64_t Bits33_init;
// CHECK:   struct { uint64_t words[2]; } Bits65_init;
// CHECK:   mpz_t Int;
// CHECK:   mpz_t Int_init;
// CHECK:   mpq_t Real;
// CHECK:   mpq_t RealPi_init;
// CHECK:   bool Bool_init;
// CHECK:   bool BoolT_init;
// CHECK:   const char* message;
// CHECK: } asl_context;
var Bits1 : bits(1);
var Bits9 : bits(9);
var Bits17 : bits(17);
var Bits33 : bits(33);
var Bits65 : bits(65);

// CHECK: static inline void asl_init_Bits1_init(asl_context* ctx) {
// CHECK:   ctx->Bits1_init = 1u;
// CHECK: }
var Bits1_init : bits(1) = '1';

// CHECK: static inline void asl_init_Bits9_init(asl_context* ctx) {
// CHECK:   ctx->Bits9_init = 341u;
// CHECK: }
var Bits9_init : bits(9) = '101010101';

// CHECK: static inline void asl_init_Bits17_init(asl_context* ctx) {
// CHECK:   ctx->Bits17_init = 87381u;
// CHECK: }
var Bits17_init : bits(17) = '10101010101010101';

// CHECK: static inline void asl_init_Bits33_init(asl_context* ctx) {
// CHECK:   ctx->Bits33_init = 5726623061ULL;
// CHECK: }
var Bits33_init : bits(33) = '101010101010101010101010101010101';

// CHECK: static inline void asl_init_Bits65_init(asl_context* ctx) {
// CHECK:   ctx->Bits65_init.words[0] = 6148914691236517205ULL;
// CHECK:   ctx->Bits65_init.words[1] = 1ULL;
// CHECK: }
var Bits65_init : bits(65) = '10101010101010101010101010101010101010101010101010101010101010101';

// CHECK: static inline void asl_init_Int(asl_context* ctx) {
// CHECK:   mpz_init_set_str(ctx->Int, "0", 10);
// CHECK: }
var Int : integer;

// CHECK: static inline void asl_init_Int_init(asl_context* ctx) {
// CHECK:   mpz_init_set_str(ctx->Int_init, "42", 10);
// CHECK: }
var Int_init : integer = 42;

// CHECK: static inline void asl_init_Real(asl_context* ctx) {
// CHECK:   mpq_init(ctx->Real);
// CHECK:   mpq_set_str(ctx->Real, "0", 10);
// CHECK:   mpq_canonicalize(ctx->Real);
// CHECK: }
var Real : real;

// CHECK: static inline void asl_init_RealPi_init(asl_context* ctx) {
// CHECK:   mpq_init(ctx->RealPi_init);
// CHECK:   mpq_set_str(ctx->RealPi_init, "157/50", 10);
// CHECK:   mpq_canonicalize(ctx->RealPi_init);
// CHECK: }
var RealPi_init : real = 3.14;

// TODO: real folding
// var RealMinusOne_init : real = -1.0;

// CHECK: static inline void asl_init_Bool_init(asl_context* ctx) {
// CHECK:   ctx->Bool_init = false;
// CHECK: }
var Bool : boolean;

// CHECK: static inline void asl_init_BoolT_init(asl_context* ctx) {
// CHECK:   ctx->BoolT_init = true;
// CHECK: }
var BoolT_init : boolean = TRUE;

// CHECK: static inline void asl_init_String(asl_context* ctx) {
// CHECK:   ctx->String = "";
// CHECK: }
var String: string;

// CHECK: static inline void asl_init_String_init(asl_context* ctx) {
// CHECK:   ctx->String_init = "Hello World!";
// CHECK: }
var String_init: string = "Hello World!";