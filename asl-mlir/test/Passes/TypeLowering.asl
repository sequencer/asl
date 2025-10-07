// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --canonicalize --run-asl-to-emitc --emitc --json-input %t.json | FileCheck %s

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
