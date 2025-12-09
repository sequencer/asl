// RUN: asl-json-backend --no-std %s > %t.json
// RUN: asl-opt --canonicalize --run-asl-to-emitc --emitc --json-input %t.json | FileCheck %s

// CHECK: typedef enum asl_Status {
// CHECK:   asl_Status_OK = 0,
// CHECK:   asl_Status_ERROR = 1,
// CHECK:   asl_Status_PENDING = 2
// CHECK: } asl_Status;
type Status of enumeration {OK, ERROR, PENDING};

// CHECK: static inline void asl_init_status(asl_context* ctx) {
// CHECK:   ctx->status = asl_Status_OK;
// CHECK: }
var status: Status;

// CHECK: static inline void asl_init_status_init(asl_context* ctx) {
// CHECK:   ctx->status_init = asl_Status_ERROR;
// CHECK: }
var status_init: Status = ERROR;

type Point of (integer, integer);

type StatusA of (boolean, string, integer);

type RecTuple of (Point, StatusA);

// CHECK: static inline void asl_init_point(asl_context* ctx) {
// CHECK:   mpz_init_set_str(ctx->point.item0, "114", 10);
// CHECK:   mpz_init_set_str(ctx->point.item1, "514", 10);
// CHECK: }
var point: Point = (114, 514);

// CHECK: static inline void asl_init_statusA(asl_context* ctx) {
// CHECK:   ctx->statusA.item0 = false;
// CHECK:   ctx->statusA.item1 = "";
// CHECK:   mpz_init_set_str(ctx->statusA.item2, "0", 10);
// CHECK: }
var statusA: StatusA;

// CHECK: static inline void asl_init_recTuple(asl_context* ctx) {
// CHECK:   mpz_init_set_str(ctx->recTuple.item0.item0, "0", 10);
// CHECK:   mpz_init_set_str(ctx->recTuple.item0.item1, "0", 10);
// CHECK:   ctx->recTuple.item1.item0 = false;
// CHECK:   ctx->recTuple.item1.item1 = "";
// CHECK:   mpz_init_set_str(ctx->recTuple.item1.item2, "0", 10);
// CHECK: }
var recTuple: RecTuple;

// CHECK: static inline void asl_init_annoTuple(asl_context* ctx) {
// CHECK:   mpz_init_set_str(ctx->annoTuple.item0, "0", 10);
// CHECK:   mpz_init_set_str(ctx->annoTuple.item1, "0", 10);
// CHECK: }
var annoTuple: (integer, integer);
