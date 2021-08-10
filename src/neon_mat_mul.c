#include <arm_neon.h>
#include "neon_mat_mul.h"
#include "android_arm_util.h"

void neon_mat_mul(const float *matrix_a, const float *matrix_b, float *matrix_r)
{
    FUNC_ENTRANCE_LOG;
    float32x4_t a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3;
    
    a0 = vld1q_f32(matrix_a);
    a1 = vld1q_f32(matrix_a + 4);
    a2 = vld1q_f32(matrix_a + 8);
    a3 = vld1q_f32(matrix_a + 12);

    r0 = vmovq_n_f32(0);
    r1 = vmovq_n_f32(0);
    r2 = vmovq_n_f32(0);
    r3 = vmovq_n_f32(0);

    b0 = vld1q_f32(matrix_b);
    // vst1_f32(matrix_r, vget_low_f32(b0));            // Show vget_low_f32(b0)

    r0 = vfmaq_laneq_f32(r0, a0, b0, 0);
    r0 = vfmaq_laneq_f32(r0, a1, b0, 1);
    r0 = vfmaq_laneq_f32(r0, a2, b0, 2);
    r0 = vfmaq_laneq_f32(r0, a3, b0, 3);
    vst1q_f32(matrix_r, r0);

    b1 = vld1q_f32(matrix_b + 4);
    r1 = vfmaq_laneq_f32(r1, a0, b1, 0);
    r1 = vfmaq_laneq_f32(r1, a1, b1, 1);
    r1 = vfmaq_laneq_f32(r1, a2, b1, 2);
    r1 = vfmaq_laneq_f32(r1, a3, b1, 3);
    vst1q_f32(matrix_r + 4, r1);

    b2 = vld1q_f32(matrix_b + 8);
    r2 = vfmaq_laneq_f32(r2, a0, b2, 0);
    r2 = vfmaq_laneq_f32(r2, a1, b2, 1);
    r2 = vfmaq_laneq_f32(r2, a2, b2, 2);
    r2 = vfmaq_laneq_f32(r2, a3, b2, 3);
    vst1q_f32(matrix_r + 8, r2);

    b3 = vld1q_f32(matrix_b + 12);
    r3 = vfmaq_laneq_f32(r3, a0, b3, 0);
    r3 = vfmaq_laneq_f32(r3, a1, b3, 1);
    r3 = vfmaq_laneq_f32(r3, a2, b3, 2);
    r3 = vfmaq_laneq_f32(r3, a3, b3, 3);
    vst1q_f32(matrix_r + 12, r3);
    
    FUNC_EXIT_LOG;
}

void neon_mat_mul2(const float *matrix_a, const float *matrix_b, float *matrix_r)
{
    FUNC_ENTRANCE_LOG;
    float32x4_t a0, a1, a2, a3, b0, b1, b2, b3, r;

    a0 = vld1q_f32(matrix_a);
    a1 = vld1q_f32(matrix_a + 4);
    a2 = vld1q_f32(matrix_a + 8);
    a3 = vld1q_f32(matrix_a + 12);
    
    b0 = vld1q_f32(matrix_b);
    b1 = vld1q_f32(matrix_b + 4);
    b2 = vld1q_f32(matrix_b + 8);
    b3 = vld1q_f32(matrix_b + 12);

    r = vmulq_lane_f32(a0, vget_low_f32(b0), 0);
    r = vmlaq_lane_f32(r, a1, vget_low_f32(b0), 1);
    r = vmlaq_lane_f32(r, a2, vget_high_f32(b0), 0);
    r = vmlaq_lane_f32(r, a3, vget_high_f32(b0), 1);
    vst1q_f32(matrix_r, r);

    r = vmulq_lane_f32(a0, vget_low_f32(b1), 0);
    r = vmlaq_lane_f32(r, a1, vget_low_f32(b1), 1);
    r = vmlaq_lane_f32(r, a2, vget_high_f32(b1), 0);
    r = vmlaq_lane_f32(r, a3, vget_high_f32(b1), 1);
    vst1q_f32(matrix_r+4, r);

    r = vmulq_lane_f32(a0, vget_low_f32(b2), 0);
    r = vmlaq_lane_f32(r, a1, vget_low_f32(b2), 1);
    r = vmlaq_lane_f32(r, a2, vget_high_f32(b2), 0);
    r = vmlaq_lane_f32(r, a3, vget_high_f32(b2), 1);
    vst1q_f32(matrix_r+8, r);

    r = vmulq_lane_f32(a0, vget_low_f32(b3), 0);
    r = vmlaq_lane_f32(r, a1, vget_low_f32(b3), 1);
    r = vmlaq_lane_f32(r, a2, vget_high_f32(b3), 0);
    r = vmlaq_lane_f32(r, a3, vget_high_f32(b3), 1);
    vst1q_f32(matrix_r+12, r);
    FUNC_EXIT_LOG;
}

void neon_mat_mul3(const float *matrix_a, const float *matrix_b, float *matrix_r)
{
    FUNC_ENTRANCE_LOG;
    float32x4_t a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3;

    a0 = vld1q_f32(matrix_a);
    a1 = vld1q_f32(matrix_a + 4);
    a2 = vld1q_f32(matrix_a + 8);
    a3 = vld1q_f32(matrix_a + 12);

    b0 = vld1q_f32(matrix_b);
    b1 = vld1q_f32(matrix_b + 4);
    b2 = vld1q_f32(matrix_b + 8);
    b3 = vld1q_f32(matrix_b + 12);

    r0 = vmulq_lane_f32(a0, vget_low_f32(b0), 0);
    r0 = vmlaq_lane_f32(r0, a0, vget_low_f32(b1), 1);
    r0 = vmlaq_lane_f32(r0, a0, vget_high_f32(b2), 0);
    r0 = vmlaq_lane_f32(r0, a0, vget_high_f32(b2), 1);

    r1 = vmulq_lane_f32(a1, vget_low_f32(b0), 0);
    r1 = vmlaq_lane_f32(r1, a1, vget_low_f32(b1), 1);
    r1 = vmlaq_lane_f32(r1, a1, vget_high_f32(b2), 0);
    r1 = vmlaq_lane_f32(r1, a1, vget_high_f32(b3), 1);

    r2 = vmulq_lane_f32(a2, vget_low_f32(b0), 0);
    r2 = vmlaq_lane_f32(r2, a2, vget_low_f32(b1), 1);
    r2 = vmlaq_lane_f32(r2, a2, vget_high_f32(b2), 0);
    r2 = vmlaq_lane_f32(r2, a2, vget_high_f32(b3), 1);

    r3 = vmulq_lane_f32(a3, vget_low_f32(b0), 0);
    r3 = vmlaq_lane_f32(r3, a3, vget_low_f32(b1), 1);
    r3 = vmlaq_lane_f32(r3, a3, vget_high_f32(b2), 0);
    r3 = vmlaq_lane_f32(r3, a3, vget_high_f32(b3), 1);

    vst1q_f32(matrix_r, r0);
    vst1q_f32(matrix_r + 4, r1);
    vst1q_f32(matrix_r + 8, r2);
    vst1q_f32(matrix_r + 12, r3);


    FUNC_EXIT_LOG;
}
