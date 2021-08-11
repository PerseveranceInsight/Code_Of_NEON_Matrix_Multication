#include <arm_neon.h>
#include <stdio.h>
#include "neon_mat_mul.h"
#include "android_arm_util.h"

/*
 *   $$
 *   A = \begin{bmatrix}
 *   a_{00}, a_{01}, a_{02}, a_{03}\\
 *   a_{10}, a_{11}, a_{12}, a_{13}\\
 *   a_{20}, a_{21}, a_{22}, a_{23}\\
 *   a_{30}, a_{31}, a_{32}, a_{33}
 *   \end{bmatrix}
 *   $$
 *   
 *   $$
 *   B = \begin{bmatrix}
 *   b_{00}, b_{01}, b_{02}, b_{03}\\
 *   b_{10}, b_{11}, b_{12}, b_{13}\\
 *   b_{20}, b_{21}, b_{22}, b_{23}\\
 *   b_{30}, b_{31}, b_{32}, b_{33}
 *   \end{bmatrix}
 *   $$
 *   
 *   $$
 *   R = \begin{bmatrix}
 *   r_{00}, r_{01}, r_{02}, r_{03}\\
 *   r_{10}, r_{11}, r_{12}, r_{13}\\
 *   r_{20}, r_{21}, r_{22}, r_{23}\\
 *   r_{30}, r_{31}, r_{32}, r_{33}
 *   \end{bmatrix}
 *   $$
 *   
 *   $$
 *   \begin{equation}
 *   \begin{split}
 *   \tilde{a}_{0} &= (a_{30}, a_{20}, a_{10}, a_{00}) \\
 *   \tilde{a}_{1} &= (a_{31}, a_{21}, a_{11}, a_{01}) \\
 *   \tilde{a}_{2} &= (a_{32}, a_{22}, a_{12}, a_{02}) \\
 *   \tilde{a}_{3} &= (a_{33}, a_{23}, a_{13}, a_{03}) \\
 *   \tilde{b}_{0} &= (b_{30}, b_{20}, b_{10}, b_{00}) \\
 *   \tilde{b}_{1} &= (b_{31}, b_{21}, b_{11}, b_{01}) \\
 *   \tilde{b}_{2} &= (b_{32}, b_{22}, b_{12}, b_{02}) \\
 *   \tilde{b}_{3} &= (b_{33}, b_{23}, b_{13}, b_{03}) \\
 *   \tilde{r}_{0} &= (r_{30}, r_{20}, r_{10}, r_{00}) \\
 *   \tilde{r}_{1} &= (r_{31}, r_{21}, r_{11}, r_{01}) \\
 *   \tilde{r}_{2} &= (r_{32}, r_{22}, r_{12}, r_{02}) \\
 *   \tilde{r}_{3} &= (r_{33}, r_{23}, r_{13}, r_{03}) \\
 *   \end{split}
 *   \end{equation}
 *   $$
 *   
 *   $$
 *   r_{00} = a_{00}b_{00} + a_{01}b_{10} + a_{02}b_{20} + a_{03}b_{30}
 *   $$
 *   
 *   $$
 *   \begin{eqaution}
 *   \begin{split}
 *   \tilde{a}_{0} \times \text{low }(\tilde{b}_{0},0) &= a_{30}b_{00} + a_{20}b_{00} + a_{10}b_{00} + a_{00}b_{00} \\
 *   \tilde{a}_{1} \times \text{low }(\tilde{b}_{0},1) &= a_{31}b_{10} + a_{21}b_{10} + a_{11}b_{10} + a_{01}b_{10} \\
 *   \tilde{a}_{2} \times \text{high}(\tilde{b}_{0},0) &= a_{32}b_{20} + a_{22}b_{20} + a_{12}b_{20} + a_{02}b_{20} \\
 *   \tilde{a}_{3} \times \text{high}(\tilde{b}_{0},1) &= a_{33}b_{30} + a_{23}b_{30} + a_{13}b_{30} + a_{03}b_{30} \\
 *   \tilde{r}_{0}                                     &=    r_{30}         r_{20}         r_{10}         r_{00}
 *   \end{split}
 *   \end{equation}
 *   $$
 */

void neon_mat_mul(const float *matrix_a, const float *matrix_b, float *matrix_r)
{
    FUNC_ENTRANCE_LOG;
    double t0, t1, time;
    float32x4_t a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3;
    t0 = now_ns();

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
    t1 = now_ns();
    time = t1 - t0;
    ex_log(LOG_DEBUG, "%s spends time %f", __func__, time);
    printf("%s spends time %f\n", __func__, time);
    FUNC_EXIT_LOG;
}

void neon_mat_mul2(const float *matrix_a, const float *matrix_b, float *matrix_r)
{
    FUNC_ENTRANCE_LOG;
    double t0, t1, time;
    float32x4_t a0, a1, a2, a3, b0, b1, b2, b3, r;
    t0 = now_ns();
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
    t1 = now_ns();
    time = t1 - t0;
    ex_log(LOG_DEBUG, "%s spends time %f", __func__, time);
    printf("%s spends time %f\n", __func__, time);
    FUNC_EXIT_LOG;
}

void neon_mat_mul3(const float *matrix_a, const float *matrix_b, float *matrix_r)
{
    FUNC_ENTRANCE_LOG;
    double t0, t1, time;
    float32x4_t a0, a1, a2, a3, b0, b1, b2, b3, r0, r1, r2, r3;
    t0 = now_ns();

    a0 = vld1q_f32(matrix_a);
    a1 = vld1q_f32(matrix_a + 4);
    a2 = vld1q_f32(matrix_a + 8);
    a3 = vld1q_f32(matrix_a + 12);

    b0 = vld1q_f32(matrix_b);
    b1 = vld1q_f32(matrix_b + 4);
    b2 = vld1q_f32(matrix_b + 8);
    b3 = vld1q_f32(matrix_b + 12);

    r0 = vmulq_lane_f32(a0, vget_low_f32(b0), 0);
    r0 = vmlaq_lane_f32(r0, a1, vget_low_f32(b0), 1);
    r0 = vmlaq_lane_f32(r0, a2, vget_high_f32(b0), 0);
    r0 = vmlaq_lane_f32(r0, a3, vget_high_f32(b0), 1);

    r1 = vmulq_lane_f32(a0, vget_low_f32(b1), 0);
    r1 = vmlaq_lane_f32(r1, a1, vget_low_f32(b1), 1);
    r1 = vmlaq_lane_f32(r1, a2, vget_high_f32(b1), 0);
    r1 = vmlaq_lane_f32(r1, a3, vget_high_f32(b1), 1);

    r2 = vmulq_lane_f32(a0, vget_low_f32(b2), 0);
    r2 = vmlaq_lane_f32(r2, a1, vget_low_f32(b2), 1);
    r2 = vmlaq_lane_f32(r2, a2, vget_high_f32(b2), 0);
    r2 = vmlaq_lane_f32(r2, a3, vget_high_f32(b2), 1);

    r3 = vmulq_lane_f32(a0, vget_low_f32(b3), 0);
    r3 = vmlaq_lane_f32(r3, a1, vget_low_f32(b3), 1);
    r3 = vmlaq_lane_f32(r3, a2, vget_high_f32(b3), 0);
    r3 = vmlaq_lane_f32(r3, a3, vget_high_f32(b3), 1);

    vst1q_f32(matrix_r, r0);
    vst1q_f32(matrix_r + 4, r1);
    vst1q_f32(matrix_r + 8, r2);
    vst1q_f32(matrix_r + 12, r3);
    t1 = now_ns();
    time = t1 - t0;
    ex_log(LOG_DEBUG, "%s spends time %f", __func__, time);
    printf("%s spends time %f\n", __func__, time);
    FUNC_EXIT_LOG;
}

void mat_mul_c_procedure(const float *matrix_a, const float *matrix_b, float *matrix_r)
{
    FUNC_ENTRANCE_LOG;
    double t0, t1, time;
    int i = 0, j = 0, k = 0;
    const float *p_element_a, *p_element_b; 
    float *p_element_r;
    t0 = now_ns();

    for (i = 0; i<4; i++) 
    {
        for (j = 0; j<4; j++)
        {
            p_element_r = matrix_r + j*4 + i;
            for(k = 0; k<4; k++)
            {
               p_element_a = matrix_a + i + k*4;
               p_element_b = matrix_b + j*4 + k;
               *p_element_r += *p_element_a * *p_element_b;
            }
        }
    }
    t1 = now_ns();
    time = t1 - t0;
    ex_log(LOG_DEBUG, "%s spends time %f", __func__, time);
    printf("%s spends time %f\n", __func__, time);
    FUNC_EXIT_LOG;
}
