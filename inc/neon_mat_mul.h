#ifndef __NEON_MAT_MUL_H__
#define __NEON_MAT_MUL_H__
void neon_mat_mul(const float *matrix_a, const float *matrix_b, float *matrix_r);
void neon_mat_mul2(const float *matrix_a, const float *matrix_b, float *matrix_r);
void neon_mat_mul3(const float *matrix_a, const float *matrix_b, float *matrix_r);
#endif
