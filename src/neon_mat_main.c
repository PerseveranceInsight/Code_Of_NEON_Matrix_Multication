#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "android_arm_util.h"
#include "neon_mat_mul.h"

int main(int argc, char* argv[])
{
    FUNC_ENTRANCE_LOG;
    float *ptr_f= NULL;
    float matrix_a[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
    float matrix_b[16] = {0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f, -9.0f, -10.0f, -11.0f, -12.0f, -13.0f, -14.0f, -15.0f};
    float matrix_r[16] = {0};
    
    ex_log(LOG_DEBUG, "matrix_a");
    printf("matrix_a col order\n");
    for (int i = 0; i<4; i++)
    {
        ptr_f = matrix_a + i*4;
        ex_log(LOG_DEBUG, "%f %f %f %f", *ptr_f,
                                         *(ptr_f+1),
                                         *(ptr_f+2),
                                         *(ptr_f+3));
        printf("%f %f %f %f\n", *ptr_f,
                                *(ptr_f+1),
                                *(ptr_f+2),
                                *(ptr_f+3));
    }

    ex_log(LOG_DEBUG, "matrix_b");
    printf("\nmatrix_b col order\n");
    for (int i = 0; i<4; i++)
    {
        ptr_f = matrix_b + i*4;
        ex_log(LOG_DEBUG, "%f %f %f %f", *ptr_f,
                                         (*ptr_f+1),
                                         (*ptr_f+2),
                                         (*ptr_f+3));
        printf("%f %f %f %f\n", *ptr_f,
                                *(ptr_f+1),
                                *(ptr_f+2),
                                *(ptr_f+3));
    }
    neon_mat_mul(matrix_a, matrix_b, matrix_r);

    ex_log(LOG_DEBUG, "matrix_r");
    printf("\nmethod 1: matrix_r col order\n");
    for (int i = 0; i<4; i++)
    {
        ptr_f = matrix_r + i*4;
        ex_log(LOG_DEBUG, "%f %f %f %f", *ptr_f,
                                         (*ptr_f+1),
                                         (*ptr_f+2),
                                         (*ptr_f+3));
        printf("%f %f %f %f\n", *ptr_f,
                                *(ptr_f+1),
                                *(ptr_f+2),
                                *(ptr_f+3));
    }

    memset(matrix_r, 0, sizeof(float)*16);
    neon_mat_mul2(matrix_a, matrix_b, matrix_r);
    ex_log(LOG_DEBUG, "matrix_r");
    printf("\nmethod 2: matrix_r col order\n");
    for (int i = 0; i<4; i++)
    {
        ptr_f = matrix_r + i*4;
        ex_log(LOG_DEBUG, "%f %f %f %f", *ptr_f,
                                         (*ptr_f+1),
                                         (*ptr_f+2),
                                         (*ptr_f+3));
        printf("%f %f %f %f\n", *ptr_f,
                                *(ptr_f+1),
                                *(ptr_f+2),
                                *(ptr_f+3));
    }

    memset(matrix_r, 0, sizeof(float)*16);
    neon_mat_mul3(matrix_a, matrix_b, matrix_r);
    ex_log(LOG_DEBUG, "matrix_r");
    printf("\nmethod 3: matrix_r col order\n");
    for (int i = 0; i<4; i++)
    {
        ptr_f = matrix_r + i*4;
        ex_log(LOG_DEBUG, "%f %f %f %f", *ptr_f,
                                         (*ptr_f+1),
                                         (*ptr_f+2),
                                         (*ptr_f+3));
        printf("%f %f %f %f\n", *ptr_f,
                                *(ptr_f+1),
                                *(ptr_f+2),
                                *(ptr_f+3));
    }

    FUNC_EXIT_LOG;
    return 0;
}
