#ifndef NN_H_
#define NN_H_
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#define MAT_AT(m, i, j) (m).es[(i) * (m).stride + (j)]
#define MAT_PRINT(m) mat_print(m)
#define NN_PRINT(nn) neural_print(nn)
#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]
// structure for the matrix
typedef struct {
  int rows;
  int cols;
  int stride;
  float *es;
} Mat;

typedef struct {
  int count;
  Mat *ws; // weights = count
  Mat *bs; // biases = count
  Mat *as; // value of perceptron = count +1 (including the ) input to
           // perceptron
} NN;

float rand_float(void);
void mat_rand(Mat m);
void mat_sig(Mat m);
float sigmoidf(float x);
Mat mat_alloc(int rows, int cols);
void mat_dot(Mat dst, Mat a, Mat b);
void mat_fill(Mat a, float fillValue);
void mat_sum(Mat dst, Mat a);
void mat_print(Mat m);
Mat mat_row(Mat m, int row);
void mat_copy(Mat dst, Mat src);
void neural_print(NN n);
void nn_rand(NN nn);
void nn_forward(NN nn);
void nn_finite_diff(NN m, NN g, float eps, Mat ti, Mat to);
float nn_cost(NN nn, Mat ti, Mat to);
void nn_learn(NN nn, NN g , float rate);
NN nn_alloc(int *arch, int count);
void nn_backprop(NN nn, NN g , Mat ti, Mat to);
void nn_zero(NN nn);
void mat_save(FILE *out, Mat m);
Mat mat_load(FILE *in);
#endif
