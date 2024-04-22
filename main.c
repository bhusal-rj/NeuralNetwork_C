#include "nn.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define NN_IMPLEMENTATION

const int stride=3;
float td[] = {0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0};


int main() {
  srand(time(0));
  int n = sizeof(td) / sizeof(td[0]) / stride;
  Mat ti = {.rows = n, .cols = 2, .stride = stride, .es = td};
  Mat to = {.rows = n, .cols = 1, .stride = stride, .es = &td[2]};
  // input layer, hidden layer/layers, output
  int arch[] = {2, 2, 1};
  int g[] = {2, 2, 1};
  int length = sizeof(arch) / sizeof(arch[0]);
  NN nn = nn_alloc(arch, length);
  NN g2 = nn_alloc(g, length);
  NN_PRINT(nn);
  nn_rand(nn);
  mat_copy(NN_INPUT(nn), mat_row(ti, 1));
  nn_forward(nn);
  printf("cost = %f\n", nn_cost(nn, ti, to));
  for (int i = 0; i < 1000000; i++) {
    // nn_finite_diff(nn, g2, 1e-1, ti, to);
    nn_backprop(nn, g2, ti, to);
    nn_learn(nn, g2, 1e-1);
  }
  printf("cost = %f\n", nn_cost(nn, ti, to));
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      MAT_AT(NN_INPUT(nn), 0, 0) = i;
      MAT_AT(NN_INPUT(nn), 0, 1) = j;
      nn_forward(nn);
      printf("%d ^ %d = %f \n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
    }
  }
  NN_PRINT(nn);
}
// float forward_xor(Xor m) {
//   // for n layer we can observe the pattern
//   mat_dot(m.a1, m.a0, m.w1);
//   mat_sum(m.a1, m.b1);
//   mat_sig(m.a1);

//   // pass the data of a1 into the second layer
//   mat_dot(m.a2, m.a1, m.w2);
//   mat_sum(m.a2, m.b2);
//   mat_sig(m.a2);
//   return *m.a2.es;
// }

// float cost(Xor m, Mat training_input, Mat training_output) {
//   // training rows must be equal to training columns
//   assert(training_input.rows == training_output.rows);
//   // no of cols of trainig must be equal to result of the xor network
//   assert(training_output.cols == m.a2.cols);
//   float cost = 0;

//   int n = training_input.rows;
//   int cols = training_input.cols;
//   for (int i = 0; i < n; i++) {
//     // input matrix
//     Mat x = mat_row(training_input, i);
//     Mat y = mat_row(training_output, i);
//     // from the training input copy the ith row into the training input
//     mat_copy(m.a0, mat_row(training_input, i));

//     // goto forward of the nn and it returns the a2 as the output
//     forward_xor(m);

//     for (int j = 0; j < cols; j++) {
//       float diff = MAT_AT(m.a2, 0, j) - MAT_AT(y, 0, j);
//       cost += diff * diff;
//     }
//   }

//   return cost / n;
// }

// float finite_difference(Xor m, Xor g, float eps, Mat ti, Mat to) {
//   float saved;
//   float c = cost(m, ti, to);
//   for (int i = 0; i < m.w1.rows; i++) {
//     for (int j = 0; j < m.w1.cols; j++) {
//       saved = MAT_AT(m.w1, i, j);
//       MAT_AT(m.w1, i, j) += eps;
//       MAT_AT(g.w1, i, j) = (cost(m, ti, to) - c) / eps;
//       MAT_AT(m.w1, i, j) = saved;
//     }
//   }
//   for (int i = 0; i < m.b1.rows; i++) {
//     for (int j = 0; j < m.b1.cols; j++) {
//       saved = MAT_AT(m.b1, i, j);
//       MAT_AT(m.b1, i, j) += eps;
//       MAT_AT(g.b1, i, j) = (cost(m, ti, to) - c) / eps;
//       MAT_AT(m.b1, i, j) = saved;
//     }
//   }

//   for (int i = 0; i < m.w2.rows; i++) {
//     for (int j = 0; j < m.w2.cols; j++) {
//       saved = MAT_AT(m.w2, i, j);
//       MAT_AT(m.w2, i, j) += eps;
//       MAT_AT(g.w2, i, j) = (cost(m, ti, to) - c) / eps;
//       MAT_AT(m.w2, i, j) = saved;
//     }
//   }
//   for (int i = 0; i < m.b2.rows; i++) {
//     for (int j = 0; j < m.b2.cols; j++) {
//       saved = MAT_AT(m.b2, i, j);
//       MAT_AT(m.b2, i, j) += eps;
//       MAT_AT(g.b2, i, j) = (cost(m, ti, to) - c) / eps;
//       MAT_AT(m.b2, i, j) = saved;
//     }
//   }
// }

// void xor_learn(Xor m, Xor g, float rate) {

//   for (int i = 0; i < m.w1.rows; i++) {
//     for (int j = 0; j < m.w1.cols; j++) {
//       MAT_AT(m.w1, i, j) -= rate * MAT_AT(g.w1, i, j);
//     }
//   }
//   for (int i = 0; i < m.b1.rows; i++) {
//     for (int j = 0; j < m.b1.cols; j++) {
//       MAT_AT(m.b1, i, j) -= rate * MAT_AT(g.b1, i, j);
//     }
//   }

//   for (int i = 0; i < m.w2.rows; i++) {
//     for (int j = 0; j < m.w2.cols; j++) {
//       MAT_AT(m.w2, i, j) -= rate * MAT_AT(g.w2, i, j);
//     }
//   }
//   for (int i = 0; i < m.b2.rows; i++) {
//     for (int j = 0; j < m.b2.cols; j++) {
//       MAT_AT(m.b2, i, j) -= rate * MAT_AT(g.b2, i, j);
//     }
//   }
// }
// int main() {
//   srand(time(0));
//   int strid = 3;
//   // move three position that is stride to get the training input data
//   int n = sizeof(td) / sizeof(td[0]) / stride;
//   Mat ti = {.rows = n, .cols = 2, .stride = stride, .es = td};
//   Mat to = {.rows = n, .cols = 1, .stride = stride, .es = &td[2]};

//   Xor m = xor_alloc();
//   Xor g = xor_alloc();
//   mat_rand(m.w1);
//   mat_rand(m.b1);
//   mat_rand(m.w2);
//   mat_rand(m.b2);

//   float eps = 1e-1;
//   float rate = 1e-1;
//   printf("%f\n", cost(m, ti, to));
//   for (int i = 0; i < 1000000; i++) {
//     finite_difference(m, g, eps, ti, to);
//     xor_learn(m, g, rate);
//   }
//   printf("%f\n", cost(m, ti, to));
//   for (int i = 0; i < 2; i++) {
//     for (int j = 0; j < 2; j++) {
//       // this is just the random value
//       MAT_AT(m.a0, 0, 0) = i;
//       MAT_AT(m.a0, 0, 1) = j;
//       forward_xor(m);
//       float y = *m.a2.es;
//       printf("%d ^ %d = %f \n", i, j, y);
//     }
//   }
//   return 0;
// }
