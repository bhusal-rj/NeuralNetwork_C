#define NN_IMPLEMENTATION
#include "nn.h"

int main(void) {
  Mat t = mat_alloc(4, 3);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
    int row= i*2 + j;
      MAT_AT(t, row, 0) = i;
      MAT_AT(t, row, 1) = j;
      MAT_AT(t, row, 2) = i & j;
    }
  }
  // for (int i = 0; i < 2; ++i) {
  //     for (int j = 0; j < 2; ++j) {
  //         int row = i*2 + j;
  //         MAT_AT(t, row, 0) = i;
  //         MAT_AT(t, row, 1) = j;
  //         MAT_AT(t, row, 2) = i^j;
  //     }
  // }

  const char *out_file_path = "add.mat";
  FILE *out = fopen(out_file_path, "wb");
  if (out == NULL) {
    fprintf(stderr, "ERROR: could not open file %s\n", out_file_path);
    return 1;
  }
  mat_save(out, t);
  fclose(out);

  printf("Generated %s\n", out_file_path);

  return 0;
}
