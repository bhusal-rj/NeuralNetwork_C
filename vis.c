#include "raylib.h"
#include <assert.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#define SV_IMPLEMENTATION
#include "sv.h"
#define NN_IMPLEMENTATION
#include "nn.h"
#include <math.h>
typedef int Errno;

#define IMG_FACTOR 80
#define IMG_WIDTH (40 * IMG_FACTOR)
#define IMG_HEIGHT (20 * IMG_FACTOR)

typedef struct {
  int *items;
  int count;
  int capacity;
} Arch;

typedef struct {
  float *items;
  int count;
  int capacity;
} Cost_Plot;

#define DA_INIT_CAP 256
#define da_append(da, item)                                                    \
  do {                                                                         \
    if ((da)->count >= (da)->capacity) {                                       \
      (da)->capacity = (da)->capacity == 0 ? DA_INIT_CAP : (da)->capacity * 2; \
      (da)->items =                                                            \
          realloc((da)->items, (da)->capacity * sizeof(*(da)->items));         \
      assert((da)->items != NULL && "Buy more RAM lol");                       \
    }                                                                          \
                                                                               \
    (da)->items[(da)->count++] = (item);                                       \
  } while (0)

char *args_shift(int *argc, char ***argv) {
  assert(*argc > 0);
  char *result = **argv;
  (*argc) -= 1;
  (*argv) += 1;
  return result;
}

void nn_render_raylib(NN nn, int rx, int ry, int rw, int rh) {
  Color low_color = {0xFF, 0x00, 0xFF, 0xFF};
  Color high_color = {0x00, 0xFF, 0x00, 0xFF};

  float neuron_radius = rh * 0.04;
  int layer_border_vpad = 100;
  int layer_border_hpad = 50;
  int nn_width = rw - layer_border_hpad;
  int nn_height = rh - layer_border_vpad;
  int nn_x = rx + rw / 2 - nn_width / 2;
  int nn_y = ry + rh / 2 - nn_height / 2;
  int arch_count = nn.count + 1;
  int layer_hpad = (nn_width / arch_count);

  for (int l = 0; l < arch_count; ++l) {
    int layer_vpad1 = nn_height / nn.as[l].cols;
    for (int i = 0; i < nn.as[l].cols; ++i) {
      int cx1 = nn_x + l * layer_hpad + layer_hpad / 2;
      int cy1 = nn_y + i * layer_vpad1 + layer_vpad1 / 2;
      if (l + 1 < arch_count) {
        int layer_vpad2 = nn_height / nn.as[l + 1].cols;
        for (int j = 0; j < nn.as[l + 1].cols; ++j) {
          int cx2 = nn_x + (l + 1) * layer_hpad + layer_hpad / 2;
          int cy2 = nn_y + j * layer_vpad2 + layer_vpad2 / 2;
          float value = sigmoidf(MAT_AT(nn.ws[l], j, i));

          // Draw the line
          high_color.a = floorf(255.f * value);
          float thick = rh * 0.004f;
          Vector2 start = {cx1, cy1};
          Vector2 end = {cx2, cy2};
          DrawLineEx(start, end, thick, ColorAlphaBlend(low_color, high_color, WHITE));

          // Draw the weight value in the middle of the line
          float weightValue = MAT_AT(nn.ws[l], j, i);
          Vector2 textPos = {(start.x + end.x) / 2, (start.y + end.y) / 2};
          char weightText[32];
          snprintf(weightText, sizeof(weightText), "%.2f", weightValue);
          DrawText(weightText, textPos.x - MeasureText(weightText, 10) / 2, textPos.y - 10, 10, WHITE);
        }
      }

      if (l > 0) {
        // Draw the neuron
        high_color.a = floorf(255.f * sigmoidf(MAT_AT(nn.bs[l - 1], 0, i)));
        DrawCircle(cx1, cy1, neuron_radius, ColorAlphaBlend(low_color, high_color, WHITE));

        // Draw the weight value at the middle of the node
        float weightValue = MAT_AT(nn.bs[l - 1], 0, i);
        Vector2 textPos = {cx1, cy1};
        char weightText[32];
        snprintf(weightText, sizeof(weightText), "%.2f", weightValue);
        DrawText(weightText, textPos.x - MeasureText(weightText, 10) / 2, textPos.y - 10, 10, WHITE);
      } else {
        // Draw the input neuron
        DrawCircle(cx1, cy1, neuron_radius, GRAY);
      }
    }
  }
}

void cost_plot_minmax(Cost_Plot plot, float *min, float *max) {
  *min = FLT_MAX;
  *max = FLT_MIN;
  for (int i = 0; i < plot.count; ++i) {
    if (*max < plot.items[i])
      *max = plot.items[i];
    if (*min > plot.items[i])
      *min = plot.items[i];
  }
}

void plot_cost(Cost_Plot plot, int rx, int ry, int rw, int rh) {
  float min, max;
  cost_plot_minmax(plot, &min, &max);
  if (min > 0)
    min = 0;
  int n = plot.count;
  if (n < 1000)
    n = 1000;
  for (int i = 0; i + 1 < plot.count; ++i) {
    float x1 = rx + (float)rw / n * i;
    float y1 = ry + (1 - (plot.items[i] - min) / (max - min)) * rh;
    float x2 = rx + (float)rw / n * (i + 1);
    float y2 = ry + (1 - (plot.items[i + 1] - min) / (max - min)) * rh;
    DrawLineEx((Vector2){x1, y1}, (Vector2){x2, y2}, rh * 0.005, RED);
  }
}

int main(int argc, char **argv) {
  const char *program = args_shift(&argc, &argv);

  if (argc <= 0) {
    fprintf(stderr, "Usage: %s <model.arch> <model.mat>\n", program);
    fprintf(stderr, "ERROR: no architecture file was provided\n");
    return 1;
  }
  const char *arch_file_path = args_shift(&argc, &argv);

  if (argc <= 0) {
    fprintf(stderr, "Usage: %s <model.arch> <model.mat>\n", program);
    fprintf(stderr, "ERROR: no data file was provided\n");
    return 1;
  }
  const char *data_file_path = args_shift(&argc, &argv);

  unsigned int buffer_len = 0;
  unsigned char *buffer = LoadFileData(arch_file_path, &buffer_len);
  if (buffer == NULL) {
    return 1;
  }

  String_View content = sv_from_parts((const char *)buffer, buffer_len);
  Arch arch = {0};
  content = sv_trim_left(content);
  while (content.count > 0 && isdigit(content.data[0])) {
    int x = sv_chop_u64(&content);
    da_append(&arch, x);
    content = sv_trim_left(content);
  }

  FILE *in = fopen(data_file_path, "rb");
  if (in == NULL) {
    fprintf(stderr, "ERROR: could not read file %s\n", data_file_path);
    return 1;
  }
  Mat t = mat_load(in);
  fclose(in);

  assert(arch.count > 1);
  int ins_sz = arch.items[0];
  int outs_sz = arch.items[arch.count - 1];
  assert(t.cols == ins_sz + outs_sz);

  Mat ti = {
      .rows = t.rows,
      .cols = ins_sz,
      .stride = t.stride,
      .es = &MAT_AT(t, 0, 0),
  };

  Mat to = {
      .rows = t.rows,
      .cols = outs_sz,
      .stride = t.stride,
      .es = &MAT_AT(t, 0, ins_sz),
  };

  NN nn = nn_alloc(arch.items, arch.count);
  NN g = nn_alloc(arch.items, arch.count);
  nn_rand(nn);
  NN_PRINT(nn);

  float rate = 0.5;

  SetConfigFlags(FLAG_WINDOW_RESIZABLE);
  InitWindow(IMG_WIDTH, IMG_HEIGHT, "gym");
  // SetWindowState(FLAG_WINDOW_RESIZABLE);
  SetTargetFPS(60);

  Cost_Plot plot = {0};

  int epoch = 0;
  int max_epoch = 20000;
  while (!WindowShouldClose()) {
    for (int i = 0; i < 10 && epoch < max_epoch; ++i) {
      if (epoch < max_epoch) {
        nn_backprop(nn, g, ti, to);
        nn_learn(nn, g, rate);
        epoch += 1;
        da_append(&plot, nn_cost(nn, ti, to));
      }
    }

    BeginDrawing();
    Color background_color = {0x18, 0x18, 0x18, 0xFF};
    ClearBackground(background_color);
    {
      int rw, rh, rx, ry;
      int w = GetRenderWidth();
      int h = GetRenderHeight();

      rw = w / 2;
      rh = h * 2 / 3;
      rx = 0;
      ry = h / 2 - rh / 2;
      plot_cost(plot, rx, ry, rw, rh);

      rw = w / 2;
      rh = h * 2 / 3;
      rx = w - rw;
      ry = h / 2 - rh / 2;
      nn_render_raylib(nn, rx, ry, rw, rh);

      char buffer[256];
      snprintf(buffer, sizeof(buffer), "Epoch: %d/%d, Rate: %f, Cost: %f",
               epoch, max_epoch, rate, plot.items[plot.count - 1]);
      DrawText(buffer, 0, 0, h * 0.04, WHITE);
    }
    for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      MAT_AT(NN_INPUT(nn), 0, 0) = i;
      MAT_AT(NN_INPUT(nn), 0, 1) = j;
      nn_forward(nn);
      printf("%d ^ %d = %f \n", i, j, MAT_AT(NN_OUTPUT(nn), 0, 0));
    }
  }

      printf("At epoach %d\n\n\n",epoch);
    if(epoch == max_epoch){
      NN_PRINT(nn);
    }
    EndDrawing();
  }
  return 0;
}
