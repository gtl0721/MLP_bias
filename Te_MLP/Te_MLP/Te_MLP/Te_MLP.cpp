// Te_MLP.cpp : 定義主控台應用程式的進入點。
//

#include <conio.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iostream>

#include "stdafx.h"

using namespace std;

constexpr auto Input_file = "in.dat";
constexpr auto Desired_output_file = "out.dat";
constexpr auto Hid_Out_weight = "w.dat";
constexpr auto In_Hid_weight = "v.dat";
constexpr auto Test_output_file = "test_y.dat";

#define a(f) 1 / (1 + exp(-f))
constexpr auto _in_varl = 4;
constexpr auto _out_varl = 1;
constexpr auto _node = 10;
constexpr auto _dat_num = 500;

void READ_W(FILE *ft);
void READ_V(FILE *ft);
void READ_input_data(FILE *fx);
void READ_output_data(FILE *fy);
void TEST_SAVE(void);

float _x[_dat_num + 1][_in_varl + 1], _d[_dat_num + 1][_out_varl + 1];
float _y[_out_varl + 1];
float _v[_node + 1][_in_varl + 1], _w[_out_varl + 1][_node + 1];

int _tmain(int argc, _TCHAR *argv[]) {
  FILE *stream;

  if (fopen_s(&stream, In_Hid_weight, "r") != 0) {
    printf("open error 1 ");
    exit(1);
  }
  READ_V(stream);
  fclose(stream);

  if (fopen_s(&stream, Hid_Out_weight, "r") != 0) {
    printf("open error 2 ");
    exit(1);
  }
  READ_W(stream);
  fclose(stream);

  if (fopen_s(&stream, Input_file, "r") != 0) {
    printf("open error 3 ");
    exit(1);
  }
  READ_input_data(stream);
  fclose(stream);

  if (fopen_s(&stream, Desired_output_file, "r") != 0) {
    printf("open error 4 ");
    exit(1);
  }
  READ_output_data(stream);
  fclose(stream);

  TEST_SAVE();
  //_getch();
  system("pause");
  return 0;
}

void READ_W(FILE *ft) {
  int j, q;

  for (j = 0; j < _out_varl; j++) {
    for (q = 0; q <= _node; q++) fscanf_s(ft, "%f ", &_w[j][q]);
  }
}

void READ_V(FILE *ft) {
  int i, q;

  for (q = 0; q < _node; q++) {
    for (i = 0; i <= _in_varl; i++) fscanf_s(ft, "%f ", &_v[q][i]);
  }
}

void READ_input_data(FILE *fx) {
  int i, j;

  for (i = 0; i < _dat_num; i++) {
    for (j = 0; j < _in_varl; j++) {
      fscanf_s(fx, "%f", &_x[i][j]);
      _x[i][j] -= 0.4;
    }
    _x[i][_in_varl] = -1;  // for bias
  }
}

void READ_output_data(FILE *fy) {
  int i, j;

  for (i = 0; i < _dat_num; i++) {
    for (j = 0; j < _out_varl; j++) {
      fscanf_s(fy, "%f ", &_d[i][j]);
      _d[i][j] -= 0.4;
    }
  }
}

void TEST_SAVE(void) {
  int i, j, q, num;
  FILE *stream;
  float h[_node + 1], Y, H, error[_out_varl + 1], difference[_out_varl + 1];
  FILE *fout;

  printf("\n   testing : ********************************** \n");

  if (fopen_s(&fout, Test_output_file, "w") != 0) exit(1);

  for (j = 0; j < _out_varl; j++) error[j] = 0;

  for (num = 0; num < _dat_num; num++) {
    for (q = 0; q < _node; q++) {
      H = 0;
      for (i = 0; i <= _in_varl; i++) H = H + _v[q][i] * _x[num][i];
      h[q] = a(H);
    }
    h[_node] = -1;  // for bias

    for (j = 0; j < _out_varl; j++) {
      Y = 0;
      for (q = 0; q <= _node; q++) Y = Y + _w[j][q] * h[q];
      _y[j] = a(Y);
      //_y[j] += 0.4;
      fprintf(fout, "%f  ", _y[j]);

      difference[j] = _d[num][j] - _y[j];
      error[j] = error[j] + difference[j] * difference[j];
    }

    fprintf(fout, " \n ");

  } /* end for num */

  printf("\n\n ++++ Final result: ");
  for (j = 0; j < _out_varl; j++)
    printf(" output %d is --- %f ", j + 1, sqrt(error[j] / _dat_num));

  fclose(fout);
}
