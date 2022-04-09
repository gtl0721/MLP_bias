
// MLP with one hidden layer

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
constexpr auto Actual_output_file = "act_y.dat";
constexpr auto Hid_Out_weight = "w.dat";
constexpr auto In_Hid_weight = "v.dat";

#define a(f) 1 / (1 + exp(-f))      // sigmoid function
constexpr auto _eta = 0.5;          // learning constant
constexpr auto _afa = 0.2;          // momentum constant
constexpr auto _in_varl = 4;        // number of input variables
constexpr auto _out_varl = 1;       // number of output variables
constexpr auto _node = 10;          // number of hidden layer nodes
constexpr auto _dat_num = 500;      // number of training samples
constexpr auto _iteration = 30000;  // number of learning iterations

void Read_data(void);  // read training data pairs
void LEARN(void);      // learning the weights
void TEST(void);       // compute the actual outputs
void SAVE(void);       // save the learned weights

float _x[_dat_num + 1][_in_varl + 1], _d[_dat_num + 1][_out_varl + 1],
    _y[_out_varl + 1];
float _v[_node + 1][_in_varl + 1], _w[_out_varl + 1][_node + 1];

int _tmain(int argc, _TCHAR *argv[]) {
  time_t t;

  srand((unsigned)time(&t));

  Read_data();
  LEARN();
  TEST();
  SAVE();
  printf("\n  ******   The  end ");
  //_getch();
  system("pause");
  return 0;
}

void Read_data() {
  int i, j;
  FILE *fm;

  if (fopen_s(&fm, Input_file, "r") != 0) exit(1);

  for (i = 0; i < _dat_num; i++) {
    for (j = 0; j < _in_varl; j++) {
      fscanf_s(fm, "%f ", &_x[i][j]);
      _x[i][j] -= 0.4;
    }
    _x[i][_in_varl] = -1;
  }
  fclose(fm);

  if (fopen_s(&fm, Desired_output_file, "r") != 0) exit(1);

  for (i = 0; i < _dat_num; i++) {
    for (j = 0; j < _out_varl; j++) {
      fscanf_s(fm, "%f ", &_d[i][j]);
      _d[i][j] -= 0.4;
    }
  }
  fclose(fm);
}

void LEARN() {
  int i, j, q, num, itera = 0;
  float H, Y, sum, *error, *h, *dely, *delh, *difference;
  float(*apv)[_in_varl + 1], (*apw)[_node + 1];  // , *a_bias_h, *a_bias_y;

  if (!(error = new float[_out_varl + 1]) || !(h = new float[_node + 1]) ||
      !(dely = new float[_out_varl + 1]) || !(delh = new float[_node + 1]) ||
      !(difference = new float[_out_varl + 1]) ||
      !(apv = new float[_node + 1][_in_varl + 1]) ||
      !(apw = new float[_out_varl + 1][_node + 1])) {
    cout << "\n Insufficient memory for learn";
    exit(1);
  }

  for (q = 0; q < _node; q++) {
    for (i = 0; i <= _in_varl; i++) {
      _v[q][i] = (rand() % (1000) - 500) / 10000.;
      apv[q][i] = 0;
    }
  }

  for (j = 0; j < _out_varl; j++) {
    for (q = 0; q <= _node; q++) {
      _w[j][q] = (rand() % (1000) - 500) / 10000.0;
      apw[j][q] = 0;
    }
  }

  while (itera < _iteration) {
    for (j = 0; j < _out_varl; j++) error[j] = 0;

    itera = itera + 1;

    for (num = 0; num < _dat_num; num++) {
      for (q = 0; q < _node; q++) {
        H = 0;
        for (i = 0; i <= _in_varl; i++) {
          H = H + _v[q][i] * _x[num][i];
        }
        h[q] = a(H);
      }
      h[_node] = -1;

      for (j = 0; j < _out_varl; j++) {
        Y = 0;
        for (q = 0; q <= _node; q++) {
          Y = Y + _w[j][q] * h[q];
        }
        _y[j] = a(Y);
        difference[j] = _y[j] - _d[num][j];
        error[j] = error[j] + difference[j] * difference[j];
        dely[j] = difference[j] * (1 - _y[j]) * _y[j];
      }

      for (q = 0; q < _node; q++) {
        sum = 0;
        for (j = 0; j < _out_varl; j++) {
          sum = sum + dely[j] * _w[j][q];
        }
        delh[q] = h[q] * (1 - h[q]) * sum;
      }

      for (j = 0; j < _out_varl; j++) {
        for (q = 0; q <= _node; q++) {
          apw[j][q] = -_eta * dely[j] * h[q] + _afa * apw[j][q];
          _w[j][q] = _w[j][q] + apw[j][q];
        }
      }

      for (q = 0; q < _node; q++) {
        for (i = 0; i <= _in_varl; i++) {
          apv[q][i] = -_eta * delh[q] * _x[num][i] + _afa * apv[q][i];
          _v[q][i] = _v[q][i] + apv[q][i];
        }
      }
    }
    if ((itera % 1) == 0) {
      printf("\n itera is %d, RMSE of  ", itera);
      for (j = 0; j < _out_varl; j++)
        printf(" output %d is --- %f ", j + 1, sqrt(error[j] / _dat_num));
    }

  } /* end while */

  delete[] error;
  delete[] h;
  delete[] dely;
  delete[] delh;
  delete[] difference;
  delete[] apv;
  delete[] apw;
}

void TEST()

{
  int i, j, num, q;
  float h[_node + 1], H, Y, error[_out_varl + 1], difference[_out_varl + 1];
  FILE *fout;

  if (fopen_s(&fout, Actual_output_file, "w") != 0) exit(1);

  for (j = 0; j < _out_varl; j++) error[j] = 0;

  for (num = 0; num < _dat_num; num++) {
    for (q = 0; q < _node; q++) {
      H = 0;
      for (i = 0; i <= _in_varl; i++) H = H + _v[q][i] * _x[num][i];
      h[q] = a(H);
    }
    h[_node] = -1;

    for (j = 0; j < _out_varl; j++) {
      Y = 0;
      for (q = 0; q <= _node; q++) Y = Y + _w[j][q] * h[q];
      _y[j] = a(Y);

      fprintf(fout, "%f   \n", _y[j]);

      difference[j] = _d[num][j] - _y[j];
      error[j] = error[j] + difference[j] * difference[j];
    }

  } /* end for num */

  printf("\n\n ++++ Final result: ");
  for (j = 0; j < _out_varl; j++)
    printf(" output %d is --- %f ", j + 1, sqrt(error[j] / _dat_num));

  fclose(fout);
}

void SAVE() {
  FILE *fp;
  int i, j, q;

  if (fopen_s(&fp, In_Hid_weight, "w") != 0) exit(1);

  for (q = 0; q < _node; q++) {
    for (i = 0; i <= _in_varl; i++) fprintf(fp, "%f ", _v[q][i]);
    fprintf(fp, "\n");
  }

  fclose(fp);

  if (fopen_s(&fp, Hid_Out_weight, "w") != 0) exit(1);

  for (j = 0; j < _out_varl; j++) {
    {
      for (q = 0; q <= _node; q++) fprintf(fp, "%f ", _w[j][q]);
      fprintf(fp, "\n");
    }
  }
  fclose(fp);
}
