#include <stdio.h>

#include <cmath>

#include "input_image.h"  //contains the first sample taken from the MNIST test set
#include "mbed.h"
#include "models/my_model/my_model.hpp"  //gernerated model file"
#include "uTensor.h"

using namespace uTensor;

int argmax(const Tensor &logits) {
  uint32_t num_elems = logits->num_elems();
  float max_value = static_cast<float>(logits(0));
  int max_index = 0;
  for (int i = 1; i < num_elems; ++i) {
    float value = static_cast<float>(logits(i));
    if (value >= max_value) {
      max_value = value;
      max_index = i;
    }
  }
  return max_index;
}

static My_model model;

int main(int argc, char **argv) {
  printf("\n");
  printf("Simple MNIST end-to-end uTensor cli example (device)\n");

  // create the input/output tensor
  for (size_t i = 0; i < 8; ++i) {
    Tensor input_image = new RomTensor({1, 28, 28, 1}, flt, arr_input_image[i]);
    Tensor logits = new RamTensor({1, 10}, flt);

    model.set_inputs({{My_model::input_0, input_image}})
        .set_outputs({{My_model::output_0, logits}})
        .eval();
    int max_index = argmax(logits);
    input_image.free();
    logits.free();

    printf("pred label: %d, expecting: %d\r\n", max_index, ref_labels[i]);
  }
  return 0;
}
