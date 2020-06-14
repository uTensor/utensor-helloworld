#include <stdio.h>

#include <cmath>
#include <iostream>

#include "input_image.h"  //contains the first sample taken from the MNIST test set
#include "mbed.h"
#include "models/my_model/my_model.hpp"  //gernerated model file"
#include "uTensor.h"

using namespace uTensor;

void onError(Error *err) {
  while (true) {
  }
}

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

// FIXME: estimated_meta_usage and estimated_ram_usage need to be gernerated
const int estimated_ram_usage = 5000;
const int estimated_meta_usage = 1000;

Serial pc(USBTX, USBRX, 115200);  // baudrate := 115200
localCircularArenaAllocator<estimated_meta_usage> meta_allocator;
localCircularArenaAllocator<estimated_ram_usage, uint32_t> ram_allocator;
SimpleErrorHandler mErrHandler(10);

int main(void) {
  printf("Simple MNIST end-to-end uTensor cli example (device)\n");

  mErrHandler.set_onError(onError);
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  static My_model model;

  // create the input/output tensor
  Tensor input_image = new RomTensor({1, 28, 28, 1}, flt, arr_input_image);
  Tensor logits = new RamTensor({1, 10}, flt);

  model.set_inputs({{My_model::input_0, input_image}})
      .set_outputs({{My_model::output_0, logits}})
      .eval();

  int max_index = argmax(logits);
  input_image.free();
  logits.free();

  printf("pred label: %d\r\n", max_index);

  return 0;
}
