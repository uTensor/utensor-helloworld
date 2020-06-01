#include "models/my_model/my_model.hpp"  //gernerated model file"
#include "uTensor/src/uTensor.h"
#include <cmath>
#include <iostream>
#include "mbed.h"
#include <stdio.h>
#include "input_image.h"  //contains the first sample taken from the MNIST test set

using namespace uTensor;

void onError(Error* err) {
  while (true) {
  }
}

int argmax(const Tensor& logits) {
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

Serial pc(USBTX, USBRX, 115200);  //baudrate := 115200
localCircularArenaAllocator<estimated_meta_usage> meta_allocator;
localCircularArenaAllocator<estimated_ram_usage, uint32_t> ram_allocator;
SimpleErrorHandler mErrHandler(10);

int main(void) {
  printf("Simple MNIST end-to-end uTensor cli example (device)\n");


  mErrHandler.set_onError(onError);
  Context::get_default_context()->set_metadata_allocator(&meta_allocator);
  Context::get_default_context()->set_ram_data_allocator(&ram_allocator);
  Context::get_default_context()->set_ErrorHandler(&mErrHandler);
  float correct_cnt = 0.0;
  
  // create the input/output tensor
  Tensor input_image = new RomTensor({1, 28, 28, 1}, flt, arr_input_image);
  Tensor logits = new RamTensor({1, 10}, flt);

  compute_my_model(input_image, logits);
  int max_index = argmax(logits);
  input_image.free();
  logits.free();
  
  printf("pred label: %d\r\n", max_index);

  return 0;
}
