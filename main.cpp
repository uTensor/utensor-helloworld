#include "models/deep_mlp.hpp"  //gernerated model file
#include "tensor.hpp"  //useful tensor classes
#include "mbed.h"
#include <stdio.h>
#include "input_data.h"  //contains the first sample taken from the MNIST test set

Serial pc(USBTX, USBRX, 115200);  //baudrate := 115200

int main(void) {
  printf("Simple MNIST end-to-end uTensor cli example (device)\n");

  Context ctx;  //creating the context class, the stage where inferences take place 
  //wrapping the input data in a tensor class
  Tensor* input_x = new WrappedRamTensor<float>({1, 784}, (float*) input_data);

  get_deep_mlp_ctx(ctx, input_x);  // pass the tensor to the context
  S_TENSOR pred_tensor = ctx.get("y_pred:0");  // getting a reference to the output tensor
  ctx.eval(); //trigger the inference

  int pred_label = *(pred_tensor->read<int>(0, 0));  //getting the result back
  printf("Predicted label: %d\r\n", pred_label);

  return 0;
}
