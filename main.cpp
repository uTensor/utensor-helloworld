#include "quant_mnist.hpp"
#include "tensorIdxImporter.hpp"
#include "tensor.hpp"
#include "FATFileSystem.h"
#include "SDBlockDevice.h"
#include "mbed.h"

Serial pc(USBTX, USBRX, 115200);
SDBlockDevice bd(MBED_CONF_APP_SD_MOSI, MBED_CONF_APP_SD_MISO,
                 MBED_CONF_APP_SD_CLK, MBED_CONF_APP_SD_CS);
FATFileSystem fs("fs");

void run_mlp(){
  TensorIdxImporter t_import;
  Tensor* input_x = t_import.float_import("/fs/tmp.idx");
  Context ctx;

  get_quant_mnist_ctx(ctx, input_x);
  S_TENSOR pred_tensor = ctx.get("y_pred:0");
  ctx.eval();

  int pred_label = *(pred_tensor->read<int>(0, 0));
  printf("Predicted label: %d\r\n", pred_label);

}

int main(void) {
  printf("Simple MNIST end-to-end uTensor cli example (device)\n");
  
  ON_ERR(bd.init(), "SDBlockDevice init ");
  ON_ERR(fs.mount(&bd), "Mounting the filesystem on \"/fs\". ");

  init_env();
  run_mlp();
  
  ON_ERR(fs.unmount(), "fs unmount ");
  ON_ERR(bd.deinit(), "SDBlockDevice de-init ");

  return 0;
}
