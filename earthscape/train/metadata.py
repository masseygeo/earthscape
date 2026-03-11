
import os
import torchinfo



def architecture_to_json(output_dir, model, loader):

    # input feature shape...
    input_size = next(iter(loader))
    input_size = {k: v for k, v in input_size.items() if k != "label"}
    input_size = list(next(iter(input_size.values()))[:1].shape)

    architecture = torchinfo.summary(model, input_size=input_size, depth=4, verbose=0, col_names=["input_size", "kernel_size", "output_size", "num_params"])
    output_path = os.path.join(output_dir, 'architecture.json')
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(str(architecture))