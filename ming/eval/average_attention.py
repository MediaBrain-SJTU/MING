import argparse 
import os 
import json
import math
import torch
import pandas as pd

TYPE = ["layer_weighted_entropy", "layer_average_entropy", "layer_max_entropy", "input_layer_average_entropy", "input_layer_max_entropy", "output_layer_average_entropy", "output_layer_max_entropy", "first_output_layer_entropy"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = f.readlines()
    head = data[0]
    data = [d.strip().split("\t") for d in data[1:]] # (sample, layer)
    data = [[t.split(",") for t in d] for d in data] # (sample, layer, type)
    #all_layer_weighted_entropy = layer_weighted_entropy.mean(-1) # (L, )
    # input_layer_weighted_entropy = layer_weighted_entropy[:, :input_length].mean(-1) # (L, )
    # output_layer_weighted_entropy = layer_weighted_entropy[:, input_length:].mean(-1) # (L, )
    # first_output_layer_weighted_entropy = layer_weighted_entropy[:, input_length] # (L, )

    data = [[[float(t) for t in d] for d in sample] for sample in data] # (sample, layer, type)
    data = torch.tensor(data)
    data_mean = data.mean(dim=0) # (layer, type)

    RESULTS = {}
    RESULTS["layer"] = [i for i in range(data_mean.shape[0])]
    for _type in TYPE:
        RESULTS[_type] = []
    for i in range(data_mean.shape[1]):
        RESULTS[TYPE[i]] = data_mean[:, i].tolist()
    
    df = pd.DataFrame(RESULTS)
    df.to_csv(args.output_file, index=False)

    # with open(args.output_file, "a") as f:
    #     f.write(f"Type\t{head}")
    #     f.write("ALL_WEIGHTED\t" + "\t".join([f"{m:.4f}" for m in mean[:, 0].tolist()]) + "\n")
    #     f.write("INPUT\t" + "\t".join([f"{m:.4f}" for m in mean[:, 1].tolist()]) + "\n")
    #     f.write("OUTPUT\t" + "\t".join([f"{m:.4f}" for m in mean[:, 2].tolist()]) + "\n")
    #     f.write("OUTFIRST\t" + "\t".join([f"{m:.4f}" for m in mean[:, 3].tolist()]) + "\n")