#!/usr/bin/env python3
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="google/gemma-2-9b-it")
    parser.add_argument("--adapter-dirs", nargs="+", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cpu")
    
    print(f"Loading base adapter: {args.adapter_dirs[0]}")
    model = PeftModel.from_pretrained(model, args.adapter_dirs[0])
    state = dict(model.named_parameters())
    
    for d in args.adapter_dirs[1:]:
        print(f"Averaging with: {d}")
        m2 = PeftModel.from_pretrained(
            AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cpu"), 
            d
        )
        for k, v in m2.named_parameters():
            if 'lora' in k:
                state[k].data += v.data
                
    for k in state:
        if 'lora' in k:
            state[k].data /= len(args.adapter_dirs)
            
    print(f"Saving merged adapter to: {args.out_dir}")
    model.save_pretrained(args.out_dir)

if __name__ == "__main__":
    main()
