import argparse

def return_parser():
    
    parser = argparse.ArgumentParser(description='Self Ablation in Transformers')
    parser.add_argument('k-attention', type=int)
    parser.add_argument('k-neurons', type=int)
    parser.add_argument('has_layer_by_layer_ablation_mask', type=bool)
    parser.add_argument('has_overall_ablation_mask', type=bool)
    
    return parser
    