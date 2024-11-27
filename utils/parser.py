import argparse

def return_parser():
    
    parser = argparse.ArgumentParser(description='Self Ablation in Transformers')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--k_attention', type=int)
    parser.add_argument('--k_neurons', type=int)
    parser.add_argument('--has_layer_by_layer_ablation_mask', action='store_true')
    parser.add_argument('--has_overall_ablation_mask', action='store_true')
    parser.add_argument('--loss_coeff_base', type=float)
    parser.add_argument('--loss_coeff_ablated', type=float)
    
    return parser
    