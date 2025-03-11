# self-ablating-transformers
A self-modeling transformer with an auxiliary output head that is an ablation mask for itself, used either in a second forward pass (global self-ablation) or seprately for each layer (layer-by-layer self ablation, called "local" in the paper).

See https://www.notion.so/apartresearch/Final-Self-Ablating-Transformer-4303c123a0ba4346bd7be95adecf6abf for detailed description (currently private to Apart lab fellows).

# Current status

The project's initial implementation used GPTNeo to match Ronen Eldan's pretrained models (e.g. https://huggingface.co/roneneldan/TinyStories-1M)

Activation function is replaced by NewGELUActivation

The initial model training stats matched other implementations 

For some of the first experiments, evidence of data leakage was found. (By "data leakage" here we mean that the so-called "ablation mask", which was intended to be an approximately binary on-and-off mask for which model components were active vs inactive for a particular token computation, was in fact being used by the model to pass information about the next token prediction directly from the first forward pass to the second forward pass. This actually makes perfect sense because we're training that second forward pass to use all the resources available to it to give a good prediction for the next token, and if anything correlated with the output of the first forward pass is available, that's a very useful information resource.)

For the final experiments there was no evidence at all. We switched to a k-WTA (winner takes all) architecture where the ablation mask used in the forward pass is actually a hard on/off binary mask, and it's only approximated by a smooth-top-K function for gradient propagation in the backward pass.
