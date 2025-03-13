# SmallLanguageModel-HybridNorm-FourierFormer (FANformer)

<div align="center">
  <img src="assets/fanformer_logo.png" alt="FANformer Logo" width="400px">
</div>

## Overview

FANformer is a compact language model implementing HybridNorm and Fourier-based attention mechanisms. It combines CoLA (low-rank projections), Fourier Analysis Network (FAN), and hybrid normalization to create an efficient decoder-only transformer architecture. By leveraging periodicity modeling and gated residuals, FANformer enhances performance while maintaining a small parameter footprint.

The model employs a novel approach to efficiently model periodicity in neural networks, significantly improving learning efficiency and reducing both training time and resource requirements compared to traditional Transformer architectures.

## Architecture Overview

The FANformer model architecture integrates several innovative components:

- **Fourier Analysis Network (FAN)**: Incorporates efficient periodicity modeling into the attention mechanism
- **CoLA Linear Layers**: Low-rank linear projections to reduce parameter count
- **HybridNorm**: A mixed normalization strategy combining pre-norm and post-norm approaches
- **Gated Residuals**: Enhanced residual connections with learnable gates for better gradient flow
- **SwiGLU**: Modified activation function in the feed-forward networks

The core of each FANformer layer is the ATtention-Fourier (ATF) module, which integrates FAN into the self-attention mechanism to explicitly model periodicity in the frequency domain.

## Model Specifications

- **Parameters**: 92.2 million parameters
- **Hardware Requirements**: Can be trained on a single NVIDIA RTX 4090
- **Training Speed**: 5.9 it/s with batch_size 16
- **Training Throughput**: 1 million examples (with sequence length 1024) in approximately 3 hours
- **Default Configuration**:
  - Hidden dimension: 512
  - Number of heads: 12
  - Number of layers: 12
  - FFN dimension: 2048
  - Base dropout: 0.1
  - GQA groups: 6

## Getting Started

### Installation

```bash
git clone https://github.com/YourUsername/SmallLanguageModel-HybridNorm-FourierFormer.git
cd SmallLanguageModel-HybridNorm-FourierFormer
pip install -r requirements.txt
```

### Training

To train a FANformer model from scratch:

```bash
python train.py --mode train --epochs_text 2 --batch_size 16 --p 0.15
```

Parameters:
- `--mode`: Choose between 'train' or 'inference'
- `--epochs_text`: Number of training epochs
- `--batch_size`: Batch size
- `--p`: Proportion of periodicity modeling (default: 0.15)

### Inference

To run inference with a pre-trained model:

```bash
python train.py --mode inference --max_length 100 --top_k 100 --top_p 0.85 --temperature 0.7
```

Parameters:
- `--max_length`: Maximum length of generated text
- `--min_length`: Minimum length before considering EOS token
- `--top_k`: Value for top-k sampling
- `--top_p`: Value for nucleus sampling
- `--temperature`: Temperature for logit scaling

## Key Features

### 1. Efficient Periodicity Modeling

FANformer's key innovation is the integration of Fourier Analysis Network (FAN) into attention mechanism to achieve efficient periodicity modeling. This significantly improves the model's ability to learn and represent periodic patterns common in language data.

### 2. Hybrid Normalization Strategy

The HybridNorm approach combines the benefits of pre-norm and post-norm architectures:
- First block uses Pre-Norm in the attention mechanism
- Other blocks use QKV-Norm
- All blocks employ Post-Norm for FFN layers

### 3. CoLA Low-Rank Projections

CoLA (Compute-efficient Low-rank Activation) enables efficient parameter utilization:
- Replaces full-rank linear layers with low-rank factorizations
- Adds non-linear transformations between factorized weight matrices
- Reduces computational costs while maintaining modeling capacity

### 4. Resource Efficiency

FANformer achieves superior efficiency metrics:
- Requires ~31% fewer parameters than comparable Transformer models
- Needs ~20% fewer training tokens to achieve similar performance
- Maintains higher throughput in both training and inference

## Performance

FANformer consistently outperforms traditional Transformer models of similar size:
- Better perplexity on language modeling tasks
- Stronger performance on downstream tasks like ARC, SCIQ, and PIQA
- Enhanced ability for rule-based reasoning

## Upcoming Improvements

We are actively working on several improvements to the FANformer architecture:

1. **Flash Attention Integration**: Implementing compatible packing methods to fully leverage Flash Attention for even faster training.

2. **Latent Space Scaling**: Exploring methods for efficient scaling during inference time in latent space to improve generation quality and speed.

3. **Multimodality Support**: Testing multimodal capabilities based on techniques mentioned in the Phi-4 Multimodal technical report.

4. **Community Training Initiative**: Exploring collaborations for distributed training of larger FANformer variants.

5. **Quantization Support**: Adding support for various quantization methods (GPTQ, AWQ, GGUF) for more efficient deployment.

## Contributing

We welcome contributions to improve FANformer! Whether it's code, documentation, or ideas, please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-improvement`)
3. Commit your changes (`git commit -m 'Add amazing improvement'`)
4. Push to the branch (`git push origin feature/amazing-improvement`)
5. Open a Pull Request

## References

1. Liu, Z., Zhang, R., Wang, Z., Yang, Z., Hovland, P., Nicolae, B., Cappello, F., & Zhang, Z. (2025). CoLA: Compute-Efficient Pre-Training of LLMs via Low-Rank Activation. arXiv:2502.10940v1.

2. Dong, Y., Li, G., Jiang, X., Tao, Y., Zhang, K., Zhu, H., Liu, H., Ding, J., Li, J., Deng, J., & Mei, H. (2025). FANformer: Improving Large Language Models Through Effective Periodicity Modeling. arXiv:2502.21309v1.

3. Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. Neural Information Processing Systems (NeurIPS).

4. Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Yang, A., Fan, A., et al. (2024). The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

5. Groeneveld, D., Beltagy, I., Walsh, E. P., Bhagia, A., Kinney, R., Tafjord, O., Jha, A. H., Ivison, H., Magnusson, I., Wang, Y., et al. (2024). OLMo: Accelerating the Science of Language Models. In ACL (1), pages 15789–15809. Association for Computational Linguistics.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We thank the authors of CoLA and FANformer for their research contributions that made this implementation possible.
