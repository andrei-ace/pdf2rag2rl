PDFS = [
    (
        "docs/examples/1706.03762v7.pdf",
        [
            ("What is the name of the new network architecture proposed in the document?", "The Transformer."),
            ("What is the primary mechanism the Transformer architecture relies on?", "Attention mechanisms."),
            ("How many layers does the encoder stack of the Transformer have?", "Six identical layers."),
            # (
            #     "What does each layer of the encoder stack consist of?",
            #     "A multi-head self-attention mechanism and a position-wise fully connected feed-forward network.",
            # ),
            # (
            #     "What are the two main benefits of the Transformer model mentioned in the abstract?",
            #     "It is more parallelizable and requires significantly less time to train.",
            # ),
            # (
            #     "What dataset was used to train the models for the English-to-German translation task?",
            #     "The WMT 2014 English-German dataset.",
            # ),
            # ("What optimizer was used for training the Transformer models?", "The Adam optimizer."),
            # (
            #     "How is positional information added to the input embeddings in the Transformer model?",
            #     "By using sine and cosine functions of different frequencies.",
            # ),
            # (
            #     "What is the BLEU score achieved by the big Transformer model on the WMT 2014 English-to-French translation task?",
            #     "41.8.",
            # ),
            # (
            #     "What is the main advantage of self-attention over recurrent layers mentioned in the document?",
            #     "Self-attention allows for significantly more parallelization.",
            # ),
        ],
    ),
    (
        "docs/examples/1607.06450v1.pdf",
        [
            (
                "What problem does batch normalization aim to solve?",
                "Batch normalization aims to reduce the covariate shift problem by normalizing the summed inputs to each hidden unit over the training cases.",
            ),
            (
                "How does layer normalization differ from batch normalization?",
                "Layer normalization computes normalization statistics from all the summed inputs to the neurons in a layer on a single training case, making it suitable for RNNs and not dependent on the mini-batch size.",
            ),
            (
                "What is the main benefit of layer normalization in RNNs?",
                "Layer normalization stabilizes the hidden state dynamics in recurrent networks, particularly for long sequences and small mini-batches.",
            ),
            # (
            #     "What is the impact of layer normalization on training speed?",
            #     "Layer normalization can substantially reduce the training time compared with other normalization techniques.",
            # ),
            # (
            #     "Why is layer normalization effective for RNNs with varying sequence lengths?",
            #     "Layer normalization's normalization terms depend only on the summed inputs to a layer at the current time-step, avoiding issues with different sequence lengths.",
            # ),
            # (
            #     "What is a key invariance property of layer normalization?",
            #     "Layer normalization is invariant to per training-case feature shifting and scaling.",
            # ),
            # (
            #     "What experimental results highlight the robustness of layer normalization?",
            #     "Experimental results show that layer normalization is robust to batch sizes and exhibits faster training convergence compared to batch normalization.",
            # ),
            # (
            #     "How does layer normalization handle the problem of vanishing and exploding gradients?",
            #     "Layer normalization makes RNNs invariant to re-scaling of the summed inputs, leading to more stable hidden-to-hidden dynamics.",
            # ),
            # (
            #     "In which types of neural networks did preliminary experiments show layer normalization to offer a speedup?",
            #     "Preliminary experiments showed that layer normalization offers a speedup in convolutional neural networks compared to the baseline model.",
            # ),
            # (
            #     "What does Figure 6 in the document compare?",
            #     "Figure 6 compares the negative log likelihood and test error of permutation invariant MNIST models using layer normalization and batch normalization with different batch sizes.",
            # ),
        ],
    ),
]
