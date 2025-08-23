import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
        """
        An embedding module.

        Arguments:
            num_embeddings: int size of the vocabulary
            embedding_dim: int dimension of the embedding vectors
        """
        super().__init__()
        # TODO: call superclass constructor
        # TODO: initialize your embedding matrix as a nn.Parameter
        #       of shape (num_embeddings, embedding_dim)
        #       with the specified device and dtype.
        #       store the embedding matrix with the d_model being the final dimension
        #       don't use nn.Embedding or nn.functional.embedding
        #       use torch.nn.init.trunc_normal_ to initialize the weights.
        self.embedding = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # Initialize weights
        torch.nn.init.trunc_normal_(self.embedding, mean=0.0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.

        Arguments:
            token_ids: torch.Tensor of shape (batch_size, sequence_length)

        Returns:
            torch.Tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        # TODO: select the embedding vector for each token ID
        #       by indexing into the embedding matrix
        return self.embedding[token_ids]
