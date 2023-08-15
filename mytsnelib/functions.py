def suma(int1: int, int2: int) -> int:
    return int1 + int2



""""
1. Calculate pairwise similarities or dissimilarities between data points in the high-dimensional space. This could be based on various metrics, such as Euclidean distance, cosine similarity, or other similarity measures.

2. Construct a probability distribution over these pairwise similarities or dissimilarities. This distribution represents the relationships between data points in the high-dimensional space.

3. Define a similar probability distribution over the lower-dimensional space (e.g., 2D or 3D).

4. Optimize the positions of data points in the lower-dimensional space so that the divergence between the two distributions (high-dimensional and lower-dimensional) is minimized. This is typically done using gradient descent optimization techniques.

5. Iterate the optimization process to achieve a stable mapping of data points in the lower-dimensional space.


"""