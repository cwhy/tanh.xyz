# NK Clustering Algorithm

An online clustering algorithm with bounded cluster sizes.

## Parameters

- **N**: Maximum number of points per cluster
- **K**: Number of neighbors used to determine the medoid (K < N)

## Key Concepts

### Medoid

The medoid is the representative point of a cluster. It is defined as the point with the **lowest distance to its K-th nearest neighbor**.

This definition finds the point in the densest region of the cluster, making it robust to outliers compared to using a centroid.

### cMin (Cluster Radius)

The distance from the medoid to the **furthest point** in the cluster. This defines the cluster's acceptance boundary.

## Algorithm

### Adding a Point

1. Find the cluster whose medoid is closest to the new point
2. If the cluster has fewer than N points, add the point
3. If the cluster is full:
   - If the new point is closer to the medoid than cMin, it can join
   - The furthest point from the medoid is **evicted**
   - The evicted point recursively tries to find another cluster
4. If no cluster can accept the point, create a new cluster

### Eviction Cascade

When a point is evicted, it attempts to join other clusters (excluding the one it was just evicted from). This can cause a chain of evictions until either:
- An evicted point finds a non-full cluster
- A new cluster is created

## Properties

- **Bounded cluster size**: Each cluster has at most N points
- **Online**: Points are processed one at a time
- **Self-organizing**: Clusters naturally form around dense regions
- **Outlier handling**: Outliers tend to form their own small clusters
