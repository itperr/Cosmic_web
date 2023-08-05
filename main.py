import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors


# Read node data from the text file
nodes_data = []
with open('nodes.txt', 'r') as file:
    for line in file:
        node, x, y, z = map(float, line.strip().split())
        nodes_data.append((node, x, y, z))

# # M1 MODEL: CONNECT NODES IF THE DISTANCE BETWEEN THEM IS SMALLER THAN R
#
# # Create a 3D complex network
# G = nx.Graph()
# # Add nodes to the graph
# for node, x, y, z in nodes_data:
#     G.add_node(node, pos=(x, y, z))
# R = 5.0  # Maximum distance for creating edges
# # Create edges based on distance
# for i, (node1, x1, y1, z1) in enumerate(nodes_data):
#     for j, (node2, x2, y2, z2) in enumerate(nodes_data[i+1:], i+1):
#         distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
#         if distance < R:
#             G.add_edge(node1, node2)
#
# # VISUALIZATION
# # Get the positions of nodes for plotting
# pos = nx.get_node_attributes(G, 'pos')
#
# # Create the plot
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot edges
# for edge in G.edges():
#     node1, node2 = edge
#     x = [pos[node1][0], pos[node2][0]]
#     y = [pos[node1][1], pos[node2][1]]
#     z = [pos[node1][2], pos[node2][2]]
#     ax.plot(x, y, z, c='yellow', linewidth=1.0)
#
# # Plot nodes
# for node in G.nodes():
#     x, y, z = pos[node]
#     ax.scatter(x, y, z, c='white', s=10, edgecolors='black', linewidths=0.5)
#
# # Customize the plot
# ax.set_facecolor('black')
# ax.view_init(elev=20, azim=30)  # Adjust the elevation and azimuth angles to rotate the plot
#
# plt.title('3D Complex Network', fontsize=16)
# plt.show()
#
# # DEGREE DISTRIBUTION
#
# # Calculate the degree for each node
# degree_sequence = [d for n, d in G.degree()]
#
# # Calculate the cumulative degree distribution
# cumulative_distribution = {}
# for degree in degree_sequence:
#     if degree in cumulative_distribution:
#         cumulative_distribution[degree] += 1
#     else:
#         cumulative_distribution[degree] = 1
#
# # Sort the cumulative distribution by degree and include zero counts for missing degrees
# max_degree = max(degree_sequence)
# sorted_cumulative = [(degree, cumulative_distribution.get(degree, 0)) for degree in range(max_degree + 1)]
#
# # Compute cumulative values
# cumulative_values = [sum(count for degree, count in sorted_cumulative if degree >= x) for x in range(max_degree + 1)]
#
# # Plot the cumulative degree distribution on a log-log scale
# degrees, counts = zip(*sorted_cumulative)
# plt.loglog(degrees, cumulative_values, marker='o', linestyle='-', color='b')
# plt.xlabel('Degree (log scale)')
# plt.ylabel('Cumulative Count (log scale)')
# plt.title('Cumulative Degree Distribution')
# plt.grid(True)
# plt.show()

# # M2 MODEL: CONNECT NODES ONLY WITH THE N CLOSEST NEIGHBORS
# Create a function to build a graph with N closest neighbors for each node
def build_graph_with_N_neighbors(nodes_data, N):
    G = nx.DiGraph()

    for node, x, y, z in nodes_data:
        G.add_node(node, pos=(x, y, z))

    # Find k-nearest neighbors for each node
    X = [pos for _, pos in G.nodes(data='pos')]
    nbrs = NearestNeighbors(n_neighbors=N + 1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Connect nodes to their k-nearest neighbors
    for i, (node, _) in enumerate(G.nodes(data='pos')):
        for j in indices[i][1:]:
            neighbor_node = list(G.nodes())[j]
            G.add_edge(node, neighbor_node)

    return G

# Number of closest neighbors
N = 4

# Create the network
G = build_graph_with_N_neighbors(nodes_data,N)

# VISUALIZATION
# Create the plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot edges
for edge in G.edges():
    node1, node2 = edge
    x = [G.nodes[node1]['pos'][0], G.nodes[node2]['pos'][0]]
    y = [G.nodes[node1]['pos'][1], G.nodes[node2]['pos'][1]]
    z = [G.nodes[node1]['pos'][2], G.nodes[node2]['pos'][2]]
    ax.plot(x,y,z, c='yellow', linewidth=1.0)

# Plot nodes
for node in G.nodes():
    x, y, z = G.nodes[node]['pos']
    ax.scatter(x, y, z, c='white', s=5, edgecolors='black', linewidths=0.5)

# Customize the plot
ax.set_facecolor('black')
ax.view_init(elev=20, azim=30)  # Adjust the elevation and azimuth angles to rotate the plot

plt.title('Cosmic Web', fontsize=16)
plt.show()

# IN-DEGREE DISTRIBUTION
# Calculate in-degree for each node
in_degree_distribution = dict(G.in_degree())

# Plot the in-degree distribution
plt.bar(in_degree_distribution.keys(), in_degree_distribution.values())
plt.xlabel('Node')
plt.ylabel('In-degree')
plt.title('In-degree Distribution')
plt.show()

#ADJACENCY MATRIX
adjacency_matrix = nx.to_numpy_array(G)
print(adjacency_matrix)

# IN-DEGREE CUMULATIVE DISTRIBUTION (doesn't reproduce the behaviour of a scale-free network!!!)
# # Get the in-degree for each node
# in_degrees = dict(G_nx.in_degree())
#
# # Count the number of nodes with each in-degree
# in_degree_counts = {}
# for degree in in_degrees.values():
#     in_degree_counts[degree] = in_degree_counts.get(degree, 0) + 1
#
# # Calculate the cumulative in-degree distribution
# total_nodes = len(G_nx.nodes())
# cumulative_distribution = {}
# cumulative_count = 0
# for degree in sorted(in_degree_counts.keys()):
#     cumulative_count += in_degree_counts[degree]
#     cumulative_distribution[degree] = cumulative_count / total_nodes
#
# # Print the cumulative in-degree distribution
# print("In-degree   Cumulative Distribution")
# for degree, cdf in cumulative_distribution.items():
#     print(f"{degree:8}   {cdf:.6f}")
#
# # Plot the cumulative in-degree distribution
# plt.plot(list(cumulative_distribution.keys()), list(cumulative_distribution.values()), marker='o')
# plt.xlabel('In-degree')
# plt.ylabel('Cumulative Distribution')
# plt.title('Cumulative In-degree Distribution')
# plt.grid(True)
# plt.show()

# # AVERAGE CLUSTERING COEFFICIENT AND GIANT STRONGLY CONNECTED COMPONENT AS A FUNCTION OF N, NUMBER OF CLOSEST NEIGHBORS
# Lists to store the parameters for different N values
average_clustering_coefficients = []
giant_component_sizes = []

# Loop through N from 0 to 20
for N in range(0, 21):
    G_nx = build_graph_with_N_neighbors(nodes_data, N)

    # GSCC
    # Find strongly connected components
    strongly_connected_components = list(nx.strongly_connected_components(G_nx))
    # Find the size of the largest strongly connected component
    giant_component = max(strongly_connected_components, key=len)
    # Normalize the size by dividing by the total number of nodes
    normalized_giant_component_size = len(giant_component) / len(G_nx.nodes())
    giant_component_sizes.append(normalized_giant_component_size)

    # Clustering coefficient accounting for 0 values
    clustering_coefficients = nx.clustering(G_nx)
    valid_clustering_coefficients = [cc for cc in clustering_coefficients.values() if not np.isnan(cc)]
    if len(valid_clustering_coefficients) > 0:
        average_clustering_coefficient = sum(valid_clustering_coefficients) / len(valid_clustering_coefficients)
    else:
        average_clustering_coefficient = np.nan
    average_clustering_coefficients.append(average_clustering_coefficient)

# Plotting the results
plt.plot(range(0,21), giant_component_sizes, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Closest Neighbors (N)')
plt.ylabel('Size of Giant Strongly Connected Component')
plt.title('Giant Strongly Connected Component Size vs. N')
plt.grid(True)
plt.show()

plt.plot(range(0, 21), average_clustering_coefficients, marker='o', linestyle='-', color='b')
plt.xticks(range(0, 21), range(0, 21))  # Set x-ticks to integer values
plt.xlabel('Number of Closest Neighbors (N)')
plt.ylabel('Average Clustering Coefficient')
plt.title('Average Clustering Coefficient vs. Number of Closest Neighbors')
plt.grid(True)
plt.show()