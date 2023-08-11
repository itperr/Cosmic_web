import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

# Read node data from the text file
nodes_data = []
with open('nodes', 'r') as file:
    for line in file:
        columns = line.split()
        node = int(columns[0])  # First column contains the node ID
        x = float(columns[1])  # Second column contains the x coordinate
        y = float(columns[2])  # Third column contains the y coordinate
        z = float(columns[3])  # Fourth column contains the z coordinate
        # uncomment the following two lines for data from the 'TracingTheCosmicWebData' dataset, when using FOF files
        # condition_value = int(columns[14])  # Fifteenth column contains the condition value
        # if condition_value == 3:
        nodes_data.append((node, (x, y, z)))

# # # M1 MODEL: CONNECT NODES IF THE DISTANCE BETWEEN THEM IS SMALLER THAN R
# #
# # # Create a 3D complex network
# # G = nx.Graph()
# # # Add nodes to the graph
# # for node, x, y, z in nodes_data:
# #     G.add_node(node, pos=(x, y, z))
# # R = 5.0  # Maximum distance for creating edges
# # # Create edges based on distance
# # for i, (node1, x1, y1, z1) in enumerate(nodes_data):
# #     for j, (node2, x2, y2, z2) in enumerate(nodes_data[i+1:], i+1):
# #         distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
# #         if distance < R:
# #             G.add_edge(node1, node2)
# #
# # # VISUALIZATION
# # # Get the positions of nodes for plotting
# # pos = nx.get_node_attributes(G, 'pos')
# #
# # # Create the plot
# # fig = plt.figure(figsize=(10, 8))
# # ax = fig.add_subplot(111, projection='3d')
# #
# # # Plot edges
# # for edge in G.edges():
# #     node1, node2 = edge
# #     x = [pos[node1][0], pos[node2][0]]
# #     y = [pos[node1][1], pos[node2][1]]
# #     z = [pos[node1][2], pos[node2][2]]
# #     ax.plot(x, y, z, c='yellow', linewidth=1.0)
# #
# # # Plot nodes
# # for node in G.nodes():
# #     x, y, z = pos[node]
# #     ax.scatter(x, y, z, c='white', s=10, edgecolors='black', linewidths=0.5)
# #
# # # Customize the plot
# # ax.set_facecolor('black')
# # ax.view_init(elev=20, azim=30)  # Adjust the elevation and azimuth angles to rotate the plot
# #
# # plt.title('3D Complex Network', fontsize=16)
# # plt.show()
# #
# # # DEGREE DISTRIBUTION
# #
# # # Calculate the degree for each node
# # degree_sequence = [d for n, d in G.degree()]
# #
# # # Calculate the cumulative degree distribution
# # cumulative_distribution = {}
# # for degree in degree_sequence:
# #     if degree in cumulative_distribution:
# #         cumulative_distribution[degree] += 1
# #     else:
# #         cumulative_distribution[degree] = 1
# #
# # # Sort the cumulative distribution by degree and include zero counts for missing degrees
# # max_degree = max(degree_sequence)
# # sorted_cumulative = [(degree, cumulative_distribution.get(degree, 0)) for degree in range(max_degree + 1)]
# #
# # # Compute cumulative values
# # cumulative_values = [sum(count for degree, count in sorted_cumulative if degree >= x) for x in range(max_degree + 1)]
# #
# # # Plot the cumulative degree distribution on a log-log scale
# # degrees, counts = zip(*sorted_cumulative)
# # plt.loglog(degrees, cumulative_values, marker='o', linestyle='-', color='b')
# # plt.xlabel('Degree (log scale)')
# # plt.ylabel('Cumulative Count (log scale)')
# # plt.title('Cumulative Degree Distribution')
# # plt.grid(True)
# # plt.show()
#
# M2 MODEL: CONNECT NODES ONLY WITH THE N CLOSEST NEIGHBORS
# Create a function to build a graph with N closest neighbors for each node
def build_graph_with_N_neighbors(nodes_data, N):
    G = nx.DiGraph()

    for node, pos in nodes_data:
        G.add_node(node, pos=pos)

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

# 3D VISUALIZATION
# Create the plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot edges
for edge in G.edges():
    node1, node2 = edge
    x = [G.nodes[node1]['pos'][0], G.nodes[node2]['pos'][0]]
    y = [G.nodes[node1]['pos'][1], G.nodes[node2]['pos'][1]]
    z = [G.nodes[node1]['pos'][2], G.nodes[node2]['pos'][2]]
    ax.plot(x,y,z, c='yellow', linewidth=0.5)

# Plot nodes with size proportional to in-degree
for node in G.nodes():
    x, y, z = G.nodes[node]['pos']
    in_degree = G.in_degree(node)
    node_size = 2*in_degree  # Adjust the size scaling factor as needed
    ax.scatter(x, y, z, c='white', s=node_size, edgecolors='black', linewidths=0.5)

# Customize the plot
ax.set_facecolor('black')
ax.view_init(elev=20, azim=30)  # Adjust the elevation and azimuth angles to rotate the plot

plt.title('Cosmic Web', fontsize=16)
plt.show()

# # # 2D VISUALIZATION
# # # Create the plot
# # fig = plt.figure(figsize=(10, 8))
# #
# # # Plot edges
# # for edge in G.edges():
# #     node1, node2 = edge
# #     x = [G.nodes[node1]['pos'][0], G.nodes[node2]['pos'][0]]
# #     y = [G.nodes[node1]['pos'][1], G.nodes[node2]['pos'][1]]
# #     plt.plot(x, y, c='yellow', linewidth=1.0)
# #
# # # Plot nodes with size proportional to in-degree
# # for node in G.nodes():
# #     x, y, _ = G.nodes[node]['pos']  # Project onto the XY-plane
# #     in_degree = G.in_degree(node)
# #     node_size = 10 + in_degree  # Adjust the size scaling factor as needed
# #     plt.scatter(x, y, c='white', s=node_size, edgecolors='black', linewidths=0.5)
# #
# # # Customize the plot
# # plt.gca().set_facecolor('black')
# # plt.title('Cosmic Web - Projected onto XY Plane', fontsize=16)
# # plt.xlabel('X')
# # plt.ylabel('Y')
# # plt.grid(True)
# #
# # plt.show()

#DEGREE DISTRIBUTION
# Calculate the degree distribution of your network (why Poisson and not scale-free??? Because in general,
# the scale-free property is absent in systems that have a limitation in
# the number of links a node can have, as such limitations limit the size of the hubs.)
degree_sequence = [degree for node, degree in G.degree()]

# Plot the degree distribution on a logarithmic scale
plt.hist(degree_sequence, bins=20, density=True, log=True, color='b', alpha=0.7)
plt.xlabel('Degree')
plt.ylabel('Probability')
plt.title('Degree Distribution of Your Network')
plt.grid(True)
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

# # #ADJACENCY MATRIX
# # adjacency_matrix = nx.to_numpy_array(G)
# # print(adjacency_matrix)
#
# # IN-DEGREE CUMULATIVE DISTRIBUTION (doesn't reproduce the behaviour of a scale-free network!!!)
# # # Get the in-degree for each node
# # in_degrees = dict(G_nx.in_degree())
# #
# # # Count the number of nodes with each in-degree
# # in_degree_counts = {}
# # for degree in in_degrees.values():
# #     in_degree_counts[degree] = in_degree_counts.get(degree, 0) + 1
# #
# # # Calculate the cumulative in-degree distribution
# # total_nodes = len(G_nx.nodes())
# # cumulative_distribution = {}
# # cumulative_count = 0
# # for degree in sorted(in_degree_counts.keys()):
# #     cumulative_count += in_degree_counts[degree]
# #     cumulative_distribution[degree] = cumulative_count / total_nodes
# #
# # # Print the cumulative in-degree distribution
# # print("In-degree   Cumulative Distribution")
# # for degree, cdf in cumulative_distribution.items():
# #     print(f"{degree:8}   {cdf:.6f}")
# #
# # # Plot the cumulative in-degree distribution
# # plt.plot(list(cumulative_distribution.keys()), list(cumulative_distribution.values()), marker='o')
# # plt.xlabel('In-degree')
# # plt.ylabel('Cumulative Distribution')
# # plt.title('Cumulative In-degree Distribution')
# # plt.grid(True)
# # plt.show()
#
# # AVERAGE CLUSTERING COEFFICIENT AND GIANT STRONGLY CONNECTED COMPONENT AS A FUNCTION OF N, NUMBER OF CLOSEST NEIGHBORS
# Lists to store the parameters for different N values
average_clustering_coefficients = []
random_average_clustering_coefficients = []
giant_component_sizes = []
average_connectivities = []
random_average_connectivities = []

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

    # Average clustering coefficient for a random network C = <N>/num of nodes
    # Calculate the mean degree of the graph
    mean_degree = sum(dict(G_nx.degree()).values()) / len(G_nx.nodes())
    random_average_clustering_coefficients.append(mean_degree/G_nx.number_of_nodes())

    # Average connectivity
    average_connectivities.append(mean_degree)
    # Calculate the average random connectivity based on clustering coefficients
    random_average_connectivities.append((mean_degree/G_nx.number_of_nodes())*(1-mean_degree/G_nx.number_of_nodes())/G_nx.number_of_nodes()   )

# Plotting the results
plt.plot(range(0,21), giant_component_sizes, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Closest Neighbors (N)')
plt.ylabel('Size of Giant Strongly Connected Component')
plt.title('Giant Strongly Connected Component Size vs. N')
plt.grid(True)
plt.show()

# Plot the average clustering coefficient for both our network and a random one (superimposed)
plt.plot(range(0, 21), average_clustering_coefficients, marker='o', linestyle='-', color='b', label='Average Clustering Coefficient')
plt.plot(range(0, 21), random_average_clustering_coefficients, marker='o', linestyle='-', color='r', label='Random Average Clustering Coefficient')
plt.xticks(range(0, 21), range(0, 21))  # Set x-ticks to integer values
plt.xlabel('Number of Closest Neighbors (N)')
plt.ylabel('Average Clustering Coefficient')
plt.title('Comparison of Average Clustering Coefficients')
plt.grid(True)
plt.legend()  # Display legend to differentiate the two plots
# Show the plot
plt.show()

# Plot the average connectivity for both our network and a random one (side by side)
# Create a figure with two subplots arranged horizontally
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Plot the average connectivities of our network
axes[0].plot(range(0,21), average_connectivities, marker='o', linestyle='-', color='b', label='Your Network')
axes[0].set_xlabel('Number of Closest Neighbors (N)')
axes[0].set_ylabel('Average Connectivity')
axes[0].set_title('Average Connectivity of Our Network')
axes[0].grid(True)
axes[0].legend()
axes[0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set integer x-axis ticks
# Plot the average connectivities of the random network
axes[1].plot(range(0,21), random_average_connectivities, marker='o', linestyle='-', color='r', label='Random Network')
axes[1].set_xlabel('Number of Closest Neighbors (N)')
axes[1].set_ylabel('Average Connectivity')
axes[1].set_title('Average Connectivity of Random Network')
axes[1].grid(True)
axes[1].legend()
axes[1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))  # Set integer x-axis ticks
# Adjust layout for better spacing
plt.tight_layout()
# Show the combined plot
plt.show()