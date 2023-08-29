import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import cKDTree
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from scipy.spatial import Voronoi
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap

# Read node data from the text file
nodes_data = []
with open('tweb', 'r') as file:
    for line in file:
        columns = line.split()
        node = int(columns[0])  # First column contains the node ID
        x = float(columns[1])  # Second column contains the x coordinate
        y = float(columns[2])  # Third column contains the y coordinate
        z = float(columns[3])  # Fourth column contains the z coordinate
        mass = float(columns[8])  # Ninth column contains the mass
        nodes_data.append((node, (x, y, z), mass))

# M1 MODEL: CONNECT NODES IF THE DISTANCE BETWEEN THEM IS SMALLER THAN R
def build_graph_dist(nodes_data, distance_threshold):
    G = nx.Graph()

    # Create a KD-tree for efficient spatial indexing
    positions = [pos for _, pos, _ in nodes_data]  # Extract positions
    kdtree = cKDTree(positions)

    # Add nodes to the graph
    for node, pos, mass in nodes_data:
        G.add_node(node, pos=pos, mass=mass)

    # Connect nodes based on distance threshold using KD-tree query
    for node, pos, mass in nodes_data:
        nearby_nodes = kdtree.query_ball_point(pos, distance_threshold)
        for neighbor_node in nearby_nodes:
            if neighbor_node != node:
                G.add_edge(node, nodes_data[neighbor_node][0])  # Adding edge using node ID

    return G

# M2 MODEL: CONNECT NODES ONLY WITH THE N CLOSEST NEIGHBORS
# Create a function to build a graph with N closest neighbors for each node
def build_graph_with_N_neighbors(nodes_data, N):
    G = nx.DiGraph()  # Create an undirected graph

    for node, pos, _ in nodes_data:  # Ignore the mass value using underscore
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


# M3 MODEL: CONNECT NODES BASED ON GRAVITATIONAL FORCE
def calculate_gravitational_force(m1, m2, r):
    G = 6.67430e-11  # Gravitational constant
    epsilon = 1e-10  # Small value to avoid division by zero
    force = (G * m1 * m2 * 1e+60) / ((r*1e+19)**2 + epsilon)
    return force

def euclidean_distance(point1, point2):
    return math.sqrt(sum((p2 - p1) ** 2 for p1, p2 in zip(point1, point2)))

def build_network_with_gravity(nodes_data, N, grid_size):
    G = nx.DiGraph()

    # Create a dictionary to store nodes in grid cells
    grid = defaultdict(list)

    # Create a dictionary to store node data indexed by node IDs
    node_dict = {node: (pos, mass) for node, pos, mass in nodes_data}

    # Populate the grid cells with nodes
    for node, pos, mass in nodes_data:
        x, y, z = pos
        grid[(int(x / grid_size), int(y / grid_size), int(z / grid_size))].append(node)

    # Connect nodes based on the N strongest gravitational forces using Euclidean distance and grid
    for node1, (x1, y1, z1), mass1 in nodes_data:
        grid_x, grid_y, grid_z = int(x1 / grid_size), int(y1 / grid_size), int(z1 / grid_size)
        force_list = []

        # Iterate over the 3x3x3 grid cells around the current cell
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_cell = (grid_x + dx, grid_y + dy, grid_z + dz)
                    for node2 in grid[neighbor_cell]:
                        if node1 != node2:
                            pos2, mass2 = node_dict[node2]
                            force = calculate_gravitational_force(mass1, mass2, euclidean_distance((x1, y1, z1), pos2))
                            force_list.append((node2, force))

        # Sort the force list by force in descending order and select the N strongest forces
        force_list.sort(key=lambda x: x[1], reverse=True)
        strongest_forces = force_list[:N]

        # Add edges for the N strongest forces
        for node2, _ in strongest_forces:
            G.add_edge(node1, node2)

    return G


# M1 MODEL: CONNECT NODES IF THE DISTANCE BETWEEN THEM IS SMALLER THAN R
R = 7.8
G_m1 = build_graph_dist(nodes_data, R)

# M2 MODEL: CONNECT NODES ONLY WITH THE N CLOSEST NEIGHBORS
N = 4
G_m2 = build_graph_with_N_neighbors(nodes_data, N)

# M3 MODEL: CONNECT NODES BASED ON GRAVITATIONAL FORCE
grid_size = 5
G_m3 = build_network_with_gravity(nodes_data, 5, grid_size)

# Create a list of all three models
networks = [G_m1, G_m2, G_m3]
model_names = ['M1', 'M2', 'M3']

# Add the 'pos' attribute to the nodes of each network
for G in [G_m1, G_m2]:
    for node, pos, _ in nodes_data:
        G.nodes[node]['pos'] = pos

# For M3, assign the 'pos' attribute to nodes using a different approach
for G in [G_m3]:
    for node in G.nodes():
        pos = nodes_data[node - 1][1]  # Node IDs are 1-indexed
        G.nodes[node]['pos'] = pos

# VISUALIZATION
# Create a figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 6), subplot_kw={'projection': '3d'})

for ax, network, model_name in zip(axes, networks, model_names):
    # Plot edges
    for edge in network.edges():
        node1, node2 = edge
        x = [network.nodes[node1]['pos'][0], network.nodes[node2]['pos'][0]]
        y = [network.nodes[node1]['pos'][1], network.nodes[node2]['pos'][1]]
        z = [network.nodes[node1]['pos'][2], network.nodes[node2]['pos'][2]]
        ax.plot(x, y, z, c='yellow', linewidth=0.5)

    # Calculate node degrees based on the model
    if model_name == 'M2':
        degrees = dict(network.in_degree())
    else:
        degrees = dict(network.degree())

    # Plot nodes with size proportional to degree/in-degree
    for node in network.nodes():
        x, y, z = network.nodes[node]['pos']
        degree = degrees.get(node, 0)
        node_size = 0.75 * degree  # Adjust the size scaling factor as needed
        ax.scatter(x, y, z, c='white', s=node_size, edgecolors='black', linewidths=0.5)

    # Customize the plot
    ax.set_facecolor('black')
    ax.view_init(elev=20, azim=30)  # Adjust the elevation and azimuth angles to rotate the plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title(f'Cosmic Web - {model_name}', fontsize=14)

# Add a common title for the entire figure
fig.suptitle('Comparison of Cosmic Web Networks', fontsize=16)
plt.tight_layout()
plt.show()

# DEGREE DISTRIBUTIONS (IN-DEGREE FOR M2, M3 SINCE DIRECTED GRAPH)
# Plot the degree distribution for all models
plt.figure(figsize=(10, 6))

for G, model_name in zip([G_m1, G_m2, G_m3], ['M1', 'M2', 'M3']):
    if model_name == 'M1':  # Use degree for M1
        degrees = dict(G.degree()).values()
    else:
        degrees = dict(G.in_degree()).values()  # Use in-degree for M2, M3 since they're directed
    plt.hist(list(degrees), bins=50, alpha=0.5, label=model_name)

plt.xlabel('Degree')
plt.ylabel('Frequency')
plt.title('Degree Distribution for Different Models')
plt.legend()
plt.grid()
plt.show()

# AVERAGE CLUSTERING COEFFICIENT, SIZE OF GIANT STRONGLY CONNECTED COMPONENT AS A FUNCTION OF R/N
# Define a range of values for R or N
param_values = range(0, 21)  # You can adjust the range as needed

# Lists to store clustering coefficients for each model
clustering_m1 = []
clustering_m2 = []
clustering_m3 = []

# Lists to store sizes of the largest connected component for each model
largest_cc_m1 = []
largest_cc_m2 = []
largest_cc_m3 = []

for param in param_values:
    # Build the networks with the current parameter value
    G_m1 = build_graph_dist(nodes_data, param)  # Adjust the function based on the parameter (R or N)
    G_m2 = build_graph_with_N_neighbors(nodes_data, param)  # Adjust the function based on the parameter (R or N)
    G_m3 = build_network_with_gravity(nodes_data, param, grid_size)  # Adjust the function based on the parameter (R or N)

    # Calculate clustering coefficients accounting for disconnected graphs
    cc_m1 = nx.average_clustering(G_m1) if G_m1.number_of_nodes() > 0 else 0.0
    cc_m2 = nx.average_clustering(G_m2) if G_m2.number_of_nodes() > 0 else 0.0
    cc_m3 = nx.average_clustering(G_m3) if G_m3.number_of_nodes() > 0 else 0.0
    clustering_m1.append(cc_m1)
    clustering_m2.append(cc_m2)
    clustering_m3.append(cc_m3)

    # Calculate sizes of the largest connected components accounting for disconnected graphs
    largest_cc_m1.append(len(max(nx.connected_components(G_m1), key=len)) if G_m1.number_of_nodes() > 0 else 0)
    largest_cc_m2.append(len(max(nx.strongly_connected_components(G_m2), key=len)) if G_m2.number_of_nodes() > 0 else 0)
    largest_cc_m3.append(len(max(nx.strongly_connected_components(G_m3), key=len)) if G_m3.number_of_nodes() > 0 else 0)

# Plot the results for CC
plt.figure(figsize=(10, 6))
plt.plot(param_values, clustering_m1, label='M1', marker='o')
plt.plot(param_values, clustering_m2, label='M2', marker='s')
plt.plot(param_values, clustering_m3, label='M3', marker='^')

plt.xticks(range(0, 21), range(0, 21))  # Set x-ticks to integer values
plt.xlabel('Parameter (R or N)')
plt.ylabel('Average Clustering Coefficient')
plt.title('Average Clustering Coefficient vs. Parameter')
plt.legend()
plt.grid()
plt.show()

# Plot the results for SSC
plt.figure(figsize=(10, 6))
plt.plot(param_values, largest_cc_m1, label='M1', marker='o')
plt.plot(param_values, largest_cc_m2, label='M2', marker='s')
plt.plot(param_values, largest_cc_m3, label='M3', marker='^')

plt.xticks(range(0, 21), range(0, 21))  # Set x-ticks to integer values
plt.xlabel('Parameter (R or N)')
plt.ylabel('Size of Largest Strongly Connected Component')
plt.title('Size of Largest Strongly Connected Component vs. Parameter')
plt.legend()
plt.grid()
plt.show()

# SMALL-WORLD
# Function to approximate average shortest path length using random walks
def approximate_average_shortest_path_length(G, num_walks):
    total_distance = 0
    num_valid_walks = 0
    for _ in range(num_walks):
        start_node = np.random.choice(list(G.nodes()))
        end_node = np.random.choice(list(G.nodes()))
        try:
            distance = nx.shortest_path_length(G, source=start_node, target=end_node)
            total_distance += distance
            num_valid_walks += 1
        except nx.NetworkXNoPath:
            pass  # Ignore cases where there is no path
    if num_valid_walks > 0:
        return total_distance / num_valid_walks
    else:
        return float('inf')  # Return infinity if no valid paths were found

# Function to generate a random network with the same number of nodes and edges as the original network
def generate_random_network(G):
    random_G = nx.fast_gnp_random_graph(len(G), nx.density(G))
    return random_G

# Calculate the clustering coefficient for each network
clustering_m1 = nx.average_clustering(G_m1)
clustering_m2 = nx.average_clustering(G_m2)
clustering_m3 = nx.average_clustering(G_m3)

# Calculate the approximate average shortest path length for each network
num_random_walks = 20000
avg_path_length_m1 = approximate_average_shortest_path_length(G_m1, num_random_walks)
avg_path_length_m2 = approximate_average_shortest_path_length(G_m2, num_random_walks)
avg_path_length_m3 = approximate_average_shortest_path_length(G_m3, num_random_walks)

# Calculate the clustering coefficient and approximate average shortest path length for the corresponding random networks
random_G_m1 = generate_random_network(G_m1)
random_G_m2 = generate_random_network(G_m2)
random_G_m3 = generate_random_network(G_m3)

clustering_random_m1 = nx.average_clustering(random_G_m1)
clustering_random_m2 = nx.average_clustering(random_G_m2)
clustering_random_m3 = nx.average_clustering(random_G_m3)

avg_path_length_random_m1 = approximate_average_shortest_path_length(random_G_m1, num_random_walks)
avg_path_length_random_m2 = approximate_average_shortest_path_length(random_G_m2, num_random_walks)
avg_path_length_random_m3 = approximate_average_shortest_path_length(random_G_m3, num_random_walks)

# Compare the metrics for each network with its corresponding random network
def is_small_world(clustering, avg_path_length, clustering_random, avg_path_length_random):
    return clustering > clustering_random and avg_path_length <= avg_path_length_random

# Check if the networks are small-world networks and print the values
small_world_m1 = is_small_world(clustering_m1, avg_path_length_m1, clustering_random_m1, avg_path_length_random_m1)
print("Network M1:")
print("    Clustering Coefficient:", clustering_m1)
print("    Avg. Shortest Path Length:", avg_path_length_m1)
print("    Clustering Coefficient (Random):", clustering_random_m1)
print("    Avg. Shortest Path Length (Random):", avg_path_length_random_m1)
print("    Is a Small-World Network:", small_world_m1)

small_world_m2 = is_small_world(clustering_m2, avg_path_length_m2, clustering_random_m2, avg_path_length_random_m2)
print("Network M2:")
print("    Clustering Coefficient:", clustering_m2)
print("    Avg. Shortest Path Length:", avg_path_length_m2)
print("    Clustering Coefficient (Random):", clustering_random_m2)
print("    Avg. Shortest Path Length (Random):", avg_path_length_random_m2)
print("    Is a Small-World Network:", small_world_m2)

small_world_m3 = is_small_world(clustering_m3, avg_path_length_m3, clustering_random_m3, avg_path_length_random_m3)
print("Network M3:")
print("    Clustering Coefficient:", clustering_m3)
print("    Avg. Shortest Path Length:", avg_path_length_m3)
print("    Clustering Coefficient (Random):", clustering_random_m3)
print("    Avg. Shortest Path Length (Random):", avg_path_length_random_m3)
print("    Is a Small-World Network:", small_world_m3)

# PLOT HOW THE IN-DEGREE DISTRIBUTION FOR M3 CHANGES AS A FUNCTION OF grid_size TO DETERMINE WHICH VALUE TO USE
# List to store grid_size values
grid_sizes = range(1, 11)  # Range of grid_size values

# Create a list to store the in-degree distributions for different grid sizes
in_degree_distributions = []

# Loop through different grid sizes
for grid_size in grid_sizes:
    G_m3 = build_network_with_gravity(nodes_data, N, grid_size)
    in_degrees = [deg for _, deg in G_m3.in_degree()]
    in_degree_distributions.append(in_degrees)

# Plot the in-degree distributions using histograms with different colors
plt.figure(figsize=(10, 6))
for i, grid_size in enumerate(grid_sizes):
    plt.hist(
        in_degree_distributions[i],
        bins=range(max(in_degree_distributions[i]) + 1),
        alpha=0.5,
        label=f'Grid Size {grid_size}',
    )

plt.xlabel('In-Degree')
plt.ylabel('Frequency')
plt.title('In-Degree Distribution of M3 for Different Grid Sizes')
plt.legend()
plt.grid()
plt.show()

# VORONOI
# Extract galaxy positions
galaxy_positions = np.array([pos for _, pos, _ in nodes_data])

# Calculate Voronoi tessellation
vor = Voronoi(galaxy_positions)

# Define criteria for voids, walls, and clusters. Parameters still to be adjusted
void_volume_threshold = 2000  # Adjust as needed
cluster_volume_threshold = 100  # Adjust as needed

void_indices = []
cluster_indices = []
wall_indices = []

for i, region in enumerate(vor.regions):
    if -1 in region or len(region) == 0:
        continue  # Skip infinite or empty Voronoi regions

    valid_vertices = [vertex for vertex in region if vertex >= 0]

    if len(valid_vertices) < 3:
        continue  # Skip regions with too few vertices

    polygon = vor.vertices[valid_vertices]
    centroid = np.mean(polygon, axis=0)
    volume = np.sum(np.linalg.norm(polygon - centroid, axis=1)) / 3

    if volume > void_volume_threshold:
        void_indices.append(i)
    elif volume < cluster_volume_threshold:
        cluster_indices.append(i)
    else:
        wall_indices.append(i)

# Write the node IDs and corresponding labels to a text file
with open('cosmic_web_labels.txt', 'w') as output_file:
    for node_id in range(len(nodes_data)):
        label = -1  # Default label if not identified
        if node_id in void_indices:
            label = 0
        elif node_id in wall_indices:
            label = 1
        elif node_id in cluster_indices:
            label = 3
        output_file.write(f"{node_id} {pos[0]} {pos[1]} {pos[2]} {label}\n")

# VISUALIZATION
# Read the labels from the generated text file
node_labels = {}
with open('cosmic_web_labels.txt', 'r') as label_file:
    for line in label_file:
        columns = line.strip().split()  # Split line into columns
        node_id = int(columns[0])  # First column is the ID
        label = int(columns[4])  # Fifth column is the label
        node_labels[node_id] = label

# Separate nodes based on their labels
void_nodes = [node for node, label in node_labels.items() if label == 0]
wall_nodes = [node for node, label in node_labels.items() if label == 1]
cluster_nodes = [node for node, label in node_labels.items() if label == 3]

# Calculate the percentages
total_nodes = len(nodes_data)
void_percentage = len(void_nodes) / total_nodes * 100
wall_percentage = len(wall_nodes) / total_nodes * 100
cluster_percentage = len(cluster_nodes) / total_nodes * 100

# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot void nodes
ax.scatter(galaxy_positions[void_nodes, 0], galaxy_positions[void_nodes, 1], galaxy_positions[void_nodes, 2], c='r', marker='o', s=5, label='Void Nodes')

# Plot wall nodes
ax.scatter(galaxy_positions[wall_nodes, 0], galaxy_positions[wall_nodes, 1], galaxy_positions[wall_nodes, 2], c='g', marker='^', s=5, label='Wall Nodes')

# Plot cluster nodes
ax.scatter(galaxy_positions[cluster_nodes, 0], galaxy_positions[cluster_nodes, 1], galaxy_positions[cluster_nodes, 2], c='b', marker='s', s=5, label='Cluster Nodes')

# Add percentage annotations to the plot
ax.text2D(0.05, 0.95, f"Void Percentage: {void_percentage:.2f}%", transform=ax.transAxes, color='red')
ax.text2D(0.05, 0.90, f"Wall Percentage: {wall_percentage:.2f}%", transform=ax.transAxes, color='green')
ax.text2D(0.05, 0.85, f"Cluster Percentage: {cluster_percentage:.2f}%", transform=ax.transAxes, color='blue')

# Customize the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Distribution of Nodes Classified as Void, Cluster, or Wall')
ax.legend()

plt.show()

# LOOPING THROUGH X
# Calculate the degree for each node in M1 and the in-degree for each node in M2 and M3
degrees_m1 = dict(G_m1.degree())
in_degrees_m2 = dict(G_m2.in_degree())
in_degrees_m3 = dict(G_m3.in_degree())

# Define the ID assignment criteria function with parameters x and y
def assign_id(d, y, x):
    if d < y:
        return 0
    elif y <= d <= x:
        return 1
    else:
        return 3

# Range of y values to loop through
y_values = range(1, 21)

# Range of x values to loop through (dependent on y)
x_values_list = [range(y+1, 41) for y in y_values]

# Create a 2D numpy array to store percentage of matching IDs for each network
percentage_matching_array = np.zeros((len(y_values), len(x_values_list[0]), 3))  # 3 columns for M1, M2, and M3

# Loop through combinations of x and y
for y_index, (y, x_values) in enumerate(zip(y_values, x_values_list)):
    for x_index, x in enumerate(x_values):
        # Calculate the degree for each node in M1 and the in-degree for each node in M2 and M3
        degrees_m1 = dict(G_m1.degree())
        in_degrees_m2 = dict(G_m2.in_degree())
        in_degrees_m3 = dict(G_m3.in_degree())

        # Assign IDs based on the provided function with the current values of x and y
        ids_m1 = {node: assign_id(degree, y, x) for node, degree in degrees_m1.items()}
        ids_m2 = {node: assign_id(in_degree, y, x) for node, in_degree in in_degrees_m2.items()}
        ids_m3 = {node: assign_id(in_degree, y, x) for node, in_degree in in_degrees_m3.items()}

        # Read the text file and compare IDs separately for each model
        matching_count_m1 = 0
        total_count_m1 = 0

        matching_count_m2 = 0
        total_count_m2 = 0

        matching_count_m3 = 0
        total_count_m3 = 0

        with open('cosmic_web_labels.txt', 'r') as file:
            for line in file:
                columns = line.split()
                node = int(columns[0])  # Node label from the first column
                id_from_file = int(columns[4])  # Node ID from the 5th column

                if id_from_file in [0, 1, 3]:
                    total_count_m1 += 1
                    total_count_m2 += 1
                    total_count_m3 += 1

                    if ids_m1.get(node) == id_from_file:
                        matching_count_m1 += 1

                    if ids_m2.get(node) == id_from_file:
                        matching_count_m2 += 1

                    if ids_m3.get(node) == id_from_file:
                        matching_count_m3 += 1

        # Calculate and store the percentage of matching IDs for each model
        percentage_matching_m1 = (matching_count_m1 / total_count_m1) * 100 if total_count_m1 > 0 else 0
        percentage_matching_m2 = (matching_count_m2 / total_count_m2) * 100 if total_count_m2 > 0 else 0
        percentage_matching_m3 = (matching_count_m3 / total_count_m3) * 100 if total_count_m3 > 0 else 0

        percentage_matching_array[y_index, x_index, 0] = percentage_matching_m1
        percentage_matching_array[y_index, x_index, 1] = percentage_matching_m2
        percentage_matching_array[y_index, x_index, 2] = percentage_matching_m3

# Create a meshgrid for x and y values
x_mesh, y_mesh = np.meshgrid(x_values, y_values)

# Create an empty array to store the z values
z_values_m1 = np.zeros((len(y_values), len(x_values)))
z_values_m2 = np.zeros((len(y_values), len(x_values)))
z_values_m3 = np.zeros((len(y_values), len(x_values)))

# Populate the z value arrays for each model
for y_idx, y_val in enumerate(y_values):
    for x_idx, x_val in enumerate(x_values):
        z_values_m1[y_idx, x_idx] = percentage_matching_array[y_idx, x_idx, 0]
        z_values_m2[y_idx, x_idx] = percentage_matching_array[y_idx, x_idx, 1]
        z_values_m3[y_idx, x_idx] = percentage_matching_array[y_idx, x_idx, 2]

# Create a 3D figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the percentage of agreement for each model as a surface plot
surf1 = ax.plot_surface(x_mesh, y_mesh, z_values_m1, cmap='viridis', alpha=0.7)
surf2 = ax.plot_surface(x_mesh, y_mesh, z_values_m2, cmap='plasma', alpha=0.7)
surf3 = ax.plot_surface(x_mesh, y_mesh, z_values_m3, cmap='inferno', alpha=0.7)

# Customize the plot labels and title
ax.set_xlabel('Parameter x')
ax.set_ylabel('Parameter y')
ax.set_zlabel('Percentage of Matching IDs')
ax.set_title('Percentage of Agreement in Node IDs for Different Values of x and y')

# Create color maps with the same colors as the surfaces
cmap_m1 = ListedColormap(surf1.get_cmap()(np.linspace(0.3, 0.9, 256)))
cmap_m2 = ListedColormap(surf2.get_cmap()(np.linspace(0.3, 0.9, 256)))
cmap_m3 = ListedColormap(surf3.get_cmap()(np.linspace(0.3, 0.9, 256)))

# Create colorbars using the custom color maps
cax = fig.add_axes([0.72, 0.15, 0.02, 0.65])  # Adjust position and size as needed
cbar_m1 = plt.colorbar(surf1, cax=cax, cmap=cmap_m1, orientation='vertical')
cbar_m1.set_label('Model M1')

cax2 = fig.add_axes([0.82, 0.15, 0.02, 0.65])  # Adjust position and size as needed
cbar_m2 = plt.colorbar(surf2, cax=cax2, cmap=cmap_m2, orientation='vertical')
cbar_m2.set_label('Model M2')

cax3 = fig.add_axes([0.92, 0.15, 0.02, 0.65])  # Adjust position and size as needed
cbar_m3 = plt.colorbar(surf3, cax=cax3, cmap=cmap_m3, orientation='vertical')
cbar_m3.set_label('Model M3')

# Show the plot
plt.show()

# Find the maximum percentage and its corresponding x and y values for each model
max_percentage_m1 = np.max(z_values_m1)
max_percentage_m2 = np.max(z_values_m2)
max_percentage_m3 = np.max(z_values_m3)

max_index_m1 = np.unravel_index(np.argmax(z_values_m1), z_values_m1.shape)
max_index_m2 = np.unravel_index(np.argmax(z_values_m2), z_values_m2.shape)
max_index_m3 = np.unravel_index(np.argmax(z_values_m3), z_values_m3.shape)

print(f"Max Percentage for M1: {max_percentage_m1:.2f}% at (x, y) = ({x_values[max_index_m1[1]]}, {y_values[max_index_m1[0]]})")
print(f"Max Percentage for M2: {max_percentage_m2:.2f}% at (x, y) = ({x_values[max_index_m2[1]]}, {y_values[max_index_m2[0]]})")
print(f"Max Percentage for M3: {max_percentage_m3:.2f}% at (x, y) = ({x_values[max_index_m3[1]]}, {y_values[max_index_m3[0]]})")
