import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Load data from CSV file
csv_file_path = 'kokkinia_data.csv'
df = pd.read_csv(csv_file_path, delimiter=',')

# Create a directed graph with weights
data = [(int(row[0]), int(row[1]), float(row[2])) for row in df.values]
G = nx.DiGraph()
i=0
for edge in data:
    G.add_edge(edge[0], edge[1], weight=edge[2])
    i+=1
    # if (i>= 20):break

# Συναρτηση Υπολογισμου του Αριθμου Αισθητηρων αναλογα με το μηκος του Σωληνα
def calculate_flow_sensors(pipe_length, thresholds, sensor_counts):
    return next((count for thresh, count in zip(thresholds, sensor_counts) if pipe_length < thresh), sensor_counts[-1])


# Function to add a node in the middle of each edge
def add_node_in_middle(graph):
    total_num_of_sensors = 0
    new_graph = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        length = data['weight']
        
        thresholds = [2.0, 10.0, 40.0, 64.0]    # Ορια μηκων σωληνων
        sensor_counts = [0, 2, 5, 6]            # Αριθμος Αισθητηρων για καθε ευρος
        number_of_sensors = calculate_flow_sensors(length, thresholds, sensor_counts)
        
        # if (0<= length< 20):number_of_sensors = 2
        # if (20<= length< 45):number_of_sensors = 3
        # if (45<= length<= 60):number_of_sensors = 4
        total_num_of_sensors += number_of_sensors
        
        if (u not in new_graph):
            new_graph.add_node(u, pos=graph.nodes[u]['pos'],node_type='manhole')
        if (v not in new_graph):
            new_graph.add_node(v, pos=graph.nodes[v]['pos'],node_type='manhole')
        
        middle_points = []
        start_pos = graph.nodes[u]['pos']
        end_pos = graph.nodes[v]['pos']
        
       
        for i in range(1, number_of_sensors + 1):
            ratio = i / (number_of_sensors + 1.0)
            middle_point = (
                (1 - ratio) * start_pos[0] + ratio * end_pos[0],
                (1 - ratio) * start_pos[1] + ratio * end_pos[1]
            )
            new_graph.add_node(f'F{u}_{v}_{i}', pos=middle_point,node_type='sensor')
            
            middle_points.append(middle_point)

        sensors_dists = [round((i / (number_of_sensors + 1.0))*length,3) for i in range(1, number_of_sensors + 1)]
        sensor_names = [f'F{u}_{v}_{i}_{sens_dist}' for i,sens_dist in zip(range(1, number_of_sensors + 1),sensors_dists)]
        sens_poses = [u] +[v] + [length] + [number_of_sensors] + sensors_dists
        print(' '.join([str(x) for x in sens_poses]))
        # print(f'{u} {v} {length} {number_of_sensors} {sensor_names}')
            
        # print(u,v,length,number_of_sensors,[np.linalg.norm(point_b - graph.nodes[u]['pos']) for point_b in middle_points])
        
            
    return new_graph,total_num_of_sensors

# Set node positions for visualization
pos = nx.kamada_kawai_layout(G, weight='weight',scale=0.50)
nx.set_node_attributes(G, pos, 'pos')

# Add nodes in the middle of each edge
G2,total_num_of_sensors = add_node_in_middle(G)
pos2 = nx.get_node_attributes(G2, 'pos')

print(f'total_num_of_sensors:{total_num_of_sensors}')
# Draw nodes with attribute 'node_type' equal to 'sensor' in red
node_types = nx.get_node_attributes(G2, 'node_type')
manhole_nodes = [key for key, value in node_types.items() if value == 'manhole']
sensor_nodes = [key for key, value in node_types.items() if value == 'sensor']
nx.draw_networkx_nodes(G2, pos2,nodelist=manhole_nodes, node_size=50, node_color='skyblue')
nx.draw_networkx_nodes(G2, pos2,nodelist=sensor_nodes, node_size=5, node_color='blue')
nx.draw_networkx_labels(G2, pos2, labels={node: node for node in manhole_nodes}, font_size=8)
# nx.draw_networkx_labels(G2, pos2, labels={node: node for node in sensor_nodes}, font_size=6, font_color='blue')

nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color='gray')
edge_labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=6)

plt.show()








