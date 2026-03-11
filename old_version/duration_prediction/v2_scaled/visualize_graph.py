
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt

# Ensure we can import modules if running as script (fallback)
# But we expect to run this via python -m
try:
    from .graph import KnowledgeGraph
except ImportError:
    # If someone runs this file directly without -m, this might help, 
    # but the relative imports in graph.py will still fail without package context.
    # So we mainly rely on the user running this correctly.
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from graph import KnowledgeGraph

def visualize():
    print("Initializing Knowledge Graph...")
    kg = KnowledgeGraph()
    G = kg.graph
    
    print(f"Graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")
    
    plt.figure(figsize=(12, 10))
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=2.0)  # k regulates the distance between nodes
    
    # Node colors by type
    node_colors = []
    labels = {}
    
    for node in G.nodes:
        node_type = G.nodes[node].get("type", "Unknown")
        labels[node] = f"{node}\n({node_type})"
        
        if node_type == "TaskType":
            node_colors.append("skyblue")
        elif node_type == "Condition":
            node_colors.append("lightgreen")
        elif node_type == "Risk":
            node_colors.append("salmon")
        else:
            node_colors.append("lightgray")

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color="gray", arrowsize=20)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family="sans-serif")
    
    # Edge Labels
    edge_labels = nx.get_edge_attributes(G, 'relationship')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=8)
    
    plt.title("Construction Knowledge Graph", size=15)
    plt.axis("off")
    
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_graph.png")
    plt.savefig(output_path, format="PNG", dpi=300, bbox_inches="tight")
    print(f"Graph visualization saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    visualize()
