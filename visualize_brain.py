import pickle
import sys
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_brain_state(filepath: str = "braincore.pkl") -> dict:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de estado en: {path.absolute()}")
    with path.open("rb") as f:
        payload = pickle.load(f)
    return payload

def visualize_cognitive_graph(graph_data: dict) -> None:
    """
    Construye y grafica la red topológica del CognitiveGraph (Nodos y Aristas).
    """
    G = nx.DiGraph()
    
    nodes = graph_data.get("nodes", {})
    edges = graph_data.get("edges", [])
    
    for nid, n_info in nodes.items():
        G.add_node(nid, kind=n_info.get("kind", "entity"), belief=n_info.get("belief", 0.5))
        
    for edge in edges:
        G.add_edge(edge["src"], edge["dst"], relation=edge["relation"], weight=edge.get("weight", 1.0))
        
    print(f"[Grafo Cognitivo] Nodos cargados: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")
    
    if G.number_of_nodes() == 0:
        print("El grafo está vacío. No hay nodos para renderizar.")
        return

    plt.figure(figsize=(12, 10))
    # Usar spring_layout para calcular la distribución topológica
    pos = nx.spring_layout(G, seed=7, k=0.15, iterations=50)
    
    # Mapear colores según el tipo de nodo (kind)
    kinds = [G.nodes[n].get('kind', 'entity') for n in G.nodes]
    unique_kinds = list(set(kinds))
    color_map = {k: i for i, k in enumerate(unique_kinds)}
    node_colors = [color_map[G.nodes[n].get('kind', 'entity')] for n in G.nodes]
    
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color=node_colors, cmap=plt.cm.Set3, alpha=0.85)
    nx.draw_networkx_edges(G, pos, width=0.4, alpha=0.25, edge_color='gray', arrows=True, arrowsize=5)
    
    plt.title(f"IAG V4 - Topología del Grafo Cognitivo ({G.number_of_nodes()} nodos)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()

def visualize_attractors(creativity_data: dict) -> None:
    """
    Reduce la dimensionalidad de los atractores latentes mediante PCA y los grafica en 3D o 2D.
    """
    attractors = creativity_data.get("attractors", [])
    if not attractors:
        print("[Atractores] No hay atractores registrados en el estado.")
        return
        
    centers = np.array([a["center"] for a in attractors], dtype=np.float32)
    labels = [a["label"] for a in attractors]
    energies = [a["energy"] for a in attractors]
    
    print(f"[Atractores] Pozos de energía detectados: {len(attractors)}")
    
    # Reducción de dimensionalidad a 3 componentes principales
    n_comp = 3 if centers.shape[0] >= 3 and centers.shape[1] >= 3 else 2
    pca = PCA(n_components=n_comp)
    reduced = pca.fit_transform(centers)
    
    fig = plt.figure(figsize=(10, 8))
    
    if n_comp == 3:
        ax = fig.add_subplot(projection='3d')
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=energies, cmap='viridis', s=120, alpha=0.9)
        ax.set_title("Espacio Latente de Atractores (Proyección PCA 3D)", fontsize=14)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        
        for i, label in enumerate(labels):
            ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], f" {label}", fontsize=8, alpha=0.7)
    else:
        ax = fig.add_subplot()
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=energies, cmap='viridis', s=120, alpha=0.9)
        ax.set_title("Espacio Latente de Atractores (Proyección PCA 2D)", fontsize=14)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        
        for i, label in enumerate(labels):
            ax.text(reduced[i, 0], reduced[i, 1], f" {label}", fontsize=8, alpha=0.7)
            
    fig.colorbar(scatter, label="Nivel de Energía del Atractor")
    plt.tight_layout()

if __name__ == "__main__":
    target_file = sys.argv[1] if len(sys.argv) > 1 else "braincore.pkl"
    
    print(f"Leyendo archivo de persistencia: {target_file}")
    try:
        brain_state = load_brain_state(target_file)
    except Exception as e:
        print(f"Error al cargar el archivo de cerebro: {e}")
        sys.exit(1)
        
    # Renderizar sub-módulos cognitivos disponibles en el volcado
    if "graph" in brain_state:
        visualize_cognitive_graph(brain_state["graph"])
    else:
        print("El estado no contiene la clave 'graph'.")
        
    if "creativity" in brain_state:
        visualize_attractors(brain_state["creativity"])
    else:
        print("El estado no contiene la clave 'creativity'.")
        
    # Mostrar todas las figuras generadas en pantalla
    plt.show()