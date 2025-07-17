"""
    Takes a tree : Tree as an input and creates a tree in real sense to see.
    See more : https://plotly.com/python/tree-plots/

    Usage : visualize_tree_structure(start_node : Node, tree : Tree)
"""

try:
    import plotly.graph_objects as go
except ImportError as e:
    raise ImportError("`plotly` not installed. Please install using `pip install plotly`")

try :
    from igraph import Graph, EdgeSeq
except ImportError as e:
    raise ImportError("`igraph` not installed. Please install using `pip install igraph`.")

# --- Pyvis optional backend ---
try:
    from pyvis.network import Network
except ImportError:
    Network = None  # handled later

from .tree_structures import Tree, Node


def format_text_for_plot(text: str) -> str:
    """
    Formats text for plotting by splitting long lines into shorter ones.
    """
    lines = text.split("\n")
    MAX_CHARS_PER_LINE = 80
    formatted_lines = []
    for line in lines:
        while len(line) > MAX_CHARS_PER_LINE:
            formatted_lines.append(line[:MAX_CHARS_PER_LINE])
            line = line[MAX_CHARS_PER_LINE:]
        formatted_lines.append(line)

    return "<br>".join(formatted_lines)


def create_visualization_figure(
    edge_x_coords,
    edge_y_coords,
    node_x_coords,
    node_y_coords,
    node_labels,
    node_colors=None,
):
    """
    Creates a Plotly figure for visualizing nodes and edges.
    """
    fig = go.Figure()

    # Add arrows for edges
    for i in range(0, len(edge_x_coords) - 1, 3):
        x_start, x_end = edge_x_coords[i], edge_x_coords[i + 1]
        y_start, y_end = edge_y_coords[i], edge_y_coords[i + 1]
        if x_start is not None and x_end is not None:  # Avoid None segments
            fig.add_annotation(
                x=x_end,
                y=y_end,
                ax=x_start,
                ay=y_start,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,  # Arrow style
                arrowsize=1.5,  # Size of the arrowhead
                arrowwidth=1,  # Arrow line width
                arrowcolor="rgb(210,210,210)",  # Color of the arrow
            )

    # Add nodes
    fig.add_trace(
        go.Scatter(
            x=node_x_coords,
            y=node_y_coords,
            mode="markers",
            name="nodes",
            marker=dict(
                symbol="circle-dot",
                size=18,
                color=node_colors if node_colors else "#6175c1",
                line=dict(color="rgb(50,50,50)", width=1),
            ),
            text=[format_text_for_plot(label) for label in node_labels],
            hoverinfo="text",
            opacity=0.8,
        )
    )

    return fig


def build_graph_from_tree(
    graph: Graph, current_node: Node, tree: Tree, parent_node_id: int = -1
) -> int:
    """
    Recursively builds a graph representation of the tree structure.
    """
    node_id = graph.vcount()
    graph.add_vertex(
        name=f"Node Index: {current_node.index}, Node Text: {current_node.text}",
        index=current_node.index,
        embeddings=current_node.embeddings,
    )

    if parent_node_id != -1:
        graph.add_edge(parent_node_id, node_id)

    for child_index in current_node.children:
        child_node = find_node_in_tree(child_index, tree)
        build_graph_from_tree(graph, child_node, tree, node_id)

    return node_id


def find_node_in_tree(node_index: int, tree: Tree) -> Node:
    """
    Finds a node in the tree by its index.
    """
    for key in tree.all_nodes:
        node = tree.all_nodes[key]
        if node.index == node_index:
            return node

    raise Exception(f"Node with index {node_index} not found")


def build_pyvis_network(
    start_node: Node,
    tree: Tree,
    highlight_node_indices: list[int] | None = None,
    highlight_color: str = "#ff5733",
    default_color: str = "#6175c1",
) -> "Network":
    """
    Builds a Pyvis Network object from the RAPTOR tree.
    """
    if Network is None:
        raise ImportError("`pyvis` not installed. Please install using `pip install pyvis`")

    if highlight_node_indices is None:
        highlight_node_indices = []

    net = Network(height="750px", width="100%", directed=True, notebook=False)
    net.barnes_hut()  # enable physics for better spacing

    visited: set[int] = set()

    def dfs(node: Node):
        """Depth‑first addition of nodes before edges."""
        if node.index not in visited:
            visited.add(node.index)
            label = (node.text[:80] + "...") if len(node.text) > 80 else node.text
            color = highlight_color if node.index in highlight_node_indices else default_color
            net.add_node(node.index, label=label, color=color)

        for c_idx in node.children:
            child = find_node_in_tree(c_idx, tree)
            dfs(child)  # ensure child node exists
            net.add_edge(node.index, child.index)  # safe to add edge now

    dfs(start_node)
    return net


def visualize_tree_structure(
    start_node: Node,
    tree: Tree,
    highlight_node_indices: list[int] | None = None,
    highlight_color: str = "#ff5733",
):
    """
    Visualizes the tree structure; optionally highlights specific nodes.
    """
    graph = Graph()
    build_graph_from_tree(graph, start_node, tree)

    # Generate layout using Reingold-Tilford (hierarchical tree layout)
    layout = graph.layout_reingold_tilford(root=[0])
    positions = {i: layout[i] for i in range(graph.vcount())}

    # Flip y-coordinates to align with traditional tree layout
    heights = [layout[i][1] for i in range(graph.vcount())]
    max_height = max(heights)
    node_x_coords = [positions[i][0] for i in range(len(positions))]
    node_y_coords = [2 * max_height - positions[i][1] for i in range(len(positions))]

    # Extract edge coordinates
    edge_x_coords = []
    edge_y_coords = []
    for edge in graph.es:
        edge_x_coords.extend([positions[edge.source][0], positions[edge.target][0], None])
        edge_y_coords.extend(
            [
                2 * max_height - positions[edge.source][1],
                2 * max_height - positions[edge.target][1],
                None,
            ]
        )

    node_labels = [vertex["name"] for vertex in graph.vs]

    # Determine per-node colors
    if highlight_node_indices is None:
        highlight_node_indices = []
    node_colors = [
        highlight_color if v["index"] in highlight_node_indices else "#6175c1"
        for v in graph.vs
    ]

    fig = create_visualization_figure(
        edge_x_coords,
        edge_y_coords,
        node_x_coords,
        node_y_coords,
        node_labels,
        node_colors,
    )

    fig.update_layout(
        title="Tree Visualization",
        font_size=12,
        showlegend=False,
        xaxis=dict(
            showline=False, zeroline=False, showgrid=False, showticklabels=False
        ),
        yaxis=dict(
            showline=False, zeroline=False, showgrid=False, showticklabels=False
        ),
        margin=dict(l=40, r=40, b=85, t=100),
        hovermode="closest",
        plot_bgcolor="rgb(248,248,248)",
    )

    fig.show()


def visualize_tree_structure_pyvis(
    start_node: Node,
    tree: Tree,
    highlight_node_indices: list[int] | None = None,
    highlight_color: str = "#ff5733",
    output_file: str = "tree.html",
    open_in_browser: bool = True,
):
    """
    Visualize the tree using Pyvis (HTML/JS). Returns the generated Network object.

    Parameters
    ----------
    start_node : Node
        Root node for visualization
    tree : Tree
        RAPTOR Tree object
    highlight_node_indices : list[int] | None
        List of node indices to accentuate with a different color
    highlight_color : str
        Hex color for highlighted nodes
    output_file : str
        HTML file path to write the network
    open_in_browser : bool
        If True (default), automatically opens generated HTML in the default browser
    """
    net = build_pyvis_network(
        start_node,
        tree,
        highlight_node_indices=highlight_node_indices,
        highlight_color=highlight_color,
    )

    net.show(output_file)
    if open_in_browser:
        import webbrowser, os

        webbrowser.open("file://" + os.path.realpath(output_file))

    return net