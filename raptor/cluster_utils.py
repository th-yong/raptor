import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import tiktoken
import umap
from sklearn.mixture import GaussianMixture

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from .tree_structures import Node

# Import necessary methods from other modules
from .utils import get_embeddings

# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        # Set n_neighbors dynamically, ensuring it's at least 2 and at most 15 or len(embeddings)-1
        n_neighbors = min(15, max(2, len(embeddings) - 1))
    try:
        reduced_embeddings = umap.UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)
    except Exception as e:
        raise RuntimeError(
            f"UMAP dimensionality reduction failed with n_neighbors={n_neighbors}, "
            f"n_components={dim}, metric={metric}, number of embeddings={len(embeddings)}. "
            f"Error: {e}"
        )
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state, reg_covar=1e-3)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(
        n_components=n_clusters, random_state=random_state, reg_covar=1e-3
    )
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray, dim: int, threshold: float, verbose: bool = False
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(
        embeddings, min(dim, len(embeddings) - 2)
    )
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    if verbose:
        logging.info(f"Total Clusters: {total_clusters}")
    return all_local_clusters


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


class RAPTOR_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 18000,  # 기존 3500
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 7,  # 기존 10
        threshold: float = 0.1,
        verbose: bool = False,
    ) -> List[List[Node]]:
        # Get the embeddings from the nodes
        embeddings = np.array([node.embeddings[embedding_model_name] for node in nodes])

        # Perform the clustering
        clusters = perform_clustering(
            embeddings, dim=reduction_dimension, threshold=threshold
        )

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate the total length of the text in the nodes
            total_length = sum(
                [len(tokenizer.encode(node.text)) for node in cluster_nodes]
            )

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                node_clusters.extend(
                    RAPTOR_Clustering.perform_clustering(
                        cluster_nodes, embedding_model_name, max_length_in_cluster
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters


class PageAwareRAPTORClustering(ClusteringAlgorithm):
    """
    페이지 번호를 고려한 RAPTOR 클러스터링
    약관 문서에서 페이지 간격이 큰 내용들은 클러스터링하지 않음
    """

    @staticmethod
    def perform_clustering(
        nodes: List[Node],
        embedding_model_name: str,
        max_length_in_cluster: int = 25000,  # o3 최적화: 대폭 확장
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 7,
        threshold: float = 0.1,
        max_page_gap: int = 10,  # 최대 페이지 간격 (기본: 10페이지)
        verbose: bool = False,
    ) -> List[List[Node]]:
        """
        페이지 제약을 고려한 클러스터링

        Args:
            max_page_gap: 클러스터링을 허용할 최대 페이지 간격
        """

        # 먼저 페이지 번호별로 노드들을 그룹핑
        page_groups = {}
        nodes_without_page = []

        for node in nodes:
            if hasattr(node, "page_number") and node.page_number is not None:
                page_num = node.page_number
                if page_num not in page_groups:
                    page_groups[page_num] = []
                page_groups[page_num].append(node)
            else:
                nodes_without_page.append(node)

        if verbose:
            logging.info(
                f"페이지별 그룹: {len(page_groups)}개, 페이지 정보 없는 노드: {len(nodes_without_page)}개"
            )

        # 페이지 번호가 없는 노드들은 기존 방식으로 클러스터링
        final_clusters = []
        if nodes_without_page:
            if verbose:
                logging.info(
                    f"페이지 정보 없는 {len(nodes_without_page)}개 노드를 기존 방식으로 클러스터링"
                )
            final_clusters.extend(
                RAPTOR_Clustering.perform_clustering(
                    nodes_without_page,
                    embedding_model_name,
                    max_length_in_cluster,
                    tokenizer,
                    reduction_dimension,
                    threshold,
                    verbose,
                )
            )

        # 페이지별로 정렬된 순서로 처리
        sorted_pages = sorted(page_groups.keys())

        # 연속된 페이지들을 묶어서 클러스터링 범위 결정
        page_ranges = []
        current_range = [sorted_pages[0]] if sorted_pages else []

        for i in range(1, len(sorted_pages)):
            current_page = sorted_pages[i]
            last_page = current_range[-1]

            # 페이지 간격이 max_page_gap 이하면 같은 범위로 묶음
            if current_page - last_page <= max_page_gap:
                current_range.append(current_page)
            else:
                # 새로운 범위 시작
                page_ranges.append(current_range)
                current_range = [current_page]

        if current_range:
            page_ranges.append(current_range)

        if verbose:
            logging.info(f"페이지 범위 {len(page_ranges)}개: {page_ranges}")

        # 각 페이지 범위 내에서 클러스터링 수행
        for page_range in page_ranges:
            range_nodes = []
            for page_num in page_range:
                range_nodes.extend(page_groups[page_num])

            if verbose:
                logging.info(
                    f"페이지 {page_range[0]}-{page_range[-1]} 범위: {len(range_nodes)}개 노드 클러스터링"
                )

            # 해당 범위의 노드들을 클러스터링
            range_clusters = RAPTOR_Clustering.perform_clustering(
                range_nodes,
                embedding_model_name,
                max_length_in_cluster,
                tokenizer,
                reduction_dimension,
                threshold,
                verbose,
            )

            final_clusters.extend(range_clusters)

        return final_clusters
