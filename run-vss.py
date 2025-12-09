#!/usr/bin/env python
# run-vss.py
# Simple Vector Similarity Search (VSS) example using ViT embeddings
#
# Usage:
#   python run-vss.py --exemplar-dir exemplars --query-image path/to/query.png
#   python run-vss.py --exemplar-dir exemplars --query-image path/to/query.png --top-k 10
#   python run-vss.py --exemplar-dir exemplars --query-image path/to/query.png --device cuda:0

import argparse
import sys
from pathlib import Path

import numpy as np

# Add the project root to the path so we can import sdcat modules
sys.path.insert(0, str(Path(__file__).parent))

from sdcat.cluster.embedding import ViTWrapper


def load_exemplar_images(exemplar_dir: Path, extensions: tuple = (".png", ".jpg", ".jpeg")) -> list[Path]:
    """
    Load all image files from the exemplar directory.
    
    Args:
        exemplar_dir: Path to directory containing exemplar images
        extensions: Tuple of valid image extensions
        
    Returns:
        List of image file paths
    """
    images = []
    for ext in extensions:
        images.extend(exemplar_dir.glob(f"*{ext}"))
        images.extend(exemplar_dir.glob(f"*{ext.upper()}"))
    
    # Filter out any embedding-related files (those with model names in them)
    images = [p for p in images if "_pred.txt" not in p.name and ".npy" not in p.name]
    
    return sorted(images)


def compute_embeddings(vit: ViTWrapper, image_paths: list[Path], batch_size: int = 32) -> np.ndarray:
    """
    Compute embeddings for a list of images using ViTWrapper.
    
    Args:
        vit: ViTWrapper instance
        image_paths: List of image file paths
        batch_size: Number of images to process in each batch
        
    Returns:
        numpy array of shape (num_images, embedding_dim)
    """
    all_embeddings = []
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = [str(p) for p in image_paths[i:i + batch_size]]
        batch_embeddings, _, _ = vit.process_images(batch_paths)
        all_embeddings.append(batch_embeddings)
    
    return np.vstack(all_embeddings)


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize embeddings for cosine similarity via dot product.
    
    Args:
        embeddings: numpy array of shape (num_images, embedding_dim)
        
    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def find_similar(query_embedding: np.ndarray, exemplar_embeddings: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the top-k most similar exemplars to the query.
    
    Args:
        query_embedding: Normalized query embedding (1, embedding_dim)
        exemplar_embeddings: Normalized exemplar embeddings (num_exemplars, embedding_dim)
        top_k: Number of similar images to return
        
    Returns:
        Tuple of (indices, similarity_scores) for top-k matches
    """
    # Cosine similarity via dot product (since embeddings are normalized)
    similarities = np.dot(exemplar_embeddings, query_embedding.T).flatten()
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    top_scores = similarities[top_indices]
    
    return top_indices, top_scores


def main():
    parser = argparse.ArgumentParser(
        description="Vector Similarity Search (VSS) example using ViT embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run-vss.py --exemplar-dir exemplars --query-image copepod.png
  python run-vss.py --exemplar-dir exemplars --query-image query.png --top-k 10
  python run-vss.py --exemplar-dir exemplars --query-image query.png --device cuda:0
        """
    )
    parser.add_argument(
        "--exemplar-dir",
        type=Path,
        required=True,
        help="Directory containing exemplar images"
    )
    parser.add_argument(
        "--query-image",
        type=Path,
        required=True,
        help="Path to query image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/Volumes/DeepSea-AI/models/CFE/cfe_isiis_final-20250509",
        help="ViT model name (default: /Volumes/DeepSea-AI/models/CFE/cfe_isiis_final-20250509)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar images to return (default: 5)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding computation (default: 32)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.exemplar_dir.exists():
        print(f"Error: Exemplar directory not found: {args.exemplar_dir}")
        sys.exit(1)
    
    if not args.query_image.exists():
        print(f"Error: Query image not found: {args.query_image}")
        sys.exit(1)
    
    # Load exemplar images
    print(f"Loading exemplar images from: {args.exemplar_dir}")
    exemplar_paths = load_exemplar_images(args.exemplar_dir)
    
    if len(exemplar_paths) == 0:
        print("Error: No images found in exemplar directory")
        sys.exit(1)
    
    print(f"Found {len(exemplar_paths)} exemplar images")
    
    # Initialize ViT model
    print(f"Initializing ViT model: {args.model} on {args.device}")
    vit = ViTWrapper(device=args.device, model_name=args.model, batch_size=args.batch_size)
    print(f"Embedding dimension: {vit.vector_dimensions}")
    
    # Compute embeddings for exemplars
    print(f"Computing embeddings for {len(exemplar_paths)} exemplar images...")
    exemplar_embeddings = compute_embeddings(vit, exemplar_paths, args.batch_size)
    print(f"Exemplar embeddings shape: {exemplar_embeddings.shape}")
    
    # Normalize embeddings for cosine similarity
    exemplar_embeddings_norm = normalize_embeddings(exemplar_embeddings)
    
    # Compute query embedding
    print(f"Computing embedding for query image: {args.query_image}")
    query_embedding, query_labels, query_scores = vit.process_images([str(args.query_image)])
    query_embedding_norm = normalize_embeddings(query_embedding)
    
    # Find similar images
    print(f"\nFinding top-{args.top_k} similar images...")
    top_indices, top_scores = find_similar(query_embedding_norm, exemplar_embeddings_norm, args.top_k)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Query Image: {args.query_image}")
    if query_labels:
        print(f"Query Prediction: {query_labels[0]} (score: {query_scores[0]})")
    print("=" * 60)
    print(f"\nTop-{args.top_k} Similar Exemplars:")
    print("-" * 60)
    
    for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
        exemplar_path = exemplar_paths[idx]
        print(f"{rank}. {exemplar_path.name}")
        print(f"   Path: {exemplar_path}")
        print(f"   Cosine Similarity: {score:.4f}")
        print()
    
    print("=" * 60)
    print("Done!")

    # Save query image and similar exemplars to a CSV file
    print(f"Saving results to run_vss.csv...")
    with open("run_vss.csv", "w") as f:
        f.write(f"query_image,{args.query_image}\n")
        for rank, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
            exemplar_path = exemplar_paths[idx]
            f.write(f"{rank},{exemplar_path}\n")


if __name__ == "__main__":
    main()

