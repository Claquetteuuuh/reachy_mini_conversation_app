# add_person_to_db.py
"""Script pour ajouter des personnes à la base de données de reconnaissance faciale."""

import argparse
import base64
from pathlib import Path
from face_database_optimized import OptimizedFaceDatabase

def image_to_base64(image_path: str) -> str:
    """Convertit une image en base64."""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

def main():
    parser = argparse.ArgumentParser(description='Add person to face database')
    parser.add_argument('name', help='Person name')
    parser.add_argument('images', nargs='+', help='Path to image files')
    parser.add_argument('--role', help='Person role (optional)')
    parser.add_argument('--db-path', default='face_database.json', help='Database path')
    parser.add_argument('--embeddings-path', default='face_embeddings.pkl', help='Embeddings path')
    
    args = parser.parse_args()
    
    db = OptimizedFaceDatabase(db_path=args.db_path, embeddings_path=args.embeddings_path)
    
    # Convertir les images en base64
    base64_images = []
    for img_path in args.images:
        if Path(img_path).exists():
            b64 = image_to_base64(img_path)
            base64_images.append(b64)
            print(f"✓ Loaded {img_path}")
        else:
            print(f"✗ File not found: {img_path}")
    
    if not base64_images:
        print("No valid images found!")
        return
    
    # Ajouter à la base
    metadata = {"role": args.role} if args.role else {}
    success = db.add_person(args.name, base64_images, metadata)
    
    if success:
        print(f"\n✓ Added {args.name} with {len(base64_images)} images")
        stats = db.get_stats()
        print(f"Total persons in database: {stats['total_persons']}")
        print(f"Total embeddings: {stats['total_embeddings']}")
    else:
        print(f"\n✗ Failed to add {args.name}")

if __name__ == "__main__":
    main()