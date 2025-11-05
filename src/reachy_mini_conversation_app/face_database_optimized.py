# face_database_optimized.py
"""Optimized face database with pre-computed embeddings and FAISS search."""

from __future__ import annotations
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from deepface import DeepFace
    import faiss
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False
    logger.warning("DeepFace or FAISS not available for optimized face recognition")


class OptimizedFaceDatabase:
    """Base de données de visages avec embeddings pré-calculés et recherche FAISS."""
    
    def __init__(
        self, 
        db_path: str = "face_database.json",
        embeddings_path: str = "face_embeddings.pkl",
        model_name: str = "VGG-Face"
    ):
        self.db_path = Path(db_path)
        self.embeddings_path = Path(embeddings_path)
        self.model_name = model_name
        
        # Données
        self.db_data: Dict[str, Dict] = {}
        self.embeddings: List[np.ndarray] = []
        self.embedding_labels: List[Tuple[str, int]] = []  # (nom, index_image)
        self.index: Optional[faiss.Index] = None
        self.embedding_dim: Optional[int] = None
        
        self.load()
    
    def load(self):
        """Charge la base de données et les embeddings."""
        # Charger les métadonnées
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                self.db_data = json.load(f)
        
        # Charger les embeddings pré-calculés
        if self.embeddings_path.exists():
            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)
                self.embeddings = data['embeddings']
                self.embedding_labels = data['labels']
                self.embedding_dim = data['dim']
            
            # Reconstruire l'index FAISS
            self._build_faiss_index()
            logger.info("Loaded %d embeddings from cache", len(self.embeddings))
    
    def _build_faiss_index(self):
        """Construit l'index FAISS pour la recherche rapide."""
        if not self.embeddings:
            return
        
        # Convertir en numpy array
        embeddings_array = np.array(self.embeddings).astype('float32')
        
        # Normaliser pour utiliser la similarité cosine
        faiss.normalize_L2(embeddings_array)
        
        # Créer un index de recherche (IndexFlatIP pour cosine similarity)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_array)
        
        logger.info("Built FAISS index with %d vectors", len(self.embeddings))
    
    def _get_embedding(self, base64_image: str) -> Optional[np.ndarray]:
        """Calcule l'embedding d'une image."""
        if not OPTIMIZED_AVAILABLE:
            return None
        
        try:
            # Sauvegarder temporairement l'image
            from face_recognition import decode_base64_image, save_temp_image
            import os
            
            pil_image = decode_base64_image(base64_image)
            temp_file = save_temp_image(pil_image)
            
            try:
                # Extraire l'embedding avec DeepFace
                embedding_objs = DeepFace.represent(
                    img_path=temp_file,
                    model_name=self.model_name,
                    enforce_detection=True
                )
                
                # DeepFace.represent retourne une liste de dicts
                if embedding_objs and len(embedding_objs) > 0:
                    embedding = np.array(embedding_objs[0]['embedding'])
                    return embedding
                else:
                    return None
                    
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    
        except Exception as e:
            logger.error("Error computing embedding: %s", e)
            return None
    
    def add_person(
        self, 
        name: str, 
        base64_images: List[str], 
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Ajoute une personne avec pré-calcul des embeddings.
        
        Returns:
            True si succès, False si erreur
        """
        if not OPTIMIZED_AVAILABLE:
            logger.error("Cannot add person: DeepFace/FAISS not available")
            return False
        
        # Calculer les embeddings pour chaque image
        new_embeddings = []
        valid_images = []
        
        logger.info("Computing embeddings for %s...", name)
        
        for i, img_b64 in enumerate(base64_images):
            embedding = self._get_embedding(img_b64)
            if embedding is not None:
                new_embeddings.append(embedding)
                valid_images.append(img_b64)
                logger.info("  Image %d/%d: ✓", i+1, len(base64_images))
            else:
                logger.warning("  Image %d/%d: ✗ (no face detected)", i+1, len(base64_images))
        
        if not new_embeddings:
            logger.error("No valid embeddings computed for %s", name)
            return False
        
        # Stocker les métadonnées
        self.db_data[name] = {
            "images": valid_images,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat(),
            "num_embeddings": len(new_embeddings)
        }
        
        # Ajouter les embeddings
        for i, emb in enumerate(new_embeddings):
            self.embeddings.append(emb)
            self.embedding_labels.append((name, i))
        
        # Déterminer la dimension si c'est le premier embedding
        if self.embedding_dim is None:
            self.embedding_dim = len(new_embeddings[0])
        
        # Reconstruire l'index FAISS
        self._build_faiss_index()
        
        # Sauvegarder
        self._save_all()
        
        logger.info("Added %s with %d embeddings", name, len(new_embeddings))
        return True
    
    def _save_all(self):
        """Sauvegarde la base de données et les embeddings."""
        # Sauvegarder les métadonnées
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.db_data, f, indent=2, ensure_ascii=False)
        
        # Sauvegarder les embeddings
        with open(self.embeddings_path, 'wb') as f:
            pickle.dump({
                'embeddings': self.embeddings,
                'labels': self.embedding_labels,
                'dim': self.embedding_dim
            }, f)
    
    def identify(
        self, 
        base64_image: str, 
        confidence_threshold: float = 0.7,
        top_k: int = 5
    ) -> Dict:
        """
        Identifie une personne en comparant avec la base (ULTRA RAPIDE).
        
        Args:
            base64_image: Image à identifier
            confidence_threshold: Seuil de confiance (0-1)
            top_k: Nombre de meilleurs matchs à récupérer
        
        Returns:
            Résultat de l'identification
        """
        if not OPTIMIZED_AVAILABLE or self.index is None:
            return {
                "success": False,
                "error": "Database not initialized or empty"
            }
        
        # Calculer l'embedding de l'image requête
        query_embedding = self._get_embedding(base64_image)
        if query_embedding is None:
            return {
                "success": False,
                "error": "No face detected in query image"
            }
        
        # Normaliser l'embedding
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Recherche dans l'index FAISS (ULTRA RAPIDE!)
        similarities, indices = self.index.search(query_embedding, top_k)
        
        # Trouver le meilleur match par personne
        person_best_scores = {}
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < len(self.embedding_labels):
                person_name, _ = self.embedding_labels[idx]
                
                # Garder le meilleur score pour chaque personne
                if person_name not in person_best_scores or sim > person_best_scores[person_name]:
                    person_best_scores[person_name] = float(sim)
        
        # Trouver le meilleur match global
        if person_best_scores:
            best_person = max(person_best_scores, key=person_best_scores.get)
            best_score = person_best_scores[best_person]
            
            # La similarité cosine de FAISS est entre -1 et 1
            # On convertit en pourcentage 0-100
            similarity_percentage = (best_score + 1) / 2 * 100
            
            if best_score >= confidence_threshold:
                return {
                    "success": True,
                    "identified": True,
                    "name": best_person,
                    "similarity": round(similarity_percentage, 2),
                    "confidence": "High" if similarity_percentage > 85 else "Medium",
                    "all_matches": {
                        name: round((score + 1) / 2 * 100, 2) 
                        for name, score in sorted(
                            person_best_scores.items(), 
                            key=lambda x: x[1], 
                            reverse=True
                        )[:3]
                    }
                }
        
        return {
            "success": True,
            "identified": False,
            "name": "Unknown",
            "best_match_similarity": round(similarity_percentage, 2) if person_best_scores else 0,
            "message": f"No person matched with confidence above {confidence_threshold*100}%"
        }
    
    def get_all_persons(self) -> List[str]:
        """Retourne la liste de toutes les personnes."""
        return list(self.db_data.keys())
    
    def remove_person(self, name: str) -> bool:
        """Supprime une personne (nécessite de reconstruire l'index)."""
        if name not in self.db_data:
            return False
        
        # Supprimer des métadonnées
        del self.db_data[name]
        
        # Supprimer des embeddings
        self.embeddings = [
            emb for emb, (label, _) in zip(self.embeddings, self.embedding_labels)
            if label != name
        ]
        self.embedding_labels = [
            label for label in self.embedding_labels if label[0] != name
        ]
        
        # Reconstruire l'index
        self._build_faiss_index()
        self._save_all()
        
        logger.info("Removed %s from database", name)
        return True
    
    def get_stats(self) -> Dict:
        """Retourne des statistiques sur la base."""
        return {
            "total_persons": len(self.db_data),
            "total_embeddings": len(self.embeddings),
            "embedding_dimension": self.embedding_dim,
            "model_used": self.model_name,
            "persons": {
                name: data["num_embeddings"] 
                for name, data in self.db_data.items()
            }
        }