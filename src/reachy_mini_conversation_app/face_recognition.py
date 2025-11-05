# face_recognition.py
"""Module for face recognition and comparison using DeepFace."""

from __future__ import annotations
import base64
import logging
import os
import tempfile
from io import BytesIO
from typing import Any, Dict

from PIL import Image
import numpy as np


logger = logging.getLogger(__name__)


def decode_base64_image(base64_string: str) -> Image.Image:
    """
    Décode une image base64 en image PIL.
    
    Args:
        base64_string: Image encodée en base64 (avec ou sans préfixe data:image)
    
    Returns:
        Image PIL
    """
    # Supprimer le préfixe data:image si présent
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    
    # Décoder le base64
    image_data = base64.b64decode(base64_string)
    
    # Convertir en image PIL
    image = Image.open(BytesIO(image_data))
    
    # Convertir en RGB si nécessaire
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image


def save_temp_image(pil_image: Image.Image) -> str:
    """
    Sauvegarde temporairement une image PIL.
    
    Args:
        pil_image: Image PIL
    
    Returns:
        Chemin du fichier temporaire
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    pil_image.save(temp_file.name, 'JPEG')
    temp_file.close()
    return temp_file.name


def compare_faces(
    base64_image1: str,
    base64_image2: str,
    model_name: str = 'VGG-Face',
    distance_metric: str = 'cosine'
) -> Dict[str, Any]:
    """
    Compare deux visages à partir d'images encodées en base64.
    
    Args:
        base64_image1: Première image en base64
        base64_image2: Deuxième image en base64
        model_name: Modèle à utiliser ('VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace')
        distance_metric: Métrique de distance ('cosine', 'euclidean', 'euclidean_l2')
    
    Returns:
        dict avec les résultats de la comparaison
    """
    try:
        from deepface import DeepFace
    except ImportError as e:
        logger.error("DeepFace not available: %s", e)
        return {
            "success": False,
            "error": "DeepFace library not installed. Install with: pip install deepface",
            "same_person": None
        }
    
    temp_file1 = None
    temp_file2 = None
    
    try:
        # Décoder les images
        logger.info("Décodage des images...")
        image1 = decode_base64_image(base64_image1)
        image2 = decode_base64_image(base64_image2)
        
        # Sauvegarder temporairement les images
        logger.info("Sauvegarde temporaire des images...")
        temp_file1 = save_temp_image(image1)
        temp_file2 = save_temp_image(image2)
        
        # Comparer les visages avec DeepFace
        logger.info("Comparaison des visages avec le modèle %s...", model_name)
        result = DeepFace.verify(
            img1_path=temp_file1,
            img2_path=temp_file2,
            model_name=model_name,
            distance_metric=distance_metric,
            enforce_detection=True
        )
        
        # Calculer un pourcentage de similitude
        distance = result['distance']
        threshold = result['threshold']
        
        # Pour cosine: 0 = identique, 1 = différent
        # Inverser pour avoir un pourcentage de similitude
        if distance_metric == 'cosine':
            similarity_percentage = max(0, (1 - distance) * 100)
        else:
            # Pour euclidean: plus petit = plus similaire
            similarity_percentage = max(0, (1 - (distance / threshold)) * 100)
        
        return {
            "success": True,
            "same_person": result['verified'],
            "distance": distance,
            "threshold": threshold,
            "similarity_percentage": similarity_percentage,
            "model_used": model_name,
            "distance_metric": distance_metric,
            "confidence": "Haute" if abs(distance - threshold) > 0.1 else "Moyenne"
        }
        
    except ValueError as e:
        # Erreur de détection de visage
        error_msg = str(e)
        if "Face could not be detected" in error_msg:
            logger.warning("Face detection failed: %s", error_msg)
            return {
                "success": False,
                "error": "Aucun visage détecté dans une ou les deux images",
                "same_person": None
            }
        else:
            logger.error("Detection error: %s", error_msg)
            return {
                "success": False,
                "error": f"Erreur de détection: {error_msg}",
                "same_person": None
            }
    
    except Exception as e:
        logger.exception("Unexpected error in face comparison")
        return {
            "success": False,
            "error": str(e),
            "same_person": None
        }
    
    finally:
        # Nettoyer les fichiers temporaires
        if temp_file1 and os.path.exists(temp_file1):
            try:
                os.unlink(temp_file1)
            except Exception:
                pass
        if temp_file2 and os.path.exists(temp_file2):
            try:
                os.unlink(temp_file2)
            except Exception:
                pass


def compare_faces_simple(base64_image1: str, base64_image2: str) -> Dict[str, Any]:
    """
    Version simplifiée avec les paramètres par défaut recommandés.
    
    Args:
        base64_image1: Première image en base64
        base64_image2: Deuxième image en base64
    
    Returns:
        dict avec les résultats de la comparaison
    """
    return compare_faces(base64_image1, base64_image2, model_name='VGG-Face', distance_metric='cosine')