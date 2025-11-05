# tools.py
from __future__ import annotations
import abc
import json
import asyncio
import inspect
import logging
from typing import Any, Dict, List, Tuple, Literal
from dataclasses import dataclass
import aiohttp  # Pour les requêtes HTTP async

from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose


logger = logging.getLogger(__name__)

# Initialize dance and emotion libraries
try:
    from reachy_mini.motion.recorded_move import RecordedMoves
    from reachy_mini_dances_library.collection.dance import AVAILABLE_MOVES
    from reachy_mini_conversation_app.dance_emotion_moves import (
        GotoQueueMove,
        DanceQueueMove,
        EmotionQueueMove,
    )

    # Initialize recorded moves for emotions
    # Note: huggingface_hub automatically reads HF_TOKEN from environment variables
    RECORDED_MOVES = RecordedMoves("pollen-robotics/reachy-mini-emotions-library")
    DANCE_AVAILABLE = True
    EMOTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dance/emotion libraries not available: {e}")
    AVAILABLE_MOVES = {}
    RECORDED_MOVES = None
    DANCE_AVAILABLE = False
    EMOTION_AVAILABLE = False

try:
    from reachy_mini_conversation_app import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Face recognition not available: {e}")
    FACE_RECOGNITION_AVAILABLE = False

try:
    from reachy_mini_conversation_app.face_database_optimized import OptimizedFaceDatabase
    from reachy_mini_conversation_app.camera_worker import get_current_camera_frame_base64
    FACE_DB_OPTIMIZED_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Optimized face database not available: {e}")
    FACE_DB_OPTIMIZED_AVAILABLE = False

_optimized_face_db: "OptimizedFaceDatabase | None" = None

def initialize_optimized_face_database(
    db_path: str = "face_database.json",
    embeddings_path: str = "face_embeddings.pkl"
) -> "OptimizedFaceDatabase | None":
    """Initialise la base de données optimisée de reconnaissance faciale.
    
    Args:
        db_path: Chemin vers le fichier JSON de métadonnées
        embeddings_path: Chemin vers le fichier pickle des embeddings
    
    Returns:
        Instance de OptimizedFaceDatabase ou None si non disponible
    """
    global _optimized_face_db
    if FACE_DB_OPTIMIZED_AVAILABLE:
        try:
            _optimized_face_db = OptimizedFaceDatabase(db_path, embeddings_path)
            logger.info("✓ Optimized face database initialized with %d persons", 
                       len(_optimized_face_db.get_all_persons()))
            return _optimized_face_db
        except Exception as e:
            logger.error("Failed to initialize face database: %s", e)
            return None
    else:
        logger.warning("Face database not available - missing dependencies")
        return None

def get_concrete_subclasses(base: type[Tool]) -> List[type[Tool]]:
    """Recursively find all concrete (non-abstract) subclasses of a base class."""
    result: List[type[Tool]] = []
    for cls in base.__subclasses__():
        if not inspect.isabstract(cls):
            result.append(cls)
        # recurse into subclasses
        result.extend(get_concrete_subclasses(cls))
    return result


# Types & state
Direction = Literal["left", "right", "up", "down", "front"]


@dataclass
class ToolDependencies:
    """External dependencies injected into tools."""

    reachy_mini: ReachyMini
    movement_manager: Any  # MovementManager from moves.py
    # Optional deps
    camera_worker: Any | None = None  # CameraWorker for frame buffering
    vision_manager: Any | None = None
    head_wobbler: Any | None = None  # HeadWobbler for audio-reactive motion
    motion_duration_s: float = 1.0


# Tool base class
class Tool(abc.ABC):
    """Base abstraction for tools used in function-calling.

    Each tool must define:
      - name: str
      - description: str
      - parameters_schema: Dict[str, Any]  # JSON Schema
    """

    name: str
    description: str
    parameters_schema: Dict[str, Any]

    def spec(self) -> Dict[str, Any]:
        """Return the function spec for LLM consumption."""
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
        }

    @abc.abstractmethod
    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Async tool execution entrypoint."""
        raise NotImplementedError


# Concrete tools


class MoveHead(Tool):
    """Move head in a given direction."""

    name = "move_head"
    description = "Move your head in a given direction: left, right, up, down or front."
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "up", "down", "front"],
            },
        },
        "required": ["direction"],
    }

    # mapping: direction -> args for create_head_pose
    DELTAS: Dict[str, Tuple[int, int, int, int, int, int]] = {
        "left": (0, 0, 0, 0, 0, 40),
        "right": (0, 0, 0, 0, 0, -40),
        "up": (0, 0, 0, 0, -30, 0),
        "down": (0, 0, 0, 0, 30, 0),
        "front": (0, 0, 0, 0, 0, 0),
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Move head in a given direction."""
        direction_raw = kwargs.get("direction")
        if not isinstance(direction_raw, str):
            return {"error": "direction must be a string"}
        direction: Direction = direction_raw  # type: ignore[assignment]
        logger.info("Tool call: move_head direction=%s", direction)

        deltas = self.DELTAS.get(direction, self.DELTAS["front"])
        target = create_head_pose(*deltas, degrees=True)

        # Use new movement manager
        try:
            movement_manager = deps.movement_manager

            # Get current state for interpolation
            current_head_pose = deps.reachy_mini.get_current_head_pose()
            _, current_antennas = deps.reachy_mini.get_current_joint_positions()

            # Create goto move
            goto_move = GotoQueueMove(
                target_head_pose=target,
                start_head_pose=current_head_pose,
                target_antennas=(0, 0),  # Reset antennas to default
                start_antennas=(
                    current_antennas[0],
                    current_antennas[1],
                ),  # Skip body_yaw
                target_body_yaw=0,  # Reset body yaw
                start_body_yaw=current_antennas[0],  # body_yaw is first in joint positions
                duration=deps.motion_duration_s,
            )

            movement_manager.queue_move(goto_move)
            movement_manager.set_moving_state(deps.motion_duration_s)

            return {"status": f"looking {direction}"}

        except Exception as e:
            logger.error("move_head failed")
            return {"error": f"move_head failed: {type(e).__name__}: {e}"}


class Camera(Tool):
    """Take a picture with the camera and ask a question about it."""

    name = "camera"
    description = "Take a picture with the camera and ask a question about it."
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask about the picture",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and ask a question about it."""
        image_query = (kwargs.get("question") or "").strip()
        if not image_query:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", image_query[:120])

        # Get frame from camera worker buffer (like main_works.py)
        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.error("No frame available from camera worker")
                return {"error": "No frame available"}
        else:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        # Use vision manager for processing if available
        if deps.vision_manager is not None:
            vision_result = await asyncio.to_thread(
                deps.vision_manager.processor.process_image, frame, image_query,
            )
            if isinstance(vision_result, dict) and "error" in vision_result:
                return vision_result
            return (
                {"image_description": vision_result}
                if isinstance(vision_result, str)
                else {"error": "vision returned non-string"}
            )
        # Return base64 encoded image like main_works.py camera tool
        import base64

        import cv2
        import tempfile, os, base64

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        temp_path = os.path.join(tempfile.gettempdir(), "camera_frame.jpg")

        # Enregistre la frame temporairement
        cv2.imwrite(temp_path, frame_bgr)

        # Lit l'image en base64
        with open(temp_path, "rb") as f:
            b64_encoded = base64.b64encode(f.read()).decode("utf-8")

        # Supprime le fichier temporaire
        try:
            os.remove(temp_path)
        except OSError:
            pass  # si déjà supprimé ou inaccessible

        return {"b64_im": b64_encoded}



class HeadTracking(Tool):
    """Toggle head tracking state."""

    name = "head_tracking"
    description = "Toggle head tracking state."
    parameters_schema = {
        "type": "object",
        "properties": {"start": {"type": "boolean"}},
        "required": ["start"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Enable or disable head tracking."""
        enable = bool(kwargs.get("start"))

        # Update camera worker head tracking state
        if deps.camera_worker is not None:
            deps.camera_worker.set_head_tracking_enabled(enable)

        status = "started" if enable else "stopped"
        logger.info("Tool call: head_tracking %s", status)
        return {"status": f"head tracking {status}"}

class RotateBody(Tool):
    """Rotate the robot's body left or right."""

    name = "rotate_body"
    description = """Rotate the robot's body to the left or right. 
                     Use this when you want to turn around, look to the side, 
                     or change your body orientation."""
    
    parameters_schema = {
        "type": "object",
        "properties": {
            "direction": {
                "type": "string",
                "enum": ["left", "right", "center"],
                "description": "Direction to rotate: 'left' (turn left), 'right' (turn right), 'center' (face forward)",
            },
            "angle": {
                "type": "number",
                "description": "Rotation angle in degrees (0-180). Default is 45 degrees. Ignored if direction is 'center'.",
                "default": 45,
            },
            "duration": {
                "type": "number",
                "description": "Duration of the movement in seconds. Default is 1.0 second.",
                "default": 1.0,
            },
        },
        "required": ["direction"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Rotate the robot's body."""
        direction = kwargs.get("direction", "center")
        angle_deg = float(kwargs.get("angle", 45))
        duration = float(kwargs.get("duration", 1.0))

        logger.info(
            "Tool call: rotate_body direction=%s angle=%s duration=%s",
            direction,
            angle_deg,
            duration,
        )

        # Limit angle to reasonable values
        angle_deg = max(0, min(180, angle_deg))
        
        # Convert to radians
        import math
        angle_rad = math.radians(angle_deg)

        # Determine target yaw based on direction
        if direction == "left":
            target_yaw = angle_rad
        elif direction == "right":
            target_yaw = -angle_rad
        elif direction == "center":
            target_yaw = 0.0
        else:
            return {"error": f"Invalid direction: {direction}. Must be 'left', 'right', or 'center'."}

        try:
            # Use goto_target to smoothly rotate the body
            deps.reachy_mini.goto_target(
                head=None,  # Don't move the head
                body_yaw=target_yaw,
                duration=duration,
            )

            return {
                "success": True,
                "direction": direction,
                "angle_degrees": angle_deg if direction != "center" else 0,
                "angle_radians": target_yaw,
                "duration": duration,
            }

        except Exception as e:
            logger.exception("Failed to rotate body")
            return {"error": f"Failed to rotate body: {str(e)}"}

class Dance(Tool):
    """Play a named or random dance move once (or repeat). Non-blocking."""

    name = "dance"
    description = "Play a named or random dance move once (or repeat). Non-blocking."
    parameters_schema = {
        "type": "object",
        "properties": {
            "move": {
                "type": "string",
                "description": """Name of the move; use 'random' or omit for random.
                                    Here is a list of the available moves:
                                        simple_nod: A simple, continuous up-and-down nodding motion.
                                        head_tilt_roll: A continuous side-to-side head roll (ear to shoulder).
                                        side_to_side_sway: A smooth, side-to-side sway of the entire head.
                                        dizzy_spin: A circular 'dizzy' head motion combining roll and pitch.
                                        stumble_and_recover: A simulated stumble and recovery with multiple axis movements. Good vibes
                                        headbanger_combo: A strong head nod combined with a vertical bounce.
                                        interwoven_spirals: A complex spiral motion using three axes at different frequencies.
                                        sharp_side_tilt: A sharp, quick side-to-side tilt using a triangle waveform.
                                        side_peekaboo: A multi-stage peekaboo performance, hiding and peeking to each side.
                                        yeah_nod: An emphatic two-part yeah nod using transient motions.
                                        uh_huh_tilt: A combined roll-and-pitch uh-huh gesture of agreement.
                                        neck_recoil: A quick, transient backward recoil of the neck.
                                        chin_lead: A forward motion led by the chin, combining translation and pitch.
                                        groovy_sway_and_roll: A side-to-side sway combined with a corresponding roll for a groovy effect.
                                        chicken_peck: A sharp, forward, chicken-like pecking motion.
                                        side_glance_flick: A quick glance to the side that holds, then returns.
                                        polyrhythm_combo: A 3-beat sway and a 2-beat nod create a polyrhythmic feel.
                                        grid_snap: A robotic, grid-snapping motion using square waveforms.
                                        pendulum_swing: A simple, smooth pendulum-like swing using a roll motion.
                                        jackson_square: Traces a rectangle via a 5-point path, with sharp twitches on arrival at each checkpoint.
                """,
            },
            "repeat": {
                "type": "integer",
                "description": "How many times to repeat the move (default 1).",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a named or random dance move once (or repeat). Non-blocking."""
        if not DANCE_AVAILABLE:
            return {"error": "Dance system not available"}

        move_name = kwargs.get("move")
        repeat = int(kwargs.get("repeat", 1))

        logger.info("Tool call: dance move=%s repeat=%d", move_name, repeat)

        if not move_name or move_name == "random":
            import random

            move_name = random.choice(list(AVAILABLE_MOVES.keys()))

        if move_name not in AVAILABLE_MOVES:
            return {"error": f"Unknown dance move '{move_name}'. Available: {list(AVAILABLE_MOVES.keys())}"}

        # Add dance moves to queue
        movement_manager = deps.movement_manager
        for _ in range(repeat):
            dance_move = DanceQueueMove(move_name)
            movement_manager.queue_move(dance_move)

        return {"status": "queued", "move": move_name, "repeat": repeat}


class StopDance(Tool):
    """Stop the current dance move."""

    name = "stop_dance"
    description = "Stop the current dance move"
    parameters_schema = {
        "type": "object",
        "properties": {
            "dummy": {
                "type": "boolean",
                "description": "dummy boolean, set it to true",
            },
        },
        "required": ["dummy"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Stop the current dance move."""
        logger.info("Tool call: stop_dance")
        movement_manager = deps.movement_manager
        movement_manager.clear_move_queue()
        return {"status": "stopped dance and cleared queue"}


class GetEmails(Tool):
    """Retrieve recent emails from the user's Gmail inbox via N8N webhook."""

    name = "get_emails"
    description = """Retrieve recent emails from the user's Gmail inbox with optional filters.
                     Use this tool when the user asks about their emails, inbox, unread messages, 
                     or specific senders."""
    
    parameters_schema = {
        "type": "object",
        "properties": {
            "max_results": {
                "type": "integer",
                "description": "Maximum number of emails to retrieve (1-50). Default is 10.",
                "default": 10,
            },
            "label_filter": {
                "type": "string",
                "description": """Filter by Gmail label name. Examples: 
                                  'UNREAD' (unread emails), 
                                  'N8N_Urgent' (urgent emails), 
                                  'N8N_Important' (important emails),
                                  'INBOX' (inbox emails)""",
            },
            "sender_filter": {
                "type": "string",
                "description": "Filter by sender email address or name (e.g., 'rodrigue@vreality.fr', 'Digiforma')",
            },
            "subject_filter": {
                "type": "string",
                "description": "Search term to filter by email subject (e.g., 'formation', 'meeting')",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Retrieve emails from Gmail via N8N webhook."""
        max_results = int(kwargs.get("max_results", 10))
        label_filter = kwargs.get("label_filter")
        sender_filter = kwargs.get("sender_filter")
        subject_filter = kwargs.get("subject_filter")

        logger.info(
            "Tool call: get_emails max_results=%d label=%s sender=%s subject=%s",
            max_results,
            label_filter,
            sender_filter,
            subject_filter,
        )

        webhook_url ="https://n8n.srv856775.hstgr.cloud/webhook/1a2ec414-2c1a-47eb-a977-199cbcc4a2ac"

        try:
            # Appel asynchrone du webhook N8N
            async with aiohttp.ClientSession() as session:
                payload = {
                    "action": "get_emails",
                    "max_results": max_results,
                    "filters": {
                        "label": label_filter,
                        "sender": sender_filter,
                        "subject": subject_filter,
                    },
                }

                async with session.get(webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error("N8N webhook error: status=%d body=%s", response.status, error_text)
                        return {"error": f"Failed to fetch emails: HTTP {response.status}"}

                    data = await response.json()

            # Parse la réponse
            emails = data.get("data", [])
            
            if not emails:
                return {
                    "success": True,
                    "count": 0,
                    "message": "No emails found matching the filters",
                    "filters_applied": {
                        "label": label_filter,
                        "sender": sender_filter,
                        "subject": subject_filter,
                    },
                }

            # Applique les filtres côté client si nécessaire
            if label_filter:
                emails = [
                    email for email in emails
                    if email.get("labels") and any(
                        label_filter.lower() in label.get("name", "").lower()
                        for label in email["labels"]
                    )
                ]

            if sender_filter:
                emails = [
                    email for email in emails
                    if email.get("From") and sender_filter.lower() in email["From"].lower()
                ]

            if subject_filter:
                emails = [
                    email for email in emails
                    if email.get("Subject") and subject_filter.lower() in email["Subject"].lower()
                ]

            # Limite au nombre demandé
            emails = emails[:max_results]

            # Formate les résultats pour l'assistant
            formatted_emails = []
            for email in emails:
                # Parse la date
                internal_date = email.get("internalDate")
                if internal_date:
                    from datetime import datetime
                    date_str = datetime.fromtimestamp(int(internal_date) / 1000).strftime("%Y-%m-%d %H:%M")
                else:
                    date_str = "Unknown date"

                # Extrait les labels
                labels = [label.get("name") for label in email.get("labels", [])]
                is_unread = "UNREAD" in labels

                formatted_emails.append({
                    "id": email.get("id"),
                    "subject": email.get("Subject", "No subject"),
                    "from": email.get("From", "Unknown sender"),
                    "to": email.get("To", ""),
                    "snippet": email.get("snippet", "")[:200],  # Limite le snippet à 200 chars
                    "date": date_str,
                    "labels": labels,
                    "is_unread": is_unread,
                })

            logger.info("Successfully retrieved %d emails", len(formatted_emails))

            return {
                "success": True,
                "count": len(formatted_emails),
                "emails": formatted_emails,
                "filters_applied": {
                    "label": label_filter,
                    "sender": sender_filter,
                    "subject": subject_filter,
                },
            }

        except asyncio.TimeoutError:
            logger.error("N8N webhook timeout")
            return {"error": "Request timeout: N8N webhook took too long to respond"}
        
        except aiohttp.ClientError as e:
            logger.exception("HTTP error calling N8N webhook")
            return {"error": f"Network error: {str(e)}"}
        
        except Exception as e:
            logger.exception("Unexpected error in get_emails tool")
            return {"error": f"Unexpected error: {str(e)}"}

def get_available_emotions_and_descriptions() -> str:
    """Get formatted list of available emotions with descriptions."""
    if not EMOTION_AVAILABLE:
        return "Emotions not available"

    try:
        emotion_names = RECORDED_MOVES.list_moves()
        output = "Available emotions:\n"
        for name in emotion_names:
            description = RECORDED_MOVES.get(name).description
            output += f" - {name}: {description}\n"
        return output
    except Exception as e:
        return f"Error getting emotions: {e}"

class PlayEmotion(Tool):
    """Play a pre-recorded emotion."""

    name = "play_emotion"
    description = "Play a pre-recorded emotion"
    parameters_schema = {
        "type": "object",
        "properties": {
            "emotion": {
                "type": "string",
                "description": f"""Name of the emotion to play.
                                    Here is a list of the available emotions:
                                    {get_available_emotions_and_descriptions()}
                                    """,
            },
        },
        "required": ["emotion"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Play a pre-recorded emotion."""
        if not EMOTION_AVAILABLE:
            return {"error": "Emotion system not available"}

        emotion_name = kwargs.get("emotion")
        if not emotion_name:
            return {"error": "Emotion name is required"}

        logger.info("Tool call: play_emotion emotion=%s", emotion_name)

        # Check if emotion exists
        try:
            emotion_names = RECORDED_MOVES.list_moves()
            if emotion_name not in emotion_names:
                return {"error": f"Unknown emotion '{emotion_name}'. Available: {emotion_names}"}

            # Add emotion to queue
            movement_manager = deps.movement_manager
            emotion_move = EmotionQueueMove(emotion_name, RECORDED_MOVES)
            movement_manager.queue_move(emotion_move)

            return {"status": "queued", "emotion": emotion_name}

        except Exception as e:
            logger.exception("Failed to play emotion")
            return {"error": f"Failed to play emotion: {e!s}"}


class StopEmotion(Tool):
    """Stop the current emotion."""

    name = "stop_emotion"
    description = "Stop the current emotion"
    parameters_schema = {
        "type": "object",
        "properties": {
            "dummy": {
                "type": "boolean",
                "description": "dummy boolean, set it to true",
            },
        },
        "required": ["dummy"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Stop the current emotion."""
        logger.info("Tool call: stop_emotion")
        movement_manager = deps.movement_manager
        movement_manager.clear_move_queue()
        return {"status": "stopped emotion and cleared queue"}


class DoNothing(Tool):
    """Choose to do nothing - stay still and silent. Use when you want to be contemplative or just chill."""

    name = "do_nothing"
    description = "Choose to do nothing - stay still and silent. Use when you want to be contemplative or just chill."
    parameters_schema = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Optional reason for doing nothing (e.g., 'contemplating existence', 'saving energy', 'being mysterious')",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Do nothing - stay still and silent."""
        reason = kwargs.get("reason", "just chilling")
        logger.info("Tool call: do_nothing reason=%s", reason)
        return {"status": "doing nothing", "reason": reason}

class CompareFaces(Tool):
    """Compare two faces from base64 encoded images to determine if they are the same person."""

    name = "compare_faces"
    description = """Compare two faces from base64 encoded images to determine if they are the same person.
    This tool uses facial recognition to analyze and compare facial features."""
    parameters_schema = {
        "type": "object",
        "properties": {
            "image1_base64": {
                "type": "string",
                "description": "First image encoded in base64 format (with or without data:image prefix)",
            },
            "image2_base64": {
                "type": "string",
                "description": "Second image encoded in base64 format (with or without data:image prefix)",
            },
            "model": {
                "type": "string",
                "description": "Model to use for face recognition. Options: VGG-Face (default), Facenet, OpenFace, DeepFace, ArcFace",
                "enum": ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "ArcFace"],
            },
        },
        "required": ["image1_base64", "image2_base64"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Compare two faces from base64 images."""
        if not FACE_RECOGNITION_AVAILABLE:
            logger.error("Face recognition not available")
            return {"error": "Face recognition system not available. Install required packages: pip install deepface tensorflow pillow"}

        image1_b64 = kwargs.get("image1_base64", "")
        image2_b64 = kwargs.get("image2_base64", "")
        model = kwargs.get("model", "VGG-Face")

        if not image1_b64 or not image2_b64:
            logger.warning("compare_faces: missing image data")
            return {"error": "Both image1_base64 and image2_base64 are required"}

        logger.info("Tool call: compare_faces model=%s", model)

        # Run face comparison in a thread to avoid blocking
        result = await asyncio.to_thread(
            face_recognition.compare_faces,
            image1_b64,
            image2_b64,
            model_name=model,
            distance_metric='cosine'
        )

        # Format response for better readability
        if result.get("success"):
            return {
                "success": True,
                "same_person": result["same_person"],
                "similarity_percentage": round(result["similarity_percentage"], 2),
                "confidence": result["confidence"],
                "details": {
                    "distance": round(result["distance"], 4),
                    "threshold": round(result["threshold"], 4),
                    "model_used": result["model_used"],
                }
            }
        else:
            return result

class IdentifyPerson(Tool):
    """Identify the person currently in front of the camera using face recognition."""
    
    name = "identify_person"
    description = """Identify the person currently in front of the camera by comparing 
    with the local face database. Returns the person's name if recognized, or 'Unknown' if not.
    This tool automatically uses the current camera frame - no image data needs to be provided.
    IMPORTANT: This never sends any image data to OpenAI - everything stays local for privacy."""
    
    parameters_schema = {
        "type": "object",
        "properties": {
            "confidence_threshold": {
                "type": "number",
                "description": "Minimum similarity percentage to consider a match (0-100, default: 60)",
                "default": 60
            }
        },
        "required": []
    }
    
    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Identifie la personne devant la caméra."""
        global _optimized_face_db
        
        if not FACE_DB_OPTIMIZED_AVAILABLE or _optimized_face_db is None:
            return {
                "error": "Face recognition database not available or not initialized. "
                        "Please run initialize_optimized_face_database() first."
            }
        
        # Récupérer l'image actuelle de la caméra
        current_image = get_current_camera_frame_base64()
        if not current_image:
            return {
                "success": False,
                "error": "No camera frame available. Please ensure camera is active."
            }
        
        confidence_threshold = float(kwargs.get("confidence_threshold", 60)) / 100  # Convertir en 0-1
        
        logger.info("🔍 Identifying person with threshold=%.1f%%", confidence_threshold * 100)
        
        # Identification ULTRA RAPIDE (quelques millisecondes!)
        result = await asyncio.to_thread(
            _optimized_face_db.identify,
            current_image,
            confidence_threshold=confidence_threshold
        )
        
        if result.get("identified"):
            logger.info("✓ Person identified: %s (%.1f%% similarity)", 
                       result.get("name"), result.get("similarity", 0))
        else:
            logger.info("✗ No person identified (best match: %.1f%%)", 
                       result.get("best_match_similarity", 0))
        
        return result

class AddPersonToDatabase(Tool):
    """Add a new person to the face recognition database using current camera frame."""
    
    name = "add_person_to_database"
    description = """Add a new person to the face recognition database using the current camera frame.
    Use this tool when someone asks to be added, registered, or saved in your memory.
    Examples: "Can you remember me?", "Add me to your database", "Save my face", "Register me as John"
    
    IMPORTANT: This captures the current camera frame and stores it locally - no data is sent to OpenAI.
    Ask for the person's name before calling this tool."""
    
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The person's name (required). Ask the user for their name before calling this tool.",
            },
            "role": {
                "type": "string",
                "description": "Optional role/description (e.g., 'friend', 'colleague', 'visitor')",
            },
            "capture_multiple": {
                "type": "boolean",
                "description": "If true, will ask the user to move their head slightly for better recognition (default: false)",
                "default": False,
            },
        },
        "required": ["name"],
    }
    
    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Ajoute une nouvelle personne à la base de données."""
        global _optimized_face_db
        
        if not FACE_DB_OPTIMIZED_AVAILABLE or _optimized_face_db is None:
            return {
                "error": "Face recognition database not available or not initialized."
            }
        
        name = kwargs.get("name", "").strip()
        if not name:
            return {
                "success": False,
                "error": "Name is required. Please provide the person's name."
            }
        
        role = kwargs.get("role", "").strip()
        capture_multiple = kwargs.get("capture_multiple", False)
        
        # Vérifier si la personne existe déjà
        existing_persons = _optimized_face_db.get_all_persons()
        if name in existing_persons:
            return {
                "success": False,
                "error": f"Person '{name}' already exists in database. Use a different name or remove the existing entry first."
            }
        
        logger.info("📸 Adding new person to database: %s (role: %s)", name, role or "none")
        
        # Récupérer l'image actuelle de la caméra
        current_image = get_current_camera_frame_base64()
        if not current_image:
            return {
                "success": False,
                "error": "No camera frame available. Please ensure camera is active."
            }
        
        # Préparer les images (une seule pour l'instant)
        images_to_add = [current_image]
        
        # TODO: Si capture_multiple est True, on pourrait capturer plusieurs frames
        # avec un délai pour permettre à l'utilisateur de bouger légèrement
        if capture_multiple:
            logger.info("Multiple capture requested - waiting for additional frames...")
            # Attendre un peu et capturer d'autres frames
            await asyncio.sleep(0.5)
            frame2 = get_current_camera_frame_base64()
            if frame2 and frame2 != current_image:
                images_to_add.append(frame2)
            
            await asyncio.sleep(0.5)
            frame3 = get_current_camera_frame_base64()
            if frame3 and frame3 not in images_to_add:
                images_to_add.append(frame3)
        
        # Ajouter à la base de données
        metadata = {}
        if role:
            metadata["role"] = role
        
        try:
            success = await asyncio.to_thread(
                _optimized_face_db.add_person,
                name,
                images_to_add,
                metadata
            )
            
            if success:
                stats = _optimized_face_db.get_stats()
                logger.info("✓ Successfully added %s to database (%d images)", 
                           name, len(images_to_add))
                
                return {
                    "success": True,
                    "name": name,
                    "images_captured": len(images_to_add),
                    "role": role or "none",
                    "message": f"Successfully added {name} to the database with {len(images_to_add)} image(s)",
                    "total_persons_in_db": stats.get("total_persons", 0),
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to add person to database. No face detected in the captured image(s)."
                }
                
        except Exception as e:
            logger.error("Failed to add person to database: %s", e)
            return {
                "success": False,
                "error": f"Failed to add person: {str(e)}"
            }


class RemovePersonFromDatabase(Tool):
    """Remove a person from the face recognition database."""
    
    name = "remove_person_from_database"
    description = """Remove a person from the face recognition database.
    Use this when someone asks to be forgotten, removed, or deleted from your memory.
    Examples: "Forget me", "Remove me from your database", "Delete John from memory"
    
    WARNING: This permanently deletes the person's data."""
    
    parameters_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "The name of the person to remove (must match exactly)",
            },
            "confirm": {
                "type": "boolean",
                "description": "Confirmation flag - must be true to proceed with deletion",
                "default": False,
            },
        },
        "required": ["name", "confirm"],
    }
    
    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Supprime une personne de la base de données."""
        global _optimized_face_db
        
        if not FACE_DB_OPTIMIZED_AVAILABLE or _optimized_face_db is None:
            return {"error": "Face database not initialized"}
        
        name = kwargs.get("name", "").strip()
        confirm = kwargs.get("confirm", False)
        
        if not name:
            return {"success": False, "error": "Name is required"}
        
        if not confirm:
            return {
                "success": False,
                "error": "Deletion not confirmed. Set confirm=true to proceed."
            }
        
        logger.info("🗑️ Removing person from database: %s", name)
        
        try:
            success = await asyncio.to_thread(
                _optimized_face_db.remove_person,
                name
            )
            
            if success:
                stats = _optimized_face_db.get_stats()
                logger.info("✓ Successfully removed %s from database", name)
                return {
                    "success": True,
                    "name": name,
                    "message": f"Successfully removed {name} from the database",
                    "total_persons_remaining": stats.get("total_persons", 0),
                }
            else:
                return {
                    "success": False,
                    "error": f"Person '{name}' not found in database"
                }
                
        except Exception as e:
            logger.error("Failed to remove person: %s", e)
            return {"success": False, "error": f"Failed to remove person: {str(e)}"}


class ListKnownPersons(Tool):
    """List all persons registered in the face recognition database."""
    
    name = "list_known_persons"
    description = """List all persons registered in the face recognition database.
    Use this when asked 'who do you know?', 'who is in your database?', or 
    'who can you recognize?'"""
    
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Liste les personnes connues."""
        global _optimized_face_db
        
        if not FACE_DB_OPTIMIZED_AVAILABLE or _optimized_face_db is None:
            return {"error": "Face database not initialized"}
        
        try:
            persons = _optimized_face_db.get_all_persons()
            stats = _optimized_face_db.get_stats()
            
            logger.info("📋 Listed %d known persons", len(persons))
            
            return {
                "success": True,
                "count": len(persons),
                "persons": persons,
                "total_embeddings": stats.get("total_embeddings", 0),
                "model_used": stats.get("model_used", "unknown")
            }
        except Exception as e:
            logger.error("Failed to list known persons: %s", e)
            return {"error": f"Failed to list persons: {str(e)}"}


class GetDatabaseStats(Tool):
    """Get statistics about the face recognition database."""
    
    name = "get_database_stats"
    description = """Get detailed statistics about the face recognition database,
    including number of persons, embeddings per person, and model information.
    Use this to check database health or answer questions about the recognition system."""
    
    parameters_schema = {
        "type": "object",
        "properties": {},
        "required": []
    }
    
    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Récupère les statistiques de la base de données."""
        global _optimized_face_db
        
        if not FACE_DB_OPTIMIZED_AVAILABLE or _optimized_face_db is None:
            return {"error": "Face database not initialized"}
        
        try:
            stats = _optimized_face_db.get_stats()
            logger.info("📊 Database stats: %d persons, %d embeddings", 
                       stats.get("total_persons", 0), 
                       stats.get("total_embeddings", 0))
            return stats
        except Exception as e:
            logger.error("Failed to get database stats: %s", e)
            return {"error": f"Failed to get stats: {str(e)}"}
        


# Registry & specs (dynamic)

# List of available tool classes
ALL_TOOLS: Dict[str, Tool] = {cls.name: cls() for cls in get_concrete_subclasses(Tool)}  # type: ignore[type-abstract]
ALL_TOOL_SPECS = [tool.spec() for tool in ALL_TOOLS.values()]


# Dispatcher
def _safe_load_obj(args_json: str) -> Dict[str, Any]:
    try:
        parsed_args = json.loads(args_json or "{}")
        return parsed_args if isinstance(parsed_args, dict) else {}
    except Exception:
        logger.warning("bad args_json=%r", args_json)
        return {}


async def dispatch_tool_call(tool_name: str, args_json: str, deps: ToolDependencies) -> Dict[str, Any]:
    """Dispatch a tool call by name with JSON args and dependencies."""
    tool = ALL_TOOLS.get(tool_name)

    if not tool:
        return {"error": f"unknown tool: {tool_name}"}

    args = _safe_load_obj(args_json)
    try:
        return await tool(deps, **args)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        logger.exception("Tool error in %s: %s", tool_name, msg)
        return {"error": msg}
