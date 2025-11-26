"""Face anonymization methods for video processing.

HIPAA-compliant face redaction using solid fills only.
Blur/pixelation methods are NOT included as they can be reversed by AI.
"""
import cv2
import numpy as np
from typing import Tuple


class Anonymizer:
    """Static methods for HIPAA-compliant face anonymization in video frames.

    Only solid redaction methods are provided (black rectangle, color fill).
    Blur and pixelation are NOT HIPAA-compliant as they can be reversed by AI.
    """

    @staticmethod
    def _apply_padding(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        padding: float,
        img_height: int,
        img_width: int
    ) -> Tuple[int, int, int, int]:
        """Apply padding to bounding box coordinates.

        Args:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            padding: Padding as fraction of box size (e.g., 0.1 = 10%)
            img_height: Image height for boundary checking
            img_width: Image width for boundary checking

        Returns:
            Tuple of padded (x1, y1, x2, y2) coordinates
        """
        width = x2 - x1
        height = y2 - y1

        pad_x = int(width * padding)
        pad_y = int(height * padding)

        x1_padded = max(0, x1 - pad_x)
        y1_padded = max(0, y1 - pad_y)
        x2_padded = min(img_width, x2 + pad_x)
        y2_padded = min(img_height, y2 + pad_y)

        return x1_padded, y1_padded, x2_padded, y2_padded

    @staticmethod
    def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to BGR tuple for OpenCV.

        Args:
            hex_color: Hex color string (e.g., "#FF0000" or "FF0000")

        Returns:
            Tuple of (B, G, R) values for OpenCV
        """
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)  # OpenCV uses BGR

    @staticmethod
    def black_rectangle(
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        padding: float = 0.1
    ) -> np.ndarray:
        """Draw black rectangle over face region.

        Args:
            image: Input image as numpy array
            x1, y1: Top-left corner of face bounding box
            x2, y2: Bottom-right corner of face bounding box
            padding: Padding fraction to expand bounding box (default: 0.1)

        Returns:
            Image with black rectangle over face
        """
        img_height, img_width = image.shape[:2]
        x1, y1, x2, y2 = Anonymizer._apply_padding(
            x1, y1, x2, y2, padding, img_height, img_width
        )

        result = image.copy()
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), -1)

        return result

    @staticmethod
    def color_fill(
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        color: str = "#000000",
        padding: float = 0.1
    ) -> np.ndarray:
        """Fill face region with solid color.

        Args:
            image: Input image as numpy array
            x1, y1: Top-left corner of face bounding box
            x2, y2: Bottom-right corner of face bounding box
            color: Hex color string (e.g., "#FF0000" for red)
            padding: Padding fraction to expand bounding box (default: 0.1)

        Returns:
            Image with colored rectangle over face
        """
        img_height, img_width = image.shape[:2]
        x1, y1, x2, y2 = Anonymizer._apply_padding(
            x1, y1, x2, y2, padding, img_height, img_width
        )

        bgr_color = Anonymizer.hex_to_bgr(color)

        result = image.copy()
        cv2.rectangle(result, (x1, y1), (x2, y2), bgr_color, -1)

        return result

    @staticmethod
    def apply(
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        mode: str,
        color: str = "#000000",
        padding: float = 0.1
    ) -> np.ndarray:
        """Route to appropriate anonymization method.

        Args:
            image: Input image as numpy array
            x1, y1: Top-left corner of face bounding box
            x2, y2: Bottom-right corner of face bounding box
            mode: Anonymization mode ("black" or "color")
            color: Hex color string for color_fill mode
            padding: Padding fraction to expand bounding box

        Returns:
            Anonymized image

        Raises:
            ValueError: If mode is not recognized
        """
        if mode == "black":
            return Anonymizer.black_rectangle(image, x1, y1, x2, y2, padding)
        elif mode == "color":
            return Anonymizer.color_fill(image, x1, y1, x2, y2, color, padding)
        else:
            raise ValueError(
                f"Unknown anonymization mode: {mode}. "
                f"Valid modes are 'black' or 'color'. "
                f"Blur is NOT supported (HIPAA non-compliant)."
            )
