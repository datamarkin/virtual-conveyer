CONFIG = {
    # Images to display on the conveyor (add your image paths here)
    "images": [
        "images/1.png",
        "images/2.png",
        "images/3.png",
        "images/4.png",
        "images/5.png"
    ],

    # Video output settings
    "output_file": "output/conveyor.mp4",
    "width": 1920,              # Default 4K
    "height": 1080,             # Default 4K
    "fps": 30,                  # Default 60 FPS
    "duration": 60,             # Duration in seconds

    # Conveyor settings
    "speed": 5,                 # Pixels per frame
    "direction": "rtl",         # "rtl" (right-to-left) or "ltr" (left-to-right)
    "max_images": 50,           # Max objects visible at once

    # Background
    "background_image": "bg-2.jpg",   # Path to background image, or None for solid color
    "background_color": (50, 50, 50),  # BGR color if no background image

    # Object appearance
    "rotation": {
        "enabled": True,
        "min_angle": 0,         # Minimum rotation in degrees
        "max_angle": 360,       # Maximum rotation in degrees
    },

    # Light source settings (dynamic shadows)
    "light_source": {
        "enabled": True,
        "x": 960,                    # Light X position (can be negative for distant light)
        "y": -500,                   # Light Y position (negative = above scene)
        "base_shadow_length": 10,    # Base shadow length at reference distance
        "reference_distance": 500,   # Distance at which shadow_length = base_shadow_length
        "max_shadow_length": 100,    # Cap to prevent extremely long shadows
        "min_opacity": 0.05,         # Minimum shadow opacity (at max distance)
        "max_opacity": 0.15,          # Maximum shadow opacity (closest to light)
        "blur": 15,                  # Shadow blur radius (must be odd)
        "color": (0, 0, 0),          # Shadow color (BGR)
    },

    # Display preview while rendering
    "show_preview": False,
}
