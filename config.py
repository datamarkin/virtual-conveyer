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
    "duration": 15,             # Duration in seconds

    # Conveyor settings
    "speed": 5,                 # Pixels per frame
    "direction": "rtl",         # "rtl" (right-to-left) or "ltr" (left-to-right)
    "max_images": 20,           # Max objects visible at once

    # Background
    "background_image": "bg-2.jpg",   # Path to background image, or None for solid color
    "background_color": (50, 50, 50),  # BGR color if no background image

    # Object appearance
    "rotation": {
        "enabled": True,
        "min_angle": 0,         # Minimum rotation in degrees
        "max_angle": 360,       # Maximum rotation in degrees
    },

    # Shadow settings
    "shadow": {
        "enabled": True,
        "offset_x": 10,         # Shadow offset X
        "offset_y": 10,         # Shadow offset Y
        "blur": 15,             # Shadow blur radius (must be odd)
        "opacity": 0.2,         # Shadow opacity (0-1)
        "color": (0, 0, 0),     # Shadow color (BGR)
    },

    # Display preview while rendering
    "show_preview": True,
}
