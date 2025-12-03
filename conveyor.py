import cv2
import numpy as np
import random
import os
import math
from config import CONFIG


class ConveyorObject:
    def __init__(self, image, x, y):
        self.image = image
        self.x = x
        self.y = y


def load_and_rotate_image(path, angle):
    """Load an image with alpha channel and rotate it."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Warning: Could not load image: {path}")
        return None

    # Add alpha channel if not present
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    if angle == 0:
        return img

    # Get image dimensions
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Calculate new bounding box size after rotation
    angle_rad = np.radians(angle)
    cos_a = abs(np.cos(angle_rad))
    sin_a = abs(np.sin(angle_rad))
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(w * sin_a + h * cos_a)

    # Rotation matrix with adjusted center
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Rotate with transparent background
    rotated = cv2.warpAffine(img, M, (new_w, new_h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
    return rotated


def create_shadow(image, opacity):
    """Create a shadow version of an image with specified opacity."""
    light_cfg = CONFIG["light_source"]

    h, w = image.shape[:2]

    # Create shadow from alpha channel
    alpha = image[:, :, 3].astype(np.float32) / 255.0

    # Apply blur
    blur_size = light_cfg["blur"]
    if blur_size % 2 == 0:
        blur_size += 1  # Must be odd
    blurred_alpha = cv2.GaussianBlur(alpha, (blur_size, blur_size), 0)

    # Apply opacity
    blurred_alpha = (blurred_alpha * opacity * 255).astype(np.uint8)

    # Create shadow image
    shadow = np.zeros((h, w, 4), dtype=np.uint8)
    shadow[:, :, 0] = light_cfg["color"][0]
    shadow[:, :, 1] = light_cfg["color"][1]
    shadow[:, :, 2] = light_cfg["color"][2]
    shadow[:, :, 3] = blurred_alpha

    return shadow


def calculate_shadow_params(obj_x, obj_y, obj_w, obj_h):
    """Calculate shadow offset and opacity based on light source position."""
    light_cfg = CONFIG["light_source"]

    # Object center
    center_x = obj_x + obj_w // 2
    center_y = obj_y + obj_h // 2

    # Vector from light to object
    dx = center_x - light_cfg["x"]
    dy = center_y - light_cfg["y"]

    # Distance from light
    distance = math.sqrt(dx * dx + dy * dy)

    # Normalize direction
    if distance > 0:
        dir_x = dx / distance
        dir_y = dy / distance
    else:
        dir_x, dir_y = 0, 1  # Default downward if at light position

    # Shadow length scales with distance
    shadow_length = (distance / light_cfg["reference_distance"]) * light_cfg["base_shadow_length"]
    shadow_length = min(shadow_length, light_cfg["max_shadow_length"])

    # Calculate offset
    offset_x = dir_x * shadow_length
    offset_y = dir_y * shadow_length

    # Opacity decreases with distance (inverse relationship)
    opacity_range = light_cfg["max_opacity"] - light_cfg["min_opacity"]
    opacity = light_cfg["max_opacity"] - (distance / (light_cfg["reference_distance"] * 3)) * opacity_range
    opacity = max(light_cfg["min_opacity"], min(light_cfg["max_opacity"], opacity))

    return offset_x, offset_y, opacity


def blend_image(background, foreground, x, y):
    """Blend a foreground image with alpha onto background."""
    h, w = foreground.shape[:2]
    bg_h, bg_w = background.shape[:2]

    # Calculate visible region
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)

    if x1 >= x2 or y1 >= y2:
        return  # Completely off screen

    # Source region in foreground
    fx1 = x1 - x
    fy1 = y1 - y
    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)

    # Get the regions
    fg_region = foreground[fy1:fy2, fx1:fx2]
    bg_region = background[y1:y2, x1:x2]

    # Alpha blending
    alpha = fg_region[:, :, 3:4].astype(np.float32) / 255.0
    blended = (1 - alpha) * bg_region + alpha * fg_region[:, :, :3]
    background[y1:y2, x1:x2] = blended.astype(np.uint8)


def check_overlap(x1, y1, w1, h1, x2, y2, w2, h2, spacing):
    """Check if two rectangles overlap with given spacing."""
    return not (x1 + w1 + spacing <= x2 or
                x2 + w2 + spacing <= x1 or
                y1 + h1 + spacing <= y2 or
                y2 + h2 + spacing <= y1)


def spawn_object(screen_width, screen_height, image_paths, existing_objects=None):
    """Spawn a new object at the entry edge."""
    # Pick random image
    path = random.choice(image_paths)

    # Random rotation
    rot_cfg = CONFIG["rotation"]
    if rot_cfg["enabled"]:
        angle = random.uniform(rot_cfg["min_angle"], rot_cfg["max_angle"])
    else:
        angle = 0

    image = load_and_rotate_image(path, angle)
    if image is None:
        return None

    h, w = image.shape[:2]

    # Position based on direction
    if CONFIG["direction"] == "rtl":
        x = screen_width  # Start just off right edge
    else:
        x = -w  # Start just off left edge

    overlap_cfg = CONFIG["overlap"]

    if overlap_cfg["allow"] or existing_objects is None:
        # Random Y position (original behavior)
        y = random.randint(-h // 4, screen_height - h + h // 4)
    else:
        # Find a non-overlapping Y position
        spacing = overlap_cfg["min_spacing"]
        min_y = -h // 4
        max_y = screen_height - h + h // 4

        # Try random positions first
        max_attempts = 50
        y = None
        for _ in range(max_attempts):
            candidate_y = random.randint(min_y, max_y)
            overlaps = False
            for obj in existing_objects:
                obj_h, obj_w = obj.image.shape[:2]
                if check_overlap(x, candidate_y, w, h, obj.x, obj.y, obj_w, obj_h, spacing):
                    overlaps = True
                    break
            if not overlaps:
                y = candidate_y
                break

        if y is None:
            # No valid position found, skip spawning
            return None

    return ConveyorObject(image, x, y)


def load_background(width, height):
    """Load and prepare background for tiling."""
    bg_path = CONFIG["background_image"]
    if bg_path and os.path.exists(bg_path):
        bg = cv2.imread(bg_path)
        if bg is not None:
            # Resize to match screen height, keep aspect ratio
            bg_h, bg_w = bg.shape[:2]
            scale = height / bg_h
            new_w = int(bg_w * scale)
            return cv2.resize(bg, (new_w, height))

    # Solid color fallback (make it wide enough to tile)
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:] = CONFIG["background_color"]
    return bg


def draw_scrolling_background(frame, bg_image, offset):
    """Draw background image with horizontal scrolling/tiling."""
    height, width = frame.shape[:2]
    bg_w = bg_image.shape[1]

    # Normalize offset to bg width
    offset = int(offset) % bg_w

    # Draw tiled background
    x = -offset
    while x < width:
        # Calculate how much of bg to draw
        src_x = 0
        dst_x = x
        copy_w = bg_w

        if x < 0:
            src_x = -x
            dst_x = 0
            copy_w = bg_w + x

        if dst_x + copy_w > width:
            copy_w = width - dst_x

        if copy_w > 0:
            frame[:, dst_x:dst_x + copy_w] = bg_image[:, src_x:src_x + copy_w]

        x += bg_w


def main():
    width = CONFIG["width"]
    height = CONFIG["height"]
    fps = CONFIG["fps"]
    duration = CONFIG["duration"]
    total_frames = fps * duration

    # Ensure output directory exists
    output_dir = os.path.dirname(CONFIG["output_file"])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Setup frame saving
    save_frames_cfg = CONFIG["save_frames"]
    if save_frames_cfg["enabled"]:
        frames_dir = save_frames_cfg["output_dir"]
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir)

    # Check for valid images
    valid_images = [p for p in CONFIG["images"] if os.path.exists(p)]
    if not valid_images:
        print("Error: No valid images found in config. Please add images to the images/ directory.")
        print("Expected paths:", CONFIG["images"])
        return

    print(f"Found {len(valid_images)} valid images")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(CONFIG["output_file"], fourcc, fps, (width, height))

    # Load background
    background = load_background(width, height)

    # Initialize objects
    objects = []
    for _ in range(CONFIG["max_images"]):
        obj = spawn_object(width, height, valid_images, objects)
        if obj:
            # Distribute across screen initially
            if CONFIG["direction"] == "rtl":
                obj.x = random.randint(0, width)
            else:
                obj.x = random.randint(-obj.image.shape[1], width - obj.image.shape[1])
            objects.append(obj)

    print(f"Generating {total_frames} frames ({duration}s at {fps}fps)...")
    print(f"Output: {CONFIG['output_file']}")
    print(f"Resolution: {width}x{height}")
    print("Press 'q' to cancel")

    light_cfg = CONFIG["light_source"]
    speed = CONFIG["speed"]
    direction = CONFIG["direction"]

    # Background scroll offset
    bg_offset = 0

    for frame_num in range(total_frames):
        # Create frame with scrolling background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_scrolling_background(frame, background, bg_offset)

        # Move background (same direction as objects)
        if direction == "rtl":
            bg_offset += speed
        else:
            bg_offset -= speed

        # Move objects
        for obj in objects:
            if direction == "rtl":
                obj.x -= speed
            else:
                obj.x += speed

        # Remove objects that have exited
        if direction == "rtl":
            objects = [o for o in objects if o.x + o.image.shape[1] > 0]
        else:
            objects = [o for o in objects if o.x < width]

        # Spawn new objects to maintain count
        while len(objects) < CONFIG["max_images"]:
            obj = spawn_object(width, height, valid_images, objects)
            if obj:
                objects.append(obj)
            else:
                break  # No valid non-overlapping position found

        # Draw shadows first
        if light_cfg["enabled"]:
            for obj in objects:
                h, w = obj.image.shape[:2]
                offset_x, offset_y, opacity = calculate_shadow_params(obj.x, obj.y, w, h)
                shadow = create_shadow(obj.image, opacity)
                sx = obj.x + int(offset_x)
                sy = obj.y + int(offset_y)
                blend_image(frame, shadow, sx, sy)

        # Draw objects
        for obj in objects:
            blend_image(frame, obj.image, obj.x, obj.y)

        # Write frame
        out.write(frame)

        # Save frame as image
        if save_frames_cfg["enabled"] and frame_num % save_frames_cfg["every_nth"] == 0:
            frame_path = os.path.join(save_frames_cfg["output_dir"], f"frame_{frame_num:06d}.png")
            cv2.imwrite(frame_path, frame)

        # Show preview
        if CONFIG["show_preview"]:
            # Scale down for preview
            preview_scale = 0.25
            preview = cv2.resize(frame, None, fx=preview_scale, fy=preview_scale)
            cv2.imshow("Conveyor Belt Preview", preview)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nCancelled by user")
                break

        # Progress
        if frame_num % fps == 0:
            progress = (frame_num / total_frames) * 100
            print(f"\rProgress: {progress:.1f}% ({frame_num}/{total_frames} frames)", end="", flush=True)

    print("\nDone!")
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved to: {CONFIG['output_file']}")


if __name__ == "__main__":
    main()
