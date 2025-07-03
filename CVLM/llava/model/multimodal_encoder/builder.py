import os
from .clip_encoder import CLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    
    if vision_tower is None:
        raise ValueError('Vision tower configuration is missing')
    
    is_absolute_path_exists = os.path.exists(vision_tower)
    
    # Check if it's a valid path (absolute or relative)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    # If the path doesn't exist but starts with known prefixes, try to resolve it
    if vision_tower.startswith("checkpoints/") or vision_tower.startswith("CVLM/"):
        # Try to find the path in the project structure
        project_paths = [
            vision_tower,  # Use as-is first
            f"CVLM/{vision_tower}" if not vision_tower.startswith("CVLM/") else vision_tower,
            f"../{vision_tower}",
            f"../../{vision_tower}"
        ]
        
        for path in project_paths:
            if os.path.exists(path):
                return CLIPVisionTower(path, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
