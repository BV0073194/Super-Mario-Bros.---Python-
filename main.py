# main.py
# Final Major Refactor: Using OpenCV Template Matching for Block Placement

import pygame
import os
import cv2
import numpy as np

# --- Game Settings ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
SCALE = 3
TILE_SIZE = 16 * SCALE

# Player physics
PLAYER_ACC = 0.5
PLAYER_FRICTION = -0.1
PLAYER_GRAVITY = 0.9
PLAYER_JUMP = -22 
MAX_RUN_SPEED = 6

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SKY_BLUE = (107, 140, 255)
DEBUG_COLOR = (255, 0, 255)
vec = pygame.math.Vector2

# Paths
WORLD_PATH_BASE = os.path.join("game", "assets", "Worlds")
SHARED_PATH = os.path.join("game", "assets", "Shared")
BLOCK_PATH = os.path.join(SHARED_PATH, "blocks")

# --- Helper function for Template Matching ---
def find_template_matches(position_map_path, template_path, threshold=0.9):
    """
    Finds all locations in position_map where the template matches.
    Returns a list of (x, y) coordinates for the top-left of each match.
    """
    pos_map_img = cv2.imread(position_map_path, cv2.IMREAD_UNCHANGED)
    template_img = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

    if pos_map_img is None or template_img is None:
        print(f"Error loading images for template matching: {position_map_path} or {template_path}")
        return []

    # Ensure images have an alpha channel for consistent matching
    if len(pos_map_img.shape) < 3 or pos_map_img.shape[2] == 3:
        pos_map_img = cv2.cvtColor(pos_map_img, cv2.COLOR_BGR2BGRA)
    if len(template_img.shape) < 3 or template_img.shape[2] == 3:
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2BGRA)

    result = cv2.matchTemplate(pos_map_img, template_img, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    matches = []
    processed_points = set()
    for pt in zip(*locations[::-1]):
        too_close = False
        for processed_pt in processed_points:
            if abs(pt[0] - processed_pt[0]) < 10 and abs(pt[1] - processed_pt[1]) < 10:
                too_close = True
                break
        if not too_close:
            matches.append(pt)
            processed_points.add(pt)

    return matches

class SpriteSheet:
    def __init__(self, filename):
        try:
            self.sheet = pygame.image.load(filename).convert()
        except pygame.error as e:
            print(f"Unable to load spritesheet: {filename}")
            raise SystemExit(e)

    def get_sprite(self, col, row, width=16, height=16, padding=1):
        x = col * (width + padding) + padding
        y = row * (height + padding) + padding
        image = pygame.Surface((width, height)).convert()
        image.blit(self.sheet, (0, 0), (x, y, width, height))
        image.set_colorkey(BLACK)
        return pygame.transform.scale(image, (width * SCALE, height * SCALE))

class Animation:
    def __init__(self, sheet, col, row, count, duration, layout='horizontal', looped=True):
        self.frames = []
        if count > 0: self.duration = duration / count
        else: self.duration = float('inf')
        self.looped = looped
        for i in range(count):
            dx = col + i if layout == 'horizontal' else col
            dy = row + i if layout == 'vertical' else row
            self.frames.append(sheet.get_sprite(dx, dy))

class Animator:
    def __init__(self, player):
        self.player = player
        self.animations = {}
        self.current = None
        self.frame_index = 0
        self.last_time = 0

    def add(self, name, anim):
        self.animations[name] = anim

    def set(self, name):
        if name in self.animations and self.current != self.animations[name]:
            self.current = self.animations[name]
            self.frame_index = 0
            self.last_time = pygame.time.get_ticks()

    def update(self):
        if not self.current or not self.current.frames: return
        now = pygame.time.get_ticks()
        if self.current.duration > 0 and now - self.last_time > self.current.duration:
            self.last_time = now
            if self.current.looped:
                self.frame_index = (self.frame_index + 1) % len(self.current.frames)
            else:
                self.frame_index = min(self.frame_index + 1, len(self.current.frames) - 1)
        if self.frame_index < len(self.current.frames):
            frame = self.current.frames[self.frame_index]
            self.player.image = pygame.transform.flip(frame, True, False) if self.player.direction == 'left' else frame

class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h):
        super().__init__()
        self.rect = pygame.Rect(x, y, w, h)

class Block(pygame.sprite.Sprite):
    def __init__(self, x, y, image):
        super().__init__()
        self.image = image
        self.rect = self.image.get_rect(topleft=(x, y))
        self.original_pos = vec(x, y)
        self.bumping = False
        self.bump_offset = 0

    def update(self):
        if self.bumping:
            self.bump_offset += 1
            self.rect.y = self.original_pos.y - 10 * (self.bump_offset / 5 - (self.bump_offset / 5)**2)
            if self.bump_offset >= 10:
                self.rect.y = self.original_pos.y
                self.bumping = False
                self.post_bump()

    def hit(self, player):
        if not self.bumping:
            self.bumping = True
            self.bump_offset = 0

    def post_bump(self):
        pass

class Brick(Block):
    def hit(self, player):
        if not self.bumping:
            if player.power_level == 'small':
                super().hit(player)
            else:
                self.kill()

class LuckyBlock(Block):
    def __init__(self, x, y, image, used_image):
        super().__init__(x,y,image)
        self.used_image = used_image
        self.is_used = False

    def hit(self, player):
        if not self.is_used:
            super().hit(player)

    def post_bump(self):
        if not self.is_used:
            self.image = self.used_image
            self.is_used = True

class Level:
    def __init__(self, world_name):
        self.world_dir = os.path.join(WORLD_PATH_BASE, world_name)
        self.all_sprites = pygame.sprite.Group() 
        self.solid_group = pygame.sprite.Group()
        self.debug_block_overlays = []

        self.load_layers()
        self.width = self.background_layer.get_width()
        self.height = self.background_layer.get_height()
        self.rect = self.background_layer.get_rect()
        self.spawn = vec(150, 400)
        self.find_spawn_pos()

    def load_layers(self):
        print("Loading level from data maps...")
        self.background_layer = self._load_visual_layer("BackgroundData.png")
        self.ground_layer = self._load_visual_layer("GroundData.png")
        self.tube_layer = self._load_visual_layer("TubeData.png")
        self._create_collision_from_map(os.path.join(self.world_dir, "GroundData.png"))
        self._create_collision_from_map(os.path.join(self.world_dir, "TubeData.png"))
        self._create_blocks_from_template_match()

    def _load_visual_layer(self, filename):
        path = os.path.join(self.world_dir, filename)
        image = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(image, (image.get_width() * SCALE, image.get_height() * SCALE))

    def _create_collision_from_map(self, path):
        image_map = pygame.image.load(path).convert_alpha()
        map_w, map_h = image_map.get_size()
        for y in range(map_h):
            x = 0
            while x < map_w:
                if image_map.get_at((x, y))[3] > 0:
                    start_x = x
                    while x < map_w and image_map.get_at((x, y))[3] > 0: x += 1
                    w = (x - start_x) * SCALE
                    platform = Platform(start_x * SCALE, y * SCALE, w, SCALE)
                    self.solid_group.add(platform)
                else: x += 1

    def _create_blocks_from_template_match(self):
        print("\n--- Starting Block Placement ---")

        # --- Breakable Blocks ---
        breakable_pos_map_path = os.path.join(self.world_dir, "BreakablePosData.png")
        breakable_templates = {
            "BreakableBlock_AG.png": pygame.transform.scale(pygame.image.load(os.path.join(BLOCK_PATH, "BreakableBlock_AG.png")).convert_alpha(), (TILE_SIZE, TILE_SIZE)),
            "BreakableBlock_BG.png": pygame.transform.scale(pygame.image.load(os.path.join(BLOCK_PATH, "BreakableBlock_BG.png")).convert_alpha(), (TILE_SIZE, TILE_SIZE))
        }
        for filename, block_image in breakable_templates.items():
            template_path = os.path.join(BLOCK_PATH, filename)
            print(f"Scanning for {filename}...")
            matches = find_template_matches(breakable_pos_map_path, template_path)
            if matches:
                print(f"Found {len(matches)} matches for {filename}:")
                for (x, y) in matches:
                    world_x, world_y = x * SCALE, y * SCALE
                    block = Brick(world_x, world_y, block_image)
                    self.all_sprites.add(block)
                    self.solid_group.add(block)
            else: print(f"No matches found for {filename}.")

        # --- Lucky Blocks ---
        lucky_pos_map_path = os.path.join(self.world_dir, "LuckyBlockPosData.png")
        lucky_template_path = os.path.join(BLOCK_PATH, "LuckyBlock_AG.png")

        # Load the full lucky block spritesheet
        lucky_sheet_img = pygame.image.load(lucky_template_path).convert_alpha()

        # Precisely extract the first 16x16 frame for template matching
        first_frame_template_surf = pygame.Surface((16,16), pygame.SRCALPHA)
        first_frame_template_surf.blit(lucky_sheet_img, (0,0), (0,0,16,16))
        temp_template_path = "temp_lucky_block_template.png"
        pygame.image.save(first_frame_template_surf, temp_template_path)

        # Precisely extract frames for the LuckyBlock object visuals
        lucky_img_surf = pygame.Surface((16, 16), pygame.SRCALPHA)
        lucky_img_surf.blit(lucky_sheet_img, (0,0), (0,0,16,16))
        lucky_img = pygame.transform.scale(lucky_img_surf, (TILE_SIZE, TILE_SIZE))

        used_img_surf = pygame.Surface((16, 16), pygame.SRCALPHA)
        used_img_surf.blit(lucky_sheet_img, (0,0), (17,0,16,16)) # Use precise coordinate
        used_img = pygame.transform.scale(used_img_surf, (TILE_SIZE, TILE_SIZE))

        print(f"Scanning for Lucky Blocks...")
        matches = find_template_matches(lucky_pos_map_path, temp_template_path)
        if matches:
            print(f"Found {len(matches)} matches for Lucky Blocks:")
            for (x,y) in matches:
                world_x, world_y = x * SCALE, y * SCALE
                block = LuckyBlock(world_x, world_y, lucky_img, used_img)
                self.all_sprites.add(block)
                self.solid_group.add(block)
        else:
            print("No Lucky Block matches found.")
        os.remove(temp_template_path)
        print("--- Finished Block Placement ---\n")

    def find_spawn_pos(self):
        if self.solid_group:
            platforms_only = [p for p in self.solid_group if not isinstance(p, Block)]
            if platforms_only:
                start_platform = min(platforms_only, key=lambda s: (s.rect.top, s.rect.left))
            else: 
                start_platform = min(self.solid_group.sprites(), key=lambda s: (s.rect.top, s.rect.left))
            self.spawn = vec(start_platform.rect.left + 48, start_platform.rect.top)

class Player(pygame.sprite.Sprite):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.animator = Animator(self)
        sheet = SpriteSheet(os.path.join(SHARED_PATH, "MarioLuigi.png"))
        self.animator.add('idle', Animation(sheet, 2, 1, 1, 500))
        self.animator.add('walk', Animation(sheet, 0, 1, 3, 360))
        self.animator.add('jump', Animation(sheet, 0, 0, 1, 500))
        self.direction = 'right'
        self.power_level = 'small'
        self.animator.set('idle')
        self.image = self.animator.current.frames[0]
        self.rect = self.image.get_rect()
        self.pos = vec(100, 400)
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)
        self.on_ground = False

    def update(self):
        self.acc = vec(0, PLAYER_GRAVITY)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]: self.acc.x = -PLAYER_ACC; self.direction = 'left'
        if keys[pygame.K_RIGHT]: self.acc.x = PLAYER_ACC; self.direction = 'right'
        self.acc.x += self.vel.x * PLAYER_FRICTION
        self.vel += self.acc
        if abs(self.vel.x) < 0.1: self.vel.x = 0
        if abs(self.vel.x) > MAX_RUN_SPEED: self.vel.x = MAX_RUN_SPEED * (1 if self.vel.x > 0 else -1)

        self.pos.x += self.vel.x
        self.rect.centerx = round(self.pos.x)
        self.check_collisions('horizontal')

        self.pos.y += self.vel.y
        self.rect.centery = round(self.pos.y)
        self.check_collisions('vertical')

        self.animator.update()
        self.set_state()
        if self.rect.top > self.game.level.rect.height:
            print("Player died: fell off map"); self.game.playing = False

    def check_collisions(self, direction):
        hits = pygame.sprite.spritecollide(self, self.game.level.solid_group, False)
        if direction == 'horizontal':
            for hit in hits:
                if self.vel.x > 0: self.rect.right = hit.rect.left
                elif self.vel.x < 0: self.rect.left = hit.rect.right
                self.pos.x = self.rect.centerx
        if direction == 'vertical':
            self.on_ground = False
            for hit in hits:
                if self.vel.y > 0:
                    self.rect.bottom = hit.rect.top
                    self.on_ground = True
                    self.vel.y = 0
                elif self.vel.y < 0:
                    self.rect.top = hit.rect.bottom
                    self.vel.y = 0
                    if isinstance(hit, Block): hit.hit(self)
            self.pos.y = self.rect.centery

    def set_state(self):
        if not self.on_ground: self.animator.set('jump')
        elif abs(self.vel.x) > 0.5: self.animator.set('walk')
        else: self.animator.set('idle')

    def jump(self):
        if self.on_ground: self.vel.y = PLAYER_JUMP

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.playing = True
        self.debug_mode = False

    def new_game(self):
        self.level = Level("W1-1")
        self.player = Player(self)
        self.player.pos = self.level.spawn
        self.player.rect.midbottom = self.level.spawn
        self.level.all_sprites.add(self.player)
        self.camera = Camera(self.level.rect.width, self.level.rect.height)
        self.run()

    def run(self):
        self.playing = True
        while self.playing and self.running:
            self.clock.tick(FPS)
            self.events()
            self.update()
            self.draw()

    def events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: self.running = False; self.playing = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_F1:
                self.debug_mode = not self.debug_mode

    def update(self):
        self.level.all_sprites.update()
        self.camera.update(self.player)

    def draw(self):
        self.screen.fill(SKY_BLUE)
        self.screen.blit(self.level.background_layer, self.camera.apply_rect(self.level.rect))
        self.screen.blit(self.level.ground_layer, self.camera.apply_rect(self.level.ground_layer.get_rect()))
        self.screen.blit(self.level.tube_layer, self.camera.apply_rect(self.level.tube_layer.get_rect()))
        for sprite in self.level.all_sprites:
            self.screen.blit(sprite.image, self.camera.apply(sprite))

        if self.debug_mode:
            for sprite in self.level.solid_group:
                pygame.draw.rect(self.screen, DEBUG_COLOR, self.camera.apply_rect(sprite.rect), 1)

        pygame.display.flip()

class Camera:
    def __init__(self, width, height):
        self.camera = pygame.Rect(0, 0, width, height)
        self.width, self.height = width, height

    def apply(self, entity):
        return entity.rect.move(self.camera.topleft)
    def apply_rect(self, rect):
        return rect.move(self.camera.topleft)

    def update(self, target):
        x = -target.rect.centerx + SCREEN_WIDTH // 3
        y = -target.rect.centery + SCREEN_HEIGHT // 1.25
        x = min(0, x)
        x = max(-(self.width - SCREEN_WIDTH), x)
        y = min(0, y)
        y = max(-(self.height - SCREEN_HEIGHT), y)
        self.camera = pygame.Rect(x, y, self.width, self.height)

if __name__ == "__main__":
    g = Game()
    while g.running:
        g.new_game()
    pygame.quit()
