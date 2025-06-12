# main.py
# Refactor 15: Adding Audio System Failsafe

import pygame
import os
import cv2
import numpy as np
import random

# --- Game Settings ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60
SCALE = 3
TILE_SIZE = 16 * SCALE

# --- NES-Style Physics Constants ---
PLAYER_ACC = 0.55
PLAYER_FRICTION = -0.25
PLAYER_GRAVITY = 0.8
PLAYER_JUMP_STRENGTH = -19
MAX_FALL_SPEED = 12
MAX_RUN_SPEED = 7

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
# --- Path for Sound Assets ---
SOUNDS_PATH = os.path.join("game", "assets", "sounds")
MUSIC_PATH = os.path.join(SOUNDS_PATH, "music")


# --- Helper function for Template Matching ---
def find_template_matches(position_map_path, template_path, threshold=0.9):
    pos_map_img = cv2.imread(position_map_path, cv2.IMREAD_UNCHANGED)
    template_img = cv2.imread(template_path, cv2.IMREAD_UNCHANGED)

    if pos_map_img is None:
        print(f"Error: Could not load position map image at {position_map_path}")
        return []
    if template_img is None:
        print(f"Error: Could not load template image at {template_path}")
        return []

    if len(pos_map_img.shape) < 3 or pos_map_img.shape[2] == 3:
        pos_map_img = cv2.cvtColor(pos_map_img, cv2.COLOR_BGR2BGRA)
    if len(template_img.shape) < 3 or template_img.shape[2] == 3:
        template_img = cv2.cvtColor(template_img, cv2.COLOR_BGR2BGRA)

    result = cv2.matchTemplate(pos_map_img, template_img, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    matches = []
    processed_points = set()
    h, w = template_img.shape[:2]
    for pt in zip(*locations[::-1]):
        too_close = False
        for processed_pt in processed_points:
            if abs(pt[0] - processed_pt[0]) < w * 0.5 and abs(pt[1] - processed_pt[1]) < h * 0.5:
                too_close = True
                break
        if not too_close:
            matches.append(pt)
            processed_points.add(pt)
    return matches


class SpriteSheet:
    def __init__(self, filename):
        try:
            self.sheet = pygame.image.load(filename).convert_alpha()
        except pygame.error as e:
            print(f"Unable to load spritesheet: {filename}")
            raise SystemExit(e)

    def get_sprite(self, col, row, width=16, height=16, padding=1):
        x = col * (width + padding) + padding
        y = row * (height + padding) + padding
        image = pygame.Surface((width, height), pygame.SRCALPHA)
        image.blit(self.sheet, (0, 0), (x, y, width, height))
        return pygame.transform.scale(image, (width * SCALE, height * SCALE))


class Animation:
    def __init__(self, sheet, col, row, count, duration, layout="horizontal", looped=True):
        self.frames = []
        if count > 0:
            self.duration = duration / count if duration > 0 else 0
        else:
            self.duration = float("inf")
        self.looped = looped
        for i in range(count):
            dx = col + i if layout == "horizontal" else col
            dy = row if layout == "horizontal" else row + i
            self.frames.append(sheet.get_sprite(dx, dy))


class Animator:
    def __init__(self, target_sprite):
        self.target = target_sprite
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
        if not self.current or not self.current.frames:
            return
        now = pygame.time.get_ticks()
        if self.current.duration > 0 and now - self.last_time > self.current.duration:
            self.last_time = now
            if self.current.looped:
                self.frame_index = (self.frame_index + 1) % len(self.current.frames)
            else:
                self.frame_index = min(self.frame_index + 1, len(self.current.frames) - 1)

        if self.frame_index < len(self.current.frames):
            frame = self.current.frames[self.frame_index]

            if hasattr(self.target, 'direction') and self.target.direction == 'left':
                frame = pygame.transform.flip(frame, True, False)

            self.target.image = frame
            if hasattr(self.target, 'hitbox'):
                self.target.rect = self.target.image.get_rect(center=self.target.hitbox.center)


class Platform(pygame.sprite.Sprite):
    def __init__(self, x, y, w, h):
        super().__init__()
        self.rect = pygame.Rect(x, y, w, h)
        self.hitbox = self.rect.copy()

class Particle(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.image = pygame.Surface((SCALE, SCALE))
        self.image.fill(color)
        self.rect = self.image.get_rect(center=(x, y))
        self.pos = vec(x, y)
        self.vel = vec(random.uniform(-4, 4), random.uniform(-12, -4))

    def update(self):
        self.vel.y += PLAYER_GRAVITY
        self.pos += self.vel
        self.rect.center = self.pos
        if self.rect.top > SCREEN_HEIGHT:
            self.kill()

class Block(pygame.sprite.Sprite):
    def __init__(self, x, y, image, game):
        super().__init__()
        self.game = game
        self.image = image
        self.rect = self.image.get_rect(topleft=(x, y))
        self.original_pos = self.rect.center
        self.bumping = False
        self.bump_offset = 0
        hitbox_size = 15 * SCALE
        self.hitbox = pygame.Rect(0, 0, hitbox_size, hitbox_size)
        self.hitbox.center = self.rect.center

    def update(self):
        if self.bumping:
            self.bump_offset += 1
            progress = self.bump_offset / 12.0
            y_offset = 48 * (progress - progress**2)
            self.rect.centery = self.original_pos[1] - y_offset

            if self.bump_offset >= 12:
                self.rect.center = self.original_pos
                self.bumping = False
                self.bump_offset = 0
                self.post_bump()

        self.hitbox.center = self.rect.center

    def hit(self, player):
        if not self.bumping:
            self.game.play_sfx("bump")
            self.bumping = True
            self.bump_offset = 0

    def post_bump(self):
        pass

class Brick(Block):
    def hit(self, player):
        if not self.bumping:
            if player.power_level == "small":
                super().hit(player)
            else:
                self.game.play_sfx("brick_break")
                self.game.create_particles(self.rect, self.image)
                self.kill()

class LuckyBlock(Block):
    def __init__(self, x, y, active_image, used_image, game):
        super().__init__(x, y, active_image, game)
        self.used_image = used_image
        self.is_used = False

    def post_bump(self):
        if not self.is_used:
            self.is_used = True
            self.image = self.used_image
            self.rect = self.image.get_rect(center=self.original_pos)
            self.game.play_sfx("powerup_appears")


class Level:
    def __init__(self, world_name, game):
        self.game = game
        self.world_dir = os.path.join(WORLD_PATH_BASE, world_name)
        self.all_sprites = pygame.sprite.Group()
        self.solid_group = pygame.sprite.Group()
        self.breakable_blocks = pygame.sprite.Group()
        self.lucky_blocks = pygame.sprite.Group()
        self.music_file = "01-main-theme-overworld.mp3"
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
        self._create_objects_from_template_match()

    def _load_visual_layer(self, filename):
        path = os.path.join(self.world_dir, filename)
        raw_image = pygame.image.load(path)
        scaled_image = pygame.transform.scale(
            raw_image, (raw_image.get_width() * SCALE, raw_image.get_height() * SCALE)
        )
        return scaled_image.convert_alpha()


    def _create_collision_from_map(self, path):
        try:
            image_map = pygame.image.load(path).convert_alpha()
        except pygame.error:
            print(f"Collision map not found: {path}. Skipping.")
            return
        map_w, map_h = image_map.get_size()
        for y in range(map_h):
            x = 0
            while x < map_w:
                if image_map.get_at((x, y))[3] > 0:
                    start_x = x
                    while x < map_w and image_map.get_at((x, y))[3] > 0:
                        x += 1
                    w = (x - start_x) * SCALE
                    platform = Platform(start_x * SCALE, y * SCALE, w, TILE_SIZE)
                    self.solid_group.add(platform)
                else:
                    x += 1

    def _create_objects_from_template_match(self):
        print("\n--- Starting Object Placement from Templates ---")
        breakable_pos_map_path = os.path.join(self.world_dir, "BreakablePosData.png")
        breakable_definitions = {
            "BAGDect.png": "BreakableBlock_AG.png",
            "BBGDect.png": "BreakableBlock_BG.png"
        }
        for dect_img, place_img in breakable_definitions.items():
            print(f"Scanning for {dect_img}...")
            template_path = os.path.join(BLOCK_PATH, dect_img)
            placement_sprite_path = os.path.join(BLOCK_PATH, place_img)
            try:
                raw_surf = pygame.image.load(placement_sprite_path)
                scaled_surf = pygame.transform.scale(raw_surf, (TILE_SIZE, TILE_SIZE))
                block_image = scaled_surf.convert_alpha()
            except pygame.error:
                print(f"Could not load placement sprite: {placement_sprite_path}. Skipping.")
                continue
            matches = find_template_matches(breakable_pos_map_path, template_path)
            print(f"Found {len(matches)} matches.")
            for x, y in matches:
                block = Brick(x * SCALE, y * SCALE, block_image, self.game)
                self.all_sprites.add(block)
                self.solid_group.add(block)
                self.breakable_blocks.add(block)

        lucky_pos_map_path = os.path.join(self.world_dir, "LuckyBlockPosData.png")
        lucky_dect_path = os.path.join(BLOCK_PATH, "LuckyBlockDect.png")
        lucky_spritesheet_path = os.path.join(BLOCK_PATH, "LuckyBlock_AG.png")
        print(f"Scanning for Lucky Blocks...")
        try:
            lucky_sheet = SpriteSheet(lucky_spritesheet_path)
            active_image = lucky_sheet.get_sprite(0, 0, 16, 16, 0)
            used_image = lucky_sheet.get_sprite(1, 0, 16, 16, 0)
        except Exception as e:
            print(f"Could not load Lucky Block assets: {e}. Skipping.")
        else:
            matches = find_template_matches(lucky_pos_map_path, lucky_dect_path)
            print(f"Found {len(matches)} Lucky Block matches.")
            for x, y in matches:
                block = LuckyBlock(x * SCALE, y * SCALE, active_image, used_image, self.game)
                self.all_sprites.add(block)
                self.solid_group.add(block)
                self.lucky_blocks.add(block)


    def find_spawn_pos(self):
        if self.solid_group:
            platforms_only = [p for p in self.solid_group if isinstance(p, Platform)]
            if platforms_only:
                start_platform = min(platforms_only, key=lambda s: (s.rect.top, s.rect.left))
            else:
                start_platform = min(self.solid_group.sprites(), key=lambda s: (s.rect.top, s.rect.left))
            self.spawn = vec(start_platform.rect.left + 48, start_platform.rect.top - TILE_SIZE)


class Player(pygame.sprite.Sprite):
    def __init__(self, game):
        super().__init__()
        self.game = game
        self.image = pygame.Surface((12 * SCALE, 15 * SCALE)).convert_alpha()
        self.image.fill((0,0,0,0))
        self.rect = self.image.get_rect()
        self.hitbox = self.rect.inflate(-4, -2)
        self.pos = vec(100, 400)
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)
        self.direction = "right"
        self.on_ground = False

        self.power_level = "small"
        self.small_anims = {}
        self.big_anims = {}
        self.animator = Animator(self)
        self.load_animations()
        self.set_power_level("small")

    def load_animations(self):
        sheet_path = os.path.join(SHARED_PATH, "MarioLuigi.png")
        try:
            sheet = SpriteSheet(sheet_path)
            self.small_anims["idle"] = Animation(sheet, 0, 1, 1, 500)
            self.small_anims["walk"] = Animation(sheet, 1, 1, 3, 300)
            self.small_anims["jump"] = Animation(sheet, 0, 0, 1, 500)
            self.small_anims["skid"] = Animation(sheet, 4, 1, 1, 500)

            self.big_anims["idle"] = Animation(sheet, 0, 3, 1, 500)
            self.big_anims["walk"] = Animation(sheet, 1, 3, 3, 300)
            self.big_anims["jump"] = Animation(sheet, 0, 2, 1, 500)
            self.big_anims["skid"] = Animation(sheet, 4, 3, 1, 500)

        except Exception as e:
            print(f"Could not load player animations from {sheet_path}: {e}")

    def set_power_level(self, level):
        self.power_level = level
        bottom = self.hitbox.bottom
        if self.power_level == "big":
            print("Player is now BIG")
            self.animator.animations = self.big_anims
            self.hitbox = pygame.Rect(0, 0, 16 * SCALE, 30 * SCALE)
        else: 
            print("Player is now SMALL")
            self.animator.animations = self.small_anims
            self.hitbox = pygame.Rect(0, 0, 12 * SCALE, 15 * SCALE)

        self.hitbox.bottom = bottom
        self.pos = vec(self.hitbox.center)
        self.animator.set("idle")


    def update(self):
        keys = pygame.key.get_pressed()
        self.acc.x = 0
        if keys[pygame.K_LEFT]:
            self.acc.x = -PLAYER_ACC
            self.direction = "left"
        if keys[pygame.K_RIGHT]:
            self.acc.x = PLAYER_ACC
            self.direction = "right"

        if self.on_ground:
            self.acc.x += self.vel.x * PLAYER_FRICTION
        self.vel.x += self.acc.x
        self.vel.y += PLAYER_GRAVITY
        if abs(self.vel.x) < 0.1: self.vel.x = 0
        self.vel.x = max(-MAX_RUN_SPEED, min(MAX_RUN_SPEED, self.vel.x))
        self.vel.y = min(MAX_FALL_SPEED, self.vel.y)

        self.pos.x += self.vel.x
        self.hitbox.centerx = round(self.pos.x)
        self.check_collisions('horizontal', self.game.level.solid_group)

        self.pos.y += self.vel.y
        self.hitbox.centery = round(self.pos.y)
        self.on_ground = False
        self.check_collisions('vertical', self.game.level.solid_group)

        self.set_anim_state()
        self.animator.update()

        if self.pos.y > self.game.level.rect.height + 100:
            print("Player died: fell off map")
            self.game.playing = False

    def check_collisions(self, direction, collidables):
        keys = pygame.key.get_pressed()
        for sprite in collidables:
            if self.hitbox.colliderect(sprite.hitbox):
                if direction == 'horizontal':
                    if self.vel.x > 0:
                        self.hitbox.right = sprite.hitbox.left
                    elif self.vel.x < 0:
                        self.hitbox.left = sprite.hitbox.right
                    self.pos.x = self.hitbox.centerx
                    self.vel.x = 0
                if direction == 'vertical':
                    if self.vel.y > 0:
                        self.hitbox.bottom = sprite.hitbox.top
                        self.on_ground = True
                        self.vel.y = 0
                        if isinstance(sprite, Block) and keys[pygame.K_DOWN]:
                             sprite.hit(self)
                    elif self.vel.y < 0:
                        self.hitbox.top = sprite.hitbox.bottom
                        self.vel.y = 0
                        if isinstance(sprite, Block):
                            sprite.hit(self)
                    self.pos.y = self.hitbox.centery

    def set_anim_state(self):
        if not self.on_ground:
            self.animator.set("jump")
        else:
            if (self.vel.x > 1 and self.direction == 'left') or \
               (self.vel.x < -1 and self.direction == 'right'):
                self.animator.set("skid")
            elif abs(self.vel.x) > 0.1:
                self.animator.set("walk")
            else:
                self.animator.set("idle")

    def jump(self):
        if self.on_ground:
            self.vel.y = PLAYER_JUMP_STRENGTH
            self.game.play_sfx("jump")


class Game:
    def __init__(self):
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        # --- FIX: Added failsafe for audio system initialization ---
        try:
            pygame.mixer.init()
            self.audio_enabled = True
        except pygame.error:
            print("Audio system not available. Sounds disabled.")
            self.audio_enabled = False

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Super Python Bro")
        self.clock = pygame.time.Clock()
        self.running = True
        self.playing = True
        self.debug_mode = False
        self.console_active = False
        self.console_text = ""
        self.console_font = pygame.font.Font(None, 32)
        self.sfx = {}
        self.load_sfx()
        self.particle_group = pygame.sprite.Group()

    def new_game(self):
        self.particle_group.empty()
        self.level = Level("W1-1", self)
        self.player = Player(self)
        self.player.pos = self.level.spawn
        self.level.all_sprites.add(self.player)
        self.camera = Camera(self.level.rect.width, self.level.rect.height)
        self.play_level_music()
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
            if event.type == pygame.QUIT:
                self.running = False
                self.playing = False
            elif event.type == pygame.KEYDOWN:
                if self.console_active:
                    if event.key == pygame.K_RETURN:
                        self.execute_command(self.console_text)
                        self.console_active = False
                    elif event.key == pygame.K_BACKSPACE:
                        self.console_text = self.console_text[:-1]
                    elif event.key == pygame.K_ESCAPE:
                        self.console_active = False
                        self.console_text = ""
                    else:
                        self.console_text += event.unicode
                else:
                    if event.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_w):
                        self.player.jump()
                    elif event.key == pygame.K_SLASH or event.key == pygame.K_BACKQUOTE:
                        self.console_active = True
                        self.console_text = "/"
                    elif event.key == pygame.K_F1:
                        self.debug_mode = not self.debug_mode
                        print(f"Debug mode toggled to {self.debug_mode}")
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
                        self.playing = False

    def execute_command(self, command):
        print(f"\nExecuting command: {command}")
        parts = command.strip().lstrip('/').split()
        if not parts: return
        cmd = parts[0].lower()
        args = parts[1:]
        if cmd == "tp":
            if len(args) == 2:
                try:
                    self.player.pos = vec(float(args[0]), float(args[1]))
                    self.player.vel = vec(0, 0)
                except ValueError: print("Usage: /tp <x> <y>")
            else: print("Usage: /tp <x> <y>")
        elif cmd == "power":
            if args and args[0].lower() in ["big", "super"]:
                self.player.set_power_level("big")
            else:
                self.player.set_power_level("small")

    def update(self):
        if not self.console_active:
            self.level.all_sprites.update()
            self.particle_group.update()
            self.camera.update(self.player)

    def draw(self):
        self.screen.fill(SKY_BLUE)
        self.screen.blit(self.level.background_layer, self.camera.apply_rect(self.level.rect))
        self.screen.blit(self.level.ground_layer, self.camera.apply_rect(self.level.ground_layer.get_rect()))
        self.screen.blit(self.level.tube_layer, self.camera.apply_rect(self.level.tube_layer.get_rect()))

        for sprite in self.level.all_sprites:
            self.screen.blit(sprite.image, self.camera.apply(sprite))

        for particle in self.particle_group:
            self.screen.blit(particle.image, self.camera.apply(particle))


        if self.debug_mode:
            pygame.draw.rect(self.screen, (255, 255, 0), self.camera.apply_rect(self.player.hitbox), 2)
            for sprite in self.solid_group:
                pygame.draw.rect(self.screen, DEBUG_COLOR, self.camera.apply_rect(sprite.hitbox), 1)

        if self.console_active:
            console_surf = pygame.Surface((SCREEN_WIDTH, 40))
            console_surf.set_alpha(180)
            console_surf.fill(BLACK)
            self.screen.blit(console_surf, (0, 0))
            text_surf = self.console_font.render(self.console_text, True, WHITE)
            self.screen.blit(text_surf, (5, 5))

        pygame.display.flip()

    # --- Sound Methods ---
    def load_sfx(self):
        if not self.audio_enabled:
            return
        try:
            self.sfx["jump"] = pygame.mixer.Sound(os.path.join(SOUNDS_PATH, "jump.mp3"))
            self.sfx["bump"] = pygame.mixer.Sound(os.path.join(SOUNDS_PATH, "bump.mp3"))
            self.sfx["brick_break"] = pygame.mixer.Sound(os.path.join(SOUNDS_PATH, "brick-smash.mp3"))
            self.sfx["powerup_appears"] = pygame.mixer.Sound(os.path.join(SOUNDS_PATH, "powerup-appears.mp3"))

        except pygame.error as e:
            print(f"Cannot load sound effect: {e}")

    def play_sfx(self, name):
        if not self.audio_enabled:
            return
        if name in self.sfx:
            self.sfx[name].play()

    def play_level_music(self):
        if not self.audio_enabled:
            print("Audio not enabled — skipping music playback.")
            return

        music_path = os.path.join(MUSIC_PATH, self.level.music_file)
        if os.path.exists(music_path):
            try:
                pygame.mixer.music.load(music_path)
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play(loops=-1)
            except pygame.error as e:
                print(f"Error playing music: {e}")
        else:
            print(f"Warning: Music file not found: {music_path}")

    def create_particles(self, rect, image):
        image_small = pygame.transform.scale(image, (16, 16))
        for x in range(image_small.get_width()):
            for y in range(image_small.get_height()):
                color = image_small.get_at((x, y))
                if color[3] > 0:
                    particle_x = rect.x + (x * SCALE)
                    particle_y = rect.y + (y * SCALE)
                    particle = Particle(particle_x, particle_y, color)
                    self.particle_group.add(particle)


class Camera:
    def __init__(self, width, height):
        self.camera = pygame.Rect(0, 0, width, height)
        self.width, self.height = width, height

    def apply(self, entity):
        return entity.rect.move(self.camera.topleft)

    def apply_rect(self, rect):
        return rect.move(self.camera.topleft)

    def update(self, target):
        x = -target.hitbox.centerx + int(SCREEN_WIDTH / 2.5)
        y = -target.hitbox.centery + int(SCREEN_HEIGHT * 0.7)
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
