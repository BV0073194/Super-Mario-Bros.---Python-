# main.py
# Final Major Refactor: Data-Driven Level Generation, Interactive Blocks, and Corrected Physics

import pygame
import os
import random

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

# --- Data Map Color Keys ---
BREAKABLE_POS_COLORS = {
    (178, 67, 10, 255): 'above_ground',
    (0, 82, 87, 255): 'below_ground'
}


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

    def get_image_at(self, rect, colorkey=BLACK):
        image = pygame.Surface(rect.size).convert()
        image.blit(self.sheet, (0, 0), rect)
        image.set_colorkey(colorkey)
        return pygame.transform.scale(
            image, (rect.width * SCALE, rect.height * SCALE))


class Animation:

    def __init__(self,
                 sheet,
                 col,
                 row,
                 count,
                 duration,
                 layout='horizontal',
                 looped=True):
        self.frames = []
        if count > 0:
            self.duration = duration / count
        else:
            self.duration = float('inf')
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
                self.frame_index = (self.frame_index + 1) % len(
                    self.current.frames)
            else:
                self.frame_index = min(self.frame_index + 1,
                                       len(self.current.frames) - 1)

        if self.frame_index < len(self.current.frames):
            frame = self.current.frames[self.frame_index]
            self.player.image = pygame.transform.flip(
                frame, True,
                False) if self.player.direction == 'left' else frame


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
            self.rect.y = self.original_pos.y - 8 * (self.bump_offset / 4 -
                                                     (self.bump_offset / 4)**2)
            if self.bump_offset >= 8:
                self.rect.y = self.original_pos.y
                self.bumping = False
                self.post_bump()

    def hit(self):
        if not self.bumping:
            self.bumping = True
            self.bump_offset = 0

    def post_bump(self):
        pass


class Brick(Block):

    def post_bump(self):
        self.kill()


class LuckyBlock(Block):

    def __init__(self, x, y, image, used_image):
        super().__init__(x, y, image)
        self.used_image = used_image
        self.is_used = False

    def post_bump(self):
        if not self.is_used:
            self.image = self.used_image
            self.is_used = True


class Pipe(Platform):

    def __init__(self, x, y, w, h, number):
        super().__init__(x, y, w, h)
        self.number = number


class Level:

    def __init__(self, world_name):
        self.world_dir = os.path.join(WORLD_PATH_BASE, world_name)
        self.all_sprites = pygame.sprite.Group()
        self.solid_group = pygame.sprite.Group()
        self.goal_group = pygame.sprite.Group()
        self.pipes_group = pygame.sprite.Group()
        self.pipe_objects = {}
        self.debug_block_overlays = []

        self.load_layers()

        self.width = self.background_layer.get_width()
        self.height = self.background_layer.get_height()
        self.rect = self.background_layer.get_rect()

        self.spawn = vec(150, 400)
        self.find_spawn_pos()

        self.teleport_map = {2: 7, 7: 2}

    def load_layers(self):
        print("Loading level from data maps...")
        self.background_layer = self._load_visual_layer("BackgroundData.png")
        self.ground_layer = self._load_visual_layer("GroundData.png")
        self.tube_layer = self._load_visual_layer("TubeData.png")

        self._create_collision_from_map(
            os.path.join(self.world_dir, "GroundData.png"))
        self._create_pipes(os.path.join(self.world_dir, "TubeData.png"))

        self._create_blocks()
        self._create_goal()

    def _load_visual_layer(self, filename):
        path = os.path.join(self.world_dir, filename)
        image = pygame.image.load(path).convert_alpha()
        return pygame.transform.scale(
            image, (image.get_width() * SCALE, image.get_height() * SCALE))

    def _create_collision_from_map(self, path):
        image_map = pygame.image.load(path).convert_alpha()
        map_w, map_h = image_map.get_size()
        for y in range(map_h):
            x = 0
            while x < map_w:
                if image_map.get_at((x, y))[3] > 0:
                    start_x = x
                    while x < map_w and image_map.get_at((x, y))[3] > 0:
                        x += 1
                    w = (x - start_x) * SCALE
                    platform = Platform(start_x * SCALE, y * SCALE, w, SCALE)
                    self.solid_group.add(platform)
                else:
                    x += 1

    def _create_pipes(self, path):
        image_map = pygame.image.load(path).convert_alpha()
        map_w, map_h = image_map.get_size()
        pipe_number = 1
        processed_coords = set()

        for y in range(map_h):
            for x in range(map_w):
                if image_map.get_at(
                    (x, y))[3] > 0 and (x, y) not in processed_coords:
                    min_x, max_x, min_y, max_y = x, x, y, y
                    q = [(x, y)]
                    visited = set([(x, y)])

                    head = 0
                    while head < len(q):
                        px, py = q[head]
                        head += 1
                        processed_coords.add((px, py))
                        min_x, max_x = min(min_x, px), max(max_x, px)
                        min_y, max_y = min(min_y, py), max(max_y, py)

                        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < map_w and 0 <= ny < map_h and \
                               (nx, ny) not in visited and image_map.get_at((nx, ny))[3] > 0:
                                q.append((nx, ny))
                                visited.add((nx, ny))

                    pipe_w = (max_x - min_x + 1) * SCALE
                    pipe_h = (max_y - min_y + 1) * SCALE
                    pipe = Pipe(min_x * SCALE, min_y * SCALE, pipe_w, pipe_h,
                                pipe_number)
                    self.solid_group.add(pipe)
                    self.pipes_group.add(pipe)
                    self.pipe_objects[pipe_number] = pipe
                    pipe_number += 1

    def _create_blocks(self):
        print("--- Placing Interactive Blocks ---")
        breakable_map_path = os.path.join(self.world_dir,
                                          "BreakablePosData.png")
        breakable_map = pygame.image.load(breakable_map_path).convert_alpha()

        brick_ag_img = pygame.transform.scale(
            pygame.image.load(os.path.join(BLOCK_PATH,
                                           "BreakableBlock_AG.png")).convert(),
            (TILE_SIZE, TILE_SIZE))
        brick_bg_img = pygame.transform.scale(
            pygame.image.load(os.path.join(BLOCK_PATH,
                                           "BreakableBlock_BG.png")).convert(),
            (TILE_SIZE, TILE_SIZE))

        map_w, map_h = breakable_map.get_size()
        for y in range(map_h):
            for x in range(map_w):
                breakable_color = tuple(breakable_map.get_at((x, y)))
                if breakable_color in BREAKABLE_POS_COLORS:
                    pixel_x = x * TILE_SIZE
                    pixel_y = y * TILE_SIZE
                    block_type = BREAKABLE_POS_COLORS[breakable_color]
                    img = brick_ag_img if block_type == 'above_ground' else brick_bg_img
                    block = Brick(pixel_x, pixel_y, img)
                    self.all_sprites.add(block)
                    self.solid_group.add(block)
                    self.debug_block_overlays.append(('breakable', block.rect))
                    print(
                        f"Placed Breakable Block at World Coords: ({pixel_x}, {pixel_y})"
                    )

    def _create_goal(self):
        goal_map_path = os.path.join(self.world_dir, "GoalData.png")
        goal_map = pygame.image.load(goal_map_path).convert_alpha()
        map_w, map_h = goal_map.get_size()
        for y in range(map_h):
            for x in range(map_w):
                if goal_map.get_at((x, y))[3] > 0:
                    self.goal_group.add(
                        Platform(x * TILE_SIZE, y * TILE_SIZE, TILE_SIZE,
                                 TILE_SIZE))

    def find_spawn_pos(self):
        if self.solid_group:
            start_platform = min(self.solid_group.sprites(),
                                 key=lambda s: (s.rect.left, s.rect.top))
            self.spawn = vec(start_platform.rect.left + 48,
                             start_platform.rect.top)


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
        if keys[pygame.K_LEFT]:
            self.acc.x = -PLAYER_ACC
            self.direction = 'left'
        if keys[pygame.K_RIGHT]:
            self.acc.x = PLAYER_ACC
            self.direction = 'right'

        self.acc.x += self.vel.x * PLAYER_FRICTION
        self.vel += self.acc
        if abs(self.vel.x) < 0.1: self.vel.x = 0
        if abs(self.vel.x) > MAX_RUN_SPEED:
            self.vel.x = MAX_RUN_SPEED * (1 if self.vel.x > 0 else -1)

        self.pos.x += self.vel.x
        self.rect.centerx = round(self.pos.x)
        self.check_collisions('horizontal')

        self.pos.y += self.vel.y
        self.rect.centery = round(self.pos.y)
        self.check_collisions('vertical')

        self.animator.update()
        self.set_state()

        if self.rect.top > self.game.level.rect.height:
            print("Player died: fell off map")
            self.game.playing = False

    def check_collisions(self, direction):
        hits = pygame.sprite.spritecollide(self, self.game.level.solid_group,
                                           False)
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
                    if isinstance(hit, Block): hit.hit()
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
        self.debug_mode = False
        self.console_active = False
        self.console_text = ""
        self.console_font = pygame.font.Font(None, 36)

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
            if event.type == pygame.QUIT: self.running = False
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
                    if event.key in (pygame.K_SPACE, pygame.K_UP):
                        self.player.jump()
                    elif event.key == pygame.K_SLASH:
                        self.console_active = True
                        self.console_text = "/"

    def execute_command(self, command):
        print(f"\nExecuting command: {command}")
        parts = command.strip().split()
        if not parts: return
        cmd = parts[0].lower()
        if cmd == "/tp":
            if len(parts) == 3:
                try:
                    x, y = float(parts[1]), float(parts[2])
                    self.player.pos = vec(x, y)
                    self.player.vel = vec(0, 0)
                except ValueError:
                    print("Invalid coordinates.")
            else:
                print("Usage: /tp <x> <y>")
        elif cmd == "/debug":
            if len(parts) == 2:
                self.debug_mode = parts[1].lower() in ['true', '1', 'on']
                print(f"Debug mode set to {self.debug_mode}")
            else:
                print("Usage: /debug <true|false>")

    def update(self):
        if not self.console_active:
            self.level.all_sprites.update()
            self.camera.update(self.player)

    def draw(self):
        self.screen.fill(SKY_BLUE)
        self.screen.blit(
            self.level.background_layer,
            self.camera.apply_rect(self.level.background_layer.get_rect()))
        self.screen.blit(
            self.level.ground_layer,
            self.camera.apply_rect(self.level.ground_layer.get_rect()))
        self.screen.blit(
            self.level.tube_layer,
            self.camera.apply_rect(self.level.tube_layer.get_rect()))

        for sprite in self.level.all_sprites:
            self.screen.blit(sprite.image, self.camera.apply(sprite))

        if self.debug_mode:
            for sprite in self.level.solid_group:
                pygame.draw.rect(self.screen, DEBUG_COLOR,
                                 self.camera.apply_rect(sprite.rect), 1)
            for sprite in self.level.goal_group:
                pygame.draw.rect(self.screen, (0, 255, 0),
                                 self.camera.apply_rect(sprite.rect), 1)
            for pipe in self.level.pipes_group:
                num_surf = self.console_font.render(str(pipe.number), True,
                                                    WHITE)
                self.screen.blit(num_surf, self.camera.apply_rect(pipe.rect))
            for tag, rect in self.level.debug_block_overlays:
                color = (255, 100, 100)  # Reddish for breakable
                debug_rect = self.camera.apply_rect(rect)
                pygame.draw.rect(self.screen, color, debug_rect, 2)

        if self.console_active:
            console_surf = pygame.Surface((SCREEN_WIDTH, 40))
            console_surf.set_alpha(180)
            console_surf.fill(BLACK)
            self.screen.blit(console_surf, (0, 0))
            text_surf = self.console_font.render(self.console_text, True,
                                                 WHITE)
            self.screen.blit(text_surf, (5, 5))

        pygame.display.flip()


class Camera:

    def __init__(self, width, height):
        self.camera = pygame.Rect(0, 0, width, height)
        self.width, self.height = width, height
        self.ground_level_y = None

    def apply(self, entity):
        return entity.rect.move(self.camera.topleft)

    def apply_rect(self, rect):
        return rect.move(self.camera.topleft)

    def update(self, target):
        x = -target.rect.centerx + SCREEN_WIDTH // 2
        y = -target.rect.centery + SCREEN_HEIGHT // 2

        if self.ground_level_y is None:
            self.ground_level_y = y

        x = min(0, x)
        x = max(-(self.width - SCREEN_WIDTH), x)

        y = min(y, self.ground_level_y)
        y = max(y, -(self.height - SCREEN_HEIGHT))

        self.camera = pygame.Rect(x, y, self.width, self.height)


if __name__ == "__main__":
    g = Game()
    while g.running:
        g.new_game()
    pygame.quit()
