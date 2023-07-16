import datetime
import os
import time as timer
from copy import deepcopy
from random import choice, randrange
from typing import List

import cv2
import numpy as np
import pygame
from matplotlib import pyplot as plt

from models.data_models import ActionZones, DetectionPoint, Circle, PreviousActions
from models.pose_detection import PoseDetection
from utils.camera import Camera
from utils.display import draw_zones_and_status, add_circle_in_image


pose_detection = PoseDetection()
action_zones = ActionZones(left=Circle(center=DetectionPoint((860, 360)), radius=100),
                           right=Circle(center=DetectionPoint((100, 360)), radius=100),
                           rotate=Circle(center=DetectionPoint((480, 100)), radius=100),
                           down=Circle(center=DetectionPoint((480, 620)), radius=100))

cam = Camera(0)

GAME_WIDTH, GAME_HEIGHT = 10, 20
TILE = 45
GAME_RES = GAME_WIDTH * TILE + 1000, GAME_HEIGHT * TILE + 200
RES = 750, 940
FPS = 60
TIMER_DURATION = 180

POSITION_WEBCAM = (1200, 540)
POSITION_TITLE = (760, 20)

POSITION_LAST_SCORE_TEXT = (690, 800)
POSITION_LAST_SCORE_VALUE = (990, 800)

POSITION_SCORE_TEXT = (740, 920)
POSITION_SCORE_VALUE = (950, 920)

POSITION_RECORD_TEXT = (1410, 100)
POSITION_RECORD_VALUE = (1610, 100)

POSITION_TIMER_TEXT = (1410, 920)
POSITION_TIMER_VALUE = (1580, 920)

pygame.init()

last_score = 0

pygame.display.set_caption('Tetris AI')

screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
game_screen = pygame.Surface(GAME_RES, pygame.SRCALPHA, 32)
clock = pygame.time.Clock()

GRID = [pygame.Rect(x * TILE, y * TILE, TILE, TILE) for x in range(GAME_WIDTH) for y in range(GAME_HEIGHT)]

BLOCK_SHAPES = [[(-1, 0), (-2, 0), (0, 0), (1, 0)],
                [(0, -1), (-1, -1), (-1, 0), (0, 0)],
                [(-1, 0), (-1, 1), (0, 0), (0, -1)],
                [(0, 0), (-1, 0), (0, 1), (-1, -1)],
                [(0, 0), (0, -1), (0, 1), (-1, -1)],
                [(0, 0), (0, -1), (0, 1), (1, -1)],
                [(0, 0), (0, -1), (0, 1), (-1, 0)]]
MAX_SIZE_BLOCK = 4

BLOCKS = [[pygame.Rect(x + GAME_WIDTH // 2, y + 1, 1, 1) for x, y in block_shape] for block_shape in BLOCK_SHAPES]
figure_rect = pygame.Rect(0, 0, TILE - 2, TILE - 2)
play_field = [[0 for i in range(GAME_WIDTH)] for j in range(GAME_HEIGHT)]

INIT_ANIM_COUNT, INIT_ANIM_SPEED, INIT_ANIM_LIMIT = 0, 120, 2000
anim_count, anim_speed, anim_limit = INIT_ANIM_COUNT, INIT_ANIM_SPEED, INIT_ANIM_LIMIT

BACKGROUND_IMAGE = pygame.image.load('img/bg-full.jpg').convert()
GAME_BACKGROUND_IMAGE = pygame.image.load('img/bg2.jpg').convert()
POWERED_IMAGE = pygame.image.load('img/powered.png')
POWERED_IMAGE.set_colorkey((0, 0, 0))
width_powered = POWERED_IMAGE.get_rect().width
height_powered = POWERED_IMAGE.get_rect().height
POWERED_IMAGE = pygame.transform.scale(POWERED_IMAGE, (width_powered / 4, height_powered / 4))
POWERED_IMAGE = pygame.Surface.convert_alpha(POWERED_IMAGE)

font = pygame.font.Font('font/font.ttf', 45)
small_font = pygame.font.Font('font/font.ttf', 20)
main_font = pygame.font.Font('font/font.ttf', 65)

TITLE_TEXT = main_font.render('TETRIS-AI', True, pygame.Color('deeppink'))
SCORE_TEXT = font.render('score:', True, pygame.Color('green'))
RECORD_TEXT = font.render('record:', True, pygame.Color('purple'))
TIMER_TEXT = font.render('time:', True, pygame.Color('violet'))
TIMER_SEC = font.render(str(TIMER_DURATION), True, pygame.Color('burlywood1'))
LAST_SCORE_TEXT = font.render('last score:', True, pygame.Color('deepskyblue1'))
LAST_SCORE_VALUE = font.render(str(last_score), True, pygame.Color('darkorange'))

get_color = lambda: (randrange(30, 256), randrange(30, 256), randrange(30, 256))

current_block, next_block = deepcopy(choice(BLOCKS)), deepcopy(choice(BLOCKS))
color, next_color = get_color(), get_color()

score, lines = 0, 0
scores = {0: 0, 1: 100, 2: 300, 3: 700, 4: 1500}

today_datetime = datetime.datetime.now()
record_filename = today_datetime.strftime('%m-%d-%Y')


def is_block_in_play_field(position: pygame.Rect, field: List[List[int]], game_width: int, game_height: int) -> bool:
    if position.x < 0 or position.x > game_width - 1:
        return False
    if position.y > game_height - 1 or field[position.y][position.x]:
        return False
    return True


def get_record():
    try:
        with open(record_filename) as f:
            best_score = 0
            for li in f:
                if li.strip():
                    current_score = int(li.strip().rsplit(' ', 3)[-2])
                    if current_score > best_score:
                        best_score = current_score
            return str(best_score)
    except FileNotFoundError:
        return "0"


def get_top_records():
    with open(record_filename, 'r') as file:
        data = []
        for line in file:
            if line.strip():
                parts = line.strip().rsplit(' ', 3)
                result = int(parts[-2])
                name = ' '.join(parts[:-2])
                time = str(parts[-1])
                data.append((name, result, time))
        data.sort(key=lambda x: x[1], reverse=True)
        top10 = data[:10] if len(data) >= 10 else data
        records = [(name[:14], str(result), time) for name, result, time in top10]
    return records


def set_record(score, firstname, lastname):
    with open(record_filename, 'a') as f:
        now = datetime.datetime.now()
        hour = str(now.hour)
        minute = str(now.minute)
        time = hour + ":" + minute
        f.write(firstname + " " + lastname + " " + str(score) + " " + str(time) + "\n")


index = 0
list_images = ["img/1.jpg", "img/2.jpg", "img/3.jpg", "img/4.jpg", "img/5.jpg"]

image_left = pygame.image.load('img/left.png').convert_alpha()
image_left = pygame.transform.scale(image_left, (50, 50))
previous_positions_status = PreviousActions(left=False, right=False, rotate=False, down=False,
                                            action_zones=action_zones, anim_limit=anim_limit)

image_right = pygame.image.load('img/right.png').convert_alpha()
image_right = pygame.transform.scale(image_right, (50, 50))

end_time = None
color_inactive = pygame.Color('white')
color_active = pygame.Color('deeppink')

# Firstname
firstname = ''
color_box_firstname = color_inactive
active_firstname = False
input_box_firstname = pygame.Rect(1270, 420, 10, 80)

# Lastname
lastname = ''
color_box_lastname = color_inactive
active_lastname = False
input_box_lastname = pygame.Rect(1270, 620, 10, 80)

# Start button
color_box_button_start = color_inactive
text_button_start = font.render("Start", True, pygame.Color('white'))
text_rect_start = text_button_start.get_rect()
button_start_x = (1920 / 2) + 500 - (text_button_start.get_width() / 2)
button_start_y = (1080 / 2) - (text_button_start.get_height() / 2) + 300
button_start = pygame.Rect(button_start_x, button_start_y, text_rect_start.width + 10, text_rect_start.height + 10)

# End button
color_box_button_end = color_active
text_button_end = font.render("End", True, pygame.Color('white'))
text_rect_end = text_button_end.get_rect()
button_end_x = 1800
button_end_y = 1000
button_end = pygame.Rect(button_end_x, button_end_y, text_rect_end.width + 10, text_rect_end.height + 10)

# Restart button
color_box_button_restart = color_active
text_button_restart = font.render("New game", True, pygame.Color('white'))
text_rect_restart = text_button_restart.get_rect()
button_restart_x = (1920 / 2) - (text_button_restart.get_width() / 2) + 600
button_restart_y = (1080 / 2) - (text_button_restart.get_height() / 2) + 150
button_restart = pygame.Rect(button_restart_x, button_restart_y, text_rect_restart.width + 10,
                             text_rect_restart.height + 10)

# Left button
left_button = pygame.Rect(330.5, 120, 60, 60)

# Right button
right_button = pygame.Rect(609.5, 120, 60, 60)

is_game_over = False
is_running = False

video = cv2.VideoCapture("img/intro.mp4")
success, video_image = video.read()
fps = video.get(cv2.CAP_PROP_FPS)
start_time_sleep = timer.time()
end_time_sleep = timer.time()
counter_frames = 4
while True:
    events = pygame.event.get()
    if len(events) == 0 and (not is_running or is_game_over):
        end_time_sleep = timer.time()
    for event in events:
        if event.type == pygame.QUIT:
            start_time_sleep = timer.time()
            end_time_sleep = timer.time()
            cam.stop()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            start_time_sleep = timer.time()
            end_time_sleep = timer.time()
            if input_box_firstname.collidepoint(event.pos):
                active_firstname = not active_firstname
            else:
                active_firstname = False
            if input_box_lastname.collidepoint(event.pos):
                active_lastname = not active_lastname
            else:
                active_lastname = False
            if button_start.collidepoint(event.pos) and not is_running and not is_game_over:
                if firstname and lastname:
                    is_running = True
            if button_end.collidepoint(event.pos) and is_running and not is_game_over:
                # Reset
                firstname = ''
                lastname = ''
                is_running = False
                active_lastname = False
                active_firstname = False
                end_time = None
                is_game_over = False
                play_field = [[0 for _ in range(GAME_WIDTH)] for _ in range(GAME_HEIGHT)]
                current_block, next_block = deepcopy(choice(BLOCKS)), deepcopy(choice(BLOCKS))
                anim_count, anim_speed, previous_positions_status.anim_limit = INIT_ANIM_COUNT, INIT_ANIM_SPEED, INIT_ANIM_LIMIT
            if button_restart.collidepoint(event.pos) and is_game_over:
                # Reset
                firstname = ''
                lastname = ''
                is_running = False
                active_lastname = False
                active_firstname = False
                end_time = None
                is_game_over = False
            if left_button.collidepoint(event.pos) and not is_running and not is_game_over:
                index -= 1
                if index < 0:
                    index = 4
            if right_button.collidepoint(event.pos) and not is_running and not is_game_over:
                index += 1
                if index > 4:
                    index = 0
        elif event.type == pygame.KEYDOWN:
            start_time_sleep = timer.time()
            end_time_sleep = timer.time()
            if event.key == pygame.K_RETURN:
                if firstname:
                    active_firstname = False
                if lastname:
                    active_lastname = False
            if active_firstname:
                if event.key == pygame.K_BACKSPACE:
                    firstname = firstname[:-1]
                else:
                    firstname += event.unicode.lower()
            if active_lastname:
                if event.key == pygame.K_BACKSPACE:
                    lastname = lastname[:-1]
                else:
                    lastname += event.unicode.lower()
        color_box_firstname = color_active if active_firstname else color_inactive
        color_box_lastname = color_active if active_lastname else color_inactive
        color_box_button_start = color_active if (firstname and lastname) else pygame.Color('grey')
    if end_time_sleep - start_time_sleep > 30:
        if counter_frames % 4 == 0:
            success, video_image = video.read()
            while not success:
                video = cv2.VideoCapture("img/intro.mp4")
                success, video_image = video.read()
        video_surf = pygame.image.frombuffer(video_image.tobytes(), video_image.shape[1::-1], "BGR")
        screen.blit(video_surf, (0, 0))
        counter_frames += 1
    elif is_game_over:
        screen.blit(BACKGROUND_IMAGE, (0, 0))
        txt_surface_game_over = font.render("End of game", True, pygame.Color('white'))
        x_position = (1920 / 2) - (txt_surface_game_over.get_width() / 2) + 600
        y_position = (1080 / 2) - (txt_surface_game_over.get_height() / 2) - 150
        screen.blit(txt_surface_game_over, (x_position, y_position))
        txt_score = font.render("Score: " + str(last_score), True, pygame.Color('white'))
        x_position_score = (1920 / 2) - (txt_score.get_width() / 2) + 600
        y_position_score = (1080 / 2) - (txt_score.get_height() / 2) - 50
        screen.blit(txt_score, (x_position_score, y_position_score))
        pygame.draw.rect(screen, color_box_button_restart, button_restart, border_radius=10)
        screen.blit(text_button_restart, (button_restart_x + 10, button_restart_y + 10))

        y = 200
        rank = 1
        color_map = plt.get_cmap('tab20b')
        colors = [color_map(i)[:3] for i in np.linspace(0, 1, 20)]
        for name, result, time in get_top_records():
            color = colors[rank % len(colors)]
            color = [i * 255 for i in color]
            text_rank = font.render(str(rank) + ".", True, color)
            text_name = font.render(str(name), True, color)
            text_score = font.render(str(result), True, color)
            text_time = font.render(str(time), True, color)
            screen.blit(text_rank, (200, y))
            screen.blit(text_name, (280, y))
            screen.blit(text_score, (800, y))
            screen.blit(text_time, (1100, y))
            y += 70
            rank += 1
        player_text = font.render('#   Player', True, pygame.Color('white'))
        screen.blit(player_text, (200, 100))
        score_text = font.render('Score', True, pygame.Color('white'))
        screen.blit(score_text, (800, 100))
        time_text = font.render('Time', True, pygame.Color('white'))
        screen.blit(time_text, (1100, 100))
        pygame.draw.line(screen, pygame.Color('white'), (200, 170), (1220, 170), 5)
        screen.blit(POWERED_IMAGE, (20, 960))
    elif not is_running:
        firstname_label = font.render("Firstname", True, pygame.Color('white'))
        lastname_label = font.render("Lastname", True, pygame.Color('white'))
        txt_surface_firstname = font.render(firstname, True, pygame.Color('white'))
        txt_surface_lastname = font.render(lastname, True, pygame.Color('white'))
        txt_surface_gdpr = small_font.render("This will be used to contact you in case you win", True,
                                             pygame.Color('white'))
        txt_tutorial = font.render("Tutorial", True, pygame.Color('white'))
        width_firstname = max(400, txt_surface_firstname.get_width() + 15)
        width_lastname = max(400, txt_surface_lastname.get_width() + 15)
        input_box_firstname.w = width_firstname
        input_box_lastname.w = width_lastname

        screen.blit(BACKGROUND_IMAGE, (0, 0))
        screen.blit(firstname_label, (input_box_firstname.x, input_box_firstname.y - 60))
        screen.blit(lastname_label, (input_box_lastname.x, input_box_lastname.y - 60))
        screen.blit(txt_surface_firstname, (input_box_firstname.x + 10, input_box_firstname.y + 10))
        screen.blit(txt_surface_lastname, (input_box_lastname.x + 10, input_box_lastname.y + 10))
        screen.blit(txt_surface_gdpr, (1460 - txt_surface_gdpr.get_width() / 2, input_box_lastname.y + 120))
        screen.blit(txt_tutorial, (20 + 960 / 2 - txt_tutorial.get_width() / 2, 120))
        pygame.draw.rect(screen, color_box_firstname, input_box_firstname, 5, 3)
        pygame.draw.rect(screen, color_box_lastname, input_box_lastname, 5, 3)
        pygame.draw.rect(screen, color_box_button_start, button_start, border_radius=10)
        screen.blit(text_button_start, (button_start_x + 10, button_start_y + 10))

        slider_image = pygame.image.load(list_images[index])
        slider_image = pygame.transform.scale(slider_image, (960, 720))
        slider_image = pygame.Surface.convert_alpha(slider_image)
        screen.blit(slider_image, (30, 215))
        pygame.draw.rect(screen, color_active, left_button)
        pygame.draw.rect(screen, color_active, right_button)
        screen.blit(image_left, image_left.get_rect(center=left_button.center))
        screen.blit(image_right, image_right.get_rect(center=right_button.center))
        screen.blit(TITLE_TEXT, (1280, 120))
        screen.blit(POWERED_IMAGE, (20, 960))
    elif is_running:
        start_time_sleep = timer.time()
        end_time_sleep = timer.time()
        screen.blit(BACKGROUND_IMAGE, (0, 0))
        if not end_time:
            end_time = datetime.datetime.now() + datetime.timedelta(seconds=TIMER_DURATION)
        today_datetime = datetime.datetime.now()
        record_filename = today_datetime.strftime('%m-%d-%Y')
        if not os.path.isfile(record_filename):
            f = open(record_filename, "x")
            f.close()
        record = get_record()

        screen.blit(game_screen, (20, 20))
        game_screen.blit(GAME_BACKGROUND_IMAGE, (0, 0))

        # delay for full lines
        for i in range(lines):
            pygame.time.wait(200)

        camera_image = cam.read()
        camera_image = cv2.resize(camera_image, (960, 720))
        pose_detected = pose_detection.predict_human_poses(camera_image)
        add_circle_in_image(pose_detected, camera_image)
        previous_positions_status.update_actions(pose_detected)
        draw_zones_and_status(action_zones, camera_image, previous_positions_status)
        camera_image = cv2.rotate(camera_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)
        camera_image = cv2.resize(camera_image, (720, 960))
        width_camera = camera_image.shape[0]
        height_camera = camera_image.shape[1]
        # move x
        if previous_positions_status.dx_action:
            figure_old = deepcopy(current_block)
            for i in range(MAX_SIZE_BLOCK):
                current_block[i].x += previous_positions_status.dx_action
                if not is_block_in_play_field(current_block[i], play_field, GAME_WIDTH, GAME_HEIGHT):
                    current_block = deepcopy(figure_old)
                    break

        # move y
        anim_count += anim_speed
        if anim_count > previous_positions_status.anim_limit:
            anim_count = 0
            figure_old = deepcopy(current_block)
            for i in range(MAX_SIZE_BLOCK):
                current_block[i].y += 1
                if not is_block_in_play_field(current_block[i], play_field, GAME_WIDTH, GAME_HEIGHT):
                    for j in range(MAX_SIZE_BLOCK):
                        play_field[figure_old[j].y][figure_old[j].x] = color
                    current_block, color = next_block, next_color
                    next_block, next_color = deepcopy(choice(BLOCKS)), get_color()
                    previous_positions_status.anim_limit = 2000
                    previous_positions_status.down = False
                    previous_positions_status.can_go_down = False
                    break

        # rotate
        if previous_positions_status.rotate_action:
            center = current_block[0]
            figure_old = deepcopy(current_block)
            for i in range(MAX_SIZE_BLOCK):
                x = current_block[i].y - center.y
                y = current_block[i].x - center.x
                current_block[i].x = center.x - x
                current_block[i].y = center.y + y
                if not is_block_in_play_field(current_block[i], play_field, GAME_WIDTH, GAME_HEIGHT):
                    current_block = deepcopy(figure_old)
                    break

        # check lines
        line, lines = GAME_HEIGHT - 1, 0
        for row in range(GAME_HEIGHT - 1, -1, -1):
            count = 0
            for i in range(GAME_WIDTH):
                if play_field[row][i]:
                    count += 1
                play_field[line][i] = play_field[row][i]
            if count < GAME_WIDTH:
                line -= 1
            else:
                anim_speed += 3
                lines += 1

        # compute score
        score += scores[lines]

        # draw grid
        [pygame.draw.rect(game_screen, (200, 200, 200), i_rect, 1) for i_rect in GRID]

        # draw play field
        for y, raw in enumerate(play_field):
            for x, col in enumerate(raw):
                if col:
                    figure_rect.x, figure_rect.y = x * TILE, y * TILE
                    pygame.draw.rect(game_screen, col, figure_rect)

        for i in range(MAX_SIZE_BLOCK):
            figure_rect.x = current_block[i].x * TILE
            figure_rect.y = current_block[i].y * TILE
            pygame.draw.rect(game_screen, color, figure_rect)
            figure_rect.x = next_block[i].x * TILE + 380
            figure_rect.y = next_block[i].y * TILE + 185
            pygame.draw.rect(screen, next_color, figure_rect)
        remaining_seconds_time = (end_time - datetime.datetime.now()).total_seconds()
        TIMER_SEC = font.render("%02d" % remaining_seconds_time, True, (255, 255, 255))

        screen.blit(TITLE_TEXT, POSITION_TITLE)
        screen.blit(SCORE_TEXT, POSITION_SCORE_TEXT)
        screen.blit(font.render(str(score), True, pygame.Color('white')), POSITION_SCORE_VALUE)
        screen.blit(RECORD_TEXT, POSITION_RECORD_TEXT)
        screen.blit(font.render(record, True, pygame.Color('gold')), POSITION_RECORD_VALUE)
        screen.blit(pygame.surfarray.make_surface(camera_image),
                    (POSITION_WEBCAM[0] - width_camera / 2, POSITION_WEBCAM[1] - height_camera / 2))
        screen.blit(TIMER_SEC, POSITION_TIMER_VALUE)
        screen.blit(TIMER_TEXT, POSITION_TIMER_TEXT)
        pygame.draw.rect(screen, color_box_button_end, button_end, border_radius=10)
        screen.blit(text_button_end, (button_end_x + 10, button_end_y + 10))

        # game over
        for i in range(GAME_WIDTH):
            if play_field[0][i] or remaining_seconds_time < 0:
                set_record(score, firstname, lastname)
                play_field = [[0 for _ in range(GAME_WIDTH)] for _ in range(GAME_HEIGHT)]
                current_block, next_block = deepcopy(choice(BLOCKS)), deepcopy(choice(BLOCKS))
                anim_count, anim_speed, previous_positions_status.anim_limit = INIT_ANIM_COUNT, INIT_ANIM_SPEED, INIT_ANIM_LIMIT
                for i_rect in GRID:
                    pygame.draw.rect(game_screen, get_color(), i_rect)
                    screen.blit(game_screen, (20, 20))
                    pygame.display.flip()
                    clock.tick(200)
                end_time = None
                remaining_seconds_time = TIMER_DURATION
                last_score = score
                score = 0
                is_game_over = True
        screen.blit(POWERED_IMAGE, (20, 960))

    pygame.display.flip()
    clock.tick(FPS)
