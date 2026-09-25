"""TinyJev in ViZDoom's Deadly Corridor, inspired by OpenDecision's Doom demo.

The tactical router and aiming actuator are disclosed and deterministic. TinyJev sees
structured telemetry and chooses between the two tactics available in the current mode.
It never receives pixels.

    pip install 'tinyjev[mlx,doom]'
    python demos/doom_corridor.py --gif tinyjev_doom_linkedin.gif
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import vizdoom as vzd

import tinyjev


ROOT = Path(__file__).resolve().parents[1]
REGULAR = ROOT / "assets" / "Poppins-Regular.ttf"
MEDIUM = ROOT / "assets" / "Poppins-Medium.ttf"
BOLD = ROOT / "assets" / "Poppins-Bold.ttf"

BG, INK, MUTED = "#F4F0E6", "#232220", "#6B675E"
RULE, TRACK, ORANGE, PANEL = "#D8D1C2", "#D5CEC0", "#E85D04", "#EBE5D8"
ENEMIES = frozenset({"ChaingunGuy", "ShotgunGuy", "Zombieman"})

MODE_ACTIONS = {
    "engage": {
        "attack": "Aim at the selected nearby enemy and fire",
        "shoot_retreat": "Fire while backing away from the selected enemy",
    },
    "approach": {
        "advance_fire": "Close on the selected enemy while aiming and firing",
        "hold_fire": "Hold position while aiming and firing",
    },
    "evade": {
        "strafe_fire": "Fire while dodging sideways away from the attacker",
        "shoot_retreat": "Fire while backing away from the attacker",
    },
    "critical_evade": {
        "retreat_fire": "Retreat while firing to preserve critical health",
        "strafe_fire": "Keep strafing sideways while firing",
    },
    "advance": {
        "move_to_goal": "Aim toward the armor goal and move forward",
        "wait": "Hold position only if the goal cannot be located",
    },
}


def ff(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


F = {
    "title": ff(BOLD, 40), "subtitle": ff(REGULAR, 18), "eyebrow": ff(MEDIUM, 13),
    "body": ff(REGULAR, 17), "bold": ff(BOLD, 17), "action": ff(BOLD, 25),
    "metric": ff(MEDIUM, 14), "small": ff(REGULAR, 13),
}


@dataclass(frozen=True)
class Target:
    name: str
    angle: float
    distance: float


@dataclass(frozen=True)
class Snapshot:
    health: int
    ammo: int
    kills: int
    target: Target | None
    enemies: int
    goal_angle: float | None
    goal_distance: float | None
    taking_damage: bool


def norm_angle(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def snapshot(state: vzd.GameState, previous_health: int | None) -> Snapshot:
    health, ammo, px, py, player_angle, kills = state.game_variables
    targets: list[Target] = []
    goal_angle = goal_distance = None
    for obj in state.objects or ():
        dx, dy = obj.position_x - px, obj.position_y - py
        distance = math.hypot(dx, dy)
        angle = norm_angle(math.degrees(math.atan2(dy, dx)) - player_angle)
        if obj.name in ENEMIES:
            targets.append(Target(obj.name, angle, distance))
        elif obj.name == "GreenArmor":
            goal_angle, goal_distance = angle, distance
    targets.sort(key=lambda item: item.distance)
    h = max(0, round(health))
    return Snapshot(h, max(0, round(ammo)), max(0, round(kills)), targets[0] if targets else None,
                    len(targets), goal_angle, goal_distance,
                    previous_health is not None and h < previous_health)


def route(s: Snapshot) -> str:
    if s.health <= 40 and s.target and s.target.distance <= 450:
        return "critical_evade"
    if s.taking_damage:
        return "evade"
    if s.target and s.target.distance <= 300:
        return "engage"
    if s.target and s.target.distance <= 450:
        return "approach"
    return "advance"


def direction(angle: float | None) -> str:
    if angle is None:
        return "none"
    if angle < -3:
        return "right"
    if angle > 3:
        return "left"
    return "ahead"


def decision_state(s: Snapshot, mode: str) -> str:
    target_angle = "none" if s.target is None else f"{s.target.angle:+.1f} degrees"
    target_distance = "none" if s.target is None else f"{s.target.distance:.0f} map units"
    goal_angle = "none" if s.goal_angle is None else f"{s.goal_angle:+.1f} degrees"
    goal_distance = "none" if s.goal_distance is None else f"{s.goal_distance:.0f} map units"
    return (
        f"Health {s.health}/100. Ammo {s.ammo}. Kills {s.kills}. Enemies alive {s.enemies}. "
        f"Current target {s.target.name if s.target else 'none'}, {direction(s.target.angle if s.target else None)}, "
        f"angle {target_angle}, distance {target_distance}. Taking damage {'yes' if s.taking_damage else 'no'}. "
        f"Goal armor {direction(s.goal_angle)}, angle {goal_angle}, distance {goal_distance}. "
        f"Tactical mode {mode}."
    )


def decide(agent: tinyjev.Agent, s: Snapshot) -> dict:
    mode = route(s)
    result = agent.predict({
        "state": decision_state(s, mode),
        "questions": {"action": {
            "type": "choice",
            "instructions": f"Which immediate action best executes the selected {mode} tactical mode?",
            "criteria": MODE_ACTIONS[mode],
        }},
    })
    answer = result["states"][0]["answers"]["action"]
    return {"mode": mode, "action": answer["choice"], "probabilities": answer["probabilities"],
            "model_ms": result["execution"]["model_ms"]}


def action_vector(action: str, s: Snapshot, mode: str) -> list[float]:
    target_angle = s.goal_angle if mode == "advance" else (s.target.angle if s.target else 0.0)
    # TURN_LEFT_RIGHT_DELTA uses the opposite sign to the world-space angle.
    turn = -max(min(target_angle or 0.0, 45.0), -45.0)
    fire = action in {"attack", "shoot_retreat", "advance_fire", "hold_fire", "strafe_fire", "retreat_fire"}
    forward = action in {"advance_fire", "move_to_goal"}
    backward = action in {"shoot_retreat", "retreat_fire"}
    strafe_left = strafe_right = False
    if action == "strafe_fire":
        enemy_right = (s.target.angle if s.target else 0.0) < 0
        strafe_left, strafe_right = enemy_right, not enemy_right
    if action == "wait":
        turn = 0.0
    return [turn, float(fire), float(forward), float(strafe_left), float(strafe_right), float(backward)]


def setup_game(skill: int, seed: int) -> vzd.DoomGame:
    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / "deadly_corridor.cfg"))
    game.set_doom_game_path("")
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_objects_info_enabled(True)
    game.set_seed(seed)
    game.set_doom_skill(skill)
    game.set_episode_start_time(0)
    game.set_available_buttons([
        vzd.Button.TURN_LEFT_RIGHT_DELTA, vzd.Button.ATTACK, vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.MOVE_BACKWARD,
    ])
    game.set_button_max_value(vzd.Button.TURN_LEFT_RIGHT_DELTA, 180)
    game.set_available_game_variables([
        vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2, vzd.GameVariable.POSITION_X,
        vzd.GameVariable.POSITION_Y, vzd.GameVariable.ANGLE, vzd.GameVariable.KILLCOUNT,
    ])
    game.init()
    return game


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    scale = max(size[0] / image.width, size[1] / image.height)
    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left, top = (image.width - size[0]) // 2, (image.height - size[1]) // 2
    return image.crop((left, top, left + size[0], top + size[1]))


def render(rgb: np.ndarray, s: Snapshot, d: dict, number: int) -> Image.Image:
    canvas = Image.new("RGB", (1080, 1080), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 37), "TinyJev runs the Deadly Corridor", font=F["title"], fill=INK)
    draw.text((51, 91), "structured game state in · typed tactical decision out · no pixels", font=F["subtitle"], fill=MUTED)
    draw.line((50, 132, 1030, 132), fill=RULE, width=2)

    game = cover(Image.fromarray(rgb), (665, 499))
    canvas.paste(game, (50, 158))
    draw.rectangle((50, 158, 715, 657), outline=INK, width=2)
    draw.rectangle((68, 176, 225, 210), fill=INK)
    draw.text((80, 182), f"DECISION {number:03d}", font=F["eyebrow"], fill=BG)

    x0, y0, x1, y1 = 744, 158, 1030, 657
    draw.rounded_rectangle((x0, y0, x1, y1), radius=17, fill=PANEL)
    draw.text((768, 182), "TACTICAL ROUTER", font=F["eyebrow"], fill=MUTED)
    draw.text((768, 209), d["mode"].replace("_", " ").upper(), font=F["action"], fill=INK)
    draw.line((768, 251, 1006, 251), fill=RULE, width=2)
    draw.text((768, 276), "TINYJEV CHOICE", font=F["eyebrow"], fill=MUTED)
    draw.text((768, 304), d["action"].replace("_", " ").upper(), font=F["action"], fill=ORANGE)

    y = 358
    for key, probability in d["probabilities"].items():
        selected = key == d["action"]
        label_font, color = (F["bold"], INK) if selected else (F["body"], MUTED)
        if selected:
            draw.polygon([(768, y + 3), (778, y + 10), (768, y + 17)], fill=ORANGE)
        draw.text((786, y), key.replace("_", " "), font=label_font, fill=color)
        draw.text((1003, y), f"{probability:.2f}", font=label_font, fill=color, anchor="ra")
        draw.rounded_rectangle((786, y + 29, 1003, y + 41), radius=6, fill=TRACK)
        draw.rounded_rectangle((786, y + 29, 786 + round(217 * probability), y + 41), radius=6,
                               fill=ORANGE if selected else "#989184")
        y += 87
    draw.text((768, 563), "DETERMINISTIC ACTUATOR", font=F["eyebrow"], fill=MUTED)
    draw.text((768, 591), "aim + press keys", font=F["bold"], fill=INK)
    draw.text((1003, 621), f"{d['model_ms']:.0f} ms", font=F["bold"], fill=ORANGE, anchor="ra")

    draw.line((50, 693, 1030, 693), fill=RULE, width=2)
    draw.text((50, 720), "LIVE STRUCTURED STATE", font=F["eyebrow"], fill=MUTED)
    draw.text((50, 752), f"health {s.health}", font=F["bold"], fill=INK)
    draw.text((176, 752), f"ammo {s.ammo}", font=F["bold"], fill=INK)
    draw.text((292, 752), f"kills {s.kills}", font=F["bold"], fill=INK)
    draw.text((398, 752), f"enemies {s.enemies}", font=F["bold"], fill=INK)
    if s.target:
        draw.text((50, 791), f"target {s.target.name} · {direction(s.target.angle)} · {s.target.angle:+.1f}° · {s.target.distance:.0f} units",
                  font=F["body"], fill=INK)
    else:
        draw.text((50, 791), "target none · advance to armor", font=F["body"], fill=INK)
    goal = "not visible" if s.goal_distance is None else f"{direction(s.goal_angle)} · {s.goal_distance:.0f} units"
    draw.text((50, 828), f"goal armor {goal}", font=F["body"], fill=MUTED)
    if s.taking_damage:
        draw.rounded_rectangle((785, 740, 1030, 786), radius=10, fill="#FAD7C2")
        draw.text((907, 752), "TAKING DAMAGE", font=F["bold"], fill="#A83A00", anchor="ma")

    draw.rounded_rectangle((50, 884, 1030, 963), radius=16, fill=INK)
    draw.text((76, 901), "DISCLOSED CONTROL LOOP", font=F["eyebrow"], fill="#BEB7AA")
    draw.text((76, 929), "telemetry > router > TinyJev > aim + keys > Doom", font=F["bold"], fill=BG)

    draw.line((50, 1006, 1030, 1006), fill=RULE, width=2)
    draw.text((50, 1026), "fixed seed · skill 1 · real model output", font=F["small"], fill=MUTED)
    draw.text((1030, 1026), "github.com/ankit-aglawe/tinyjev", font=F["small"], fill=ORANGE, anchor="ra")
    return canvas


def save_gif(frames: list[Image.Image], output: Path, fps: int) -> None:
    sample_count = min(10, len(frames))
    sample = Image.new("RGB", (1080, 1080 * sample_count))
    step = max(1, len(frames) // sample_count)
    for i, frame in enumerate(frames[::step][:sample_count]):
        sample.paste(frame, (0, i * 1080))
    palette = sample.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
    paletted = [frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for frame in frames]
    paletted[0].save(output, save_all=True, append_images=paletted[1:], duration=round(1000 / fps),
                     loop=0, optimize=True, disposal=1)


def run(output: Path, decisions: int, fps: int, skill: int, seed: int) -> None:
    agent = tinyjev.load("TinyJev-0.6B", quantize=8)
    # Warm the exact two-choice inference path.
    agent.predict({"state": "Health 100. No enemy nearby. Goal ahead.", "questions": {"action": {
        "type": "choice", "instructions": "Which action best executes advance mode?",
        "criteria": MODE_ACTIONS["advance"]}}})
    game = setup_game(skill, seed)
    frames: list[Image.Image] = []
    counts: Counter[str] = Counter()
    previous_health = None
    last_snapshot = None
    last_decision = None
    try:
        game.new_episode()
        for number in range(1, decisions + 1):
            if game.is_episode_finished():
                break
            state = game.get_state()
            if state is None:
                break
            s = snapshot(state, previous_health)
            previous_health = s.health
            d = decide(agent, s)
            counts[d["mode"]] += 1
            vector = action_vector(d["action"], s, d["mode"])
            last_snapshot, last_decision = s, d
            # Two visible frames per decision, five Doom tics total: seven decisions/second.
            for tics in (2, 3):
                current = game.get_state()
                if current is None:
                    break
                frames.append(render(current.screen_buffer, s, d, number))
                game.make_action(vector, tics)
                # Delta turns should happen once; keep the movement/fire portion active.
                vector[0] = 0.0
        if last_snapshot and last_decision:
            if game.is_episode_finished():
                outcome = "PLAYER DIED" if game.is_player_dead() else "GOAL REACHED"
            else:
                outcome = "RUN COMPLETE"
            for _ in range(fps * 2):
                frame = frames[-1].copy()
                draw = ImageDraw.Draw(frame)
                draw.rounded_rectangle((330, 448, 750, 548), radius=18, fill=INK)
                draw.text((540, 476), outcome, font=F["action"], fill=ORANGE, anchor="ma")
                frames.append(frame)
    finally:
        game.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, output, fps)
    summary = ", ".join(f"{k}={v}" for k, v in counts.items())
    print(f"wrote {output} ({len(frames)} frames, {output.stat().st_size / 1_000_000:.1f} MB; {summary})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gif", type=Path, default=ROOT / "assets" / "tinyjev_doom.gif")
    parser.add_argument("--decisions", type=int, default=120)
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--skill", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args.gif, args.decisions, args.fps, args.skill, args.seed)


if __name__ == "__main__":
    main()
