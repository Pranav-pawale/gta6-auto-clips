import os
import re
import json
import time
import subprocess
import requests
from datetime import datetime, timedelta, timezone
from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# --- CONFIG ---
WHITELIST_STREAMERS = ["caseoh_", "darkviperau", "summit1g", "loserfruit", "kaicenat"]
TARGET_GAME_NAME = "Grand Theft Auto VI"
GEMINI_MODEL = "gemini-3.5-flash-lite"
POSTED_FILE = "posted_ids.json"

# Words/slangs to automatically beep out in audio and censor in subtitles
BAD_WORDS = {
    "fuck", "fucking", "fucked", "fucker", "motherfucker", "motherfucking", "fucks",
    "shit", "shitty", "bullshit", "shitting", "bitch", "bitches", "bitching",
    "asshole", "bastard", "dick", "dickhead", "pussy", "cunt", "whore", "slut",
    "nigga", "niggas", "nigger", "niggers", "faggot", "fag", "retard", "retarded",
    "cock", "cocksucker", "tits", "twat", "wanker", "stfu", "wtf"
}


def get_gemini_client():
    return genai.Client(
        api_key=os.environ["GEMINI_API_KEY"].strip(),
        http_options={"api_version": "v1beta"}
    )


def get_twitch_token():
    r = requests.post("https://id.twitch.tv/oauth2/token", data={
        "client_id": os.environ["TWITCH_CLIENT_ID"].strip(),
        "client_secret": os.environ["TWITCH_CLIENT_SECRET"].strip(),
        "grant_type": "client_credentials"
    }).json()
    if "access_token" not in r:
        raise RuntimeError(f"Twitch Auth Failed: {r}")
    return r["access_token"]


def get_game_id(headers, game_name):
    r = requests.get(
        "https://api.twitch.tv/helix/games",
        headers=headers,
        params={"name": game_name}
    ).json().get("data", [])
    return r[0]["id"] if r else None


def get_top_clip(token):
    headers = {
        "Client-ID": os.environ["TWITCH_CLIENT_ID"].strip(),
        "Authorization": f"Bearer {token}"
    }
    with open(POSTED_FILE, "r") as f:
        posted = set(json.load(f))

    gta6_id = get_game_id(headers, TARGET_GAME_NAME)
    if not gta6_id:
        print(f"Could not find Twitch category for '{TARGET_GAME_NAME}'.")
        return None

    started_at = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    long_clips = []
    medium_clips = []

    # 1. Check whitelisted streamers strictly for GTA 6 clips
    users = requests.get(
        "https://api.twitch.tv/helix/users",
        headers=headers,
        params=[("login", u) for u in WHITELIST_STREAMERS]
    ).json().get("data", [])

    for u in users:
        clips = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
            "broadcaster_id": u["id"],
            "started_at": started_at,
            "first": 20
        }).json().get("data", [])
        for c in clips:
            if c["id"] in posted or c.get("game_id") != gta6_id:
                continue
            if 45 <= c["duration"] <= 60:
                long_clips.append(c)
            elif 30 <= c["duration"] < 45:
                medium_clips.append(c)

    # 2. If none of the whitelisted streamers played GTA 6 recently, pull top clips from the GTA 6 category
    if not long_clips and not medium_clips:
        print("No whitelisted GTA 6 clips found; fetching top clips from GTA 6 category...")
        clips = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
            "game_id": gta6_id,
            "started_at": started_at,
            "first": 50
        }).json().get("data", [])
        for c in clips:
            if c["id"] in posted:
                continue
            if 45 <= c["duration"] <= 60:
                long_clips.append(c)
            elif 30 <= c["duration"] < 45:
                medium_clips.append(c)

    candidates = long_clips if long_clips else medium_clips
    if not candidates:
        return None

    top_pool = sorted(candidates, key=lambda x: x["view_count"], reverse=True)[:15]
    clip_list = "\n".join(
        f"{i}: [{c['broadcaster_name']}] {c['title']} ({c['view_count']} views, {int(c['duration'])}s)"
        for i, c in enumerate(top_pool)
    )

    client = get_gemini_client()
    pick_prompt = f"""
    Pick the SINGLE best GTA 6 clip index (0 to {len(top_pool)-1}) for a viral gaming Short.
    Prioritize clips with a clear story/moment where the streamer is surprised, excited, panicking, or laughing hard.
    Avoid clips with profanity/slurs in the title or boring/quiet gameplay.
    Return ONLY JSON: {{"index": <int>}}

    Clips:
    {clip_list}
    """
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=pick_prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    idx = json.loads(resp.text).get("index", 0)
    return top_pool[idx] if 0 <= idx < len(top_pool) else top_pool[0]


def detect_facecam_box(video_path):
    """Checks 3 timestamps to reliably locate a real-life streamer facecam."""
    client = get_gemini_client()
    locate_prompt = (
        "Find the real-life human streamer's webcam/facecam overlay in this Twitch gaming screenshot. "
        "Note: Streamers often sit in dark rooms, wear headsets, or use green screens—that STILL counts as a facecam. "
        "Ignore Twitch chat text boxes, game minimaps, and in-game 3D characters. "
        "If a real human webcam is visible, return JSON: "
        "{\"has_cam\": true, \"box_2d\": [ymin, xmin, ymax, xmax]} (0-1000 normalized coordinates around the webcam/person). "
        "If there is NO real human webcam anywhere on screen, return JSON: {\"has_cam\": false, \"box_2d\": null}."
    )

    for ts in ["2", "15", "30"]:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", ts, "-i", video_path, "-frames:v", "1", "frame.jpg"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        if not os.path.exists("frame.jpg"):
            continue
        with open("frame.jpg", "rb") as f:
            img_bytes = f.read()

        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), locate_prompt],
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            data = json.loads(resp.text)
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict) or not data.get("has_cam") or not data.get("box_2d"):
                continue

            box = data["box_2d"]
            if isinstance(box, list) and len(box) == 1 and isinstance(box[0], list):
                box = box[0]
            if not isinstance(box, list) or len(box) != 4:
                continue

            ymin, xmin, ymax, xmax = [float(v) for v in box]
            xf = max(0.0, (xmin / 1000.0) - 0.02)
            yf = max(0.0, (ymin / 1000.0) - 0.02)
            wf = min(1.0 - xf, ((xmax - xmin) / 1000.0) + 0.04)
            hf = min(1.0 - yf, ((ymax - ymin) / 1000.0) + 0.04)

            if 0.06 <= wf <= 0.65 and 0.06 <= hf <= 0.65:
                return (xf, yf, wf, hf)
        except Exception as e:
            print(f"Facecam check skipped at {ts}s: {e}")
            continue

    return None


def is_bad_word(word):
    clean = re.sub(r"[^a-zA-Z]", "", word).lower()
    return clean in BAD_WORDS


def format_srt_time(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"


def process_video(clip_url):
    subprocess.run(["yt-dlp", "-f", "best[ext=mp4]/best", "-o", "raw.mp4", clip_url], check=True)

    # 1. Transcribe with word-level timestamps for audio censorship + 3-word minimal captions
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        "raw.mp4
