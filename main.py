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
        model="gemini-2.5-flash",
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

        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"), locate_prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(resp.text)
        if not data.get("has_cam") or not data.get("box_2d"):
            continue

        ymin, xmin, ymax, xmax = data["box_2d"]
        # Add slight padding so the streamer's head/shoulders aren't cut off
        xf = max(0.0, (xmin / 1000.0) - 0.02)
        yf = max(0.0, (ymin / 1000.0) - 0.02)
        wf = min(1.0 - xf, ((xmax - xmin) / 1000.0) + 0.04)
        hf = min(1.0 - yf, ((ymax - ymin) / 1000.0) + 0.04)

        if 0.06 <= wf <= 0.65 and 0.06 <= hf <= 0.65:
            return (xf, yf, wf, hf)

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

    # 1. Transcribe with word-level timestamps to detect & beep bad words + build 3-word minimal captions
    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        "raw.mp4",
        vad_filter=True,
        vad_parameters={"threshold": 0.25, "min_silence_duration_ms": 400},
        condition_on_previous_text=False,
        word_timestamps=True
    )
    transcript_text = []
    beep_ranges = []

    with open("subs.srt", "w", encoding="utf-8") as srt:
        idx = 1
        for seg in segments:
            if seg.words:
                # Split into 3-word chunks so captions stay short and never cover the screen
                for i in range(0, len(seg.words), 3):
                    chunk = seg.words[i:i + 3]
                    censored = []
                    for w in chunk:
                        if is_bad_word(w.word):
                            beep_ranges.append((max(0.0, w.start - 0.05), w.end + 0.05))
                            censored.append("****")
                        else:
                            censored.append(w.word.strip())
                    text = " ".join(censored).strip()
                    if not text:
                        continue
                    transcript_text.append(text)
                    srt.write(
                        f"{idx}\n{format_srt_time(chunk[0].start)} --> {format_srt_time(chunk[-1].end)}\n{text}\n\n"
                    )
                    idx += 1
            else:
                text = seg.text.strip()
                if not text:
                    continue
                transcript_text.append(text)
                srt.write(f"{idx}\n{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}\n{text}\n\n")
                idx += 1

    print(f"Censored {len(beep_ranges)} offensive word(s).")

    # 2. Auto-detect facecam box via Gemini Vision
    cam_box = detect_facecam_box("raw.mp4")
    sub_filter = (
        ",subtitles=subs.srt:force_style='FontName=Arial,FontSize=11,Bold=1,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=1.5,Shadow=0,Alignment=5'"
        if transcript_text else ""
    )

    if cam_box:
        xf, yf, wf, hf = cam_box
        print(f"Facecam detected via Gemini Vision: x={xf:.2f}, y={yf:.2f}, w={wf:.2f}, h={hf:.2f}")
        v_fc = (
            f"[0:v]crop=iw*{wf:.4f}:ih*{hf:.4f}:iw*{xf:.4f}:ih*{yf:.4f},"
            f"scale=1080:640:force_original_aspect_ratio=increase,crop=1080:640[cam];"
            f"[0:v]scale=1080:1280:force_original_aspect_ratio=increase,crop=1080:1280[game];"
            f"[cam][game]vstack=inputs=2{sub_filter}[outv]"
        )
    else:
        print("No facecam detected; using clean full-screen 9:16 crop.")
        v_fc = f"[0:v]crop=ih*9/16:ih,scale=1080:1920{sub_filter}[outv]"

    # 3. Audio: beep out offensive words + loudness normalization + limiter
    if beep_ranges:
        cond = "+".join(f"between(t,{s:.2f},{e:.2f})" for s, e in beep_ranges)
        a_fc = (
            f"[0:a]volume=enable='{cond}':volume=0[muted];"
            f"sine=frequency=1000:sample_rate=48000,volume=0.25,volume=enable='not({cond})':volume=0[beep];"
            f"[muted][beep]amix=inputs=2:duration=first:dropout_transition=0,"
            f"loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.8[outa]"
        )
    else:
        a_fc = "[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,alimiter=limit=0.8[outa]"

    full_fc = f"{v_fc};{a_fc}"

    subprocess.run([
        "ffmpeg", "-y", "-i", "raw.mp4",
        "-filter_complex", full_fc, "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-ar", "48000", "-b:a", "128k",
        "-movflags", "+faststart", "final.mp4"
    ], check=True)

    return " ".join(transcript_text)


def generate_metadata(transcript, streamer_name, clip_title):
    client = get_gemini_client()
    prompt = f"""
    You are a viral gaming shorts editor for GTA 6.
    Streamer: {streamer_name}
    Original Clip Title: {clip_title}
    Transcript: {transcript}

    Return ONLY valid JSON with these keys (keep all text family-friendly, no profanity):
    - "yt_title": Punchy curiosity-driven title under 65 characters + #shorts
    - "yt_description": 2 short lines of context + "Credit: @{streamer_name} on Twitch" + 4 GTA 6 hashtags
    - "ig_caption": 1-line hook + blank line + "Credit: @{streamer_name} (Twitch)" + 4 GTA 6 hashtags
    """
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    return json.loads(resp.text)


def upload_youtube(meta):
    creds = Credentials(
        token=None,
        refresh_token=os.environ["YT_REFRESH_TOKEN"].strip(),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["YT_CLIENT_ID"].strip(),
        client_secret=os.environ["YT_CLIENT_SECRET"].strip()
    )
    yt = build("youtube", "v3", credentials=creds)
    body = {
        "snippet": {
            "title": meta["yt_title"],
            "description": meta["yt_description"],
            "categoryId": "20"
        },
        "status": {"privacyStatus": "public"}
    }
    media = MediaFileUpload("final.mp4", mimetype="video/mp4", resumable=True)
    yt.videos().insert(part="snippet,status", body=body, media_body=media).execute()
    print("YouTube upload complete (Public).")


def upload_instagram(public_video_url, caption):
    ig_id = os.environ["IG_USER_ID"].strip()
    token = os.environ["IG_ACCESS_TOKEN"].strip()
    base = "https://graph.facebook.com/v21.0"

    c_res = requests.post(f"{base}/{ig_id}/media", data={
        "media_type": "REELS",
        "video_url": public_video_url,
        "caption": caption,
        "share_to_feed": "true",
        "access_token": token
    }).json()
    if "id" not in c_res:
        raise RuntimeError(f"Instagram API Error: {c_res}")
    creation_id = c_res["id"]

    for _ in range(20):
        status = requests.get(f"{base}/{creation_id}", params={
            "fields": "status_code", "access_token": token
        }).json().get("status_code")
        if status == "FINISHED":
            break
        if status == "ERROR":
            raise RuntimeError(f"Instagram processing failed: {c_res}")
        time.sleep(10)

    requests.post(f"{base}/{ig_id}/media_publish", data={
        "creation_id": creation_id,
        "access_token": token
    }).raise_for_status()
    print("Instagram Reel published!")


if __name__ == "__main__":
    token = get_twitch_token()
    clip = get_top_clip(token)
    if not clip:
        print("No new GTA 6 clips found.")
        exit(0)

    print(f"Selected GTA 6 clip: [{clip['broadcaster_name']}] {clip['title']} ({int(clip['duration'])}s)")
    transcript = process_video(clip["url"])
    meta = generate_metadata(transcript, clip["broadcaster_name"], clip["title"])
    upload_youtube(meta)

    with open("meta.json", "w") as f:
        json.dump({"clip_id": clip["id"], "ig_caption": meta["ig_caption"]}, f)
