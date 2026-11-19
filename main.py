import os
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
# Twitch usernames & their facecam corner: "tl" (top-left), "tr" (top-right), "bl" (bottom-left), "br" (bottom-right)
STREAMER_CAMS = {
    "caseoh_": "bl",
    "darkviperau": "tl",
    "summit1g": "tl",
    "loserfruit": "bl",
    "kaicenat": "bl",
}
WHITELIST_STREAMERS = list(STREAMER_CAMS.keys())
POSTED_FILE = "posted_ids.json"


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


def get_top_clip(token):
    headers = {
        "Client-ID": os.environ["TWITCH_CLIENT_ID"].strip(),
        "Authorization": f"Bearer {token}"
    }
    with open(POSTED_FILE, "r") as f:
        posted = set(json.load(f))

    users = requests.get(
        "https://api.twitch.tv/helix/users",
        headers=headers,
        params=[("login", u) for u in WHITELIST_STREAMERS]
    ).json().get("data", [])

    started_at = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    candidates = []

    for u in users:
        clips = requests.get("https://api.twitch.tv/helix/clips", headers=headers, params={
            "broadcaster_id": u["id"],
            "started_at": started_at,
            "first": 10
        }).json().get("data", [])
        for c in clips:
            if c["id"] not in posted and 15 <= c["duration"] <= 55:
                candidates.append(c)

    if not candidates:
        return None

    top_pool = sorted(candidates, key=lambda x: x["view_count"], reverse=True)[:15]
    clip_list = "\n".join(
        f"{i}: [{c['broadcaster_name']}] {c['title']} ({c['view_count']} views, {int(c['duration'])}s)"
        for i, c in enumerate(top_pool)
    )

    client = get_gemini_client()
    pick_prompt = f"""
    Pick the SINGLE best clip index (0 to {len(top_pool)-1}) for a viral GTA/gaming Short.
    Prioritize clips where the streamer is angry, shocked, screaming, panicking, laughing hard, or having an emotional/chaotic reaction that works as an instant hook.
    Avoid boring gameplay, quiet moments, or chill driving/flying.
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


def format_srt_time(seconds):
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    ms = int((s - int(s)) * 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"


def process_video(clip_url, streamer_login):
    subprocess.run(["yt-dlp", "-f", "best[ext=mp4]/best", "-o", "raw.mp4", clip_url], check=True)

    model = WhisperModel("base", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        "raw.mp4",
        vad_filter=True,
        condition_on_previous_text=False
    )
    transcript_text = []

    with open("subs.srt", "w", encoding="utf-8") as srt:
        idx = 1
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            transcript_text.append(text)
            srt.write(f"{idx}\n{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}\n{text}\n\n")
            idx += 1

    # Facecam corner coordinates (crops 28% width x 32% height from the streamer's cam corner)
    cam_pos = STREAMER_CAMS.get(streamer_login.lower(), "bl")
    cam_coords = {
        "tl": "0:0",
        "tr": "iw-iw*0.28:0",
        "bl": "0:ih-ih*0.32",
        "br": "iw-iw*0.28:ih-ih*0.32",
    }[cam_pos]

    sub_filter = (
        ",subtitles=subs.srt:force_style='FontName=Arial,FontSize=18,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BorderStyle=1,Outline=2,Alignment=2,MarginV=80'"
        if transcript_text else ""
    )

    fc = (
        f"[0:v]crop=iw*0.28:ih*0.32:{cam_coords},scale=1080:640:force_original_aspect_ratio=increase,crop=1080:640[cam];"
        f"[0:v]scale=1080:1280:force_original_aspect_ratio=increase,crop=1080:1280[game];"
        f"[cam][game]vstack=inputs=2{sub_filter}[outv]"
    )

    subprocess.run([
        "ffmpeg", "-y", "-i", "raw.mp4",
        "-filter_complex", fc, "-map", "[outv]", "-map", "0:a?",
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

    Return ONLY valid JSON with these keys:
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
        "status": {"privacyStatus": "private"}
    }
    media = MediaFileUpload("final.mp4", mimetype="video/mp4", resumable=True)
    yt.videos().insert(part="snippet,status", body=body, media_body=media).execute()
    print("YouTube upload complete (Private).")


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
        print("No new clips found.")
        exit(0)

    print(f"Selected clip: [{clip['broadcaster_name']}] {clip['title']}")
    transcript = process_video(clip["url"], clip["broadcaster_name"])
    meta = generate_metadata(transcript, clip["broadcaster_name"], clip["title"])
    upload_youtube(meta)

    with open("meta.json", "w") as f:
        json.dump({"clip_id": clip["id"], "ig_caption": meta["ig_caption"]}, f)
