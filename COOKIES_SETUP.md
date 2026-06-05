# YouTube Cookies Setup

YouTube blocks requests from Hugging Face's server IP ranges. To allow
the chord engine to download YouTube audio on HF Spaces, you need to
provide a **browser cookies.txt file** exported from your real browser
and stored as a Hugging Face Space Secret.

## How it works

1. Your browser has cookies that prove you're a real YouTube user.
2. You export those cookies as a `cookies.txt` file.
3. The cookies are stored as an HF Secret and injected at container
   startup.
4. The server uses those cookies when making requests to YouTube,
   bypassing the IP-based block.

## How to export cookies.txt

### Install a cookie-export extension

| Browser | Extension |
|---------|-----------|
| Chrome / Edge | **Get cookies.txt LOCALLY** by Rahul Shaw |
| Firefox | **cookies.txt** by Lennon Hill |
| Safari | **cookies.txt** (search the App Store or use a Safari-specific extension) |

### Export the cookies

1. Open a new browser tab and navigate to **https://www.youtube.com**
2. **Make sure you are signed into your Google account** — a logged-in
   session has stronger cookies that are more effective at bypassing
   blocks.
3. Click the cookie-export extension icon in your browser toolbar.
4. Select **Export** (or **Export as cookies.txt**).
5. Save the file — it should be named `cookies.txt`.
6. **Open the file and check** that it contains lines for `youtube.com`
   with `__Secure-3PSID`, `__Secure-3PAPISID`, or similar session
   cookies. If it only has a `CONSENT` cookie, it won't work — you need
   to be logged in.

## How to add it to Hugging Face Spaces

1. Go to your Space on Hugging Face:  
   `https://huggingface.co/spaces/Jeremyszs/chord-engine`

2. Click **Settings** → **Variables and secrets** (or go directly to  
   `https://huggingface.co/spaces/Jeremyszs/chord-engine/settings`)

3. Scroll to **Repository Secrets**

4. Click **New secret**

5. Set:
   - **Name**: `YOUTUBE_COOKIES`
   - **Value**: Paste the **entire contents** of your `cookies.txt` file
     (not the file path — the actual text content)

6. Click **Save**

## How to update expired cookies

YouTube cookies expire. If YouTube downloads stop working after weeks
or months:

1. Re-visit https://www.youtube.com in your browser (stay logged in)
2. Re-export the cookies using the same extension
3. Update the `YOUTUBE_COOKIES` secret on HF Spaces with the new content
4. The Space will automatically restart with the fresh cookies

## Security notes

- **Never commit cookies.txt to git.** It is listed in `.gitignore` and
  `.dockerignore`.
- The cookies file is only used at runtime — it is never baked into the
  Docker image.
- The cookies give access to **your YouTube account**. Treat them like a
  password.
- If you suspect your cookies have been compromised, sign out of all
  Google sessions (Settings → Security → Your devices → Manage devices)
  and re-export.
- This only works for **public** YouTube videos. Private videos,
  age-restricted content, and member-only streams still won't work
  because the server's IP is still the blocked HF IP — only the
  **authentication** is provided via cookies.

## Local development

On your personal machine, yt-dlp usually works without cookies. If you
do want to test with cookies locally:

```bash
export YOUTUBE_COOKIES_PATH=/path/to/your/cookies.txt
uvicorn api.main:app --reload --port 8000
```

Alternatively, just skip the cookie setup for local dev and use the
file-upload endpoint instead.
