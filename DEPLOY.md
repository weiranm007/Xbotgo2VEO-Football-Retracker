# Deploying FootTrack to Railway

Railway is the easiest way to get this live. Free tier included, upgrade when you have users.
Total time: ~15 minutes.

---

## Step 1 — Create a GitHub repo

Railway deploys from Git. You need to push your code there first.

1. Go to https://github.com/new and create a **private** repo called `football-retracker`
2. On your computer, open a terminal in the `football-retracker` folder and run:

```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/football-retracker.git
git push -u origin main
```

---

## Step 2 — Deploy on Railway

1. Go to https://railway.app and sign up (free)
2. Click **New Project → Deploy from GitHub repo**
3. Connect your GitHub account and select `football-retracker`
4. Railway auto-detects the `railway.toml` — click **Deploy**
5. Wait ~3 minutes for the build (it installs ffmpeg + Python packages)
6. Click your deployment → **Settings → Networking → Generate Domain**
7. You'll get a URL like `football-retracker-production.up.railway.app`

That's it — your app is live.

---

## Step 3 — Set environment variables (optional but recommended)

In Railway dashboard → your service → **Variables**, add:

| Key | Value | Purpose |
|-----|-------|---------|
| `MAX_UPLOAD_MB` | `500` | Max video upload size in MB |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Enables AI scene analysis (optional) |

---

## Step 4 — Upgrade for paying users

Railway's free tier gives you $5/month credit — enough for light testing.
For real traffic, upgrade to the **Hobby plan ($20/month)** which gives you:
- 8GB RAM (needed for long videos)
- 100GB disk
- No sleep on inactivity

To handle more than ~3 concurrent users processing video, you'll need to either:
- **Scale vertically**: Railway Pro → larger instance (8 vCPU / 32GB RAM = ~$150/mo)
- **Scale horizontally**: Add a job queue (Redis + Celery) so videos process in background — contact me if you want this built

---

## Step 5 — Custom domain

1. Buy a domain (Namecheap ~$12/year, Cloudflare ~$10/year)
2. In Railway → Settings → Networking → Custom Domain → enter your domain
3. Add the CNAME record Railway gives you to your DNS provider
4. HTTPS is automatic (Let's Encrypt)

---

## Updating the app

Whenever you want to push a new version:

```bash
git add .
git commit -m "your change description"
git push
```

Railway auto-redeploys within ~2 minutes.

---

## Troubleshooting

**Build fails with "ffmpeg not found"**
→ Check `railway.toml` has `nixPkgs = ["ffmpeg", "python311"]`

**Upload fails with 413 error**
→ Go to Railway → Settings → add env var `NIXPACKS_APT_PACKAGES=nginx` or increase `MAX_CONTENT_LENGTH` in app.py

**Processing times out**
→ Railway has a 5 minute request timeout on free tier. Upgrade to Hobby or add async job processing.

**Video is processed but download fails**
→ Railway's ephemeral disk means files disappear on redeploy. For production, add Cloudflare R2 storage (see below).

---

## Production storage (for serious use)

Railway's disk is ephemeral — files vanish on redeploy. For a real product:

1. Sign up at https://cloudflare.com → R2 (free for first 10GB)
2. Create a bucket called `foottrack-outputs`
3. Add to Railway env vars:
   - `R2_ACCOUNT_ID`
   - `R2_ACCESS_KEY_ID`  
   - `R2_SECRET_ACCESS_KEY`
   - `R2_BUCKET=foottrack-outputs`
4. Run: `pip install boto3`

Then in `app.py`, after processing completes, upload to R2 and return a pre-signed URL instead of serving the file directly. Let me know if you want this built.
