# OpenEval — Manual Execution Checklist

All content is prepared. You just need credentials. Estimated: 2 hours.

---

## 1. npm Publish (15 min)

```bash
# Step 1: Log in to npm
npm login

# Step 2: Publish TypeScript SDK
cd /Users/sk/Desktop/openeval/sdk/typescript
npm publish --access public

# Step 3: Publish CLI
cd /Users/sk/Desktop/openeval/cli
npm publish --access public

# Step 4: Verify
npm view @openeval/sdk
npm view @openeval/cli
```

Verify: https://www.npmjs.com/package/@openeval/sdk should show v1.0.0

---

## 2. PyPI Publish (15 min)

```bash
# Step 1: Create PyPI account at https://pypi.org (if needed)
# Step 2: Generate API token at https://pypi.org/manage/account/tokens/

# Step 3: Build and upload
cd /Users/sk/Desktop/openeval/sdk/python
python -m pip install build twine
python -m build
twine upload dist/*
# Username: __token__
# Password: [paste your PyPI API token]

# Step 4: Verify
pip install openeval
python -c "from openeval.validate import validate_suite; print('OK')"
```

Verify: https://pypi.org/project/openeval/ should show v1.0.0

---

## 3. Send 5 Founder Emails (20 min)

Content ready at: `docs/founder-emails.md`

Recipients to find:
- Jeffrey Ip (DeepEval) — search LinkedIn or check confident-ai/deepeval README
- Michael D'Amour (Promptfoo) — check promptfoo/promptfoo or michael@promptfoo.dev
- UK AISI (Inspect AI) — contact via uksecurity.ai or inspect_ai docs
- LangChain team — contact via langchain.com
- EleutherAI — contact via Discord or GitHub

Just copy each email from `docs/founder-emails.md`, personalize, and send from your Gmail.

---

## 4. Social Media Posts (45 min)

All content ready at: `docs/social-media-posts.md`

### Hacker News (post at 9 AM PT)
- Go to https://news.ycombinator.com/ → submit
- Title: "OpenEval: Why LLM Evaluation Needs a Standard Format"
- URL: https://github.com/adhabnr-ux/openeval
- Engage with comments for 2 hours

### Reddit (3 posts)
- r/MachineLearning: Title from docs/social-media-posts.md
- r/LocalLLaMA: Title from docs/social-media-posts.md
- r/ArtificialIntelligence: Title from docs/social-media-posts.md

### Dev.to
- Create post, paste content from docs/blog/launch-post.md
- Add frontmatter from docs/social-media-posts.md

### LinkedIn
- Start a post, paste LinkedIn content from docs/social-media-posts.md
- Tag: #LLM #AI #Evaluation #OpenSource

---

## 5. Gmail Re-auth (optional, 10 min)

If you want to send emails from Cline:
- In Claude/Cline, find Gmail in connected tools
- Click Reconnect/Re-authorize
- Follow OAuth flow

Alternative: Just send founder emails manually from Gmail web.

---

## Summary of What's Already Done

✅ GitHub repo: https://github.com/adhabnr-ux/openeval (66 files, tagged v1.0.0-rc.1)
✅ 17 GitHub issues filed on target repos
✅ Founder emails drafted (docs/founder-emails.md)
✅ Social media posts prepared (docs/social-media-posts.md)
✅ Blog post written (docs/blog/launch-post.md)
✅ Landing page built (docs/landing-page.html)
✅ Full documentation (getting-started, grader-reference, migration guides)
✅ Converters for Promptfoo, DeepEval, Inspect AI, OpenAI Evals
✅ CI/CD pipeline (.github/workflows/ci.yml)

## What You Need To Do

- [ ] npm login + publish (2 packages)
- [ ] PyPI account + token + publish (1 package)
- [ ] Send 5 founder emails
- [ ] Post to HN (engage 2 hours)
- [ ] Post to Reddit (3 subreddits)
- [ ] Post to Dev.to
- [ ] Post to LinkedIn
