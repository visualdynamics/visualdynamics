# visualdynamics.org — the holding page

**`public/` is the holding page and `launch/` is the site from the
first release; everything else here is neither.** Only one of those
two directories is ever deployed, because a note like this one served
at `visualdynamics.org/README.md` is an internal document on a public
host. Keep new assets inside them and new notes outside.

One self-contained page. No build step, no framework, no third-party
request: the CSS is inline and the only assets are three PNGs beside it.

## The mark is generated, not drawn

`mark.png`, `apple-touch-icon.png` and `favicon.png` come from
`visualdynamics.gui.app_icon.draw_app_icon` — the same plate carrying a
mode that the application wears in the Dock. The page and the program
it announces cannot drift apart, because one produced the other:

```bash
python - <<'PY'
import os; os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QBuffer, QByteArray
app = QApplication.instance() or QApplication([])
from visualdynamics.gui.app_icon import draw_app_icon
for size, name in ((512, 'web/public/mark.png'),
                   (180, 'web/public/apple-touch-icon.png'),
                   (64, 'web/public/favicon.png')):
    store = QByteArray(); buf = QBuffer(store)
    buf.open(QBuffer.OpenModeFlag.WriteOnly)
    draw_app_icon(size).save(buf, 'PNG'); buf.close()
    open(name, 'wb').write(bytes(store))
PY
```

The colors in the page are the application's own, from
`visualdynamics.theme`: the dark scene's flat `#000000` ground (the
page's gradient is the site's, not the app's, which draws no
gradient), `#e8e8ea` for text, `#ffb020` — the 3-D view's highlight — for the one
accent.

## Publishing it

The domain is on Cloudflare, so Cloudflare Pages is the short road.

**From this repo**, now that the machine holds a wrangler OAuth token:

```bash
cd web/launch && npx wrangler pages deploy . --project-name=visualdynamics --branch=main
```

From *inside* the site directory, deploying `.`: wrangler reads Pages
Functions from `./functions` relative to where it runs, not from the
directory it deploys — deployed from the repo root, the pages went up
without `/api/requests` and the site served `index.html` in its place
(2026-09-14). "Uploading Functions bundle" in the output is the line
that says the function shipped. (`web/public/` was the holding page
the site served until the first release.) This is a token for pushing assets, which is a
narrower grant than the GitHub connection ruled out below — Cloudflare
never sees the source.

**After the first release the site deploys itself.** The release
workflow's `site` job (`.github/workflows/release.yml`) runs when a
release is *published*: it builds the documentation, writes
`latest.json` from the release, and deploys `web/launch` — so the
site, the docs and the update manifest move together and match the
release. It needs two repository secrets on the shared repository:

1. Cloudflare dashboard → Manage account → Account API tokens →
   Create a token → *Custom token*, one permission policy: **Account ·
   Cloudflare Pages · Write** (the dashboard's 2026 wording; the
   review page reads "Entire … account — Pages Write", the account
   scope folded into the policy line — there is no separate Account
   Resources section any more). No expiration is fine: revoke it on
   the same page if it ever leaks. Copy the token once; it is shown
   once.
2. The account ID is the hex string in the dashboard's address bar,
   `dash.cloudflare.com/<account id>/…`, and on the overview page for
   the domain (right-hand column, "Account ID").
3. GitHub → the repository → Settings → Secrets and variables →
   Actions → New repository secret: `CLOUDFLARE_API_TOKEN`, then
   `CLOUDFLARE_ACCOUNT_ID`.

Until those exist the job fails and the site stays as it was. The
manual route below still works and is the fallback.

Where they live, settled 2026-09-11: the token is an *account* API
token (Manage account → Account API tokens), not a user token under
My Profile — an older user token from the first pass was deleted —
and the two secrets are *repository* secrets on the shared
repository, not organization secrets. An organization copy from the
first pass was removed too: the free plan scopes organization secrets
to all repositories or to public ones, so one scoped to public would
have left the workflow blind while the repository is still private,
and a token in two places is two to rotate.

**The documentation is a page of the site** (Brandon, 2026-09-01):
`properdocs.yml` builds straight into `web/launch/documentation/`, which
is gitignored, so one deploy carries both and the docs answer at
`visualdynamics.org/documentation/` — a link in the site's nav, and the
address the pyproject and the README name. The preview shows it too
once built (http://127.0.0.1:8710/documentation/). Order on the day:

```bash
./.venv/bin/python packaging/stage_downloads.py clear
./.venv/bin/properdocs build --strict
cd web/launch && npx wrangler pages deploy . --project-name=visualdynamics --branch=main
```

Skip the middle line and the deployed site's Documentation link is a
404; CI's docs job keeps building into `site/` for its artifact.

**Direct upload** through the dashboard does the same thing by hand:

1. Cloudflare dashboard → Workers & Pages → Create → Pages → *Upload
   assets*.
2. Drag `web/public/` in — that folder only. Name the project
   `visualdynamics`.
3. Custom domains → add `visualdynamics.org` and `www`. Cloudflare
   writes the DNS itself because the domain is already there.

**Do not connect it to the GitHub repository.** That grants Cloudflare
read access to a private repo in exchange for saving one drag-and-drop
per change, and this page changes about once a year.

## The feature-requests page

`launch/requests.html` ranks the project's GitHub discussions in the
*Ideas* category by upvotes and links to them — requests and votes
live on GitHub, one upvote per account, and the page is the ranking
(Brandon, 2026-09-11). The ranking comes from `/api/requests`,
`launch/functions/api/requests.js`, a Pages Function that wrangler
deploys with the site: GitHub's GraphQL API needs a token even for
public discussions, and the function is the one place it lives.

1. GitHub → Settings → Developer settings → Fine-grained tokens →
   Generate: no repository access, no permissions — public data is
   all it reads. Expiry a year.
2. Cloudflare dashboard → Workers & Pages → visualdynamics →
   Settings → Variables and Secrets → add `GITHUB_TOKEN` as a
   *secret*, for Production.
3. On the shared repository, enable Discussions and keep the default
   *Ideas* category; its slug is what the function filters on.

When the query fails the function answers `{open: false}`, which the
page reads as "requests could not be loaded" rather than as an empty
list. The function caches its answer at the edge for ten
minutes. Previewing locally needs `npx wrangler pages dev web/launch`
rather than a plain file server, or the page shows the could-not-load
note.

## What belongs on the holding page

`public/index.html` was the page that stood in for a project that was
not yet public (it has been, since 2026-09-14; `web/launch` is what
the site serves now), and its rule is kept for the record:
narrow — the name, the mark, a way to make contact. **Not what the thing
does** — that is deliberate, and it has to be kept out of the places
that are read by machines rather than people: `<meta name=description>`,
`og:description`, and the image's `alt`. A sentence deleted from the
body while it still sits in a meta tag has not been deleted.

- **No screenshots of real data.** Nothing from `stressdata/`, nothing
  from a customer's run. If a picture is ever wanted, the demonstration
  drone is the only safe subject — it is synthetic and it is ours.
- **No feature list and no dates.** Both are promises, and neither
  helps a holding page do its one job.
- **The address is a role, not a person.** `contact@` can be deleted
  and remade in Email Routing the day it is buried in spam; a personal
  address on a public page cannot be taken back. Catch-all stays off
  for the same reason — it turns every dictionary attack into inbox.
  The rule generalized on 2026-08-27: the domain now also carries
  `quickbooks-dev@`, forwarding to the same inbox, so that the Intuit
  developer app's public pages could stop printing a personal address.
  Both are literal rules in Cloudflare Email Routing; adding another
  costs nothing and is the right answer any time a page needs a
  contact.
- **Keep the copyright line.** Publishing the name under a dated notice
  is the cheapest evidence of first use there is, which matters if the
  mark is ever worth registering.

## Before deploying `web/launch`

`web/launch/downloads/` is where `packaging/stage_downloads.py` hardlinks
the locally built installers for the preview. It is gitignored, but
`wrangler pages deploy` uploads a directory as it stands — run
`packaging/stage_downloads.py clear` first, or hundreds of megabytes of
builds go up with the pages.

## The example projects

The two example tiles wear a picture of what is inside them — the
plate's mesh and the quadcopter's airframe, rendered from the app's
own scene on a transparent ground — made by
`tools/make_example_icons.py` into `web/launch/plate.png` and
`web/launch/drone.png` (Brandon, 2026-09-16). Re-run it if the
geometry or the theme changes; the renders are not byte-stable across
VTK builds, so the test holds them to their shape rather than their
bytes.

The downloads page links two zips — the plate and the quadcopter
projects, one `.vdyn` per workflow — that live on a GitHub release
of their own, tagged `examples`, so the package stays small.
`tools/cut_examples.py` builds and uploads them; re-cut them whenever
the project format moves. That release is created with
`--latest=false`, so `releases/latest` — which the downloads page's
tiles and `latest.json` both read — keeps naming the newest version.

## `latest.json` is the update check's manifest

`latest.json` is what the application's *File → Check for Updates…*
reads (`src/visualdynamics/update.py`): a version, a URL to open, a
line of notes. `web/public/latest.json` is the holding page's copy,
saying 0.0.1. From the first release on, the release workflow's
`site` job writes `web/launch/latest.json` from the published release
and deploys it with the site — so it can never name assets that do
not exist yet, which is the whole rule. It is a check, not an
updater: nothing is downloaded or run from it.
