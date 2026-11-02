# Project page

GitHub Pages serves the root of the repository's `gh-pages` branch. The page
is static and needs no build step.

Preview from this branch:

~~~bash
python3 -m http.server 8000
~~~

Edit index.html for page content, static/css/index.css for styles and
static/js/index.js for video controls. Keep comparison clips and captions
matched to the evaluation described in the paper.

The page adapts the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
and [Nerfies](https://nerfies.github.io/). See
static/css/BULMA-LICENSE.txt and the font license files in static/fonts/ for
third-party notices. The adapted template uses CC BY-SA 4.0, Bulma uses MIT,
and the fonts use the SIL Open Font License. These notices do not change the
license of the paper, generated videos or AB-Attention code.
