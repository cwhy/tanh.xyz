"""Separate the readable article from the full experiment record."""
import re
from pathlib import Path
from article import render_article


def organize(text):
    start = text.index('<section class="intro">')
    end = text.index('</main>', start)
    original = text[start:end]
    sections = re.findall(r'<section\b[^>]*>.*?</section>', original, re.S)
    intro = sections.pop(0)
    intro = re.sub(r'<p><a href="#first-failure">.*?</p>', '<nav class="article-nav" aria-label="On this page"><a href="#article">Article</a><a href="#references">References</a><a href="#appendix">Appendix &amp; evidence</a></nav>', intro)
    sources = re.search(r'<ul class="source-list">.*?</ul>', original, re.S).group()
    papers = re.findall(r'<a href="(https://(?:arxiv.org|www.anthropic.com)[^"]+)">([^<]+)</a>', original)
    refs = '<section id="references" class="part"><p class="eyebrow">02 / References</p><h2>Sources behind the experiment</h2><p>The model claims below come from their authors. The results in this article come from my recorded experiments.</p>'+sources+'<ul class="source-list">'+''.join(f'<li><a href="{url}">{label}</a></li>' for url,label in dict(papers).items())+'</ul><p><a href="https://github.com/cwhy/decision-injection-bench">Experiment code and reproduction instructions</a>. Full local records are in the <a href="#followup-evidence">evidence appendix</a>.</p></section>'
    appendix = '<section id="appendix" class="part"><p class="eyebrow">03 / Appendix</p><h2>The full experiment record</h2><p>Open a section for exact prompts, policies, counts, limitations, or downloadable records. Counts from different campaigns are kept separate.</p>'
    for i,section in enumerate(sections):
        title = re.search(r'<h2>(.*?)</h2>', section).group(1)
        section = section.replace('<h3>Primary sources</h3>'+sources, '')
        section = section.replace('<h2>'+title+'</h2>', '')
        section = section.replace('<details open>', '<details>')
        appendix += f'<details class="appendix-entry"><summary>{title}</summary>{section}</details>'
    appendix += '</section>'
    article = render_article(Path(__file__).resolve().parents[2] / 'public/data/demos/jev-prompt-manipulation')
    text = text[:start]+intro+article+refs+appendix+text[end:]
    # Fragment links reveal the relevant disclosure, including on direct navigation.
    reveal = '''<script>
function revealAnchor(){const id=decodeURIComponent(location.hash.slice(1));const target=document.getElementById(id);if(!target)return;let parent=target.parentElement;while(parent){if(parent.tagName==='DETAILS')parent.open=true;parent=parent.parentElement;}requestAnimationFrame(()=>target.scrollIntoView());}
window.addEventListener('hashchange',revealAnchor);revealAnchor();
</script>'''
    return text.replace('</body>', reveal+'</body>')
