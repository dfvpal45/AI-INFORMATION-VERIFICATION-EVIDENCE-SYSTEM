"""Evidence verification pipeline. ML models and external APIs are optional; deterministic fallbacks keep local demo usable."""
import os, re, json, math, hashlib, logging
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse
from xml.etree import ElementTree
import httpx
from bs4 import BeautifulSoup
import trafilatura
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .config import settings

logger = logging.getLogger(__name__)

with (Path(__file__).parent.parent / 'data' / 'trusted_sources.json').open(encoding='utf-8') as source_file:
    TRUSTED_SOURCES = json.load(source_file)

TRUSTED_DOMAINS = {
    urlparse(source['url']).hostname.removeprefix('www.').lower(): source['credibility']
    for source in TRUSTED_SOURCES
}
TRUSTED_SOURCE_NAMES = {
    source['name'].casefold(): source['credibility']
    for source in TRUSTED_SOURCES
}

TRUSTED_CORPUS = [
 {"title":"World Health Organization: health topics", "text":"The World Health Organization publishes evidence-based public health guidance and technical information.", "url":"https://www.who.int/health-topics", "source":"WHO", "credibility":0.96},
 {"title":"United Nations: global issues", "text":"The United Nations publishes information, reports and data on international development, population and climate.", "url":"https://www.un.org/en/global-issues", "source":"United Nations", "credibility":0.95},
 {"title":"NASA climate change evidence", "text":"NASA provides climate indicators, observations and scientific explanations based on satellite and ground measurements.", "url":"https://science.nasa.gov/climate-change/", "source":"NASA", "credibility":0.96},
 {"title":"Government of India portal", "text":"The National Portal of India provides official information about government services, programmes and public institutions.", "url":"https://www.india.gov.in/", "source":"Government of India", "credibility":0.95},
 {"title":"Our World in Data", "text":"Our World in Data publishes research and visualisations using documented data sources on health, population, energy and development.", "url":"https://ourworldindata.org/", "source":"Our World in Data", "credibility":0.90},
 {"title":"Wikipedia", "text":"Wikipedia is a collaboratively edited encyclopaedia and can be a useful starting point, but important claims should be checked against primary sources.", "url":"https://en.wikipedia.org/", "source":"Wikipedia", "credibility":0.70},
]

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text or '').strip()
    return text[:30000]

def extract_from_url(url: str) -> str:
    if not url.startswith(('https://','http://')): raise ValueError('URL must begin with http:// or https://')
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        extracted = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        if extracted and len(extracted.strip()) > 30: return clean_text(extracted)
    with httpx.Client(timeout=15, follow_redirects=True, headers={'User-Agent':'AIVerificationEvidenceSystem/1.0'}) as client:
        response = client.get(url); response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    for tag in soup(['script','style','nav','footer','header','aside']): tag.decompose()
    return clean_text(soup.get_text(' ', strip=True))

def extract_image_text(data: bytes) -> str:
    try:
        import cv2, numpy as np, pytesseract
        from PIL import Image
        from io import BytesIO
        image = Image.open(BytesIO(data)).convert('RGB')
        arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if settings.tesseract_cmd: pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd
        text = pytesseract.image_to_string(gray)
        if len(text.strip()) < 3: return 'OCR found no readable text. Please upload a clearer image or paste the claim.'
        return clean_text(text)
    except Exception as e:
        return f'OCR is unavailable or failed ({type(e).__name__}). Install Tesseract OCR and upload a clear PNG/JPG, or paste the text directly.'

def split_claims(text: str) -> list[str]:
    # spaCy sentence splitting and NER are used when the model is installed; regex fallback avoids downloads.
    sentences = []
    try:
        import spacy
        try: nlp = spacy.load('en_core_web_sm')
        except OSError: nlp = spacy.blank('en'); nlp.add_pipe('sentencizer')
        doc = nlp(text[:30000])
        sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 18]
    except Exception:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    out=[]
    for s in sentences:
        s=clean_text(s).strip(' •-\t')
        if len(s) >= 20 and re.search(r'\b(is|are|was|were|has|have|will|can|causes|increased|decreased|according|percent|%|million|billion|202[0-9])\b', s, re.I): out.append(s)
    if not out and len(text.strip()) >= 3: out=[clean_text(text)[:1000]]
    return list(dict.fromkeys(out[:8]))

def external_evidence(claim: str) -> list[dict]:
    results=[]
    if settings.google_factcheck_api_key:
        try:
            with httpx.Client(timeout=8) as c:
                r=c.get('https://factchecktools.googleapis.com/v1alpha1/claims:search', params={'query':claim,'key':settings.google_factcheck_api_key,'pageSize':5}); r.raise_for_status()
                for item in r.json().get('claims',[]):
                    for review in item.get('claimReview',[]):
                        results.append({'title':item.get('text','Fact check'),'text':f"Rating: {review.get('textualRating','Unknown')}. {item.get('text','')}", 'url':review.get('url',''), 'source':review.get('publisher',{}).get('name','Fact-check publisher'), 'credibility':0.90, 'rating':review.get('textualRating','')})
        except Exception: pass
    if settings.newsapi_key:
        try:
            with httpx.Client(timeout=8) as c:
                r=c.get('https://newsapi.org/v2/everything',params={'q':claim,'language':'en','sortBy':'relevancy','pageSize':4,'apiKey':settings.newsapi_key}); r.raise_for_status()
                for a in r.json().get('articles',[]):
                    desc=(a.get('description') or a.get('content') or '')
                    if desc: results.append({'title':a.get('title','News article'),'text':desc,'url':a.get('url',''),'source':(a.get('source') or {}).get('name','News source'),'credibility':0.72})
        except Exception: pass
    # Google News RSS provides current reporting without requiring an API key.
    try:
        headers={'User-Agent':'AIVerificationEvidenceSystem/1.0 (evidence search)'}
        with httpx.Client(timeout=8, follow_redirects=True, headers=headers) as c:
            r=c.get(
                'https://news.google.com/rss/search',
                params={'q':claim,'hl':'en-US','gl':'US','ceid':'US:en'},
            )
            r.raise_for_status()
            root=ElementTree.fromstring(r.content)
            for item in root.findall('./channel/item')[:8]:
                title=item.findtext('title','').strip()
                link=item.findtext('link','').strip()
                description=item.findtext('description','')
                source_name=item.findtext('source','Google News').strip()
                snippet=BeautifulSoup(description,'html.parser').get_text(' ',strip=True)
                if title and link:
                    results.append({
                        'title':title,
                        'text':snippet or title,
                        'url':link,
                        'source':source_name or 'Google News',
                        'credibility':_source_credibility(link,source_name),
                    })
    except (httpx.HTTPError, ElementTree.ParseError) as error:
        logger.warning("Google News evidence search failed: %s", error)
    # Public web search provides claim-specific and Wikipedia results.
    try:
        headers={'User-Agent':'Mozilla/5.0 (compatible; AIVES/1.0; evidence search)'}
        with httpx.Client(timeout=8, follow_redirects=True, headers=headers) as c:
            r=c.get('https://html.duckduckgo.com/html/',params={'q':claim}); r.raise_for_status()
            soup=BeautifulSoup(r.text,'html.parser')
            for item in soup.select('.result'):
                link=item.select_one('a.result__a')
                snippet=item.select_one('.result__snippet')
                if not link or not snippet:
                    continue
                href=urljoin('https://duckduckgo.com',link.get('href',''))
                redirect=parse_qs(urlparse(href).query).get('uddg')
                page=redirect[0] if redirect else href
                if urlparse(page).scheme not in ('http','https'):
                    continue
                title=link.get_text(' ',strip=True)
                text=snippet.get_text(' ',strip=True)
                if title and text:
                    results.append({'title':title,'text':text,'url':page,'source':urlparse(page).netloc,'credibility':_source_credibility(page)})
                if len(results)>=16:
                    break
    except httpx.HTTPError as error:
        logger.warning("Public evidence search failed: %s", error)
    return results

def _source_credibility(url: str, source_name: str='') -> float:
    hostname=(urlparse(url).hostname or '').removeprefix('www.').lower()
    matches=[
        (domain,credibility)
        for domain,credibility in TRUSTED_DOMAINS.items()
        if hostname == domain or hostname.endswith('.'+domain)
    ]
    domain_score=max(matches,key=lambda match:len(match[0]))[1] if matches else 0.72
    normalized_name=re.sub(r'[^a-z0-9]','',source_name.casefold())
    name_matches=[
        (re.sub(r'[^a-z0-9]','',name),credibility)
        for name,credibility in TRUSTED_SOURCE_NAMES.items()
        if normalized_name.startswith(re.sub(r'[^a-z0-9]','',name))
    ]
    name_score=max(name_matches,key=lambda match:len(match[0]))[1] if name_matches else 0.0
    return max(domain_score,name_score)

def rank_evidence(claim: str, evidence: list[dict]) -> list[dict]:
    if not evidence: return []
    docs=[claim]+[e.get('text','')+' '+e.get('title','') for e in evidence]
    try:
        if settings.enable_sentence_transformers:
            from sentence_transformers import SentenceTransformer
            model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            emb=model.encode(docs, normalize_embeddings=True)
            # FAISS inner-product search over normalized vectors is cosine-similarity ranking.
            import faiss
            index=faiss.IndexFlatIP(emb.shape[1])
            index.add(emb[1:].astype('float32'))
            distances, indices=index.search(emb[0:1].astype('float32'), len(evidence))
            score_by_index={int(idx):float(score) for idx,score in zip(indices[0],distances[0]) if idx >= 0}
            scores=[score_by_index.get(i,0.0) for i in range(len(evidence))]
        else:
            vec=TfidfVectorizer(stop_words='english', ngram_range=(1,2)).fit_transform(docs)
            scores=cosine_similarity(vec[1:],vec[0]).ravel().tolist()
    except Exception:
        vec=TfidfVectorizer(stop_words='english').fit_transform(docs)
        scores=cosine_similarity(vec[1:],vec[0]).ravel().tolist()
    for e,s in zip(evidence,scores): e['similarity']=round(float(s),4); e['rank_score']=round(float(s)*float(e.get('credibility',.6)),4)
    evidence=[e for e in evidence if e['similarity'] > 0]
    return sorted(evidence,key=lambda x:x['rank_score'],reverse=True)[:5]

def _confidence_from_evidence(evidence: list[dict]) -> float:
    if not evidence:
        return 25.0
    strongest=max(evidence,key=lambda item:item.get('rank_score',0))
    relevant_trusted_sources={
        item.get('source') or urlparse(item.get('url','')).hostname
        for item in evidence
        if item.get('similarity',0)>=0.1 and item.get('credibility',0)>=0.85
    }
    confidence=(
        25
        + 20*strongest.get('similarity',0)
        + 15*strongest.get('credibility',0.6)
        + 5*min(len(relevant_trusted_sources),4)
    )
    return round(max(25,min(80,confidence)),1)

def _evidence_analysis(evidence: list[dict]) -> str:
    if not evidence:
        return 'No sufficiently relevant search results were found, so this claim cannot be assessed from available evidence.'
    strongest=max(evidence,key=lambda item:item.get('rank_score',0))
    relevant_trusted_sources={
        item.get('source') or urlparse(item.get('url','')).hostname
        for item in evidence
        if item.get('similarity',0)>=0.1 and item.get('credibility',0)>=0.85
    }
    source_count=len(relevant_trusted_sources)
    source_text=f'{source_count} independent high-credibility source(s)' if source_count else 'no independent high-credibility sources'
    return (
        f"Reviewed {len(evidence)} relevant result(s); found {source_text}. "
        f"The strongest match is “{strongest.get('title','Evidence')}” "
        f"(relevance {strongest.get('similarity',0)*100:.0f}%, "
        f"source credibility {strongest.get('credibility',0.6)*100:.0f}%). "
        'Search similarity is not proof of truth.'
    )

def classify_claim(claim: str, evidence: list[dict]) -> tuple[str,float,str]:
    confidence=_confidence_from_evidence(evidence)
    analysis=_evidence_analysis(evidence)
    # Optional DeBERTa NLI. It is loaded only when ENABLE_TRANSFORMERS=true to avoid surprise multi-GB downloads.
    if settings.enable_transformers and evidence:
        try:
            from transformers import pipeline
            if not hasattr(classify_claim,'nli'): classify_claim.nli=pipeline('text-classification',model='MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli')
            scores=[]
            for e in evidence[:3]:
                out=classify_claim.nli({'text':e.get('text','')[:1500],'text_pair':claim[:800]}, truncation=True)[0]
                scores.append((out['label'].lower(),float(out['score'])))
            best=max(scores,key=lambda x:x[1])
            label=best[0]
            confidence=round(max(25,min(80,best[1]*100)),1)
            if 'entail' in label or 'support' in label: return 'Supported',confidence,f'NLI model found evidence consistent with the claim. {analysis}'
            if 'contradict' in label: return 'Refuted',confidence,f'NLI model found evidence that appears to contradict the claim. {analysis}'
            return 'Unverified',confidence,f'Evidence was neutral or insufficient to verify the claim. {analysis}'
        except Exception: pass
    if not evidence: return 'Unverified',confidence,f'{analysis} This is not proof that the claim is false.'
    top=evidence[0]; sim=top.get('similarity',0); cred=top.get('credibility',.6)
    # A source's presence and text similarity are not themselves truth proof; conservative rule.
    fact_checks=[
        str(item.get('rating','')).lower()
        for item in evidence
        if item.get('rating') and item.get('similarity',0)>=0.15
    ]
    for fact_rating in fact_checks:
        if re.search(r'\b(false|incorrect|pants on fire|misleading)\b',fact_rating):
            return 'Refuted',confidence,f'A retrieved fact-check labels a related claim as false or misleading; review its wording and date. {analysis}'
        if re.search(r'\b(true|correct|accurate)\b',fact_rating):
            return 'Supported',confidence,f'A retrieved fact-check labels a related claim as true; confirm it refers to the same claim and context. {analysis}'
    if sim >= .28 and cred >= .85:
        return 'Evidence found',confidence,f'Relevant material was found from a higher-credibility source, but similarity alone cannot establish truth. {analysis}'
    return 'Unverified',confidence,f'The available results do not establish whether the claim is true or false. {analysis}'

def explanation(claim: str, verdict: str, evidence: list[dict], reason: str) -> str:
    if settings.anthropic_api_key:
        try:
            import anthropic
            client=anthropic.Anthropic(api_key=settings.anthropic_api_key)
            snippets='\n'.join(f"- {e['title']}: {e.get('text','')[:400]} ({e.get('url','')})" for e in evidence[:3])
            msg=client.messages.create(model='claude-3-5-haiku-latest',max_tokens=220,system='Explain verification results cautiously. Use only supplied evidence. Never claim certainty beyond evidence.',messages=[{'role':'user','content':f'Claim: {claim}\nVerdict: {verdict}\nReason: {reason}\nEvidence:\n{snippets}\nWrite a concise explanation.'}])
            return ''.join(getattr(b,'text','') for b in msg.content).strip()
        except Exception: pass
    if evidence:
        e=evidence[0]
        return f"Assessment: {verdict}. {reason} Best match: {e.get('title','Evidence')} from {e.get('source','source')} (similarity {e.get('similarity',0):.2f}). Read the linked source and check its publication date, scope and original context before drawing a conclusion."
    return f"Assessment: {verdict}. {reason} Try a more specific claim, provide a source URL, or configure Google Fact Check Tools/NewsAPI credentials for broader retrieval."

def verify(text: str, input_type: str='text') -> dict:
    text=clean_text(text)
    claims=split_claims(text)
    if not claims: claims=[text[:1000]]
    results=[]
    for claim in claims:
        ev=TRUSTED_CORPUS.copy()+external_evidence(claim)
        ev=rank_evidence(claim,ev)
        verdict,confidence,reason=classify_claim(claim,ev)
        results.append({'claim':claim,'verdict':verdict,'confidence':confidence,'reason':reason,'evidence':ev,'explanation':explanation(claim,verdict,ev,reason)})
    counts={v:sum(1 for r in results if r['verdict']==v) for v in set(r['verdict'] for r in results)}
    if not results: overall='Unverified'
    elif any(r['verdict']=='Refuted' for r in results): overall='Refuted'
    elif all(r['verdict']=='Supported' for r in results): overall='Supported'
    elif any(r['verdict']=='Evidence found' for r in results): overall='Evidence found'
    else: overall='Unverified'
    conf=round(max(25,min(80,sum(r['confidence'] for r in results)/len(results))),1) if results else 25.0
    return {'input_type':input_type,'verdict':overall,'confidence':conf,'confidence_range':[25,80],'confidence_basis':'Evidence-strength estimate based on result relevance, source credibility and independent-source agreement; not a probability that the verdict is true.','claims':results,'summary':f"Checked {len(results)} claim(s). Results: {json.dumps(counts)}. Confidence estimates evidence strength on a 25–80% scale, not truth probability. Verdicts are evidence-based assistance, not a substitute for expert review.", 'pipeline': ['Preprocessing','Claim extraction','Fact-check/API and knowledge retrieval','Evidence ranking','Claim verification','Verdict and confidence','Explanation and report']}
