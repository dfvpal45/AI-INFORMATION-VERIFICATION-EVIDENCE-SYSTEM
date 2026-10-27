from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import desc
from .db import Base, engine, get_db
from .models import VerificationRecord
from .schemas import VerifyTextRequest, FeedbackRequest
from .pipeline import verify, extract_from_url, extract_image_text
from .config import settings
from .cache import get_cached, set_cached

Base.metadata.create_all(bind=engine)
app=FastAPI(title='AI Information Verification & Evidence System', version='1.0.0', description='Evidence retrieval and claim assessment API')
app.add_middleware(CORSMiddleware, allow_origins=[x.strip() for x in settings.cors_origins.split(',') if x.strip()], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.get('/api/health')
def health(): return {'status':'ok','service':'AI Information Verification & Evidence System','database':'configured','optional_models':{'transformers':settings.enable_transformers,'sentence_transformers':settings.enable_sentence_transformers}}

def save_result(db:Session, original:str, input_type:str, report:dict):
    row=VerificationRecord(input_type=input_type,original_input=original[:30000],claims=report['claims'],verdict=report['verdict'],confidence=report['confidence'],report=report)
    db.add(row); db.commit(); db.refresh(row)
    report['id']=row.id; report['created_at']=str(row.created_at)
    return report

@app.post('/api/verify/text')
def verify_text(req:VerifyTextRequest, db:Session=Depends(get_db)):
    text=req.text
    if req.source_url:
        try: text += '\n\nSource article text: '+extract_from_url(req.source_url)
        except Exception as e: raise HTTPException(status_code=400,detail=f'Could not retrieve source URL: {e}')
    try:
        report=get_cached(text,'text')
        if report is None:
            report=verify(text,'text'); set_cached(text,'text',report)
        return save_result(db,req.text,'text',report)
    except Exception as e: raise HTTPException(status_code=500,detail=f'Verification pipeline error: {type(e).__name__}: {e}')

@app.post('/api/verify/url')
def verify_url(url:str, db:Session=Depends(get_db)):
    try: text=extract_from_url(url)
    except Exception as e: raise HTTPException(status_code=400,detail=f'Could not read URL: {e}')
    if len(text)<20: raise HTTPException(status_code=400,detail='No readable article text found at this URL.')
    try:
        report=get_cached(text,'url')
        if report is None:
            report=verify(text,'url'); set_cached(text,'url',report)
        return save_result(db,url,'url',report)
    except Exception as e: raise HTTPException(status_code=500,detail=f'Verification pipeline error: {type(e).__name__}: {e}')

@app.post('/api/verify/image')
async def verify_image(file:UploadFile=File(...), db:Session=Depends(get_db)):
    if file.content_type not in ('image/png','image/jpeg','image/webp','image/bmp'): raise HTTPException(status_code=415,detail='Upload a PNG, JPG, WEBP or BMP image.')
    data=await file.read()
    if len(data)>10*1024*1024: raise HTTPException(status_code=413,detail='Image must be 10 MB or smaller.')
    text=extract_image_text(data)
    if len(text)<3: raise HTTPException(status_code=400,detail='No text could be extracted from image.')
    report=verify(text,'image'); report['extracted_text']=text
    return save_result(db,'[uploaded image: '+file.filename+']','image',report)

@app.get('/api/history')
def history(limit:int=20, db:Session=Depends(get_db)):
    limit=max(1,min(limit,100)); rows=db.query(VerificationRecord).order_by(desc(VerificationRecord.id)).limit(limit).all()
    return [{'id':r.id,'input_type':r.input_type,'original_input':r.original_input[:180],'verdict':r.verdict,'confidence':r.confidence,'created_at':str(r.created_at)} for r in rows]

@app.get('/api/history/{record_id}')
def get_record(record_id:int,db:Session=Depends(get_db)):
    r=db.get(VerificationRecord,record_id)
    if not r: raise HTTPException(status_code=404,detail='Record not found')
    return {'id':r.id,'input_type':r.input_type,'original_input':r.original_input,'verdict':r.verdict,'confidence':r.confidence,'report':r.report,'user_feedback':r.user_feedback,'created_at':str(r.created_at)}

@app.post('/api/history/{record_id}/feedback')
def feedback(record_id:int, req:FeedbackRequest, db:Session=Depends(get_db)):
    r=db.get(VerificationRecord,record_id)
    if not r: raise HTTPException(status_code=404,detail='Record not found')
    r.user_feedback=req.feedback; db.commit(); return {'status':'saved','id':record_id}

frontend_dist = Path(__file__).resolve().parents[2] / 'frontend' / 'dist'
if not frontend_dist.is_dir():
    frontend_dist = Path(__file__).resolve().parents[1] / 'frontend' / 'dist'
if frontend_dist.is_dir():
    app.mount('/', StaticFiles(directory=frontend_dist, html=True), name='frontend')
