from fastapi import APIRouter, UploadFile, File, HTTPException
from ml_pipeline.audio_analysis import AudioAnalyzer
import shutil
import os
import tempfile

router = APIRouter(
    tags=["audio"]
)

audio_analyzer = AudioAnalyzer()

@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload an audio file.")

    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        result = await audio_analyzer.analyze_audio(temp_file_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
