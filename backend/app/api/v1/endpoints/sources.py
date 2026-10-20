from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.dependencies import CurrentUserDep, DocumentServiceDep
from app.core.config import settings
from app.schemas.agent import DocumentUploadResponse, IndexedDocumentsResponse
from app.services.document_service import DocumentProcessingError
from app.services.errors import ProviderUnavailableError

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("", summary="List indexed PDF sources")
async def list_sources(
    document_service: DocumentServiceDep,
    current: CurrentUserDep,
) -> IndexedDocumentsResponse:
    return IndexedDocumentsResponse(
        documents=await document_service.list_documents(current.user.id)
    )


@router.post("", summary="Upload and index a PDF source")
async def upload_source(
    file: Annotated[UploadFile, File()],
    document_service: DocumentServiceDep,
    current: CurrentUserDep,
) -> DocumentUploadResponse:
    filename = file.filename or "document.pdf"
    try:
        file_bytes = await file.read(settings.MAX_UPLOAD_BYTES + 1)
        if len(file_bytes) > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="The PDF exceeds the 25 MiB upload limit.")
        if not file_bytes:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        document_id, chunks_indexed = await document_service.ingest_pdf(
            file_bytes, filename, current.user.id
        )
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await file.close()

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        chunks_indexed=chunks_indexed,
    )
