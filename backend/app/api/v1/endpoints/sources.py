from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.dependencies import CurrentUserDep, DocumentServiceDep, rate_limit
from app.core.config import settings
from app.schemas.agent import IndexedDocument, IndexedDocumentsResponse
from app.services.document_service import DocumentProcessingError, DuplicateDocumentError
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


@router.post(
    "",
    summary="Upload and index a PDF source",
    status_code=status.HTTP_201_CREATED,
    dependencies=[rate_limit("upload", lambda: settings.RATE_LIMIT_UPLOAD)],
)
async def upload_source(
    file: Annotated[UploadFile, File()],
    document_service: DocumentServiceDep,
    current: CurrentUserDep,
) -> IndexedDocument:
    filename = file.filename or "document.pdf"
    try:
        file_bytes = await file.read(settings.MAX_UPLOAD_BYTES + 1)
        if len(file_bytes) > settings.MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="The PDF exceeds the 25 MiB upload limit.")
        if not file_bytes:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        return await document_service.ingest_pdf(file_bytes, filename, current.user.id)
    except DuplicateDocumentError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DocumentProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        await file.close()


@router.delete(
    "/{document_id}",
    summary="Delete an indexed PDF and its chunks",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_source(
    document_id: str,
    document_service: DocumentServiceDep,
    current: CurrentUserDep,
) -> None:
    if not await document_service.delete_document(current.user.id, document_id):
        raise HTTPException(status_code=404, detail="Document not found.")
