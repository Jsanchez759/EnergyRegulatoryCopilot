import { useRef, useState } from "react";

import { useDocuments, useUploadDocument } from "./queries";
import Icon from "../../shared/Icon";

function messageForError(error) {
  return error?.message || "No fue posible completar la solicitud.";
}

export default function SourceLibrary({ onClose }) {
  const documentsQuery = useDocuments();
  const uploadMutation = useUploadDocument();
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState("");
  const fileInputRef = useRef(null);
  const documents = documentsQuery.data?.documents || [];

  async function onUpload(event) {
    event.preventDefault();
    if (!selectedFile) {
      setStatus("Selecciona un PDF para indexarlo.");
      return;
    }
    setStatus("Indexando documento…");
    try {
      const result = await uploadMutation.mutateAsync(selectedFile);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      setStatus(`${result.filename} · ${result.chunks_indexed} fragmentos indexados`);
    } catch (error) {
      setStatus(messageForError(error));
    }
  }

  const visibleStatus = status || (documentsQuery.isError ? messageForError(documentsQuery.error) : "");

  return (
    <aside className="sources-pane is-open" aria-label="Fuentes y documentos">
      <div className="sources-header">
        <div><div className="eyebrow">BIBLIOTECA</div><h2>Fuentes de consulta</h2></div>
        <button className="icon-button close-sources" type="button" onClick={onClose} aria-label="Ocultar fuentes">
          <Icon name="close" size={17} />
        </button>
      </div>
      <p className="sources-description">Añade PDFs para buscar respuestas basadas en tus documentos.</p>

      <form className="upload-zone" onSubmit={onUpload}>
        <label className={`file-picker ${selectedFile ? "has-file" : ""}`}>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf,.pdf"
            onChange={(event) => setSelectedFile(event.target.files?.[0] || null)}
          />
          <span className="upload-symbol"><Icon name="plus" size={17} /></span>
          <span><strong>{selectedFile ? selectedFile.name : "Seleccionar PDF"}</strong><small>Hasta 25 MB por archivo</small></span>
        </label>
        <button className="upload-button" type="submit" disabled={uploadMutation.isPending || !selectedFile}>
          {uploadMutation.isPending ? "Indexando…" : "Añadir a la biblioteca"}
        </button>
      </form>
      {visibleStatus && <p className={`document-status ${uploadMutation.isPending ? "is-loading" : ""}`} role="status">{visibleStatus}</p>}

      <div className="library-heading"><span>Documentos indexados</span><span>{documents.length.toString().padStart(2, "0")}</span></div>
      {documents.length ? <div className="document-list">
        {documents.map((doc) => <div key={doc.document_id} className="document-entry">
          <span className="document-icon"><Icon name="file" size={17} /></span>
          <span className="document-copy"><strong title={doc.filename}>{doc.filename}</strong><small>{doc.chunks_indexed} fragmentos indexados</small></span>
        </div>)}
      </div> : <div className="empty-library">
        <Icon name="book" size={19} />
        <span>{documentsQuery.isLoading ? "Cargando biblioteca…" : documentsQuery.isError ? "No se pudo cargar la biblioteca." : "Todavía no hay documentos. Añade un PDF para empezar."}</span>
      </div>}
      <button className="refresh-documents" type="button" onClick={() => void documentsQuery.refetch()}>
        Actualizar biblioteca <span>↻</span>
      </button>
      <div className="sources-footnote"><span className="footnote-line" />Las referencias muestran archivo y página de origen.</div>
    </aside>
  );
}
