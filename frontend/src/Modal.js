import React from 'react';

// A simple Overlay + Modal component
function Modal({ isOpen, onClose, record }) {
  if (!isOpen || !record) return null;
  const { text, metadata, stance, score, spans } = record;

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-lg shadow-lg w-11/12 max-w-xl p-6 relative"
        onClick={e => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-3 right-3 text-gray-500 hover:text-gray-800 cursor-pointer"
        >
          ✕
        </button>

        <h2 className="text-xl font-semibold mb-4">Article Details</h2>

        <div className="space-y-2 text-sm text-gray-700">
          <p>
            <span className="font-medium">Source file:</span>{" "}
            {metadata.filename}
          </p>
          <p>
            <span className="font-medium">News source:</span>{" "}
            {metadata.source}
          </p>
          <p>
            <span className="font-medium">Date:</span> {metadata.date}
          </p>
          <p className="mt-2">
            <span className="font-medium">Text snippet:</span>{" "}
            <em className="text-gray-600">
              {text.length > 150 ? text.slice(0, 150) + "…" : text}
            </em>
          </p>

          <hr className="my-4" />

          <p>
            <span className="font-medium">Stance:</span>{" "}
            <span
              className={
                stance === "STANCE_POS"
                  ? "text-green-700"
                  : stance === "STANCE_NEG"
                  ? "text-red-700"
                  : "text-gray-700"
              }
            >
              {stance.replace("STANCE_", "")}
            </span>{" "}
            (score: {score.toFixed(3)})
          </p>

          <div>
            <span className="font-medium">Extracted entities:</span>
            {spans.length === 0 ? (
              <span className="ml-2 italic text-gray-500">None</span>
            ) : (
              <ul className="mt-1 list-disc list-inside">
                {spans.map((s, i) => (
                  <li key={i} className="mb-1">
                    <span className="font-medium">{s.label}</span> — “
                    <mark className="bg-yellow-100 text-yellow-800">
                      {text.slice(s.start, s.end)}
                    </mark>”{" "}
                    (indices: {s.start}-{s.end})
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Modal;