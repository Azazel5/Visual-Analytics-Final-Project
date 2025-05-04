import sources from "./sources.json";
import React, { useCallback, useState } from 'react';


function App() {
  const [results, setResults] = useState([]);
  const [source, setSource] = useState('All News Today');
  const [entities, setEntities] = useState('PER');
  const [stances, setStances] = useState('STANCE_POS');
  const [minScore, setMinScore] = useState(0.8);
  const [limit, setLimit] = useState(25);
  const [loading, setLoading] = useState(false);


  const handleFetch = useCallback(async () => {
    setLoading(true);

    const qs = new URLSearchParams({
      source,
      entities,
      stances,
      min_score: minScore,
      limit,
    });

    const res = await fetch(`http://127.0.0.1:5005/predictions?${qs}`);
    const data = await res.json();
    setResults(data);
    setLoading(false);

  },  [source, entities, stances, minScore, limit]);

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <h1 className="text-4xl font-extrabold mb-6 text-center">
        News Entity & Stance Explorer
      </h1>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-6 mb-8">
        {/* Source */}
        <div className="flex flex-col">
          {/* <label className="text-sm font-medium text-gray-700">Source</label> */}
          {/* <input
            type="text"
            value={source}
            onChange={(e) => setSource(e.target.value)}
            placeholder="e.g. The Continent"
            className="mt-1 w-64 px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-400"
          /> */}

          <label className="mb-2">Pick a source:</label>
          <select
            className="border rounded p-2"
            value={source}
            onChange={e => setSource(e.target.value)}
          >
            <option value="">All sources</option>
            {sources.map(src => (
              <option key={src} value={src}>{src}</option>
            ))}
          </select>
        </div>

        {/* Entity */}
        <div className="flex flex-col">
          <label className="text-sm font-medium text-gray-700">Entity</label>
          <select
            value={entities}
            onChange={(e) => setEntities(e.target.value)}
            className="mt-1 w-48 px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-400"
          >
            <option value="">Any</option>
            <option value="PER">Person</option>
            <option value="LOC">Location</option>
            <option value="ORG">Organization</option>
          </select>
        </div>

        {/* Stance */}
        <div className="flex flex-col">
          <label className="text-sm font-medium text-gray-700">Stance</label>
          <select
            value={stances}
            onChange={(e) => setStances(e.target.value)}
            className="mt-1 w-48 px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-400"
          >
            <option value="">Any</option>
            <option value="STANCE_POS">Positive</option>
            <option value="STANCE_NEU">Neutral</option>
            <option value="STANCE_NEG">Negative</option>
          </select>
        </div>

        {/* Min Score */}
        <div className="flex flex-col items-center">
          <label className="text-sm font-medium text-gray-700">
            Min Confidence: {minScore.toFixed(2)}
          </label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={minScore}
            onChange={(e) => setMinScore(Number(e.target.value))}
            className="mt-1 w-48"
          />
        </div>

        {/* Limit */}
        <div className="flex flex-col">
          <label className="text-sm font-medium text-gray-700">Max Results</label>
          <input
            type="number"
            min="1"
            max="500"
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="mt-1 w-24 px-3 py-2 rounded-lg border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-400"
          />
        </div>

        {/* Fetch Button */}
        <button
          onClick={handleFetch}
          disabled={loading}
          className="h-12 px-6 ml-4 bg-indigo-600 text-white font-semibold rounded-lg shadow hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-400"
        >
          {loading ? 'Loading…' : 'Generate Predictions'}
        </button>
      </div>

      {/* Results */}
      <div className="grid gap-6 grid-cols-1 sm:grid-cols-2 lg:grid-cols-3">
        {results.map((rec, idx) => (
          <div
            key={idx}
            className="p-4 bg-white rounded-2xl shadow hover:shadow-lg transform hover:-translate-y-1 transition duration-200"
          >
            <p className="font-medium text-gray-800 mb-2 truncate">{rec.text}</p>

            <div className="flex flex-wrap gap-2 mb-3">
              <span
                className={`px-2 py-1 text-xs font-medium rounded-full ${rec.stance === 'STANCE_POS'
                  ? 'bg-green-100 text-green-800'
                  : rec.stance === 'STANCE_NEG'
                    ? 'bg-red-100 text-red-800'
                    : 'bg-gray-100 text-gray-800'
                  }`}
              >
                {rec.stance.replace('STANCE_', '')}
              </span>
              

              {rec.spans && (
                  <span
                    key={`${rec.metadata.filename}-${rec.spans[0].start}`}
                    className="px-2 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800"
                  >
                    {rec.spans[0].label}
                  </span>
              )}
            </div>

            <div className="flex justify-between text-sm text-gray-500">
              <span>{rec.metadata.source}</span>
              <span>{rec.metadata.date}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;