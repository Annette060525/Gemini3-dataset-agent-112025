
import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI } from "@google/genai";
import { 
  LayoutDashboard, 
  Upload, 
  Search, 
  BarChart2, 
  Settings, 
  BrainCircuit, 
  FileText, 
  Database, 
  Play, 
  AlertCircle,
  Loader2,
  Table as TableIcon,
  X,
  Key
} from 'lucide-react';

// --- Types & Constants ---

type Row = Record<string, any>;

interface Dataset {
  name: string;
  data: Row[];
  columns: string[];
}

interface AgentConfig {
  model: string;
  maxTokens: number;
  temperature: number;
  systemInstruction: string;
}

type AgentMode = 'semantic_search' | 'visualization' | 'insight';

const AVAILABLE_MODELS = [
  { value: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
  { value: 'gemini-2.5-flash-lite-latest', label: 'Gemini 2.5 Flash Lite' },
  { value: 'gemini-2.5-flash-thinking-exp-0121', label: 'Gemini 2.5 Thinking (Exp)' },
  { value: 'gemini-2.5-flash-image', label: 'Gemini 2.5 Flash Image' },
];

const DEFAULT_AGENTS: Record<AgentMode, string> = {
  semantic_search: `You are a semantic search engine. 
Your task is to analyze the provided dataset and find rows that match the user's query based on meaning and intent.
Return a JSON object with a property "indices" containing an array of the original row indices (0-based) that match the query.
Example output: { "indices": [0, 5, 12] }
If no matches found, return { "indices": [] }.`,
  visualization: `You are a data visualization expert using Vega-Lite v5.
Analyze the dataset structure and the user's request.
Generate a valid Vega-Lite v5 JSON specification to visualize the data.
The data values will be injected automatically, so in your spec, simply use "data": { "name": "dataset" }.
Focus on clear, aesthetic charts. Handle messy data gracefully.
Return ONLY the JSON specification.`,
  insight: `You are a data mining and analysis expert.
Analyze the provided dataset sample and the user's query to provide deep insights, patterns, and anomalies.
Format your response in Markdown with clear headers, bullet points, and actionable conclusions.`
};

// --- Helper Functions ---

const parseCSV = (text: string): Row[] => {
  const lines = text.split('\n').filter(l => l.trim());
  if (lines.length < 2) return [];
  
  // Robust header parsing
  const headers = lines[0].split(',').map(h => (h || '').trim().replace(/^"|"$/g, ''));
  
  return lines.slice(1).map(line => {
    // Robust value parsing
    const values = line.split(',').map(v => (v || '').trim().replace(/^"|"$/g, ''));
    const row: Row = {};
    headers.forEach((h, i) => {
      if (h) row[h] = values[i] || '';
    });
    return row;
  });
};

// --- Components ---

const App = () => {
  // --- State ---
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [query, setQuery] = useState('');
  const [mode, setMode] = useState<AgentMode>('semantic_search');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  
  // API Key State - Initialize with environment variable if available, else empty
  const [apiKey, setApiKey] = useState<string>(() => {
    try {
      return process.env.API_KEY || '';
    } catch {
      return '';
    }
  });

  // Config State
  const [config, setConfig] = useState<AgentConfig>({
    model: 'gemini-2.5-flash',
    maxTokens: 2000,
    temperature: 0.7,
    systemInstruction: DEFAULT_AGENTS['semantic_search']
  });
  const [showSettings, setShowSettings] = useState(false);

  // Update prompt when mode changes
  useEffect(() => {
    setConfig(prev => ({ ...prev, systemInstruction: DEFAULT_AGENTS[mode] }));
    setResults(null);
    setError(null);
  }, [mode]);

  // --- Handlers ---

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const text = event.target?.result as string;
      let data: Row[] = [];
      let columns: string[] = [];

      try {
        if (file.name.endsWith('.json')) {
          data = JSON.parse(text);
        } else if (file.name.endsWith('.csv') || file.name.endsWith('.txt')) {
          data = parseCSV(text);
        } else {
          setError("Unsupported file format. Please use CSV, JSON, or TXT.");
          return;
        }

        if (!Array.isArray(data) || data.length === 0) {
           if (typeof data === 'object' && data !== null && !Array.isArray(data)) {
             data = [data];
           } else {
             throw new Error("Invalid data structure");
           }
        }

        columns = Object.keys(data[0] || {});
        setDataset({ name: file.name, data, columns });
        setError(null);
        setResults(null);
      } catch (err) {
        console.error(err);
        setError("Failed to parse file. Ensure it is valid JSON or CSV.");
      }
    };
    reader.readAsText(file);
  };

  const executeAgent = async () => {
    if (!apiKey) {
      setError("Please enter your Gemini API Key in the Configuration settings.");
      setShowSettings(true);
      return;
    }

    if (!dataset) {
      setError("Please upload a dataset first.");
      return;
    }
    if (!query.trim()) {
      setError("Please enter a query or instruction.");
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const ai = new GoogleGenAI({ apiKey: apiKey });
      
      // Prepare context (Truncate if too large for a demo)
      const sampleSize = 50; 
      const dataSample = dataset.data.slice(0, sampleSize);
      const context = `
Dataset Columns: ${dataset.columns.join(', ')}
Total Rows: ${dataset.data.length}
Data Sample (First ${sampleSize} rows):
${JSON.stringify(dataSample, null, 2)}
      `;

      const finalPrompt = `
${config.systemInstruction}

--- USER CONTEXT ---
${context}

--- USER QUERY ---
${query}
      `;

      // Call API
      if (mode === 'semantic_search') {
        const response = await ai.models.generateContent({
          model: config.model,
          contents: finalPrompt,
          config: {
            responseMimeType: "application/json",
            temperature: config.temperature,
            maxOutputTokens: config.maxTokens,
          }
        });
        
        const jsonText = response.text;
        const parsed = JSON.parse(jsonText || "{}");
        const indices = parsed.indices || [];
        
        // Filter dataset
        const filteredData = dataset.data.filter((_, idx) => indices.includes(idx));
        setResults({ type: 'table', data: filteredData, count: filteredData.length });

      } else if (mode === 'visualization') {
        const response = await ai.models.generateContent({
          model: config.model,
          contents: finalPrompt,
          config: {
            responseMimeType: "application/json",
            temperature: config.temperature,
            maxOutputTokens: config.maxTokens,
          }
        });
        
        const spec = JSON.parse(response.text || "{}");
        // Inject full data (or a larger sample) into the spec for client-side rendering
        const renderData = dataset.data.slice(0, 2000);
        spec.data = { values: renderData };
        setResults({ type: 'chart', spec });

      } else if (mode === 'insight') {
        const response = await ai.models.generateContent({
          model: config.model,
          contents: finalPrompt,
          config: {
            temperature: config.temperature,
            maxOutputTokens: config.maxTokens,
          }
        });
        
        const textContent = response.text || "No insights could be generated.";
        setResults({ type: 'text', content: textContent });
      }

    } catch (err: any) {
      console.error(err);
      setError(err.message || "An error occurred during execution.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-50 text-gray-900 font-sans overflow-hidden">
      
      {/* Sidebar */}
      <aside className="w-80 bg-white border-r border-gray-200 flex flex-col shadow-sm z-10">
        <div className="p-6 border-b border-gray-100 flex items-center space-x-3">
          <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white">
            <BrainCircuit size={20} />
          </div>
          <h1 className="font-bold text-xl tracking-tight text-indigo-900">Agentic Miner</h1>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          
          {/* Data Upload Section */}
          <section>
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 flex items-center">
              <Database size={14} className="mr-2" /> Data Source
            </h2>
            
            {!dataset ? (
              <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-xl cursor-pointer bg-gray-50 hover:bg-indigo-50 hover:border-indigo-300 transition-all group">
                <div className="flex flex-col items-center justify-center pt-5 pb-6 text-gray-400 group-hover:text-indigo-500">
                  <Upload className="w-8 h-8 mb-2" />
                  <p className="text-sm font-medium">Click to upload CSV/JSON</p>
                </div>
                <input type="file" className="hidden" onChange={handleFileUpload} accept=".csv,.json,.txt" />
              </label>
            ) : (
              <div className="bg-indigo-50 rounded-xl p-4 border border-indigo-100 relative group">
                <button 
                  onClick={() => { setDataset(null); setResults(null); }}
                  className="absolute top-2 right-2 p-1 bg-white rounded-full shadow-sm text-gray-400 hover:text-red-500 opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X size={14} />
                </button>
                <div className="flex items-center space-x-3 mb-2">
                  <div className="p-2 bg-white rounded-lg text-indigo-600">
                    <FileText size={20} />
                  </div>
                  <div className="overflow-hidden">
                    <p className="font-semibold text-sm truncate text-gray-900">{dataset.name}</p>
                    <p className="text-xs text-gray-500">{dataset.data.length.toLocaleString()} rows • {dataset.columns.length} cols</p>
                  </div>
                </div>
                <div className="flex flex-wrap gap-1 mt-2">
                   {dataset.columns.slice(0, 3).map(c => (
                     <span key={c} className="text-[10px] px-2 py-0.5 bg-white border border-indigo-100 rounded-full text-gray-600">{c}</span>
                   ))}
                   {dataset.columns.length > 3 && <span className="text-[10px] text-gray-400 px-1">+{dataset.columns.length - 3}</span>}
                </div>
              </div>
            )}
          </section>

          {/* Mode Selection */}
          <section>
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4 flex items-center">
              <LayoutDashboard size={14} className="mr-2" /> Agent Mode
            </h2>
            <div className="space-y-2">
              <button 
                onClick={() => setMode('semantic_search')}
                className={`w-full flex items-center p-3 rounded-xl transition-all text-sm font-medium border ${mode === 'semantic_search' ? 'bg-indigo-600 text-white border-indigo-600 shadow-md' : 'bg-white text-gray-700 border-gray-200 hover:border-indigo-300 hover:bg-gray-50'}`}
              >
                <Search size={18} className="mr-3" /> Semantic Search
              </button>
              <button 
                onClick={() => setMode('visualization')}
                className={`w-full flex items-center p-3 rounded-xl transition-all text-sm font-medium border ${mode === 'visualization' ? 'bg-indigo-600 text-white border-indigo-600 shadow-md' : 'bg-white text-gray-700 border-gray-200 hover:border-indigo-300 hover:bg-gray-50'}`}
              >
                <BarChart2 size={18} className="mr-3" /> Visualization
              </button>
              <button 
                onClick={() => setMode('insight')}
                className={`w-full flex items-center p-3 rounded-xl transition-all text-sm font-medium border ${mode === 'insight' ? 'bg-indigo-600 text-white border-indigo-600 shadow-md' : 'bg-white text-gray-700 border-gray-200 hover:border-indigo-300 hover:bg-gray-50'}`}
              >
                <BrainCircuit size={18} className="mr-3" /> Data Insights
              </button>
            </div>
          </section>

          {/* Settings Toggle */}
          <section>
            <button 
              onClick={() => setShowSettings(!showSettings)}
              className="w-full flex items-center justify-between text-xs font-semibold text-gray-500 hover:text-indigo-600 transition-colors"
            >
              <span className="flex items-center"><Settings size={14} className="mr-2" /> CONFIGURATION</span>
              <span className={`transform transition-transform ${showSettings ? 'rotate-180' : ''}`}>▼</span>
            </button>
            
            {showSettings && (
              <div className="mt-4 space-y-4 p-4 bg-gray-100 rounded-xl border border-gray-200 animate-in fade-in slide-in-from-top-2">
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1 flex items-center">
                    <Key size={12} className="mr-1" /> API Key
                  </label>
                  <input 
                    type="password" 
                    placeholder="Enter Gemini API Key"
                    className="w-full text-xs p-2 rounded border border-gray-300 bg-white focus:ring-2 focus:ring-indigo-500 outline-none"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">Model</label>
                  <select 
                    className="w-full text-xs p-2 rounded border border-gray-300 bg-white focus:ring-2 focus:ring-indigo-500 outline-none"
                    value={config.model}
                    onChange={(e) => setConfig({...config, model: e.target.value})}
                  >
                    {AVAILABLE_MODELS.map(m => (
                      <option key={m.value} value={m.value}>{m.label}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">Max Tokens ({config.maxTokens})</label>
                  <input 
                    type="range" min="100" max="8192" step="100"
                    className="w-full h-1 bg-gray-300 rounded-lg appearance-none cursor-pointer"
                    value={config.maxTokens}
                    onChange={(e) => setConfig({...config, maxTokens: parseInt(e.target.value)})}
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-gray-700 mb-1">System Instruction</label>
                  <textarea 
                    className="w-full text-xs p-2 rounded border border-gray-300 bg-white focus:ring-2 focus:ring-indigo-500 outline-none"
                    rows={4}
                    value={config.systemInstruction}
                    onChange={(e) => setConfig({...config, systemInstruction: e.target.value})}
                  />
                </div>
              </div>
            )}
          </section>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 bg-gray-50 text-[10px] text-gray-500 text-center">
          Powered by Gemini 2.5 Flash
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col h-full relative overflow-hidden">
        
        {/* Header Area */}
        <div className="p-8 pb-4">
          <div className="max-w-4xl mx-auto w-full">
            <h2 className="text-2xl font-bold text-gray-800 mb-2">
              {mode === 'semantic_search' && "Find what matters."}
              {mode === 'visualization' && "Visualize your data."}
              {mode === 'insight' && "Uncover deep insights."}
            </h2>
            <p className="text-gray-500 mb-6">
              {mode === 'semantic_search' && "Use natural language to find specific rows in your dataset without complex filtering."}
              {mode === 'visualization' && "Describe the chart you want to see, and our AI agent will build it for you."}
              {mode === 'insight' && "Ask high-level questions about trends, anomalies, and summary statistics."}
            </p>

            <div className="relative shadow-lg rounded-2xl">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                className="block w-full pl-11 pr-32 py-4 bg-white border-0 rounded-2xl text-gray-900 placeholder-gray-400 focus:ring-2 focus:ring-indigo-500 text-lg"
                placeholder={
                  mode === 'semantic_search' ? "e.g., 'Find users who signed up last week with high activity'" :
                  mode === 'visualization' ? "e.g., 'Show me a scatter plot of revenue vs time'" :
                  "e.g., 'What are the top 3 drivers of customer churn?'"
                }
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && executeAgent()}
              />
              <div className="absolute inset-y-0 right-2 flex items-center">
                <button 
                  onClick={executeAgent}
                  disabled={loading || !dataset}
                  className={`flex items-center px-4 py-2 rounded-xl text-sm font-bold text-white transition-all ${loading || !dataset ? 'bg-gray-300 cursor-not-allowed' : 'bg-indigo-600 hover:bg-indigo-700 shadow-md hover:shadow-lg'}`}
                >
                  {loading ? <Loader2 className="animate-spin w-5 h-5" /> : <span className="flex items-center">Run <Play className="ml-2 w-4 h-4 fill-current" /></span>}
                </button>
              </div>
            </div>
            {error && (
               <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg flex items-center text-sm text-red-700">
                 <AlertCircle className="w-4 h-4 mr-2 flex-shrink-0" />
                 {error}
               </div>
            )}
          </div>
        </div>

        {/* Results Area */}
        <div className="flex-1 overflow-y-auto p-8 pt-0">
          <div className="max-w-4xl mx-auto w-full pb-20">
            
            {!results && !loading && !dataset && (
              <div className="flex flex-col items-center justify-center h-64 text-center text-gray-400">
                <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                  <TableIcon size={32} className="opacity-50" />
                </div>
                <p>Upload data to get started.</p>
              </div>
            )}

            {!results && !loading && dataset && (
               <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden animate-in fade-in slide-in-from-bottom-4">
                 <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
                   <h3 className="font-semibold text-gray-800 flex items-center">
                     <Database size={18} className="mr-2 text-indigo-600"/> Data Preview
                   </h3>
                   <span className="text-xs text-gray-500 bg-white px-2 py-1 rounded border border-gray-200">
                     First 20 of {dataset.data.length.toLocaleString()} rows
                   </span>
                 </div>
                 <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                        <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-b">
                          <tr>
                            {dataset.columns.map((col) => (
                              <th key={col} className="px-6 py-3 font-medium tracking-wider whitespace-nowrap">{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {dataset.data.slice(0, 20).map((row: Row, i: number) => (
                            <tr key={i} className="hover:bg-indigo-50/50 transition-colors">
                              {dataset.columns.map((col) => (
                                <td key={`${i}-${col}`} className="px-6 py-3 text-gray-700 whitespace-nowrap max-w-xs truncate" title={String(row[col] ?? '')}>
                                  {String(row[col] ?? '')}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                 </div>
               </div>
            )}

            {loading && (
              <div className="space-y-4 animate-pulse">
                <div className="h-8 bg-gray-200 rounded w-1/4"></div>
                <div className="h-64 bg-gray-200 rounded-xl"></div>
              </div>
            )}

            {results && (
              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden animate-in fade-in slide-in-from-bottom-4">
                
                {/* Result Header */}
                <div className="px-6 py-4 border-b border-gray-100 flex justify-between items-center bg-gray-50">
                   <h3 className="font-semibold text-gray-800 flex items-center">
                     {results.type === 'table' && <><TableIcon size={18} className="mr-2 text-indigo-600"/> Matching Rows ({results.count})</>}
                     {results.type === 'chart' && <><BarChart2 size={18} className="mr-2 text-indigo-600"/> Generated Visualization</>}
                     {results.type === 'text' && <><BrainCircuit size={18} className="mr-2 text-indigo-600"/> Analysis Report</>}
                   </h3>
                   <span className="text-xs text-gray-400 uppercase font-medium">{config.model}</span>
                </div>

                {/* Table Result */}
                {results.type === 'table' && (
                  <div className="overflow-x-auto">
                    {results.data.length > 0 ? (
                      <table className="w-full text-sm text-left">
                        <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-b">
                          <tr>
                            {dataset && dataset.columns.map((col) => (
                              <th key={col} className="px-6 py-3 font-medium tracking-wider whitespace-nowrap">{col}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-100">
                          {results.data.map((row: Row, i: number) => (
                            <tr key={i} className="hover:bg-indigo-50/50 transition-colors">
                              {dataset && dataset.columns.map((col) => (
                                <td key={`${i}-${col}`} className="px-6 py-3 text-gray-700 whitespace-nowrap max-w-xs truncate" title={String(row[col] ?? '')}>
                                  {String(row[col] ?? '')}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <div className="p-8 text-center text-gray-500">No matching rows found.</div>
                    )}
                  </div>
                )}

                {/* Chart Result */}
                {results.type === 'chart' && (
                  <div className="p-6 flex justify-center min-h-[400px]">
                    <VisualizationEmbed spec={results.spec} />
                  </div>
                )}

                {/* Insight Result */}
                {results.type === 'text' && (
                  <div className="p-8 prose prose-indigo max-w-none">
                     {results.content ? (
                       <div dangerouslySetInnerHTML={{ __html: results.content.replace(/\n/g, '<br/>') }} />
                     ) : (
                       <p className="text-gray-500 italic">No analysis output generated.</p>
                     )}
                  </div>
                )}

              </div>
            )}
          </div>
        </div>

      </main>
    </div>
  );
};

// --- Visualization Helper Component ---
const VisualizationEmbed = ({ spec }: { spec: any }) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current && spec) {
      // @ts-ignore - vegaEmbed is loaded via CDN in index.html
      if (window.vegaEmbed) {
        // @ts-ignore
        window.vegaEmbed(ref.current, spec, { actions: true, theme: 'excel' }).catch(console.warn);
      }
    }
  }, [spec]);

  return <div ref={ref} className="w-full h-full" />;
};

const root = createRoot(document.getElementById('root')!);
root.render(<App />);
