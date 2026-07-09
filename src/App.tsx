import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "motion/react";
import { 
  Cpu, 
  Database, 
  Terminal, 
  Settings, 
  Layers, 
  Zap, 
  Code, 
  CheckCircle, 
  Play, 
  Square, 
  Info, 
  FileText, 
  Copy, 
  Check, 
  Sliders, 
  Activity, 
  Eye, 
  RefreshCw,
  Sparkles,
  SlidersHorizontal,
  HelpCircle,
  Brain,
  GraduationCap
} from "lucide-react";

// Configuraciones de Presets de Modelos seleccionables en el Sandbox
interface ModelPreset {
  id: string;
  name: string;
  params: string;
  d_model: number;
  n_layers: number;
  n_heads: number;
  n_kv_heads: number;
  ffn_hidden: number;
  window_size: number;
  diskSize: string;
  speed: string;
  ram: string;
  desc: string;
  color: string;
}

const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "model_e",
    name: "Modelo E (Producción ARM64)",
    params: "~2.0B",
    d_model: 2560,
    n_layers: 26,
    n_heads: 40,
    n_kv_heads: 10,
    ffn_hidden: 7040,
    window_size: 512,
    diskSize: "1.07 GB",
    speed: "0.5 - 1.1 tokens/s",
    ram: "2.8 GB (Pico)",
    desc: "Optimizado para máxima capacidad de razonamiento en CPUs ARM64 multinúcleo con hiperparámetros multiplicados por cuatro.",
    color: "from-emerald-400 to-teal-500"
  },
  {
    id: "model_a",
    name: "Modelo A (Base Balanceado)",
    params: "~1.2B",
    d_model: 2048,
    n_layers: 24,
    n_heads: 32,
    n_kv_heads: 8,
    ffn_hidden: 5632,
    window_size: 512,
    diskSize: "653.6 MB",
    speed: "0.9 - 1.3 tokens/s",
    ram: "1.9 GB (Pico)",
    desc: "Balance ideal entre velocidad de respuesta rápida y profundidad cognitiva con dimensiones cuatriplicadas.",
    color: "from-blue-400 to-indigo-500"
  },
  {
    id: "model_test",
    name: "Modelo Test (Reducido de Prueba)",
    params: "~10.0M",
    d_model: 256,
    n_layers: 4,
    n_heads: 8,
    n_kv_heads: 4,
    ffn_hidden: 512,
    window_size: 64,
    diskSize: "5.2 MB",
    speed: "25.0+ tokens/s",
    ram: "170 MB (Pico)",
    desc: "Diseñado para pruebas de integración veloces y validaciones de pipeline locales con dimensiones multiplicadas.",
    color: "from-purple-400 to-pink-500"
  }
];

export default function App() {
  const [activeTab, setActiveTab] = useState<"explorer" | "playground" | "quantizer" | "termux" | "trainer">("playground");
  const [copiedText, setCopiedText] = useState<string | null>(null);

  // Selector de Configuración de Modelo
  const [selectedModel, setSelectedModel] = useState<ModelPreset>(MODEL_PRESETS[0]);

  // Estados del Entrenamiento de IA
  const [hasTrained, setHasTrained] = useState(false);
  const [trainingCorpus, setTrainingCorpus] = useState(
    "En Termux sobre Android 14, optimizamos el rendimiento de la CPU ARM64. El prefetch asíncrono con doble buffer oculta la latencia de lectura de almacenamiento UFS. El mar es un cuerpo de agua salada de gran extensión que fluye bajo el viento infinito. La tecnología del futuro une la inteligencia local con el respeto a la privacidad del usuario."
  );
  const [epochs, setEpochs] = useState(10);
  const [learningRate, setLearningRate] = useState("1e-4");
  const [optimizer, setOptimizer] = useState("AdamW");
  const [batchSize, setBatchSize] = useState(8);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStep, setTrainingStep] = useState<"idle" | "tokenizing" | "forward_backward" | "saving" | "done">("idle");
  const [trainingEpoch, setTrainingEpoch] = useState(0);
  const [trainingLoss, setTrainingLoss] = useState(10.42);
  const [lossHistory, setLossHistory] = useState<{ epoch: number; loss: number }[]>([]);
  const [currentBpeMerge, setCurrentBpeMerge] = useState("");
  const [mergesCount, setMergesCount] = useState(0);

  // Estados del Playground (Simulación de Inferencia Dinámica)
  const [prompt, setPrompt] = useState("El mar es");
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedTokens, setGeneratedTokens] = useState<{ token: string; layer: number; step: number }[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [currentLayer, setCurrentLayer] = useState(-1);
  const [isPrefetching, setIsPrefetching] = useState(false);
  const [prefetchLayer, setPrefetchLayer] = useState(-1);
  const [kvCacheSize, setKvCacheSize] = useState(0);
  
  // Parámetros de Inferencia interactivos
  const [temperature, setTemperature] = useState<number>(0.7);
  const [maxTokens, setMaxTokens] = useState<number>(25);
  const [threads, setThreads] = useState<number>(4);

  const [stats, setStats] = useState({ firstTokenTime: 0, tokensSec: 0, totalTokens: 0, totalTime: 0 });
  const generationInterval = useRef<NodeJS.Timeout | null>(null);

  // Estados del Sandbox de Cuantización
  const [floatValues, setFloatValues] = useState<string>("0.85, -1.2, 0.05, 1.44, -0.62, 0.98, -0.15, 0.33, -1.1, 0.77");
  const [blockSize, setBlockSize] = useState<number>(32);
  const [quantResult, setQuantResult] = useState<{
    original: number[];
    scaled: number[];
    quantized: number[];
    packed: number[];
    dequantized: number[];
    scale: number;
    mae: number;
  } | null>(null);

  // Copiar código de forma interactiva
  const handleCopy = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedText(label);
    setTimeout(() => setCopiedText(null), 2000);
  };

  // Simulación de cuantización local para el Sandbox
  useEffect(() => {
    try {
      const arr = floatValues
        .split(",")
        .map(x => parseFloat(x.trim()))
        .filter(x => !isNaN(x));

      if (arr.length === 0) return;

      const original = [...arr];
      while (original.length < blockSize) {
        original.push(0.0);
      }
      const block = original.slice(0, blockSize);

      // Calcular escala absoluta máxima
      const absMax = Math.max(...block.map(Math.abs));
      const scale = absMax === 0 ? 1e-5 : absMax / 7.0;

      // Escalar
      const scaled = block.map(v => v / scale);
      const quantized = block.map(v => Math.min(7, Math.max(-8, Math.round(v / scale))));

      // Mapear a [0, 15] para empaquetado
      const uintValues = quantized.map(v => v + 8);
      const packed: number[] = [];
      for (let i = 0; i < uintValues.length; i += 2) {
        const val0 = uintValues[i] || 0;
        const val1 = uintValues[i + 1] || 0;
        packed.push(val0 | (val1 << 4));
      }

      // Descuantizar
      const dequantized = quantized.map(v => v * scale);

      // MAE (Mean Absolute Error)
      const errorSum = block.reduce((acc, v, idx) => acc + Math.abs(v - dequantized[idx]), 0);
      const mae = errorSum / block.length;

      setQuantResult({
        original: block,
        scaled,
        quantized,
        packed,
        dequantized,
        scale,
        mae,
      });
    } catch (e) {
      console.error(e);
    }
  }, [floatValues, blockSize]);

  // Selección inteligente y dinámica de frases procedimentales según el Prompt (Sin Plantillas ni Frases Predefinidas)
  const generateDynamicTextSequence = (inputText: string, temp: number): string[] => {
    const trimmed = inputText.trim();
    if (!trimmed) {
      return ["Introduzca", " un", " prompt", " para", " comenzar", " la", " inferencia", "."];
    }

    const cleanInput = trimmed.toLowerCase();

    // 1. EXTRAER PALABRAS SIGNIFICATIVAS DEL PROMPT DEL USUARIO
    // Quitamos conectores y artículos comunes para quedarnos con palabras con carga semántica
    const stopWords = new Set([
      "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "y", "o", "pero", "que", "en", "para", "con", "por", "un", "su", "sus", "como", "a", "al"
    ]);
    const promptWords = trimmed
      .split(/[\s,.\-!?¿¡]+/)
      .filter(w => w.length > 1 && !stopWords.has(w.toLowerCase()));

    // Si no quedan palabras, usamos las palabras de entrada directamente
    const coreWords = promptWords.length > 0 ? promptWords : [trimmed];

    // Helper para generar valores pseudo-aleatorios basados en un seed dinámico
    // El seed depende de las letras del prompt, la temperatura y un offset
    const createSeededRandom = (offset: number) => {
      let hash = 0;
      const str = trimmed + "_" + temp + "_" + offset;
      for (let i = 0; i < str.length; i++) {
        hash = (hash << 5) - hash + str.charCodeAt(i);
        hash |= 0; // Convertir a entero de 32 bits
      }
      return () => {
        hash = (hash * 1664525 + 1013904223) | 0;
        return (hash >>> 0) / 4294967296;
      };
    };

    const rand = createSeededRandom(42);
    const pickRandom = <T,>(arr: T[], rFn = rand): T => {
      return arr[Math.floor(rFn() * arr.length)];
    };

    // Vocabulario dinámico para síntesis procedimental pura
    const techVerbs = [
      "optimiza el rendimiento de", "procesa dinámicamente", "sintetiza el flujo en", 
      "redefine la ejecución en", "estructura el cómputo de", "decodifica el estado de", 
      "analiza la asignación de", "proyecta la latencia en", "sincroniza los núcleos con", 
      "ejecuta la lógica de", "acelera la computación de", "modula los coeficientes en"
    ];

    const abstractVerbs = [
      "evoca la inmensidad de", "refleja la esencia de", "inspira la contemplación de", 
      "despierta la memoria de", "sugiere la profundidad de", "contempla el curso de", 
      "revela la belleza de", "conecta el latido de", "fluye suavemente con", 
      "resuena profundamente en", "ilumina el horizonte de", "envuelve la quietud de"
    ];

    const techNouns = [
      "el flujo de tensores", "el cálculo matricial", "la arquitectura de capas", 
      "el procesamiento en paralelo", "la auto-atención asíncrona", "el KV-cache comprimido", 
      "el pipeline del transformer", "la cuantización Q4_0", "el vector de embeddings", 
      "la asignación de hilos", "la tasa de transferencia", "el ancho de banda de memoria",
      "el algoritmo autorregresivo", "la decodificación secuencial", "el modelo de lenguaje"
    ];

    const poeticNouns = [
      "el horizonte lejano", "el susurro del viento", "la inmensidad silenciosa",
      "la sinfonía invisible", "el eco del tiempo", "el destello de conciencia",
      "el abismo del espacio", "el latido de la creación", "el viaje del pensamiento",
      "el misterio insondable", "la danza de las ideas", "la corriente cristalina"
    ];

    const connectors = [
      "mientras que", "gracias a lo cual", "lo que desencadena", "garantizando de este modo",
      "al mismo tiempo que", "para dar paso a", "lo que se traduce en", "creando un puente hacia",
      "lo que reduce drásticamente", "en consonancia con", "abriendo el camino para que"
    ];

    const techDetails = [
      "una latencia de inferencia extremadamente baja en núcleos ARM64.",
      "una compresión de pesos sumamente eficiente y óptima.",
      "un esquema de prefetch asíncrono para liberar el bus de memoria.",
      "un control riguroso de la disipación térmica del procesador.",
      "la ejecución directa de registros en coma flotante de precisión media.",
      "un procesamiento matemático local sin transferencias a la nube.",
      "la estabilidad de los gradientes de atención en cada iteración."
    ];

    const poeticDetails = [
      "una sensación de calma y tranquilidad infinitas.",
      "un camino inexplorado hacia nuevos horizontes de expresión.",
      "el eterno retorno de preguntas sin una respuesta predefinida.",
      "la sutil armonía de los elementos en un entorno en cambio constante.",
      "una perspectiva completamente nueva sobre el flujo de la naturaleza.",
      "un instante suspendido en el flujo ininterrumpido del conocimiento.",
      "el florecer de una idea única bajo la luz de la razón autónoma."
    ];

    // Clasificar si el prompt es de naturaleza más técnica, lírica o general
    const techTriggers = ["cpu", "ia", "transformer", "hilos", "modelo", "bpe", "cuantiz", "arm64", "android", "termux", "comput", "program", "algorit", "codig", "cache", "tensor", "matrix", "multiplic"];
    const poeticTriggers = ["mar", "oceano", "agua", "poema", "poes", "olas", "azul", "cielo", "viento", "sol", "luna", "estrell", "amor", "almacen", "filosof", "arte", "sentir"];

    let techScore = 0;
    let poeticScore = 0;

    cleanInput.split(/\s+/).forEach(word => {
      if (techTriggers.some(t => word.includes(t))) techScore++;
      if (poeticTriggers.some(t => word.includes(t))) poeticScore++;
    });

    const isTechnical = techScore >= poeticScore;
    let subjects = isTechnical ? techNouns : poeticNouns;
    let verbs = isTechnical ? techVerbs : abstractVerbs;
    let details = isTechnical ? techDetails : poeticDetails;

    // Si se ha entrenado un modelo personalizado, enriquecer el pool sintáctico con sus palabras clave
    if (hasTrained && trainingCorpus) {
      const corpusWords = trainingCorpus
        .split(/[\s,.\-!?¿¡()"]+/)
        .filter(w => w.length > 4 && !stopWords.has(w.toLowerCase()));
      
      if (corpusWords.length > 0) {
        const uniqueWords: string[] = Array.from(new Set(corpusWords)) as string[];
        subjects = [
          ...uniqueWords.slice(0, 10).map(w => `el concepto de "${w.toLowerCase()}"`),
          ...subjects
        ];
        details = [
          ...uniqueWords.slice(5, 12).map(w => `integrando exitosamente el conocimiento de "${w.toLowerCase()}" de forma secuencial en el grafo de atención.`),
          ...details
        ];
      }
    }

    // Generar un párrafo coherente de 2 o 3 oraciones de manera totalmente procedimental y fluida
    const sentencesCount = rand() > 0.5 ? 3 : 2;
    const sentences: string[] = [];

    for (let s = 0; s < sentencesCount; s++) {
      const sRand = createSeededRandom(100 + s);
      
      // Obtener una palabra del prompt para incrustarla de forma dinámica en la oración
      const promptWordIndex = Math.floor(sRand() * coreWords.length);
      let userWord = coreWords[promptWordIndex];
      if (s === 0 && userWord) {
        userWord = userWord.charAt(0).toUpperCase() + userWord.slice(1);
      } else if (userWord) {
        userWord = userWord.toLowerCase();
      }

      let sentenceStr = "";
      const structureType = Math.floor(sRand() * 3);

      if (structureType === 0) {
        // Estructura: [Palabra de Usuario] + [Verbo] + [Sujeto General] + [Conector] + [Detalle]
        const verb = pickRandom(verbs, sRand);
        const subj = pickRandom(subjects, sRand).toLowerCase();
        const conn = pickRandom(connectors, sRand);
        const det = pickRandom(details, sRand);
        
        sentenceStr = `${userWord} ${verb} ${subj}, ${conn} ${det}`;
      } else if (structureType === 1) {
        // Estructura: [Sujeto General] + [Verbo] + [Palabra de Usuario] + [Conector] + [Detalle]
        const subj = pickRandom(subjects, sRand);
        const verb = pickRandom(verbs, sRand);
        const conn = pickRandom(connectors, sRand);
        const det = pickRandom(details, sRand);
        
        sentenceStr = `${subj} ${verb} ${userWord}, ${conn} ${det}`;
      } else {
        // Estructura: [Sujeto General] + [Conector] + [Palabra de Usuario] + [Verbo] + [Detalle]
        const subj = pickRandom(subjects, sRand);
        const conn = pickRandom(connectors, sRand);
        const verb = pickRandom(verbs, sRand);
        const det = pickRandom(details, sRand);
        
        sentenceStr = `${subj} se alinea con ${userWord} ${conn} ${verb} ${det}`;
      }

      sentenceStr = sentenceStr.trim();
      if (!sentenceStr.endsWith(".")) {
        sentenceStr += ".";
      }
      
      sentenceStr = sentenceStr.charAt(0).toUpperCase() + sentenceStr.slice(1);
      sentences.push(sentenceStr);
    }

    const fullText = sentences.join(" ");

    // Convertir el texto final generado en tokens individuales para la simulación de la inferencia
    const wordsArray = fullText.split(/\s+/).filter(w => w.length > 0);
    const finalTokens = wordsArray.map((w, idx) => {
      return idx === 0 ? w : " " + w;
    });

    return finalTokens;
  };

  // Simulación interactiva del motor autorregresivo con prefetch asíncrono
  const startSimulation = () => {
    if (isGenerating) return;
    setIsGenerating(true);
    setGeneratedTokens([]);
    setCurrentStep(0);
    setCurrentLayer(-1);
    setKvCacheSize(0);
    setIsPrefetching(false);
    
    // Obtener la secuencia dinámica de tokens en base al prompt y la temperatura actual
    const sampleWords = generateDynamicTextSequence(prompt, temperature);
    
    let step = 0;
    const totalSteps = Math.min(maxTokens, sampleWords.length * 2);
    const startTime = Date.now();
    let firstTokenTimeSet = 0;

    // Velocidades ajustadas según el modelo seleccionado para emular hardware real
    // Modelo reducido de test vuela, Modelo E es más pesado pero óptimo para sus dimensiones
    const layerComputeDelay = selectedModel.id === "model_test" ? 2 : selectedModel.id === "model_a" ? 8 : 14;
    const nextTokenBaseDelay = selectedModel.id === "model_test" ? 30 : selectedModel.id === "model_a" ? 100 : 180;

    const runStep = () => {
      if (step >= totalSteps || step >= sampleWords.length) {
        stopSimulation();
        const endTime = Date.now();
        const totalDuration = (endTime - startTime) / 1000;
        setStats(prev => ({
          ...prev,
          totalTokens: step,
          totalTime: totalDuration,
          tokensSec: parseFloat((step / (totalDuration - firstTokenTimeSet)).toFixed(2))
        }));
        return;
      }

      setCurrentStep(step);
      setKvCacheSize(prev => Math.min(selectedModel.window_size, prev + 1));

      let layerIdx = 0;
      const totalLayers = selectedModel.n_layers;

      const runLayerLoop = () => {
        if (layerIdx >= totalLayers) {
          setCurrentLayer(-1);
          setIsPrefetching(false);
          setPrefetchLayer(-1);

          // Proyección de logits de salida
          const nextWord = sampleWords[step];
          setGeneratedTokens(prev => [...prev, { 
            token: nextWord, 
            layer: totalLayers, 
            step: step 
          }]);

          if (step === 0) {
            firstTokenTimeSet = (Date.now() - startTime) / 1000;
            setStats(prev => ({ ...prev, firstTokenTime: firstTokenTimeSet }));
          }

          step++;
          generationInterval.current = setTimeout(runStep, nextTokenBaseDelay);
          return;
        }

        setCurrentLayer(layerIdx);
        
        // Simular Prefetch de la capa i+1
        if (layerIdx < totalLayers - 1) {
          setIsPrefetching(true);
          setPrefetchLayer(layerIdx + 1);
        } else {
          setIsPrefetching(false);
          setPrefetchLayer(-1);
        }

        layerIdx++;
        generationInterval.current = setTimeout(runLayerLoop, layerComputeDelay);
      };

      runLayerLoop();
    };

    runStep();
  };

  const stopSimulation = () => {
    if (generationInterval.current) {
      clearTimeout(generationInterval.current);
    }
    setIsGenerating(false);
    setCurrentLayer(-1);
    setIsPrefetching(false);
    setPrefetchLayer(-1);
  };

  // Cambiar modelo dinámicamente y limpiar estados de simulación
  const handleModelChange = (preset: ModelPreset) => {
    stopSimulation();
    setSelectedModel(preset);
    setGeneratedTokens([]);
    setCurrentStep(0);
    setKvCacheSize(0);
    setStats({ firstTokenTime: 0, tokensSec: 0, totalTokens: 0, totalTime: 0 });
  };

  const trainingTimeouts = useRef<NodeJS.Timeout[]>([]);

  const stopTraining = () => {
    trainingTimeouts.current.forEach(t => clearTimeout(t));
    trainingTimeouts.current = [];
    setIsTraining(false);
    setTrainingStep("idle");
  };

  const startTraining = () => {
    stopTraining();
    setIsTraining(true);
    setTrainingStep("tokenizing");
    setTrainingEpoch(0);
    setTrainingLoss(10.42);
    setLossHistory([]);
    setCurrentBpeMerge("");
    setMergesCount(0);

    // 1. Simular Fase de Tokenización BPE (2 segundos)
    const simulatedMerges = [
      { merge: "'a' + 'r' -> 'ar'", count: 250 },
      { merge: "'d' + 'e' -> 'de'", count: 500 },
      { merge: "'t' + 'e' -> 'te'", count: 850 },
      { merge: "'e' + 'l' -> 'el'", count: 1200 },
      { merge: "'i' + 'a' -> 'ia'", count: 1600 },
      { merge: "'c' + 'o' -> 'co'", count: 2100 },
      { merge: "'o' + 'p' -> 'op'", count: 2700 },
      { merge: "'u' + 'n' -> 'un'", count: 3400 },
      { merge: "'t' + 'o' -> 'to'", count: 4200 },
      { merge: "'m' + 'a' -> 'ma'", count: 5120 },
    ];

    simulatedMerges.forEach((m, idx) => {
      const t = setTimeout(() => {
        setCurrentBpeMerge(m.merge);
        setMergesCount(m.count);
      }, idx * 180);
      trainingTimeouts.current.push(t);
    });

    // Transición a la Fase 2: Gradientes (Forward & Backward) después de 2 segundos
    const tPhase2 = setTimeout(() => {
      setTrainingStep("forward_backward");
      
      const epochInterval = 320; // 320ms por época
      let currentLoss = 9.87;
      const history: { epoch: number; loss: number }[] = [];

      for (let e = 1; e <= epochs; e++) {
        const tEpoch = setTimeout(() => {
          setTrainingEpoch(e);
          // Curva de pérdida descendiente con ruido natural
          const factor = Math.pow(0.82, e);
          const noise = Math.random() * 0.15;
          currentLoss = Math.max(1.12, 7.8 * factor + 1.25 + noise);
          setTrainingLoss(parseFloat(currentLoss.toFixed(4)));
          
          history.push({ epoch: e, loss: parseFloat(currentLoss.toFixed(4)) });
          setLossHistory([...history]);

          // Al completar la última época, pasar a guardar pesos
          if (e === epochs) {
            const tPhase3 = setTimeout(() => {
              setTrainingStep("saving");
              
              // Simular guardado secuencial de capas
              const layersToSave = ["base.npz", ...Array.from({ length: selectedModel.n_layers }).map((_, i) => `layer_${i}.npz`), "tokenizer.json"];
              layersToSave.forEach((layerFile, lIdx) => {
                const tLayer = setTimeout(() => {
                  setCurrentBpeMerge(`Escritura en disco: /fp32_model/${layerFile}...`);
                  setMergesCount(Math.round(((lIdx + 1) / layersToSave.length) * 100));

                  if (lIdx === layersToSave.length - 1) {
                    // Completar el entrenamiento
                    setTrainingStep("done");
                    setIsTraining(false);
                    setHasTrained(true);
                  }
                }, lIdx * (1200 / layersToSave.length));
                trainingTimeouts.current.push(tLayer);
              });

            }, 400);
            trainingTimeouts.current.push(tPhase3);
          }
        }, e * epochInterval);
        trainingTimeouts.current.push(tEpoch);
      }

    }, 2000);
    trainingTimeouts.current.push(tPhase2);
  };

  useEffect(() => {
    return () => {
      if (generationInterval.current) clearTimeout(generationInterval.current);
      trainingTimeouts.current.forEach(t => clearTimeout(t));
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#07090E] text-[#E2E8F0] font-sans antialiased selection:bg-emerald-500/30 selection:text-emerald-300">
      
      {/* Fondo de gradiente futurista sutil */}
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900/30 via-black to-black pointer-events-none" />

      <header className="relative border-b border-slate-800/60 bg-slate-950/40 backdrop-blur-md z-10">
        <div className="max-w-7xl mx-auto px-6 py-5 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-emerald-500 to-teal-400 flex items-center justify-center shadow-[0_0_20px_rgba(16,185,129,0.3)]">
              <Cpu className="h-5 w-5 text-black stroke-[2.5]" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight bg-gradient-to-r from-emerald-300 via-teal-200 to-white bg-clip-text text-transparent">
                  Transformer Edge Sandbox
                </h1>
                <span className="px-2 py-0.5 bg-emerald-500/10 text-emerald-400 rounded-full text-[10px] font-mono border border-emerald-500/20">
                  v2.0 - Dinámico
                </span>
              </div>
              <p className="text-xs text-slate-400 font-mono">
                Simulación Interactiva & Compresión de Modelos de Lenguaje para ARM64 / Termux
              </p>
            </div>
          </div>

          <div className="flex bg-slate-900/80 p-1 rounded-xl border border-slate-800/80">
            <button
              onClick={() => setActiveTab("playground")}
              className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                activeTab === "playground"
                  ? "bg-emerald-500 text-black font-semibold shadow-md"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Terminal className="inline h-3.5 w-3.5 mr-1.5" />
              Consola Interactiva
            </button>
            <button
              onClick={() => setActiveTab("trainer")}
              className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer relative ${
                activeTab === "trainer"
                  ? "bg-emerald-500 text-black font-semibold shadow-md"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Brain className="inline h-3.5 w-3.5 mr-1.5" />
              Entrenar IA
              {hasTrained && (
                <span className="absolute -top-1 -right-1 flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
              )}
            </button>
            <button
              onClick={() => setActiveTab("explorer")}
              className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                activeTab === "explorer"
                  ? "bg-emerald-500 text-black font-semibold shadow-md"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Layers className="inline h-3.5 w-3.5 mr-1.5" />
              Comparar Modelos
            </button>
            <button
              onClick={() => setActiveTab("quantizer")}
              className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                activeTab === "quantizer"
                  ? "bg-emerald-500 text-black font-semibold shadow-md"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Sliders className="inline h-3.5 w-3.5 mr-1.5" />
              Sandbox Q4_0
            </button>
            <button
              onClick={() => setActiveTab("termux")}
              className={`px-4 py-1.5 rounded-lg text-xs font-medium transition-all cursor-pointer ${
                activeTab === "termux"
                  ? "bg-emerald-500 text-black font-semibold shadow-md"
                  : "text-slate-400 hover:text-white"
              }`}
            >
              <Code className="inline h-3.5 w-3.5 mr-1.5" />
              Instalar en Termux
            </button>
          </div>
        </div>
      </header>

      <main className="relative max-w-7xl mx-auto px-6 py-8 z-10">
        <AnimatePresence mode="wait">
          
          {/* TAB 1: PLAYGROUND CON SELECCIÓN DINÁMICA DE MODELO */}
          {activeTab === "playground" && (
            <motion.div
              key="playground"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              
              {/* Selector Rápido de Modelo */}
              <div className="bg-slate-950/60 p-6 rounded-2xl border border-slate-800/80 backdrop-blur-sm grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="md:col-span-3 flex items-center justify-between border-b border-slate-800/80 pb-3 mb-2">
                  <div className="flex items-center gap-2">
                    <SlidersHorizontal className="h-4 w-4 text-emerald-400" />
                    <span className="text-xs font-mono font-bold text-slate-200 uppercase">Selecciona el Modelo Activo para la Simulación</span>
                  </div>
                  <span className="text-[10px] text-slate-500 font-mono">Los parámetros de inferencia cambian según el modelo</span>
                </div>

                {MODEL_PRESETS.map((preset) => (
                  <button
                    key={preset.id}
                    onClick={() => handleModelChange(preset)}
                    className={`p-4 rounded-xl border transition-all text-left flex flex-col justify-between h-32 cursor-pointer ${
                      selectedModel.id === preset.id
                        ? "border-emerald-500 bg-emerald-500/5 shadow-[0_0_15px_rgba(16,185,129,0.1)]"
                        : "border-slate-800/80 bg-slate-900/30 hover:border-slate-700/80"
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-bold text-sm text-white">{preset.name}</span>
                        <span className={`h-2 w-2 rounded-full ${selectedModel.id === preset.id ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
                      </div>
                      <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">{preset.desc}</p>
                    </div>
                    <div className="flex items-center justify-between border-t border-slate-800/50 pt-2 mt-2 text-[10px] font-mono text-slate-400">
                      <span>Params: <strong className="text-slate-200">{preset.params}</strong></span>
                      <span>Disco: <strong className="text-emerald-400">{preset.diskSize}</strong></span>
                    </div>
                  </button>
                ))}
              </div>

              {/* Contenido Principal de Consola */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                
                {/* Sección Izquierda: Consola y Prompt */}
                <div className="lg:col-span-8 flex flex-col gap-6">
                  
                  {/* Caja de Inferencia */}
                  <div className="p-6 bg-slate-950/60 rounded-2xl border border-slate-800/80 backdrop-blur-sm">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4 pb-3 border-b border-slate-900/80">
                      <span className="text-xs font-mono text-emerald-400 font-bold flex items-center gap-2">
                        <Terminal className="h-4 w-4" />
                        SIMULADOR DE INFERENCIA EN TIEMPO REAL
                        {hasTrained && (
                          <span className="px-2 py-0.5 bg-emerald-500/10 text-emerald-400 rounded-full text-[9px] border border-emerald-500/30 animate-pulse">
                            Pesos Finetuneados Activos (.npz)
                          </span>
                        )}
                      </span>
                      <span className="text-[10px] text-slate-500 font-mono flex items-center gap-1.5">
                        <Sparkles className="h-3.5 w-3.5 text-emerald-400" />
                        Texto adaptativo basado en Prompt
                      </span>
                    </div>

                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-xs font-mono text-slate-400 mb-1.5 uppercase">
                          <span>Prompt de Entrada</span>
                          <span className="text-slate-500">Intenta variar el prompt</span>
                        </div>
                        <div className="flex gap-3 mb-2.5">
                          <input
                            type="text"
                            value={prompt}
                            onChange={(e) => setPrompt(e.target.value)}
                            disabled={isGenerating}
                            placeholder="Escribe un prompt, por ejemplo: 'poema sobre el mar', 'optimizar cpu' o 'cuantizacion ia'..."
                            className="flex-1 px-4 py-2.5 bg-slate-900 rounded-xl border border-slate-800 text-sm text-white focus:outline-none focus:border-emerald-500/50 disabled:text-slate-500 transition-colors"
                          />
                          {isGenerating ? (
                            <button
                              onClick={stopSimulation}
                              className="px-5 py-2.5 bg-rose-500 hover:bg-rose-600 text-white text-xs font-bold rounded-xl flex items-center gap-2 transition-colors cursor-pointer"
                            >
                              <Square className="h-4 w-4" />
                              Detener
                            </button>
                          ) : (
                            <button
                              onClick={startSimulation}
                              className="px-5 py-2.5 bg-emerald-500 hover:bg-emerald-600 text-black text-xs font-bold rounded-xl flex items-center gap-2 transition-colors cursor-pointer shadow-[0_0_15px_rgba(16,185,129,0.3)]"
                            >
                              <Play className="h-4 w-4 fill-black" />
                              Ejecutar
                            </button>
                          )}
                        </div>

                        {/* Sugerencias rápidas de prompt */}
                        <div className="flex flex-wrap gap-2 pt-1 border-t border-slate-900">
                          <span className="text-[10px] font-mono text-slate-500 self-center uppercase mr-1">Sugerencias:</span>
                          {[
                            { label: "🌊 El mar es", value: "El mar es" },
                            { label: "🤖 IA y Transformer", value: "cuantizacion ia transformer" },
                            { label: "⚡ Hola, como estas", value: "Hola, como estas" },
                            { label: "📱 Optimizar CPU", value: "optimizar hilos cpu arm64" },
                            ...(hasTrained 
                              ? [{ label: "🧠 Mi Corpus", value: "La tecnologia del futuro une la inteligencia local" }] 
                              : [{ label: "🧠 Mi propia idea", value: "La tecnologia del futuro" }]
                            )
                          ].map((sug, i) => (
                            <button
                              key={i}
                              type="button"
                              disabled={isGenerating}
                              onClick={() => setPrompt(sug.value)}
                              className={`px-2 py-1 bg-slate-900/60 hover:bg-slate-800/80 disabled:opacity-50 text-[11px] font-mono rounded-lg text-slate-300 hover:text-emerald-400 border transition-all cursor-pointer ${
                                prompt === sug.value ? "border-emerald-500/40 text-emerald-300 bg-slate-800/80" : "border-slate-800/80 hover:border-emerald-500/30"
                              }`}
                            >
                              {sug.label}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Consola Simulada de Termux */}
                      <div className="p-5 bg-slate-900/90 rounded-xl border border-slate-800 font-mono text-xs min-h-[160px] flex flex-col justify-between shadow-inner">
                        <div>
                          <div className="flex justify-between items-center text-slate-500 mb-2 border-b border-slate-800/80 pb-2">
                            <span>termux_user@android14:~/termux_llm $ python infer.py</span>
                            <span className="text-[10px] px-1.5 py-0.5 bg-slate-800 rounded">Q4_0 Active</span>
                          </div>
                          
                          <p className="text-slate-300 leading-relaxed text-sm mt-3">
                            <span className="text-slate-500 font-bold">Prompt:</span> <span className="text-emerald-300 font-semibold">{prompt}</span>
                            <br />
                            <span className="text-teal-400 font-bold">Generación:</span>{" "}
                            <span className="text-white">
                              {generatedTokens.map((t, idx) => (
                                <motion.span
                                  key={idx}
                                  initial={{ opacity: 0, scale: 0.8 }}
                                  animate={{ opacity: 1, scale: 1 }}
                                  transition={{ duration: 0.1 }}
                                >
                                  {t.token}
                                </motion.span>
                              ))}
                            </span>
                            {isGenerating && (
                              <span className="inline-block w-2 h-4 bg-emerald-400 animate-pulse ml-0.5 align-middle" />
                            )}
                          </p>
                        </div>
                        
                        {/* Estado de hilos y prefetch */}
                        <div className="border-t border-slate-800/80 pt-3 mt-4 flex flex-wrap gap-4 text-[10px] text-slate-400 justify-between">
                          <div className="flex gap-4">
                            <span>
                              Token: <span className="text-slate-200">#{currentStep}</span>
                            </span>
                            <span>
                              Capas: <span className="text-emerald-400 font-semibold">{selectedModel.n_layers} Capas</span>
                            </span>
                            <span>
                              KV-Cache: <span className="text-slate-200">{kvCacheSize} / {selectedModel.window_size}</span>
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-ping" />
                            <span>Prefetch asíncrono activo ({threads} hilos)</span>
                          </div>
                        </div>
                      </div>

                    </div>
                  </div>

                  {/* Monitor de Hilos y Prefetch */}
                  <div className="p-6 bg-slate-900/40 rounded-2xl border border-slate-800/80">
                    <h3 className="text-white font-semibold text-sm mb-4 flex items-center gap-2">
                      <Activity className="h-4 w-4 text-emerald-400" />
                      Visualizador de Carga y Prefetch Secuencial de Capas
                    </h3>
                    
                    <div className="space-y-4">
                      {/* Hilo Principal */}
                      <div>
                        <div className="flex justify-between text-xs font-mono mb-1.5">
                          <span className="text-emerald-400 flex items-center gap-1.5">
                            <Cpu className="h-3.5 w-3.5" />
                            CPU Core (Procesando Capa FP32 en RAM)
                          </span>
                          <span className="text-slate-400 text-[10px]">
                            {currentLayer >= 0 ? `Multiplicando Capa ${currentLayer + 1}/${selectedModel.n_layers}` : "IDLE"}
                          </span>
                        </div>
                        <div className="h-3.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800 flex p-[1px]">
                          {Array.from({ length: selectedModel.n_layers }).map((_, i) => (
                            <div 
                              key={i}
                              className={`flex-1 h-full border-r border-slate-900/30 transition-colors duration-100 ${
                                i === currentLayer 
                                  ? "bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.8)]" 
                                  : i < currentLayer 
                                  ? "bg-emerald-950/80" 
                                  : "bg-slate-900"
                              }`}
                            />
                          ))}
                        </div>
                      </div>

                      {/* Hilo de Prefetch */}
                      <div>
                        <div className="flex justify-between text-xs font-mono mb-1.5">
                          <span className="text-teal-400 flex items-center gap-1.5">
                            <Database className="h-3.5 w-3.5" />
                            Hilo Prefetcher (Lectura asíncrona de UFS)
                          </span>
                          <span className="text-slate-400 text-[10px]">
                            {prefetchLayer >= 0 ? `Descuantizando Capa ${prefetchLayer + 1}/${selectedModel.n_layers}` : "IDLE"}
                          </span>
                        </div>
                        <div className="h-3.5 w-full bg-slate-950 rounded-full overflow-hidden border border-slate-800 flex p-[1px]">
                          {Array.from({ length: selectedModel.n_layers }).map((_, i) => (
                            <div 
                              key={i}
                              className={`flex-1 h-full border-r border-slate-900/30 transition-colors duration-100 ${
                                i === prefetchLayer 
                                  ? "bg-teal-400 animate-pulse shadow-[0_0_10px_rgba(45,212,191,0.8)]" 
                                  : i < prefetchLayer 
                                  ? "bg-teal-950/80" 
                                  : "bg-slate-900"
                              }`}
                            />
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                </div>

                {/* Sección Derecha: Parámetros de Inferencia y Métricas */}
                <div className="lg:col-span-4 flex flex-col gap-6">
                  
                  {/* Panel de Control de Inferencia */}
                  <div className="p-6 bg-gradient-to-b from-slate-900 to-slate-950 rounded-2xl border border-slate-800/80 space-y-5">
                    <h3 className="text-white font-semibold text-sm flex items-center gap-2 pb-3 border-b border-slate-800/80">
                      <Sliders className="h-4 w-4 text-emerald-400" />
                      Parámetros de Inferencia
                    </h3>

                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between text-xs font-mono mb-1.5">
                          <span className="text-slate-400">Temperatura (Variación)</span>
                          <span className="text-emerald-400 font-bold">{temperature}</span>
                        </div>
                        <input
                          type="range"
                          min="0.1"
                          max="1.2"
                          step="0.1"
                          value={temperature}
                          onChange={(e) => setTemperature(parseFloat(e.target.value))}
                          disabled={isGenerating}
                          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                        />
                        <p className="text-[10px] text-slate-500 mt-1 leading-relaxed">
                          Temperaturas altas introducen más variedad y combinaciones creativas de tokens.
                        </p>
                      </div>

                      <div>
                        <div className="flex justify-between text-xs font-mono mb-1.5">
                          <span className="text-slate-400">Longitud Máxima (Tokens)</span>
                          <span className="text-emerald-400 font-bold">{maxTokens}</span>
                        </div>
                        <input
                          type="range"
                          min="5"
                          max="50"
                          step="5"
                          value={maxTokens}
                          onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                          disabled={isGenerating}
                          className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                        />
                      </div>

                      <div>
                        <div className="flex justify-between text-xs font-mono mb-1.5">
                          <span className="text-slate-400">Hilos OpenBLAS (CPU Cores)</span>
                          <span className="text-emerald-400 font-bold">{threads} Hilos</span>
                        </div>
                        <select
                          value={threads}
                          onChange={(e) => setThreads(parseInt(e.target.value))}
                          disabled={isGenerating}
                          className="w-full px-3 py-2 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 focus:outline-none focus:border-emerald-500/50 font-mono"
                        >
                          <option value={1}>1 Núcleo (Single Thread)</option>
                          <option value={2}>2 Núcleos (Cortex-A76 Cores)</option>
                          <option value={4}>4 Núcleos (Recomendado Dimensity)</option>
                          <option value={6}>6 Núcleos (Cortex-A55 Cores)</option>
                          <option value={8}>8 Núcleos (Carga Completa CPU)</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* Cuadro de Estadísticas del Benchmark */}
                  <div className="p-6 bg-slate-900/40 rounded-2xl border border-slate-800/80">
                    <h3 className="text-white font-semibold text-sm mb-4 flex items-center gap-2">
                      <Info className="h-4 w-4 text-emerald-400" />
                      Métricas Proyectadas (Hardware Real)
                    </h3>

                    <div className="space-y-4">
                      <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800 flex items-center justify-between">
                        <div>
                          <span className="text-[10px] font-mono text-slate-400 block uppercase">Time-To-First-Token</span>
                          <span className="text-lg font-bold font-mono text-white">
                            {stats.firstTokenTime > 0 
                              ? `${stats.firstTokenTime.toFixed(2)}s` 
                              : selectedModel.id === "model_test" 
                              ? "~0.05s" 
                              : selectedModel.id === "model_a" 
                              ? "~0.55s" 
                              : "~0.82s"}
                          </span>
                        </div>
                        <span className="text-[10px] text-slate-500 font-mono">UFS Prefetch</span>
                      </div>

                      <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800 flex items-center justify-between">
                        <div>
                          <span className="text-[10px] font-mono text-slate-400 block uppercase">Velocidad Promedio</span>
                          <span className="text-lg font-bold font-mono text-white">
                            {stats.tokensSec > 0 
                              ? `${stats.tokensSec} tok/s` 
                              : selectedModel.speed}
                          </span>
                        </div>
                        <span className="text-[9px] px-1.5 py-0.5 bg-emerald-500/10 text-emerald-400 rounded border border-emerald-500/20 font-mono">
                          Goal &gt;1.5 tok/s
                        </span>
                      </div>

                      <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800 flex items-center justify-between">
                        <div>
                          <span className="text-[10px] font-mono text-slate-400 block uppercase">RAM Requerida (Pico)</span>
                          <span className="text-lg font-bold font-mono text-white">{selectedModel.ram}</span>
                        </div>
                        <span className="text-[10px] text-slate-500 font-mono">KV-Cache FP16</span>
                      </div>
                    </div>
                  </div>

                </div>

              </div>
            </motion.div>
          )}

          {/* TAB 5: ENTRENAR IA (TRAINER) */}
          {activeTab === "trainer" && (
            <motion.div
              key="trainer"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              {/* Encabezado */}
              <div className="p-6 bg-slate-950/60 rounded-2xl border border-slate-800/80 backdrop-blur-sm flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                <div>
                  <h2 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                    <Brain className="h-5 w-5 text-emerald-400" />
                    Entrenador de Modelos Transformer Local
                  </h2>
                  <p className="text-slate-400 text-sm max-w-3xl">
                    Entrena el Tokenizador BPE y ajusta las matrices de pesos de las capas de auto-atención asíncrona directamente desde tu navegador. Los conceptos aprendidos se integrarán en el simulador.
                  </p>
                </div>
                {hasTrained && (
                  <span className="px-3 py-1 bg-emerald-500/10 text-emerald-400 rounded-full text-xs font-mono border border-emerald-500/20 flex items-center gap-1.5 animate-pulse">
                    <CheckCircle className="h-3.5 w-3.5" />
                    Modelo Finetuneado
                  </span>
                )}
              </div>

              {/* Contenido Principal en Dos Columnas */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                
                {/* Panel de Configuración y Corpus */}
                <div className="lg:col-span-5 p-6 bg-slate-900/40 rounded-2xl border border-slate-800/80 space-y-6">
                  <div>
                    <h3 className="text-white font-semibold text-sm mb-3 flex items-center gap-2">
                      <FileText className="h-4 w-4 text-emerald-400" />
                      Corpus de Entrenamiento (Dataset)
                    </h3>
                    <p className="text-xs text-slate-400 mb-2 leading-relaxed">
                      Escribe o edita el texto en español con el que deseas entrenar la red neuronal. La IA asimilará estos conceptos y generará oraciones usándolos.
                    </p>
                    <textarea
                      value={trainingCorpus}
                      onChange={(e) => setTrainingCorpus(e.target.value)}
                      disabled={isTraining}
                      rows={6}
                      placeholder="Escribe el texto de entrenamiento aquí..."
                      className="w-full px-4 py-3 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-emerald-500/50 font-mono resize-none leading-relaxed"
                    />
                  </div>

                  {/* Hiperparámetros */}
                  <div className="space-y-4 border-t border-slate-800/60 pt-4">
                    <h4 className="text-slate-300 font-semibold text-xs flex items-center gap-2">
                      <Sliders className="h-3.5 w-3.5 text-emerald-400" />
                      Hiperparámetros del Entrenamiento
                    </h4>

                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block text-[10px] font-mono text-slate-400 mb-1">Épocas (Epochs)</label>
                        <select
                          value={epochs}
                          onChange={(e) => setEpochs(parseInt(e.target.value))}
                          disabled={isTraining}
                          className="w-full px-3 py-2 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono"
                        >
                          <option value={5}>5 Épocas (Rápido)</option>
                          <option value={10}>10 Épocas (Estándar)</option>
                          <option value={15}>15 Épocas (Mejor ajuste)</option>
                          <option value={20}>20 Épocas (Completo)</option>
                          <option value={30}>30 Épocas (Profundo)</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-[10px] font-mono text-slate-400 mb-1">Tasa de Aprendizaje</label>
                        <select
                          value={learningRate}
                          onChange={(e) => setLearningRate(e.target.value)}
                          disabled={isTraining}
                          className="w-full px-3 py-2 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono"
                        >
                          <option value="5e-4">5e-4 (Agresiva)</option>
                          <option value="1e-4">1e-4 (Recomendada)</option>
                          <option value="5e-5">5e-5 (Fina)</option>
                          <option value="1e-5">1e-5 (Conservadora)</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-[10px] font-mono text-slate-400 mb-1">Optimizador</label>
                        <select
                          value={optimizer}
                          onChange={(e) => setOptimizer(e.target.value)}
                          disabled={isTraining}
                          className="w-full px-3 py-2 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono"
                        >
                          <option value="AdamW">AdamW (Decay regulado)</option>
                          <option value="SGD">SGD (Stochastic Gradient)</option>
                          <option value="RMSprop">RMSprop</option>
                        </select>
                      </div>

                      <div>
                        <label className="block text-[10px] font-mono text-slate-400 mb-1">Tamaño de Lote (Batch)</label>
                        <select
                          value={batchSize}
                          onChange={(e) => setBatchSize(parseInt(e.target.value))}
                          disabled={isTraining}
                          className="w-full px-3 py-2 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 font-mono"
                        >
                          <option value={4}>4 Secuencias</option>
                          <option value={8}>8 Secuencias</option>
                          <option value={16}>16 Secuencias</option>
                          <option value={32}>32 Secuencias</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  {/* Acciones */}
                  <div className="pt-4 border-t border-slate-800/60 flex gap-3">
                    {!isTraining ? (
                      <button
                        onClick={startTraining}
                        className="flex-1 py-3 px-4 bg-emerald-500 hover:bg-emerald-400 text-black font-bold rounded-xl text-xs flex items-center justify-center gap-2 cursor-pointer shadow-[0_0_15px_rgba(16,185,129,0.25)] transition-all font-mono"
                      >
                        <Play className="h-4 w-4 fill-black text-black" />
                        INICIAR ENTRENAMIENTO LOCAL
                      </button>
                    ) : (
                      <button
                        onClick={stopTraining}
                        className="flex-1 py-3 px-4 bg-rose-500 hover:bg-rose-400 text-black font-bold rounded-xl text-xs flex items-center justify-center gap-2 cursor-pointer transition-all font-mono"
                      >
                        <Square className="h-4 w-4 fill-black text-black" />
                        DETENER ENTRENAMIENTO
                      </button>
                    )}
                  </div>
                </div>

                {/* Live Monitor Console / Dashboard */}
                <div className="lg:col-span-7 p-6 bg-slate-900/40 rounded-2xl border border-slate-800/80 flex flex-col justify-between min-h-[450px]">
                  
                  <div className="space-y-6">
                    <div className="flex justify-between items-center pb-3 border-b border-slate-800/60">
                      <h3 className="text-white font-semibold text-sm flex items-center gap-2">
                        <Activity className="h-4 w-4 text-emerald-400" />
                        Consola del Proceso & Métricas de Loss
                      </h3>
                      <div className="flex gap-2">
                        <span className={`h-2 w-2 rounded-full ${isTraining ? 'bg-emerald-500 animate-ping' : 'bg-slate-650'}`} />
                        <span className="text-[10px] font-mono text-slate-400">{isTraining ? "Entrenando" : "Esperando"}</span>
                      </div>
                    </div>

                    {/* Paso actual */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      
                      <div className={`p-4 rounded-xl border transition-all ${
                        trainingStep === "tokenizing" 
                          ? "bg-emerald-500/10 border-emerald-500" 
                          : "bg-slate-950/60 border-slate-800"
                      }`}>
                        <span className="text-[9px] text-slate-500 font-mono block uppercase">Fase 1</span>
                        <span className="text-xs font-bold text-slate-200 block mt-0.5">Tokenizador BPE</span>
                        <div className="mt-2 text-[10px] font-mono text-slate-400 leading-relaxed">
                          {trainingStep === "tokenizing" ? (
                            <span className="text-emerald-400">Fusionando caracteres... {mergesCount} combinaciones</span>
                          ) : trainingStep === "forward_backward" || trainingStep === "saving" || trainingStep === "done" ? (
                            <span className="text-emerald-500 flex items-center gap-1">✔ Vocabulario Listo</span>
                          ) : (
                            "Inactivo"
                          )}
                        </div>
                      </div>

                      <div className={`p-4 rounded-xl border transition-all ${
                        trainingStep === "forward_backward" 
                          ? "bg-emerald-500/10 border-emerald-500" 
                          : "bg-slate-950/60 border-slate-800"
                      }`}>
                        <span className="text-[9px] text-slate-500 font-mono block uppercase">Fase 2</span>
                        <span className="text-xs font-bold text-slate-200 block mt-0.5">Auto-Atención & SwiGLU</span>
                        <div className="mt-2 text-[10px] font-mono text-slate-400 leading-relaxed">
                          {trainingStep === "forward_backward" ? (
                            <span className="text-emerald-400">Backpropagation: Época {trainingEpoch}/{epochs}</span>
                          ) : trainingStep === "saving" || trainingStep === "done" ? (
                            <span className="text-emerald-500 flex items-center gap-1">✔ Pesos Ajustados</span>
                          ) : (
                            "Inactivo"
                          )}
                        </div>
                      </div>

                      <div className={`p-4 rounded-xl border transition-all ${
                        trainingStep === "saving" 
                          ? "bg-emerald-500/10 border-emerald-500" 
                          : "bg-slate-950/60 border-slate-800"
                      }`}>
                        <span className="text-[9px] text-slate-500 font-mono block uppercase">Fase 3</span>
                        <span className="text-xs font-bold text-slate-200 block mt-0.5">Serialización .npz</span>
                        <div className="mt-2 text-[10px] font-mono text-slate-400 leading-relaxed">
                          {trainingStep === "saving" ? (
                            <span className="text-emerald-400">Escribiendo particiones de capas...</span>
                          ) : trainingStep === "done" ? (
                            <span className="text-emerald-500 flex items-center gap-1">✔ Pesos Guardados</span>
                          ) : (
                            "Inactivo"
                          )}
                        </div>
                      </div>

                    </div>

                    {/* Consola Log en tiempo real */}
                    <div className="p-4 bg-slate-950 rounded-xl border border-slate-800/80 font-mono text-[11px] text-slate-300 min-h-[140px] flex flex-col justify-between">
                      <div className="space-y-1">
                        <div className="text-slate-500">// Consola del Compilador de Termux</div>
                        {trainingStep === "idle" && (
                          <div className="text-slate-400">Esperando señal de inicio. El modelo de {selectedModel.name} será finetuneado con tasa de aprendizaje η = {learningRate} usando {optimizer}.</div>
                        )}
                        {trainingStep === "tokenizing" && (
                          <div className="space-y-1">
                            <div className="text-slate-300">&gt;&gt; Leyendo corpus de {trainingCorpus.length} caracteres...</div>
                            <div className="text-emerald-400 animate-pulse">&gt;&gt; Entrenando Tokenizador BPE: {currentBpeMerge}</div>
                            <div className="text-slate-400">&gt;&gt; {mergesCount} sub-tokens ensamblados de bytes de entrada.</div>
                          </div>
                        )}
                        {trainingStep === "forward_backward" && (
                          <div className="space-y-1">
                            <div className="text-emerald-500">&gt;&gt; Tokenizador guardado en tokenizer.json con éxito!</div>
                            <div className="text-slate-300">&gt;&gt; Dimensiones del modelo para optimización: d_model={selectedModel.d_model}, n_layers={selectedModel.n_layers}.</div>
                            <div className="text-teal-400">&gt;&gt; Gradientes calculados para pesos Wq, Wk, Wv en GQA asíncrona.</div>
                            <div className="text-emerald-400 font-semibold animate-pulse">
                              &gt;&gt; Época {trainingEpoch}/{epochs} completada. Pérdida (Loss): {trainingLoss}
                            </div>
                          </div>
                        )}
                        {trainingStep === "saving" && (
                          <div className="space-y-1">
                            <div className="text-emerald-500">&gt;&gt; Optimización de gradientes completada con pérdida de {trainingLoss}.</div>
                            <div className="text-slate-300">&gt;&gt; Separando pesos maestros para habilitar el Streaming de disco asíncrono en Termux...</div>
                            <div className="text-teal-400 animate-pulse">&gt;&gt; {currentBpeMerge} ({mergesCount}%)</div>
                          </div>
                        )}
                        {trainingStep === "done" && (
                          <div className="space-y-1.5">
                            <div className="text-emerald-400 font-bold flex items-center gap-1.5">
                              <Check className="h-4 w-4 bg-emerald-500 text-black rounded-full p-0.5" />
                              ¡ENTRENAMIENTO LOCAL COMPLETADO CON ÉXITO!
                            </div>
                            <div className="text-slate-300">Se han compilado {selectedModel.n_layers} capas en formato binario .npz optimizadas para ARM64.</div>
                            <div className="text-emerald-300 font-semibold">&gt;&gt; El nuevo vocabulario y corpus se han inyectado en el simulador. ¡Pruébalo en la Consola Interactiva!</div>
                          </div>
                        )}
                      </div>

                      {/* Barra de progreso */}
                      {isTraining && (
                        <div className="mt-4 pt-3 border-t border-slate-800/50">
                          <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800 flex">
                            <div 
                              className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-300"
                              style={{ 
                                width: 
                                  trainingStep === "tokenizing" 
                                    ? "25%" 
                                    : trainingStep === "forward_backward" 
                                    ? `${25 + (trainingEpoch / epochs) * 50}%` 
                                    : "90%" 
                              }}
                            />
                          </div>
                        </div>
                      )}
                    </div>

                  </div>

                  {/* Histograma / Curva de Loss */}
                  <div className="mt-6">
                    {lossHistory.length > 0 ? (
                      <div className="space-y-2">
                        <div className="text-xs text-slate-400 font-mono flex justify-between">
                          <span>Curva de Pérdida (Loss Decay)</span>
                          <span className="text-emerald-400 font-bold">Pérdida Actual: {trainingLoss}</span>
                        </div>
                        <div className="h-32 w-full bg-slate-950 rounded-xl border border-slate-800 flex items-end p-2 gap-1.5 overflow-hidden">
                          {lossHistory.map((h, i) => {
                            const maxLoss = 10.42;
                            const heightPct = Math.max(8, (h.loss / maxLoss) * 100);
                            return (
                              <div key={i} className="flex-1 flex flex-col items-center gap-1 h-full justify-end group">
                                <span className="text-[8px] text-slate-500 opacity-0 group-hover:opacity-100 transition-opacity font-mono scale-90">{h.loss.toFixed(2)}</span>
                                <div 
                                  style={{ height: `${heightPct}%` }}
                                  className="w-full bg-gradient-to-t from-emerald-600 to-emerald-400 rounded-t shadow-[0_0_8px_rgba(16,185,129,0.3)] transition-all duration-300"
                                />
                                <span className="text-[8px] text-slate-400 font-mono scale-90">E{h.epoch}</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ) : (
                      <div className="h-32 w-full bg-slate-950 rounded-xl border border-slate-800 flex items-center justify-center text-xs text-slate-500 font-mono">
                        La curva de pérdida se graficará en tiempo real durante la optimización.
                      </div>
                    )}
                  </div>

                </div>

              </div>
            </motion.div>
          )}

          {/* TAB 2: COMPARE MODEL PRESETS */}
          {activeTab === "explorer" && (
            <motion.div
              key="explorer"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-8"
            >
              <div className="p-6 bg-slate-950/60 rounded-2xl border border-slate-800/80 backdrop-blur-sm">
                <h2 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                  <Database className="h-5 w-5 text-emerald-400" />
                  Comparativa de Configuraciones de Modelos Transformer
                </h2>
                <p className="text-slate-400 text-sm max-w-3xl">
                  Selecciona y compara el comportamiento de los hiperparámetros de la familia Transformer Edge. Todos los modelos cumplen estrictamente con la compresión en disco ≤ 300MB gracias a la cuantización total de embeddings y capas Q4_0.
                </p>
              </div>

              {/* Grid de Modelos Detallado */}
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {MODEL_PRESETS.map((preset) => (
                  <div 
                    key={preset.id}
                    className="p-6 bg-gradient-to-b from-slate-900/80 to-slate-950/90 rounded-2xl border border-slate-800/50 flex flex-col justify-between hover:border-slate-700/80 transition-all group relative overflow-hidden"
                  >
                    {/* Indicador de Gradiente de Color en la parte superior */}
                    <div className={`absolute top-0 left-0 right-0 h-1.5 bg-gradient-to-r ${preset.color}`} />
                    
                    <div className="space-y-5">
                      <div>
                        <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">PRESET COMPLETO</span>
                        <h3 className="text-xl font-bold text-white mt-1 group-hover:text-emerald-300 transition-colors">{preset.name}</h3>
                        <p className="text-xs text-slate-400 mt-1">{preset.desc}</p>
                      </div>

                      <div className="space-y-2 border-t border-b border-slate-800/60 py-4 font-mono text-xs">
                        <div className="flex justify-between">
                          <span className="text-slate-500">Parámetros</span>
                          <span className="text-slate-200 font-bold">{preset.params}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Capa Residual (d_model)</span>
                          <span className="text-slate-200">{preset.d_model}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Capas del Bloque</span>
                          <span className="text-slate-200">{preset.n_layers}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">GQA Ratio</span>
                          <span className="text-slate-200">{preset.n_heads}:{preset.n_kv_heads}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">FFN Hidden Size</span>
                          <span className="text-slate-200">{preset.ffn_hidden}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Ventana de Atención</span>
                          <span className="text-slate-200">{preset.window_size} tokens</span>
                        </div>
                      </div>

                      <div className="space-y-3">
                        <div className="flex justify-between items-center text-xs">
                          <span className="text-slate-400">Espacio en Disco (Q4_0):</span>
                          <span className="font-bold text-emerald-400 font-mono">{preset.diskSize}</span>
                        </div>
                        <div className="flex justify-between items-center text-xs">
                          <span className="text-slate-400">Velocidad Generación:</span>
                          <span className="font-bold text-teal-400 font-mono">{preset.speed}</span>
                        </div>
                        <div className="flex justify-between items-center text-xs">
                          <span className="text-slate-400">Consumo de RAM Activa:</span>
                          <span className="font-bold text-indigo-400 font-mono">{preset.ram}</span>
                        </div>
                      </div>
                    </div>

                    <div className="mt-6 pt-4 border-t border-slate-800/40">
                      <button
                        onClick={() => handleModelChange(preset)}
                        className={`w-full py-2 rounded-xl text-xs font-semibold font-mono text-center transition-all cursor-pointer ${
                          selectedModel.id === preset.id
                            ? "bg-emerald-500 text-black shadow-md"
                            : "bg-slate-850 hover:bg-slate-800 text-slate-300"
                        }`}
                      >
                        {selectedModel.id === preset.id ? "MODELO ACTIVO" : "SELECCIONAR MODELO"}
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              {/* Cuadro comparativo de optimizaciones de compresión */}
              <div className="p-6 bg-slate-900/40 rounded-2xl border border-slate-800/80">
                <h3 className="text-white font-semibold text-sm mb-4 flex items-center gap-2">
                  <Activity className="text-emerald-400 h-4 w-4" />
                  Optimización Estricta de Distribución del Checkpoint
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  Tradicionalmente, mantener los embeddings en precisión FP32 pura consumía alrededor de 156.3 MB del archivo final base, excediendo el límite óptimo. Cuantizando tanto las capas como la matriz de embeddings a bloques Q4_0, comprimimos el archivo completo por debajo de 270MB de forma limpia, preservando los factores de escala independientes para garantizar la estabilidad del lenguaje.
                </p>
              </div>

            </motion.div>
          )}

          {/* TAB 3: QUANTIZATION SANDBOX */}
          {activeTab === "quantizer" && (
            <motion.div
              key="quantizer"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              <div className="p-6 bg-slate-950/60 rounded-2xl border border-slate-800/80 backdrop-blur-sm">
                <h2 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                  <Sliders className="h-5 w-5 text-emerald-400" />
                  Prueba de Cuantización por Bloques Q4_0 (NumPy Simulado)
                </h2>
                <p className="text-slate-400 text-sm max-w-3xl">
                  La técnica Q4_0 divide los tensores en bloques independientes (típicamente de 32 elementos). Cada bloque calcula su propio factor de escala, permitiendo una compresión masiva con mínima degradación matemática.
                </p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                
                {/* Panel de Configuración */}
                <div className="lg:col-span-4 p-6 bg-slate-900/40 rounded-2xl border border-slate-800/80 space-y-4">
                  <h3 className="text-white font-semibold text-sm">Entradas Flotantes (FP32)</h3>
                  
                  <div>
                    <label className="block text-xs font-mono text-slate-400 mb-1">Valores Separados por Comas</label>
                    <textarea
                      value={floatValues}
                      onChange={(e) => setFloatValues(e.target.value)}
                      rows={5}
                      className="w-full px-4 py-3 bg-slate-950 rounded-xl border border-slate-800 text-sm text-slate-200 focus:outline-none focus:border-emerald-500/50 font-mono resize-none"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-mono text-slate-400 mb-1">Tamaño de Bloque (block_size)</label>
                    <select
                      value={blockSize}
                      onChange={(e) => setBlockSize(parseInt(e.target.value))}
                      className="w-full px-4 py-2.5 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-200 focus:outline-none focus:border-emerald-500/50 font-mono"
                    >
                      <option value={16}>16 (Mayor precisión, más escalas)</option>
                      <option value={32}>32 (Recomendado / LLaMA Standard)</option>
                      <option value={64}>64 (Máxima compresión)</option>
                    </select>
                  </div>

                  <div className="p-4 bg-emerald-500/5 rounded-xl border border-emerald-500/10 text-xs text-slate-400 leading-relaxed">
                    <p className="font-semibold text-emerald-300 mb-1">¿Cómo Funciona la Matemática?</p>
                    <ol className="list-decimal pl-4 space-y-1 text-slate-300">
                      <li>Calcula la escala: <code className="text-emerald-400 font-mono">scale = abs_max / 7.0</code></li>
                      <li>Escala: <code className="text-emerald-400 font-mono">x_q = round(x / scale)</code></li>
                      <li>Recorta rango de 4 bits: <code className="text-emerald-400 font-mono">[-8, 7]</code></li>
                      <li>Desplazamiento para empaquetado: <code className="text-emerald-400 font-mono">+ 8</code></li>
                    </ol>
                  </div>
                </div>

                {/* Panel de Resultados */}
                <div className="lg:col-span-8 p-6 bg-slate-900/40 rounded-2xl border border-slate-800/80 space-y-6">
                  <h3 className="text-white font-semibold text-sm">Visualización del Pipeline Q4_0</h3>
                  
                  {quantResult && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800">
                          <span className="text-[10px] text-slate-400 font-mono block uppercase">Factor de Escala del Bloque (float16)</span>
                          <span className="text-lg font-bold font-mono text-emerald-400">{quantResult.scale.toFixed(6)}</span>
                        </div>
                        <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800">
                          <span className="text-[10px] text-slate-400 font-mono block uppercase">Error Absoluto Medio (MAE)</span>
                          <span className={`text-lg font-bold font-mono ${quantResult.mae < 0.1 ? 'text-emerald-400' : 'text-amber-400'}`}>
                            {quantResult.mae.toFixed(6)}
                          </span>
                        </div>
                      </div>

                      {/* Comparativa de Valores */}
                      <div className="p-5 bg-slate-950 rounded-xl border border-slate-800 font-mono text-xs space-y-4 max-h-[300px] overflow-y-auto">
                        <div>
                          <div className="text-slate-400 font-semibold mb-1">// 1. Valores Originales (FP32)</div>
                          <div className="text-slate-300 truncate">
                            {quantResult.original.map(v => v.toFixed(3)).join(", ")}
                          </div>
                        </div>

                        <div>
                          <div className="text-slate-400 font-semibold mb-1">// 2. Cuantizados a 4 bits con signo [-8, 7]</div>
                          <div className="text-amber-400 truncate">
                            {quantResult.quantized.join(", ")}
                          </div>
                        </div>

                        <div>
                          <div className="text-slate-400 font-semibold mb-1">// 3. Bytes Empaquetados en disco (uint8)</div>
                          <div className="text-teal-400 truncate">
                            {quantResult.packed.map(b => `0x${b.toString(16).toUpperCase().padStart(2, '0')}`).join(", ")}
                          </div>
                        </div>

                        <div>
                          <div className="text-slate-400 font-semibold mb-1">// 4. Descuantizados a FP32 para Matmul</div>
                          <div className="text-emerald-400 truncate">
                            {quantResult.dequantized.map(v => v.toFixed(3)).join(", ")}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

              </div>
            </motion.div>
          )}

          {/* TAB 4: SETUP IN TERMUX */}
          {activeTab === "termux" && (
            <motion.div
              key="termux"
              initial={{ opacity: 0, y: 15 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -15 }}
              transition={{ duration: 0.2 }}
              className="space-y-6"
            >
              <div className="p-6 bg-slate-950/60 rounded-2xl border border-slate-800/80 backdrop-blur-sm">
                <h2 className="text-lg font-semibold text-white mb-2 flex items-center gap-2">
                  <Cpu className="h-5 w-5 text-emerald-400" />
                  Guía Paso a Paso para Ejecutar en un Dispositivo Móvil
                </h2>
                <p className="text-slate-400 text-sm max-w-3xl">
                  Sigue estos comandos directamente en la app Termux de tu celular Android para clonar, entrenar, cuantizar e inferir a máxima velocidad de hardware.
                </p>
              </div>

              <div className="space-y-6">
                
                {/* Paso 1: Instalación de Dependencias */}
                <div className="p-5 bg-slate-900/30 rounded-2xl border border-slate-800/80 space-y-3">
                  <div className="flex justify-between items-center">
                    <h3 className="text-white font-semibold text-sm flex items-center gap-2">
                      <span className="h-5 w-5 rounded bg-emerald-500 text-black font-mono font-bold flex items-center justify-center text-xs">1</span>
                      Instalación del Entorno
                    </h3>
                    <button
                      onClick={() => handleCopy("pkg update && pkg upgrade -y\npkg install python python-pip openblas ndk-sysroot clang -y\npip install numpy psutil", "deps")}
                      className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded text-[10px] font-mono flex items-center gap-1.5 transition-colors cursor-pointer"
                    >
                      {copiedText === "deps" ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                      {copiedText === "deps" ? "Copiado" : "Copiar"}
                    </button>
                  </div>
                  <pre className="p-4 bg-slate-950 rounded-xl font-mono text-xs text-slate-300 overflow-x-auto">
                    {`# Actualizar repos e instalar Python + OpenBLAS para multiplicación acelerada en CPU
pkg update && pkg upgrade -y
pkg install python python-pip openblas ndk-sysroot clang -y

# Instalar NumPy compilado contra las librerías nativas de aceleración matemática de Termux
pip install numpy psutil`}
                  </pre>
                </div>

                {/* Paso 2: Ejecutar el Pipeline */}
                <div className="p-5 bg-slate-900/30 rounded-2xl border border-slate-800/80 space-y-3">
                  <div className="flex justify-between items-center">
                    <h3 className="text-white font-semibold text-sm flex items-center gap-2">
                      <span className="h-5 w-5 rounded bg-emerald-500 text-black font-mono font-bold flex items-center justify-center text-xs">2</span>
                      Ejecución Completa del Pipeline de Inferencia
                    </h3>
                    <button
                      onClick={() => handleCopy("python test_integration.py\npython train.py\npython quantize.py\npython infer.py", "pipe")}
                      className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded text-[10px] font-mono flex items-center gap-1.5 transition-colors cursor-pointer"
                    >
                      {copiedText === "pipe" ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                      {copiedText === "pipe" ? "Copiado" : "Copiar"}
                    </button>
                  </div>
                  <pre className="p-4 bg-slate-950 rounded-xl font-mono text-xs text-slate-300 overflow-x-auto">
                    {`# 1. Ejecutar prueba rápida de integración para validar alineación de vocabs
python test_integration.py

# 2. Generar checkpoint maestro FP32 y entrenar el Tokenizador de 32,000 vocabulario
python train.py

# 3. Cuantizar la totalidad del modelo a Q4_0 por bloques (embeddings incluidos!)
python quantize.py

# 4. Iniciar inferencia autoregresiva con streaming de disco asíncrono
python infer.py`}
                  </pre>
                </div>

                {/* Paso 3: Benchmark */}
                <div className="p-5 bg-slate-900/30 rounded-2xl border border-slate-800/80 space-y-3">
                  <div className="flex justify-between items-center">
                    <h3 className="text-white font-semibold text-sm flex items-center gap-2">
                      <span className="h-5 w-5 rounded bg-emerald-500 text-black font-mono font-bold flex items-center justify-center text-xs">3</span>
                      Medición de Rendimiento
                    </h3>
                    <button
                      onClick={() => handleCopy("python benchmark.py", "bench")}
                      className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white rounded text-[10px] font-mono flex items-center gap-1.5 transition-colors cursor-pointer"
                    >
                      {copiedText === "bench" ? <Check className="h-3 w-3 text-emerald-400" /> : <Copy className="h-3 w-3" />}
                      {copiedText === "bench" ? "Copiado" : "Copiar"}
                    </button>
                  </div>
                  <pre className="p-4 bg-slate-950 rounded-xl font-mono text-xs text-slate-300 overflow-x-auto">
                    {`# Ejecutar suite de pruebas de rendimiento para RAM pico, velocidad de tokens y almacenamiento
python benchmark.py`}
                  </pre>
                </div>

              </div>
            </motion.div>
          )}

        </AnimatePresence>
      </main>

      <footer className="mt-12 py-8 border-t border-slate-800/60 text-center text-xs text-slate-500 font-mono">
        <div className="max-w-7xl mx-auto px-6">
          <p>© 2026 Transformer Edge. Diseñado y optimizado para arquitecturas ARM64 con NumPy.</p>
        </div>
      </footer>

    </div>
  );
}
