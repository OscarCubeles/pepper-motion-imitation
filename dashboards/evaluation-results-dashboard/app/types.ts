export type MethodId =
  | "current_solution_constrained"
  | "current_solution_raw"
  | "ikpy"
  | "original_solution";

export type MetricId = "EEAr" | "SOAx" | "HJL" | "WOM" | "HJAr" | "TSE" | "SYN";

export interface MethodRecord {
  id: MethodId;
  label: string;
  shortLabel: string;
  role: "proposed" | "ablation" | "baseline";
  color: string;
  dash: string;
}

export interface MetricRecord {
  id: MetricId;
  label: string;
  unit: string;
  direction: "lower" | "higher";
  definition: string;
}

export interface CategoryRecord {
  id: string;
  label: string;
}

export interface QualityRow {
  videoId: string;
  category: string;
  categoryLabel: string;
  method: MethodId;
  metric: MetricId;
  value: number | null;
}

export type SingularityFilter = "all" | "any" | "dual" | "elbow" | "shoulder" | "none";

export interface VideoSingularityRecord {
  videoId: string;
  category: string;
  categoryLabel: string;
  classificationAvailable: boolean;
  hasAny: boolean;
  hasDual: boolean;
  hasElbowRoll: boolean;
  hasShoulderRoll: boolean;
  totalFrameCount: number;
  singularFrameCount: number;
  dualFrameCount: number;
  elbowRollFrameCount: number;
  shoulderRollFrameCount: number;
  leftArmDualFrameCount: number;
  leftArmElbowRollFrameCount: number;
  leftArmShoulderRollFrameCount: number;
  leftArmNoSingularityFrameCount: number;
  rightArmDualFrameCount: number;
  rightArmElbowRollFrameCount: number;
  rightArmShoulderRollFrameCount: number;
  rightArmNoSingularityFrameCount: number;
  diagnosticArmCount: number;
  expectedDiagnosticArmCount: number;
}

export interface PairedComparison {
  baseline: MethodId;
  metric: MetricId;
  n: number;
  meanDifference: number;
  medianDifference: number;
  ciLow: number;
  ciHigh: number;
  rankBiserial: number;
  pValue: number;
  pAdjusted: number;
  wins: number;
  ties: number;
  losses: number;
}

export interface TrajectoryRow {
  videoId: string;
  category: string;
  method: MethodId;
  frameId: number;
  timestamp: number;
  side: "left" | "right";
  EEAr: number | null;
  SOAx: number | null;
  HJL: number | null;
  WOM: number | null;
  HJAr: number | null;
  TSE: number | null;
  SYN: number | null;
}

export interface PerformanceRow {
  videoId: string;
  category: string;
  categoryLabel: string;
  method: MethodId;
  repeatIndex?: number;
  repeatCount?: number;
  frameCount?: number;
  latency_ms: number | null;
  latency_jitter_ms: number | null;
  cpu_percent: number | null;
  gpu_percent: number | null;
  memory_percent: number | null;
  memory_mb: number | null;
}

export interface LatencySample {
  videoId: string;
  category: string;
  method: MethodId;
  repeatIndex: number;
  frameId: number;
  value: number;
}

export interface PerformanceFrameRow {
  videoId: string;
  category: string;
  method: MethodId;
  repeatIndex: number;
  frameId: number;
  timestamp: number;
  latency_ms: number | null;
  cpu_percent: number | null;
  gpu_percent: number | null;
  memory_percent: number | null;
  memory_mb: number | null;
}

export interface AnalysisData {
  metadata: {
    generatedAt: string;
    seed: number;
    bootstrapSamples: number;
    permutationSamples: number;
    videoCount: number;
    frameCount: number;
    categoryCount: number;
    primaryUnit: string;
    sources: string[];
    caveats: string[];
  };
  methods: MethodRecord[];
  metrics: MetricRecord[];
  performanceMetrics: Array<{
    id: keyof PerformanceRow;
    label: string;
    unit: string;
    direction: "lower";
  }>;
  categories: CategoryRecord[];
  videoSingularities: VideoSingularityRecord[];
  qualityRows: QualityRow[];
  pairedComparisons: PairedComparison[];
  identicalSeries: Array<{ left: MethodId; right: MethodId; metrics: MetricId[] }>;
  trajectoryRows: TrajectoryRow[];
  performanceRepeats: PerformanceRow[];
  performanceVideos: PerformanceRow[];
  performanceFrames: PerformanceFrameRow[];
  latencySamples: LatencySample[];
  sourceNotes: {
    ik: string[];
    performance: string[];
    environment: Record<string, unknown>;
  };
}
