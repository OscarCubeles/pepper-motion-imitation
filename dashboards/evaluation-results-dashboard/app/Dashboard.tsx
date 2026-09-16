"use client";

/* eslint-disable react/no-unescaped-entities */

import { useEffect, useMemo, useState } from "react";
import type { CSSProperties, Dispatch, SetStateAction } from "react";
import {
  downloadRowsCsv,
  ExportButtons,
  formatValue,
  getSingularityCategoryCounts,
  LatencyEcdfPlot,
  MethodLegend,
  PerformanceTimelinePlot,
  qualityMetricAxisBounds,
  QualityMatrixPlot,
  CategoryTrendPlot,
  TrajectoryPlot,
  VerticalCategoryBarPlot,
  SingularityCategoryBarPlot,
  VerticalDistributionPlot,
  VerticalEffectBarPlot,
  VerticalMeanBarPlot,
} from "./charts";
import type {
  AnalysisData,
  MethodId,
  MethodRecord,
  MetricId,
  MetricRecord,
  PairedComparison,
  PerformanceRow,
  QualityRow,
  SingularityFilter,
  VideoSingularityRecord,
} from "./types";
import type { FigureDisplayOptions } from "./charts";

const methodOrder: MethodId[] = [
  "current_solution_constrained",
  "current_solution_raw",
  "ikpy",
  "original_solution",
];
const metricDisplayOrder: MetricId[] = ["EEAr", "SOAx", "HJL", "TSE", "WOM", "SYN", "HJAr"];

const academicPalettes: Record<string, { label: string; colors: string[] }> = {
  current: { label: "Current teal / indigo", colors: ["#147d86", "#8eb3b2", "#e4860b", "#6d72a6"] },
  muted: { label: "Muted academic", colors: ["#2f6f73", "#8aa6a3", "#c47a35", "#6f7597"] },
  nature: { label: "Sage and ochre", colors: ["#2f6b5f", "#94ad9a", "#c8873e", "#626b88"] },
  contrast: { label: "Journal contrast", colors: ["#176b87", "#78a6a8", "#c96b35", "#5d679c"] },
  slate: { label: "Slate and copper", colors: ["#315b67", "#91a6ae", "#bd7445", "#646d8f"] },
  blue: { label: "Blue tones", colors: ["#d6e2f0", "#bdd7e7", "#6baed6", "#2171b5"] },
  pink: { label: "Pink tones", colors: ["#f5d4c9", "#fbb4b9", "#f768a1", "#ae017e"] },
  purple: { label: "Purple tones", colors: ["#dedbec", "#cbc9e2", "#9e9ac8", "#6a51a3"] },
  mixed: { label: "Mixed dark tones", colors: ["#ae017e", "#2171b5", "#6a51a3", "#238b45"] },
  mixedMedium: { label: "Mixed medium tones", colors: ["#6baed6", "#f768a1", "#9e9ac8", "#41ab5d"] },
};

const performanceMatrixMetrics = [
  { id: "latency_ms", label: "Average latency", unit: "ms" },
  { id: "latency_jitter_ms", label: "Average latency jitter", unit: "ms" },
] as const;

const defaultFigureOptions: FigureDisplayOptions = {
  methods: methodOrder,
  showValues: true,
  showInstances: false,
  showMinMax: false,
  showMinMaxValues: false,
  showLegend: false,
  showDirection: true,
  showSampleSize: true,
  showCategoryNote: true,
};

function FigureHeading({ code, title, note, svgId, filename }: { code?: string; title: string; note: string; svgId: string; filename: string }) {
  return <div className="figure-heading"><div>{code && <span className="metric-code">{code}</span>}<h3>{title}</h3><p>{note}</p></div><ExportButtons svgId={svgId} filename={filename} /></div>;
}

function FigureControls({ id, data, options, setOptions, availableMethods = methodOrder, includeValueControl = true, includeInstanceControl = true, includeMinMaxControl = true, includeQualityContextControls = false }: {
  id: string;
  data: AnalysisData;
  options: FigureDisplayOptions;
  setOptions: Dispatch<SetStateAction<FigureDisplayOptions>>;
  availableMethods?: MethodId[];
  includeValueControl?: boolean;
  includeInstanceControl?: boolean;
  includeMinMaxControl?: boolean;
  includeQualityContextControls?: boolean;
}) {
  const visibleMethods = data.methods.filter((method) => availableMethods.includes(method.id));
  const toggleMethod = (method: MethodId) => setOptions((current) => {
    const included = current.methods.includes(method);
    if (included && current.methods.filter((item) => availableMethods.includes(item)).length === 1) return current;
    return { ...current, methods: included ? current.methods.filter((item) => item !== method) : [...current.methods, method] };
  });
  const toggle = (key: "showValues" | "showInstances" | "showMinMax" | "showMinMaxValues" | "showLegend" | "showDirection" | "showSampleSize" | "showCategoryNote") => setOptions((current) => ({ ...current, [key]: !current[key] }));
  return <>
    <fieldset className="figure-controls" aria-describedby={`${id}-help`}>
      <legend>Figure controls</legend>
      <span id={`${id}-help`} className="sr-only">These choices are included in SVG and PNG exports.</span>
      <div className="method-checks" aria-label="Methods to show">
        {visibleMethods.map((method) => <label key={method.id} className="method-check"><input type="checkbox" checked={options.methods.includes(method.id)} onChange={() => toggleMethod(method.id)} /><i style={{ background: method.color }} aria-hidden="true" />{method.shortLabel}</label>)}
      </div>
      <div className="display-checks">
        {includeValueControl && <label><input type="checkbox" checked={options.showValues} onChange={() => toggle("showValues")} /> Show mean values</label>}
        {includeInstanceControl && <label><input type="checkbox" checked={options.showInstances} onChange={() => toggle("showInstances")} /> Show video instances</label>}
        {includeMinMaxControl && <label><input type="checkbox" checked={options.showMinMax} onChange={() => toggle("showMinMax")} /> Show min-max range</label>}
        {includeMinMaxControl && <label><input type="checkbox" checked={options.showMinMaxValues} onChange={() => toggle("showMinMaxValues")} /> Show min-max values</label>}
        {includeQualityContextControls && <label><input type="checkbox" checked={options.showDirection} onChange={() => toggle("showDirection")} /> Show higher/lower-is-better text</label>}
        {includeQualityContextControls && <label><input type="checkbox" checked={options.showSampleSize} onChange={() => toggle("showSampleSize")} /> Show sample size (n)</label>}
        {id === "categories" && <label><input type="checkbox" checked={options.showCategoryNote} onChange={() => toggle("showCategoryNote")} /> Show category description</label>}
        <label><input type="checkbox" checked={options.showLegend} onChange={() => toggle("showLegend")} /> Show legend in figure</label>
      </div>
    </fieldset>
    {id === "performance" && <PerformanceSingularityMatrixSection data={data} />}
    {id === "categories" && <><QualityMatrixSections data={data} /><CategoryTrendSection data={data} /><CategoryImprovementMatrix data={data} /><SingularityCategorySection data={data} /></>}
  </>;
}

function QualityFigure({ metric, rows, data, options, sampleSize, scopeId = "quality", filenamePrefix = "quality", populationLabel }: { metric: MetricRecord; rows: QualityRow[]; data: AnalysisData; options: FigureDisplayOptions; sampleSize: number; scopeId?: string; filenamePrefix?: string; populationLabel?: string }) {
  const svgId = `${scopeId}-${metric.id}`;
  const figureRows = rows.filter((row) => row.metric === metric.id && Number.isFinite(row.value)).map((row) => ({ ...row, value: row.value as number }));
  const contextParts = [
    populationLabel,
    options.showDirection ? `${metric.direction === "lower" ? "Lower" : "Higher"} is better` : undefined,
    options.showSampleSize ? `n = ${sampleSize} ${sampleSize === 1 ? "video" : "videos"}` : undefined,
  ].filter((part): part is string => Boolean(part));
  const note = contextParts.join(" - ");
  const populationDescription = populationLabel ? ` for ${populationLabel.toLowerCase()}` : "";
  return <article className="figure-card"><FigureHeading code={metric.id} title={metric.label} note={note} svgId={svgId} filename={`${filenamePrefix}-${metric.id.toLowerCase()}`} /><VerticalMeanBarPlot rows={figureRows} methods={data.methods} options={options} svgId={svgId} ariaLabel={`${metric.label} across ${sampleSize} ${sampleSize === 1 ? "video" : "videos"}${populationDescription}`} yDomain={qualityMetricAxisBounds(metric.id)} /><p className="figure-note">{populationLabel && <><strong>Video classification:</strong> {populationLabel}. </>}Bars are video-level means{options.showInstances ? `; circles are the ${sampleSize} individual ${sampleSize === 1 ? "video" : "videos"}.` : "."}</p></article>;
}

function DistributionQualityFigure({ metric, rows, data, options, sampleSize, scopeId, filenamePrefix, populationLabel }: { metric: MetricRecord; rows: QualityRow[]; data: AnalysisData; options: FigureDisplayOptions; sampleSize: number; scopeId: string; filenamePrefix: string; populationLabel: string }) {
  const svgId = `${scopeId}-${metric.id}`;
  const figureRows = rows.filter((row) => row.metric === metric.id && Number.isFinite(row.value)).map((row) => ({ ...row, value: row.value as number }));
  const contextParts = [
    populationLabel,
    options.showDirection ? `${metric.direction === "lower" ? "Lower" : "Higher"} is better` : undefined,
    options.showSampleSize ? `n = ${sampleSize} ${sampleSize === 1 ? "video" : "videos"}` : undefined,
  ].filter((part): part is string => Boolean(part));
  const note = contextParts.join(" - ");
  const bounds = qualityMetricAxisBounds(metric.id);
  const boundsNote = bounds[1] === undefined ? "The y-axis starts at 0 and uses a data-driven upper limit." : `The y-axis is fixed from ${bounds[0]} to ${bounds[1]}.`;
  return <article className="figure-card distribution-figure-card"><FigureHeading code={metric.id} title={metric.label} note={note} svgId={svgId} filename={`${filenamePrefix}-${metric.id.toLowerCase()}`} /><VerticalDistributionPlot rows={figureRows} methods={data.methods} options={options} svgId={svgId} ariaLabel={`${metric.label} across ${sampleSize} ${sampleSize === 1 ? "video" : "videos"}`} yDomain={bounds} /><p className="figure-note"><strong>Video classification:</strong> {populationLabel}. Translucent diamonds are individual video summaries; the outlined diamond is their mean. {boundsNote}</p></article>;
}

const singularityFilterLabels: Record<SingularityFilter, string> = {
  all: "All videos",
  any: "Videos with at least one singularity",
  dual: "Dual-singularity videos",
  elbow: "Elbow-roll-singularity videos",
  shoulder: "Shoulder-roll-singularity videos",
  none: "Videos without singularities",
};

const correlationStates: Array<{ id: Exclude<SingularityFilter, "all">; label: string }> = [
  { id: "any", label: "Any singularity" },
  { id: "dual", label: "Dual singularity" },
  { id: "elbow", label: "Elbow-roll singularity" },
  { id: "shoulder", label: "Shoulder-roll singularity" },
  { id: "none", label: "No singularity" },
];

const videoKey = (category: string, videoId: string) => `${category}\u0000${videoId}`;

function matchesSingularityFilter(record: VideoSingularityRecord, filter: SingularityFilter): boolean {
  if (filter === "all") return true;
  if (filter === "any") return record.hasAny;
  if (filter === "dual") return record.hasDual;
  if (filter === "elbow") return record.hasElbowRoll;
  if (filter === "shoulder") return record.hasShoulderRoll;
  return record.classificationAvailable && !record.hasAny;
}

function SingularityBadge({ active, children }: { active: boolean; children: string }) {
  return <span className={`singularity-badge ${active ? "is-active" : "is-inactive"}`}>{children}</span>;
}

function SingularityVideoTable({ records }: { records: VideoSingularityRecord[] }) {
  return <div className="table-wrap singularity-table"><table><caption>Video-level classification from unconstrained HJL singularity diagnostics. These summary counts identify complete frames in which either arm has the listed state.</caption><thead><tr><th>Category</th><th>Video</th><th>Classification</th><th>Any frames</th><th>Dual</th><th>Elbow</th><th>Shoulder</th></tr></thead><tbody>{records.map((record) => <tr key={videoKey(record.category, record.videoId)}><td>{record.categoryLabel}</td><td><strong>{record.videoId}</strong></td><td>{record.classificationAvailable ? <div className="singularity-badges">{record.hasAny ? <><SingularityBadge active={record.hasDual}>Dual</SingularityBadge><SingularityBadge active={record.hasElbowRoll}>Elbow</SingularityBadge><SingularityBadge active={record.hasShoulderRoll}>Shoulder</SingularityBadge></> : <SingularityBadge active>No singularity</SingularityBadge>}</div> : <SingularityBadge active={false}>Unavailable</SingularityBadge>}</td><td>{record.singularFrameCount}</td><td>{record.dualFrameCount}</td><td>{record.elbowRollFrameCount}</td><td>{record.shoulderRollFrameCount}</td></tr>)}</tbody></table></div>;
}

function PerformanceRangeTable({ rows, methods }: { rows: PerformanceRow[]; methods: MethodRecord[] }) {
  const summary = (methodId: MethodId, metricId: typeof performanceMatrixMetrics[number]["id"]) => {
    const values = rows
      .filter((row) => row.method === methodId)
      .map((row) => row[metricId])
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    if (!values.length) return null;
    return {
      mean: values.reduce((sum, value) => sum + value, 0) / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      n: values.length,
    };
  };

  return <div className="table-wrap performance-range-wrap"><table className="performance-range-table"><caption>Latency summaries across the selected video population. Minimum and maximum values are the observed per-video summary range.</caption><thead><tr><th rowSpan={2} scope="col">Metric</th>{methods.map((method) => <th key={method.id} scope="colgroup" colSpan={3}><i className="table-color-key" style={{ background: method.color }} aria-hidden="true" />{method.shortLabel}</th>)}</tr><tr>{methods.flatMap((method) => [<th key={`${method.id}-mean`} scope="col">Average</th>, <th key={`${method.id}-min`} scope="col">Min</th>, <th key={`${method.id}-max`} scope="col">Max</th>])}</tr></thead><tbody>{performanceMatrixMetrics.map((metric) => <tr key={metric.id}><th scope="row">{metric.label}<small>{metric.unit}</small></th>{methods.flatMap((method) => { const result = summary(method.id, metric.id); return [<td key={`${method.id}-${metric.id}-mean`} title={result ? `Mean across ${result.n} selected videos` : "Unavailable"}>{result ? formatValue(result.mean, 2) : "—"}</td>, <td key={`${method.id}-${metric.id}-min`}>{result ? formatValue(result.min, 2) : "—"}</td>, <td key={`${method.id}-${metric.id}-max`}>{result ? formatValue(result.max, 2) : "—"}</td>]; })}</tr>)}</tbody></table></div>;
}

function PerformanceSingularityMatrixSection({ data }: { data: AnalysisData }) {
  const [filter, setFilter] = useState<SingularityFilter>("all");
  const selectedVideos = useMemo(() => (data.videoSingularities ?? []).filter((record) => matchesSingularityFilter(record, filter)), [data, filter]);
  const selectedKeys = useMemo(() => new Set(selectedVideos.map((record) => videoKey(record.category, record.videoId))), [selectedVideos]);
  const rows = useMemo(() => data.performanceVideos.filter((row) => selectedKeys.has(videoKey(row.category, row.videoId))), [data, selectedKeys]);
  const classificationByVideo = new Map((data.videoSingularities ?? []).map((record) => [videoKey(record.category, record.videoId), record]));

  return <section className="performance-distribution-section" id="performance-distribution"><div className="section-heading"><div><p className="section-kicker">05B - Performance by singularity classification</p><h2>Compare latency across the same singularity-defined populations.</h2></div><button className="data-link" type="button" disabled={!rows.length} onClick={() => downloadRowsCsv(`performance-singularity-${filter}.csv`, rows.map((row) => { const classification = classificationByVideo.get(videoKey(row.category, row.videoId)); return { videoId: row.videoId, category: row.category, method: row.method, classificationAvailable: classification?.classificationAvailable, hasAny: classification?.hasAny, hasDual: classification?.hasDual, hasElbowRoll: classification?.hasElbowRoll, hasShoulderRoll: classification?.hasShoulderRoll, ...Object.fromEntries(performanceMatrixMetrics.map((metric) => [metric.id, row[metric.id]])) }; }))}>Download selected CSV</button></div><p className="section-copy">The two rows report average latency and average latency jitter. For every method, the table shows the mean and the observed minimum and maximum per-video summaries, all in milliseconds.</p><div className="singularity-toolbar"><label className="select-field"><span>Video classification</span><select value={filter} onChange={(event) => setFilter(event.target.value as SingularityFilter)}>{(Object.keys(singularityFilterLabels) as SingularityFilter[]).map((item) => <option key={item} value={item}>{singularityFilterLabels[item]} ({(data.videoSingularities ?? []).filter((record) => matchesSingularityFilter(record, item)).length})</option>)}</select></label><div className="selection-summary" aria-live="polite"><strong>{selectedVideos.length}</strong><span>{selectedVideos.length === 1 ? "video selected" : "videos selected"}</span></div></div>{selectedVideos.length ? <><details className="singularity-video-detail"><summary>Show classified videos and singular-frame counts</summary><SingularityVideoTable records={selectedVideos} /></details><article className="wide-figure performance-matrix-card"><div className="figure-heading"><div><h3>Latency summary by method</h3><p>{singularityFilterLabels[filter]} - n = {selectedVideos.length} {selectedVideos.length === 1 ? "video" : "videos"}</p></div></div><PerformanceRangeTable rows={rows} methods={data.methods} /><p className="figure-note">Average, minimum, and maximum are calculated across the selected videos using each video&apos;s performance summary; they are not frame-level extremes.</p></article></> : <div className="empty-selection" role="status"><strong>No videos match this classification.</strong><p>Choose another singularity group or refresh the analysis after new evaluation results are available.</p></div>}</section>;
}

function QualityMatrixSections({ data }: { data: AnalysisData }) {
  const [categoryMetric, setCategoryMetric] = useState<MetricId>(data.metrics[0]?.id ?? "TSE");
  const [categoryOptions, setCategoryOptions] = useState<FigureDisplayOptions>({ ...defaultFigureOptions, showInstances: false, showLegend: true });
  const [selectedCategory, setSelectedCategory] = useState(data.categories[0]?.id ?? "");
  const [metricOptions, setMetricOptions] = useState<FigureDisplayOptions>({ ...defaultFigureOptions, showInstances: false, showLegend: true });
  const category = data.categories.find((item) => item.id === selectedCategory) ?? data.categories[0];
  const categoryRows = data.qualityRows.filter((row) => row.category === category?.id);

  return <>
    <section className="quality-matrix-section" id="quality-category-matrix"><div className="section-heading"><div><p className="section-kicker">03A - Categories by method</p><h2>Compare one metric across every motion category.</h2></div><label className="select-field"><span>Metric</span><select value={categoryMetric} onChange={(event) => setCategoryMetric(event.target.value as MetricId)}>{data.metrics.map((metric) => <option key={metric.id} value={metric.id}>{metric.id} - {metric.label}</option>)}</select></label></div><p className="section-copy">Categories are rows and methods are columns. Each colored bar is the category mean for the selected metric.</p><FigureControls id="quality-category-matrix" data={data} options={categoryOptions} setOptions={setCategoryOptions} /><article className="wide-figure matrix-figure-card"><FigureHeading code={categoryMetric} title={`${data.metrics.find((metric) => metric.id === categoryMetric)?.label ?? categoryMetric} by category and method`} note="Category means - descriptive view" svgId="quality-category-matrix-plot" filename={`quality-category-matrix-${categoryMetric.toLowerCase()}`} /><QualityMatrixPlot rows={data.qualityRows} methods={data.methods} rowLabels={data.categories} metricForRows={() => categoryMetric} options={categoryOptions} svgId="quality-category-matrix-plot" ariaLabel={`${categoryMetric} by motion category and method`} /><p className="figure-note">Rows are motion categories; columns are methods. Each row is scaled to its largest selected method mean.</p></article></section>
    <section className="quality-matrix-section" id="quality-metric-matrix"><div className="section-heading"><div><p className="section-kicker">03B - Metrics by category</p><h2>Inspect the metric profile of one motion category.</h2></div><label className="select-field"><span>Video category</span><select value={category?.id ?? ""} onChange={(event) => setSelectedCategory(event.target.value)}>{data.categories.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label></div><p className="section-copy">Metrics are rows and methods are columns. Select a category to see where each method performs better or worse across the seven quality measures.</p><FigureControls id="quality-metric-matrix" data={data} options={metricOptions} setOptions={setMetricOptions} /><article className="wide-figure matrix-figure-card"><FigureHeading title={`${category?.label ?? "Selected category"} metric profile`} note="Metric means - descriptive view" svgId="quality-metric-matrix-plot" filename={`quality-metric-matrix-${category?.id ?? "category"}`} /><QualityMatrixPlot rows={categoryRows} methods={data.methods} rowLabels={data.metrics.map((metric) => ({ id: metric.id, label: metric.id }))} metricForRows={(rowId) => rowId as MetricId} options={metricOptions} svgId="quality-metric-matrix-plot" ariaLabel={`Quality metrics for ${category?.label ?? "selected category"}`} /><p className="figure-note">Rows are quality metrics; columns are methods. Native metric values are preserved, while bar lengths are normalized within each row.</p></article></section>
  </>;
}

function CategoryTrendSection({ data }: { data: AnalysisData }) {
  const [metric, setMetric] = useState<MetricId>(data.metrics[0]?.id ?? "TSE");
  const [selectedCategories, setSelectedCategories] = useState<string[]>(() => data.categories.map((category) => category.id));
  const [options, setOptions] = useState<FigureDisplayOptions>({ ...defaultFigureOptions, showInstances: false, showLegend: true });
  const categories = data.categories.filter((category) => selectedCategories.includes(category.id));
  const selectedMetric = data.metrics.find((item) => item.id === metric);
  const toggleCategory = (categoryId: string) => setSelectedCategories((current) => current.includes(categoryId) ? (current.length > 1 ? current.filter((item) => item !== categoryId) : current) : [...current, categoryId]);

  return <section className="quality-trend-section" id="quality-category-trend"><div className="section-heading"><div><p className="section-kicker">03C - Category trend lines</p><h2>Follow each method across video categories.</h2></div><label className="select-field"><span>Metric</span><select value={metric} onChange={(event) => setMetric(event.target.value as MetricId)}>{data.metrics.map((item) => <option key={item.id} value={item.id}>{item.id} - {item.label}</option>)}</select></label></div><p className="section-copy">Each line joins the category-level mean for one method. Use the category selectors to focus the comparison; missing combinations remain absent rather than being treated as zero.</p><div className="trend-category-toolbar" aria-label="Video category selector">{data.categories.map((category) => <label key={category.id}><input type="checkbox" checked={selectedCategories.includes(category.id)} onChange={() => toggleCategory(category.id)} />{category.label}</label>)}</div><FigureControls id="quality-category-trend" data={data} options={options} setOptions={setOptions} includeInstanceControl={false} includeMinMaxControl={false} /><article className="wide-figure trend-figure-card"><FigureHeading code={metric} title={`${selectedMetric?.label ?? metric} by video category`} note={`${selectedMetric?.direction === "lower" ? "Lower" : "Higher"} is better - category means`} svgId="quality-category-trend-plot" filename={`quality-category-trend-${metric.toLowerCase()}`} /><CategoryTrendPlot rows={data.qualityRows} methods={data.methods} categories={categories} metric={metric} options={options} svgId="quality-category-trend-plot" ariaLabel={`${selectedMetric?.label ?? metric} trend by video category`} /><p className="figure-note">Dots are category means; lines connect only categories with available observations. Method colours follow the selected palette.</p></article></section>;
}

function SingularityCategorySection({ data }: { data: AnalysisData }) {
  const counts = getSingularityCategoryCounts(data);
  const methodColors = data.methods.map((method) => method.color);
  const colors = [methodColors[0] ?? "#ae017e", methodColors[1] ?? "#2171b5", methodColors[2] ?? "#6a51a3", methodColors[3] ?? "#238b45"];
  const singularityColumns = [
    { id: "dual", label: "Dual singularity", color: colors[0] },
    { id: "elbow", label: "Elbow-roll singularity", color: colors[1] },
    { id: "shoulder", label: "Shoulder-roll singularity", color: colors[2] },
    { id: "none", label: "No singularity", color: colors[3] },
  ] as const;
  const arms = [{ id: "left", label: "Left arm" }, { id: "right", label: "Right arm" }] as const;
  return <>
    <section className="quality-trend-section singularity-counts-section" id="singularity-category-counts"><div className="section-heading"><div><p className="section-kicker">03F - Singularity prevalence by category</p><h2>How often do singularity patterns occur in each arm?</h2></div></div><p className="section-copy">Each category has separate left-arm and right-arm stacked bars. Within an arm, every frame belongs to exactly one mutually exclusive state.</p><article className="wide-figure singularity-counts-card"><FigureHeading title="Singularity classifications by arm and video category" note="L = left arm; R = right arm. Unavailable diagnostics are excluded." svgId="singularity-category-counts" filename="singularity-category-counts" /><SingularityCategoryBarPlot data={data} colors={colors} svgId="singularity-category-counts" /><p className="figure-note">Each arm is counted independently, so both the left and right bar total the classified-frame population for that category.</p></article></section>
    <section className="quality-trend-section singularity-table-section" id="singularity-category-table"><div className="section-heading"><div><p className="section-kicker">03G - Singularity counts by category</p><h2>Compare each arm in separate matrix tables.</h2></div></div><p className="section-copy">Motion categories are rows and mutually exclusive arm states are columns, following the structure of the 03D improvement matrix.</p><article className="wide-figure singularity-summary-card">{arms.map((arm) => <div className="arm-matrix-block" key={arm.id}><h3>{arm.label}</h3><div className="table-wrap singularity-matrix-wrap"><table className="singularity-summary-table"><caption>{arm.label} classified-frame counts by motion category. Cell intensity is relative to all classified frames in the category.</caption><thead><tr><th scope="col">Video category</th>{singularityColumns.map((column) => <th scope="col" key={column.id}><i className="table-color-key" style={{ background: column.color }} aria-hidden="true" /><span>{column.label}</span><small>frame count</small></th>)}</tr></thead><tbody>{counts.map((item) => <tr key={item.category.id}><th scope="row">{item.category.label}<small>{item.totalFrames.toLocaleString()} total frames</small></th>{singularityColumns.map((column) => { const value = item.arms[arm.id][column.id]; const strength = item.totalFrames ? value / item.totalFrames : 0; return <td key={column.id} style={{ backgroundColor: `color-mix(in srgb, ${column.color} ${Math.round(12 + strength * 58)}%, white)` }} title={`${item.category.label}, ${arm.label.toLowerCase()}: ${value} ${column.label.toLowerCase()} frames (${formatValue(strength * 100, 1)}%)`}>{value.toLocaleString()}</td>; })}</tr>)}</tbody></table></div></div>)}<p className="figure-note">In each arm table, dual, elbow-roll, shoulder-roll, and no-singularity counts form an exclusive partition; every row sums to the displayed total.</p></article></section>
  </>;
}

function CategoryImprovementMatrix({ data }: { data: AnalysisData }) {
  const [showMetricNames, setShowMetricNames] = useState(true);
  const [showMatrixDescription, setShowMatrixDescription] = useState(true);
  useEffect(() => {
    document.body.classList.toggle("hide-improvement-metric-names", !showMetricNames);
    document.body.classList.toggle("hide-improvement-caption", !showMatrixDescription);
    return () => {
      document.body.classList.remove("hide-improvement-metric-names");
      document.body.classList.remove("hide-improvement-caption");
    };
  }, [showMetricNames, showMatrixDescription]);
  return <><div className="metric-name-toggle-group"><label className="metric-name-toggle"><input type="checkbox" checked={showMetricNames} onChange={(event) => setShowMetricNames(event.target.checked)} /> Show full metric names</label><label className="metric-name-toggle"><input type="checkbox" checked={showMatrixDescription} onChange={(event) => setShowMatrixDescription(event.target.checked)} /> Show matrix description</label></div><CategoryImprovementHeatmap data={data} /><SingularityImprovementCorrelation data={data} /></>;
  const grouped = new Map<string, number[]>();
  data.qualityRows.filter((row) => Number.isFinite(row.value)).forEach((row) => {
    const key = `${row.category}\u0000${row.metric}\u0000${row.method}`;
    const values = grouped.get(key) ?? [];
    values.push(row.value as number);
    grouped.set(key, values);
  });
  const meanFor = (category: string, metric: MetricId, method: MethodId) => {
    const values = grouped.get(`${category}\u0000${metric}\u0000${method}`) ?? [];
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  const metricMap = new Map(data.metrics.map((metric) => [metric.id, metric]));
  const improvement = (category: string, metricId: MetricId) => {
    const original = meanFor(category, metricId, "original_solution");
    const constrained = meanFor(category, metricId, "current_solution_constrained");
    if (original === null || constrained === null || original === 0) return null;
    const metric = metricMap.get(metricId);
    const signedChange = metric?.direction === "lower" ? original - constrained : constrained - original;
    return (signedChange / Math.abs(original)) * 100;
  };

  return <CategoryImprovementHeatmap data={data} />;

  return <section className="quality-improvement-section" id="quality-category-improvement"><div className="section-heading"><div><p className="section-kicker">03D - Constrained improvement by category</p><h2>Where does the constrained solution improve on Baseline?</h2></div></div><p className="section-copy">Cells show the percentage improvement of the constrained method over the Baseline for each category and metric. Positive values indicate improvement after respecting the metric's direction; negative values indicate deterioration.</p><article className="wide-figure improvement-matrix-card"><div className="table-wrap improvement-matrix-wrap"><table className="improvement-matrix"><caption>Percentage improvement of constrained over Baseline, averaged across videos within each category.</caption><thead><tr><th scope="col">Video category</th>{data.metrics.map((metric) => <th scope="col" key={metric.id}><span>{metric.id}</span><small>{metric.direction === "lower" ? "↓ lower" : "↑ higher"}</small></th>)}</tr></thead><tbody>{data.categories.map((category) => <tr key={category.id}><th scope="row">{category.label}</th>{data.metrics.map((metric) => { const value = improvement(category.id, metric.id); const intensity = value === null ? 0 : Math.min(Math.abs(value) / 100, 1); const tone = value === null ? "missing" : value >= 0 ? "positive" : "negative"; return <td key={metric.id} className={`improvement-cell ${tone}`} style={value === null ? undefined : { "--improvement-strength": intensity } as CSSProperties} title={value === null ? "Unavailable: no valid Original or constrained value" : `${formatValue(value, 1)}% improvement`}>{value === null ? "—" : `${value >= 0 ? "+" : ""}${formatValue(value, 1)}%`}</td>; })}</tr>)}</tbody></table></div><p className="figure-note">Green cells indicate improvement and coral cells indicate deterioration. Color intensity reflects the absolute percentage change; the printed value is the original percentage.</p></article></section>;
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function CategoryImprovementHeatmapOld({ data }: { data: AnalysisData }) {
  const [nativeUnits, setNativeUnits] = useState(false);
  const grouped = new Map<string, number[]>();
  data.qualityRows.filter((row) => Number.isFinite(row.value)).forEach((row) => {
    const key = `${row.category}\u0000${row.metric}\u0000${row.method}`;
    grouped.set(key, [...(grouped.get(key) ?? []), row.value as number]);
  });
  const meanFor = (category: string, metric: MetricId, method: MethodId) => {
    const values = grouped.get(`${category}\u0000${metric}\u0000${method}`) ?? [];
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  const metricMap = new Map(data.metrics.map((metric) => [metric.id, metric]));
  const changeFor = (category: string, metricId: MetricId) => {
    const original = meanFor(category, metricId, "original_solution");
    const constrained = meanFor(category, metricId, "current_solution_constrained");
    if (original === null || constrained === null) return null;
    const change = metricMap.get(metricId)?.direction === "lower" ? original - constrained : constrained - original;
    return { native: change, percent: original === 0 ? null : change / Math.abs(original) * 100 };
  };
  return <section className="quality-improvement-section" id="quality-category-improvement"><div className="section-heading"><div><p className="section-kicker">03D - Constrained improvement by category</p><h2>Constrained improvement over Baseline</h2></div><label className="toggle-control"><input type="checkbox" checked={nativeUnits} onChange={(event) => setNativeUnits(event.target.checked)} /> Show native metric units</label></div><p className="section-copy">Positive values mean the constrained method improves the metric after applying its direction. Negative values mean it is worse. Toggle native units to replace percentages with the direction-adjusted metric difference.</p><article className="wide-figure improvement-matrix-card"><div className="table-wrap improvement-matrix-wrap"><table className="improvement-matrix"><caption>{nativeUnits ? "Direction-adjusted native-unit difference" : "Percentage improvement"} of constrained over Baseline, averaged across videos within each category.</caption><thead><tr><th scope="col">Video category</th>{data.metrics.map((metric) => <th scope="col" key={metric.id}><span>{metric.id}</span><small>{metric.unit || (metric.direction === "lower" ? "lower is better" : "higher is better")}</small></th>)}</tr></thead><tbody>{data.categories.map((category) => <tr key={category.id}><th scope="row">{category.label}</th>{data.metrics.map((metric) => { const result = changeFor(category.id, metric.id); const value = nativeUnits ? result?.native ?? null : result?.percent ?? null; const strength = result?.percent === null || result === null ? 0 : Math.min(Math.abs(result.percent) / 100, 1); const tone = value === null ? "missing" : value >= 0 ? "positive" : "negative"; return <td key={metric.id} className={`improvement-cell ${tone}`} style={value === null ? undefined : { "--improvement-strength": strength } as CSSProperties} title={value === null ? "Unavailable" : `${formatValue(value, 2)}${nativeUnits ? ` ${metric.unit}` : "%"}`}>{value === null ? "—" : `${value >= 0 ? "+" : ""}${formatValue(value, 2)}${nativeUnits ? ` ${metric.unit}` : "%"}`}</td>; })}</tr>)}</tbody></table></div><div className="improvement-legend" aria-label="Improvement legend"><span><i className="improvement-swatch positive" />Improvement</span><span><i className="improvement-swatch negative" />Deterioration</span><span><i className="improvement-swatch missing" />Unavailable</span></div><p className="figure-note">The minus sign indicates a deterioration relative to Baseline. Cell color intensity reflects the percentage magnitude even when native units are displayed.</p></article></section>;
}

function CategoryImprovementHeatmapLegacy({ data }: { data: AnalysisData }) {
  const [nativeUnits, setNativeUnits] = useState(false);
  const [compareIkpy, setCompareIkpy] = useState(false);
  const grouped = new Map<string, number[]>();
  data.qualityRows.filter((row) => Number.isFinite(row.value)).forEach((row) => {
    const key = `${row.category}\u0000${row.metric}\u0000${row.method}`;
    grouped.set(key, [...(grouped.get(key) ?? []), row.value as number]);
  });
  const meanFor = (category: string, metric: MetricId, method: MethodId) => {
    const values = grouped.get(`${category}\u0000${metric}\u0000${method}`) ?? [];
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  const metricMap = new Map(data.metrics.map((metric) => [metric.id, metric]));
  const nativeUnit = (metric: MetricRecord) => metric.id === "HJL" || metric.id === "WOM" ? "percentage points" : metric.id === "TSE" ? "rad²" : metric.unit === "normalized error" || metric.unit === "normalized distance" || metric.unit === "normalized MSE" ? "normalized units" : metric.unit === "descriptor distance" ? "descriptor units" : metric.unit;
  const changeFor = (category: string, metricId: MetricId) => {
    const baselineMethod: MethodId = compareIkpy ? "ikpy" : "original_solution";
    const original = meanFor(category, metricId, baselineMethod);
    const constrained = meanFor(category, metricId, "current_solution_constrained");
    if (original === null || constrained === null) return null;
    const metric = metricMap.get(metricId);
    const nativeChange = metric?.direction === "lower" ? original - constrained : constrained - original;
    return { native: metricId === "HJL" || metricId === "WOM" ? nativeChange * 100 : nativeChange, percent: original === 0 ? null : nativeChange / Math.abs(original) * 100 };
  };
  return <section className="quality-improvement-section" id="quality-category-improvement"><div className="section-heading"><div><p className="section-kicker">03D - Constrained improvement by category</p><h2>Constrained improvement over Baseline</h2></div><label className="toggle-control"><input type="checkbox" checked={nativeUnits} onChange={(event) => setNativeUnits(event.target.checked)} /> Show native metric differences</label></div><p className="section-copy">Positive values indicate improvement after accounting for each metric&apos;s direction. Negative values indicate deterioration. Percentage mode is relative improvement; native mode shows the direction-adjusted difference in the metric&apos;s own units.</p><article className="wide-figure improvement-matrix-card"><div className="table-wrap improvement-matrix-wrap"><table className="improvement-matrix"><caption>{nativeUnits ? "Direction-adjusted native-unit differences" : "Relative percentage improvement"} of constrained over Baseline, averaged across videos within each category.</caption><thead><tr><th scope="col">Video category</th>{data.metrics.map((metric) => <th scope="col" key={metric.id}><span>{metric.id}</span><small>{metric.label}</small><em>{nativeUnits ? nativeUnit(metric) : "% improvement"}</em></th>)}</tr></thead><tbody>{data.categories.map((category) => <tr key={category.id}><th scope="row">{category.label}</th>{data.metrics.map((metric) => { const result = changeFor(category.id, metric.id); const value = nativeUnits ? result?.native ?? null : result?.percent ?? null; const strength = result?.percent === null || result === null ? 0 : Math.min(Math.abs(result.percent) / 100, 1); const neutral = value !== null && Math.abs(value) < 0.005; const tone = value === null ? "missing" : neutral ? "neutral" : value >= 0 ? "positive" : "negative"; return <td key={metric.id} className={`improvement-cell ${tone}`} style={value === null || neutral ? undefined : { "--improvement-strength": strength } as CSSProperties} title={value === null ? "Unavailable" : neutral ? "No meaningful change at displayed precision" : `${formatValue(value, 2)} ${nativeUnits ? nativeUnit(metric) : "% improvement"}`}>{value === null ? "—" : `${value >= 0 ? "+" : ""}${formatValue(value, 2)}`}</td>; })}</tr>)}</tbody></table></div><div className="improvement-legend" aria-label="Improvement legend"><span><i className="improvement-swatch positive" />Improvement</span><span><i className="improvement-swatch negative" />Deterioration</span><span><i className="improvement-swatch neutral" />No meaningful change</span><span><i className="improvement-swatch missing" />Unavailable</span></div><p className="figure-note">A minus sign means the constrained solution is worse than Baseline for that category and metric. Values that round to 0.00 are neutral gray. Cell intensity reflects relative percentage magnitude.</p></article></section>;
}

function CategoryImprovementHeatmap({ data }: { data: AnalysisData }) {
  const [nativeUnits, setNativeUnits] = useState(false);
  const [compareIkpy, setCompareIkpy] = useState(false);
  const [filter, setFilter] = useState<SingularityFilter>("all");
  const selectedVideos = (data.videoSingularities ?? []).filter((record) => matchesSingularityFilter(record, filter));
  const selectedVideoKeys = new Set(selectedVideos.map((record) => videoKey(record.category, record.videoId)));
  const grouped = new Map<string, number[]>();
  data.qualityRows.filter((row) => selectedVideoKeys.has(videoKey(row.category, row.videoId)) && Number.isFinite(row.value)).forEach((row) => {
    const key = `${row.category}\u0000${row.metric}\u0000${row.method}`;
    grouped.set(key, [...(grouped.get(key) ?? []), row.value as number]);
  });
  const meanFor = (category: string, metric: MetricId, method: MethodId) => {
    const values = grouped.get(`${category}\u0000${metric}\u0000${method}`) ?? [];
    return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  };
  const metricMap = new Map(data.metrics.map((metric) => [metric.id, metric]));
  const baselineLabel = compareIkpy ? "IKPy" : "Baseline";
  const nativeUnit = (metric: MetricRecord) => metric.id === "HJL" || metric.id === "WOM" ? "percentage points" : metric.id === "TSE" ? "rad²" : metric.unit === "normalized error" || metric.unit === "normalized distance" || metric.unit === "normalized MSE" ? "normalized units" : metric.unit === "descriptor distance" ? "descriptor units" : metric.unit;
  const changeFor = (category: string, metricId: MetricId) => {
    const baseline = meanFor(category, metricId, compareIkpy ? "ikpy" : "original_solution");
    const constrained = meanFor(category, metricId, "current_solution_constrained");
    if (baseline === null || constrained === null) return null;
    const metric = metricMap.get(metricId);
    const nativeChange = metric?.direction === "lower" ? baseline - constrained : constrained - baseline;
    return { native: metricId === "HJL" || metricId === "WOM" ? nativeChange * 100 : nativeChange, percent: baseline === 0 ? null : nativeChange / Math.abs(baseline) * 100 };
  };
  return <section className="quality-improvement-section" id="quality-category-improvement"><div className="section-heading"><div><p className="section-kicker">03D - Constrained improvement by category</p><h2>Constrained improvement over {compareIkpy ? "IKPy" : "Baseline"}</h2></div><div className="improvement-toggles"><label className="toggle-control"><input type="checkbox" checked={compareIkpy} onChange={(event) => setCompareIkpy(event.target.checked)} /> Compare with IKPy</label><label className="toggle-control"><input type="checkbox" checked={nativeUnits} onChange={(event) => setNativeUnits(event.target.checked)} /> Show native metric differences</label></div></div><p className="section-copy">Positive values indicate improvement after accounting for each metric&apos;s direction. Negative values indicate deterioration. The comparison baseline is {compareIkpy ? "IKPy" : "Baseline"}.</p><div className="singularity-toolbar"><label className="select-field"><span>Video classification</span><select value={filter} onChange={(event) => setFilter(event.target.value as SingularityFilter)}>{(Object.keys(singularityFilterLabels) as SingularityFilter[]).map((item) => <option key={item} value={item}>{singularityFilterLabels[item]} ({(data.videoSingularities ?? []).filter((record) => matchesSingularityFilter(record, item)).length})</option>)}</select></label><div className="selection-summary" aria-live="polite"><strong>{selectedVideos.length}</strong><span>{selectedVideos.length === 1 ? "video selected" : "videos selected"}</span></div></div><article className="wide-figure improvement-matrix-card"><div className="table-wrap improvement-matrix-wrap"><table className="improvement-matrix"><caption>{nativeUnits ? "Direction-adjusted native-unit differences" : "Relative percentage improvement"} of Constrained over {compareIkpy ? "IKPy" : "Baseline"}, averaged across {singularityFilterLabels[filter].toLowerCase()} within each category.</caption><thead><tr><th scope="col">Video category</th>{data.metrics.map((metric) => <th scope="col" key={metric.id}><span>{metric.id}</span><small>{metric.label}</small><em>{nativeUnits ? nativeUnit(metric) : "% improvement"}</em></th>)}</tr></thead><tbody>{data.categories.map((category) => <tr key={category.id}><th scope="row">{category.label}</th>{data.metrics.map((metric) => { const result = changeFor(category.id, metric.id); const value = nativeUnits ? result?.native ?? null : result?.percent ?? null; const strength = result?.percent === null || result === null ? 0 : Math.min(Math.abs(result.percent) / 100, 1); const neutral = value !== null && Math.abs(value) < 0.005; const tone = value === null ? "missing" : neutral ? "neutral" : value > 0 ? "positive" : "negative"; const text = value === null ? "—" : neutral || value === 0 ? formatValue(value, 2) : `${value > 0 ? "+" : ""}${formatValue(value, 2)}`; return <td key={metric.id} className={`improvement-cell ${tone}`} style={value === null || neutral ? undefined : { "--improvement-strength": strength } as CSSProperties} title={value === null ? `Unavailable for ${singularityFilterLabels[filter].toLowerCase()}` : neutral ? "No meaningful change at displayed precision" : `${formatValue(value, 2)} ${nativeUnits ? nativeUnit(metric) : "% improvement"}`}>{text}</td>; })}</tr>)}</tbody></table></div><div className="improvement-legend" aria-label="Improvement legend"><span><i className="improvement-swatch positive" />Improvement</span><span><i className="improvement-swatch negative" />Deterioration</span><span><i className="improvement-swatch neutral" />No meaningful change</span><span><i className="improvement-swatch missing" />Unavailable</span></div><p className="figure-note">A minus sign means Constrained is worse than {baselineLabel} for the selected video population, category, and metric. Empty cells mean that classification has no valid paired videos in that category.</p></article></section>;
}

function SingularityImprovementCorrelation({ data }: { data: AnalysisData }) {
  const [compareIkpy, setCompareIkpy] = useState(false);
  const baselineMethod: MethodId = compareIkpy ? "ikpy" : "original_solution";
  const baselineLabel = compareIkpy ? "IKPy" : "Baseline";
  const classifiedVideos = (data.videoSingularities ?? []).filter((record) => record.classificationAvailable);
  const values = new Map<string, number>();
  data.qualityRows.forEach((row) => {
    if (Number.isFinite(row.value)) values.set(`${videoKey(row.category, row.videoId)}\u0000${row.metric}\u0000${row.method}`, row.value as number);
  });
  const mean = (items: number[]) => items.length ? items.reduce((sum, value) => sum + value, 0) / items.length : null;
  const pointBiserial = (samples: Array<{ triggered: boolean; improvement: number }>) => {
    if (samples.length < 3 || !samples.some((sample) => sample.triggered) || !samples.some((sample) => !sample.triggered)) return null;
    const xMean = samples.filter((sample) => sample.triggered).length / samples.length;
    const yMean = samples.reduce((sum, sample) => sum + sample.improvement, 0) / samples.length;
    const numerator = samples.reduce((sum, sample) => sum + ((sample.triggered ? 1 : 0) - xMean) * (sample.improvement - yMean), 0);
    const xSquares = samples.reduce((sum, sample) => sum + ((sample.triggered ? 1 : 0) - xMean) ** 2, 0);
    const ySquares = samples.reduce((sum, sample) => sum + (sample.improvement - yMean) ** 2, 0);
    const denominator = Math.sqrt(xSquares * ySquares);
    return denominator > 0 ? numerator / denominator : null;
  };
  const resultFor = (state: Exclude<SingularityFilter, "all">, metric: MetricRecord) => {
    const samples = classifiedVideos.flatMap((record) => {
      const prefix = `${videoKey(record.category, record.videoId)}\u0000${metric.id}\u0000`;
      const constrained = values.get(`${prefix}current_solution_constrained`);
      const baseline = values.get(`${prefix}${baselineMethod}`);
      if (constrained === undefined || baseline === undefined) return [];
      const improvement = metric.direction === "lower" ? baseline - constrained : constrained - baseline;
      return [{ triggered: matchesSingularityFilter(record, state), improvement }];
    });
    const triggered = samples.filter((sample) => sample.triggered).map((sample) => sample.improvement);
    const notTriggered = samples.filter((sample) => !sample.triggered).map((sample) => sample.improvement);
    return { correlation: pointBiserial(samples), n: samples.length, triggeredN: triggered.length, triggeredMean: mean(triggered), notTriggeredMean: mean(notTriggered) };
  };

  return <section className="quality-correlation-section" id="singularity-improvement-correlation"><div className="section-heading"><div><p className="section-kicker">03E - Singularity state and metric improvement</p><h2>Are triggered states associated with larger improvements?</h2></div><label className="toggle-control"><input type="checkbox" checked={compareIkpy} onChange={(event) => setCompareIkpy(event.target.checked)} /> Compare with IKPy</label></div><p className="section-copy">Each cell is a point-biserial correlation between a video-level state indicator and the direction-adjusted improvement of Constrained over {baselineLabel}. Positive values mean videos triggering that state tend to improve more; negative values mean they tend to improve less.</p><article className="wide-figure correlation-matrix-card"><h3>State–improvement association across paired videos</h3><p className="correlation-subtitle">Point-biserial r ranges from −1 to +1. This is a descriptive association, not evidence that a singularity state causes the improvement.</p><div className="table-wrap improvement-matrix-wrap"><table className="improvement-matrix correlation-matrix"><caption>Correlation between singularity-state presence and direction-adjusted metric improvement over {baselineLabel}.</caption><thead><tr><th scope="col">Triggered state</th>{data.metrics.map((metric) => <th scope="col" key={metric.id}><span>{metric.id}</span><small>{metric.label}</small><em>point-biserial r</em></th>)}</tr></thead><tbody>{correlationStates.map((state) => <tr key={state.id}><th scope="row">{state.label}<small>{classifiedVideos.filter((record) => matchesSingularityFilter(record, state.id)).length} of {classifiedVideos.length} videos</small></th>{data.metrics.map((metric) => { const result = resultFor(state.id, metric); const value = result.correlation; const neutral = value !== null && Math.abs(value) < 0.005; const tone = value === null ? "missing" : neutral ? "neutral" : value > 0 ? "positive" : "negative"; const detail = value === null ? "Unavailable: both state groups and varying improvements are required" : `r = ${formatValue(value, 3)}; triggered mean improvement = ${formatValue(result.triggeredMean ?? Number.NaN, 4)}; other videos = ${formatValue(result.notTriggeredMean ?? Number.NaN, 4)}; n = ${result.triggeredN}/${result.n}`; return <td key={metric.id} className={`improvement-cell ${tone}`} style={value === null || neutral ? undefined : { "--improvement-strength": Math.abs(value) } as CSSProperties} title={detail}>{value === null ? "—" : `${value > 0 ? "+" : ""}${formatValue(value, 2)}`}</td>; })}</tr>)}</tbody></table></div><div className="improvement-legend" aria-label="Correlation legend"><span><i className="improvement-swatch positive" />Associated with more improvement</span><span><i className="improvement-swatch negative" />Associated with less improvement</span><span><i className="improvement-swatch neutral" />Little or no linear association</span></div><p className="figure-note">Hover a cell for the coefficient, group means, and paired sample size. States overlap: a dual-singularity video can also be counted in the elbow-roll and shoulder-roll rows.</p></article></section>;
}

function ComparisonTable({ data }: { data: AnalysisData }) {
  const methodMap = Object.fromEntries(data.methods.map((method) => [method.id, method]));
  return <div className="table-wrap"><table><caption>Direction-adjusted comparisons. Positive differences and effect sizes favour the constrained solution.</caption><thead><tr><th>Metric</th><th>Baseline</th><th>Mean delta [95% CI]</th><th>Rank-biserial</th><th>Win / tie / loss</th><th>Holm p</th></tr></thead><tbody>{data.pairedComparisons.map((row) => <tr key={`${row.metric}-${row.baseline}`}><td><strong>{row.metric}</strong></td><td>{methodMap[row.baseline].shortLabel}</td><td>{formatValue(row.meanDifference, 4)} [{formatValue(row.ciLow, 4)}, {formatValue(row.ciHigh, 4)}]</td><td>{formatValue(row.rankBiserial, 3)}</td><td>{row.wins} / {row.ties} / {row.losses}</td><td>{row.pAdjusted < 0.001 ? "< 0.001" : formatValue(row.pAdjusted, 3)}</td></tr>)}</tbody></table></div>;
}

export default function Dashboard() {
  const [sourceData, setData] = useState<AnalysisData | null>(null);
  const [error, setError] = useState("");
  const [palette, setPalette] = useState("mixed");
  const [useOursLabels, setUseOursLabels] = useState(false);
  const [qualityOptions, setQualityOptions] = useState<FigureDisplayOptions>({ ...defaultFigureOptions });
  const [singularityQualityOptions, setSingularityQualityOptions] = useState<FigureDisplayOptions>({ ...defaultFigureOptions });
  const [singularityFilter, setSingularityFilter] = useState<SingularityFilter>("all");
  const [distributionQualityOptions, setDistributionQualityOptions] = useState<FigureDisplayOptions>({ ...defaultFigureOptions, showInstances: true });
  const [distributionFilter, setDistributionFilter] = useState<SingularityFilter>("all");
  const [effectOptions, setEffectOptions] = useState<FigureDisplayOptions>({ ...defaultFigureOptions, methods: methodOrder.slice(1), showInstances: false });
  const [categoryOptions, setCategoryOptions] = useState<FigureDisplayOptions>({ ...defaultFigureOptions });
  const [performanceOptions, setPerformanceOptions] = useState<FigureDisplayOptions>({ ...defaultFigureOptions });
  const [trajectoryOptions, setTrajectoryOptions] = useState<FigureDisplayOptions>({ ...defaultFigureOptions });
  const [effectMetric, setEffectMetric] = useState<MetricId>("TSE");
  const [categoryMetric, setCategoryMetric] = useState<MetricId>("TSE");
  const [performanceMetric, setPerformanceMetric] = useState("latency_ms");
  const [trajectoryCategory, setTrajectoryCategory] = useState("");
  const [trajectoryVideo, setTrajectoryVideo] = useState("");
  const [trajectorySide, setTrajectorySide] = useState<"left" | "right">("right");
  const [trajectoryMetric, setTrajectoryMetric] = useState<MetricId>("TSE");
  const [trajectoryStartFrame, setTrajectoryStartFrame] = useState(0);
  const [trajectoryEndFrame, setTrajectoryEndFrame] = useState(119);
  const [performanceVideo, setPerformanceVideo] = useState("");
  const [performanceStartFrame, setPerformanceStartFrame] = useState(0);
  const [performanceEndFrame, setPerformanceEndFrame] = useState(71);
  const [performanceTimelineMetric, setPerformanceTimelineMetric] = useState<"latency_ms" | "cpu_percent" | "gpu_percent" | "memory_percent" | "memory_mb">("latency_ms");

  useEffect(() => {
    fetch("/data/analysis.json").then((response) => {
      if (!response.ok) throw new Error(`Analysis data request failed (${response.status})`);
      return response.json();
    }).then((payload: AnalysisData) => { setData(payload); setTrajectoryCategory(payload.categories[0]?.id ?? ""); setPerformanceVideo(payload.performanceFrames[0]?.videoId ?? ""); }).catch((reason) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, []);

  const availableVideos = useMemo(() => !sourceData || !trajectoryCategory ? [] : Array.from(new Set(sourceData.trajectoryRows.filter((row) => row.category === trajectoryCategory).map((row) => row.videoId))).sort(), [sourceData, trajectoryCategory]);
  const activeTrajectoryVideo = availableVideos.includes(trajectoryVideo) ? trajectoryVideo : (availableVideos[0] ?? "");
  const trajectoryFrames = sourceData ? sourceData.trajectoryRows.filter((row) => row.videoId === activeTrajectoryVideo).map((row) => row.frameId) : [];
  const trajectoryMin = Math.min(...trajectoryFrames, 0);
  const trajectoryMax = Math.max(...trajectoryFrames, 119);
  const trajectoryStart = Math.min(Math.max(trajectoryStartFrame, trajectoryMin), trajectoryMax);
  const trajectoryEnd = Math.max(trajectoryStart, Math.min(trajectoryEndFrame, trajectoryMax));
  const performanceVideos = sourceData ? Array.from(new Set(sourceData.performanceFrames.map((row) => row.videoId))).sort() : [];
  const activePerformanceVideo = performanceVideos.includes(performanceVideo) ? performanceVideo : (performanceVideos[0] ?? "");
  const performanceFrames = sourceData ? sourceData.performanceFrames.filter((row) => row.videoId === activePerformanceVideo).map((row) => row.frameId) : [];
  const performanceMin = Math.min(...performanceFrames, 0);
  const performanceMax = Math.max(...performanceFrames, 71);
  const performanceStart = Math.min(Math.max(performanceStartFrame, performanceMin), performanceMax);
  const performanceEnd = Math.max(performanceStart, Math.min(performanceEndFrame, performanceMax));
  const singularityFilterCounts = useMemo(() => {
    const records = sourceData?.videoSingularities ?? [];
    return Object.fromEntries(
      (Object.keys(singularityFilterLabels) as SingularityFilter[]).map((filter) => [
        filter,
        records.filter((record) => matchesSingularityFilter(record, filter)).length,
      ]),
    ) as Record<SingularityFilter, number>;
  }, [sourceData]);
  const selectedSingularityVideos = useMemo(
    () => (sourceData?.videoSingularities ?? []).filter((record) => matchesSingularityFilter(record, singularityFilter)),
    [sourceData, singularityFilter],
  );
  const selectedSingularityVideoKeys = useMemo(
    () => new Set(selectedSingularityVideos.map((record) => videoKey(record.category, record.videoId))),
    [selectedSingularityVideos],
  );
  const singularityQualityRows = useMemo(
    () => (sourceData?.qualityRows ?? []).filter((row) => selectedSingularityVideoKeys.has(videoKey(row.category, row.videoId))),
    [sourceData, selectedSingularityVideoKeys],
  );
  const selectedDistributionVideos = useMemo(
    () => (sourceData?.videoSingularities ?? []).filter((record) => matchesSingularityFilter(record, distributionFilter)),
    [sourceData, distributionFilter],
  );
  const selectedDistributionVideoKeys = useMemo(
    () => new Set(selectedDistributionVideos.map((record) => videoKey(record.category, record.videoId))),
    [selectedDistributionVideos],
  );
  const distributionQualityRows = useMemo(
    () => (sourceData?.qualityRows ?? []).filter((row) => selectedDistributionVideoKeys.has(videoKey(row.category, row.videoId))),
    [sourceData, selectedDistributionVideoKeys],
  );
  const displayData = useMemo(() => sourceData ? ({ ...sourceData, metrics: [...sourceData.metrics].sort((left, right) => metricDisplayOrder.indexOf(left.id) - metricDisplayOrder.indexOf(right.id)), methods: sourceData.methods.map((method, index) => {
    const shortLabel = method.id === "current_solution_constrained"
      ? (useOursLabels ? "Ours (constrained)" : "Constrained")
      : method.id === "current_solution_raw"
        ? (useOursLabels ? "Ours (unconstrained)" : "Unconstrained")
        : method.id === "original_solution"
          ? "Baseline"
          : method.shortLabel;
    return { ...method, label: method.id === "original_solution" ? "Baseline" : method.label, shortLabel, color: academicPalettes[palette].colors[index] ?? method.color };
  }) }) : null, [sourceData, palette, useOursLabels]);

  if (error) return <main className="status-page"><h1>Analysis data could not be loaded</h1><p>{error}</p><p>Run the refresh command in the README, then reload this page.</p></main>;
  if (!sourceData || !displayData) return <main className="status-page" aria-live="polite"><span className="loading-mark" /><h1>Preparing the evaluation atlas</h1><p>Loading paired video-level results and scientific figures...</p></main>;

  const data = displayData;

  const identicalText = data.identicalSeries.map((item) => `${data.methods.find((method) => method.id === item.left)?.shortLabel} and ${data.methods.find((method) => method.id === item.right)?.shortLabel}: ${item.metrics.join(", ")}`).join("; ");
  const performanceFigures = [
    ["latency_ms", "Latency", "ms"], ["latency_jitter_ms", "Latency jitter", "ms"], ["cpu_percent", "CPU use", "%"], ["gpu_percent", "GPU use", "%"], ["memory_mb", "Memory", "MiB"],
  ] as const;
  const selectedPerformance = performanceFigures.find(([metric]) => metric === performanceMetric) ?? performanceFigures[0];
  const [performanceKey, performanceLabel, performanceUnit] = selectedPerformance;
  const performanceRows = data.performanceVideos.filter((row) => Number.isFinite(row[performanceKey])).map((row) => ({ method: row.method, value: row[performanceKey] as number, videoId: row.videoId }));
  const selectedEffectMetric = data.metrics.find((metric) => metric.id === effectMetric);
  const singularityByVideo = new Map(
    (data.videoSingularities ?? []).map((record) => [videoKey(record.category, record.videoId), record]),
  );

  return <main>
    <header className="hero"><nav aria-label="Dashboard sections"><a className="wordmark" href="#top">IK / EVALUATION</a><div className="nav-links"><a href="#quality">Quality</a><a href="#singularity-quality">Singularities</a><a href="#singularity-distribution">Distributions</a><a href="#effects">Effects</a><a href="#categories">Categories</a><a href="#trajectory">Trajectories</a><a href="#performance">Performance</a><a href="#methods">Methods</a></div></nav><div className="hero-content" id="top"><p className="eyebrow">Pepper robot imitation - Chapter 4 evidence companion</p><h1>Constrained inverse kinematics, examined beyond averages.</h1><p className="hero-lede">A paired, video-level comparison of motion fidelity, human-likeness, smoothness, feasibility, and computation across the proposed method, its ablation, IKPy, and the original solution.</p><div className="study-strip" aria-label="Study design summary"><div><strong>{data.metadata.videoCount}</strong><span>paired videos</span></div><div><strong>{data.metadata.categoryCount}</strong><span>motion categories</span></div><div><strong>{data.metadata.frameCount}</strong><span>quality frames</span></div><div><strong>25</strong><span>inferential units</span></div></div><div style={{ display: "flex", alignItems: "flex-end", flexWrap: "wrap", gap: "14px 22px", marginTop: 30 }}><label className="select-field palette-picker"><span>Bar colour palette</span><select value={palette} onChange={(event) => setPalette(event.target.value)}>{Object.entries(academicPalettes).map(([key, option]) => <option key={key} value={key}>{option.label}</option>)}</select></label><label className="toggle-control" style={{ minHeight: 42, padding: "8px 12px", border: "1px solid rgb(255 255 255 / 0.3)", color: "#dfe7e4" }}><input type="checkbox" checked={useOursLabels} onChange={(event) => setUseOursLabels(event.target.checked)} /> Use “Ours” method labels</label></div><MethodLegend methods={displayData.methods} /></div></header>

    <section className="section intro-section"><div className="section-number">00</div><div><p className="section-kicker">Reading guide</p><h2>Evidence first, interaction second.</h2></div><div className="reading-note"><p>Each point in the primary analysis represents one complete video. Left and right arm summaries are averaged where appropriate; HJL is counted once because it is a complete two-arm frame indicator.</p><p>Frame curves remain available for diagnosis, but they are never treated as independent samples in the inferential results.</p></div></section>

    <section className="section" id="quality"><div className="section-heading"><div><p className="section-kicker">01 - Overall quality</p><h2>Seven complementary views of imitation quality.</h2></div><a className="data-link" href="/data/video_quality.csv" download>Download video-level CSV</a></div><p className="section-copy">Native scales are preserved. EEAh is intentionally excluded: it is a human reference signal shared by every method, not a method outcome.</p><FigureControls id="quality" data={data} options={qualityOptions} setOptions={setQualityOptions} includeQualityContextControls />{identicalText && <aside className="overlap-note"><strong>Exact overlap detected.</strong> {identicalText}. Separate method bars keep coincident series visible.</aside>}<div className="figure-grid">{data.metrics.map((metric) => <QualityFigure key={metric.id} metric={metric} rows={data.qualityRows} data={data} options={qualityOptions} sampleSize={data.metadata.videoCount} />)}</div></section>

    <section className="section section-tinted" id="singularity-quality"><div className="section-heading"><div><p className="section-kicker">01B - Quality by singularity classification</p><h2>Compare complete videos by the singularities they contain.</h2></div><button className="data-link" type="button" disabled={!singularityQualityRows.length} onClick={() => downloadRowsCsv(`singularity-quality-${singularityFilter}.csv`, singularityQualityRows.map((row) => { const classification = singularityByVideo.get(videoKey(row.category, row.videoId)); return { videoId: row.videoId, category: row.category, method: row.method, metric: row.metric, value: row.value, classificationAvailable: classification?.classificationAvailable, hasAny: classification?.hasAny, hasDual: classification?.hasDual, hasElbowRoll: classification?.hasElbowRoll, hasShoulderRoll: classification?.hasShoulderRoll }; }))}>Download selected CSV</button></div><p className="section-copy">The selector changes the complete-video population used by every chart. Metrics remain full-video summaries for both arms; frames are not removed or recomputed.</p><div className="singularity-toolbar"><label className="select-field"><span>Video classification</span><select value={singularityFilter} onChange={(event) => setSingularityFilter(event.target.value as SingularityFilter)}>{(Object.keys(singularityFilterLabels) as SingularityFilter[]).map((filter) => <option key={filter} value={filter}>{singularityFilterLabels[filter]} ({singularityFilterCounts[filter]})</option>)}</select></label><div className="selection-summary" aria-live="polite"><strong>{selectedSingularityVideos.length}</strong><span>{selectedSingularityVideos.length === 1 ? "video selected" : "videos selected"}</span></div></div>{selectedSingularityVideos.length ? <><details className="singularity-video-detail"><summary>Show classified videos and singular-frame counts</summary><SingularityVideoTable records={selectedSingularityVideos} /></details><FigureControls id="singularity-quality" data={data} options={singularityQualityOptions} setOptions={setSingularityQualityOptions} includeQualityContextControls /><div className="figure-grid">{data.metrics.map((metric) => <QualityFigure key={metric.id} metric={metric} rows={singularityQualityRows} data={data} options={singularityQualityOptions} sampleSize={selectedSingularityVideos.length} scopeId="singularity-quality" filenamePrefix={`singularity-${singularityFilter}`} populationLabel={singularityFilterLabels[singularityFilter]} />)}</div></> : <div className="empty-selection" role="status"><strong>No videos match this classification.</strong><p>Choose another singularity group or refresh the analysis after new evaluation results are available.</p></div>}</section>

    <section className="section distribution-quality-section" id="singularity-distribution"><div className="section-heading"><div><p className="section-kicker">01C - Quality distributions by singularity classification</p><h2>See every selected video behind each method mean.</h2></div><button className="data-link" type="button" disabled={!distributionQualityRows.length} onClick={() => downloadRowsCsv(`singularity-distribution-${distributionFilter}.csv`, distributionQualityRows.map((row) => { const classification = singularityByVideo.get(videoKey(row.category, row.videoId)); return { videoId: row.videoId, category: row.category, method: row.method, metric: row.metric, value: row.value, classificationAvailable: classification?.classificationAvailable, hasAny: classification?.hasAny, hasDual: classification?.hasDual, hasElbowRoll: classification?.hasElbowRoll, hasShoulderRoll: classification?.hasShoulderRoll }; }))}>Download selected CSV</button></div><p className="section-copy">This is the same complete-video population and seven quality metrics as 01B. Each translucent diamond is one video; the outlined diamond is the selected method mean.</p><div className="singularity-toolbar"><label className="select-field"><span>Video classification</span><select value={distributionFilter} onChange={(event) => setDistributionFilter(event.target.value as SingularityFilter)}>{(Object.keys(singularityFilterLabels) as SingularityFilter[]).map((filter) => <option key={filter} value={filter}>{singularityFilterLabels[filter]} ({singularityFilterCounts[filter]})</option>)}</select></label><div className="selection-summary" aria-live="polite"><strong>{selectedDistributionVideos.length}</strong><span>{selectedDistributionVideos.length === 1 ? "video selected" : "videos selected"}</span></div></div>{selectedDistributionVideos.length ? <><details className="singularity-video-detail"><summary>Show classified videos and singular-frame counts</summary><SingularityVideoTable records={selectedDistributionVideos} /></details><FigureControls id="singularity-distribution" data={data} options={distributionQualityOptions} setOptions={setDistributionQualityOptions} includeQualityContextControls /><div className="figure-grid">{data.metrics.map((metric) => <DistributionQualityFigure key={metric.id} metric={metric} rows={distributionQualityRows} data={data} options={distributionQualityOptions} sampleSize={selectedDistributionVideos.length} scopeId="singularity-distribution" filenamePrefix={`distribution-${distributionFilter}`} populationLabel={singularityFilterLabels[distributionFilter]} />)}</div></> : <div className="empty-selection" role="status"><strong>No videos match this classification.</strong><p>Choose another singularity group or refresh the analysis after new evaluation results are available.</p></div>}</section>

    <section className="section section-tinted" id="effects"><div className="section-heading"><div><p className="section-kicker">02 - Paired effects</p><h2>Does the constrained solution improve this metric?</h2></div><a className="data-link" href="/data/paired_comparisons.csv" download>Download inference CSV</a></div><div className="effect-explainer"><p><strong>How to read this figure.</strong> For each of the same 25 videos, the constrained result is compared with a selected baseline. Scores are direction-adjusted first, so positive always means the constrained solution performed better.</p><p>The bar is the mean video-level improvement. Its whisker is a 95% paired bootstrap confidence interval; if it crosses zero, the observed improvement is uncertain at that level.</p></div><div className="section-toolbar"><label className="select-field"><span>Metric</span><select value={effectMetric} onChange={(event) => setEffectMetric(event.target.value as MetricId)}>{data.metrics.map((metric) => <option key={metric.id} value={metric.id}>{metric.id} - {metric.label}</option>)}</select></label></div><FigureControls id="effects" data={data} options={effectOptions} setOptions={setEffectOptions} availableMethods={methodOrder.slice(1)} includeInstanceControl={false} includeMinMaxControl={false} /><article className="wide-figure"><FigureHeading code={effectMetric} title={`Constrained improvement in ${selectedEffectMetric?.label ?? effectMetric}`} note="Positive favours constrained - paired 95% bootstrap CI" svgId="paired-effect" filename={`paired-effect-${effectMetric.toLowerCase()}`} /><VerticalEffectBarPlot methods={data.methods} comparisons={data.pairedComparisons.filter((row: PairedComparison) => row.metric === effectMetric)} selectedBaselines={effectOptions.methods} showValues={effectOptions.showValues} showLegend={effectOptions.showLegend} svgId="paired-effect" ariaLabel={`Paired constrained improvement for ${selectedEffectMetric?.label ?? effectMetric}`} /><p className="figure-note">The constrained method is the fixed reference. Choose one or more baselines above; numerical bar values can be hidden before export.</p></article><h3 className="table-title">Robustness and inferential summary</h3><ComparisonTable data={data} /></section>

    <section className="section" id="categories"><div className="section-heading"><div><p className="section-kicker">03 - Which motion categories are hardest for each method?</p><h2>Use the same metric to see where patterns differ.</h2></div><label className="select-field"><span>Metric</span><select value={categoryMetric} onChange={(event) => setCategoryMetric(event.target.value as MetricId)}>{data.metrics.map((metric) => <option key={metric.id} value={metric.id}>{metric.id} - {metric.label}</option>)}</select></label></div><p className="section-copy">Each category contains five clips. A bar is the mean across those clips: use this view to reveal where patterns differ, not to make category-level significance claims.</p><FigureControls id="categories" data={data} options={categoryOptions} setOptions={setCategoryOptions} /><article className="wide-figure"><FigureHeading code={categoryMetric} title={`${data.metrics.find((metric) => metric.id === categoryMetric)?.label} by motion category`} note="Five videos per category - descriptive view only" svgId="category-comparison" filename={`category-${categoryMetric.toLowerCase()}`} /><VerticalCategoryBarPlot data={data} metric={categoryMetric} options={categoryOptions} svgId="category-comparison" /><p className="figure-note">Bars are category means{categoryOptions.showInstances ? "; circles are the five underlying videos." : "."}</p></article></section>

    <section className="section section-dark" id="trajectory"><div className="section-heading"><div><p className="section-kicker">04 - Frame explorer</p><h2>Inspect the temporal shape behind a sequence score.</h2></div></div><div className="control-row" aria-label="Trajectory controls"><label className="select-field"><span>Category</span><select value={trajectoryCategory} onChange={(event) => setTrajectoryCategory(event.target.value)}>{data.categories.map((category) => <option key={category.id} value={category.id}>{category.label}</option>)}</select></label><label className="select-field"><span>Video</span><select value={activeTrajectoryVideo} onChange={(event) => setTrajectoryVideo(event.target.value)}>{availableVideos.map((video) => <option key={video} value={video}>{video}</option>)}</select></label><label className="select-field"><span>Arm</span><select value={trajectorySide} onChange={(event) => setTrajectorySide(event.target.value as "left" | "right")}><option value="left">Left</option><option value="right">Right</option></select></label><label className="select-field"><span>Metric</span><select value={trajectoryMetric} onChange={(event) => setTrajectoryMetric(event.target.value as MetricId)}>{data.metrics.map((metric) => <option key={metric.id} value={metric.id}>{metric.id}</option>)}</select></label><label className="select-field"><span>Start frame</span><input type="number" min={trajectoryMin} max={trajectoryEnd} value={trajectoryStart} onChange={(event) => setTrajectoryStartFrame(Number(event.target.value))} /></label><label className="select-field"><span>End frame</span><input type="number" min={trajectoryStart} max={trajectoryMax} value={trajectoryEnd} onChange={(event) => setTrajectoryEndFrame(Number(event.target.value))} /></label></div><FigureControls id="trajectory" data={data} options={trajectoryOptions} setOptions={setTrajectoryOptions} includeValueControl={false} includeInstanceControl={false} includeMinMaxControl={false} />{activeTrajectoryVideo && <article className="wide-figure"><FigureHeading code={trajectoryMetric} title={`${activeTrajectoryVideo.replace("_", " ")} - ${trajectorySide} arm`} note={`Frame-synchronised trajectories, frames ${trajectoryStart}-${trajectoryEnd}`} svgId="trajectory-figure" filename={`trajectory-${activeTrajectoryVideo}-${trajectorySide}-${trajectoryMetric.toLowerCase()}`} /><TrajectoryPlot data={data} category={trajectoryCategory} videoId={activeTrajectoryVideo} side={trajectorySide} metric={trajectoryMetric} methods={trajectoryOptions.methods} showLegend={trajectoryOptions.showLegend} startFrame={trajectoryStart} endFrame={trajectoryEnd} svgId="trajectory-figure" /><p className="figure-note">{trajectoryMetric === "TSE" ? "Frames 0-1 are intentionally undefined because a second-order difference requires three frames." : trajectoryMetric === "HJL" ? "HJL is a two-arm frame indicator; the arm selector does not change this series." : "All values remain on the metric's native scale."}</p></article>}</section>

    <section className="section" id="performance"><div className="section-heading"><div><p className="section-kicker">05 - Computational performance</p><h2>Responsiveness, stability, and resource use.</h2></div><a className="data-link" href="/data/performance_repeats.csv" download>Download repeat-level CSV</a></div><aside className="benchmark-note"><strong>Offline benchmark.</strong> Measurements cover IK/method computation only. Capture, MeTRAbs inference, networking, and Pepper actuation are outside the timing boundary.</aside><div className="section-toolbar"><label className="select-field"><span>Performance metric</span><select value={performanceMetric} onChange={(event) => setPerformanceMetric(event.target.value)}>{performanceFigures.map(([metric, label, unit]) => <option key={metric} value={metric}>{label} ({unit})</option>)}</select></label></div><FigureControls id="performance" data={data} options={performanceOptions} setOptions={setPerformanceOptions} /><article className="wide-figure"><FigureHeading title={performanceLabel} note={`Lower is better - ${performanceUnit} - n = 25 videos`} svgId="performance-primary" filename={`performance-${performanceKey}`} /><VerticalMeanBarPlot rows={performanceRows} methods={data.methods} options={performanceOptions} svgId="performance-primary" ariaLabel={`${performanceLabel} across 25 videos`} unit={performanceUnit} /><p className="figure-note">Bars are per-video means{performanceOptions.showInstances ? "; circles are individual video summaries." : "."}{performanceOptions.showMinMax ? " Whiskers show the minimum and maximum video summary." : ""}</p></article><div className="control-row" aria-label="Performance evolution controls"><label className="select-field"><span>Video</span><select value={activePerformanceVideo} onChange={(event) => setPerformanceVideo(event.target.value)}>{performanceVideos.map((video) => <option key={video} value={video}>{video}</option>)}</select></label><label className="select-field"><span>Frame metric</span><select value={performanceTimelineMetric} onChange={(event) => setPerformanceTimelineMetric(event.target.value as typeof performanceTimelineMetric)}><option value="latency_ms">Latency (ms)</option><option value="cpu_percent">CPU use (%)</option><option value="gpu_percent">GPU use (%)</option><option value="memory_percent">Memory (%)</option><option value="memory_mb">Memory (MiB)</option></select></label><label className="select-field"><span>Start frame</span><input type="number" min={performanceMin} max={performanceEnd} value={performanceStart} onChange={(event) => setPerformanceStartFrame(Number(event.target.value))} /></label><label className="select-field"><span>End frame</span><input type="number" min={performanceStart} max={performanceMax} value={performanceEnd} onChange={(event) => setPerformanceEndFrame(Number(event.target.value))} /></label></div><article className="wide-figure"><FigureHeading title="Performance evolution by frame" note="Each point is the mean of three repeats for the selected video" svgId="performance-timeline" filename={`performance-timeline-${activePerformanceVideo}-${performanceTimelineMetric}`} /><PerformanceTimelinePlot data={data} videoId={activePerformanceVideo} metric={performanceTimelineMetric} methods={performanceOptions.methods} showLegend={performanceOptions.showLegend} startFrame={performanceStart} endFrame={performanceEnd} svgId="performance-timeline" unit={performanceTimelineMetric === "latency_ms" ? "ms" : performanceTimelineMetric === "memory_mb" ? "MiB" : "%"} /></article><details className="latency-detail"><summary>Latency distribution detail</summary><p>The ECDF uses repeat-level latency observations and a logarithmic time axis. It is retained for deeper inspection rather than primary thesis figures.</p><div className="detail-heading"><ExportButtons svgId="latency-ecdf" filename="performance-latency-ecdf" /></div><LatencyEcdfPlot data={data} methods={performanceOptions.methods} showLegend={performanceOptions.showLegend} svgId="latency-ecdf" /></details></section>

    <section className="section section-tinted" id="methods"><div className="section-heading"><div><p className="section-kicker">06 - Methods & data</p><h2>A transparent path from source JSON to thesis figure.</h2></div></div><div className="methods-layout"><div><h3>Metric dictionary</h3><div className="metric-dictionary">{data.metrics.map((metric) => <article key={metric.id}><span className="metric-code">{metric.id}</span><div><strong>{metric.label}</strong><p>{metric.definition}</p></div><small>{metric.direction === "lower" ? "lower is better" : "higher is better"}</small></article>)}</div></div><aside className="methodology-card"><h3>Analysis protocol</h3><dl><div><dt>Primary unit</dt><dd>Video (n = 25)</dd></div><div><dt>Bootstrap</dt><dd>{data.metadata.bootstrapSamples.toLocaleString()} paired resamples</dd></div><div><dt>Permutation</dt><dd>{data.metadata.permutationSamples.toLocaleString()} sign flips</dd></div><div><dt>Multiplicity</dt><dd>Holm correction, 21 tests</dd></div><div><dt>Random seed</dt><dd>{data.metadata.seed}</dd></div></dl><h4>Interpretive limits</h4><ul>{data.metadata.caveats.map((caveat) => <li key={caveat}>{caveat}</li>)}</ul></aside></div><div className="download-band"><div><strong>Reproducible data package</strong><span>Video summaries, corrected comparisons, and repeat-level performance.</span></div><div><a href="/data/video_quality.csv" download>Quality CSV</a><a href="/data/paired_comparisons.csv" download>Effects CSV</a><a href="/data/performance_repeats.csv" download>Performance CSV</a><button type="button" onClick={() => downloadRowsCsv(`active-${categoryMetric.toLowerCase()}-analysis.csv`, data.qualityRows.filter((row) => row.metric === categoryMetric && categoryOptions.methods.includes(row.method)).map((row) => ({ video: row.videoId, category: row.category, method: row.method, metric: row.metric, value: row.value })))}>Active metric CSV</button></div></div></section>
    <footer><span>Scientific IK Evaluation Dashboard</span><span>Generated from authoritative repository results - {new Date(data.metadata.generatedAt).toLocaleDateString()}</span></footer>
  </main>;
}
