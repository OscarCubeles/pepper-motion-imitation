"use client";

import { useState } from "react";
import type {
  AnalysisData,
  MethodId,
  MethodRecord,
  MetricId,
  PairedComparison,
  PerformanceRow,
  QualityRow,
} from "./types";

const WIDTH = 860;
const METHODS: MethodId[] = [
  "current_solution_constrained",
  "current_solution_raw",
  "ikpy",
  "original_solution",
];

export type FigureDisplayOptions = {
  methods: MethodId[];
  showValues: boolean;
  showInstances: boolean;
  showMinMax: boolean;
  showMinMaxValues: boolean;
  showLegend: boolean;
  showDirection: boolean;
  showSampleSize: boolean;
  showCategoryNote: boolean;
};

const finite = (values: Array<number | null | undefined>): number[] => values.filter((value): value is number => typeof value === "number" && Number.isFinite(value));

export type MetricAxisBounds = readonly [minimum: number, maximum?: number];

export function qualityMetricAxisBounds(metric: MetricId): MetricAxisBounds {
  if (metric === "EEAr" || metric === "SOAx") return [0, 2];
  if (metric === "HJL" || metric === "WOM" || metric === "HJAr") return [0, 1];
  return [0];
}

function niceUpperBound(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 1;
  const roughStep = value / 4;
  const magnitude = 10 ** Math.floor(Math.log10(roughStep));
  const normalized = roughStep / magnitude;
  const niceNormalized = [1, 2, 2.5, 5, 10].find((candidate) => candidate >= normalized) ?? 10;
  const step = niceNormalized * magnitude;
  return Math.ceil(value / step) * step;
}

function resolveAxisDomain(values: number[], bounds?: MetricAxisBounds): readonly [number, number] {
  const rawMinimum = values.length ? Math.min(...values) : 0;
  const rawMaximum = values.length ? Math.max(...values) : 1;
  if (bounds) {
    const minimum = bounds[0];
    const maximum = bounds[1] ?? niceUpperBound(Math.max(rawMaximum, minimum));
    return [minimum, maximum > minimum ? maximum : minimum + 1];
  }
  const spread = rawMaximum - rawMinimum;
  const padding = spread > 0 ? spread * 0.12 : Math.max(Math.abs(rawMaximum) * 0.12, 1);
  return [rawMinimum - padding, rawMaximum + padding];
}

export function formatValue(value: number, unit: string | number = ""): string {
  if (!Number.isFinite(value)) return "NA";
  const magnitude = Math.abs(value);
  const digits = typeof unit === "number" ? unit : magnitude >= 100 ? 0 : magnitude >= 10 ? 1 : 2;
  const formatted = value.toFixed(digits);
  const displayNumber = Number(formatted) === 0 ? (0).toFixed(digits) : formatted;
  return `${displayNumber}${typeof unit === "string" && unit ? ` ${unit}` : ""}`;
}

function mean(values: number[]): number {
  const valid = finite(values);
  return valid.length ? valid.reduce((total, value) => total + value, 0) / valid.length : NaN;
}

function tickValues(minimum: number, maximum: number, count = 5): number[] {
  if (minimum === maximum) return [minimum];
  return Array.from({ length: count }, (_, index) => minimum + ((maximum - minimum) * index) / (count - 1));
}

function logTickValues(minimum: number, maximum: number, count = 5): number[] {
  if (minimum <= 0 || maximum <= 0 || minimum === maximum) return [Math.max(minimum, 0.000001)];
  const logMinimum = Math.log10(minimum);
  const logMaximum = Math.log10(maximum);
  return Array.from({ length: count }, (_, index) => 10 ** (logMinimum + ((logMaximum - logMinimum) * index) / (count - 1)));
}

function selectedMethodRecords(methods: MethodRecord[], selected: MethodId[]): MethodRecord[] {
  return METHODS.filter((id) => selected.includes(id))
    .map((id) => methods.find((method) => method.id === id))
    .filter((method): method is MethodRecord => Boolean(method));
}

function SvgLegend({ methods, x = 18, y = 18, columns = 2, columnGap = 190 }: { methods: MethodRecord[]; x?: number; y?: number; columns?: number; columnGap?: number }) {
  return (
    <g aria-label="Method legend" transform={`translate(${x} ${y})`}>
      {methods.map((method, index) => {
        const column = index % columns;
        const row = Math.floor(index / columns);
        return (
          <g key={method.id} transform={`translate(${column * columnGap} ${row * 18})`}>
            <rect width="11" height="11" rx="2" fill={method.color} />
            <text x="16" y="9" className="chart-legend-text">
              {method.shortLabel}
            </text>
          </g>
        );
      })}
    </g>
  );
}

export function downloadRowsCsv(filename: string, rows: Array<Record<string, unknown>>) {
  if (!rows.length) return;
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))];
  const escape = (value: unknown) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  const csv = [columns.join(","), ...rows.map((row) => columns.map((column) => escape(row[column])).join(","))].join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function downloadSvg(svgId: string, filename: string) {
  const svg = document.getElementById(svgId);
  if (!(svg instanceof SVGSVGElement)) return;
  const source = new XMLSerializer().serializeToString(svg);
  const url = URL.createObjectURL(new Blob([source], { type: "image/svg+xml;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `${filename}.svg`;
  link.click();
  URL.revokeObjectURL(url);
}

async function downloadPng(svgId: string, filename: string) {
  const svg = document.getElementById(svgId);
  if (!(svg instanceof SVGSVGElement)) return;
  const source = new XMLSerializer().serializeToString(svg);
  const image = new Image();
  const sourceUrl = URL.createObjectURL(new Blob([source], { type: "image/svg+xml;charset=utf-8" }));
  image.onload = () => {
    const canvas = document.createElement("canvas");
    const scale = 3;
    canvas.width = Number(svg.getAttribute("width") || WIDTH) * scale;
    canvas.height = Number(svg.getAttribute("height") || 420) * scale;
    const context = canvas.getContext("2d");
    context?.drawImage(image, 0, 0, canvas.width, canvas.height);
    URL.revokeObjectURL(sourceUrl);
    const link = document.createElement("a");
    link.href = canvas.toDataURL("image/png");
    link.download = `${filename}.png`;
    link.click();
  };
  image.src = sourceUrl;
}

export function ExportButtons({ svgId, filename }: { svgId: string; filename: string }) {
  return null;
}

export function MethodLegend({ methods }: { methods: MethodRecord[] }) {
  return (
    <div className="method-legend" aria-label="Method colour legend">
      {methods.map((method) => (
        <span key={method.id}>
          <i style={{ background: method.color }} aria-hidden="true" />
          <b>{method.shortLabel}</b>
          <small>{method.role === "ablation" ? "ablation" : method.label}</small>
        </span>
      ))}
    </div>
  );
}

type MeanBarProps = {
  rows: Array<Pick<QualityRow, "method" | "value"> & Record<string, unknown>>;
  methods: MethodRecord[];
  options: FigureDisplayOptions;
  svgId: string;
  ariaLabel: string;
  unit?: string;
  scale?: "linear" | "log";
  yDomain?: MetricAxisBounds;
};

export function VerticalMeanBarPlot({ rows, methods, options, svgId, ariaLabel, unit = "", scale = "linear", yDomain }: MeanBarProps) {
  const isPerformancePlot = unit === "ms" || unit === "%" || unit === "MiB";
  const [logScale, setLogScale] = useState(false);
  const activeScale = isPerformancePlot && scale === "linear" && logScale ? "log" : scale;
  const visibleMethods = selectedMethodRecords(methods, options.methods);
  const grouped = new Map<MethodId, number[]>();
  visibleMethods.forEach((method) => grouped.set(method.id, []));
  rows.forEach((row) => {
    if (grouped.has(row.method) && Number.isFinite(row.value)) grouped.get(row.method)?.push(row.value as number);
  });
  const allValues = finite([...grouped.values()].flat());
  const plottedValues = activeScale === "log" ? allValues.filter((value) => value > 0) : allValues;
  const means = visibleMethods.map((method) => mean(grouped.get(method.id) ?? []));
  const plottedMeans = activeScale === "log" ? finite(means).filter((value) => value > 0) : finite(means);
  const maximum = Math.max(0, ...plottedValues, ...plottedMeans);
  const minimum = activeScale === "log" ? Math.max(Math.min(...plottedValues, ...plottedMeans, 1) / 1.18, 0.000001) : (yDomain?.[0] ?? 0);
  const domainMaximum = activeScale === "log"
    ? (maximum > minimum ? maximum * 1.18 : Math.max(minimum * 10, 1))
    : (yDomain?.[1] ?? niceUpperBound(maximum));
  const height = 420;
  const left = 64;
  const right = 24;
  const top = options.showLegend ? 58 : 26;
  const bottom = 74;
  const plotWidth = WIDTH - left - right;
  const plotHeight = height - top - bottom;
  const y = (value: number) => activeScale === "log"
    ? top + plotHeight - ((Math.log10(Math.max(value, minimum)) - Math.log10(minimum)) / (Math.log10(domainMaximum) - Math.log10(minimum) || 1)) * plotHeight
    : top + plotHeight - ((value - minimum) / (domainMaximum - minimum || 1)) * plotHeight;
  const baseline = activeScale === "log" ? top + plotHeight : y(0);
  const slot = plotWidth / Math.max(visibleMethods.length, 1);
  const barWidth = Math.min(92, slot * 0.62);
  const description = `${ariaLabel}. It shows video-level means${options.showInstances ? " and individual video values" : ""}${options.showMinMax ? " with min-max ranges" : ""} for ${visibleMethods.length} selected methods${activeScale === "log" ? " on a logarithmic y-axis" : ""}.`;

  return (
    <>
      {isPerformancePlot && <label className="chart-scale-toggle"><input type="checkbox" checked={logScale} onChange={() => setLogScale((current) => !current)} /> Logarithmic y-axis</label>}
      <svg id={svgId} className="chart-svg" viewBox={`0 0 ${WIDTH} ${height}`} width={WIDTH} height={height} role="img" aria-label={description}>
      <title>{description}</title>
      <rect width={WIDTH} height={height} fill="#ffffff" />
      {options.showLegend && <SvgLegend methods={visibleMethods} />}
      {(activeScale === "log" ? logTickValues(minimum, domainMaximum) : tickValues(minimum, domainMaximum)).map((tick) => (
        <g key={tick}>
          <line x1={left} x2={WIDTH - right} y1={y(tick)} y2={y(tick)} className="chart-grid" />
          <text x={left - 9} y={y(tick) + 4} className="chart-axis-label" textAnchor="end">{formatValue(tick)}</text>
        </g>
      ))}
      <line x1={left} x2={WIDTH - right} y1={baseline} y2={baseline} className="chart-axis" />
      {visibleMethods.map((method, methodIndex) => {
        const values = grouped.get(method.id) ?? [];
        const methodMean = means[methodIndex];
        const rangeMin = values.length > 0 ? Math.min(...values) : undefined;
        const rangeMax = values.length > 0 ? Math.max(...values) : undefined;
        const centre = left + slot * (methodIndex + 0.5);
        const barY = Number.isFinite(methodMean) && (activeScale !== "log" || methodMean > 0) ? y(methodMean) : baseline;
        return (
          <g key={method.id}>
            <rect x={centre - barWidth / 2} y={barY} width={barWidth} height={Math.max(0, baseline - barY)} rx="4" fill={method.color} opacity="0.88">
              <title>{`${method.label}: mean ${formatValue(methodMean, unit)} (${values.length} videos)`}</title>
            </rect>
            {options.showMinMax && rangeMin !== undefined && rangeMax !== undefined && <g className="min-max-range"><line x1={centre} x2={centre} y1={y(rangeMin)} y2={y(rangeMax)} /><line x1={centre - 7} x2={centre + 7} y1={y(rangeMin)} y2={y(rangeMin)} /><line x1={centre - 7} x2={centre + 7} y1={y(rangeMax)} y2={y(rangeMax)} /><title>{`Min-max range: ${formatValue(rangeMin, unit)} to ${formatValue(rangeMax, unit)}`}</title>{options.showMinMaxValues && <><text x={centre} y={Math.max(top + 12, y(rangeMax) - 8)} className="range-value" textAnchor="middle">{formatValue(rangeMax, unit)}</text><text x={centre} y={Math.min(baseline - 4, y(rangeMin) + 15)} className="range-value" textAnchor="middle">{formatValue(rangeMin, unit)}</text></>}</g>}
            {options.showInstances && values.filter((value) => activeScale !== "log" || value > 0).map((value, valueIndex) => {
              const offset = ((valueIndex * 17) % 11) - 5;
              return <circle key={`${value}-${valueIndex}`} cx={centre + offset * Math.min(3, barWidth / 16)} cy={y(value)} r="3.2" fill="#ffffff" stroke={method.color} strokeWidth="1.6"><title>{`${method.label}: ${formatValue(value, unit)}`}</title></circle>;
            })}
            {options.showValues && Number.isFinite(methodMean) && options.showMinMax && rangeMin !== undefined && rangeMax !== undefined ? <text x={centre} y={Math.max(top + 12, y(rangeMax) - 24)} className="bar-value" textAnchor="middle">{formatValue(methodMean, unit)}</text> : options.showValues && Number.isFinite(methodMean) && <text x={centre} y={Math.max(top + 13, barY - 8)} className="bar-value" textAnchor="middle">{formatValue(methodMean, unit)}</text>}
            <text x={centre} y={height - 47} className="chart-method-label" textAnchor="middle">{method.shortLabel}</text>
          </g>
        );
      })}
      <text x={left} y={height - 16} className="chart-axis-title">Video-level mean ({unit || "native units"})</text>
      </svg>
    </>
  );
}

type DistributionProps = {
  rows: Array<Pick<QualityRow, "method" | "value" | "videoId">>;
  methods: MethodRecord[];
  options: FigureDisplayOptions;
  svgId: string;
  ariaLabel: string;
  unit?: string;
  yDomain?: MetricAxisBounds;
};

export function VerticalDistributionPlot({ rows, methods, options, svgId, ariaLabel, unit = "", yDomain }: DistributionProps) {
  const visibleMethods = selectedMethodRecords(methods, options.methods);
  const grouped = new Map<MethodId, number[]>();
  visibleMethods.forEach((method) => grouped.set(method.id, []));
  rows.forEach((row) => {
    if (grouped.has(row.method) && Number.isFinite(row.value)) grouped.get(row.method)?.push(row.value as number);
  });
  const allValues = finite([...grouped.values()].flat());
  const [minimum, maximum] = resolveAxisDomain(allValues, yDomain);
  const height = 440;
  const left = 64;
  const right = 24;
  const top = options.showLegend ? 64 : 30;
  const bottom = 76;
  const plotWidth = WIDTH - left - right;
  const plotHeight = height - top - bottom;
  const y = (value: number) => top + plotHeight - ((value - minimum) / (maximum - minimum || 1)) * plotHeight;
  const slot = plotWidth / Math.max(visibleMethods.length, 1);
  const description = `${ariaLabel}. Each column shows the selected video-level values for one method, with a highlighted mean marker.`;

  return (
    <svg id={svgId} className="chart-svg distribution-chart" viewBox={`0 0 ${WIDTH} ${height}`} width={WIDTH} height={height} role="img" aria-label={description}>
      <title>{description}</title>
      <rect width={WIDTH} height={height} fill="#ffffff" />
      {options.showLegend && <SvgLegend methods={visibleMethods} />}
      {tickValues(minimum, maximum).map((tick) => (
        <g key={tick}>
          <line x1={left} x2={WIDTH - right} y1={y(tick)} y2={y(tick)} className="chart-grid" />
          <text x={left - 9} y={y(tick) + 4} className="chart-axis-label" textAnchor="end">{formatValue(tick, unit)}</text>
        </g>
      ))}
      {visibleMethods.map((method, methodIndex) => {
        const values = grouped.get(method.id) ?? [];
        const methodMean = mean(values);
        const rangeMin = values.length ? Math.min(...values) : undefined;
        const rangeMax = values.length ? Math.max(...values) : undefined;
        const centre = left + slot * (methodIndex + 0.5);
        const meanY = Number.isFinite(methodMean) ? y(methodMean) : undefined;
        return (
          <g key={method.id}>
            {options.showMinMax && rangeMin !== undefined && rangeMax !== undefined && <g className="distribution-range"><line x1={centre} x2={centre} y1={y(rangeMin)} y2={y(rangeMax)} /><line x1={centre - 7} x2={centre + 7} y1={y(rangeMin)} y2={y(rangeMin)} /><line x1={centre - 7} x2={centre + 7} y1={y(rangeMax)} y2={y(rangeMax)} />{options.showMinMaxValues && <><text x={centre + 11} y={y(rangeMax) + 3} className="range-value">{formatValue(rangeMax, unit)}</text><text x={centre + 11} y={y(rangeMin) + 3} className="range-value">{formatValue(rangeMin, unit)}</text></>}</g>}
            {options.showInstances && values.map((value, valueIndex) => {
              const jitter = (((valueIndex * 17) % 13) - 6) * Math.min(2.8, slot / 42);
              return <polygon key={`${value}-${valueIndex}`} points={`${centre + jitter},${y(value) - 6} ${centre + jitter + 6},${y(value)} ${centre + jitter},${y(value) + 6} ${centre + jitter - 6},${y(value)}`} fill={method.color} opacity="0.16"><title>{`${method.label}: ${formatValue(value, unit)}`}</title></polygon>;
            })}
            {meanY !== undefined && <g className="distribution-mean">
              <polygon points={`${centre},${meanY - 9} ${centre + 9},${meanY} ${centre},${meanY + 9} ${centre - 9},${meanY}`} fill={method.color} stroke="#173f42" strokeWidth="1.4"><title>{`${method.label}: mean ${formatValue(methodMean, unit)} (${values.length} videos)`}</title></polygon>
              {options.showValues && <text x={centre} y={Math.max(top + 13, meanY - 14)} className="bar-value" textAnchor="middle">{formatValue(methodMean, unit)}</text>}
            </g>}
            <text x={centre} y={height - 43} className="chart-method-label" textAnchor="middle">{method.shortLabel}</text>
          </g>
        );
      })}
      <text x={left} y={height - 14} className="chart-axis-title">Video-level values ({unit || "native units"})</text>
    </svg>
  );
}

type PerformanceMetricDefinition = {
  id: "latency_ms" | "latency_jitter_ms" | "cpu_percent" | "gpu_percent" | "memory_percent" | "memory_mb";
  label: string;
  unit: string;
};

type PerformanceMatrixProps = {
  rows: PerformanceRow[];
  methods: MethodRecord[];
  metrics: ReadonlyArray<PerformanceMetricDefinition>;
  options: FigureDisplayOptions;
  svgId: string;
  ariaLabel: string;
};

export function PerformanceMatrixPlot({ rows, methods, metrics, options, svgId, ariaLabel }: PerformanceMatrixProps) {
  const visibleMethods = selectedMethodRecords(methods, options.methods);
  const width = 1100;
  const left = 190;
  const right = 28;
  const top = options.showLegend ? 64 : 28;
  const rowHeight = 58;
  const bottom = 38;
  const height = top + metrics.length * rowHeight + bottom;
  const plotWidth = width - left - right;
  const columnWidth = plotWidth / Math.max(visibleMethods.length, 1);
  const cellPadding = 9;
  const cellWidth = Math.max(columnWidth - cellPadding * 2, 12);
  const description = `${ariaLabel}. Rows are performance metrics and columns are selected methods; each bar is the mean across the selected videos.`;

  return (
    <svg id={svgId} className="chart-svg performance-matrix-chart" viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={description}>
      <title>{description}</title>
      <rect width={width} height={height} fill="#ffffff" />
      {options.showLegend && <SvgLegend methods={visibleMethods} x={left} y={12} />}
      {visibleMethods.map((method, methodIndex) => {
        const centre = left + columnWidth * (methodIndex + 0.5);
        return <text key={method.id} x={centre} y={top - 10} className="chart-method-label" textAnchor="middle">{method.shortLabel}</text>;
      })}
      {metrics.map((metric, metricIndex) => {
        const y = top + metricIndex * rowHeight;
        const metricValues = visibleMethods.flatMap((method) => rows.filter((row) => row.method === method.id).map((row) => row[metric.id]).filter((value): value is number => typeof value === "number" && Number.isFinite(value)));
        const rowMaximum = Math.max(...metricValues, 0);
        return <g key={metric.id}>
          <line x1={left} x2={width - right} y1={y + rowHeight} y2={y + rowHeight} className="chart-grid" />
          <text x={left - 14} y={y + 24} className="performance-row-label" textAnchor="end">{metric.label}</text>
          <text x={left - 14} y={y + 40} className="performance-row-unit" textAnchor="end">{metric.unit}</text>
          {visibleMethods.map((method, methodIndex) => {
            const centre = left + columnWidth * (methodIndex + 0.5);
            const cellX = centre - cellWidth / 2;
            const values = rows.filter((row) => row.method === method.id).map((row) => row[metric.id]).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
            const methodMean = mean(values);
            const rangeMin = values.length ? Math.min(...values) : undefined;
            const rangeMax = values.length ? Math.max(...values) : undefined;
            const ratio = rowMaximum > 0 && Number.isFinite(methodMean) ? methodMean / rowMaximum : 0;
            return <g key={method.id}>
              <rect x={cellX} y={y + 10} width={cellWidth} height="25" rx="2" fill="#d6dedb" />
              <rect x={cellX} y={y + 10} width={cellWidth * Math.max(0, Math.min(1, ratio))} height="25" rx="2" fill={method.color} opacity="0.92"><title>{`${method.label}, ${metric.label}: ${formatValue(methodMean)} ${metric.unit}`}</title></rect>
              {options.showInstances && rowMaximum > 0 && values.map((value, valueIndex) => <line key={`${value}-${valueIndex}`} x1={cellX + cellWidth * Math.max(0, Math.min(1, value / rowMaximum))} x2={cellX + cellWidth * Math.max(0, Math.min(1, value / rowMaximum))} y1={y + 7} y2={y + 38} stroke={method.color} strokeWidth="1.2" opacity="0.32"><title>{`${method.label}, ${metric.label}: ${formatValue(value)} ${metric.unit}`}</title></line>)}
              {options.showMinMax && rangeMin !== undefined && rangeMax !== undefined && rowMaximum > 0 && <line x1={cellX + cellWidth * rangeMin / rowMaximum} x2={cellX + cellWidth * rangeMax / rowMaximum} y1={y + 43} y2={y + 43} className="performance-range" />}
              {options.showValues && Number.isFinite(methodMean) && <text x={cellX + cellWidth / 2} y={y + 27} className="performance-value" textAnchor="middle">{formatValue(methodMean)}</text>}
            </g>;
          })}
        </g>;
      })}
      <text x={left} y={height - 10} className="chart-axis-title">Each row is scaled to its largest selected method mean</text>
    </svg>
  );
}

type QualityMatrixProps = {
  rows: QualityRow[];
  methods: MethodRecord[];
  rowLabels: Array<{ id: string; label: string }>;
  metricForRows: (rowId: string) => MetricId;
  options: FigureDisplayOptions;
  svgId: string;
  ariaLabel: string;
};

export function QualityMatrixPlot({ rows, methods, rowLabels, metricForRows, options, svgId, ariaLabel }: QualityMatrixProps) {
  const visibleMethods = selectedMethodRecords(methods, options.methods);
  const width = 1100;
  const left = 190;
  const right = 28;
  const top = options.showLegend ? 64 : 28;
  const rowHeight = 58;
  const bottom = 38;
  const height = top + rowLabels.length * rowHeight + bottom;
  const plotWidth = width - left - right;
  const columnWidth = plotWidth / Math.max(visibleMethods.length, 1);
  const cellPadding = 9;
  const cellWidth = Math.max(columnWidth - cellPadding * 2, 12);
  const description = `${ariaLabel}. Rows are selected quality groups and columns are methods; each bar is the mean across the corresponding videos.`;

  return (
    <svg id={svgId} className="chart-svg performance-matrix-chart" viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={description}>
      <title>{description}</title>
      <rect width={width} height={height} fill="#ffffff" />
      {options.showLegend && <SvgLegend methods={visibleMethods} x={left} y={12} />}
      {visibleMethods.map((method, methodIndex) => <text key={method.id} x={left + columnWidth * (methodIndex + 0.5)} y={top - 10} className="chart-method-label" textAnchor="middle">{method.shortLabel}</text>)}
      {rowLabels.map((rowLabel, rowIndex) => {
        const y = top + rowIndex * rowHeight;
        const metric = metricForRows(rowLabel.id);
        const metricRows = rows.filter((row) => row.metric === metric && row.category === rowLabel.id);
        const metricValues = visibleMethods.flatMap((method) => metricRows.filter((row) => row.method === method.id).map((row) => row.value).filter((value): value is number => typeof value === "number" && Number.isFinite(value)));
        const rowMaximum = Math.max(...metricValues, 0);
        return <g key={rowLabel.id}>
          <line x1={left} x2={width - right} y1={y + rowHeight} y2={y + rowHeight} className="chart-grid" />
          <text x={left - 14} y={y + 29} className="performance-row-label" textAnchor="end">{rowLabel.label}</text>
          {visibleMethods.map((method, methodIndex) => {
            const centre = left + columnWidth * (methodIndex + 0.5);
            const cellX = centre - cellWidth / 2;
            const values = metricRows.filter((row) => row.method === method.id).map((row) => row.value).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
            const methodMean = mean(values);
            const rangeMin = values.length ? Math.min(...values) : undefined;
            const rangeMax = values.length ? Math.max(...values) : undefined;
            const ratio = rowMaximum > 0 && Number.isFinite(methodMean) ? methodMean / rowMaximum : 0;
            return <g key={method.id}>
              <rect x={cellX} y={y + 10} width={cellWidth} height="25" rx="2" fill="#d6dedb" />
              <rect x={cellX} y={y + 10} width={cellWidth * Math.max(0, Math.min(1, ratio))} height="25" rx="2" fill={method.color} opacity="0.92"><title>{`${rowLabel.label}, ${method.label}: ${formatValue(methodMean)}`}</title></rect>
              {options.showInstances && rowMaximum > 0 && values.map((value, valueIndex) => <line key={`${value}-${valueIndex}`} x1={cellX + cellWidth * Math.max(0, Math.min(1, value / rowMaximum))} x2={cellX + cellWidth * Math.max(0, Math.min(1, value / rowMaximum))} y1={y + 7} y2={y + 38} stroke={method.color} strokeWidth="1.2" opacity="0.32"><title>{`${rowLabel.label}, ${method.label}: ${formatValue(value)}`}</title></line>)}
              {options.showMinMax && rangeMin !== undefined && rangeMax !== undefined && rowMaximum > 0 && <line x1={cellX + cellWidth * rangeMin / rowMaximum} x2={cellX + cellWidth * rangeMax / rowMaximum} y1={y + 43} y2={y + 43} className="performance-range" />}
              {options.showValues && Number.isFinite(methodMean) && <text x={cellX + cellWidth / 2} y={y + 27} className="performance-value" textAnchor="middle">{formatValue(methodMean)}</text>}
            </g>;
          })}
        </g>;
      })}
      <text x={left} y={height - 10} className="chart-axis-title">Each row is scaled to its largest selected method mean</text>
    </svg>
  );
}

type CategoryTrendProps = {
  rows: QualityRow[];
  methods: MethodRecord[];
  categories: Array<{ id: string; label: string }>;
  metric: MetricId;
  options: FigureDisplayOptions;
  svgId: string;
  ariaLabel: string;
};

export function CategoryTrendPlot({ rows, methods, categories, metric, options, svgId, ariaLabel }: CategoryTrendProps) {
  const visibleMethods = selectedMethodRecords(methods, options.methods);
  const grouped = new Map<MethodId, Map<string, number>>();
  visibleMethods.forEach((method) => grouped.set(method.id, new Map()));
  const observations = new Map<string, number[]>();
  rows.filter((row) => row.metric === metric && grouped.has(row.method) && Number.isFinite(row.value)).forEach((row) => {
    const key = `${row.method}\u0000${row.category}`;
    const values = observations.get(key) ?? [];
    values.push(row.value as number);
    observations.set(key, values);
  });
  observations.forEach((values, key) => {
    const [methodId, categoryId] = key.split("\u0000");
    grouped.get(methodId as MethodId)?.set(categoryId, mean(values));
  });
  const values = [...grouped.values()].flatMap((categoryValues) => [...categoryValues.values()]);
  const [minimum, maximum] = resolveAxisDomain(values, qualityMetricAxisBounds(metric));
  const width = 1100;
  const height = 470;
  const left = 76;
  const right = 34;
  const top = options.showLegend ? 74 : 34;
  const bottom = 78;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const x = (index: number) => left + (categories.length <= 1 ? plotWidth / 2 : (plotWidth * index) / (categories.length - 1));
  const y = (value: number) => top + plotHeight - ((value - minimum) / (maximum - minimum || 1)) * plotHeight;
  const description = `${ariaLabel}. Four method lines connect category means for the selected metric.`;

  return (
    <svg id={svgId} className="chart-svg category-trend-chart" viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={description}>
      <title>{description}</title>
      <rect width={width} height={height} fill="#ffffff" />
      {options.showLegend && <SvgLegend methods={visibleMethods} x={left} y={14} columns={visibleMethods.length} columnGap={210} />}
      {tickValues(minimum, maximum).map((tick) => <g key={tick}><line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} className="chart-grid" /><text x={left - 10} y={y(tick) + 4} className="chart-axis-label" textAnchor="end">{formatValue(tick)}</text></g>)}
      <line x1={left} x2={width - right} y1={top + plotHeight} y2={top + plotHeight} className="chart-axis" />
      {categories.map((category, index) => <text key={category.id} x={x(index)} y={height - 47} className="chart-category-label" textAnchor="middle">{category.label}</text>)}
      {visibleMethods.map((method) => {
        const categoryValues = grouped.get(method.id) ?? new Map<string, number>();
        const pathParts: string[] = [];
        categories.forEach((category, index) => {
          const value = categoryValues.get(category.id);
          if (value === undefined) return;
          const previous = index > 0 ? categoryValues.get(categories[index - 1].id) : undefined;
          pathParts.push(`${previous === undefined ? "M" : "L"}${x(index)},${y(value)}`);
        });
        return <g key={method.id}>
          <path d={pathParts.join(" ")} fill="none" stroke={method.color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
          {categories.map((category, index) => {
            const value = categoryValues.get(category.id);
            if (value === undefined) return null;
            return <g key={category.id}><circle cx={x(index)} cy={y(value)} r="5" fill="#ffffff" stroke={method.color} strokeWidth="2.5"><title>{`${method.label}, ${category.label}: ${formatValue(value)}`}</title></circle>{options.showValues && <text x={x(index)} y={y(value) - 10} className="category-value" fill={method.color} textAnchor="middle">{formatValue(value)}</text>}</g>;
          })}
        </g>;
      })}
      <text x={left} y={height - 15} className="chart-axis-title">Video category</text>
    </svg>
  );
}

type EffectProps = {
  methods: MethodRecord[];
  comparisons: PairedComparison[];
  selectedBaselines: MethodId[];
  showValues: boolean;
  showLegend: boolean;
  svgId: string;
  ariaLabel: string;
};

export function VerticalEffectBarPlot({ methods, comparisons, selectedBaselines, showValues, showLegend, svgId, ariaLabel }: EffectProps) {
  const rows = comparisons.filter((comparison) => selectedBaselines.includes(comparison.baseline));
  const visibleMethods = selectedMethodRecords(methods, rows.map((row) => row.baseline));
  const extent = rows.flatMap((row) => [row.ciLow, row.ciHigh, row.meanDifference, 0]);
  const rawMinimum = Math.min(...extent);
  const rawMaximum = Math.max(...extent);
  const padding = Math.max((rawMaximum - rawMinimum) * 0.16, 0.05);
  const minimum = Math.min(0, rawMinimum - padding);
  const maximum = Math.max(0, rawMaximum + padding);
  const height = 430;
  const left = 64;
  const right = 24;
  const top = showLegend ? 58 : 26;
  const bottom = 82;
  const plotWidth = WIDTH - left - right;
  const plotHeight = height - top - bottom;
  const y = (value: number) => top + plotHeight - ((value - minimum) / (maximum - minimum || 1)) * plotHeight;
  const zero = y(0);
  const slot = plotWidth / Math.max(rows.length, 1);
  const barWidth = Math.min(96, slot * 0.6);
  const description = `${ariaLabel}. Positive values favour the constrained solution; whiskers are 95 percent paired bootstrap confidence intervals.`;

  return (
    <svg id={svgId} className="chart-svg" viewBox={`0 0 ${WIDTH} ${height}`} width={WIDTH} height={height} role="img" aria-label={description}>
      <title>{description}</title>
      <rect width={WIDTH} height={height} fill="#ffffff" />
      {showLegend && <SvgLegend methods={visibleMethods} />}
      {tickValues(minimum, maximum).map((tick) => (
        <g key={tick}>
          <line x1={left} x2={WIDTH - right} y1={y(tick)} y2={y(tick)} className="chart-grid" />
          <text x={left - 9} y={y(tick) + 4} className="chart-axis-label" textAnchor="end">{formatValue(tick)}</text>
        </g>
      ))}
      <line x1={left} x2={WIDTH - right} y1={zero} y2={zero} className="chart-zero" />
      {rows.map((row, index) => {
        const method = methods.find((candidate) => candidate.id === row.baseline);
        if (!method) return null;
        const centre = left + slot * (index + 0.5);
        const barY = y(row.meanDifference);
        const rectY = Math.min(zero, barY);
        const labelY = row.meanDifference >= 0 ? Math.max(top + 13, y(row.ciHigh) - 8) : Math.min(height - bottom - 6, y(row.ciLow) + 18);
        return (
          <g key={row.baseline}>
            <rect x={centre - barWidth / 2} y={rectY} width={barWidth} height={Math.abs(zero - barY)} rx="4" fill={method.color} opacity="0.9"><title>{`${method.label}: ${formatValue(row.meanDifference)} improvement`}</title></rect>
            <line x1={centre} x2={centre} y1={y(row.ciLow)} y2={y(row.ciHigh)} className="ci-line" />
            <line x1={centre - 7} x2={centre + 7} y1={y(row.ciLow)} y2={y(row.ciLow)} className="ci-line" />
            <line x1={centre - 7} x2={centre + 7} y1={y(row.ciHigh)} y2={y(row.ciHigh)} className="ci-line" />
            {showValues && <text x={centre} y={labelY} className="bar-value" textAnchor="middle">{formatValue(row.meanDifference)}</text>}
            <text x={centre} y={height - 52} className="chart-method-label" textAnchor="middle">{method.shortLabel}</text>
          </g>
        );
      })}
      <text x={left} y={height - 17} className="chart-axis-title">Direction-adjusted paired improvement (positive favours constrained)</text>
    </svg>
  );
}

type CategoryProps = {
  data: AnalysisData;
  metric: MetricId;
  options: FigureDisplayOptions;
  svgId: string;
};

export function VerticalCategoryBarPlot({ data, metric, options, svgId }: CategoryProps) {
  const visibleMethods = selectedMethodRecords(data.methods, options.methods);
  const rows = data.qualityRows.filter((row) => row.metric === metric && options.methods.includes(row.method) && Number.isFinite(row.value));
  const values = finite(rows.map((row) => row.value));
  const [domainMinimum, domainMaximum] = resolveAxisDomain(values, qualityMetricAxisBounds(metric));
  const height = 455;
  const left = 64;
  const right = 24;
  const top = options.showLegend ? 58 : 26;
  const bottom = 90;
  const plotWidth = WIDTH - left - right;
  const plotHeight = height - top - bottom;
  const y = (value: number) => top + plotHeight - ((value - domainMinimum) / (domainMaximum - domainMinimum || 1)) * plotHeight;
  const baseline = y(0);
  // Reserve a clear gutter between category groups so the five summaries read as
  // distinct blocks while method bars remain tightly grouped within each block.
  const categoryGap = 44;
  const groupWidth = (plotWidth - categoryGap * Math.max(data.categories.length - 1, 0)) / data.categories.length;
  const methodSlot = groupWidth / Math.max(visibleMethods.length, 1);
  const barWidth = Math.min(34, methodSlot * 0.72);
  const metricRecord = data.metrics.find((item) => item.id === metric);
  const description = `${metricRecord?.label ?? metric} by motion category. It shows category means${options.showInstances ? " and five underlying clips per category" : ""}${options.showMinMax ? " with min-max ranges" : ""}.`;

  return (
    <svg id={svgId} className="chart-svg" viewBox={`0 0 ${WIDTH} ${height}`} width={WIDTH} height={height} role="img" aria-label={description}>
      <title>{description}</title>
      <rect width={WIDTH} height={height} fill="#ffffff" />
      {options.showLegend && <SvgLegend methods={visibleMethods} />}
      {tickValues(domainMinimum, domainMaximum).map((tick) => <g key={tick}><line x1={left} x2={WIDTH - right} y1={y(tick)} y2={y(tick)} className="chart-grid" /><text x={left - 9} y={y(tick) + 4} className="chart-axis-label" textAnchor="end">{formatValue(tick)}</text></g>)}
      <line x1={left} x2={WIDTH - right} y1={baseline} y2={baseline} className="chart-axis" />
      {data.categories.map((category, categoryIndex) => {
        const categoryStart = left + categoryIndex * (groupWidth + categoryGap);
        return <g key={category.id}>
          {visibleMethods.map((method, methodIndex) => {
            const methodRows = rows.filter((row) => row.category === category.id && row.method === method.id);
            const methodMean = mean(methodRows.map((row) => row.value as number));
            const rangeMin = methodRows.length > 0 ? Math.min(...methodRows.map((row) => row.value as number)) : undefined;
            const rangeMax = methodRows.length > 0 ? Math.max(...methodRows.map((row) => row.value as number)) : undefined;
            const centre = categoryStart + methodSlot * (methodIndex + 0.5);
            const barY = Number.isFinite(methodMean) ? y(methodMean) : baseline;
            return <g key={method.id}>
              <rect x={centre - barWidth / 2} y={barY} width={barWidth} height={Math.max(0, baseline - barY)} rx="3" fill={method.color} opacity="0.88"><title>{`${category.label}, ${method.label}: ${formatValue(methodMean)} mean`}</title></rect>
              {options.showMinMax && rangeMin !== undefined && rangeMax !== undefined && <g className="min-max-range"><line x1={centre} x2={centre} y1={y(rangeMin)} y2={y(rangeMax)} /><line x1={centre - 5} x2={centre + 5} y1={y(rangeMin)} y2={y(rangeMin)} /><line x1={centre - 5} x2={centre + 5} y1={y(rangeMax)} y2={y(rangeMax)} />{options.showMinMaxValues && <><text x={centre - 8} y={y(rangeMax) + 3} className="range-value" textAnchor="end">{formatValue(rangeMax)}</text><text x={centre - 8} y={y(rangeMin) + 3} className="range-value" textAnchor="end">{formatValue(rangeMin)}</text></>}</g>}
              {options.showInstances && methodRows.map((row, index) => <circle key={row.videoId} cx={centre + (((index * 7) % 5) - 2) * 2.1} cy={y(row.value as number)} r="2.8" fill="#ffffff" stroke={method.color} strokeWidth="1.3"><title>{`${row.videoId}: ${formatValue(row.value as number)}`}</title></circle>)}
              {options.showValues && Number.isFinite(methodMean) && options.showMinMax && rangeMin !== undefined && rangeMax !== undefined ? <g className="mean-marker"><circle cx={centre + 7} cy={y(methodMean)} r="3" fill="#173f42" stroke="#ffffff" strokeWidth="1.2" /><text x={centre + 13} y={y(methodMean) + 3} className="category-value" textAnchor="start">{formatValue(methodMean)}</text></g> : options.showValues && Number.isFinite(methodMean) && <text x={centre} y={Math.max(top + 13, barY - 6)} className="category-value" textAnchor="middle">{formatValue(methodMean)}</text>}
            </g>;
          })}
          <text x={categoryStart + groupWidth / 2} y={height - 53} className="chart-category-label" textAnchor="middle">{category.label}</text>
        </g>;
      })}
      <text x={left} y={height - 17} className="chart-axis-title">Mean across five clips (native metric units)</text>
    </svg>
  );
}

export function getSingularityCategoryCounts(data: AnalysisData) {
  const records = data.videoSingularities ?? [];
  return data.categories.map((category) => {
    const categoryRecords = records.filter((record) => record.category === category.id && record.classificationAvailable);
    const armValues = (side: "left" | "right") => ({
      dual: categoryRecords.reduce((sum, record) => sum + record[`${side}ArmDualFrameCount`], 0),
      elbow: categoryRecords.reduce((sum, record) => sum + record[`${side}ArmElbowRollFrameCount`], 0),
      shoulder: categoryRecords.reduce((sum, record) => sum + record[`${side}ArmShoulderRollFrameCount`], 0),
      none: categoryRecords.reduce((sum, record) => sum + record[`${side}ArmNoSingularityFrameCount`], 0),
    });
    return {
      category,
      totalFrames: categoryRecords.reduce((sum, record) => sum + record.totalFrameCount, 0),
      arms: { left: armValues("left"), right: armValues("right") },
    };
  });
}

export function SingularityCategoryBarPlot({ data, svgId = "singularity-category-counts", colors }: { data: AnalysisData; svgId?: string; colors?: string[] }) {
  const segments = [
    { key: "dual", label: "Dual singularity", color: "#176b87" },
    { key: "elbow", label: "Elbow-roll singularity", color: "#2f9e89" },
    { key: "shoulder", label: "Shoulder-roll singularity", color: "#72d8bb" },
    { key: "none", label: "No singularity", color: "#d9dedb" },
  ] as const;
  const segmentColors = colors?.length === 4 ? colors : ["#176b87", "#2f9e89", "#72d8bb", "#d9dedb"];
  const counts = getSingularityCategoryCounts(data);
  const max = Math.max(1, ...counts.flatMap(({ arms }) => Object.values(arms).map((values) => Object.values(values).reduce((sum, value) => sum + value, 0))));
  const height = 430;
  const left = 70;
  const right = 28;
  const top = 82;
  const bottom = 82;
  const plotHeight = height - top - bottom;
  const plotWidth = WIDTH - left - right;
  const groupWidth = plotWidth / Math.max(counts.length, 1);
  const barWidth = Math.min(46, groupWidth * 0.3);
  const barGap = 10;
  const y = (value: number) => top + plotHeight - (value / max) * plotHeight;
  const description = "Separate left-arm and right-arm frame counts for dual, elbow-roll, shoulder-roll, or no singularity by category";
  return <svg id={svgId} className="chart-svg" viewBox={`0 0 ${WIDTH} ${height}`} width={WIDTH} height={height} role="img" aria-label={description}>
    <title>{description}</title>
    <rect width={WIDTH} height={height} fill="#ffffff" />
    <g className="svg-legend">{segments.map((segment, index) => <g key={segment.key} transform={`translate(${left + index * 190}, 18)`}><rect width="13" height="13" rx="2" fill={segmentColors[index]} /><text x="19" y="11" className="chart-legend-text">{segment.label}</text></g>)}</g>
    {tickValues(0, max).map((tick) => <g key={tick}><line x1={left} x2={WIDTH - right} y1={y(tick)} y2={y(tick)} className="chart-grid" /><text x={left - 10} y={y(tick) + 4} className="chart-axis-label" textAnchor="end">{tick}</text></g>)}
    <line x1={left} x2={WIDTH - right} y1={y(0)} y2={y(0)} className="chart-axis" />
    {counts.map(({ category, arms }, index) => { const centre = left + index * groupWidth + groupWidth / 2; return <g key={category.id}>{(["left", "right"] as const).map((side, sideIndex) => { const values = arms[side]; let cursor = 0; const x = centre + (sideIndex ? barGap / 2 : -barGap / 2 - barWidth); const total = Object.values(values).reduce((sum, value) => sum + value, 0); return <g key={side}>{segments.map((segment, segmentIndex) => { const value = values[segment.key]; const segmentStart = cursor; const barTop = y(cursor + value); const barBottom = y(cursor); cursor += value; const segmentHeight = Math.max(0, barBottom - barTop); return value ? <g key={segment.key}><rect x={x} y={barTop} width={barWidth} height={segmentHeight} fill={segmentColors[segmentIndex]}><title>{`${category.label}, ${side} arm: ${segment.label}, ${value} frame${value === 1 ? "" : "s"}`}</title></rect>{segmentHeight > 18 && <text x={x + barWidth / 2} y={y(segmentStart + value / 2) + 4} className="bar-value" textAnchor="middle" fill={segmentIndex >= 2 ? "#173b38" : "#ffffff"}>{value}</text>}</g> : null; })}<text x={x + barWidth / 2} y={Math.max(top + 14, y(total) - 8)} className="category-value" textAnchor="middle">{total}</text><text x={x + barWidth / 2} y={height - 66} className="chart-axis-label" textAnchor="middle">{side === "left" ? "L" : "R"}</text></g>; })}<text x={centre} y={height - 47} className="chart-category-label" textAnchor="middle">{category.label}</text></g>; })}
    <text x={left} y={height - 16} className="chart-axis-title">Number of classified frames</text>
  </svg>;
}

type TrajectoryProps = {
  data: AnalysisData;
  category: string;
  videoId: string;
  side: "left" | "right";
  metric: MetricId;
  methods: MethodId[];
  showLegend: boolean;
  startFrame: number;
  endFrame: number;
  svgId: string;
};

export function TrajectoryPlot({ data, category, videoId, side, metric, methods, showLegend, startFrame, endFrame, svgId }: TrajectoryProps) {
  const rows = data.trajectoryRows.filter((row) => row.category === category && row.videoId === videoId && row.side === side && methods.includes(row.method) && row.frameId >= startFrame && row.frameId <= endFrame);
  const visibleMethods = selectedMethodRecords(data.methods, methods);
  const valid = finite(rows.map((row) => row[metric]));
  const [min, max] = resolveAxisDomain(valid, qualityMetricAxisBounds(metric));
  const height = 390;
  const left = 58;
  const right = 26;
  const top = showLegend ? 58 : 24;
  const bottom = 54;
  const y = (value: number) => top + (height - top - bottom) * (1 - (value - min) / (max - min || 1));
  const x = (frame: number) => left + ((WIDTH - left - right) * (frame - startFrame)) / (endFrame - startFrame || 1);
  const description = `Frame trajectory for ${videoId}, frames ${startFrame} through ${endFrame}. ${methods.length} methods are shown; missing values are marked with gaps.`;
  return <svg id={svgId} className="chart-svg" viewBox={`0 0 ${WIDTH} ${height}`} width={WIDTH} height={height} role="img" aria-label={description}>
    <title>{description}</title><rect width={WIDTH} height={height} fill="#ffffff" />
    {showLegend && <SvgLegend methods={visibleMethods} />}
    {tickValues(min, max).map((tick) => <g key={tick}><line x1={left} x2={WIDTH - right} y1={y(tick)} y2={y(tick)} className="chart-grid" /><text x={left - 8} y={y(tick) + 4} className="chart-axis-label" textAnchor="end">{formatValue(tick)}</text></g>)}
    {tickValues(startFrame, endFrame).map((frame) => <g key={frame}><line x1={x(frame)} x2={x(frame)} y1={top} y2={height - bottom} className="chart-grid chart-grid-vertical" /><text x={x(frame)} y={height - 26} className="chart-axis-label" textAnchor="middle">{Math.round(frame)}</text></g>)}
    {visibleMethods.map((method) => {
      const series = rows.filter((row) => row.method === method.id).sort((a, b) => a.frameId - b.frameId);
      let path = "";
      series.forEach((row) => { const value = row[metric]; path += Number.isFinite(value) ? `${path && !path.endsWith("M") ? " L" : "M"}${x(row.frameId)},${y(value as number)}` : "M"; });
      return <g key={method.id}><path d={path} fill="none" stroke={method.color} strokeWidth="2.2" strokeDasharray={method.dash ?? undefined} /><title>{method.label}</title></g>;
    })}
    {rows.filter((row) => !Number.isFinite(row[metric])).map((row) => <line key={`${row.method}-${row.frameId}`} x1={x(row.frameId)} x2={x(row.frameId)} y1={height - bottom - 9} y2={height - bottom} stroke="#a44" strokeWidth="2" />)}
    <text x={left} y={height - 7} className="chart-axis-title">Frame index ({startFrame}-{endFrame})</text>
  </svg>;
}

export function PerformanceTimelinePlot({ data, videoId, metric, methods, showLegend, startFrame, endFrame, svgId, unit }: { data: AnalysisData; videoId: string; metric: "latency_ms" | "cpu_percent" | "gpu_percent" | "memory_percent" | "memory_mb"; methods: MethodId[]; showLegend: boolean; startFrame: number; endFrame: number; svgId: string; unit: string }) {
  const visibleMethods = selectedMethodRecords(data.methods, methods);
  const grouped = new Map<string, number[]>();
  data.performanceFrames.filter((row) => row.videoId === videoId && methods.includes(row.method) && row.frameId >= startFrame && row.frameId <= endFrame && Number.isFinite(row[metric])).forEach((row) => {
    const key = `${row.method}-${row.frameId}`;
    grouped.set(key, [...(grouped.get(key) ?? []), row[metric] as number]);
  });
  const series = visibleMethods.map((method) => Array.from({ length: endFrame - startFrame + 1 }, (_, offset) => {
    const frameId = startFrame + offset;
    const values = grouped.get(`${method.id}-${frameId}`) ?? [];
    return { frameId, value: mean(values) };
  }));
  const valid = finite(series.flatMap((items) => items.map((item) => item.value)));
  const min = Math.min(...valid, 0);
  const max = Math.max(...valid, 1);
  const height = 390;
  const left = 58;
  const right = 26;
  const top = showLegend ? 58 : 24;
  const bottom = 54;
  const y = (value: number) => top + (height - top - bottom) * (1 - (value - min) / (max - min || 1));
  const x = (frame: number) => left + ((WIDTH - left - right) * (frame - startFrame)) / (endFrame - startFrame || 1);
  const description = `${metric} performance evolution for ${videoId}, frames ${startFrame} through ${endFrame}. Each value is the mean of three repeats.`;
  return <svg id={svgId} className="chart-svg" viewBox={`0 0 ${WIDTH} ${height}`} width={WIDTH} height={height} role="img" aria-label={description}>
    <title>{description}</title><rect width={WIDTH} height={height} fill="#ffffff" />
    {showLegend && <SvgLegend methods={visibleMethods} />}
    {tickValues(min, max).map((tick) => <g key={tick}><line x1={left} x2={WIDTH - right} y1={y(tick)} y2={y(tick)} className="chart-grid" /><text x={left - 8} y={y(tick) + 4} className="chart-axis-label" textAnchor="end">{formatValue(tick)}</text></g>)}
    {tickValues(startFrame, endFrame).map((frame) => <g key={frame}><line x1={x(frame)} x2={x(frame)} y1={top} y2={height - bottom} className="chart-grid chart-grid-vertical" /><text x={x(frame)} y={height - 26} className="chart-axis-label" textAnchor="middle">{Math.round(frame)}</text></g>)}
    {visibleMethods.map((method, index) => { const path = series[index].filter((item) => Number.isFinite(item.value)).map((item, itemIndex) => `${itemIndex ? "L" : "M"}${x(item.frameId)},${y(item.value)}`).join(" "); return <path key={method.id} d={path} fill="none" stroke={method.color} strokeWidth="2.2" strokeDasharray={method.dash ?? undefined} />; })}
    <text x={left} y={height - 7} className="chart-axis-title">Frame index ({startFrame}-{endFrame}); mean of 3 repeats ({unit})</text>
  </svg>;
}

export function LatencyEcdfPlot({ data, methods, showLegend, svgId }: { data: AnalysisData; methods: MethodId[]; showLegend: boolean; svgId: string }) {
  const visibleMethods = selectedMethodRecords(data.methods, methods);
  const values = finite(data.performanceRepeats.filter((row) => methods.includes(row.method)).map((row) => row.latency_ms));
  const min = Math.min(...values, 0.1);
  const max = Math.max(...values, 1);
  const logMin = Math.log10(Math.max(min * 0.8, 0.01));
  const logMax = Math.log10(max * 1.2);
  const height = 370;
  const left = 62;
  const right = 26;
  const top = showLegend ? 58 : 24;
  const bottom = 58;
  const x = (value: number) => left + ((Math.log10(value) - logMin) / (logMax - logMin || 1)) * (WIDTH - left - right);
  const y = (value: number) => top + (1 - value) * (height - top - bottom);
  return <svg id={svgId} className="chart-svg" viewBox={`0 0 ${WIDTH} ${height}`} width={WIDTH} height={height} role="img" aria-label="Log-scale empirical cumulative latency distribution for the selected methods.">
    <title>Log-scale empirical cumulative latency distribution</title><rect width={WIDTH} height={height} fill="#ffffff" />
    {showLegend && <SvgLegend methods={visibleMethods} />}
    {[0, 0.25, 0.5, 0.75, 1].map((tick) => <g key={tick}><line x1={left} x2={WIDTH - right} y1={y(tick)} y2={y(tick)} className="chart-grid" /><text x={left - 9} y={y(tick) + 4} className="chart-axis-label" textAnchor="end">{tick.toFixed(2)}</text></g>)}
    {visibleMethods.map((method) => {
      const series = finite(data.performanceRepeats.filter((row) => row.method === method.id).map((row) => row.latency_ms)).sort((a, b) => a - b);
      const path = series.map((value, index) => `${index ? "L" : "M"}${x(value)},${y((index + 1) / series.length)}`).join(" ");
      return <path key={method.id} d={path} fill="none" stroke={method.color} strokeWidth="2.3" strokeDasharray={method.dash ?? undefined} />;
    })}
    <text x={left} y={height - 13} className="chart-axis-title">Latency in ms (log scale)</text>
  </svg>;
}
