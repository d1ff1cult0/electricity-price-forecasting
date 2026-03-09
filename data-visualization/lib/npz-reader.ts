import AdmZip from "adm-zip";
import fs from "fs";

interface NpyArray {
  shape: number[];
  dtype: string;
  data: number[];
}

function parseNpy(buffer: Buffer): NpyArray {
  if (
    buffer[0] !== 0x93 ||
    buffer[1] !== 0x4e || // N
    buffer[2] !== 0x55 || // U
    buffer[3] !== 0x4d || // M
    buffer[4] !== 0x50 || // P
    buffer[5] !== 0x59    // Y
  ) {
    throw new Error("Invalid npy file: bad magic number");
  }

  const major = buffer[6];
  let headerLen: number;
  let headerOffset: number;

  if (major === 1) {
    headerLen = buffer.readUInt16LE(8);
    headerOffset = 10;
  } else {
    headerLen = buffer.readUInt32LE(8);
    headerOffset = 12;
  }

  const headerStr = buffer.toString("latin1", headerOffset, headerOffset + headerLen);

  const descrMatch = headerStr.match(/'descr'\s*:\s*'([^']+)'/);
  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
  const orderMatch = headerStr.match(/'fortran_order'\s*:\s*(True|False)/);

  if (!descrMatch || !shapeMatch) {
    throw new Error(`Cannot parse npy header: ${headerStr}`);
  }

  const dtype = descrMatch[1];
  const fortranOrder = orderMatch?.[1] === "True";
  const shape = shapeMatch[1]
    .split(",")
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number);

  if (fortranOrder) {
    throw new Error("Fortran-order arrays not supported");
  }

  const dataOffset = headerOffset + headerLen;
  const totalElements = shape.reduce((a, b) => a * b, 1);
  const data: number[] = new Array(totalElements);

  if (dtype === "<f8" || dtype === "<f4") {
    const bytesPerElement = dtype === "<f8" ? 8 : 4;
    const view = new DataView(
      buffer.buffer,
      buffer.byteOffset + dataOffset,
      totalElements * bytesPerElement
    );
    for (let i = 0; i < totalElements; i++) {
      data[i] =
        dtype === "<f8"
          ? view.getFloat64(i * 8, true)
          : view.getFloat32(i * 4, true);
    }
  } else if (dtype === "<i8" || dtype === "<i4") {
    const bytesPerElement = dtype === "<i8" ? 8 : 4;
    const view = new DataView(
      buffer.buffer,
      buffer.byteOffset + dataOffset,
      totalElements * bytesPerElement
    );
    for (let i = 0; i < totalElements; i++) {
      data[i] =
        dtype === "<i8"
          ? Number(view.getBigInt64(i * 8, true))
          : view.getInt32(i * 4, true);
    }
  } else {
    throw new Error(`Unsupported dtype: ${dtype}`);
  }

  return { shape, dtype, data };
}

function reshape2D(flat: number[], rows: number, cols: number): number[][] {
  const result: number[][] = new Array(rows);
  for (let i = 0; i < rows; i++) {
    result[i] = flat.slice(i * cols, (i + 1) * cols);
  }
  return result;
}

export interface PredictionArrays {
  [key: string]: number[][];
}

export function readNpz(filePath: string): PredictionArrays {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File not found: ${filePath}`);
  }
  const zip = new AdmZip(filePath);
  const entries = zip.getEntries();
  const arrays: PredictionArrays = {};

  for (const entry of entries) {
    if (!entry.entryName.endsWith(".npy")) continue;
    const name = entry.entryName.replace(".npy", "");
    const buffer = entry.getData();
    const parsed = parseNpy(buffer);

    if (parsed.shape.length === 2) {
      arrays[name] = reshape2D(parsed.data, parsed.shape[0], parsed.shape[1]);
    } else if (parsed.shape.length === 1) {
      arrays[name] = [parsed.data];
    } else {
      arrays[name] = [parsed.data];
    }
  }

  return arrays;
}

export function getNpzTotalDays(filePath: string): number {
  const totalRows = getNpzTotalRows(filePath);
  return Math.ceil(totalRows / 24);
}

export function getNpzTotalRows(filePath: string): number {
  if (!fs.existsSync(filePath)) return 0;
  const zip = new AdmZip(filePath);
  const entry = zip.getEntries().find((e) => e.entryName === "y_test.npy");
  if (!entry) return 0;
  const buffer = entry.getData();
  if (
    buffer[0] !== 0x93 || buffer[1] !== 0x4e || buffer[2] !== 0x55 ||
    buffer[3] !== 0x4d || buffer[4] !== 0x50 || buffer[5] !== 0x59
  ) return 0;
  const major = buffer[6];
  const headerLen = major === 1 ? buffer.readUInt16LE(8) : buffer.readUInt32LE(8);
  const headerOffset = major === 1 ? 10 : 12;
  const headerStr = buffer.toString("latin1", headerOffset, headerOffset + headerLen);
  const shapeMatch = headerStr.match(/'shape'\s*:\s*\(([^)]*)\)/);
  if (!shapeMatch) return 0;
  const shape = shapeMatch[1].split(",").map((s) => s.trim()).filter((s) => s.length > 0).map(Number);
  return shape[0] ?? 0;
}

export function readNpzSlice(
  filePath: string,
  startRow: number,
  endRow: number,
  stride: number = 1
): PredictionArrays {
  const full = readNpz(filePath);
  const sliced: PredictionArrays = {};

  for (const [key, arr] of Object.entries(full)) {
    if (arr.length > 1) {
      const s = Math.max(0, startRow);
      const e = Math.min(arr.length, endRow);
      if (stride <= 1) {
        sliced[key] = arr.slice(s, e);
      } else {
        const result: number[][] = [];
        for (let i = s; i < e; i += stride) {
          result.push(arr[i]);
        }
        sliced[key] = result;
      }
    } else {
      sliced[key] = arr;
    }
  }

  return sliced;
}

export function averageArrays(
  allRuns: PredictionArrays[]
): PredictionArrays {
  if (allRuns.length === 0) return {};
  const keys = Object.keys(allRuns[0]);
  const result: PredictionArrays = {};
  const nRuns = allRuns.length;

  for (const key of keys) {
    if (key === "y_test") {
      result[key] = allRuns[0][key];
      continue;
    }
    const rows = allRuns[0][key].length;
    const cols = allRuns[0][key][0]?.length ?? 0;
    const averaged: number[][] = Array.from({ length: rows }, () =>
      new Array(cols).fill(0)
    );
    for (const run of allRuns) {
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          averaged[r][c] += run[key][r][c] / nRuns;
        }
      }
    }
    result[key] = averaged;
  }

  return result;
}
