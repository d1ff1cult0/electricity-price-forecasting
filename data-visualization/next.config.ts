import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  output: "standalone",
  serverExternalPackages: ["adm-zip"],
  env: {
    RESULTS_DIR: process.env.RESULTS_DIR || path.resolve(process.cwd(), "../results"),
  },
};

export default nextConfig;
