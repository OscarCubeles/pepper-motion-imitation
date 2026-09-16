import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Scientific IK Evaluation | Pepper Imitation",
  description:
    "Video-level scientific comparison of constrained IK, its ablation, IKPy, and the original Pepper imitation solution.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
