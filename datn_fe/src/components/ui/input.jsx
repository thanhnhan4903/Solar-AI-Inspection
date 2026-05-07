import React from "react";

export function Input({ className = "", ...props }) {
  return (
    <input
      className={`flex h-10 w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-slate-400 ${className}`}
      {...props}
    />
  );
}
